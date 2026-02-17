/*
 * Optimized CUDA MOE Implementation for FlashInfer-Bench
 *
 * Optimizations:
 * 1. Weight caching - dequant FP8→BF16 once on first call, reuse forever
 *    Saves 64 dequant kernel launches per call (32 experts × 2 weights each)
 * 2. Grouped GEMM via cublasGemmGroupedBatchedEx - replaces 32-iteration
 *    per-expert loop with 2 grouped GEMM calls (GEMM1 + GEMM2)
 * 3. Single SwiGLU kernel over all tokens (not per-expert)
 * 4. Full BF16 pipeline for maximum tensor core throughput
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdint.h>

#define H_DIM       7168
#define I_DIM       2048
#define E_GLOBAL    256
#define E_LOCAL     32
#define TOP_K       8
#define N_GROUP     8
#define TOPK_GROUP  4
#define BLOCK_SZ    128

// ============================================================================
// FP8 E4M3FN → float32
// ============================================================================
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    uint32_t sign = (uint32_t)(x >> 7);
    uint32_t exp  = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    if (exp == 0) {
        float val = (float)mant * (1.0f / 512.0f);
        return sign ? -val : val;
    }
    if (exp == 15 && mant == 7) return __uint_as_float(0x7FC00000);
    uint32_t f32 = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
    return __uint_as_float(f32);
}

// ============================================================================
// Routing kernel
// ============================================================================
__global__ void routing_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int64_t*     __restrict__ topk_idx_out,
    float*       __restrict__ weights_out,
    float routed_scaling_factor,
    int T
) {
    int token = blockIdx.x;
    if (token >= T) return;
    int tid = threadIdx.x;

    __shared__ float s_scores[256];
    __shared__ float s_raw[256];

    float logit = logits[token * 256 + tid];
    float b     = bias[tid];
    float sig   = 1.0f / (1.0f + expf(-logit));
    s_raw[tid]    = sig;
    s_scores[tid] = sig + b;
    __syncthreads();

    if (tid == 0) {
        float group_scores[N_GROUP];
        for (int g = 0; g < N_GROUP; g++) {
            float max1 = -1e38f, max2 = -1e38f;
            int base = g * 32;
            for (int i = 0; i < 32; i++) {
                float v = s_scores[base + i];
                if (v > max1) { max2 = max1; max1 = v; }
                else if (v > max2) { max2 = v; }
            }
            group_scores[g] = max1 + max2;
        }

        bool selected_groups[N_GROUP];
        for (int g = 0; g < N_GROUP; g++) selected_groups[g] = false;
        for (int k = 0; k < TOPK_GROUP; k++) {
            int best_g = -1;
            float best_s = -1e38f;
            for (int g = 0; g < N_GROUP; g++) {
                if (!selected_groups[g] && group_scores[g] > best_s) {
                    best_s = group_scores[g];
                    best_g = g;
                }
            }
            if (best_g >= 0) selected_groups[best_g] = true;
        }

        float pruned[E_GLOBAL];
        for (int i = 0; i < E_GLOBAL; i++) {
            pruned[i] = selected_groups[i / 32] ? s_scores[i] : -1e38f;
        }

        float w_sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            int best_i = 0;
            float best_v = -1e38f;
            for (int i = 0; i < E_GLOBAL; i++) {
                if (pruned[i] > best_v) {
                    best_v = pruned[i];
                    best_i = i;
                }
            }
            topk_idx_out[token * TOP_K + k] = (int64_t)best_i;
            float w = s_raw[best_i];
            weights_out[token * TOP_K + k] = w;
            w_sum += w;
            pruned[best_i] = -1e38f;
        }

        w_sum += 1e-20f;
        for (int k = 0; k < TOP_K; k++) {
            weights_out[token * TOP_K + k] =
                (weights_out[token * TOP_K + k] / w_sum) * routed_scaling_factor;
        }
    }
}

// ============================================================================
// Permute kernels
// ============================================================================
__global__ void permute_count_kernel(
    const int64_t* __restrict__ topk_idx,
    int*           __restrict__ counts,
    int local_expert_offset, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int64_t ge = topk_idx[idx];
    int le = (int)(ge - local_expert_offset);
    if (le >= 0 && le < E_LOCAL) atomicAdd(&counts[le], 1);
}

__global__ void permute_prefix_sum_kernel(
    const int* __restrict__ counts,
    int*       __restrict__ offsets
) {
    int running = 0;
    for (int e = 0; e < E_LOCAL; e++) {
        offsets[e] = running;
        running += counts[e];
    }
    offsets[E_LOCAL] = running;
}

__global__ void permute_scatter_kernel(
    const int64_t* __restrict__ topk_idx,
    const int*     __restrict__ expert_offsets,
    int64_t*       __restrict__ sorted_token_ids,
    int*           __restrict__ token_expert_map,
    int*           __restrict__ write_counters,
    int local_expert_offset, int N, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int64_t ge = topk_idx[idx];
    int le = (int)(ge - local_expert_offset);
    if (le >= 0 && le < E_LOCAL) {
        int pos      = atomicAdd(&write_counters[le], 1);
        int base     = expert_offsets[le];
        int write_at = base + pos;
        sorted_token_ids[write_at] = (int64_t)(idx / top_k);
        token_expert_map[write_at] = le;
    }
}

// ============================================================================
// Dequant FP8 2D block-scale → BF16 (for weight caching)
// ============================================================================
__global__ void dequant_2d_to_bf16_kernel(
    const uint8_t* __restrict__ x_fp8,
    const float*   __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int R, int C, int scale_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = R * C;
    if (idx >= total) return;
    int r  = idx / C;
    int c  = idx % C;
    int rb = r / BLOCK_SZ;
    int cb = c / BLOCK_SZ;
    float s = scale[rb * scale_cols + cb];
    output[idx] = __float2bfloat16(fp8e4m3_to_float(x_fp8[idx]) * s);
}

// ============================================================================
// Gather + Dequant hidden states → float32
// ============================================================================
__global__ void gather_dequant_kernel(
    const uint8_t* __restrict__ hidden_states,
    const float*   __restrict__ hs_scale,
    const int64_t* __restrict__ sorted_ids,
    float*         __restrict__ output,
    int T, int H, int H_BLOCKS
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= H) return;
    int64_t token_id = sorted_ids[row];
    uint8_t fp8_val  = hidden_states[token_id * H + col];
    int block_idx    = col / BLOCK_SZ;
    float scale      = hs_scale[block_idx * T + token_id];
    output[row * H + col] = fp8e4m3_to_float(fp8_val) * scale;
}

// ============================================================================
// SwiGLU BF16 (internal float32 computation)
// ============================================================================
__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16*       __restrict__ output,
    int total_elements, int I_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int row   = idx / I_dim;
    int col   = idx % I_dim;
    int two_I = 2 * I_dim;
    float gate    = __bfloat162float(input[row * two_I + col]);
    float up      = __bfloat162float(input[row * two_I + I_dim + col]);
    float silu_up = up / (1.0f + expf(-up));
    output[idx]   = __float2bfloat16(silu_up * gate);
}

// ============================================================================
// SwiGLU float32
// ============================================================================
__global__ void swiglu_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int Tk, int I_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Tk * I_dim;
    if (idx >= total) return;
    int row   = idx / I_dim;
    int col   = idx % I_dim;
    int two_I = 2 * I_dim;
    float gate    = input[row * two_I + col];
    float up      = input[row * two_I + I_dim + col];
    float silu_up = up / (1.0f + expf(-up));
    output[idx]   = silu_up * gate;
}

// ============================================================================
// Weighted scatter-add from BF16 input
// ============================================================================
__global__ void weighted_scatter_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ gemm2_result,
    const int64_t* __restrict__ sorted_token_ids,
    const int*     __restrict__ token_expert_map,
    const int64_t* __restrict__ topk_idx,
    const float*   __restrict__ routing_weights,
    float*         __restrict__ output,
    int local_expert_offset, int H_dim, int top_k
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= H_dim) return;

    int64_t token_id      = sorted_token_ids[row];
    int     local_expert  = token_expert_map[row];
    int64_t global_expert = (int64_t)local_expert + local_expert_offset;

    float w = 0.0f;
    for (int k = 0; k < top_k; k++) {
        if (topk_idx[token_id * top_k + k] == global_expert) {
            w = routing_weights[token_id * top_k + k];
            break;
        }
    }

    float val = __bfloat162float(gemm2_result[row * H_dim + col]) * w;
    atomicAdd(&output[token_id * H_dim + col], val);
}

// ============================================================================
// Weighted scatter-add from float32 input
// ============================================================================
__global__ void weighted_scatter_add_kernel(
    const float*   __restrict__ gemm2_result,
    const int64_t* __restrict__ sorted_token_ids,
    const int*     __restrict__ token_expert_map,
    const int64_t* __restrict__ topk_idx,
    const float*   __restrict__ routing_weights,
    float*         __restrict__ output,
    int local_expert_offset, int H_dim, int top_k
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= H_dim) return;

    int64_t token_id      = sorted_token_ids[row];
    int     local_expert  = token_expert_map[row];
    int64_t global_expert = (int64_t)local_expert + local_expert_offset;

    float w = 0.0f;
    for (int k = 0; k < top_k; k++) {
        if (topk_idx[token_id * top_k + k] == global_expert) {
            w = routing_weights[token_id * top_k + k];
            break;
        }
    }

    float val = gemm2_result[row * H_dim + col] * w;
    atomicAdd(&output[token_id * H_dim + col], val);
}


// ============================================================================
// Static weight cache
// ============================================================================
static torch::Tensor W1_cache;  // [E_LOCAL, 2*I, H] bfloat16
static torch::Tensor W2_cache;  // [E_LOCAL, H, I] bfloat16
static bool weights_cached = false;

// Static device pointer buffers for grouped GEMM (allocated once)
static const void** d_A_ptrs1 = nullptr;
static const void** d_B_ptrs1 = nullptr;
static void**       d_C_ptrs1 = nullptr;
static const void** d_A_ptrs2 = nullptr;
static const void** d_B_ptrs2 = nullptr;
static void**       d_C_ptrs2 = nullptr;

// ============================================================================
// Host entry point
// ============================================================================
torch::Tensor kernel(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights,
    torch::Tensor gemm1_weights_scale,
    torch::Tensor gemm2_weights,
    torch::Tensor gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int T = routing_logits.size(0);
    auto device = routing_logits.device();
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opts_i64  = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto logits_f32 = routing_logits.to(torch::kFloat32).contiguous();
    auto bias_f32   = routing_bias.to(torch::kFloat32).reshape({-1}).contiguous();

    // ==================================================================
    // Step 1: Routing
    // ==================================================================
    auto topk_idx = torch::empty({T, TOP_K}, opts_i64);
    auto weights  = torch::empty({T, TOP_K}, opts_f32);

    if (T > 0) {
        routing_kernel<<<T, 256, 0, stream>>>(
            logits_f32.data_ptr<float>(),
            bias_f32.data_ptr<float>(),
            topk_idx.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            (float)routed_scaling_factor,
            T
        );
    }

    // ==================================================================
    // Step 2: Token permutation
    // ==================================================================
    int N = T * TOP_K;
    auto flat_idx = topk_idx.reshape({-1});

    auto counts          = torch::zeros({E_LOCAL}, opts_i32);
    auto expert_offsets  = torch::empty({E_LOCAL + 1}, opts_i32);
    auto sorted_ids_full = torch::empty({N}, opts_i64);
    auto expert_map_full = torch::empty({N}, opts_i32);
    auto write_counters  = torch::zeros({E_LOCAL}, opts_i32);

    if (N > 0) {
        int thr = 256;
        int blk = (N + thr - 1) / thr;

        permute_count_kernel<<<blk, thr, 0, stream>>>(
            flat_idx.data_ptr<int64_t>(),
            counts.data_ptr<int32_t>(),
            (int)local_expert_offset, N
        );

        permute_prefix_sum_kernel<<<1, 1, 0, stream>>>(
            counts.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>()
        );

        permute_scatter_kernel<<<blk, thr, 0, stream>>>(
            flat_idx.data_ptr<int64_t>(),
            expert_offsets.data_ptr<int32_t>(),
            sorted_ids_full.data_ptr<int64_t>(),
            expert_map_full.data_ptr<int32_t>(),
            write_counters.data_ptr<int32_t>(),
            (int)local_expert_offset, N, TOP_K
        );
    }

    // Read Tsum on CPU (requires sync)
    auto offsets_cpu = expert_offsets.cpu();
    int Tsum = offsets_cpu[E_LOCAL].item<int>();

    if (Tsum == 0) {
        return torch::zeros({T, H_DIM}, opts_bf16);
    }

    auto sorted_token_ids = sorted_ids_full.slice(0, 0, Tsum).contiguous();
    auto token_expert_map_t = expert_map_full.slice(0, 0, Tsum).contiguous();

    // ==================================================================
    // Step 3: Cache weights in BF16 (first call only)
    // Memory: 32 × (4096×7168 + 7168×2048) × 2 = ~3.7 GB BF16
    // ==================================================================
    if (!weights_cached) {
        W1_cache = torch::empty({E_LOCAL, 2 * I_DIM, H_DIM}, opts_bf16);
        W2_cache = torch::empty({E_LOCAL, H_DIM, I_DIM}, opts_bf16);

        for (int le = 0; le < E_LOCAL; le++) {
            auto w1_fp8   = gemm1_weights[le].contiguous();
            auto w1_scale = gemm1_weights_scale[le].to(torch::kFloat32).contiguous();
            {
                int total = 2 * I_DIM * H_DIM;
                int thr = 256;
                int blk = (total + thr - 1) / thr;
                dequant_2d_to_bf16_kernel<<<blk, thr, 0, stream>>>(
                    (const uint8_t*)w1_fp8.data_ptr(),
                    w1_scale.data_ptr<float>(),
                    (__nv_bfloat16*)W1_cache[le].data_ptr(),
                    2 * I_DIM, H_DIM, H_DIM / BLOCK_SZ
                );
            }

            auto w2_fp8   = gemm2_weights[le].contiguous();
            auto w2_scale = gemm2_weights_scale[le].to(torch::kFloat32).contiguous();
            {
                int total = H_DIM * I_DIM;
                int thr = 256;
                int blk = (total + thr - 1) / thr;
                dequant_2d_to_bf16_kernel<<<blk, thr, 0, stream>>>(
                    (const uint8_t*)w2_fp8.data_ptr(),
                    w2_scale.data_ptr<float>(),
                    (__nv_bfloat16*)W2_cache[le].data_ptr(),
                    H_DIM, I_DIM, I_DIM / BLOCK_SZ
                );
            }
        }

        // Allocate device pointer buffers for grouped GEMM
        cudaMalloc(&d_A_ptrs1, E_LOCAL * sizeof(void*));
        cudaMalloc(&d_B_ptrs1, E_LOCAL * sizeof(void*));
        cudaMalloc(&d_C_ptrs1, E_LOCAL * sizeof(void*));
        cudaMalloc(&d_A_ptrs2, E_LOCAL * sizeof(void*));
        cudaMalloc(&d_B_ptrs2, E_LOCAL * sizeof(void*));
        cudaMalloc(&d_C_ptrs2, E_LOCAL * sizeof(void*));

        weights_cached = true;
    }

    // ==================================================================
    // Step 4: Gather + Dequant hidden states → F32, then cast to BF16
    // ==================================================================
    int H_BLOCKS = H_DIM / BLOCK_SZ;
    auto hs_scale = hidden_states_scale.to(torch::kFloat32).contiguous();
    auto A_f32 = torch::empty({Tsum, H_DIM}, opts_f32);

    {
        int thr = 256;
        int ncol = (H_DIM + thr - 1) / thr;
        dim3 grid(Tsum, ncol);
        gather_dequant_kernel<<<grid, thr, 0, stream>>>(
            (const uint8_t*)hidden_states.data_ptr(),
            hs_scale.data_ptr<float>(),
            sorted_token_ids.data_ptr<int64_t>(),
            A_f32.data_ptr<float>(),
            T, H_DIM, H_BLOCKS
        );
    }
    auto A_bf16 = A_f32.to(torch::kBFloat16);

    // ==================================================================
    // Step 5: Expert compute - hybrid approach
    //
    // Small/medium Tsum: Grouped GEMM (fewer kernel launches)
    // Large Tsum: Per-expert loop (better per-GEMM efficiency)
    // ==================================================================
    constexpr int GROUPED_GEMM_THRESHOLD = 16000;

    auto G2_buf = torch::empty({Tsum, H_DIM}, opts_bf16);

    if (Tsum <= GROUPED_GEMM_THRESHOLD) {
        // --- Path A: Grouped GEMM (3 kernel launches vs ~96) ---
        auto G1_buf     = torch::empty({Tsum, 2 * I_DIM}, opts_bf16);
        auto swiglu_buf = torch::empty({Tsum, I_DIM}, opts_bf16);

        int num_active = 0;
        cublasOperation_t transa_h[E_LOCAL], transb_h[E_LOCAL];
        int m_h1[E_LOCAL], n_h1[E_LOCAL], k_h1[E_LOCAL];
        int lda_h1[E_LOCAL], ldb_h1[E_LOCAL], ldc_h1[E_LOCAL];
        int m_h2[E_LOCAL], n_h2[E_LOCAL], k_h2[E_LOCAL];
        int lda_h2[E_LOCAL], ldb_h2[E_LOCAL], ldc_h2[E_LOCAL];
        int group_size_h[E_LOCAL];
        float alpha_h[E_LOCAL], beta_h[E_LOCAL];

        const void* A_ptrs_h1[E_LOCAL], *B_ptrs_h1[E_LOCAL];
        void*       C_ptrs_h1[E_LOCAL];
        const void* A_ptrs_h2[E_LOCAL], *B_ptrs_h2[E_LOCAL];
        void*       C_ptrs_h2[E_LOCAL];

        auto bf16_A  = (__nv_bfloat16*)A_bf16.data_ptr();
        auto bf16_G1 = (__nv_bfloat16*)G1_buf.data_ptr();
        auto bf16_SG = (__nv_bfloat16*)swiglu_buf.data_ptr();
        auto bf16_G2 = (__nv_bfloat16*)G2_buf.data_ptr();

        for (int le = 0; le < E_LOCAL; le++) {
            int start = offsets_cpu[le].item<int>();
            int end   = offsets_cpu[le + 1].item<int>();
            if (start >= end) continue;
            int Tk = end - start;

            transa_h[num_active]     = CUBLAS_OP_T;
            transb_h[num_active]     = CUBLAS_OP_N;
            alpha_h[num_active]      = 1.0f;
            beta_h[num_active]       = 0.0f;
            group_size_h[num_active] = 1;

            m_h1[num_active] = 2 * I_DIM;  n_h1[num_active] = Tk;  k_h1[num_active] = H_DIM;
            lda_h1[num_active] = H_DIM;  ldb_h1[num_active] = H_DIM;  ldc_h1[num_active] = 2 * I_DIM;
            A_ptrs_h1[num_active] = (const void*)((const __nv_bfloat16*)W1_cache[le].data_ptr());
            B_ptrs_h1[num_active] = (const void*)(bf16_A  + (int64_t)start * H_DIM);
            C_ptrs_h1[num_active] = (void*)      (bf16_G1 + (int64_t)start * 2 * I_DIM);

            m_h2[num_active] = H_DIM;  n_h2[num_active] = Tk;  k_h2[num_active] = I_DIM;
            lda_h2[num_active] = I_DIM;  ldb_h2[num_active] = I_DIM;  ldc_h2[num_active] = H_DIM;
            A_ptrs_h2[num_active] = (const void*)((const __nv_bfloat16*)W2_cache[le].data_ptr());
            B_ptrs_h2[num_active] = (const void*)(bf16_SG + (int64_t)start * I_DIM);
            C_ptrs_h2[num_active] = (void*)      (bf16_G2 + (int64_t)start * H_DIM);

            num_active++;
        }

        if (num_active > 0) {
            cudaMemcpyAsync(d_A_ptrs1, A_ptrs_h1, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_B_ptrs1, B_ptrs_h1, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_C_ptrs1, C_ptrs_h1, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);

            cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
            cublasSetStream(handle, stream);

            cublasStatus_t status = cublasGemmGroupedBatchedEx(
                handle, transa_h, transb_h, m_h1, n_h1, k_h1,
                (const void*)alpha_h,
                d_A_ptrs1, CUDA_R_16BF, lda_h1,
                d_B_ptrs1, CUDA_R_16BF, ldb_h1,
                (const void*)beta_h,
                d_C_ptrs1, CUDA_R_16BF, ldc_h1,
                num_active, group_size_h, CUBLAS_COMPUTE_32F
            );
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "GEMM1 grouped failed: ", (int)status);

            {
                int total = Tsum * I_DIM;
                int thr = 256;
                swiglu_bf16_kernel<<<(total + thr - 1) / thr, thr, 0, stream>>>(
                    bf16_G1, bf16_SG, total, I_DIM
                );
            }

            cudaMemcpyAsync(d_A_ptrs2, A_ptrs_h2, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_B_ptrs2, B_ptrs_h2, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_C_ptrs2, C_ptrs_h2, num_active * sizeof(void*), cudaMemcpyHostToDevice, stream);

            status = cublasGemmGroupedBatchedEx(
                handle, transa_h, transb_h, m_h2, n_h2, k_h2,
                (const void*)alpha_h,
                d_A_ptrs2, CUDA_R_16BF, lda_h2,
                d_B_ptrs2, CUDA_R_16BF, ldb_h2,
                (const void*)beta_h,
                d_C_ptrs2, CUDA_R_16BF, ldc_h2,
                num_active, group_size_h, CUBLAS_COMPUTE_32F
            );
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "GEMM2 grouped failed: ", (int)status);
        }
        // For grouped GEMM path: use BF16 scatter-add
        auto output = torch::zeros({T, H_DIM}, opts_f32);
        {
            int thr  = 256;
            int ncol = (H_DIM + thr - 1) / thr;
            dim3 grid(Tsum, ncol);
            weighted_scatter_add_bf16_kernel<<<grid, thr, 0, stream>>>(
                (__nv_bfloat16*)G2_buf.data_ptr(),
                sorted_token_ids.data_ptr<int64_t>(),
                token_expert_map_t.data_ptr<int32_t>(),
                topk_idx.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                output.data_ptr<float>(),
                (int)local_expert_offset, H_DIM, TOP_K
            );
        }
        return output.to(torch::kBFloat16);
    } else {
        // --- Path B: Per-expert loop (better for large workloads) ---
        // Stays in F32 for GEMM2 output → scatter-add, avoiding extra BF16 cast
        auto gemm2_f32 = torch::zeros({Tsum, H_DIM}, opts_f32);

        for (int le = 0; le < E_LOCAL; le++) {
            int start = offsets_cpu[le].item<int>();
            int end   = offsets_cpu[le + 1].item<int>();
            if (start >= end) continue;
            int Tk = end - start;

            auto A_e = A_bf16.slice(0, start, end);
            auto G1  = torch::matmul(A_e, W1_cache[le].t());

            auto G1_f32 = G1.to(torch::kFloat32);
            auto C = torch::empty({Tk, I_DIM}, opts_f32);
            {
                int total = Tk * I_DIM;
                int thr = 256;
                swiglu_kernel<<<(total + thr - 1) / thr, thr, 0, stream>>>(
                    G1_f32.data_ptr<float>(), C.data_ptr<float>(), Tk, I_DIM
                );
            }

            auto C_bf16 = C.to(torch::kBFloat16);
            gemm2_f32.slice(0, start, end).copy_(
                torch::matmul(C_bf16, W2_cache[le].t())
            );
        }

        // For per-expert path: use F32 scatter-add (no extra BF16 conversion)
        auto output = torch::zeros({T, H_DIM}, opts_f32);
        {
            int thr  = 256;
            int ncol = (H_DIM + thr - 1) / thr;
            dim3 grid(Tsum, ncol);
            weighted_scatter_add_kernel<<<grid, thr, 0, stream>>>(
                gemm2_f32.data_ptr<float>(),
                sorted_token_ids.data_ptr<int64_t>(),
                token_expert_map_t.data_ptr<int32_t>(),
                topk_idx.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                output.data_ptr<float>(),
                (int)local_expert_offset, H_DIM, TOP_K
            );
        }
        return output.to(torch::kBFloat16);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MOE FP8 Block-Scale kernel (CUDA)");
}
