/*
 * CUDA Multi-Kernel MOE Implementation for FlashInfer-Bench
 *
 * Implements DeepSeek-V3 style Mixture-of-Experts with FP8 block-scale
 * quantization as a PyTorch C++ extension (pybind11).
 *
 * Non-GEMM ops  → custom CUDA kernels
 * GEMM1 / GEMM2 → torch::matmul (cuBLAS)
 *
 * Constants (DeepSeek-V3/R1 geometry):
 *   H=7168, I=2048, E_GLOBAL=256, E_LOCAL=32
 *   TOP_K=8, N_GROUP=8, TOPK_GROUP=4, BLOCK=128
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================
#define H_DIM       7168
#define I_DIM       2048
#define E_GLOBAL    256
#define E_LOCAL     32
#define TOP_K       8
#define N_GROUP     8
#define TOPK_GROUP  4
#define BLOCK_SZ    128

// ============================================================================
// FP8 E4M3FN → float32 software conversion
// ============================================================================
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    uint32_t sign = (uint32_t)(x >> 7);
    uint32_t exp  = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }
    if (exp == 0) {
        float val = (float)mant * (1.0f / 512.0f);
        return sign ? -val : val;
    }
    if (exp == 15 && mant == 7) {
        return __uint_as_float(0x7FC00000);  // NaN
    }
    uint32_t f32 = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
    return __uint_as_float(f32);
}


// ============================================================================
// Kernel 1: DeepSeek-V3 No-Aux Routing
// Grid: (T,), Block: (256,)
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
        // Group scoring: sum of top-2 per group
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

        // Top-4 group selection
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

        // Pruned scores (only selected groups)
        float pruned[E_GLOBAL];
        for (int i = 0; i < E_GLOBAL; i++) {
            pruned[i] = selected_groups[i / 32] ? s_scores[i] : -1e38f;
        }

        // Top-8 experts
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

        // Normalize and scale
        w_sum += 1e-20f;
        for (int k = 0; k < TOP_K; k++) {
            weights_out[token * TOP_K + k] =
                (weights_out[token * TOP_K + k] / w_sum) * routed_scaling_factor;
        }
    }
}


// ============================================================================
// Kernel 2a: Permute — Count tokens per local expert
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


// ============================================================================
// Kernel 2b: Permute — Exclusive prefix sum
// ============================================================================
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


// ============================================================================
// Kernel 2c: Permute — Scatter tokens to sorted positions
// ============================================================================
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
// Kernel 3: Fused Gather + Dequant (hidden states)
// Grid: (Tsum, ceil(H/256)), Block: (256,)
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
// Kernel 4: 2D block-scale dequantization (for expert weights)
// ============================================================================
__global__ void dequant_2d_blockscale_kernel(
    const uint8_t* __restrict__ x_fp8,
    const float*   __restrict__ scale,
    float*         __restrict__ output,
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
    output[idx] = fp8e4m3_to_float(x_fp8[idx]) * s;
}


// ============================================================================
// Kernel 5: SwiGLU activation
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
// Kernel 6: Weighted scatter-add
// Grid: (Tsum, ceil(H/256)), Block: (256,)
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
// Host entry point: full MOE pipeline
//
// Exported as "kernel" via pybind11 for the FlashInfer-Bench CUDA builder.
// ============================================================================
torch::Tensor kernel(
    torch::Tensor routing_logits,       // [T, E_GLOBAL] float32
    torch::Tensor routing_bias,         // [E_GLOBAL] float32
    torch::Tensor hidden_states,        // [T, H] float8_e4m3fn
    torch::Tensor hidden_states_scale,  // [H/BLOCK, T] float32
    torch::Tensor gemm1_weights,        // [E_LOCAL, 2*I, H] float8_e4m3fn
    torch::Tensor gemm1_weights_scale,  // [E_LOCAL, (2*I)/BLOCK, H/BLOCK] float32
    torch::Tensor gemm2_weights,        // [E_LOCAL, H, I] float8_e4m3fn
    torch::Tensor gemm2_weights_scale,  // [E_LOCAL, H/BLOCK, I/BLOCK] float32
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int T = routing_logits.size(0);
    auto device = routing_logits.device();
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Ensure float32 contiguous routing inputs
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
    // Step 2: Token permutation (3 kernels)
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
        return torch::zeros({T, H_DIM},
            torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    }

    auto sorted_token_ids = sorted_ids_full.slice(0, 0, Tsum).contiguous();
    auto token_expert_map = expert_map_full.slice(0, 0, Tsum).contiguous();

    // ==================================================================
    // Step 3: Gather + Dequant hidden states
    // ==================================================================
    int H_BLOCKS = H_DIM / BLOCK_SZ;
    auto hs_scale = hidden_states_scale.to(torch::kFloat32).contiguous();
    auto A = torch::empty({Tsum, H_DIM}, opts_f32);

    {
        int thr = 256;
        int ncol = (H_DIM + thr - 1) / thr;
        dim3 grid(Tsum, ncol);
        gather_dequant_kernel<<<grid, thr, 0, stream>>>(
            (const uint8_t*)hidden_states.data_ptr(),
            hs_scale.data_ptr<float>(),
            sorted_token_ids.data_ptr<int64_t>(),
            A.data_ptr<float>(),
            T, H_DIM, H_BLOCKS
        );
    }

    // ==================================================================
    // Step 4: Per-expert compute loop
    // ==================================================================
    auto gemm2_buffer = torch::zeros({Tsum, H_DIM}, opts_f32);
    auto W1_buf = torch::empty({2 * I_DIM, H_DIM}, opts_f32);
    auto W2_buf = torch::empty({H_DIM, I_DIM}, opts_f32);

    for (int le = 0; le < E_LOCAL; le++) {
        int start = offsets_cpu[le].item<int>();
        int end   = offsets_cpu[le + 1].item<int>();
        if (start >= end) continue;
        int Tk = end - start;

        // 4a: Dequant weight1 for this expert
        auto w1_fp8   = gemm1_weights[le].contiguous();
        auto w1_scale = gemm1_weights_scale[le].to(torch::kFloat32).contiguous();
        {
            int total = 2 * I_DIM * H_DIM;
            int thr = 256;
            int blk = (total + thr - 1) / thr;
            dequant_2d_blockscale_kernel<<<blk, thr, 0, stream>>>(
                (const uint8_t*)w1_fp8.data_ptr(),
                w1_scale.data_ptr<float>(),
                W1_buf.data_ptr<float>(),
                2 * I_DIM, H_DIM, H_DIM / BLOCK_SZ
            );
        }

        // 4b: GEMM1 — [Tk, H] @ [H, 2*I] → [Tk, 2*I]
        auto A_e = A.slice(0, start, end);
        auto G1  = torch::matmul(A_e, W1_buf.t());

        // 4c: SwiGLU
        auto C = torch::empty({Tk, I_DIM}, opts_f32);
        {
            int total = Tk * I_DIM;
            int thr = 256;
            int blk = (total + thr - 1) / thr;
            swiglu_kernel<<<blk, thr, 0, stream>>>(
                G1.data_ptr<float>(),
                C.data_ptr<float>(),
                Tk, I_DIM
            );
        }

        // 4d: Dequant weight2 for this expert
        auto w2_fp8   = gemm2_weights[le].contiguous();
        auto w2_scale = gemm2_weights_scale[le].to(torch::kFloat32).contiguous();
        {
            int total = H_DIM * I_DIM;
            int thr = 256;
            int blk = (total + thr - 1) / thr;
            dequant_2d_blockscale_kernel<<<blk, thr, 0, stream>>>(
                (const uint8_t*)w2_fp8.data_ptr(),
                w2_scale.data_ptr<float>(),
                W2_buf.data_ptr<float>(),
                H_DIM, I_DIM, I_DIM / BLOCK_SZ
            );
        }

        // 4e: GEMM2 — [Tk, I] @ [I, H] → [Tk, H]
        gemm2_buffer.slice(0, start, end).copy_(torch::matmul(C, W2_buf.t()));
    }

    // ==================================================================
    // Step 5: Weighted scatter-add
    // ==================================================================
    auto output = torch::zeros({T, H_DIM}, opts_f32);

    {
        int thr  = 256;
        int ncol = (H_DIM + thr - 1) / thr;
        dim3 grid(Tsum, ncol);
        weighted_scatter_add_kernel<<<grid, thr, 0, stream>>>(
            gemm2_buffer.data_ptr<float>(),
            sorted_token_ids.data_ptr<int64_t>(),
            token_expert_map.data_ptr<int32_t>(),
            topk_idx.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            output.data_ptr<float>(),
            (int)local_expert_offset, H_DIM, TOP_K
        );
    }

    // ==================================================================
    // Step 6: Cast to bfloat16
    // ==================================================================
    return output.to(torch::kBFloat16);
}


// ============================================================================
// pybind11 module
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MOE FP8 Block-Scale kernel (CUDA)");
}
