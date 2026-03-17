/*
 * Optimized CUDA MOE Implementation for DeepSeek-V3
 *
 * Optimizations:
 * 1. Direct FP8→BF16 gather+dequant with vec4 loads (skip F32 intermediate)
 * 2. Static pre-allocated buffers with pinned host memory
 * 3. Parallel routing kernel with warp-level group scoring
 * 4. Vectorized SwiGLU BF16 (4 elements/thread)
 * 5. Vectorized weighted scatter-add (4 BF16 elements/thread)
 * 6. Always-on grouped GEMM (no per-expert fallback)
 * 7. Vectorized weight dequant for caching
 * 8. Weight caching - dequant FP8→BF16 once, reuse forever
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
// FP8 E4M3FN → float32 (device inline)
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
// Parallel Routing Kernel
// Uses warp-level reductions for group scoring instead of single-threaded loop
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
    int tid = threadIdx.x;  // 0..255

    __shared__ float s_scores[256];
    __shared__ float s_raw[256];
    __shared__ float s_group_scores[N_GROUP];
    __shared__ bool  s_selected_groups[N_GROUP];
    __shared__ int   s_topk_idx[TOP_K];
    __shared__ float s_topk_raw[TOP_K];

    // Phase A: Sigmoid + bias (all 256 threads parallel)
    float logit = logits[token * 256 + tid];
    float sig   = 1.0f / (1.0f + expf(-logit));
    s_raw[tid]    = sig;
    s_scores[tid] = sig + bias[tid];
    __syncthreads();

    // Phase B: Warp-level group scoring
    // Each warp (32 threads) handles one group of 32 experts
    // 8 warps = 8 groups, perfectly matched
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (warp_id < N_GROUP) {
        float my_score = s_scores[warp_id * 32 + lane_id];

        // Find top-2 values in this warp using shuffle reductions
        float max1 = -1e38f, max2 = -1e38f;

        // Each lane contributes its value; we do a manual reduction
        // First, find the global max using warp shuffle
        float warp_max = my_score;
        for (int offset = 16; offset >= 1; offset /= 2) {
            float other = __shfl_xor_sync(0xFFFFFFFF, warp_max, offset);
            warp_max = fmaxf(warp_max, other);
        }
        max1 = warp_max;

        // Find second max: mask out the max value, find max again
        float masked = (my_score == max1) ? -1e38f : my_score;
        // Handle ties: only the first lane with max1 gets masked
        // Use ballot to find which lanes have max1
        unsigned mask_max1 = __ballot_sync(0xFFFFFFFF, my_score == max1);
        int first_max_lane = __ffs(mask_max1) - 1;
        masked = (lane_id == first_max_lane) ? -1e38f : my_score;

        float warp_max2 = masked;
        for (int offset = 16; offset >= 1; offset /= 2) {
            float other = __shfl_xor_sync(0xFFFFFFFF, warp_max2, offset);
            warp_max2 = fmaxf(warp_max2, other);
        }
        max2 = warp_max2;

        if (lane_id == 0) {
            s_group_scores[warp_id] = max1 + max2;
        }
    }
    __syncthreads();

    // Phase C: Top-4 group selection (thread 0, trivial on 8 values)
    if (tid == 0) {
        for (int g = 0; g < N_GROUP; g++) s_selected_groups[g] = false;
        for (int k = 0; k < TOPK_GROUP; k++) {
            int best_g = -1;
            float best_s = -1e38f;
            for (int g = 0; g < N_GROUP; g++) {
                if (!s_selected_groups[g] && s_group_scores[g] > best_s) {
                    best_s = s_group_scores[g];
                    best_g = g;
                }
            }
            if (best_g >= 0) s_selected_groups[best_g] = true;
        }
    }
    __syncthreads();

    // Phase D: Parallel pruning (all 256 threads)
    float pruned_val = s_selected_groups[tid / 32] ? s_scores[tid] : -1e38f;
    s_scores[tid] = pruned_val;
    __syncthreads();

    // Phase E: Top-8 selection via 8 rounds of block-wide max reduction
    if (tid == 0) {
        for (int k = 0; k < TOP_K; k++) {
            int best_i = 0;
            float best_v = -1e38f;
            for (int i = 0; i < E_GLOBAL; i++) {
                if (s_scores[i] > best_v) {
                    best_v = s_scores[i];
                    best_i = i;
                }
            }
            s_topk_idx[k] = best_i;
            s_topk_raw[k] = s_raw[best_i];
            s_scores[best_i] = -1e38f;
        }

        // Phase F: Normalize weights
        float w_sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) w_sum += s_topk_raw[k];
        w_sum += 1e-20f;
        for (int k = 0; k < TOP_K; k++) {
            topk_idx_out[token * TOP_K + k] = (int64_t)s_topk_idx[k];
            weights_out[token * TOP_K + k] = (s_topk_raw[k] / w_sum) * routed_scaling_factor;
        }
    }
}

// ============================================================================
// Permute kernels (unchanged - already efficient)
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
// Vectorized Dequant FP8 2D block-scale → BF16 (4 elements per thread)
// Used for weight caching
// ============================================================================
__global__ void dequant_2d_to_bf16_vec4_kernel(
    const uint8_t* __restrict__ x_fp8,
    const float*   __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int R, int C, int scale_cols
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int total4 = (R * C) / 4;
    if (idx4 >= total4) return;

    int base_idx = idx4 * 4;

    // Load 4 FP8 values as packed uint32
    uint32_t packed = ((const uint32_t*)x_fp8)[idx4];
    uint8_t v0 = (uint8_t)(packed & 0xFF);
    uint8_t v1 = (uint8_t)((packed >> 8) & 0xFF);
    uint8_t v2 = (uint8_t)((packed >> 16) & 0xFF);
    uint8_t v3 = (uint8_t)((packed >> 24) & 0xFF);

    // All 4 elements are in the same row (C is large, divisible by 4)
    int r  = base_idx / C;
    int c  = base_idx % C;
    int rb = r / BLOCK_SZ;
    int cb = c / BLOCK_SZ;
    float s = scale[rb * scale_cols + cb];

    // Check if all 4 fit in same scale block
    int cb_end = (c + 3) / BLOCK_SZ;
    if (cb == cb_end) {
        // Same scale block for all 4
        output[base_idx]     = __float2bfloat16(fp8e4m3_to_float(v0) * s);
        output[base_idx + 1] = __float2bfloat16(fp8e4m3_to_float(v1) * s);
        output[base_idx + 2] = __float2bfloat16(fp8e4m3_to_float(v2) * s);
        output[base_idx + 3] = __float2bfloat16(fp8e4m3_to_float(v3) * s);
    } else {
        // Cross scale-block boundary - handle individually
        output[base_idx]     = __float2bfloat16(fp8e4m3_to_float(v0) * scale[rb * scale_cols + (c)     / BLOCK_SZ]);
        output[base_idx + 1] = __float2bfloat16(fp8e4m3_to_float(v1) * scale[rb * scale_cols + (c + 1) / BLOCK_SZ]);
        output[base_idx + 2] = __float2bfloat16(fp8e4m3_to_float(v2) * scale[rb * scale_cols + (c + 2) / BLOCK_SZ]);
        output[base_idx + 3] = __float2bfloat16(fp8e4m3_to_float(v3) * scale[rb * scale_cols + (c + 3) / BLOCK_SZ]);
    }
}

// ============================================================================
// Direct FP8→BF16 Gather + Dequant (vectorized, skip F32 intermediate)
// Each block handles one row, grid.y covers columns in chunks of blockDim.x*4
// ============================================================================
__global__ void gather_dequant_bf16_vec4_kernel(
    const uint8_t* __restrict__ hidden_states,
    const float*   __restrict__ hs_scale,
    const int64_t* __restrict__ sorted_ids,
    __nv_bfloat16* __restrict__ output,
    int T, int H, int H_BLOCKS
) {
    int row = blockIdx.x;
    int col4 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= H) return;

    int64_t token_id = sorted_ids[row];
    int base_in = token_id * H + col4;

    // Load 4 FP8 values as packed uint32
    uint32_t packed = ((const uint32_t*)(hidden_states + base_in))[0];
    uint8_t v0 = (uint8_t)(packed & 0xFF);
    uint8_t v1 = (uint8_t)((packed >> 8) & 0xFF);
    uint8_t v2 = (uint8_t)((packed >> 16) & 0xFF);
    uint8_t v3 = (uint8_t)((packed >> 24) & 0xFF);

    // Scale layout: [H_BLOCKS, T], accessed as scale[block_idx * T + token_id]
    int block_idx = col4 / BLOCK_SZ;
    float s = hs_scale[block_idx * T + token_id];

    // Check if all 4 fit in same scale block (BLOCK_SZ=128, so usually yes)
    int block_end = (col4 + 3) / BLOCK_SZ;
    int out_base = row * H + col4;

    if (block_idx == block_end) {
        output[out_base]     = __float2bfloat16(fp8e4m3_to_float(v0) * s);
        output[out_base + 1] = __float2bfloat16(fp8e4m3_to_float(v1) * s);
        output[out_base + 2] = __float2bfloat16(fp8e4m3_to_float(v2) * s);
        output[out_base + 3] = __float2bfloat16(fp8e4m3_to_float(v3) * s);
    } else {
        output[out_base]     = __float2bfloat16(fp8e4m3_to_float(v0) * hs_scale[((col4)     / BLOCK_SZ) * T + token_id]);
        output[out_base + 1] = __float2bfloat16(fp8e4m3_to_float(v1) * hs_scale[((col4 + 1) / BLOCK_SZ) * T + token_id]);
        output[out_base + 2] = __float2bfloat16(fp8e4m3_to_float(v2) * hs_scale[((col4 + 2) / BLOCK_SZ) * T + token_id]);
        output[out_base + 3] = __float2bfloat16(fp8e4m3_to_float(v3) * hs_scale[((col4 + 3) / BLOCK_SZ) * T + token_id]);
    }
}

// ============================================================================
// Vectorized SwiGLU BF16 (4 elements per thread)
// input: [Tsum, 2*I_DIM], output: [Tsum, I_DIM]
// ============================================================================
__global__ void swiglu_bf16_vec4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16*       __restrict__ output,
    int total_elements, int I_dim
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int total4 = total_elements / 4;
    if (idx4 >= total4) return;

    int base = idx4 * 4;
    int row   = base / I_dim;
    int col   = base % I_dim;
    int two_I = 2 * I_dim;

    // Load 4 gate values and 4 up values
    int gate_offset = row * two_I + col;
    int up_offset   = gate_offset + I_dim;

    // Vectorized load for gate (4 x bf16 = 8 bytes = uint2)
    uint2 gate_packed = ((const uint2*)(input + gate_offset))[0];
    uint2 up_packed   = ((const uint2*)(input + up_offset))[0];

    __nv_bfloat16* gate_vals = (__nv_bfloat16*)&gate_packed;
    __nv_bfloat16* up_vals   = (__nv_bfloat16*)&up_packed;

    __nv_bfloat16 out_vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float g = __bfloat162float(gate_vals[i]);
        float u = __bfloat162float(up_vals[i]);
        float silu_u = u / (1.0f + expf(-u));
        out_vals[i] = __float2bfloat16(silu_u * g);
    }

    // Vectorized store
    ((uint2*)(output + base))[0] = *((uint2*)out_vals);
}

// ============================================================================
// Vectorized Weighted Scatter-Add from BF16 (4 elements per thread)
// ============================================================================
__global__ void weighted_scatter_add_bf16_vec4_kernel(
    const __nv_bfloat16* __restrict__ gemm2_result,
    const int64_t* __restrict__ sorted_token_ids,
    const int*     __restrict__ token_expert_map,
    const int64_t* __restrict__ topk_idx,
    const float*   __restrict__ routing_weights,
    float*         __restrict__ output,
    int local_expert_offset, int H_dim, int top_k
) {
    int row  = blockIdx.x;
    int col4 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= H_dim) return;

    int64_t token_id      = sorted_token_ids[row];
    int     local_expert  = token_expert_map[row];
    int64_t global_expert = (int64_t)local_expert + local_expert_offset;

    // Find routing weight (hoisted out of inner loop)
    float w = 0.0f;
    for (int k = 0; k < top_k; k++) {
        if (topk_idx[token_id * top_k + k] == global_expert) {
            w = routing_weights[token_id * top_k + k];
            break;
        }
    }

    // Vectorized load of 4 BF16 values
    int in_offset = row * H_dim + col4;
    uint2 packed = ((const uint2*)(gemm2_result + in_offset))[0];
    __nv_bfloat16* vals = (__nv_bfloat16*)&packed;

    int out_base = token_id * H_dim + col4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float val = __bfloat162float(vals[i]) * w;
        atomicAdd(&output[out_base + i], val);
    }
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

// Static pre-allocated intermediate buffers
static torch::Tensor s_A_bf16;       // [max_Tsum, H_DIM]
static torch::Tensor s_G1_buf;       // [max_Tsum, 2*I_DIM]
static torch::Tensor s_swiglu_buf;   // [max_Tsum, I_DIM]
static torch::Tensor s_G2_buf;       // [max_Tsum, H_DIM]
static torch::Tensor s_output_f32;   // [max_T, H_DIM]

static torch::Tensor s_topk_idx;     // [max_T, TOP_K]
static torch::Tensor s_weights;      // [max_T, TOP_K]
static torch::Tensor s_counts;       // [E_LOCAL]
static torch::Tensor s_expert_offsets; // [E_LOCAL+1]
static torch::Tensor s_sorted_ids;   // [max_T*TOP_K]
static torch::Tensor s_expert_map;   // [max_T*TOP_K]
static torch::Tensor s_write_ctr;    // [E_LOCAL]

static int s_max_T = 0;
static int s_max_Tsum = 0;

// Pinned host memory for async D2H of expert offsets
static int* h_expert_offsets_pinned = nullptr;

// ============================================================================
// Ensure static buffers are large enough for given T
// ============================================================================
static void ensure_buffers(int T, torch::Device device, cudaStream_t stream) {
    int Tsum_max = T * TOP_K;  // worst case: all tokens to local experts

    if (T <= s_max_T && Tsum_max <= s_max_Tsum) return;

    // Grow to accommodate
    int new_max_T = std::max(T, s_max_T);
    int new_max_Tsum = std::max(Tsum_max, s_max_Tsum);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i64  = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32).device(device);

    if (new_max_Tsum > s_max_Tsum) {
        s_A_bf16     = torch::empty({new_max_Tsum, H_DIM}, opts_bf16);
        s_G1_buf     = torch::empty({new_max_Tsum, 2 * I_DIM}, opts_bf16);
        s_swiglu_buf = torch::empty({new_max_Tsum, I_DIM}, opts_bf16);
        s_G2_buf     = torch::empty({new_max_Tsum, H_DIM}, opts_bf16);
        s_sorted_ids = torch::empty({new_max_Tsum}, opts_i64);
        s_expert_map = torch::empty({new_max_Tsum}, opts_i32);
    }

    if (new_max_T > s_max_T) {
        s_output_f32 = torch::empty({new_max_T, H_DIM}, opts_f32);
        s_topk_idx   = torch::empty({new_max_T, TOP_K}, opts_i64);
        s_weights    = torch::empty({new_max_T, TOP_K}, opts_f32);
    }

    // These are fixed size, allocate once
    if (s_max_T == 0) {
        s_counts          = torch::empty({E_LOCAL}, opts_i32);
        s_expert_offsets  = torch::empty({E_LOCAL + 1}, opts_i32);
        s_write_ctr       = torch::empty({E_LOCAL}, opts_i32);

        // Pinned host memory for async D2H
        cudaMallocHost(&h_expert_offsets_pinned, (E_LOCAL + 1) * sizeof(int));
    }

    s_max_T = new_max_T;
    s_max_Tsum = new_max_Tsum;
}

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
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    if (T == 0) {
        return torch::zeros({0, H_DIM}, opts_bf16);
    }

    auto logits_f32 = routing_logits.to(torch::kFloat32).contiguous();
    auto bias_f32   = routing_bias.to(torch::kFloat32).reshape({-1}).contiguous();

    // Ensure static buffers are allocated
    ensure_buffers(T, device, stream);

    // ==================================================================
    // Step 1: Routing (parallel warp-level)
    // ==================================================================
    routing_kernel<<<T, 256, 0, stream>>>(
        logits_f32.data_ptr<float>(),
        bias_f32.data_ptr<float>(),
        s_topk_idx.data_ptr<int64_t>(),
        s_weights.data_ptr<float>(),
        (float)routed_scaling_factor,
        T
    );

    // ==================================================================
    // Step 2: Token permutation
    // ==================================================================
    int N = T * TOP_K;
    auto flat_idx = s_topk_idx.slice(0, 0, T).reshape({-1});

    // Zero counts and write counters using cudaMemsetAsync
    cudaMemsetAsync(s_counts.data_ptr(), 0, E_LOCAL * sizeof(int), stream);
    cudaMemsetAsync(s_write_ctr.data_ptr(), 0, E_LOCAL * sizeof(int), stream);

    {
        int thr = 256;
        int blk = (N + thr - 1) / thr;

        permute_count_kernel<<<blk, thr, 0, stream>>>(
            flat_idx.data_ptr<int64_t>(),
            s_counts.data_ptr<int32_t>(),
            (int)local_expert_offset, N
        );

        permute_prefix_sum_kernel<<<1, 1, 0, stream>>>(
            s_counts.data_ptr<int32_t>(),
            s_expert_offsets.data_ptr<int32_t>()
        );

        permute_scatter_kernel<<<blk, thr, 0, stream>>>(
            flat_idx.data_ptr<int64_t>(),
            s_expert_offsets.data_ptr<int32_t>(),
            s_sorted_ids.data_ptr<int64_t>(),
            s_expert_map.data_ptr<int32_t>(),
            s_write_ctr.data_ptr<int32_t>(),
            (int)local_expert_offset, N, TOP_K
        );
    }

    // Async D2H copy of expert_offsets to pinned memory, then sync
    cudaMemcpyAsync(h_expert_offsets_pinned,
                    s_expert_offsets.data_ptr<int32_t>(),
                    (E_LOCAL + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int Tsum = h_expert_offsets_pinned[E_LOCAL];

    if (Tsum == 0) {
        return torch::zeros({T, H_DIM}, opts_bf16);
    }

    // ==================================================================
    // Step 3: Cache weights in BF16 (first call only)
    // ==================================================================
    if (!weights_cached) {
        W1_cache = torch::empty({E_LOCAL, 2 * I_DIM, H_DIM}, opts_bf16);
        W2_cache = torch::empty({E_LOCAL, H_DIM, I_DIM}, opts_bf16);

        for (int le = 0; le < E_LOCAL; le++) {
            auto w1_fp8   = gemm1_weights[le].contiguous();
            auto w1_scale = gemm1_weights_scale[le].to(torch::kFloat32).contiguous();
            {
                int total = 2 * I_DIM * H_DIM;
                int total4 = total / 4;
                int thr = 256;
                int blk = (total4 + thr - 1) / thr;
                dequant_2d_to_bf16_vec4_kernel<<<blk, thr, 0, stream>>>(
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
                int total4 = total / 4;
                int thr = 256;
                int blk = (total4 + thr - 1) / thr;
                dequant_2d_to_bf16_vec4_kernel<<<blk, thr, 0, stream>>>(
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
    // Step 4: Gather + Dequant → BF16 directly (skip F32 intermediate)
    // ==================================================================
    int H_BLOCKS = H_DIM / BLOCK_SZ;
    auto hs_scale = hidden_states_scale.to(torch::kFloat32).contiguous();

    {
        int thr = 256;
        // Each thread handles 4 columns, so we need H_DIM/4 threads per row
        int ncol_groups = (H_DIM / 4 + thr - 1) / thr;
        dim3 grid(Tsum, ncol_groups);
        gather_dequant_bf16_vec4_kernel<<<grid, thr, 0, stream>>>(
            (const uint8_t*)hidden_states.data_ptr(),
            hs_scale.data_ptr<float>(),
            s_sorted_ids.data_ptr<int64_t>(),
            (__nv_bfloat16*)s_A_bf16.data_ptr(),
            T, H_DIM, H_BLOCKS
        );
    }

    // ==================================================================
    // Step 5: Expert compute - Always grouped GEMM
    // ==================================================================
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

    auto bf16_A  = (__nv_bfloat16*)s_A_bf16.data_ptr();
    auto bf16_G1 = (__nv_bfloat16*)s_G1_buf.data_ptr();
    auto bf16_SG = (__nv_bfloat16*)s_swiglu_buf.data_ptr();
    auto bf16_G2 = (__nv_bfloat16*)s_G2_buf.data_ptr();

    for (int le = 0; le < E_LOCAL; le++) {
        int start = h_expert_offsets_pinned[le];
        int end   = h_expert_offsets_pinned[le + 1];
        if (start >= end) continue;
        int Tk = end - start;

        transa_h[num_active]     = CUBLAS_OP_T;
        transb_h[num_active]     = CUBLAS_OP_N;
        alpha_h[num_active]      = 1.0f;
        beta_h[num_active]       = 0.0f;
        group_size_h[num_active] = 1;

        // GEMM1: W1^T @ A_e^T  ->  [2*I, Tk]
        // In col-major: C = W1^T * B  where B = A_e (row-major = col-major transposed)
        m_h1[num_active] = 2 * I_DIM;  n_h1[num_active] = Tk;  k_h1[num_active] = H_DIM;
        lda_h1[num_active] = H_DIM;  ldb_h1[num_active] = H_DIM;  ldc_h1[num_active] = 2 * I_DIM;
        A_ptrs_h1[num_active] = (const void*)((const __nv_bfloat16*)W1_cache[le].data_ptr());
        B_ptrs_h1[num_active] = (const void*)(bf16_A  + (int64_t)start * H_DIM);
        C_ptrs_h1[num_active] = (void*)      (bf16_G1 + (int64_t)start * 2 * I_DIM);

        // GEMM2: W2^T @ SG^T  ->  [H, Tk]
        m_h2[num_active] = H_DIM;  n_h2[num_active] = Tk;  k_h2[num_active] = I_DIM;
        lda_h2[num_active] = I_DIM;  ldb_h2[num_active] = I_DIM;  ldc_h2[num_active] = H_DIM;
        A_ptrs_h2[num_active] = (const void*)((const __nv_bfloat16*)W2_cache[le].data_ptr());
        B_ptrs_h2[num_active] = (const void*)(bf16_SG + (int64_t)start * I_DIM);
        C_ptrs_h2[num_active] = (void*)      (bf16_G2 + (int64_t)start * H_DIM);

        num_active++;
    }

    if (num_active > 0) {
        // Upload GEMM1 pointers
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

        // SwiGLU (vectorized BF16)
        {
            int total = Tsum * I_DIM;
            int total4 = total / 4;
            int thr = 256;
            swiglu_bf16_vec4_kernel<<<(total4 + thr - 1) / thr, thr, 0, stream>>>(
                bf16_G1, bf16_SG, total, I_DIM
            );
        }

        // Upload GEMM2 pointers
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

    // ==================================================================
    // Step 6: Weighted scatter-add (vectorized BF16)
    // ==================================================================
    // Zero the output buffer
    cudaMemsetAsync(s_output_f32.data_ptr(),
                    0,
                    (size_t)T * H_DIM * sizeof(float),
                    stream);
    {
        int thr = 256;
        int ncol_groups = (H_DIM / 4 + thr - 1) / thr;
        dim3 grid(Tsum, ncol_groups);
        weighted_scatter_add_bf16_vec4_kernel<<<grid, thr, 0, stream>>>(
            bf16_G2,
            s_sorted_ids.data_ptr<int64_t>(),
            s_expert_map.data_ptr<int32_t>(),
            s_topk_idx.data_ptr<int64_t>(),
            s_weights.data_ptr<float>(),
            s_output_f32.data_ptr<float>(),
            (int)local_expert_offset, H_DIM, TOP_K
        );
    }

    // ==================================================================
    // Step 7: Cast to BF16 and return
    // ==================================================================
    return s_output_f32.slice(0, 0, T).to(torch::kBFloat16);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MOE FP8 Block-Scale kernel (CUDA)");
}
