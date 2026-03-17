# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
MoE FP8 Block-Scale Kernel for FlashInfer-Bench
================================================

Implements a fused Mixture-of-Experts kernel with FP8 block-scale quantization
targeting NVIDIA Blackwell SM100a architecture.

Computation flow:
  1. DeepSeek-V3 no-aux routing: sigmoid → bias → group scoring → top-k=8
  2. Token permutation: group tokens by expert
  3. Per-expert: GEMM1 (FP8 blockwise) → SwiGLU → GEMM2 (FP8 blockwise)
  4. Weighted scatter-add back to output

Parameters (fixed DeepSeek-V3/R1 geometry):
  H=7168, I=2048, E_global=256, E_local=32
  top_k=8, n_group=8, topk_group=4, block_size=128

Usage:
  python examples/python/CuTeDSL/blackwell/moe_fp8_blockscale.py --num_tokens 4
"""

import argparse
import math
import sys
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ==============================================================================
# Constants
# ==============================================================================
H = 7168            # hidden_size
I = 2048            # intermediate_size
E_GLOBAL = 256      # total experts
E_LOCAL = 32        # local experts on this rank
TOP_K = 8           # experts per token
N_GROUP = 8         # routing groups
TOPK_GROUP = 4      # groups to keep
BLOCK = 128         # quantization block size


# ==============================================================================
# Part 1: DeepSeek-V3 No-Aux Routing (Fused Triton Kernel)
#
# Replaces ~15 PyTorch kernel launches with a single Triton kernel.
# Each program handles one token, processing all 256 expert scores in registers.
# ==============================================================================
@triton.jit
def _routing_kernel(
    logits_ptr,             # [T, E_GLOBAL] float32
    bias_ptr,               # [E_GLOBAL] float32
    topk_idx_ptr,           # [T, TOP_K] int64 output
    weights_ptr,            # [T, TOP_K] float32 output
    routed_scaling_factor,  # float scalar
    T,                      # number of tokens
    E_GLOBAL: tl.constexpr,     # 256
    TOP_K: tl.constexpr,        # 8
    N_GROUP: tl.constexpr,      # 8
    TOPK_GROUP: tl.constexpr,   # 4
):
    """Fused DeepSeek-V3 routing kernel. One program per token."""
    pid = tl.program_id(0)

    GROUP_SIZE: tl.constexpr = E_GLOBAL // N_GROUP  # 32

    # Load all expert logits and bias for this token
    e_range = tl.arange(0, E_GLOBAL)
    logits = tl.load(logits_ptr + pid * E_GLOBAL + e_range)
    bias = tl.load(bias_ptr + e_range)

    # Sigmoid scoring
    s = 1.0 / (1.0 + tl.exp(-logits))
    s_with_bias = s + bias

    # --- Group scoring: top-2 per group → sum ---
    group_id = e_range // GROUP_SIZE  # [256]: group index 0-7
    group_score_vec = tl.zeros([E_GLOBAL], dtype=tl.float32)

    for g in range(N_GROUP):
        mask_g = (group_id == g)
        vals = tl.where(mask_g, s_with_bias, float('-inf'))
        # Find max1
        max1 = tl.max(vals)
        # Remove first occurrence of max1 (position-based for tie safety)
        is_max1 = (vals == max1)
        first_pos = tl.min(tl.where(is_max1, e_range, E_GLOBAL))
        vals2 = tl.where(e_range == first_pos, float('-inf'), vals)
        # Find max2
        max2 = tl.max(vals2)
        g_score = max1 + max2
        # Replicate score across all positions in this group
        group_score_vec = tl.where(mask_g, g_score, group_score_vec)

    # --- Top-4 group selection (iterative max on replicated group scores) ---
    selected_mask = tl.zeros([E_GLOBAL], dtype=tl.int32)
    group_scores_remaining = group_score_vec

    for _g in range(TOPK_GROUP):
        max_score = tl.max(group_scores_remaining)
        is_max = (group_scores_remaining == max_score)
        # Find which group won (lowest-index element breaks ties)
        first_pos = tl.min(tl.where(is_max, e_range, E_GLOBAL))
        winning_group = first_pos // GROUP_SIZE
        in_winning_group = (group_id == winning_group)
        selected_mask = tl.where(in_winning_group, 1, selected_mask)
        group_scores_remaining = tl.where(
            in_winning_group, float('-inf'), group_scores_remaining
        )

    # --- Global top-8 from selected groups ---
    pruned_scores = tl.where(selected_mask == 1, s_with_bias, float('-inf'))

    out_range = tl.arange(0, TOP_K)
    out_idx = tl.zeros([TOP_K], dtype=tl.int64)
    out_weights = tl.zeros([TOP_K], dtype=tl.float32)

    for k in range(TOP_K):
        max_val = tl.max(pruned_scores)
        is_max = (pruned_scores == max_val)
        first_pos = tl.min(tl.where(is_max, e_range, E_GLOBAL))
        # Get weight from s (without bias) at first_pos
        w = tl.sum(tl.where(e_range == first_pos, s, 0.0))
        # Store in output position k
        out_idx = tl.where(out_range == k, first_pos.to(tl.int64), out_idx)
        out_weights = tl.where(out_range == k, w, out_weights)
        # Mask out selected expert
        pruned_scores = tl.where(e_range == first_pos, float('-inf'), pruned_scores)

    # Normalize and scale
    w_sum = tl.sum(out_weights) + 1e-20
    out_weights = (out_weights / w_sum) * routed_scaling_factor

    # Store results
    tl.store(topk_idx_ptr + pid * TOP_K + out_range, out_idx)
    tl.store(weights_ptr + pid * TOP_K + out_range, out_weights)


def deepseek_routing_triton(
    routing_logits: torch.Tensor,   # [T, E_global]
    routing_bias: torch.Tensor,     # [E_global]
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused Triton routing kernel."""
    T = routing_logits.shape[0]
    device = routing_logits.device

    logits = routing_logits.float().contiguous()
    bias = routing_bias.float().reshape(-1).contiguous()

    topk_idx = torch.empty((T, TOP_K), dtype=torch.int64, device=device)
    weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)

    _routing_kernel[(T,)](
        logits, bias,
        topk_idx, weights,
        routed_scaling_factor,
        T,
        E_GLOBAL=E_GLOBAL, TOP_K=TOP_K,
        N_GROUP=N_GROUP, TOPK_GROUP=TOPK_GROUP,
    )

    return topk_idx, weights


def deepseek_routing(
    routing_logits: torch.Tensor,   # [T, E_global]
    routing_bias: torch.Tensor,     # [E_global]
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-V3 no-aux routing.

    Returns:
        topk_idx: [T, TOP_K] - global expert indices selected per token
        weights:  [T, TOP_K] - normalized routing weights per token
    """
    return deepseek_routing_triton(routing_logits, routing_bias, routed_scaling_factor)


# ==============================================================================
# Part 2: Token Permutation
#
# Two implementations:
#   - Single-program kernel for small N (low launch overhead)
#   - Parallel 3-kernel approach for large N (>4K elements)
# ==============================================================================

# --- Single-program kernel (kept for small N) ---
@triton.jit
def _permute_kernel(
    topk_idx_ptr,           # [N] int64 input (flattened topk_idx)
    sorted_token_ids_ptr,   # [N] int64 output (pre-allocated at max size)
    expert_offsets_ptr,     # [E_LOCAL + 1] int32 output
    token_expert_map_ptr,   # [N] int32 output (pre-allocated at max size)
    counts_ptr,             # [E_LOCAL] int32 scratch
    write_counter_ptr,      # [E_LOCAL] int32 scratch
    local_expert_offset,    # scalar
    num_local_experts,      # scalar
    top_k: tl.constexpr,
    N: tl.constexpr,        # T * TOP_K
    E_LOCAL: tl.constexpr,  # 32
    BLOCK_SIZE: tl.constexpr,
):
    """Fused count + cumsum + scatter permutation kernel. Single program."""
    expert_range = tl.arange(0, E_LOCAL)

    # Initialize scratch to zero
    tl.store(counts_ptr + expert_range, tl.zeros([E_LOCAL], dtype=tl.int32))
    tl.store(write_counter_ptr + expert_range, tl.zeros([E_LOCAL], dtype=tl.int32))
    tl.debug_barrier()

    # Phase 1: Count tokens per expert
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        ge = tl.load(topk_idx_ptr + offsets, mask=mask, other=-1)
        le = ge - local_expert_offset
        valid = (le >= 0) & (le < num_local_experts) & mask
        le_safe = tl.where(valid, le, 0)
        tl.atomic_add(counts_ptr + le_safe, 1, mask=valid)

    tl.debug_barrier()

    # Phase 2: Exclusive prefix sum → expert_offsets
    counts = tl.load(counts_ptr + expert_range)
    offsets_vec = tl.zeros([E_LOCAL], dtype=tl.int32)
    running_sum = 0
    for e in range(E_LOCAL):
        mask_e = (expert_range == e)
        offsets_vec = tl.where(mask_e, running_sum, offsets_vec)
        running_sum = running_sum + tl.sum(tl.where(mask_e, counts, 0))
    tl.store(expert_offsets_ptr + expert_range, offsets_vec)
    tl.store(expert_offsets_ptr + E_LOCAL, running_sum)

    tl.debug_barrier()

    # Phase 3: Scatter tokens to sorted positions
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        ge = tl.load(topk_idx_ptr + offsets, mask=mask, other=-1)
        le = ge - local_expert_offset
        valid = (le >= 0) & (le < num_local_experts) & mask
        le_safe = tl.where(valid, le, 0)
        token_id = (offsets // top_k).to(tl.int64)
        pos = tl.atomic_add(write_counter_ptr + le_safe, 1, mask=valid)
        base = tl.load(expert_offsets_ptr + le_safe, mask=valid, other=0)
        write_idx = base + pos
        tl.store(sorted_token_ids_ptr + write_idx, token_id, mask=valid)
        tl.store(token_expert_map_ptr + write_idx, le_safe.to(tl.int32), mask=valid)


# --- Parallel kernels for large N ---
@triton.jit
def _permute_count_kernel(
    topk_idx_ptr,           # [N] int64 input
    counts_ptr,             # [E_LOCAL] int32 output (atomically incremented)
    local_expert_offset,
    num_local_experts,
    N,                      # runtime value (not constexpr — varies per call)
    E_LOCAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: Each program counts tokens in its chunk via atomics."""
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    ge = tl.load(topk_idx_ptr + offsets, mask=mask, other=-1)
    le = ge - local_expert_offset
    valid = (le >= 0) & (le < num_local_experts) & mask
    le_safe = tl.where(valid, le, 0)
    tl.atomic_add(counts_ptr + le_safe, 1, mask=valid)


@triton.jit
def _permute_cumsum_kernel(
    counts_ptr,             # [E_LOCAL] int32 input
    expert_offsets_ptr,     # [E_LOCAL + 1] int32 output
    E_LOCAL: tl.constexpr,
):
    """Phase 2: Single-program exclusive prefix sum on 32 expert counts."""
    expert_range = tl.arange(0, E_LOCAL)
    counts = tl.load(counts_ptr + expert_range)
    offsets_vec = tl.zeros([E_LOCAL], dtype=tl.int32)
    running_sum = 0
    for e in range(E_LOCAL):
        mask_e = (expert_range == e)
        offsets_vec = tl.where(mask_e, running_sum, offsets_vec)
        running_sum = running_sum + tl.sum(tl.where(mask_e, counts, 0))
    tl.store(expert_offsets_ptr + expert_range, offsets_vec)
    tl.store(expert_offsets_ptr + E_LOCAL, running_sum)


@triton.jit
def _permute_scatter_kernel(
    topk_idx_ptr,           # [N] int64 input
    sorted_token_ids_ptr,   # [N] int64 output
    expert_offsets_ptr,     # [E_LOCAL + 1] int32 input (from cumsum)
    token_expert_map_ptr,   # [N] int32 output
    write_counter_ptr,      # [E_LOCAL] int32 scratch (atomically incremented)
    local_expert_offset,
    num_local_experts,
    N,                      # runtime value
    top_k: tl.constexpr,
    E_LOCAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 3: Each program scatters tokens from its chunk via atomics."""
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    ge = tl.load(topk_idx_ptr + offsets, mask=mask, other=-1)
    le = ge - local_expert_offset
    valid = (le >= 0) & (le < num_local_experts) & mask
    le_safe = tl.where(valid, le, 0)
    token_id = (offsets // top_k).to(tl.int64)
    pos = tl.atomic_add(write_counter_ptr + le_safe, 1, mask=valid)
    base = tl.load(expert_offsets_ptr + le_safe, mask=valid, other=0)
    write_idx = base + pos
    tl.store(sorted_token_ids_ptr + write_idx, token_id, mask=valid)
    tl.store(token_expert_map_ptr + write_idx, le_safe.to(tl.int32), mask=valid)


# Threshold: use parallel kernels when N > this value
_PERMUTE_PARALLEL_THRESHOLD = 4096


def permute_tokens(
    topk_idx: torch.Tensor,        # [T, TOP_K]
    local_expert_offset: int,
    num_local_experts: int = E_LOCAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute which tokens go to which local expert.

    Returns:
        sorted_token_ids: [total_selected] - token indices sorted by expert
        expert_offsets:    [E_LOCAL + 1] - start offset for each expert in sorted_token_ids
        token_expert_map: [total_selected] - which local expert each entry maps to
    """
    T = topk_idx.shape[0]
    device = topk_idx.device
    N = T * TOP_K
    BLOCK_SIZE = 1024

    flat_idx = topk_idx.reshape(-1)  # [T * TOP_K]

    # Pre-allocate at max size (torch.empty = allocation only, no kernel launch)
    sorted_token_ids = torch.empty(N, dtype=torch.long, device=device)
    token_expert_map = torch.empty(N, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(num_local_experts + 1, dtype=torch.int32, device=device)

    if N <= _PERMUTE_PARALLEL_THRESHOLD:
        # Small N: single-program kernel (lower launch overhead)
        counts = torch.empty(num_local_experts, dtype=torch.int32, device=device)
        write_counter = torch.empty(num_local_experts, dtype=torch.int32, device=device)

        _permute_kernel[(1,)](
            flat_idx, sorted_token_ids, expert_offsets, token_expert_map,
            counts, write_counter,
            local_expert_offset, num_local_experts,
            top_k=TOP_K, N=N, E_LOCAL=num_local_experts,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Large N: parallel 3-kernel approach
        num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

        # torch.zeros launches a fast memset kernel
        counts = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
        write_counter = torch.zeros(num_local_experts, dtype=torch.int32, device=device)

        # Phase 1: Parallel count — each of num_blocks programs processes BLOCK_SIZE elements
        _permute_count_kernel[(num_blocks,)](
            flat_idx, counts,
            local_expert_offset, num_local_experts,
            N, E_LOCAL=num_local_experts, BLOCK_SIZE=BLOCK_SIZE,
        )

        # Phase 2: Cumsum — single program, 32 elements (trivial)
        _permute_cumsum_kernel[(1,)](
            counts, expert_offsets,
            E_LOCAL=num_local_experts,
        )

        # Phase 3: Parallel scatter — each program scatters its chunk
        _permute_scatter_kernel[(num_blocks,)](
            flat_idx, sorted_token_ids, expert_offsets, token_expert_map,
            write_counter,
            local_expert_offset, num_local_experts,
            N, top_k=TOP_K, E_LOCAL=num_local_experts, BLOCK_SIZE=BLOCK_SIZE,
        )

    Tsum = expert_offsets[num_local_experts].item()

    if Tsum == 0:
        sorted_token_ids = torch.empty(0, dtype=torch.long, device=device)
        token_expert_map = torch.empty(0, dtype=torch.int32, device=device)
        return sorted_token_ids, expert_offsets, token_expert_map

    return sorted_token_ids[:Tsum], expert_offsets, token_expert_map[:Tsum]


# ==============================================================================
# Part 2b: Weighted Scatter-Add (Triton)
# ==============================================================================
@triton.jit
def _weighted_scatter_add_kernel(
    gemm2_result_ptr,      # [Tsum, H_dim] float32
    sorted_token_ids_ptr,  # [Tsum] int64
    token_expert_map_ptr,  # [Tsum] int32
    topk_idx_ptr,          # [T, TOP_K] int64
    weights_ptr,           # [T, TOP_K] float32
    output_ptr,            # [T, H_dim] float32
    local_expert_offset,   # int scalar
    H_dim: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused weight-scale-scatter kernel.

    Each program handles one (sorted_position, H_tile) pair:
    1. Look up token_id and local_expert for this sorted position
    2. Compute global_expert = local_expert + local_expert_offset
    3. Find the routing weight for this token-expert pair
    4. Scale gemm2_result by the weight
    5. Atomic-add to output[token_id]
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Load token_id and expert for this sorted position
    token_id = tl.load(sorted_token_ids_ptr + pid_row)
    local_expert = tl.load(token_expert_map_ptr + pid_row)
    global_expert = (local_expert + local_expert_offset).to(tl.int64)

    # Vectorized weight lookup across TOP_K slots
    topk_offsets = token_id * TOP_K + tl.arange(0, TOP_K)
    topk_experts = tl.load(topk_idx_ptr + topk_offsets)
    topk_weights = tl.load(weights_ptr + topk_offsets)
    match = topk_experts == global_expert
    w = tl.sum(tl.where(match, topk_weights, 0.0))

    # Load gemm2_result tile and scale
    h_start = pid_col * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H_dim
    gemm2_vals = tl.load(
        gemm2_result_ptr + pid_row * H_dim + h_offsets,
        mask=h_mask, other=0.0,
    )
    scaled = (gemm2_vals * w).to(tl.float32)

    # Atomic add to output
    tl.atomic_add(output_ptr + token_id * H_dim + h_offsets, scaled, mask=h_mask)


def weighted_scatter_add(
    gemm2_result: torch.Tensor,       # [Tsum, H_dim] float32
    sorted_token_ids: torch.Tensor,   # [Tsum] int64
    token_expert_map: torch.Tensor,   # [Tsum] int32
    topk_idx: torch.Tensor,           # [T, TOP_K] int64
    weights: torch.Tensor,            # [T, TOP_K] float32
    local_expert_offset: int,
    T: int,
    H_dim: int = H,
) -> torch.Tensor:
    """Fused weighted scatter-add using a single Triton kernel.

    Replaces ~13 PyTorch kernel launches with 1 Triton kernel + 1 torch.zeros.
    """
    Tsum = sorted_token_ids.shape[0]
    device = gemm2_result.device
    output = torch.zeros((T, H_dim), dtype=torch.float32, device=device)

    if Tsum == 0:
        return output

    BLOCK_H = 1024
    grid = (Tsum, (H_dim + BLOCK_H - 1) // BLOCK_H)

    _weighted_scatter_add_kernel[grid](
        gemm2_result, sorted_token_ids, token_expert_map,
        topk_idx, weights, output,
        local_expert_offset,
        H_dim=H_dim, TOP_K=topk_idx.shape[1], BLOCK_H=BLOCK_H,
    )

    return output


# ==============================================================================
# Part 3: FP8 Block-Scale Dequantization Helper
# ==============================================================================
def dequant_fp8_blockscale(
    x_fp8: torch.Tensor,           # [..., D] in FP8
    scale: torch.Tensor,            # [..., D // BLOCK] in float32 (or needs reshaping)
    dim: int = -1,
) -> torch.Tensor:
    """Dequantize FP8 tensor with per-block float32 scale factors."""
    x = x_fp8.float()
    # Expand scale to match x shape along the quantized dimension
    shape = list(x.shape)
    num_blocks = shape[dim] // BLOCK
    # Reshape x to expose blocks
    new_shape = shape[:dim] + [num_blocks, BLOCK] + (shape[dim+1:] if dim != -1 and dim != len(shape)-1 else [])
    if dim == -1 or dim == len(shape) - 1:
        new_shape = shape[:-1] + [num_blocks, BLOCK]

    x_blocked = x.reshape(new_shape)
    # scale should broadcast: [..., num_blocks, 1]
    scale_expanded = scale.unsqueeze(-1)    # add block dimension
    result = (x_blocked * scale_expanded).reshape(shape)
    return result


# ==============================================================================
# Part 4: Expert Compute (Initial PyTorch Implementation)
#
# This will be replaced with CuTe DSL kernels based on the blockwise
# contiguous grouped GEMM pattern for peak performance.
# ==============================================================================
def expert_compute_pytorch(
    hidden_states: torch.Tensor,        # [T, H] FP8
    hidden_states_scale: torch.Tensor,  # [H/BLOCK, T]
    gemm1_weights: torch.Tensor,        # [E_local, 2*I, H] FP8
    gemm1_weights_scale: torch.Tensor,  # [E_local, (2*I)/BLOCK, H/BLOCK]
    gemm2_weights: torch.Tensor,        # [E_local, H, I] FP8
    gemm2_weights_scale: torch.Tensor,  # [E_local, H/BLOCK, I/BLOCK]
    sorted_token_ids: torch.Tensor,     # [total_selected]
    expert_offsets: torch.Tensor,       # [E_local + 1]
    topk_idx: torch.Tensor,            # [T, TOP_K]
    weights: torch.Tensor,             # [T, TOP_K]
    local_expert_offset: int,
    token_expert_map: torch.Tensor = None,  # [total_selected] int32
) -> torch.Tensor:
    """
    Per-expert compute: GEMM1 → SwiGLU → GEMM2 → weighted accumulation.

    TODO: Replace with CuTe DSL blockwise contiguous grouped GEMM kernel
    following the pattern in blockwise_gemm/contiguous_grouped_gemm.py
    """
    T = hidden_states.shape[0]
    Tsum = sorted_token_ids.shape[0]
    device = hidden_states.device

    if Tsum == 0:
        return torch.zeros((T, H), dtype=torch.float32, device=device)

    # Dequantize hidden states: [T, H]
    # scale is [H/BLOCK, T], need [T, H/BLOCK] for per-token-per-block scaling
    hs_scale_t = hidden_states_scale.float().permute(1, 0).contiguous()  # [T, H/BLOCK]
    A = dequant_fp8_blockscale(hidden_states, hs_scale_t, dim=-1)        # [T, H] float32

    # Pre-allocate buffer for all expert GEMM2 outputs
    gemm2_buffer = torch.zeros((Tsum, H), dtype=torch.float32, device=device)

    # Convert expert_offsets to CPU once for loop indexing
    offsets_cpu = expert_offsets.cpu()

    for le in range(E_LOCAL):
        start = offsets_cpu[le].item()
        end = offsets_cpu[le + 1].item()
        if start >= end:
            continue

        token_idx = sorted_token_ids[start:end]         # [Tk]

        # Gather activations for this expert
        A_e = A[token_idx]                               # [Tk, H]

        # Dequantize weight1: [2*I, H], scale [2*I/BLOCK, H/BLOCK]
        w1_scale = gemm1_weights_scale[le].float()       # [(2*I)/BLOCK, H/BLOCK]
        W1 = dequant_2d_blockscale(
            gemm1_weights[le], w1_scale, BLOCK
        )                                                 # [2*I, H] float32

        # GEMM1: [Tk, H] @ [H, 2*I] → [Tk, 2*I]
        G1 = A_e @ W1.t()

        # SwiGLU
        X1 = G1[:, :I]                                   # gate
        X2 = G1[:, I:]                                    # up
        C = F.silu(X2) * X1                               # [Tk, I]

        # Dequantize weight2: [H, I], scale [H/BLOCK, I/BLOCK]
        w2_scale = gemm2_weights_scale[le].float()        # [H/BLOCK, I/BLOCK]
        W2 = dequant_2d_blockscale(
            gemm2_weights[le], w2_scale, BLOCK
        )                                                  # [H, I] float32

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        gemm2_buffer[start:end] = C @ W2.t()

    # Fused weighted scatter-add (single Triton kernel)
    return weighted_scatter_add(
        gemm2_buffer, sorted_token_ids, token_expert_map,
        topk_idx, weights, local_expert_offset, T,
    )


def dequant_2d_blockscale(
    x_fp8: torch.Tensor,    # [R, C] FP8
    scale: torch.Tensor,     # [R/BLOCK, C/BLOCK] float32
    block_size: int = BLOCK,
) -> torch.Tensor:
    """Dequantize a 2D FP8 tensor with 2D block-scale factors."""
    R, C = x_fp8.shape
    rb = R // block_size
    cb = C // block_size
    x = x_fp8.float().reshape(rb, block_size, cb, block_size)
    s = scale.float().reshape(rb, 1, cb, 1)
    return (x * s).reshape(R, C)


# ==============================================================================
# Part 4b: CuTe DSL Expert Compute using Blockwise Contiguous Grouped GEMM
#
# This adapts the blockwise_gemm/contiguous_grouped_gemm.py pattern from
# CUTLASS examples for the MoE use case.
#
# Data layout mapping:
#   MoE concept              → Contiguous Grouped GEMM layout
#   ----------------------------------------------------------
#   sorted_tokens [Tsum, K]  → A [1, valid_m, K]  (contiguous by expert)
#   expert_weights [E, N, K] → B [L, N, K]        (L = num_experts)
#   hs_scale [Tsum, K/128]   → SFA [1, valid_m, K/128]  (per-row, per-K-block)
#   w_scale [E, N/128, K/128]→ SFB [L, N/128, K/128]    (per-block)
#   expert_ids [Tsum]        → gidx_mapping [valid_m]    (expert per row)
# ==============================================================================

@triton.jit
def _gather_and_scale_kernel(
    hidden_states_ptr,       # [T, H] int8 (FP8 viewed as int8)
    hs_scale_ptr,            # [H_BLOCKS, T] float32
    sorted_ids_ptr,          # [Tsum] int64
    a_out_ptr,               # [Tsum, H] int8 output
    sfa_out_ptr,             # [Tsum, H_BLOCKS] float32 output
    T,                       # number of tokens
    H: tl.constexpr,         # 7168
    H_BLOCKS: tl.constexpr,  # H // BLOCK = 56
    H_BLOCKS_PADDED: tl.constexpr,  # next power-of-2 >= H_BLOCKS (64)
    COPY_BLOCK: tl.constexpr,       # bytes per program along H (1024)
):
    """Fused gather: row-gather FP8 activations + column-gather float32 scales."""
    row = tl.program_id(0)   # sorted token index
    col = tl.program_id(1)   # which COPY_BLOCK chunk of H

    token_id = tl.load(sorted_ids_ptr + row)

    # Copy COPY_BLOCK FP8 bytes: hidden_states[token_id, col*COPY_BLOCK:] → a_out[row, :]
    # H = 7 * COPY_BLOCK exactly, so no masking needed
    h_start = col * COPY_BLOCK
    h_offs = h_start + tl.arange(0, COPY_BLOCK)
    vals = tl.load(hidden_states_ptr + token_id * H + h_offs)
    tl.store(a_out_ptr + row * H + h_offs, vals)

    # Scale copy: col=0 programs handle the full [H_BLOCKS] column gather
    # hs_scale layout is [H_BLOCKS, T]: element [hb, t] = ptr + hb*T + t
    if col == 0:
        hb_range = tl.arange(0, H_BLOCKS_PADDED)
        hb_mask = hb_range < H_BLOCKS
        scales = tl.load(hs_scale_ptr + hb_range * T + token_id, mask=hb_mask, other=0.0)
        tl.store(sfa_out_ptr + row * H_BLOCKS + hb_range, scales, mask=hb_mask)


def prepare_gemm1_inputs(
    hidden_states: torch.Tensor,        # [T, H] FP8
    hidden_states_scale: torch.Tensor,  # [H/BLOCK, T] float32
    gemm1_weights: torch.Tensor,        # [E_local, 2*I, H] FP8
    gemm1_weights_scale: torch.Tensor,  # [E_local, (2*I)/BLOCK, H/BLOCK] float32
    sorted_token_ids: torch.Tensor,     # [Tsum] long
    gidx_mapping: torch.Tensor,         # [Tsum] int32 (token_expert_map from permute_tokens)
):
    """
    Prepare data for GEMM1 in the contiguous grouped GEMM format.

    Fuses row-gather of FP8 activations and column-gather of float32 scales
    into a single Triton kernel. gidx_mapping is passed through directly
    from permute_tokens (same values as the old repeat_interleave computation).
    """
    Tsum = sorted_token_ids.shape[0]
    T = hidden_states.shape[0]
    device = hidden_states.device

    H_BLOCKS = H // BLOCK           # 56
    H_BLOCKS_PADDED = 64            # next power-of-2 >= 56
    COPY_BLOCK = 1024               # H = 7 * 1024 exactly
    n_col = H // COPY_BLOCK         # 7

    a_sorted = torch.empty(Tsum, H, dtype=hidden_states.dtype, device=device)
    sfa_sorted = torch.empty(Tsum, H_BLOCKS, dtype=torch.float32, device=device)

    _gather_and_scale_kernel[(Tsum, n_col)](
        hidden_states.view(torch.int8),
        hidden_states_scale.float().contiguous(),
        sorted_token_ids,
        a_sorted.view(torch.int8),
        sfa_sorted,
        T,
        H=H, H_BLOCKS=H_BLOCKS, H_BLOCKS_PADDED=H_BLOCKS_PADDED,
        COPY_BLOCK=COPY_BLOCK,
    )

    return a_sorted, sfa_sorted, gemm1_weights, gemm1_weights_scale, gidx_mapping


@triton.jit
def _quantize_fp8_kernel(
    inter_ptr,              # [Tsum, I] float32 input
    a_fp8_ptr,              # [Tsum, I] int8 output (FP8 as int8)
    sfa_ptr,                # [Tsum, I_BLOCKS] float32 output
    I: tl.constexpr,        # 2048
    I_BLOCKS: tl.constexpr, # 16
    BLOCK: tl.constexpr,    # 128
    FP8_MAX: tl.constexpr,  # 448.0
):
    """Per-(token, block) FP8 quantization: compute amax, scale, quantize."""
    row = tl.program_id(0)
    col = tl.program_id(1)

    b_offs = col * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(inter_ptr + row * I + b_offs)

    # Compute per-block scale
    amax = tl.max(tl.abs(vals))
    scale = tl.maximum(amax / FP8_MAX, 1e-12)
    tl.store(sfa_ptr + row * I_BLOCKS + col, scale)

    # Quantize to FP8 and store as int8
    fp8_vals = (vals / scale).to(tl.float8e4nv)
    tl.store(a_fp8_ptr + row * I + b_offs, fp8_vals.to(tl.int8, bitcast=True))


def prepare_gemm2_inputs(
    intermediate: torch.Tensor,         # [Tsum, I] float32 (SwiGLU output)
    gemm2_weights: torch.Tensor,        # [E_local, H, I] FP8
    gemm2_weights_scale: torch.Tensor,  # [E_local, H/BLOCK, I/BLOCK] float32
    gidx_mapping: torch.Tensor,         # [Tsum] int32
):
    """
    Fused FP8 quantization of SwiGLU output in a single Triton kernel.

    Each program handles one (token, I-block): computes amax → scale → FP8.
    Replaces 5+ PyTorch kernel launches with grid=(Tsum, I/BLOCK).
    """
    Tsum = intermediate.shape[0]
    device = intermediate.device
    I_BLOCKS = I // BLOCK   # 16

    a_fp8 = torch.empty(Tsum, I, dtype=torch.float8_e4m3fn, device=device)
    sfa_quant = torch.empty(Tsum, I_BLOCKS, dtype=torch.float32, device=device)

    _quantize_fp8_kernel[(Tsum, I_BLOCKS)](
        intermediate.contiguous(),
        a_fp8.view(torch.int8),
        sfa_quant,
        I=I, I_BLOCKS=I_BLOCKS, BLOCK=BLOCK, FP8_MAX=448.0,
    )

    return a_fp8, sfa_quant, gemm2_weights, gemm2_weights_scale, gidx_mapping


def _make_cute_fp8_tensor(from_dlpack_fn, tensor_int8, cutlass_mod, leading_dim=1):
    """Create a CuTe tensor from an int8 view of an FP8 tensor."""
    ct = from_dlpack_fn(tensor_int8, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)
    ct.element_type = cutlass_mod.Float8E4M3FN
    return ct


def _make_cute_tensor(from_dlpack_fn, tensor, leading_dim=1):
    """Create a CuTe tensor from a regular tensor."""
    return from_dlpack_fn(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


# ==============================================================================
# Fused SwiGLU + FP8 block-scale quantize (single Triton kernel)
#
# Replaces: gemm1_result[:, :I].float(), gemm1_result[:, I:].float(),
#           F.silu(up) * gate, and _quantize_fp8_kernel  (4 launches → 1)
# ==============================================================================
@triton.jit
def _swiglu_quantize_kernel(
    gemm1_out_ptr,   # [valid_m, 2*I] bfloat16
    a2_out_ptr,      # [Tsum, I] int8 (FP8 as int8)
    sfa2_out_ptr,    # [Tsum, I_BLOCKS] float32
    Tsum,            # actual token count (grid rows == Tsum, guard for safety)
    I: tl.constexpr,
    I_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TWO_I: tl.constexpr,
):
    """Fused SwiGLU activation + per-block FP8 quantization.

    Each program handles one (token-row, I-block) pair:
      gate = gemm1_out[row, :I]   (first half of 2*I)
      up   = gemm1_out[row, I:]   (second half)
      intermediate = silu(up) * gate  = up * sigmoid(up) * gate
      scale = amax(|intermediate|) / FP8_MAX
      output = float8(intermediate / scale)
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= Tsum:
        return

    i_offs = col * BLOCK + tl.arange(0, BLOCK)

    # Load gate and up from bfloat16 GEMM1 output, cast to float32
    gate = tl.load(gemm1_out_ptr + row * TWO_I + i_offs).to(tl.float32)
    up   = tl.load(gemm1_out_ptr + row * TWO_I + I + i_offs).to(tl.float32)

    # SwiGLU: silu(up) * gate = up * sigmoid(up) * gate
    intermediate = up * tl.sigmoid(up) * gate

    # Per-block FP8 quantize
    amax  = tl.max(tl.abs(intermediate))
    scale = tl.maximum(amax / FP8_MAX, 1e-12)
    tl.store(sfa2_out_ptr + row * I_BLOCKS + col, scale)

    fp8_vals = (intermediate / scale).to(tl.float8e4nv)
    tl.store(a2_out_ptr + row * I + i_offs, fp8_vals.to(tl.int8, bitcast=True))


# Module-level cache for compiled GEMM kernels and hardware info
_gemm_cache = {}
_hw_info_cache = {}
# CUDA graph registry: (valid_m, T, local_expert_offset) → graph + static bufs
_cuda_graph_registry = {}


def _get_compiled_gemm(cache_key, cute_mod, kernel, a, b, c, sfa, sfb, gidx,
                       max_active_clusters, stream):
    """Compile a GEMM kernel once and cache it by key."""
    if cache_key not in _gemm_cache:
        _gemm_cache[cache_key] = cute_mod.compile(
            kernel, a, b, c, sfa, sfb, gidx,
            max_active_clusters, stream,
        )
    return _gemm_cache[cache_key]


def _build_cuda_graph(
    cache_key, valid_m, T, Tsum, pad_m, local_expert_offset,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    sorted_token_ids, topk_idx, weights, token_expert_map,
    device,
):
    """
    Build and cache a CUDA graph for the full expert-compute pipeline.

    All intermediate tensors are pre-allocated as static buffers.
    CuTe tensor wrappers are created once and reused across replays.
    On each replay the caller updates the static INPUT buffers in-place,
    then calls graph.replay() — zero Python overhead in the hot path.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda
    from contiguous_grouped_gemm import BlockwiseContiguousGroupedGemmKernel

    H_BLOCKS = H // BLOCK
    I_BLOCKS = I // BLOCK

    # ── Static INPUT buffers (caller copies new data here before replay) ──────
    s_hidden     = hidden_states.clone()
    s_hs_scale   = hidden_states_scale.float().contiguous().clone()
    s_sorted_ids = torch.zeros(valid_m, dtype=torch.int64, device=device)
    s_sorted_ids[:Tsum].copy_(sorted_token_ids)
    s_token_map  = torch.zeros(valid_m, dtype=torch.int32, device=device)
    s_token_map[:Tsum].copy_(token_expert_map)
    s_topk_idx   = topk_idx.clone()
    s_weights    = weights.clone()

    # ── Static INTERMEDIATE buffers ───────────────────────────────────────────
    a1_buf       = torch.empty(valid_m, H,     dtype=hidden_states.dtype, device=device)
    sfa1_buf     = torch.empty(valid_m, H_BLOCKS, dtype=torch.float32, device=device)
    gemm1_out    = torch.empty(valid_m, 2 * I, dtype=torch.bfloat16,  device=device)
    a2_buf       = torch.empty(valid_m, I,     dtype=torch.float8_e4m3fn, device=device)
    sfa2_buf     = torch.empty(valid_m, I_BLOCKS, dtype=torch.float32, device=device)
    gemm2_out    = torch.empty(valid_m, H,     dtype=torch.bfloat16,  device=device)
    output_buf   = torch.zeros(T, H,           dtype=torch.float32,   device=device)

    # ── CuTe wrappers for WEIGHT tensors (stable: weights never change) ───────
    b1_cute  = from_dlpack(gemm1_weights.view(torch.int8).permute(1, 2, 0),
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
    b1_cute.element_type = cutlass.Float8E4M3FN
    sfb1_cute = from_dlpack(gemm1_weights_scale.permute(1, 2, 0),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)

    b2_cute  = from_dlpack(gemm2_weights.view(torch.int8).permute(1, 2, 0),
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
    b2_cute.element_type = cutlass.Float8E4M3FN
    sfb2_cute = from_dlpack(gemm2_weights_scale.permute(1, 2, 0),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # ── CuTe wrappers for ACTIVATION tensors (point to static buffers) ────────
    a1_cute  = from_dlpack(a1_buf.view(torch.int8).unsqueeze(-1),
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
    a1_cute.element_type = cutlass.Float8E4M3FN
    sfa1_cute = from_dlpack(sfa1_buf.unsqueeze(-1),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)
    c1_cute   = from_dlpack(gemm1_out.unsqueeze(-1),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)
    gidx_cute = from_dlpack(s_token_map, assumed_align=4).mark_layout_dynamic()

    a2_cute  = from_dlpack(a2_buf.view(torch.int8).unsqueeze(-1),
                           assumed_align=16).mark_layout_dynamic(leading_dim=1)
    a2_cute.element_type = cutlass.Float8E4M3FN
    sfa2_cute = from_dlpack(sfa2_buf.unsqueeze(-1),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)
    c2_cute   = from_dlpack(gemm2_out.unsqueeze(-1),
                            assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # ── Compile CuTe GEMM kernels (cached by valid_m) ─────────────────────────
    cluster_size = 2 * 2
    if cluster_size not in _hw_info_cache:
        _hw_info_cache[cluster_size] = (
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
        )
    max_active_clusters = _hw_info_cache[cluster_size]

    torch_stream = torch.cuda.current_stream()
    cu_stream    = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm1 = _get_compiled_gemm(
        ("gemm1", valid_m), cute,
        BlockwiseContiguousGroupedGemmKernel(
            acc_dtype=cutlass.Float32, use_2cta_instrs=True,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(2, 2),
        ),
        a1_cute, b1_cute, c1_cute, sfa1_cute, sfb1_cute, gidx_cute,
        max_active_clusters, cu_stream,
    )
    compiled_gemm2 = _get_compiled_gemm(
        ("gemm2", valid_m), cute,
        BlockwiseContiguousGroupedGemmKernel(
            acc_dtype=cutlass.Float32, use_2cta_instrs=True,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(2, 2),
        ),
        a2_cute, b2_cute, c2_cute, sfa2_cute, sfb2_cute, gidx_cute,
        max_active_clusters, cu_stream,
    )

    n_col         = H // 1024          # = 7
    scatter_grid  = (Tsum, (H + 1023) // 1024)

    # ── Inner function: pure GPU operations, captured in CUDA graph ───────────
    def _inner():
        # 1. Gather FP8 activations (Triton)
        _gather_and_scale_kernel[(Tsum, n_col)](
            s_hidden.view(torch.int8),
            s_hs_scale,
            s_sorted_ids,
            a1_buf.view(torch.int8),
            sfa1_buf,
            T,
            H=H, H_BLOCKS=H_BLOCKS, H_BLOCKS_PADDED=64, COPY_BLOCK=1024,
        )
        if pad_m > 0:
            a1_buf[Tsum:valid_m].zero_()
            sfa1_buf[Tsum:valid_m].zero_()

        # 2. GEMM1 (CuTe DSL)
        gemm1_out.zero_()
        compiled_gemm1(
            a1_cute, b1_cute, c1_cute,
            sfa1_cute, sfb1_cute, gidx_cute,
            cu_stream,
        )

        # 3. Fused SwiGLU + FP8 quantize (Triton, 4 ops → 1)
        _swiglu_quantize_kernel[(Tsum, I_BLOCKS)](
            gemm1_out,
            a2_buf.view(torch.int8),
            sfa2_buf,
            Tsum,
            I=I, I_BLOCKS=I_BLOCKS, BLOCK=BLOCK, FP8_MAX=448.0, TWO_I=2 * I,
        )
        if pad_m > 0:
            a2_buf[Tsum:valid_m].zero_()
            sfa2_buf[Tsum:valid_m].zero_()

        # 4. GEMM2 (CuTe DSL)
        gemm2_out.zero_()
        compiled_gemm2(
            a2_cute, b2_cute, c2_cute,
            sfa2_cute, sfb2_cute, gidx_cute,
            cu_stream,
        )

        # 5. Weighted scatter-add (Triton)
        output_buf.zero_()
        _weighted_scatter_add_kernel[scatter_grid](
            gemm2_out, s_sorted_ids, s_token_map,
            s_topk_idx, s_weights, output_buf,
            local_expert_offset,
            H_dim=H, TOP_K=TOP_K, BLOCK_H=1024,
        )

    # ── Warmup: JIT-compile all Triton kernels before graph capture ───────────
    for _ in range(3):
        _inner()
    torch.cuda.synchronize()

    # ── Capture CUDA graph ────────────────────────────────────────────────────
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _inner()
    torch.cuda.synchronize()

    # Store everything that must stay alive for the lifetime of the graph
    _cuda_graph_registry[cache_key] = dict(
        graph=g, _inner_fn=_inner,
        s_hidden=s_hidden, s_hs_scale=s_hs_scale,
        s_sorted_ids=s_sorted_ids, s_token_map=s_token_map,
        s_topk_idx=s_topk_idx, s_weights=s_weights,
        a1_buf=a1_buf, sfa1_buf=sfa1_buf, gemm1_out=gemm1_out,
        a2_buf=a2_buf, sfa2_buf=sfa2_buf, gemm2_out=gemm2_out,
        output_buf=output_buf,
        a1_cute=a1_cute, sfa1_cute=sfa1_cute, c1_cute=c1_cute,
        b1_cute=b1_cute, sfb1_cute=sfb1_cute,
        a2_cute=a2_cute, sfa2_cute=sfa2_cute, c2_cute=c2_cute,
        b2_cute=b2_cute, sfb2_cute=sfb2_cute, gidx_cute=gidx_cute,
        Tsum=Tsum, pad_m=pad_m,
    )


def expert_compute_cutedsl(
    hidden_states: torch.Tensor,        # [T, H] FP8
    hidden_states_scale: torch.Tensor,  # [H/BLOCK, T]
    gemm1_weights: torch.Tensor,        # [E_local, 2*I, H] FP8
    gemm1_weights_scale: torch.Tensor,  # [E_local, (2*I)/BLOCK, H/BLOCK]
    gemm2_weights: torch.Tensor,        # [E_local, H, I] FP8
    gemm2_weights_scale: torch.Tensor,  # [E_local, H/BLOCK, I/BLOCK]
    sorted_token_ids: torch.Tensor,     # [total_selected]
    expert_offsets: torch.Tensor,       # [E_local + 1]
    topk_idx: torch.Tensor,             # [T, TOP_K]
    weights: torch.Tensor,              # [T, TOP_K]
    local_expert_offset: int,
    token_expert_map: torch.Tensor = None,  # [total_selected] int32
) -> torch.Tensor:
    """
    CuTe DSL Expert Compute with CUDA graph acceleration.

    On first call for a given (valid_m, T, local_expert_offset):
      - Compiles CuTe DSL GEMM kernels
      - Captures the full pipeline in a CUDA graph:
          gather(FP8) → GEMM1 → fused SwiGLU+quant → GEMM2 → scatter-add
    On subsequent calls:
      - Copies inputs into pre-allocated static buffers (async GPU copies)
      - Replays the CUDA graph (zero Python overhead between kernels)
    """
    try:
        import cutlass  # noqa: F401 – trigger ImportError early if missing
    except ImportError:
        return expert_compute_pytorch(
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            sorted_token_ids, expert_offsets,
            topk_idx, weights, local_expert_offset,
            token_expert_map,
        )

    T    = hidden_states.shape[0]
    Tsum = sorted_token_ids.shape[0]
    device = hidden_states.device

    if Tsum == 0:
        return torch.zeros((T, H), dtype=torch.float32, device=device)

    pad_m   = (BLOCK - Tsum % BLOCK) % BLOCK
    valid_m = Tsum + pad_m

    cache_key = (valid_m, T, local_expert_offset)
    if cache_key not in _cuda_graph_registry:
        _build_cuda_graph(
            cache_key, valid_m, T, Tsum, pad_m, local_expert_offset,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            sorted_token_ids, topk_idx, weights, token_expert_map,
            device,
        )

    reg = _cuda_graph_registry[cache_key]

    # ── Copy dynamic inputs into static buffers (async GPU copies) ────────────
    reg['s_hidden'].copy_(hidden_states, non_blocking=True)
    reg['s_hs_scale'].copy_(hidden_states_scale.float(), non_blocking=True)
    reg['s_sorted_ids'][:Tsum].copy_(sorted_token_ids, non_blocking=True)
    if pad_m > 0:
        reg['s_sorted_ids'][Tsum:].fill_(0)
    reg['s_token_map'][:Tsum].copy_(token_expert_map, non_blocking=True)
    if pad_m > 0:
        reg['s_token_map'][Tsum:].fill_(0)
    reg['s_topk_idx'].copy_(topk_idx, non_blocking=True)
    reg['s_weights'].copy_(weights, non_blocking=True)

    # ── Replay CUDA graph (zero Python overhead between GPU kernels) ───────────
    reg['graph'].replay()

    return reg['output_buf'].float()


# ==============================================================================
# Part 5: Top-level run() Function (FlashInfer-Bench Interface)
# ==============================================================================
@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    use_cutedsl: bool = False,
):
    """
    MoE FP8 Block-Scale kernel entry point.

    Implements DeepSeek-V3 style MoE with:
    - No-aux routing (sigmoid + bias + grouped top-k)
    - FP8 block-scale quantized expert weights
    - SwiGLU activation between two GEMMs
    - Weighted accumulation of expert outputs

    Args:
        use_cutedsl: If True, use CuTe DSL blockwise contiguous grouped GEMM
                     for the expert compute (requires SM100a GPU + CuTe DSL setup).
                     If False, use PyTorch reference implementation.
    """
    T = routing_logits.shape[0]
    device = hidden_states.device

    # Step 1: Routing
    topk_idx, weights = deepseek_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # Step 2: Token permutation
    sorted_token_ids, expert_offsets, token_expert_map = permute_tokens(
        topk_idx, local_expert_offset
    )

    # Step 3: Expert compute (GEMM1 → SwiGLU → GEMM2 → weighted accumulation)
    compute_fn = expert_compute_cutedsl if use_cutedsl else expert_compute_pytorch
    output = compute_fn(
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        sorted_token_ids, expert_offsets,
        topk_idx, weights,
        local_expert_offset,
        token_expert_map,
    )

    return output.to(torch.bfloat16)


# ==============================================================================
# Part 6: Test Harness
# ==============================================================================
def create_test_tensors(T: int, device: str = "cuda"):
    """Create test tensors matching the FlashInfer-Bench interface."""
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Routing
    routing_logits = torch.randn(T, E_GLOBAL, device=device, dtype=torch.float32)
    routing_bias = torch.randn(E_GLOBAL, device=device, dtype=torch.float32) * 0.1

    # Hidden states (FP8 + scale)
    hidden_states = torch.randn(T, H, device=device, dtype=torch.float32)
    hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(num_hidden_blocks, T, device=device, dtype=torch.float32)
    # Compute proper scales
    hs_blocked = hidden_states.reshape(T, num_hidden_blocks, BLOCK)
    hidden_states_scale = hs_blocked.abs().amax(dim=-1).permute(1, 0).contiguous() / 448.0
    hidden_states_scale = hidden_states_scale.clamp(min=1e-12)

    # Expert weights (FP8 + scale)
    gemm1_weights = torch.randn(E_LOCAL, 2 * I, H, device=device, dtype=torch.float32)
    gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.ones(E_LOCAL, num_gemm1_out_blocks, num_hidden_blocks,
                                      device=device, dtype=torch.float32)
    for e in range(E_LOCAL):
        w_blocked = gemm1_weights[e].reshape(num_gemm1_out_blocks, BLOCK, num_hidden_blocks, BLOCK)
        gemm1_weights_scale[e] = w_blocked.abs().amax(dim=(1, 3)) / 448.0
    gemm1_weights_scale = gemm1_weights_scale.clamp(min=1e-12)

    gemm2_weights = torch.randn(E_LOCAL, H, I, device=device, dtype=torch.float32)
    gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.ones(E_LOCAL, num_hidden_blocks, num_intermediate_blocks,
                                      device=device, dtype=torch.float32)
    for e in range(E_LOCAL):
        w_blocked = gemm2_weights[e].reshape(num_hidden_blocks, BLOCK, num_intermediate_blocks, BLOCK)
        gemm2_weights_scale[e] = w_blocked.abs().amax(dim=(1, 3)) / 448.0
    gemm2_weights_scale = gemm2_weights_scale.clamp(min=1e-12)

    local_expert_offset = 0
    routed_scaling_factor = 1.0

    return (
        routing_logits, routing_bias,
        hidden_states_fp8, hidden_states_scale,
        gemm1_weights_fp8, gemm1_weights_scale,
        gemm2_weights_fp8, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
    )


def reference_run(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    """Reference implementation matching the original Python reference code."""
    T = routing_logits.shape[0]
    device = hidden_states.device

    num_hidden_blocks = H // BLOCK
    num_gemm1_out_blocks = (2 * I) // BLOCK
    num_intermediate_blocks = I // BLOCK

    # FP8 dequant
    A_fp32 = hidden_states.float()
    A_scale = hidden_states_scale.float().permute(1, 0).contiguous()
    A_scale_expanded = A_scale.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H)
    A = A_fp32 * A_scale_expanded

    W13_fp32 = gemm1_weights.float()
    S13 = gemm1_weights_scale.float()
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)
    W13 = W13_fp32 * S13_expanded

    W2_fp32 = gemm2_weights.float()
    S2 = gemm2_weights_scale.float()
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)
    W2 = W2_fp32 * S2_expanded

    # Routing
    topk_idx, weights = deepseek_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # Expand weights to [T, E_GLOBAL]
    weights_full = torch.zeros(T, E_GLOBAL, dtype=torch.float32, device=device)
    weights_full.scatter_(1, topk_idx, weights)

    # Expert compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A[token_idx]
        W13_e = W13[le]
        W2_e = W2[le]

        G1 = A_e @ W13_e.t()
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1
        O = C @ W2_e.t()

        w_tok = weights_full[token_idx, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


def verify(output: torch.Tensor, reference: torch.Tensor, atol: float = 1e-1, rtol: float = 1e-1):
    """Verify output against reference with tolerances appropriate for FP8."""
    if output.shape != reference.shape:
        print(f"FAIL: Shape mismatch {output.shape} vs {reference.shape}")
        return False

    diff = (output.float() - reference.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_norm = reference.float().abs().mean().item()

    # Relative check accounting for FP8 quantization noise
    passed = torch.allclose(output.float(), reference.float(), atol=atol, rtol=rtol)

    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Ref norm: {ref_norm:.6f}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def benchmark_run(test_args, use_cutedsl, warmup=5, iterations=20):
    """Benchmark the MoE kernel using CUDA events for accurate GPU timing."""
    # Warmup (includes JIT compilation on first run)
    for _ in range(warmup):
        _ = run(*test_args, use_cutedsl=use_cutedsl)
    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(iterations):
        start_event.record()
        output = run(*test_args, use_cutedsl=use_cutedsl)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    times_ms.sort()
    # Drop top/bottom 10% for stability
    trim = max(1, iterations // 10)
    trimmed = times_ms[trim:-trim] if len(times_ms) > 2 * trim else times_ms
    avg_ms = sum(trimmed) / len(trimmed)
    min_ms = times_ms[0]
    med_ms = times_ms[len(times_ms) // 2]

    return output, avg_ms, min_ms, med_ms


def main():
    parser = argparse.ArgumentParser(description="MoE FP8 Block-Scale Kernel")
    parser.add_argument("--num_tokens", type=int, default=4, help="Number of tokens")
    parser.add_argument("--local_expert_offset", type=int, default=0)
    parser.add_argument("--routed_scaling_factor", type=float, default=1.0)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cutedsl", action="store_true",
                        help="Use CuTe DSL blockwise grouped GEMM (requires SM100a)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()

    T = args.num_tokens
    backend = "CuTe DSL" if args.use_cutedsl else "PyTorch"
    print(f"MoE FP8 Block-Scale Kernel Test")
    print(f"  Backend: {backend}")
    print(f"  Tokens: {T}, H: {H}, I: {I}")
    print(f"  E_global: {E_GLOBAL}, E_local: {E_LOCAL}")
    print(f"  top_k: {TOP_K}, n_group: {N_GROUP}, topk_group: {TOPK_GROUP}")
    print(f"  Block size: {BLOCK}")
    print()

    # Create test data
    print("Creating test tensors...")
    test_args = create_test_tensors(T)

    if args.benchmark:
        print(f"Benchmarking ({args.warmup} warmup, {args.iterations} iterations)...")
        output, avg_ms, min_ms, med_ms = benchmark_run(
            test_args, args.use_cutedsl, args.warmup, args.iterations
        )
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"  Avg: {avg_ms:.3f} ms | Min: {min_ms:.3f} ms | Median: {med_ms:.3f} ms")

        # Compute TFLOPS (2 GEMMs per token-expert pair)
        # GEMM1: [Tsum, H] x [2*I, H].T  → 2 * Tsum * H * 2*I FLOPs
        # GEMM2: [Tsum, I] x [H, I].T     → 2 * Tsum * I * H FLOPs
        # Tsum ≈ T * TOP_K (each token routed to TOP_K local experts on average)
        # But only local experts matter: ~T * TOP_K * (E_LOCAL / E_GLOBAL)
        avg_tokens_per_expert = T * TOP_K / E_GLOBAL
        tsum_est = avg_tokens_per_expert * E_LOCAL
        gemm1_flops = 2.0 * tsum_est * H * (2 * I)
        gemm2_flops = 2.0 * tsum_est * I * H
        total_flops = gemm1_flops + gemm2_flops
        tflops = total_flops / (avg_ms * 1e-3) / 1e12
        print(f"  Est. Tsum: {tsum_est:.0f} | Total FLOPs: {total_flops:.2e} | {tflops:.2f} TFLOPS")
    else:
        # Run our implementation
        print("Running MoE kernel...")
        output = run(*test_args, use_cutedsl=args.use_cutedsl)
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")

    if not args.skip_ref_check:
        # Run reference
        print("Running reference...")
        ref_output = reference_run(*test_args)
        print(f"  Reference shape: {ref_output.shape}, dtype: {ref_output.dtype}")

        # Verify
        print("Verifying...")
        verify(output, ref_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
