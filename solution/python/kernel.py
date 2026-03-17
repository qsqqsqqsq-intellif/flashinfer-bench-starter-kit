"""
CUTLASS CuTe DSL MoE Kernel for FlashInfer-Bench.

Wraps the CUTLASS moe_fp8_blockscale_cursor implementation for the
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 definition.

Computation flow:
  1. DeepSeek-V3 no-aux routing (fused Triton kernel)
  2. Token permutation (fused Triton kernel)
  3. Per-expert: GEMM1 (FP8 blockwise) -> SwiGLU -> GEMM2 (FP8 blockwise)
  4. Weighted scatter-add back to output (fused Triton kernel)
"""

import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_fp8_blockscale import run as _run


@torch.no_grad()
def kernel(
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
) -> torch.Tensor:
    """
    DeepSeek-V3/R1 FP8 block-scale MoE kernel using CUTLASS CuTe DSL.

    Uses Triton-fused routing + permutation + scatter, and optionally
    the CuTe DSL blockwise contiguous grouped GEMM for expert compute.
    """
    return _run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
        use_cutedsl=True,
    )
