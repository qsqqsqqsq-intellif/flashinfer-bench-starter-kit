#!/usr/bin/env python3
"""
MoE kernel profiler - measures latency of each sub-kernel and the total kernel.

Uses FlashInfer-Bench framework to load workload[0] and profiles each stage
using CUDA events (GPU time) and perf_counter (CPU-side overhead).

Run:
  python scripts/profile_moe.py                          # PyTorch backend
  python scripts/profile_moe.py --use-cutedsl             # CuTe DSL backend
  python scripts/profile_moe.py --use-cutedsl --detail    # + GEMM/dlpack breakdown
  python scripts/profile_moe.py --warmup 20 --iterations 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from the original source tree so __file__-relative paths
# (e.g. blockwise_gemm/) resolve correctly inside moe_fp8_blockscale.py.
BLACKWELL_DIR = str(PROJECT_ROOT.parent / "examples" / "python" / "CuTeDSL" / "blackwell")
if BLACKWELL_DIR not in sys.path:
    sys.path.insert(0, BLACKWELL_DIR)

from flashinfer_bench import TraceSet
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

import moe_fp8_blockscale as moe

DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
DEVICE = "cuda:0"


# ─── Timing helpers ───────────────────────────────────────────────────────────

def cuda_time(fn, warmup: int, iterations: int) -> dict:
    """GPU time via CUDA events (captures kernel execution + GPU-idle gaps)."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))

    times.sort()
    trim = max(1, iterations // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return {"min_ms": times[0],
            "median_ms": times[len(times) // 2],
            "mean_ms": sum(trimmed) / len(trimmed)}


def cpu_wall_time(fn, warmup: int, iterations: int) -> dict:
    """CPU wall time via perf_counter (captures Python/CPU overhead).
    Syncs the GPU before starting so prior GPU work doesn't inflate the result."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    trim = max(1, iterations // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return {"min_ms": times[0],
            "median_ms": times[len(times) // 2],
            "mean_ms": sum(trimmed) / len(trimmed)}


def print_row(label: str, t: dict, indent: int = 0, tag: str = "GPU"):
    pad = "  " * indent
    print(f"  {pad}{label:<46s}[{tag}]  "
          f"min={t['min_ms']:7.3f} ms  "
          f"median={t['median_ms']:7.3f} ms  "
          f"mean={t['mean_ms']:7.3f} ms")


# ─── CuTe DSL detailed breakdown ──────────────────────────────────────────────

def profile_cutedsl_detail(
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    sorted_ids, expert_offsets, token_expert_map,
    topk_idx, weights,
    local_expert_offset, gidx,
    Tsum, T, W, N,
):
    """Profile the internals of expert_compute_cutedsl step by step."""
    import cutlass
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    BLOCK = moe.BLOCK
    I     = moe.I
    H     = moe.H

    # Ensure compiled GEMMs are in _gemm_cache (warm up expert_compute first).
    with torch.no_grad():
        for _ in range(3):
            moe.expert_compute_cutedsl(
                hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale,
                sorted_ids, expert_offsets,
                topk_idx, weights,
                local_expert_offset, token_expert_map,
            )
    torch.cuda.synchronize()

    # ── Prepare static tensors for GEMM1 ──
    a1, sfa1, b1, sfb1, gidx1 = moe.prepare_gemm1_inputs(
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        sorted_ids, token_expert_map,
    )
    pad_m   = (BLOCK - Tsum % BLOCK) % BLOCK
    valid_m = Tsum + pad_m
    a1_int8 = a1.view(torch.int8)
    if pad_m > 0:
        a1_int8 = F.pad(a1_int8, (0, 0, 0, pad_m))
        sfa1_p  = F.pad(sfa1,    (0, 0, 0, pad_m))
        gidx1_p = F.pad(gidx1,   (0, pad_m), value=0)
    else:
        sfa1_p, gidx1_p = sfa1, gidx1

    gemm1_out = torch.zeros(valid_m, 2 * I, dtype=torch.bfloat16, device=DEVICE)

    torch_stream  = torch.cuda.current_stream()
    cuda_stream   = cuda_driver.CUstream(torch_stream.cuda_stream)

    # ── 3d: from_dlpack + tensor layout overhead for GEMM1 (CPU-side) ──
    def make_cute_gemm1():
        a_c   = moe._make_cute_fp8_tensor(from_dlpack, a1_int8.unsqueeze(-1), cutlass)
        sfa_c = moe._make_cute_tensor(from_dlpack, sfa1_p.unsqueeze(-1))
        b_c   = moe._make_cute_fp8_tensor(
                    from_dlpack, b1.view(torch.int8).permute(1, 2, 0), cutlass)
        sfb_c = moe._make_cute_tensor(from_dlpack, sfb1.permute(1, 2, 0))
        c_c   = moe._make_cute_tensor(from_dlpack, gemm1_out.unsqueeze(-1))
        g_c   = from_dlpack(gidx1_p).mark_layout_dynamic()
        return a_c, sfa_c, b_c, sfb_c, c_c, g_c

    t = cpu_wall_time(make_cute_gemm1, W, N)
    print_row("  3d. from_dlpack + layout (GEMM1)", t, indent=1, tag="CPU")

    # ── 3e: CuTe DSL GEMM1 kernel ──
    a_c, sfa_c, b_c, sfb_c, c_c, g_c = make_cute_gemm1()
    compiled_g1 = moe._gemm_cache.get(("gemm1", valid_m))
    if compiled_g1:
        t = cuda_time(
            lambda: compiled_g1(a_c, b_c, c_c, sfa_c, sfb_c, g_c, cuda_stream),
            W, N)
        print_row("  3e. CuTe DSL GEMM1 kernel", t, indent=1)
    else:
        print("  3e. CuTe DSL GEMM1  (not found in cache)")

    # ── 3f: SwiGLU ──
    gemm1_result = gemm1_out[:Tsum].detach()
    t = cuda_time(
        lambda: F.silu(gemm1_result[:, I:].float()) * gemm1_result[:, :I].float(),
        W, N)
    print_row("  3f. SwiGLU", t, indent=1)

    # ── Prepare real intermediate for GEMM2 ──
    with torch.no_grad():
        gate         = gemm1_result[:, :I].float()
        up           = gemm1_result[:, I:].float()
        intermediate = F.silu(up) * gate

    a2, sfa2, b2, sfb2, gidx2 = moe.prepare_gemm2_inputs(
        intermediate, gemm2_weights, gemm2_weights_scale, gidx1[:Tsum],
    )
    a2_int8 = a2.view(torch.int8)
    if pad_m > 0:
        a2_int8 = F.pad(a2_int8, (0, 0, 0, pad_m))
        sfa2_p  = F.pad(sfa2,    (0, 0, 0, pad_m))
        gidx2_p = F.pad(gidx2,   (0, pad_m), value=0)
    else:
        sfa2_p, gidx2_p = sfa2, gidx2

    gemm2_out = torch.zeros(valid_m, H, dtype=torch.bfloat16, device=DEVICE)

    # ── 3g: from_dlpack + tensor layout overhead for GEMM2 (CPU-side) ──
    def make_cute_gemm2():
        a_c   = moe._make_cute_fp8_tensor(from_dlpack, a2_int8.unsqueeze(-1), cutlass)
        sfa_c = moe._make_cute_tensor(from_dlpack, sfa2_p.unsqueeze(-1))
        b_c   = moe._make_cute_fp8_tensor(
                    from_dlpack, b2.view(torch.int8).permute(1, 2, 0), cutlass)
        sfb_c = moe._make_cute_tensor(from_dlpack, sfb2.permute(1, 2, 0))
        c_c   = moe._make_cute_tensor(from_dlpack, gemm2_out.unsqueeze(-1))
        g_c   = from_dlpack(gidx2_p).mark_layout_dynamic()
        return a_c, sfa_c, b_c, sfb_c, c_c, g_c

    t = cpu_wall_time(make_cute_gemm2, W, N)
    print_row("  3g. from_dlpack + layout (GEMM2)", t, indent=1, tag="CPU")

    # ── 3h: CuTe DSL GEMM2 kernel ──
    a_c, sfa_c, b_c, sfb_c, c_c, g_c = make_cute_gemm2()
    compiled_g2 = moe._gemm_cache.get(("gemm2", valid_m))
    if compiled_g2:
        t = cuda_time(
            lambda: compiled_g2(a_c, b_c, c_c, sfa_c, sfb_c, g_c, cuda_stream),
            W, N)
        print_row("  3h. CuTe DSL GEMM2 kernel", t, indent=1)
    else:
        print("  3h. CuTe DSL GEMM2  (not found in cache)")

    # ── 3i: torch.cuda.synchronize ──
    t = cpu_wall_time(torch.cuda.synchronize, W, N)
    print_row("  3i. torch.cuda.synchronize()", t, indent=1, tag="CPU")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cutedsl", action="store_true",
                        help="Use CuTe DSL expert compute (default: PyTorch)")
    parser.add_argument("--detail", action="store_true",
                        help="Break down GEMM1/GEMM2 + from_dlpack overhead "
                             "(only meaningful with --use-cutedsl)")
    parser.add_argument("--warmup",     type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dataset", default=os.environ.get(
        "FIB_DATASET_PATH", "/home/qsq/mlsys2026/mlsys26-contest"))
    args = parser.parse_args()

    use_cutedsl = args.use_cutedsl
    W, N  = args.warmup, args.iterations
    backend = "CuTe DSL" if use_cutedsl else "PyTorch"

    # ── Load workload ──
    print(f"Loading trace set from {args.dataset}...")
    trace_set = TraceSet.from_path(args.dataset)
    defn      = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads[DEFINITION]

    trace = workloads[0]
    wl = trace.workload if hasattr(trace, "workload") else trace
    print(f"Workload[0]: uuid={wl.uuid[:8]}  axes={wl.axes}")

    stensors = load_safetensors(defn, wl, Path(args.dataset))
    inputs   = gen_inputs(defn, wl, DEVICE, stensors)

    routing_logits        = inputs["routing_logits"]
    routing_bias          = inputs["routing_bias"]
    hidden_states         = inputs["hidden_states"]
    hidden_states_scale   = inputs["hidden_states_scale"]
    gemm1_weights         = inputs["gemm1_weights"]
    gemm1_weights_scale   = inputs["gemm1_weights_scale"]
    gemm2_weights         = inputs["gemm2_weights"]
    gemm2_weights_scale   = inputs["gemm2_weights_scale"]
    local_expert_offset   = inputs["local_expert_offset"]
    routed_scaling_factor = inputs["routed_scaling_factor"]

    T = routing_logits.shape[0]
    print(f"Tokens (T): {T}")

    # ── Pre-compute intermediates ──
    with torch.no_grad():
        topk_idx, weights = moe.deepseek_routing(
            routing_logits, routing_bias, routed_scaling_factor)
        sorted_ids, expert_offsets, token_expert_map = moe.permute_tokens(
            topk_idx, local_expert_offset)
        Tsum = sorted_ids.shape[0]
        print(f"Tsum (tokens assigned to local experts): {Tsum}")

        _, _, _, _, gidx = moe.prepare_gemm1_inputs(
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            sorted_ids, token_expert_map,
        )

    dummy_intermediate = torch.zeros(Tsum, moe.I, device=DEVICE, dtype=torch.float32)
    dummy_gemm2_out    = torch.zeros(Tsum, moe.H, device=DEVICE, dtype=torch.float32)

    print(f"\nBackend: {backend} | warmup={W} | iterations={N}")
    print(f"Note: [GPU] = CUDA event time (kernel exec + GPU-idle gaps); "
          f"[CPU] = wall-clock time (Python/C++ overhead)\n")
    print(f"{'Sub-kernel':<50s}       {'min':>10s}   {'median':>10s}   {'mean':>10s}")
    print("  " + "-" * 95)

    # ── 1. Routing ──
    t = cuda_time(
        lambda: moe.deepseek_routing(routing_logits, routing_bias, routed_scaling_factor),
        W, N)
    print_row("1. routing (Triton)", t)

    # ── 2. Token permutation ──
    t = cuda_time(lambda: moe.permute_tokens(topk_idx, local_expert_offset), W, N)
    print_row("2. permute_tokens (Triton)", t)

    # ── 3. Expert compute (total) ──
    compute_fn = moe.expert_compute_cutedsl if use_cutedsl else moe.expert_compute_pytorch
    t = cuda_time(
        lambda: compute_fn(
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            sorted_ids, expert_offsets,
            topk_idx, weights,
            local_expert_offset, token_expert_map,
        ),
        W, N)
    print_row(f"3. expert_compute ({backend})", t)

    # ── 3a. prepare_gemm1 (Triton gather+scale) ──
    t = cuda_time(
        lambda: moe.prepare_gemm1_inputs(
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            sorted_ids, token_expert_map,
        ),
        W, N)
    print_row("  3a. prepare_gemm1 (Triton gather+scale)", t, indent=1)

    # ── 3b. prepare_gemm2 (Triton FP8 quantize) ──
    t = cuda_time(
        lambda: moe.prepare_gemm2_inputs(
            dummy_intermediate, gemm2_weights, gemm2_weights_scale, gidx,
        ),
        W, N)
    print_row("  3b. prepare_gemm2 (Triton FP8 quant)", t, indent=1)

    # ── 3c. weighted_scatter_add (Triton) ──
    t = cuda_time(
        lambda: moe.weighted_scatter_add(
            dummy_gemm2_out, sorted_ids, token_expert_map,
            topk_idx, weights, local_expert_offset, T,
        ),
        W, N)
    print_row("  3c. weighted_scatter_add (Triton)", t, indent=1)

    # ── 3d-3i. CuTe DSL internals (--detail only) ──
    if use_cutedsl and args.detail:
        try:
            profile_cutedsl_detail(
                hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale,
                sorted_ids, expert_offsets, token_expert_map,
                topk_idx, weights,
                local_expert_offset, gidx,
                Tsum, T, W, N,
            )
        except Exception as e:
            import traceback
            print(f"  (detail profiling error: {e})")
            traceback.print_exc()

    # ── Total end-to-end kernel ──
    print("  " + "-" * 95)
    t = cuda_time(
        lambda: moe.run(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            use_cutedsl=use_cutedsl,
        ),
        W, N)
    print_row("TOTAL kernel", t)
    print()


if __name__ == "__main__":
    main()
