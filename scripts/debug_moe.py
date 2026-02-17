#!/usr/bin/env python3
"""
Debug script for MoE kernel - runs kernel and checks correctness against reference.
Run: FIB_DATASET_PATH=/home/qsq/mlsys2026/mlsys26-contest python scripts/debug_moe.py
With NCU profiling: python scripts/debug_moe.py --ncu_profile
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flashinfer_bench import TraceSet, Solution
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from flashinfer_bench.compile.registry import get_builder_registry
from flashinfer_bench.utils import dtype_str_to_torch_dtype


def main():
    parser = argparse.ArgumentParser(description="Debug MoE kernel")
    parser.add_argument(
        "--ncu_profile",
        action="store_true",
        help="NCU profiling mode: skip reference run and result check",
    )
    args = parser.parse_args()
    ncu_profile = args.ncu_profile
    dataset_path = os.environ.get("FIB_DATASET_PATH", "/home/qsq/mlsys2026/mlsys26-contest")
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print("Loading trace set...")
    trace_set = TraceSet.from_path(dataset_path)
    defn = trace_set.definitions["moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]
    workloads = trace_set.workloads["moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]

    # Load solution
    solution_path = PROJECT_ROOT / "solution.json"
    if not solution_path.exists():
        print("Run pack_solution first: python scripts/pack_solution.py")
        sys.exit(1)

    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Building solution: {solution.name}...")

    registry = get_builder_registry()
    runnable = registry.build(defn, solution)

    # Use workload index (change to debug different workloads)
    wl_index = int(os.environ.get("WL_INDEX", "0"))
    # trace = workloads[wl_index]
    trace = workloads[8]
    wl = trace.workload if hasattr(trace, "workload") else trace
    print(f"Testing workload: {wl.uuid[:8]}... (seq_len={wl.axes.get('seq_len')})")

    traceset_root = Path(dataset_path)
    stensors = load_safetensors(defn, wl, traceset_root)
    device = "cuda:0"
    inputs = gen_inputs(defn, wl, device, stensors)

    print("Input shapes:")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v} (scalar)")

    output_names = list(defn.outputs.keys())
    output_dtypes = {k: dtype_str_to_torch_dtype(v.dtype) for k, v in defn.outputs.items()}

    if not ncu_profile:
        # Build reference and run
        print("\nRunning reference implementation...")
        ref_runnable = registry.build_reference(defn)
        with torch.no_grad():
            ref_out = ref_runnable(**inputs)
        torch.cuda.synchronize(device)
        ref_out = normalize_outputs(
            ref_out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )

    print("Running solution kernel...")
    try:
        with torch.no_grad():
            out = runnable(**inputs)
        torch.cuda.synchronize(device)
    except Exception as e:
        print(f"\n*** RUNTIME_ERROR ***\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    out = normalize_outputs(
        out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
    )

    if ncu_profile:
        print("\n*** NCU profile mode: skipped reference run and result check ***")
        sys.exit(0)

    # Result check: compare against reference
    cfg = BenchmarkConfig(rtol=1e-2, atol=1e-2, required_matched_ratio=0.95)
    ref_tensor = ref_out["output"]
    out_tensor = out["output"]

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(
        out_tensor, ref_tensor, cfg
    )

    print(f"\n--- Result Check ---")
    print(f"  Output shape: {out_tensor.shape}")
    print(f"  Max absolute error: {max_abs:.2e}")
    print(f"  Max relative error: {max_rel:.2e}")
    print(f"  Matched ratio (within atol={cfg.atol}, rtol={cfg.rtol}): {matched_ratio:.2%}")
    print(f"  Required matched ratio: {cfg.required_matched_ratio or 1.0:.0%}")

    # Diagnostic: sample comparison
    ref_f = ref_tensor.float()
    out_f = out_tensor.float()
    diff = (out_f - ref_f).abs()
    max_idx = diff.argmax().item()
    row, col = max_idx // ref_tensor.shape[1], max_idx % ref_tensor.shape[1]
    print(f"\n--- Diagnostic (worst element at [{row},{col}]) ---")
    print(f"  Reference: {ref_f.flatten()[max_idx].item():.6f}")
    print(f"  FlashInfer: {out_f.flatten()[max_idx].item():.6f}")
    ratio_val = (out_f.flatten()[max_idx] / (ref_f.flatten()[max_idx].abs() + 1e-8)).item()
    print(f"  Ratio (out/ref): {ratio_val:.4f}")

    # Ratio distribution: where ref != 0, compute out/ref
    mask = ref_f.abs() > 1e-6
    if mask.any():
        ratios = (out_f[mask] / ref_f[mask])
        print(f"  Ratio stats (where ref!=0): min={ratios.min().item():.4f} max={ratios.max().item():.4f} mean={ratios.mean().item():.4f}")

    # Check for scale/offset pattern
    ref_mean, ref_std = ref_f.mean().item(), ref_f.std().item()
    out_mean, out_std = out_f.mean().item(), out_f.std().item()
    print(f"  Ref  mean={ref_mean:.4f} std={ref_std:.4f}")
    print(f"  Out  mean={out_mean:.4f} std={out_std:.4f}")

    if exceeds_tol:
        print(f"\n*** FAIL: Correctness check failed (matched ratio {matched_ratio:.2%} < {cfg.required_matched_ratio or 1.0:.0%})")
        print("\nNote: FlashInfer trtllm_fp8_block_scale_moe may use a different FP8 block-scale")
        print("layout than the benchmark reference. See FlashInfer/TensorRT-LLM docs.")
        sys.exit(1)
    else:
        print(f"\n*** PASS: Correctness check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
