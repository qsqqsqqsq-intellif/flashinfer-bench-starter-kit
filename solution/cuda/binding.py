"""
CUDA Multi-Kernel MOE — Python Orchestration (binding.py)

Compiles kernel.cu into a shared library, loads it via ctypes, and
orchestrates the full DeepSeek-V3 MoE pipeline:

  1. Routing           → CUDA kernel
  2. Token permutation → 3 CUDA kernels (count → prefix-sum → scatter)
  3. Gather + Dequant  → CUDA kernel (fused FP8 gather + 1D blockscale dequant)
  4. Per-expert loop:
       Dequant weights → CUDA kernel
       GEMM1           → torch.matmul (cuBLAS)
       SwiGLU          → CUDA kernel
       GEMM2           → torch.matmul (cuBLAS)
  5. Weighted scatter-add → CUDA kernel
  6. Cast to bfloat16     → torch.to()
"""

import ctypes
import os
import subprocess
import sys

import torch

# ==============================================================================
# Constants (must match kernel.cu)
# ==============================================================================
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128


# ==============================================================================
# Compilation and loading
# ==============================================================================
_lib = None


def _compile_and_load():
    """Compile kernel.cu with nvcc and load the shared library."""
    global _lib
    if _lib is not None:
        return _lib

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(src_dir, "kernel.cu")
    so_path = os.path.join(src_dir, "moe_kernels.so")

    # Recompile if source is newer than .so
    needs_compile = (
        not os.path.exists(so_path)
        or os.path.getmtime(cu_path) > os.path.getmtime(so_path)
    )

    if needs_compile:
        # Detect GPU architecture
        arch_flag = _detect_arch()

        # Find nvcc
        nvcc = _find_nvcc()

        cmd = [
            nvcc,
            "-shared",
            "-Xcompiler", "-fPIC",
            "-o", so_path,
            cu_path,
            "-O3",
            arch_flag,
        ]
        print(f"[binding.py] Compiling CUDA kernels: {' '.join(cmd)}", file=sys.stderr)
        subprocess.check_call(cmd)
        print(f"[binding.py] Compiled to {so_path}", file=sys.stderr)

    _lib = ctypes.CDLL(so_path)
    return _lib


def _detect_arch():
    """Detect GPU architecture and return appropriate nvcc flag."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        major, minor = cap
        if major >= 10:
            return f"-arch=sm_{major}0a"
        elif major == 9:
            return "-arch=sm_90a"
        elif major == 8 and minor >= 9:
            return "-arch=sm_89"
        else:
            return f"-arch=sm_{major}{minor}"
    return "-arch=sm_90a"  # default


def _find_nvcc():
    """Find the nvcc compiler."""
    import shutil

    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    for env_var in ["CUDA_HOME", "CUDA_INSTALL_PATH", "CUDACXX"]:
        cuda_dir = os.environ.get(env_var, "")
        if not cuda_dir:
            continue
        # CUDACXX might point directly to nvcc
        if env_var == "CUDACXX" and os.path.isfile(cuda_dir):
            return cuda_dir
        candidate = os.path.join(cuda_dir, "bin", "nvcc")
        if os.path.isfile(candidate):
            return candidate

    # Common locations
    for path in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]:
        if os.path.isfile(path):
            return path

    raise RuntimeError("Cannot find nvcc. Set CUDA_HOME or add nvcc to PATH.")


# ==============================================================================
# ctypes helpers
# ==============================================================================
def _ptr(tensor):
    """Get raw data pointer from a torch tensor for ctypes."""
    return ctypes.c_void_p(tensor.data_ptr())


def _stream():
    """Get current PyTorch CUDA stream handle for ctypes."""
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


# ==============================================================================
# Main kernel function
# ==============================================================================
@torch.no_grad()
def kernel(
    routing_logits,         # [T, E_GLOBAL] float32
    routing_bias,           # [E_GLOBAL] float32
    hidden_states,          # [T, H] float8_e4m3fn
    hidden_states_scale,    # [H/BLOCK, T] float32
    gemm1_weights,          # [E_LOCAL, 2*I, H] float8_e4m3fn
    gemm1_weights_scale,    # [E_LOCAL, (2*I)/BLOCK, H/BLOCK] float32
    gemm2_weights,          # [E_LOCAL, H, I] float8_e4m3fn
    gemm2_weights_scale,    # [E_LOCAL, H/BLOCK, I/BLOCK] float32
    local_expert_offset,    # int scalar
    routed_scaling_factor,  # float scalar
):
    """
    MoE FP8 Block-Scale kernel entry point.

    Implements the full DeepSeek-V3 MoE pipeline using custom CUDA
    kernels for non-GEMM operations and cuBLAS (via torch.matmul)
    for the two expert GEMMs.
    """
    lib = _compile_and_load()
    T = routing_logits.shape[0]
    device = routing_logits.device
    local_expert_offset = int(local_expert_offset)
    routed_scaling_factor = float(routed_scaling_factor)
    stream = _stream()

    # Ensure float32 contiguous inputs for routing
    logits_f32 = routing_logits.float().contiguous()
    bias_f32 = routing_bias.float().reshape(-1).contiguous()

    # ==================================================================
    # Step 1: Routing — CUDA kernel
    # ==================================================================
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int64, device=device)
    weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)

    lib.launch_routing(
        _ptr(logits_f32), _ptr(bias_f32),
        _ptr(topk_idx), _ptr(weights),
        ctypes.c_float(routed_scaling_factor),
        ctypes.c_int(T),
        stream,
    )

    # ==================================================================
    # Step 2: Token permutation — 3 CUDA kernels
    # ==================================================================
    N = T * TOP_K
    flat_idx = topk_idx.reshape(-1)  # [N]

    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(E_LOCAL + 1, dtype=torch.int32, device=device)
    sorted_token_ids = torch.empty(N, dtype=torch.int64, device=device)
    token_expert_map = torch.empty(N, dtype=torch.int32, device=device)
    write_counters = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)

    # 2a: Count tokens per expert
    lib.launch_permute_count(
        _ptr(flat_idx), _ptr(counts),
        ctypes.c_int(local_expert_offset),
        ctypes.c_int(N),
        stream,
    )

    # 2b: Prefix sum
    lib.launch_permute_prefix_sum(
        _ptr(counts), _ptr(expert_offsets),
        stream,
    )

    # 2c: Scatter
    lib.launch_permute_scatter(
        _ptr(flat_idx), _ptr(expert_offsets),
        _ptr(sorted_token_ids), _ptr(token_expert_map),
        _ptr(write_counters),
        ctypes.c_int(local_expert_offset),
        ctypes.c_int(N),
        ctypes.c_int(TOP_K),
        stream,
    )

    # Need Tsum on CPU to size subsequent allocations
    torch.cuda.synchronize()
    Tsum = expert_offsets[E_LOCAL].item()

    if Tsum == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    sorted_token_ids = sorted_token_ids[:Tsum].contiguous()
    token_expert_map = token_expert_map[:Tsum].contiguous()
    offsets_cpu = expert_offsets.cpu()

    # ==================================================================
    # Step 3: Gather + Dequant hidden states — CUDA kernel
    # ==================================================================
    H_BLOCKS = H // BLOCK
    hs_scale_f32 = hidden_states_scale.float().contiguous()

    A = torch.empty((Tsum, H), dtype=torch.float32, device=device)

    lib.launch_gather_dequant(
        _ptr(hidden_states),  # FP8 bytes via data_ptr
        _ptr(hs_scale_f32),
        _ptr(sorted_token_ids),
        _ptr(A),
        ctypes.c_int(T),
        ctypes.c_int(Tsum),
        ctypes.c_int(H),
        ctypes.c_int(H_BLOCKS),
        stream,
    )

    # ==================================================================
    # Step 4: Per-expert compute loop
    # ==================================================================
    gemm2_buffer = torch.zeros((Tsum, H), dtype=torch.float32, device=device)

    # Pre-allocate weight dequant buffers (reused each iteration)
    W1_buf = torch.empty((2 * I, H), dtype=torch.float32, device=device)
    W2_buf = torch.empty((H, I), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        start = offsets_cpu[le].item()
        end = offsets_cpu[le + 1].item()
        if start >= end:
            continue
        Tk = end - start

        # --- 4a: Dequant weight1 for this expert ---
        w1_fp8 = gemm1_weights[le].contiguous()         # [2*I, H] FP8
        w1_scale = gemm1_weights_scale[le].float().contiguous()  # [(2*I)/BLOCK, H/BLOCK]

        lib.launch_dequant_2d(
            _ptr(w1_fp8),
            _ptr(w1_scale),
            _ptr(W1_buf),
            ctypes.c_int(2 * I),
            ctypes.c_int(H),
            ctypes.c_int(H // BLOCK),
            stream,
        )

        # --- 4b: GEMM1 via cuBLAS ---
        # A_e: [Tk, H], W1_buf.t(): [H, 2*I] → G1: [Tk, 2*I]
        A_e = A[start:end]
        G1 = torch.matmul(A_e, W1_buf.t())

        # --- 4c: SwiGLU activation ---
        C = torch.empty((Tk, I), dtype=torch.float32, device=device)
        lib.launch_swiglu(
            _ptr(G1),
            _ptr(C),
            ctypes.c_int(Tk),
            ctypes.c_int(I),
            stream,
        )

        # --- 4d: Dequant weight2 for this expert ---
        w2_fp8 = gemm2_weights[le].contiguous()          # [H, I] FP8
        w2_scale = gemm2_weights_scale[le].float().contiguous()  # [H/BLOCK, I/BLOCK]

        lib.launch_dequant_2d(
            _ptr(w2_fp8),
            _ptr(w2_scale),
            _ptr(W2_buf),
            ctypes.c_int(H),
            ctypes.c_int(I),
            ctypes.c_int(I // BLOCK),
            stream,
        )

        # --- 4e: GEMM2 via cuBLAS ---
        # C: [Tk, I], W2_buf.t(): [I, H] → [Tk, H]
        gemm2_buffer[start:end] = torch.matmul(C, W2_buf.t())

    # ==================================================================
    # Step 5: Weighted scatter-add — CUDA kernel
    # ==================================================================
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    lib.launch_weighted_scatter_add(
        _ptr(gemm2_buffer),
        _ptr(sorted_token_ids),
        _ptr(token_expert_map),
        _ptr(topk_idx),
        _ptr(weights),
        _ptr(output),
        ctypes.c_int(local_expert_offset),
        ctypes.c_int(Tsum),
        ctypes.c_int(H),
        ctypes.c_int(TOP_K),
        stream,
    )

    # ==================================================================
    # Step 6: Cast to bfloat16
    # ==================================================================
    return output.to(torch.bfloat16)


# ==============================================================================
# TVM FFI registration (for FlashInfer-Bench framework)
# ==============================================================================
try:
    from tvm.ffi import register_func
    register_func("flashinfer.kernel")(kernel)
except ImportError:
    pass
