"""
CUDA MOE Kernel Binding (unused - framework compiles kernel.cu directly)

The FlashInfer-Bench framework compiles kernel.cu via torch.utils.cpp_extension.load()
and accesses the kernel function through pybind11. This file is not used at runtime.
"""
