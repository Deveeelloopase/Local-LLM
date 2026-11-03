"""
cuda_kernels.py — JIT‑compiles and loads the CUDA 1‑bit matmul kernels.
======================================================================
Uses ctypes to load the compiled .so and provides a Python callable
that can be inserted into PyTorch models.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)

_CUDA_AVAIL = torch.cuda.is_available()
_CUDA_LIB = None

def _find_nvcc() -> Optional[str]:
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]:
        if subprocess.call(["which", candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return candidate
    return None

def _compile_kernel() -> Optional[str]:
    nvcc = _find_nvcc()
    if not nvcc:
        log.warning("nvcc not found; cannot compile CUDA kernels")
        return None
    # Use a fixed temporary directory for caching
    cache_dir = Path(tempfile.gettempdir()) / "llm_cuda_kernels"
    cache_dir.mkdir(exist_ok=True)
    so_path = cache_dir / "ternary_matmul.so"
    if so_path.exists():
        return str(so_path)

    cu_src = Path(__file__).parent / "cuda_matmul.cu"
    cmd = [
        nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC",
        str(cu_src), "-o", str(so_path),
        "-I", os.path.dirname(torch.utils.cmake_prefix_path),
    ]
    log.info("Compiling CUDA kernels: %s", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        log.error("CUDA compile failed:\n%s", ret.stderr)
        return None
    return str(so_path)

def load_cuda_kernel() -> Optional[Callable]:
    global _CUDA_LIB
    if _CUDA_LIB is not None:
        return _CUDA_LIB
    so_path = _compile_kernel()
    if not so_path:
        return None
    try:
        lib = ctypes.CDLL(so_path)
        # Set up argument types for the kernels launcher wrapper (we'll use a helper)
        # Store the library and return a wrapper function
        _CUDA_LIB = lib
        return lib
    except Exception as e:
        log.error("Failed to load CUDA kernels: %s", e)
        return None

def ternary_matmul(a: torch.Tensor, b_weights: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Call the custom CUDA kernels if loaded, else fallback to PyTorch."""
    if not _CUDA_AVAIL:
        return torch.matmul(a.float(), b_weights.float())  # fallback
    lib = load_cuda_kernel()
    if lib is None:
        return torch.matmul(a.float(), b_weights.float())

    # b_weights expected as packed int8 tensor; for simplicity assume we pack beforehand
    M, K = a.shape
    K2, N = b_weights.shape
    c = torch.empty(M, N, dtype=torch.float16, device=a.device)

    grid = ((M + 15) // 16, (N + 15) // 16)
    block = (16, 16, 1)
    lib.ternary_matmul_kernel(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b_weights.data_ptr()),
        ctypes.c_void_p(scale.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        grid=grid, block=block,
    )
    return c
