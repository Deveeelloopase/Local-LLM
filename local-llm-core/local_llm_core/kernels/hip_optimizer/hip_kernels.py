"""
hip_kernels.py — Production‑ready HIP 1‑bit matmul kernels
====================================================================
JIT‑compiles hip_matmul.cpp, loads the shared library, and exposes
`hip_ternary_matmul()`, which falls back to PyTorch on failure.

Fixes:
  • Host wrapper now uses triple‑chevron <<<>>> launch.
  • ctypes signatures precisely match the new 'launch_ternary_matmul'.
  • Extra hipcc discovery paths and environment variable override.
  • Cache staleness recompilation.
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

_HIP_AVAIL = torch.cuda.is_available() and hasattr(torch.version, 'hip')
_HIP_LIB: Optional[ctypes.CDLL] = None
_HIP_LAUNCHER: Optional[Callable[..., int]] = None

HIPCC_ENV_VAR = "HIPCC"


def _find_hipcc() -> Optional[str]:
    """Locate hipcc compiler, allowing override via environment variable."""
    env_hipcc = os.environ.get(HIPCC_ENV_VAR)
    if env_hipcc and Path(env_hipcc).is_file():
        return env_hipcc

    # 2. Check common paths
    candidates = [
        "hipcc",
        "/opt/rocm/bin/hipcc",
        "/opt/rocm/llvm/bin/hipcc",
        "/usr/local/cuda/bin/hipcc",
    ]
    for cand in candidates:
        try:
            subprocess.run(
                [cand, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return cand
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _compile_kernel() -> Optional[str]:
    """Compile hip_matmul.cpp to a shared library if stale or missing."""
    hipcc = _find_hipcc()
    if not hipcc:
        log.warning("hipcc not found; cannot compile HIP kernels")
        return None

    cache_dir = Path(tempfile.gettempdir()) / "llm_hip_kernels"
    cache_dir.mkdir(exist_ok=True)

    so_path = cache_dir / "ternary_matmul.so"
    cpp_src = Path(__file__).parent / "hip_matmul.cpp"

    if not cpp_src.exists():
        log.error("HIP source not found: %s", cpp_src)
        return None

    # Recompile if .so missing or source is newer
    need_compile = True
    if so_path.exists():
        src_mtime = cpp_src.stat().st_mtime
        so_mtime = so_path.stat().st_mtime
        if so_mtime >= src_mtime:
            need_compile = False

    if need_compile:
        cmd = [
            hipcc, "-O3", "-shared", "-fPIC",
            str(cpp_src), "-o", str(so_path),
        ]
        log.info("Compiling HIP kernels: %s", " ".join(cmd))
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            log.error("HIP compile failed:\n%s", ret.stderr)
            try:
                so_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        log.info("HIP kernel compiled successfully")
    else:
        log.debug("HIP kernel .so up to date, skipping compilation")

    return str(so_path)


def load_hip_library() -> Optional[ctypes.CDLL]:
    """Load the compiled .so and return the CDLL handle.
    Stores the launcher function pointer globally for fast reuse."""
    global _HIP_LIB, _HIP_LAUNCHER
    if _HIP_LIB is not None and _HIP_LAUNCHER is not None:
        return _HIP_LIB

    so_path = _compile_kernel()
    if not so_path:
        return None

    try:
        lib = ctypes.CDLL(so_path)
    except OSError as e:
        log.error("Failed to load HIP kernel library: %s", e)
        return None

    # Define the prototype of launch_ternary_matmul
    # int launch_ternary_matmul(
    #     const __half* a, const int8_t* b, const float* scale, __half* c,
    #     int M, int N, int K,
    #     unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    #     unsigned int block_x, unsigned int block_y, unsigned int block_z
    # );
    func = lib.launch_ternary_matmul
    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.c_void_p,  # a
        ctypes.c_void_p,  # b
        ctypes.c_void_p,  # scale
        ctypes.c_void_p,  # c
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_uint,    # grid_x
        ctypes.c_uint,    # grid_y
        ctypes.c_uint,    # grid_z
        ctypes.c_uint,    # block_x
        ctypes.c_uint,    # block_y
        ctypes.c_uint,    # block_z
    ]

    _HIP_LIB = lib
    _HIP_LAUNCHER = func
    log.info("HIP kernel library loaded and launcher ready")
    return lib


def hip_ternary_matmul(
    a: torch.Tensor,
    b_weights: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Perform 1‑bit ternary matrix multiplication using the HIP kernel.
    Falls back to torch.matmul if HIP is not available or fails.

    Args:
        a: Input activations [M, K] (float16)
        b_weights: Packed ternary weights [K/4, N] (int8)
        scale: Per‑column scaling factors [N] (float32)
    Returns:
        Output tensor [M, N] (float16)
    """
    if not _HIP_AVAIL:
        return torch.matmul(a.float(), b_weights.float()).to(a.dtype)

    lib = load_hip_library()
    if lib is None or _HIP_LAUNCHER is None:
        return torch.matmul(a.float(), b_weights.float()).to(a.dtype)

    M, K = a.shape
    K2, N = b_weights.shape
    a_f16 = a.to(torch.float16) if a.dtype != torch.float16 else a
    b_int8 = b_weights.to(torch.int8) if b_weights.dtype != torch.int8 else b_weights
    scale_f32 = scale.to(torch.float32) if scale.dtype != torch.float32 else scale

    c = torch.empty(M, N, dtype=torch.float16, device=a.device)

    block = (16, 16, 1)
    grid_x = (M + block[0] - 1) // block[0]
    grid_y = (N + block[1] - 1) // block[1]
    grid_z = 1

    try:
        ret = _HIP_LAUNCHER(
            ctypes.c_void_p(a_f16.data_ptr()),
            ctypes.c_void_p(b_int8.data_ptr()),
            ctypes.c_void_p(scale_f32.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.c_uint(grid_x),
            ctypes.c_uint(grid_y),
            ctypes.c_uint(grid_z),
            ctypes.c_uint(block[0]),
            ctypes.c_uint(block[1]),
            ctypes.c_uint(block[2]),
        )
        if ret != 0:
            log.error("HIP kernel launch failed with code %d", ret)
            return torch.matmul(a.float(), b_weights.float()).to(a.dtype)
        return c
    except Exception as e:
        log.error("Error during HIP kernel launch: %s", e)
        return torch.matmul(a.float(), b_weights.float()).to(a.dtype)


# Convenience alias for legacy code
load_hip_kernel = load_hip_library
