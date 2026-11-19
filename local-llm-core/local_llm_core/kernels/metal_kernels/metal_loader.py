"""
metal_loader.py — Compiles and loads the Metal 1‑bit matmul shader.
====================================================================
Requires pyobjc‑framework‑Metal (pip install pyobjc‑framework‑Metal).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

_METAL_AVAIL = False
try:
    import Metal
    _METAL_AVAIL = True
except ImportError:
    pass

_device = None
_library = None
_pipeline = None
_lock = threading.Lock()

def _load_metal_kernel() -> bool:
    global _device, _library, _pipeline
    if not _METAL_AVAIL:
        return False
    with _lock:
        if _pipeline is not None:
            return True
        try:
            _device = Metal.MTLCreateSystemDefaultDevice()
            source = (Path(__file__).parent / "matmul_1bit.metal").read_text()
            library = _device.newLibraryWithSource_options_error_(source, None, None)[0]
            func = library.newFunctionWithName_("ternary_matmul")
            _pipeline = _device.newComputePipelineStateWithFunction_error_(func, None)[0]
            return True
        except Exception as e:
            log.error("Metal kernels init failed: %s", e)
            return False

def metal_ternary_matmul(a: torch.Tensor, b_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not _load_metal_kernel():
        return torch.matmul(a.float(), b_packed.float())  # fallback
    # a: (M, K) half; b_packed: (K, N) int8; scale: (N) float
    M, K = a.shape
    _, N = b_packed.shape
    c = torch.empty(M, N, dtype=torch.float16, device='mps')

    cmd_buffer = _device.newCommandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(_pipeline)
    encoder.setBuffer_offset_atIndex_(a, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_packed, 0, 1)
    encoder.setBuffer_offset_atIndex_(scale, 0, 2)
    encoder.setBuffer_offset_atIndex_(c, 0, 3)

    # set constants M,N,K via a small buffer
    consts = torch.tensor([M, N, K], dtype=torch.int32, device='cpu')
    encoder.setBytes_length_atIndex_(consts.data_ptr(), 12, 4)

    threads_per_group = max(_pipeline.maxTotalThreadsPerThreadgroup(), 1)
    threads = min(256, threads_per_group)
    grid_size = ((M + threads-1)//threads, (N + threads-1)//threads, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, threads)

    encoder.endEncoding()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    return c
