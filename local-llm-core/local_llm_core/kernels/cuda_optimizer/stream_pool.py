"""
kernels/cuda_optimizer/stream_pool.py — CUDA DMA Pool with Async Streaming
============================================================================
Provides CudaStreamPool for asynchronous memory transfers and
overlapped kernels execution, reducing latency and improving GPU utilisation.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import List, Optional

import torch

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load the compiled CUDA DMA pool library
# -----------------------------------------------------------------------------
import os as _os
_DMA_SO = _os.path.join(_os.path.dirname(__file__), "cuda_dma_pool.so")
try:
    _lib = ctypes.CDLL(_DMA_SO)
except OSError:
    _lib = None

if _lib is not None:
    _lib.cuda_pool_device_init.argtypes = [ctypes.c_size_t]
    _lib.cuda_pool_device_malloc.argtypes = [ctypes.c_size_t]
    _lib.cuda_pool_device_free.argtypes = [ctypes.c_void_p]

# -----------------------------------------------------------------------------
# Stream Management
# -----------------------------------------------------------------------------
class CudaStream:
    """Wrapper around a CUDA stream for asynchronous operations."""
    def __init__(self, device: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._stream = torch.cuda.Stream(device=device)

    def __enter__(self):
        torch.cuda.synchronize()
        self._stream.__enter__()
        return self._stream

    def __exit__(self, *args):
        self._stream.__exit__(*args)
        self._stream.synchronize()

class CudaStreamPool:
    """Pool of CUDA streams for parallel memory transfers and kernels execution."""

    def __init__(self, num_streams: int = 4, device: int = 0):
        self._streams = [CudaStream(device) for _ in range(num_streams)]
        self._index = 0
        self._lock = threading.Lock()

    def get_stream(self) -> CudaStream:
        with self._lock:
            stream = self._streams[self._index]
            self._index = (self._index + 1) % len(self._streams)
            return stream

    def synchronize_all(self):
        for s in self._streams:
            s._stream.synchronize()

# -----------------------------------------------------------------------------
# DMA Pool with Streaming
# -----------------------------------------------------------------------------
class StreamingDmaPool:
    """
    Uses the pre‑allocated DMA pool and CUDA streams to asynchronously
    copy data between host and device, enabling overlap with kernels.
    """

    def __init__(self, device_mb: int = 512, host_mb: int = 256, num_streams: int = 4):
        self._stream_pool = CudaStreamPool(num_streams)
        self._device_buffers: List[torch.Tensor] = []
        self._host_buffers: List[torch.Tensor] = []

        # Pre-allocate device and pinned host memory pools
        for _ in range(8):
            dev_buf = torch.empty(device_mb * 1024 * 1024, dtype=torch.uint8, device='cuda')
            host_buf = torch.empty(host_mb * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
            self._device_buffers.append(dev_buf)
            self._host_buffers.append(host_buf)

    def async_copy_host_to_device(self, host_src: torch.Tensor, device_dst: torch.Tensor):
        """Asynchronously copy from host to device using a stream."""
        stream = self._stream_pool.get_stream()
        with stream:
            device_dst.copy_(host_src, non_blocking=True)

    def async_copy_device_to_host(self, device_src: torch.Tensor, host_dst: torch.Tensor):
        stream = self._stream_pool.get_stream()
        with stream:
            host_dst.copy_(device_src, non_blocking=True)

    def allocate_device_buffer(self, size: int) -> torch.Tensor:
        # Simple bump allocator from the pool (production would have a proper manager)
        for buf in self._device_buffers:
            if buf.numel() >= size:
                return buf[:size]
        raise RuntimeError("No free device buffer of requested size")

    def allocate_host_buffer(self, size: int) -> torch.Tensor:
        for buf in self._host_buffers:
            if buf.numel() >= size:
                return buf[:size]
        raise RuntimeError("No free host buffer of requested size")

    def synchronize(self):
        self._stream_pool.synchronize_all()
