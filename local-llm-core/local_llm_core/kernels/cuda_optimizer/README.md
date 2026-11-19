# kernels/cuda_optimizer/

NVIDIA CUDA backend.

| File | Purpose |
|---|---|
| `cuda_matmul.cu` | CUDA kernel(s) for ternary / 1-bit matrix multiplication |
| `cuda_dma_pool.c` | Userspace DMA buffer pool (C) |
| `cuda_kernels.py` | Python loader: finds `nvcc`, JIT-compiles `cuda_matmul.cu` and loads it with `ctypes` (`load_cuda_kernel`, `ternary_matmul`, `CudaDmaPool`) — re-exported lazily by `engine/kernels/cuda_kernels.py` |
| `stream_pool.py` | CUDA stream pool |
| `Makefile` | Builds `cuda_dma_pool.so` from `cuda_dma_pool.c` (needs `nvcc`) |
| `__init__.py` | Package marker |

Used when `UniversalKernelManager` selects the `cuda` backend (first in the auto-select order). Requires the CUDA toolkit; the `gpu` extra installs `cuda-python`.
