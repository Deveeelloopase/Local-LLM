# engine/kernels/cuda_kernels.py
# ==============================
# Robust wrapper that imports the real CUDA kernels on demand.
# This avoids IDE warnings and is safe in production.

__all__ = ["load_cuda_kernel", "ternary_matmul", "CudaDmaPool"]

def __getattr__(name):
    if name in ("load_cuda_kernel", "ternary_matmul"):
        from local_llm_core.kernels.cuda_optimizer.cuda_kernels import load_cuda_kernel, ternary_matmul
        # cache in globals
        globals()[name] = locals()[name]
        return locals()[name]
    if name == "CudaDmaPool":
        try:
            from local_llm_core.kernels.cuda_optimizer.cuda_dma_pool import CudaDmaPool
        except ImportError:
            class CudaDmaPool:
                pass
        globals()["CudaDmaPool"] = CudaDmaPool
        return CudaDmaPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(__all__)
