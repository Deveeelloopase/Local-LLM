# kernels/hip_optimizer/

AMD ROCm/HIP backend.

| File | Purpose |
|---|---|
| `hip_matmul.cpp` | HIP kernel(s) for ternary / 1-bit matrix multiplication |
| `hip_kernels.py` | Python loader used by `UniversalKernelManager` |
| `Makefile` | Builds `ternary_matmul.so` from `hip_matmul.cpp` with `hipcc -O3 -shared -fPIC` |
| `__init__.py` | Package marker |

Second in the auto-select order (`cuda → hip → metal → vulkan`). Needs a ROCm install; on APUs and older gfx targets you may need `HSA_OVERRIDE_GFX_VERSION`.
