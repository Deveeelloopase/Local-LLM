# engine/kernels/metal/

| File | Purpose |
|---|---|
| `loader.py` | The canonical import point for the Metal backend used by `UniversalKernelManager`. It re-exports the production 1-bit loader from `local_llm_core/kernels/metal_kernels/` |
| `matmul.metal` | Reserved for future general-purpose Metal kernels (fp16, bf16, int8); the production BitNet 1-bit shader is `kernels/metal_kernels/matmul_1bit.metal` |
| `__init__.py` | Package marker |

Requires macOS and `pyobjc-framework-Metal` / `pyobjc-framework-MetalPerformanceShaders` at runtime (not installed by any `setup.py` extra, despite a comment in `loader.py` that mentions an `[apple]` extra).
