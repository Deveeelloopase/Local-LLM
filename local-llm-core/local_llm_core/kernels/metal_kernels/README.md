# kernels/metal_kernels/

Apple Metal backend (macOS).

| File | Purpose |
|---|---|
| `matmul_1bit.metal` | Metal shader for 1-bit matrix multiplication (the production kernel) |
| `metal_loader.py` | Python loader; re-exported by `engine/kernels/metal/loader.py` |
| `Makefile` | Placeholder: shaders are compiled at runtime by `metal_loader.py` |
| `__init__.py` | Package marker |

Needs `pyobjc-framework-Metal` and `pyobjc-framework-MetalPerformanceShaders` at runtime; no `setup.py` extra installs them.
