# kernels/vulkan_optimizer/

Vulkan compute backend (portable fallback).

| File | Purpose |
|---|---|
| `matmul_1bit.comp` | GLSL compute shader for 1-bit matrix multiplication |
| `vulkan_loader.py` | Python loader used by `UniversalKernelManager` |
| `Makefile` | Compiles the shader with `glslc` to `matmul_1bit.spv` (the loader also looks for `glslc` itself) |
| `__init__.py` | Package marker |

Last in the auto-select order. Needs a Vulkan driver (`vulkaninfo` working). The `vulkan` extra installs no Python packages.
