# local_llm_core/ (Python package)

| Path | Purpose |
|---|---|
| `__init__.py` | Lazy export table (`_LAZY_IMPORTS`, ~80 public names). `__version__ = "1.0.0"`. Accessing `local_llm_core._fast` imports the C extension if it was built |
| `_setup_cli.py` | `local-llm-setup` command |
| `_fast.c`, `_fast.*.so` | Optional CPU accelerator: `dot_product_f32`, `dot_product_f16`, `sum_f32`, ternary `matmul_1bit` (AVX2/AVX-512 with scalar fallback). Build is driven by `setup.py` |
| `Makefile` | Developer shortcuts (`make lint`, `test`, …); see the stale-path note in the [core README](../README.md#known-issues) |
| [`hal/`](hal/README.md) | Hardware abstraction: detection, GPU analysis, OS tuning |
| [`memory/`](memory/README.md) | VRAM / RAM / swap / NVMe memory tiers |
| [`engine/`](engine/README.md) | Model loading, inference, scheduling, BitNet, kernels |
| [`kernels/`](kernels/README.md) | **Source code** of optional low-level kernels (Linux kernel modules, CUDA/HIP/Vulkan/Metal compute) — not importable as a Python package |
| [`compatibility/`](compatibility/README.md) | `transformers` / BitLinear compatibility layer |

Dependency direction: `engine` → `memory`, `hal`, `kernels` (through `engine/kernels`), `compatibility`; `hal` and `memory` do not import `engine`.