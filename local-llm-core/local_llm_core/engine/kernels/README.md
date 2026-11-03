# engine/kernels/

Runtime selection and activation of custom compute kernels. The kernel **source code** is in [`../../kernels/`](../../kernels/README.md).

| File | Purpose |
|---|---|
| `manager.py` | `UniversalKernelManager`: with `kernels.auto_select` it tries backends in the order `cuda → hip → metal → vulkan`, activates the first one whose hardware is present and whose loader succeeds; `kernels.force_backend` overrides; `kernels.cpu_optimizer` is forwarded to the CPU tuning module. Disabled by `kernels.enabled: false` |
| `patcher.py` | `install_patch()` / `remove_patch()`: replace `torch.matmul` and `F.linear` with conditional wrappers, and restore the original PyTorch functions |
| `cuda_kernels.py` | Lazy forwarding module: `load_cuda_kernel`, `ternary_matmul`, `CudaDmaPool` from `local_llm_core.kernels.cuda_optimizer` |
| [`metal/`](metal/README.md) | Metal (macOS) loader entry point |
| `__init__.py` | Re-exports `UniversalKernelManager` |

Config (`kernels:`): `enabled`, `auto_select`, `force_backend`, `cpu_optimizer`.
