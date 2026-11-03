# local_llm_core/kernels/

**Source code** for optional low-level acceleration. Two different kinds of things live here:

| Kind | Folders | What they are |
|---|---|---|
| Linux kernel modules (C) | [`amd_kali_compat/`](amd_kali_compat/README.md), [`gpu_scheduler/`](gpu_scheduler/README.md), [`memory_optimizer/`](memory_optimizer/README.md) | Built with `make` against your kernel headers and loaded with `insmod` by `scripts/install.sh`. Linux only, root required |
| Compute kernels with a Python loader | [`cuda_optimizer/`](cuda_optimizer/README.md), [`hip_optimizer/`](hip_optimizer/README.md), [`vulkan_optimizer/`](vulkan_optimizer/README.md), [`metal_kernels/`](metal_kernels/README.md), [`cpu_optimizer/`](cpu_optimizer/README.md) | 1-bit/ternary matmul kernels per GPU API, plus CPU core isolation. Selected at runtime by [`engine/kernels/manager.py`](../engine/kernels/README.md) |

All of this is **optional**. The engine runs without any of it (llama.cpp does its own GPU work); the kernels matter mainly for BitNet-style 1-bit models and for Linux hosts where the kernel modules were installed.

## How it is wired

- `UniversalKernelManager` (`engine/kernels/manager.py`) imports the Python loaders (`cuda_kernels.py`, `hip_kernels.py`, `vulkan_loader.py`, `metal_loader.py`) and activates the first backend that works.
- The three kernel-module folders are **not imported by Python code**; they are referenced by `scripts/install.sh` (build + `insmod`) and by hardware detection that looks for the loaded modules.
- Configuration: the `kernels:` section (`enabled`, `auto_select`, `force_backend`, `cpu_optimizer`).

## Building a kernel module by hand

```bash
cd local_llm_core/kernels/gpu_scheduler
make                      # needs /lib/modules/$(uname -r)/build (kernel headers) and a compiler
sudo insmod gpu_scheduler.ko
sudo rmmod gpu_scheduler  # to unload
```

## Known issue: build output is committed

The folders `amd_kali_compat/`, `gpu_scheduler/` and `memory_optimizer/` contain compiled `.o`, `.ko`, `.mod`, `.mod.c`, `.cmd`, `modules.order` and `Module.symvers` files from one machine and one kernel version. They do not work elsewhere and should be removed from git (`make clean`) and ignored.
