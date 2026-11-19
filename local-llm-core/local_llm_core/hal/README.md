# local_llm_core/hal/ — hardware abstraction layer

Detects what the machine is and applies host-level tuning. Nothing here loads models.

| File | Key classes / functions | Purpose |
|---|---|---|
| `hardware_detector.py` | `HardwareDetector`, `GPUInfo`, `SystemInfo`, `GPUVendor`, `MemoryType` | Detects CPU, RAM, GPUs (vendor, VRAM, unified vs dedicated memory) with configurable command timeouts and paths. Has a `main()` for manual runs |
| `gpu_analyzer.py` | `GPUAnalyzer`, `ComputeProfile`, `ThermalStatus` | Estimates GPU capability (compute throughput, memory/PCIe bandwidth) from known specs and sysfs probing; reports thermal state |
| `system_optimizer.py` | `SystemOptimizer` | Optional OS tuning for throughput: huge pages / THP, CPU governor, I/O scheduler, sysctls. Controlled by the `system:` config section (`allow_kernel_tuning` etc.) |
| `suspend_inhibitor.py` | suspend inhibitor | Takes a systemd-logind inhibitor lock so the host does not suspend while serving; no-op where unavailable |
| `config_loader.py` | `HardwareConfig`, `GpuAnalyzerConfig`, `SystemConfig`, `HugePagesConfig`, `VMConfig` | Typed config for the `hardware`, `gpu_analyzer` and `system` sections |
| `__init__.py` | – | Lazy re-exports |

## Config

`hardware` (`auto_detect`, `preferred_backend`, `timeout_seconds`, `vulkan_icd_paths`), `gpu_analyzer` (default FP32 TFLOPS, memory/PCIe bandwidth fallbacks, `thermal_throttle_temp_c`, `model_overrides`), `system` (kernel tuning switches).

## Related

The *profile* chosen from this detection (e.g. `apu_2gb`, `gpu_24gb`) is computed in [`engine/hardware/`](../engine/hardware/README.md).
