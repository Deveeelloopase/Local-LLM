# engine/bitnet/

Support for BitNet b1.58 (ternary-weight) models.

| File | Key classes | Purpose |
|---|---|---|
| `bitnet_cpp_backend.py` | `BitNetCppBackend`, `BitNetCppChatBackend`, `BitNetCliBackend`, `BitNetRunInferenceBackend`, `BitNetCppConfig`, `SubprocessEnvironmentBuilder`, `CircuitBreaker`, `create_best_backend()`, `is_bitnet_cpp_available()`, `get_recommended_thread_count()` | Runs BitNet.cpp (`llama-server`/`llama-cli`/`run_inference.py`) as a subprocess; `create_best_backend()` tries the available backends in tier order |
| `bitnet_loader.py` | `BitNetLoader`, `BitNetModelConfig`, `PRISMModelClassifier`, `HardwareDetector`, `auto_detect_backend()` | Loads a BitNet model through the best backend (BitNet.cpp or the `transformers` path) with retry |
| `bitnet_engine.py` | `BitNetInferenceWrapper`, `BitNetInferenceConfig`, `BitNetBackendType`, `GenerationMetrics`, `StopStringCriteria`, `get_router()` | Generation wrapper that gives BitNet the same interface as the GGUF engine |
| `bitnet_memory_manager.py` | `BitNetMemoryManager`, `BitNetMemoryCalculator`, `BitNetMemoryProfile`, `MemoryPressureLevel` | Memory estimates and pressure handling for BitNet models |
| `speculative_bitnet.py` | `BitNetSpeculativeEngine`, `BitNetSpecStats` | Speculative decoding for BitNet |
| `bitnet_kernels.py` | alias | Backward-compatibility alias; kernel selection moved to `engine/kernels/manager.py` (`UniversalKernelManager`) |
| `__init__.py` | `initialize()`, `get_status()` | Lazy re-exports plus package init/status helpers |

## Setup dependency

BitNet.cpp itself is not part of this repository. `scripts/bitnet_setup.sh` clones and builds it (default location `…/BitNet/`) and records the result in `bitnet_state.json`.

## Config (`bitnet:` section)

`enabled`, `default_quantization`, `use_custom_kernels`, `auto_detect_kernel_backend`, `force_kernel_backend`, `memory_fraction`, `enable_speculative_decoding`, `enable_context_caching`, `max_loaded_models`, `load_retries`; plus `speculative:` for draft-model settings.
