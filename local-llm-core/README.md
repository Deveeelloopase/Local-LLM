# local-llm-core

The inference engine library of Local LLM Cloud. It detects the machine (CPU, GPU vendor/VRAM, APU, memory, thermal state), chooses a hardware profile, loads GGUF and BitNet models with settings that fit that profile, schedules requests, and generates tokens. It has **no HTTP server of its own**; [`local-llm-server`](../local-llm-server/README.md) imports it.

- Package: `local-llm-core` · import name `local_llm_core` · version `1.0.0` · Python ≥ 3.10 · Linux (primary) and macOS
- License: MIT (per `setup.py` classifiers)
- Console script: `local-llm-setup` → `local_llm_core._setup_cli:main` (one-time system setup)
- Optional C extension `_fast` (SIMD dot product / ternary matmul) built by `setup.py`; the library works without it

## Folder contents

| Path | Purpose |
|---|---|
| [`local_llm_core/`](local_llm_core/README.md) | The Python package |
| [`scripts/`](scripts/README.md) | `install.sh` (system setup) and `bitnet_setup.sh` (BitNet.cpp build/model setup) with helper libraries and patches |
| [`tests/`](tests/README.md) | `pytest` suite |
| `config.yaml` | Engine configuration reference (all sections the engine reads) |
| `requirements.txt`, `setup.py`, `pyproject.toml`, `MANIFEST.in` | Packaging |
| `local_llm_core.egg-info/` | Build output committed by accident — safe to delete, should be git-ignored |

## Install

```bash
pip install -e .                 # editable install; builds the optional _fast extension if a compiler is present
local-llm-setup --status         # has the one-time setup run?
local-llm-setup                  # runs scripts/install.sh then scripts/bitnet_setup.sh
```

`local-llm-setup` options: `--status`, `--reset` (delete the "done" marker and redo everything), `--skip-bitnet` (install.sh only), plus pass-through flags for `bitnet_setup.sh` (`--non-interactive/-y`, `--dry-run`, `--use-pypi-gguf`, `--apply-patch`, `--minimal-build`, `--quant-embd`, `--skip-clone`, `--skip-build`, `--skip-model`, …; see [`scripts/README.md`](scripts/README.md)).

Extras defined in `setup.py`: `dev` (test/lint tools), `gpu` (`cuda-python`), `vulkan`, `onnx` (`onnxruntime`, `onnx`), `all` (gpu + vulkan + onnx) and `full` (all + dev).

## Using it as a library

Public names are loaded lazily from the package root (about 80 of them), so `import local_llm_core` is cheap:

```python
from local_llm_core import (
    detect_profile, HardwareDetector, ModelLoader, InferenceEngine,
    InferenceRequest, Scheduler, load_config,
)
```

Grouped exports: engine (`InferenceEngine`, `ModelLoader`, `ModelConfig`, `LoadedModel`, `SpeculativeEngine`, `Scheduler`, `Priority`), BitNet (`BitNetLoader`, `BitNetInferenceWrapper`, `BitNetCppBackend`, `create_best_backend`, …), routing (`classify_gguf`, `route_model`, `auto_detect_backend`, `validate_environment`), hardware (`HardwareDetector`, `GPUAnalyzer`, `SystemOptimizer`, `HardwareProfile`, `detect_profile`), memory (`VRAMManager`, `RAMOptimizer`, `SwapHandler`, `HybridMemoryManager`, `HybridCache`), kernels (`UniversalKernelManager`), errors (`LocalLLMError` and subclasses, `exception_to_http_response`), config classes (`ServerConfig`, `APIConfig`, `HardwareConfig`, `InferenceConfig`, `ModelsConfig`, `LoggingConfig`).

## How a request flows

```
detect hardware (hal) ──▶ pick profile (engine/hardware) ──▶ plan context + layers (engine/inference/stream_guards, engine/pipelines)
        │                                                           │
        ▼                                                           ▼
 memory tiers (memory/)                      ModelLoader (engine/gguf)  or  BitNetLoader (engine/bitnet)
                                                           │
 request ─▶ Scheduler (engine/scheduling) ─▶ InferenceEngine (engine/inference) ─▶ tokens
```

Which loader runs is decided by `engine/adaptive_awareness` (it classifies a `.gguf` file — standard GGUF vs BitNet i2_s/TL1/TL2 — and checks the environment).

## Configuration

`config.yaml` sections read by the engine:

`kernels`, `hardware`, `hardware_profiles`, `gpu_analyzer`, `memory`, `system`, `inference`, `throughput_optimizer`, `nvme_kv_cache`, `layer_streamer`, `drive_pipeline`, `scheduler`, `bitnet`, `speculative`, `multi_gpu`, `circuit_breaker`, `models`, `model_loader`, `model_lifecycle`.

The server's `config.yaml` uses the same section names, so the server file is the one you edit in practice. Notable settings:

- `inference.default_timeout_s` — engine request deadline; **`0` means no deadline**.
- `inference.default_max_tokens` — default generation limit; `max_tokens: 0` in a request means unlimited.
- `hardware_profiles.force_profile` and `.overrides` — pin or tune a profile (context length, target tokens/s, KV-cache types).
- `models.allowed_dir` and `models.profiles` — where GGUF files live and which models exist.

Hardware profile keys: `apu_2gb`, `apu_4gb`, `gpu_4gb`, `gpu_6gb`, `gpu_8gb`, `gpu_12gb`, `gpu_16gb`, `gpu_24gb`, `gpu_32gb`, `gpu_48gb`, `gpu_80gb`, `gpu_192gb+`, `cpu_only`.

## Testing

```bash
pytest tests -v                        # unit + integration without model files
LLM_TEST_REAL_MODELS=1 pytest tests   # include tests that load real GGUF files
LLM_TEST_STRESS=1 pytest tests        # include the concurrency stress test
```

## Known issues

- `local_llm_core/Makefile` is a developer Makefile whose `SRC_DIRS` (`serving engine memory monitoring hal tui`) refers to directories that do not exist in this package (`serving`, `monitoring`, `tui` live in `local-llm-server`).
- A compiled `_fast.cpython-314-x86_64-linux-gnu.so`, `*.egg-info` and kernel-module build output are committed; they are build artifacts for one machine and should be untracked.
- `config.yaml` examples contain absolute paths from the author's machine.
- 