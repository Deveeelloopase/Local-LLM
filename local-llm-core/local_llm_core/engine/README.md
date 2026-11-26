# local_llm_core/engine/

The core of the library: model loading, generation, scheduling and backend selection. Each subfolder has its own README.

| Folder | Responsibility |
|---|---|
| [`adaptive_awareness/`](adaptive_awareness/README.md) | Classifies a GGUF file (standard vs BitNet), validates the environment, routes to the right backend |
| [`bitnet/`](bitnet/README.md) | BitNet b1.58 support: BitNet.cpp backends, loader, memory manager, speculative decoding |
| [`exceptions/`](exceptions/README.md) | Exception hierarchy with HTTP-status mapping |
| [`gguf/`](gguf/README.md) | `ModelLoader` for GGUF models (llama.cpp) and the speculative-decoding engine |
| [`hardware/`](hardware/README.md) | Hardware **profiles** (`apu_2gb` … `gpu_192gb+`, `cpu_only`) |
| [`inference/`](inference/README.md) | `InferenceEngine`, stream guards, throughput optimizer, tensor-parallel planning, async weight pipeline |
| [`kernels/`](kernels/README.md) | `UniversalKernelManager` (backend selection for compute kernels) and the PyTorch patcher |
| [`models/`](models/README.md) | Loaded-model lifecycle (idle eviction) and the NVMe KV cache |
| [`pipelines/`](pipelines/README.md) | Layer streaming, GPU/CPU offload planning, drive/NVMe pipeline |
| [`scheduling/`](scheduling/README.md) | Priority request scheduler with AIMD control |
| [`utils/`](utils/README.md) | Typed configuration models and the config loader |

`engine/__init__.py` provides the lazy re-exports that `local_llm_core/__init__.py` forwards.

## Typical flow

1. `utils.config_loader` parses `config.yaml` into typed models.
2. `hardware.detect_profile()` (using `hal`) picks a profile and applies overrides.
3. `adaptive_awareness` classifies each configured model file and picks GGUF or BitNet.
4. `gguf.ModelLoader` / `bitnet.BitNetLoader` load it, using `pipelines` + `memory` to fit it in the available memory.
5. `scheduling.Scheduler` orders requests; `inference.InferenceEngine` runs them (timeouts, circuit breaker, rate limiter, stream guards).
