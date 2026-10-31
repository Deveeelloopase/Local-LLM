# engine/pipelines/

Fitting models that are bigger than the available GPU memory, and keeping disk/CPU/memory work balanced.

| File | Key names | Purpose |
|---|---|---|
| `layer_streamer.py` | `read_gguf_meta()`, `calc_optimal_gpu_layers()`, `compute_optimal_offload_plan()`, `FLPConfig`, `BackendProbe`, `AdaptiveLoader`, `torch_available()` | Reads GGUF metadata, computes how many layers go to the GPU, and loads adaptively: on out-of-memory it retries with fewer layers and a smaller context (`layer_streamer:` config) |
| `drive_pipeline.py` | `NVMeModelAdvisor`, `advise_model_file()`, `ContextCache`, `LayerPrefetcher`, `CPUThrottler`, `ConcurrencyTokenManager`, `ThroughputController`, `MemoryPressureMonitor`, `NVMeConfig` | Storage-aware pipeline: advice about where a model file should live, context cache, layer prefetch, CPU throttling, concurrency tokens and a throughput controller (`drive_pipeline:` config) |
| `__init__.py` | – | Lazy re-exports |

## Config

`layer_streamer:` — `vram_overhead_mb`, `oom_max_attempts`, `oom_layer_reduction_factor`, `ctx_reduction_steps`, `min_context`, `layer_overcommit_factor`.
`drive_pipeline:` — `target_vram_pct`, `target_ram_pct`, `target_tps`, `cpu_thread_ratio` and nested `context_cache`, `layer_prefetcher`, `cpu_throttler`, `concurrency`, `throughput_controller`, `nvme_advisor`.

The `vram_overhead_mb` value matters on small GPUs/APUs: a constant tuned for large cards over-reserves memory on a 2 GB device.
