# engine/inference/

Generation runtime.

| File | Key classes / functions | Purpose |
|---|---|---|
| `inference_engine.py` | `InferenceEngine`, `InferenceRequest`, `InferenceResponse`, `GenerationCheckpoint`, `RequestQueue`, `AsyncTokenBuffer`, `CircuitBreaker`, `RateLimiter`, `TokenBucket`, `AutoTuner`, `NGramCache`, `PrefillTracker`, `BoundedThreadPoolExecutor`, `StreamStallError` | Runs requests against loaded models: `generate()` (non-streaming) and token streaming, deadlines, circuit breaker on repeated failures, rate limiting, prefill/decode speed tracking (EWMA), stall detection, checkpoints for resumable generation |
| `stream_guards.py` | `resolve_max_tokens()` / `UNLIMITED_TOKENS`, `kv_bytes_per_token()`, `kv_budget_mb()`, `plan_context_length()`, `ThinkStripper`, `strip_think_text()`, `RepetitionGuard`, `slide_window()` | Pure helper functions shared by engine and server: one definition of "no token limit", memory-aware context planning, `<think>` tag removal (including orphan tags), repetition detection |
| `throughput_optimizer.py` | `ThroughputOptimizer`, `ContinuousBatchScheduler`, `SlotRequest` | Continuous batching over a fixed number of slots (`throughput_optimizer:` config) |
| `tensor_parallel.py` | `compute_tensor_split()`, `should_split()`, `TensorSplitPlan`, `SplitStrategy` | Plans how to split a model across several GPUs (`multi_gpu:` config) |
| `async_weight_pipe.py` | `AsyncWeightPipeline`, `ModelWeightPipelineManager`, `LayerRegion`, `estimate_layer_regions()` | Overlaps weight loading with compute for models that do not fit entirely in VRAM |
| `__init__.py` | – | Lazy re-exports |

## Behaviour to know

- **Deadlines.** A request's `timeout_s` overrides `inference.default_timeout_s`. A value `<= 0` for both means "no deadline" on the non-streaming path; when a deadline is hit the response has `finish_reason == "timeout"`.
- **Token limits.** `max_tokens <= 0` means unlimited (`UNLIMITED_TOKENS` in `stream_guards.py`).
- **Thinking.** `inference.suppress_thinking` and the per-request `no_think` control reasoning blocks; `ThinkStripper` removes them from output.

## Config

`inference:` (`batch_size`, `default_temperature`, `default_top_p`, `default_top_k`, `default_max_tokens`, `default_timeout_s`, `default_stop`, `suppress_thinking`, `reasoning_budget_tokens`, `thinking_indicator_enabled`), `throughput_optimizer:`, `multi_gpu:`, `circuit_breaker:`.
