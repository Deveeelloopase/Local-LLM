# local_llm_client/core/

The SDK. Everything here is importable without the CLI.

| File | Purpose |
|---|---|
| `client.py` | `LLMCloudClient` — synchronous client with retry and a circuit breaker |
| `async_client.py` | `AsyncLLMCloudClient` — asyncio client; uses `aiohttp` when installed, with a fallback path otherwise. Same method names as the sync client |
| `models.py` | Data classes: `ClientConfig` (all settings), `GenerationResult`, `ServerStatus`, `ModelInfo`, `ModelList`, `ChatMessage`, `CompletionRequest`, `CompletionResponse`, `TokenMetricsCompat`, `ModelStatus`, `ModelType`, `FinishReason`, `ConfigMerger` |
| `exceptions.py` | `LLMCloudError` and ~20 subclasses: `ConnectionError`, `TimeoutError`, `AuthError`, `InvalidAPIKeyError`, `MasterKeyRequiredError`, `ModelNotLoadedError`, `ModelBusyError`, `RateLimitError`, `QueueFullError`, `ServerError`, … |
| `pool.py` | `ConnectionPool`, `AsyncConnectionPool` |
| `adapter.py` | `LocalLLMCloudAdapter`, `AdaptiveContextManager`, `TokenMetrics`, `get_model_manager()` — an adapter shaped for an agent framework that expects a `ModelConfig`-like object (mirrors the `ModelConfig` of AMB-APP's `ai_agent.py`); not used by the CLI |
| `_accel.py` | Optional C accelerator via `cffi` (token estimation, word-boundary search, chunk buffers) with a pure-Python fallback |
| `examples.py` | Runnable examples (`example_sync`, `example_stream`, `example_async`, `example_batch`, `example_agent`, key and model management) |

## Method map (both clients)

| Group | Methods |
|---|---|
| Generation | `generate`, `chat_completions`, `chat_stream`, `stream`, `resume` (sync client also `resume_stream`) |
| Server info | `health`, `is_healthy`, `ready`, `models`, `current_model`, `queue_status`, `system_info`, `metrics`, `get_logs` |
| Models | `model_ready`, `load_model`, `unload_model`, `hot_swap`, `load_profile`, `clear_cache` |
| Keys (master key) | `create_key`, `revoke_key`, `activate_key`, `delete_key`, `list_keys`, `key_stats` |
| Misc | `set_api_key`, `set_master_key`, `is_authenticated`, `count_tokens`, `close` |

## Behaviour worth knowing

- `max_tokens=0` means "no limit"; the value is sent as is.
- Generation calls go to the server's `/v1/chat/completions` family; request fields such as `no_think`, `timeout_s` and `history_mode` are documented in the server's [`serving/README.md`](../../../local-llm-server/local_llm_server/serving/README.md).
- The clients retry transient failures and open a circuit breaker after repeated failures (`circuit_breaker_*` fields of `ClientConfig`).
- Timeouts, pool sizes and defaults come from `ClientConfig`; the package-root `config.yaml` shows the YAML form.
- 