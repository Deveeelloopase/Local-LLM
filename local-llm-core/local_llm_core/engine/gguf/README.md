# engine/gguf/

Standard GGUF models through llama.cpp (`llama-cpp-python`).

| File | Key classes | Purpose |
|---|---|---|
| `model_loader.py` | `ModelLoader`, `ModelConfig`, `LoadedModel` | Full model lifecycle: load / unload / hot swap, hardware-adaptive parameters (GPU layers, context, KV cache types), a loader circuit breaker, covers both GGUF and BitNet models in `LoadedModel` |
| `speculative_engine.py` | `SpeculativeEngine`, `SpecStats` | Speculative decoding with a draft model (`speculative:` config: `draft_model_id`, `k`, `adaptive_k`, …) |
| `__init__.py` | – | Lazy re-exports |

Config: `models:` (`allowed_dir`, `profiles`), `model_loader:` (`load_timeout_s`, `shutdown_timeout_s`, `min_vram_budget_mb`, `circuit_breaker`).
