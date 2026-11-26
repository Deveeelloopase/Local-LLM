# local_llm_client/ (Python package)

Import name of the `local-llm-client` distribution. Version `__version__ = "1.0.0"`.

| Path | Purpose |
|---|---|
| `__init__.py` | Public exports: `LLMCloudClient`, `AsyncLLMCloudClient`, `ClientConfig`, `GenerationResult`, `ServerStatus`, `TokenMetrics`, `ModelInfo`, `ModelList`, the exception hierarchy, `LocalLLMCloudAdapter`, `get_model_manager`, `AdaptiveContextManager`, `HTML`, `get_web_ui` |
| `__main__.py` | `python -m local_llm_client` entry: Python ≥ 3.10 guard, several import strategies, exit-code propagation; set `LLM_DEBUG=1` for import traces |
| [`cli/`](cli/README.md) | The `llm` command line |
| [`core/`](core/README.md) | The SDK (HTTP clients, data models, errors, pools, agent adapter) |
| [`web/`](web/README.md) | Embedded web chat UI |

## How the pieces connect

```
llm (cli/main.py) ─▶ cli/commands/* ─▶ core.client / core.async_client ─▶ HTTP ─▶ local-llm-server
                          │                          ▲
                          └─ cli/config.py ──────────┘  (base_url, api_key, defaults)
web/web_ui.py ── imported by local-llm-server and served at "/" and "/demo"
```

## Where to start reading

1. `core/models.py` → `ClientConfig` (all client settings).
2. `core/async_client.py` or `core/client.py` (identical method sets).
3. `cli/main.py` → `register_commands()` → `cli/commands/*.py`.
4. 