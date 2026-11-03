# local_llm_server/ (Python package)

| Path | Purpose |
|---|---|
| `__main__.py` | Command-line entry (`python -m local_llm_server`, `llm-server`). Parses flags, locates/validates the config, then hands over to `serving.api_server.main()` |
| `__init__.py` | Package metadata (`__version__ = "1.0.0"`) |
| [`serving/`](serving/README.md) | The FastAPI application and everything it serves |
| [`auth/`](auth/README.md) | API-key store (SQLite / PostgreSQL / MySQL) |
| [`memory/`](memory/README.md) | Conversation (chat history) store |
| [`monitoring/`](monitoring/README.md) | Metrics, analysis and the terminal dashboard (`llm-monitor`) |
| [`cli/`](cli/README.md) | `llm-server-install` |

## Startup flow

```
llm-server (__main__.py)
  └─ validate config.yaml, then call serving.api_server.main(argv)
      └─ main(): create_app(config)  (${ENV} expansion happens here) and run it with uvicorn
           ├─ build engine from local-llm-core (hardware detection → model loader → scheduler)
           ├─ auth.key_store            (if api.key_store_enabled)
           ├─ memory.conversation_store (chat history)
           ├─ middleware (request id, IP rate limit, CORS)
           └─ routes: /v1/chat/completions, /v1/models, /health, /ready, …
```

The server imports `local_llm_core` for all inference work. Nothing in this package talks to a model directly.
