# local_llm_server/serving/

The HTTP layer. Almost everything lives in one large module, `api_server.py` (~5,100 lines): configuration loading, the FastAPI app factory `create_app(config_path)`, all routes, authentication, history merging, streaming and the process entry `main(argv)`.

| File | Purpose |
|---|---|
| `api_server.py` | App factory, routes, auth dependencies, request/response models used by the routes, streaming (SSE) with segment-based long generation, `main()` which validates `--workers`, runs uvicorn (import-string + factory mode for `--reload`) and writes `/tmp/llm_server.pid` |
| `middleware.py` | `ProductionMiddleware` (ASGI), `IPRateLimiter` with trusted-proxy handling, request-id logging filter |
| `log_handler.py` | `TUILogHandler`: in-memory ring buffer feeding `GET /v1/logs` and the `/v1/ws/logs` WebSocket, with de-duplication and overflow file |
| `models.py` | Pydantic schemas for the OpenAI-style API (`ChatRequest`, `ChatResponse`, `ModelList`, …). Exported through `__init__`; `api_server.py` defines its own request models for the routes, so treat `models.py` as a shared schema reference |
| `diagnose_web.py` | Stand-alone diagnostic tool: `python -m local_llm_server.serving.diagnose_web --help` (endpoint tests, DNS/TLS checks, JSON output, exit codes 0 = OK / 1 = warnings / 2 = critical) |
| `__init__.py` | Lazy re-exports of the modules in this folder |

## Authentication

| Scope | Header | Routes |
|---|---|---|
| Public | – | `/`, `/demo`, `/health`, `/ready`, `/v1/models`, `/v1/models/current`, `/v1/runtime/limits` |
| API key | `X-API-Key` | everything else under `/v1/*` |
| Master key | `X-Master-Key` | `/admin/keys/*` |

`Authorization: Bearer …` is not read. The log WebSocket `/v1/ws/logs` uses the same key rules as the HTTP routes. With neither a static key nor a key store configured the server accepts all requests.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | Liveness; readiness (503 until a model is loaded) |
| GET | `/`, `/demo` | Web chat page (from `local-llm-client`) |
| GET | `/v1/me` | Info about the calling key |
| GET | `/v1/system`, `/v1/hardware/backends`, `/v1/gpu/analyze`, `/v1/backend` | Hardware and backend information |
| GET | `/v1/runtime/limits` | Effective context length, measured tokens/s, tier, parallelism — use this instead of guessing limits |
| GET | `/v1/models`, `/v1/models/current`, `/v1/model-status`, `/v1/config/models` | Loaded models, current model, status, configured profiles |
| POST | `/v1/models/load`, `/v1/models/load-profile/{profile_id}` | Load a model / profile |
| GET | `/v1/models/loading/{model_id}` | Load progress |
| POST | `/v1/models/{model_id}/swap` | Hot swap |
| DELETE | `/v1/models/{model_id}` | Unload |
| POST | `/v1/chat/completions` | **Main chat endpoint** (OpenAI-compatible; streaming with `"stream": true`) |
| POST | `/v1/chat/stream` | Streaming chat variant |
| POST | `/v1/chat/smart` | "Smart" chat (`SmartChatRequest`): chooses among the models that are ready |
| POST | `/v1/chat/resume` | Continue an interrupted generation from a checkpoint saved in the conversation store |
| POST | `/v1/completions`, `/v1/embeddings` | Text completion, embeddings |
| GET/POST | `/v1/bitnet/status`, `/v1/bitnet/info`, `/v1/bitnet/backend`, `/v1/bitnet/generate` | BitNet backend |
| GET | `/v1/queue/status`, `/v1/metrics`, `/v1/performance`, `/v1/telemetry`, `/v1/throughput/stats`, `/v1/middleware/stats`, `/v1/cluster` (static single-node answer: `{"nodes": 1, "role": "standalone"}`), `/v1/drive/stats` | Operational data (JSON; Prometheus export is a separate listener, see `monitoring` config) |
| DELETE | `/v1/context-cache/clear` | Clear cached contexts |
| GET / WS | `/v1/logs`, `/v1/ws/logs` | Log tail, log stream. The WebSocket takes the key in the `X-API-Key` header and validates it against the static key **and** the key store |
| POST/GET/PATCH/DELETE | `/admin/keys/create`, `/list`, `/stats/{id}`, `/update/{id}`, `/revoke/{id}`, `/activate/{id}`, `/delete/{id}` | Key administration |

Docs: `/docs` (Swagger UI), `/redoc`.

### `/v1/chat/completions` request

Standard OpenAI fields (`model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`) plus:

| Field | Meaning |
|---|---|
| `max_tokens` | `0` = no limit |
| `timeout_s` | Total deadline for non-streaming generation, seconds. `0`/absent = server default (`inference.default_timeout_s`; `0` there means no deadline) |
| `no_think` | Disable the model's reasoning ("think") block for models that support it |
| `history_mode` | `auto` (default) / `append` / `replace` — see [`memory/`](../memory/README.md) |
| Session | `X-Session-ID` header or `llm_session` cookie selects the stored conversation |

Messages are `{role, content}` with string content; `tools` / `tool_calls` are **not** supported (extra fields are ignored). An engine error or deadline is returned as an error status, not as completion text.

### Streaming

`"stream": true` returns Server-Sent Events in the OpenAI delta format. The server recognises the stream markers `[⚡TRUNCATED…]`, `[⚡INTERRUPTED…]` and `[⚡STALLED…]`; when one is present (or the client disconnects mid-stream) it stores a checkpoint of the text generated so far, which `/v1/chat/resume` can continue.

## Configuration used by this package

`server.*` (host, port, workers, TLS, CORS, `trusted_proxies`, size limits), `api.*` (keys, rate limits), `logging.*` (buffer and WebSocket settings), `inference.*`, `models.*`. See the [server README](../../README.md#configuration).

## Security notes

`GET /` and `GET /demo` substitute the configured static `api.key` into the page (`__API_KEY_PLACEHOLDER__`). See [Security notes](../../README.md#security-notes).

## Process model

The server is a **single process**: the model, the scheduler and the key/conversation caches live in it. `--workers` greater than 1 is rejected at start-up (each worker would load its own copy of the model into VRAM). To serve more load, run one server per machine and distribute traffic with a reverse proxy — see [DEPLOYMENT.md](../../DEPLOYMENT.md#scaling-out).

`--reload` (development only) starts uvicorn with the import string `local_llm_server.serving.api_server:create_app` in factory mode. The config path reaches the factory through `LLM_CONFIG_PATH`, which `create_app()` reads when called without an argument. Every reload restarts the process and reloads the model.
