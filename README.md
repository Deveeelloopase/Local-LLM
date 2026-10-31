# Local LLM Cloud

A self-hosted LLM inference platform that detects the hardware it runs on (CPU-only up to multi-GPU, ~2 GB to 192 GB+ of VRAM) and adapts model loading, context length and scheduling to fit. It exposes an OpenAI-style HTTP API and ships a CLI, an SDK and a web chat UI.

This repository is a **monorepo of three Python packages** that are installed separately:

| Folder | Package | Role | Runs as |
|---|---|---|---|
| [`local-llm-core/`](local-llm-core/README.md) | `local-llm-core` (`local_llm_core`) | Inference engine library: hardware detection, model loading (llama.cpp / BitNet), memory tiers, scheduling, kernels | A library imported by the server (not a process of its own) |
| [`local-llm-server/`](local-llm-server/README.md) | `local-llm-server` (`local_llm_server`) | FastAPI HTTP server around the engine: auth, chat history, monitoring, TUI dashboard | Long-running service (`llm-server`, systemd unit `local-llm-cloud`) |
| [`local-llm-client/`](local-llm-client/README.md) | `local-llm-client` (`local_llm_client`) | CLI (`llm`), sync/async SDK, web UI HTML | Run on demand on any machine that can reach the server |
| [`logs/`](logs/README.md) | – | Runtime log output directory | – |

```
 you ── llm CLI / SDK / browser ──HTTP──▶  local-llm-server (port 8080)
        (local-llm-client)                   │  auth · sessions · rate limits · monitoring
                                             ▼
                                         local-llm-core
                                    hardware profile → model loader → scheduler → llama.cpp / BitNet
```

There is **no separate "core" network service**: `local-llm-core` is imported inside the server process. The server listens on **port 8080** by default.

## Requirements

- Linux (primary target; the code paths for macOS/Metal exist but are less exercised), Python ≥ 3.10.
- `systemd` for the optional service install.
- GPU backends are optional: AMD ROCm/HIP is the backend this project is developed against; CUDA, Vulkan and Metal code paths exist (see [`local-llm-core/local_llm_core/kernels/`](local-llm-core/local_llm_core/kernels/README.md)). CPU-only works.
- PostgreSQL is optional. The server can keep API keys and chat history in SQLite, PostgreSQL or MySQL.
- GGUF model files, placed in the directory named by `models.allowed_dir` in the server `config.yaml`.

## Quick start

```bash
git clone https://github.com/Deveeelloopase/Local-LLM.git
cd Local-LLM
python3 -m venv .venv && source .venv/bin/activate

# 1. engine library (editable) + one-time system setup
pip install -e local-llm-core
local-llm-setup --status            # check whether setup has been done
local-llm-setup                     # runs scripts/install.sh then scripts/bitnet_setup.sh

# 2. server (+ optional extras: postgres, mysql, tui)
pip install -e "local-llm-server[postgres,tui]"
sudo llm-server-install             # DB setup, systemd unit, writes credentials into config.yaml

# 3. client
pip install -e local-llm-client

# 4. configure + run
cp local-llm-server/config.yaml /path/you/own/config.yaml   # then edit: keys, model paths, DB
llm-server serve --config /path/you/own/config.yaml
```

Verify:

```bash
curl http://localhost:8080/health                  # liveness (public)
curl http://localhost:8080/ready                   # 200 only when a model is loaded (public)
curl http://localhost:8080/v1/models               # model list (public)
llm status ping                                    # via the CLI
llm chat "Hello" -m qwen3-8b
```

Interactive API docs: `http://localhost:8080/docs` (Swagger) and `/redoc`.

> `local-llm-core` and `local-llm-client` are installed **editable** so the server imports your working copy. A running service only re-imports at start: after changing code, reinstall if needed and `sudo systemctl restart local-llm-cloud`.

## API in one minute

OpenAI-compatible chat endpoint:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-API-Key: $LLM_API_KEY" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":64}'
```

- Auth header is **`X-API-Key`** (the key-management routes `/admin/keys/*` use **`X-Master-Key`**). `Authorization: Bearer` is not read by the server.
- Public (no key needed): `/health`, `/ready`, `/v1/models`, `/v1/models/current`, `/v1/runtime/limits`, plus the web UI pages `/` and `/demo`.
- Full endpoint list: [`local-llm-server/local_llm_server/serving/README.md`](local-llm-server/local_llm_server/serving/README.md).
- `max_tokens: 0` means "no limit".
- Request-level extras the server understands: `timeout_s`, `no_think`, `history_mode` (`auto` | `append` | `replace`). Conversation state is keyed by the `X-Session-ID` header or the `llm_session` cookie (a new id is created and set as a cookie if neither is sent).

## Configuration

| File | Used by | Notes |
|---|---|---|
| `local-llm-server/config.yaml` | the server (passed with `--config` or `LLM_CONFIG_PATH`) | The main file: API keys, DB, models, hardware overrides, scheduler, monitoring, logging |
| `local-llm-core/config.yaml` | the engine when used standalone | Same section names as the server file for the engine parts |
| `local-llm-client/config.yaml` | reference/defaults for the client | The CLI persists user settings to `~/.llmcloud/` (override with `LLM_CONFIG_DIR`) |

Credentials: use environment-variable expansion (`${VAR}`) rather than literals where the loader supports it, and keep real `config.yaml` files out of git.

## Repository map

```
Local-LLM/
├── local-llm-core/      engine library + install scripts
│   ├── local_llm_core/  hal/ memory/ engine/ kernels/ compatibility/
│   ├── scripts/         install.sh, bitnet_setup.sh, lib/, patches/
│   └── tests/
├── local-llm-server/    FastAPI server
│   ├── local_llm_server/ serving/ auth/ memory/ monitoring/ cli/
│   └── tests/
├── local-llm-client/    CLI + SDK + web UI
│   ├── local_llm_client/ cli/ core/ web/
│   └── tests/
└── logs/                runtime logs
```

Every folder has its own README describing its files.

## Known issues in this repository state

These are facts about what is currently committed, not roadmap items:

1. **Secrets in git.** `local-llm-server/config.yaml` contains an API key, a master key and database passwords in plain text. Treat them as compromised: rotate them, replace the file with a template, and remove it from history.
2. **The web UI page can reveal the API key.** `GET /` and `GET /demo` are public, and when a static `api.key` is configured the server substitutes it into the served HTML (`__API_KEY_PLACEHOLDER__`). Anyone who can reach the port gets the key. Use the key store (`api.key_store_enabled`) and bind to localhost or put the server behind an authenticating proxy.
3. **Hard-coded paths.** `config.yaml` files and the `.service` files contain absolute paths under `/home/deve-oops/...`. Edit them before use.
4. **Build artifacts and logs are committed:** `*.egg-info/`, a compiled `_fast.cpython-314-…so`, kernel-module build output under `local-llm-core/local_llm_core/kernels/*/` (`*.o`, `*.ko`, `*.cmd`) and `logs/llm_server.log`. They should be in `.gitignore`.
5. The unit files run the services as `root` and one still references a different GitHub URL in `Documentation=`.
6. The `[tool.hatch.build.hooks.custom]` comments in `local-llm-server/pyproject.toml` say `install.sh` runs on pip install; in fact the hook only **prints** the instruction to run `sudo llm-server-install`.

## Testing

```bash
pytest local-llm-core/tests            # set LLM_TEST_REAL_MODELS=1 to include model-loading tests
pytest local-llm-server/tests
pytest local-llm-client/tests          # integration tests skip if no server is reachable (LLM_TEST_SERVER_URL)
```

## Conventions for contributors

- Version string is `1.0.0` in all three packages (`__version__` and packaging metadata) — keep them aligned.
- No absolute paths in code; resolve from config or `Path(__file__)`.
- Errors returned to API clients are structured and human-readable, never a raw traceback and never a fabricated success.
- Fix root causes; do not paper over hardware-specific symptoms.
- 