# local-llm-server

FastAPI HTTP server that wraps the [`local-llm-core`](../local-llm-core/README.md) engine. It provides an OpenAI-style chat API, API-key authentication, per-session chat history, rate limiting, live logs, a terminal dashboard and the web chat page.

- Package: `local-llm-server` · import name `local_llm_server` · version `1.0.0` · Python ≥ 3.10
- Build backend: hatchling (`hatch_build.py` is a build hook that prints next-step instructions — it does not install anything)
- Default listen address: `0.0.0.0:8080` (`server.host` / `server.port` in `config.yaml`)

## Folder contents

| Path | Purpose |
|---|---|
| [`local_llm_server/`](local_llm_server/README.md) | The Python package (entry point, auth, history, monitoring, API) |
| [`tests/`](tests/README.md) | Test suite |
| `config.yaml` | The server configuration (see [Configuration](#configuration)) |
| `pyproject.toml` | Packaging, extras, console scripts |
| `requirements.txt` | Pinned runtime requirements for manual installs |
| `install.sh` | Interactive installer behind `llm-server-install` |
| `local-llm-cloud.service` | systemd unit |
| `hatch_build.py` | Build hook (prints "run `sudo llm-server-install`") |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Step-by-step deployment guide |

## Console scripts

| Command | Entry point | Purpose |
|---|---|---|
| `llm-server` | `local_llm_server.__main__:main` | Run the server (`llm-server serve`, the only subcommand and the default) |
| `llm-server-install` | `local_llm_server.cli.server_install_cmd:main` | Production install: database, systemd unit, credentials — run with `sudo`. Flags: `--non-interactive`, `--config`, `--venv-dir`, `--log-dir`, `--force`, `--dry-run`, `--status` |
| `llm-monitor` | `local_llm_server.monitoring.monitor:main` | Terminal dashboard (curses) |

`llm-server` options: `--config PATH` (default `~/config.yaml` or `./config.yaml`; env `LLM_CONFIG_PATH`), `--host`, `--port`, `--workers` (only `1` is supported; larger values exit with an error), `--log-level`, `--log-file`, `--reload` (development; restarts the process and reloads the model on every code change), `--restart` (stop any instance on the port, then start), `--stop` (stop the instance on the port and exit), `--debug`, `--version`, `--check-config`.

## Install and run

```bash
pip install -e ../local-llm-core           # the engine must be installed first
pip install -e ".[postgres,tui]"           # extras: postgres, mysql, tui (see pyproject.toml)
cp config.yaml ~/llm-config.yaml           # edit it: keys, models, DB, paths
llm-server serve --config ~/llm-config.yaml
```

Production (systemd): `sudo llm-server-install`, then `sudo systemctl enable --now local-llm-cloud`. Details in [DEPLOYMENT.md](DEPLOYMENT.md).

The web UI (`/`, `/demo`) comes from `local-llm-client`. Install it in the same environment if you want the full page: `pip install -e ../local-llm-client`. Without it the server serves a placeholder page.

## Configuration

`config.yaml` is organised into top-level sections. The ones you will touch most:

| Section | Controls |
|---|---|
| `server` | `host`, `port`, `workers`, request-size limits, TLS, CORS, `trusted_proxies` |
| `api` | static `key`, `master_key`, `key_store_enabled`, key-store backend (`storage`, `key_store_db`, DB host/port/credentials), per-IP and per-key rate limits |
| `database` / `conversation` | where API keys and chat history are stored (SQLite, PostgreSQL, MySQL, or in-memory) |
| `models` | `allowed_dir`, `autoload_all`, warm-up settings, `profiles` (the models you can load) |
| `hardware` | hardware-profile overrides (context length, target tokens/s, KV-cache type per profile such as `apu_2gb`) |
| `inference` | engine behaviour, including `default_timeout_s` (**`0` = no deadline**) and max-token handling |
| `scheduler`, `memory`, `model_loader`, `kernels` | queueing, VRAM/RAM management, loading strategy, kernel selection (passed to `local-llm-core`) |
| `monitoring` | metrics collection, Prometheus export (`prometheus_enabled`, `prometheus_port`) |
| `logging` | level, file, rotation |

`${ENV_VAR}` placeholders are expanded in string values when the server loads its config. Validate a file without starting the server: `llm-server --check-config --config PATH`.

## Security notes

1. **Never commit a real `config.yaml`.** The file in this repository currently contains literal API keys and database passwords. Treat any value that has been in git as compromised: rotate it and ship a template with `${ENV}` placeholders instead.
2. **Authentication model.** Requests authenticate with `X-API-Key`; the key-management routes `/admin/keys/*` need `X-Master-Key`. Model load/unload/swap routes (`/v1/models/*`) only need a normal API key, so any key holder can change the loaded model. If neither a static key nor a key store is configured, the server runs **without authentication** (it logs this).
3. **The web page and the key.** `GET /` and `GET /demo` are public. When a static `api.key` is configured, the server inserts it into the served HTML so the page works out of the box — anyone who can load the page can read the key. Prefer the key store (`api.key_store_enabled: true`, keys created with `llm keys create`) and keep the port off the public internet or behind an authenticating reverse proxy.
4. The service runs as `root` in the shipped unit file; run it as a dedicated user and give that user access to the GPU devices.

## Troubleshooting

| Symptom | Check |
|---|---|
| `GET /ready` returns 503 | No model is loaded yet — see `llm status system`, `llm models list`, and the log |
| Requests return text like `[timed out after Ns]` | Deadline reached. Send a larger `timeout_s`, or set `inference.default_timeout_s: 0` for no deadline. Use streaming for long generations |
| 401 / 403 | Missing or wrong `X-API-Key` / `X-Master-Key` |
| `--workers` exits with an error | Intended: the server is single-process (one model copy). Scale out with several servers behind a reverse proxy, see [DEPLOYMENT.md](DEPLOYMENT.md#scaling-out) |
| TUI cannot start | `llm-monitor` needs a real terminal and the `tui` extra |

Diagnostics: `python -m local_llm_server.serving.diagnose_web --help`.
