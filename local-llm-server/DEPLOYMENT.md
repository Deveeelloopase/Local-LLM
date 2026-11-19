# Deployment guide (local-llm-server)

Targets a single Linux host with systemd. Paths below use `<project root>` for your clone of the monorepo.

## 1. Prerequisites

- Python ≥ 3.10, `git`, a virtual environment.
- `local-llm-core` installed and set up (`local-llm-setup`), see [`../local-llm-core/README.md`](../local-llm-core/README.md).
- GGUF model files in the directory you will set as `models.allowed_dir`.
- Optional: PostgreSQL or MySQL server for keys and chat history (SQLite needs nothing).

## 2. Install the packages

```bash
cd <project root>
python3 -m venv .venv && source .venv/bin/activate
pip install -e local-llm-core
pip install -e "local-llm-server[postgres,tui]"
pip install -e local-llm-client            # web UI + CLI
```

## 3. Configure

Copy `local-llm-server/config.yaml` somewhere you control and edit at least:

- `server.host` / `server.port` (use `127.0.0.1` if a reverse proxy sits in front),
- `api.key` / `api.master_key` — or enable the key store and create keys afterwards,
- `database` / `conversation` — storage backend and credentials (use `${ENV_VAR}` placeholders, never literals),
- `models.allowed_dir` and the model definitions,
- `logging.file` and any other absolute paths.

## 4. Install the service

```bash
sudo llm-server-install
```

The installer (`install.sh`, v4.3.0; also `sudo bash install.sh`; non-interactive mode via `NONINTERACTIVE=1` plus `PG_ADMIN_USER`, `API_DB_*`, `CONV_DB_*` environment variables) prepares the database user/schemas, writes generated credentials into the config, creates the log directory and installs `local-llm-cloud.service`. Review the generated unit: the committed template contains paths from the author's machine and runs as `root`; change `User=`, `WorkingDirectory=` and `ExecStart=` to match your host.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now local-llm-cloud
systemctl status local-llm-cloud
journalctl -u local-llm-cloud -f
```

The unit's `ExecStart` runs the `llm-server` executable of your virtualenv by absolute path (systemd does not expand `${VAR}`), and `ExecStartPre` frees port 8080 if a stale process holds it.

## 5. Verify

```bash
curl -s localhost:8080/health
curl -s localhost:8080/ready
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/chat/completions -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"model":"<model id>","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
python -m local_llm_server.serving.diagnose_web --help     # full endpoint diagnostics
```

## 6. Operate

| Task | Command |
|---|---|
| Restart after code/config change | `sudo systemctl restart local-llm-cloud` |
| Live dashboard | `llm-monitor` |
| Logs | `tail -f <project root>/logs/llm_server.log` |
| Create/revoke keys | `llm keys create`, `llm keys revoke …` (needs master key) |
| Load / swap models | `llm models load --id <id>`, `llm models swap …` |
| Remove the service | `sudo systemctl disable --now local-llm-cloud && sudo rm /etc/systemd/system/local-llm-cloud.service && sudo systemctl daemon-reload` (there is no uninstall command) |

## 7. Hardening checklist

- [ ] Real credentials only in environment variables or a root-owned config (mode 600), never in git.
- [ ] Key store enabled; static `api.key` removed (the static key is embedded in the public web page).
- [ ] Service runs as a dedicated non-root user.
- [ ] Port reachable only from trusted networks or via a TLS-terminating reverse proxy that sets `X-Forwarded-For` for a configured trusted proxy.
- [ ] Log rotation configured under `logging:`.
- [ ] `inference.default_timeout_s` and `timeout_s` strategy decided (0 = no deadline; clients should stream long generations).

## Scaling out

One server process serves one machine's GPU/CPU. For more capacity or availability run one server per host and put a reverse proxy in front (nginx, HAProxy, Caddy, Traefik):

- balance with **least connections**; health-check `GET /ready` (200 only when a model is loaded);
- keep a conversation on one node by hashing the `X-Session-ID` header (or the `llm_session` cookie), or point every node at the same PostgreSQL/MySQL **conversation store** so any node can continue a conversation;
- use the same key store database on all nodes so API keys are valid everywhere;
- disable proxy buffering for streaming responses (`proxy_buffering off` in nginx) so Server-Sent Events arrive token by token;
- give streaming/long requests generous read timeouts (generation on small hardware can take minutes).

`GET /v1/cluster` always answers `{"nodes": 1, "role": "standalone"}`; the cluster view is the proxy's job.
