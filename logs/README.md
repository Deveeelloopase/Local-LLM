# logs/

Runtime log output directory for the Local LLM services.

| File | Written by |
|---|---|
| `llm_server.log` | `local-llm-server` — the systemd unit (`local-llm-cloud.service`) appends stdout/stderr here, and the server's `--log-file` option points here in that unit |

## Notes

- The directory is created by `local-llm-server/install.sh` (`LOG_DIR`, default `<project root>/logs`) and by the unit's `ExecStartPre`.
- The server's own default log file (when started by hand without `--log-file`) is `/tmp/llm_server.log`; rotation size/backups are set under `logging:` in `config.yaml`.
- `local-llm-client.service` writes to `logs/client.log` (see that unit), the CLI itself logs to `~/.llmcloud/client.log` by default.
- **Do not commit logs.** The current repository contains a sample `llm_server.log`; it should be added to `.gitignore` and removed from version control, since logs can contain request metadata and key identifiers.
- The server installs a request-id log filter (`serving/middleware.py`) and exposes the `X-Request-ID` response header, which lets one request be followed through the log.

## Reading the server log

```bash
tail -f logs/llm_server.log
grep " 500 " logs/llm_server.log         # failed requests
journalctl -u local-llm-cloud -f          # same stream via systemd
```
