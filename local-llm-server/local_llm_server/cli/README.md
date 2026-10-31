# local_llm_server/cli/

| File | Purpose |
|---|---|
| `server_install_cmd.py` | Implements `llm-server-install` (also `python -m local_llm_server.cli.server_install_cmd`). Locates `install.sh` and runs it, passing along the options below |
| `__init__.py` | Package marker |

```bash
sudo llm-server-install                     # interactive
sudo llm-server-install --non-interactive   # take everything from config/env
sudo llm-server-install --status            # show installation state
sudo llm-server-install --dry-run           # print actions only
```

Other flags: `--config PATH`, `--venv-dir PATH`, `--log-dir PATH`, `--force`.

What `install.sh` does (v4.3.0): reads `config.yaml` (prompting for missing values), creates PostgreSQL/MySQL users and databases for the key store and the conversation store (or prepares SQLite files), writes the credentials back into `config.yaml`, creates the log directory and installs the systemd unit.
