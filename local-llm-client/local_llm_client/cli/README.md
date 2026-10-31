# local_llm_client/cli/

The `llm` / `local-llm` command line, built with [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/).

| File | Purpose |
|---|---|
| `main.py` | Entry points: `cli` (Click group), `main()`, the `doctor` command, and `register_commands()` which attaches `chat`, `models`, `keys`, `status`, `config`, `doctor`; also traceback/logging setup |
| `__main__.py` | `python -m local_llm_client.cli` |
| `config.py` | `Config` class plus `get_config()`, `save_config()`, `reload_config()`, `reset_config()`, `get_endpoint()`, `get_api_key()`, `get_default_model()`. Persists settings under `~/.llmcloud/` (`LLM_CONFIG_DIR` overrides the directory, `LLM_CONFIG_FILE` the file) |
| `install_cmd.py` | `llm-install` — setup helper |
| `utils.py` | Output helpers: coloured messages, tables, JSON printing, Markdown/syntax rendering, spinners, progress, `LiveDisplay`, streaming display |
| [`commands/`](commands/README.md) | One module per command group |

## Command overview

```
llm chat [MESSAGE] [-m MODEL] [-i] [--no-stream] ...    chat (one-shot or interactive)
llm models  list | load | unload | swap | load-profile | status | ...
llm keys    create | revoke | activate | delete | list | stats | revoke-many | delete-many   (master key)
llm status  ping | health | ready | system | metrics | queue | logs | dashboard | all
llm config  show | get | set | unset | list | validate | backup | restore | reload | path | edit | export | import | reset
llm doctor
llm help
```

Run `llm <group> --help` for exact options.

## Configuration precedence

command-line flag → environment variable (e.g. `LLM_API_KEY`) → stored config in `~/.llmcloud/` → built-in default.

## Notes

- `prompt_toolkit` drives interactive input (history, completion, multi-line); without it the CLI falls back to plain readline.
- `LLM_DEBUG=1` enables verbose import/exception output.
