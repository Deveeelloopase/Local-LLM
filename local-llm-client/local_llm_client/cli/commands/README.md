# local_llm_client/cli/commands/

One module per top-level CLI command. `__init__.py` exposes `get_all_commands()`; `cli/main.py` registers `chat`, `models`, `keys`, `status`, `config` and `doctor`.

| Module | Command | What it does |
|---|---|---|
| `chat.py` | `llm chat` | One-shot and interactive chat. Streams by default and renders Markdown live; `<think>` blocks are shown separately (`ThinkParser` handles tags split across chunks); conversations can be saved/loaded (`save_conversation`, `load_conversation`, `list_conversations`) |
| `models.py` | `llm models …` | List models, load/unload/swap, load a configured profile, show model status |
| `keys.py` | `llm keys …` | API-key administration (create, revoke, activate, delete, list, stats, bulk revoke/delete). Requires the server **master key** |
| `status.py` | `llm status …` | `ping`, `health`, `ready`, `system`, `metrics`, `queue`, `logs`, `dashboard`, `all` — each maps to a server endpoint |
| `config_cmd.py` | `llm config …` | Show/get/set/unset/validate/backup/restore/export/import/reset client configuration |

## `llm chat` options (from `chat.py`)

`MESSAGE` (positional, optional) · `--model/-m` · `--interactive/-i` · `--no-stream` · `--temperature` · `--max-tokens` (0 = unlimited) · `--top-p` (0.95) · `--top-k` (40) · `--stop` (comma separated) · `--no-think` · `--system` · `--timeout` · `--api-key` (env `LLM_API_KEY`) · `--base-url` · `--conv-id` · `--save` · `--json` · `--load` · `--list-convs` · `--resume` · `--resume-conv` · `--show-internal`

By default draft/internal models (see `is_draft_model`) are hidden from selection; `--show-internal` reveals them.

## Adding a command

1. Create `mycmd.py` with a Click command/group named like the module.
2. Add its name to the list in `register_commands()` in `cli/main.py`.
3. Add a CLI test in `tests/test_all.py` (the suite uses Click's runner).
4. 