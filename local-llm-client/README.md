# local-llm-client

CLI (`llm`), Python SDK (sync + async) and web-UI source for talking to a Local LLM Cloud server. It only needs HTTP access to a running `local-llm-server`; it does **not** need `local-llm-core` installed.

- Package: `local-llm-client` · import name `local_llm_client` · version `1.0.0` · Python ≥ 3.10 (`<4.0`)
- Console scripts (from `setup.py`): `llm`, `local-llm` (both → `local_llm_client.cli.main:cli`) and `llm-install` (→ `local_llm_client.cli.install_cmd:main`)
- Also runnable as `python -m local_llm_client`

## Folder contents

| Path | What it is |
|---|---|
| [`local_llm_client/`](local_llm_client/README.md) | The Python package |
| [`local_llm_client/cli/`](local_llm_client/cli/README.md) | Click CLI entry point, config handling, output helpers |
| [`local_llm_client/cli/commands/`](local_llm_client/cli/commands/README.md) | One module per CLI command group (`chat`, `models`, `keys`, `status`, `config`) |
| [`local_llm_client/core/`](local_llm_client/core/README.md) | SDK: sync/async clients, models, exceptions, connection pools, agent adapter |
| [`local_llm_client/web/`](local_llm_client/web/README.md) | The single-file web chat UI (HTML/JS as a Python string) served by the server |
| [`tests/`](tests/README.md) | Test suite |
| `config.yaml` | Reference configuration with every client option and its default |
| `requirements.txt` | Runtime dependencies (`httpx`, `urllib3`, `click`, `rich`, `prompt_toolkit`, `pyyaml`, `pydantic`, `psutil`, `aiohttp`, `cffi`, `pygments`) |
| `setup.py` | Packaging, extras, console scripts |
| `install.sh` | Installer used by `llm-install` / manual setup |
| `local-llm-client.service` | Optional systemd unit — a placeholder daemon (`sleep` loop) that only keeps an environment alive; the client does not need a daemon |

## Install

```bash
pip install -e .                  # core dependencies
pip install -e ".[perf]"          # + aiohttp, cffi (faster async streaming, C accelerator)
pip install -e ".[pygments]"      # + syntax highlighting for JSON output
pip install -e ".[all]"           # perf, aiohttp, cffi, http2, prometheus, otel, pygments
pip install -e ".[dev]"           # + pytest, ruff, mypy, black
```

Extras defined in `setup.py`: `dev`, `perf`, `aiohttp`, `cffi`, `http2`, `prometheus`, `otel`, `pygments`, `all`.

## Configure

```bash
llm config set base_url http://localhost:8080
llm config set api_key  <your-key>
llm config show
```

Settings are stored under `~/.llmcloud/` (override the directory with `LLM_CONFIG_DIR`, the file with `LLM_CONFIG_FILE`). Environment variables recognised by the client code: `LLM_API_KEY`, `LLM_MASTER_KEY`, `LLM_CONFIG_DIR`, `LLM_CONFIG_FILE`, `LLM_SESSION_DIR`, `LLM_DEBUG`. Per-command flags (`--api-key`, `--base-url`, …) override stored values for one invocation.

## Use the CLI

```bash
llm chat "Explain mutexes in two sentences" -m qwen3-8b     # one-shot, streams by default
llm chat -i                                                 # interactive session
llm chat "..." --no-stream --temperature 0.2 --max-tokens 256 --no-think
llm chat --list-convs                                       # saved conversations
llm models list                                             # models on the server
llm models load --id qwen3-8b
llm status ping                                             # reachability
llm status health | ready | system | metrics | queue | logs | dashboard | all
llm keys list                                               # admin; needs the master key
llm config show
llm doctor                                                  # diagnose local setup
```

Note that `-m` means `--model`, not "message". The message is the positional argument. Full option lists are in [`cli/commands/`](local_llm_client/cli/commands/README.md).

## Use the SDK

```python
import asyncio
from local_llm_client import AsyncLLMCloudClient, ClientConfig

async def main():
    cfg = ClientConfig(base_url="http://localhost:8080", api_key="YOUR_KEY", model="qwen3-8b")
    async with AsyncLLMCloudClient(cfg) as client:
        print(await client.generate("Hello!", max_tokens=64))
        async for chunk in client.chat_stream(
            messages=[{"role": "user", "content": "Tell me a story"}],
            model_id="qwen3-8b",
        ):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

- The constructor takes a **`ClientConfig`** (not `base_url=`/`api_key=` keyword arguments).
- Streaming methods take `model_id=` (not `model=`).
- Sync equivalent: `LLMCloudClient(cfg)` with the same method names (`generate`, `chat_completions`, `chat_stream`, `stream`, `resume`, …).
- Admin helpers (`create_key`, `list_keys`, `load_model`, `unload_model`, `hot_swap`, `get_logs`, …) need a master key (`client.set_master_key(...)`).
- Errors derive from `LLMCloudError` (see [`core/`](local_llm_client/core/README.md)).

## Web UI

`local_llm_client/web/web_ui.py` holds the HTML/JS of the chat page. The **server** imports it (`from local_llm_client.web.web_ui import HTML`) and serves it at `/` and `/demo`; if the client package is not installed the server falls back to a minimal placeholder page. Restart the server after changing it.

## Testing

```bash
pytest tests/test_all.py -v
LLM_TEST_SERVER_URL=http://localhost:8080 pytest tests/test_all.py -v    # include live-server tests
```

## Known issues

- `local-llm-client.service` and `config.yaml` examples contain absolute paths from the author's machine (`/home/deve-oops/...`); adjust before use.
- The committed `local_llm_client.egg-info/` directory is build output and should be untracked.
- 