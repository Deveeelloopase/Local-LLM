# scripts/

System setup for `local-llm-core`. Normally run through `local-llm-setup` (`local_llm_core/_setup_cli.py`), which runs `install.sh` and then `bitnet_setup.sh` and records completion in a marker file.

| File | Purpose |
|---|---|
| `install.sh` | System installer (Linux). Runs checks, installs system packages, creates/uses the Python virtualenv, installs Python dependencies, detects GPU backends (ROCm/CUDA/Vulkan), optionally builds and loads the kernel modules from `local_llm_core/kernels/`, installs the `gguf_tl_shim` helper into site-packages, and makes `lib/*.sh` executable. Sections are named in the script (ATLAS hardware probe, GUARDIAN pre-checks, SPECTRE gguf-py conflict handling, NEXUS venv management) |
| `bitnet_setup.sh` | Clones and builds **BitNet.cpp**, converts/downloads BitNet models, applies the patches in `patches/`, and writes the state file (`~/.config/local-llm/bitnet_state.json`) that the engine reads (`BitNetStateManager`) |
| [`lib/`](lib/README.md) | Helper tools (shell and Python) used or installed by the scripts |
| [`patches/`](patches/README.md) | Files that replace upstream BitNet.cpp files |

## Running

```bash
./scripts/install.sh --help
./scripts/bitnet_setup.sh --help
```

Common `bitnet_setup.sh` flags: `--non-interactive` / `-y`, `--dry-run`, `--skip-clone`, `--skip-build`, `--skip-model`, `--quant-embd`, `--use-pypi-gguf`, `--apply-patch`, `--minimal-build`. `local-llm-setup --reset` removes the completion marker so both scripts run again.

## Conventions

- Scripts print what they change; `--dry-run` shows the plan where supported.
- Paths are derived from the script's own location and the environment (`PROJECT_ROOT`, `VENV_DIR`, `LOCAL_LLM_STATE_DIR`); avoid adding absolute paths.
