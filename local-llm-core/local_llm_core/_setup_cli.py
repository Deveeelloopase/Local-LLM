#!/usr/bin/env python3
"""
local-llm-setup
===============
One-time system setup for local-llm-core.

Run this once after `pip install local-llm-core`:

    local-llm-setup              # full setup (install.sh + bitnet_setup.sh)
    local-llm-setup --status     # check whether setup has already been done
    local-llm-setup --reset      # delete sentinel and re-run everything
    local-llm-setup --skip-bitnet  # run install.sh only (no BitNet)

Advanced flags (passed directly to bitnet_setup.sh):
    --non-interactive, -y     Accept all defaults
    --dry-run                 No filesystem changes
    --use-pypi-gguf           Force gguf from PyPI and patch setup_env.py
    --apply-patch             Apply patch to setup_env.py
    --minimal-build           Use pre-converted GGUF model only
    --quant-embd              Enable Q6_K embedding quantization
    --skip-clone              Skip BitNet git clone
    --skip-build              Skip BitNet build
    --skip-model              Skip model download/selection
    --bitnet-dir PATH         Override BitNet.cpp install directory
    --model-dir PATH          Override model download directory

Environment variables (alternative to flags):
    BITNET_USE_PYPI_GGUF, BITNET_APPLY_PATCH, BITNET_MINIMAL_BUILD,
    BITNET_QUANT_EMBD, BITNET_SKIP_CLONE, BITNET_SKIP_BUILD, BITNET_SKIP_MODEL,
    BITNET_DIR, BITNET_MODEL_DIR, NON_INTERACTIVE, DRY_RUN
"""

from __future__ import annotations

import argparse
import datetime
import os
import stat
import subprocess
import sys
from pathlib import Path


# ── Sentinel (same location as _auto_setup.py uses) ───────────────────────
_SENTINEL_DIR = Path.home() / ".config" / "local-llm"
_SENTINEL_FILE = _SENTINEL_DIR / ".core_setup_done"


def _find_scripts_dir() -> Path | None:
    """Find scripts/ — works for both editable and wheel installs."""
    override = os.environ.get("LLM_SCRIPTS_DIR", "")
    if override:
        p = Path(override)
        return p if p.is_dir() else None

    here = Path(__file__).parent  # local_llm_core/

    # Editable install: scripts/ is a sibling of local_llm_core/
    candidate = here.parent / "scripts"
    if (candidate / "install.sh").exists():
        return candidate

    # Wheel install: scripts/ copied inside the package
    candidate = here / "scripts"
    if (candidate / "install.sh").exists():
        return candidate

    return None


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    script: Path,
    label: str,
    extra_env: dict | None = None,
    extra_args: list[str] | None = None,
) -> bool:
    """Run a bash script, streaming output live. Returns True on success."""
    _make_executable(script)
    env = {**os.environ, **(extra_env or {})}
    env["PROJECT_ROOT"] = str(script.parent.parent)

    cmd = ["bash", str(script)]
    if extra_args:
        cmd.extend(extra_args)

    hr = "─" * 64
    print(f"\n{hr}")
    print(f"  ⚙   Running: {label}")
    print(f"  →   {' '.join(cmd)}")
    print(f"{hr}\n", flush=True)

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    proc.wait()

    if proc.returncode == 0:
        print(f"\n  ✅  {label} completed.\n", flush=True)
        return True
    else:
        print(
            f"\n  ❌  {label} failed (exit {proc.returncode}).\n"
            f"      Fix the error above, then re-run: local-llm-setup\n",
            file=sys.stderr,
        )
        return False


def _status() -> None:
    done = _SENTINEL_FILE.exists()
    scripts_dir = _find_scripts_dir()
    print()
    print("  local-llm-core setup status")
    print("  " + "─" * 40)
    print(f"  Setup done   : {'✅ yes' if done else '❌ no — run: local-llm-setup'}")
    print(f"  Scripts dir  : {scripts_dir or 'NOT FOUND'}")
    if done:
        ts = _SENTINEL_FILE.read_text().strip()
        print(f"  Completed at : {ts}")
    print()


def _bool_env(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "").lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    return default


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="local-llm-setup",
        description="One-time system setup for local-llm-core (install.sh + bitnet_setup.sh).",
    )
    parser.add_argument("--status", action="store_true", help="Show setup status and exit")
    parser.add_argument("--reset", action="store_true", help="Delete sentinel and re-run setup")
    parser.add_argument("--skip-bitnet", action="store_true", help="Run install.sh only, skip bitnet_setup.sh")
    parser.add_argument("--force", action="store_true", help="Run even if setup was already done")

    # BitNet flags (passed through)
    parser.add_argument("--non-interactive", "-y", action="store_true", help="Accept all defaults in BitNet setup")
    parser.add_argument("--dry-run", action="store_true", help="No filesystem changes in BitNet setup")
    parser.add_argument("--use-pypi-gguf", action="store_true", help="Force gguf from PyPI and patch setup_env.py")
    parser.add_argument("--apply-patch", action="store_true", help="Apply patch to setup_env.py")
    parser.add_argument("--minimal-build", action="store_true", help="Use pre-converted GGUF model only")
    parser.add_argument("--quant-embd", action="store_true", help="Enable Q6_K embedding quantization")
    parser.add_argument("--skip-clone", action="store_true", help="Skip BitNet git clone")
    parser.add_argument("--skip-build", action="store_true", help="Skip BitNet build")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download/selection")
    parser.add_argument("--bitnet-dir", type=str, help="Override BitNet.cpp install directory")
    parser.add_argument("--model-dir", type=str, help="Override model download directory")

    # Model mode selection (mutually exclusive shortcuts)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--transformers-only",
        action="store_true",
        help=(
            "Use packed BitNet model via transformers API only (no bitnet.cpp build). "
            "Sets model source to option 4. No 1-bit hardware acceleration. "
            "Good for API compatibility or when you already have the packed model downloaded."
        ),
    )
    mode_group.add_argument(
        "--model-source",
        type=str,
        choices=["1", "2", "3", "4"],
        help=(
            "Select model source directly (non-interactive): "
            "1=local directory, 2=HuggingFace download (bf16), "
            "3=pre-built GGUF (fastest, recommended), 4=packed transformers-only"
        ),
    )

    args = parser.parse_args()

    if args.status:
        _status()
        return

    if args.reset:
        if _SENTINEL_FILE.exists():
            _SENTINEL_FILE.unlink()
            print("  🗑   Sentinel removed — re-running setup…")
        else:
            print("  ℹ   No sentinel found — running setup fresh…")

    if _SENTINEL_FILE.exists() and not args.force:
        print()
        print("  ✅  local-llm-core is already set up.")
        print()
        print("  To re-run:  local-llm-setup --reset")
        print("  To check:   local-llm-setup --status")
        print()
        return

    # ── Find scripts ────────────────────────────────────────────────────────
    scripts_dir = _find_scripts_dir()
    if scripts_dir is None:
        print(
            "\n  ❌  Cannot find scripts directory.\n"
            "      Expected install.sh and bitnet_setup.sh in:\n"
            "        - <repo>/scripts/  (editable install)\n"
            "        - <site-packages>/local_llm_core/scripts/  (wheel install)\n"
            "      Set LLM_SCRIPTS_DIR=/path/to/scripts to override.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    install_sh = scripts_dir / "install.sh"
    bitnet_sh = scripts_dir / "bitnet_setup.sh"

    # ── Banner ──────────────────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   🚀  local-llm-core — System Setup                     ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Scripts : {scripts_dir}")
    print()

    # ── Build shared environment (install.sh + bitnet_setup.sh) ────────────
    extra_env: dict[str, str] = {}
    for var in (
        "LLM_INSTALL_DIR",
        "LLM_VENV_DIR",
        "LLM_SKIP_KERNEL",
        "LLM_MODEL_DIR",
        "BITNET_DIR",
        "BITNET_MODEL_DIR",
        "BITNET_SKIP_CLONE",
        "BITNET_SKIP_MODEL",
        "BITNET_QUANT_EMBD",
    ):
        if os.environ.get(var):
            extra_env[var] = os.environ[var]

    # ── Step 1: install.sh ──────────────────────────────────────────────────
    if install_sh.exists():
        install_ok = _run(install_sh, "install.sh", extra_env)
    else:
        print(f"\n  ❌  install.sh not found at {install_sh}", file=sys.stderr)
        install_ok = False

    if not install_ok:
        print(
            "\n  ❌  install.sh failed — stopping before BitNet setup.\n"
            "      Fix the errors shown above, then re-run: local-llm-setup\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 2: bitnet_setup.sh (only if install.sh succeeded) ──────────────
    if not args.skip_bitnet:
        # Build argument list for bitnet_setup.sh
        bitnet_args = []

        # CLI flags take precedence over environment variables
        non_interactive = args.non_interactive or _bool_env("NON_INTERACTIVE")
        dry_run = args.dry_run or _bool_env("DRY_RUN")
        use_pypi_gguf = args.use_pypi_gguf or _bool_env("BITNET_USE_PYPI_GGUF")
        apply_patch = args.apply_patch or _bool_env("BITNET_APPLY_PATCH")
        minimal_build = args.minimal_build or _bool_env("BITNET_MINIMAL_BUILD")
        quant_embd = args.quant_embd or _bool_env("BITNET_QUANT_EMBD")
        skip_clone = args.skip_clone or _bool_env("BITNET_SKIP_CLONE")
        skip_build = args.skip_build or _bool_env("BITNET_SKIP_BUILD")
        skip_model = args.skip_model or _bool_env("BITNET_SKIP_MODEL")

        if non_interactive:
            bitnet_args.append("--non-interactive")
        if dry_run:
            bitnet_args.append("--dry-run")
        if use_pypi_gguf:
            bitnet_args.append("--use-pypi-gguf")
        if apply_patch:
            bitnet_args.append("--apply-patch")
        if minimal_build:
            bitnet_args.append("--minimal-build")
        if quant_embd:
            bitnet_args.append("--quant-embd")
        if skip_clone:
            bitnet_args.append("--skip-clone")
        if skip_build:
            bitnet_args.append("--skip-build")
        if skip_model:
            bitnet_args.append("--skip-model")
        if args.bitnet_dir:
            bitnet_args.append(f"--bitnet-dir={args.bitnet_dir}")
        elif os.environ.get("BITNET_DIR"):
            bitnet_args.append(f"--bitnet-dir={os.environ['BITNET_DIR']}")
        if args.model_dir:
            bitnet_args.append(f"--model-dir={args.model_dir}")
        elif os.environ.get("BITNET_MODEL_DIR"):
            bitnet_args.append(f"--model-dir={os.environ['BITNET_MODEL_DIR']}")

        # Model source / mode flags
        if getattr(args, "transformers_only", False):
            bitnet_args.append("--model-source=4")
            # transformers-only path doesn't need user interaction — force non-interactive
            if "--non-interactive" not in bitnet_args:
                bitnet_args.append("--non-interactive")
        elif getattr(args, "model_source", None):
            bitnet_args.append(f"--model-source={args.model_source}")
            # model-source=3 (pre-built GGUF) is also fully non-interactive safe
            if args.model_source in ("3", "4") and "--non-interactive" not in bitnet_args:
                bitnet_args.append("--non-interactive")

        if bitnet_sh.exists():
            ok = _run(bitnet_sh, "bitnet_setup.sh", extra_env, bitnet_args)
        else:
            print(f"  ❌  bitnet_setup.sh not found at {bitnet_sh}", file=sys.stderr)
            ok = False
    else:
        print("  ⏭   Skipping bitnet_setup.sh (--skip-bitnet)")
        ok = True

    # ── Write sentinel ──────────────────────────────────────────────────────
    if ok:
        _SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        _SENTINEL_FILE.write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║   ✅  Setup complete — local-llm-core is ready!         ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()
        print("  You can now start the server:")
        print("    local-llm-server start")
        print()
    else:
        print(
            "\n  ❌  Setup finished with errors.\n"
            "      Fix the issues above and re-run: local-llm-setup\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
