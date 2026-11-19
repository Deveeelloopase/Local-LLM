"""
hatch_build.py — local-llm-server build hook
=============================================
Hatchling calls initialize() on EVERY pip install, including `pip install -e .`.

This replaces the old setup.py PostInstallCommand / PostDevelopCommand which
were silently skipped because modern pip (PEP 517/660) never calls setup.py
cmdclass hooks.

Skip conditions (same logic as the old setup.py):
  SKIP_POST_INSTALL=1    — CI/CD, Docker build layers
  LLM_SETUP_RUNNING=1   — recursion guard (install.sh called pip internally)
"""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Marker file lives in ~/.config/local-llm/ and is keyed to the current
# Python executable so that each virtual environment gets its own marker.
# Re-installing the package in the same venv never re-runs install.sh.
_MARKER_DIR = Path.home() / ".config" / "local-llm"


def _marker_path() -> Path:
    venv_hash = hashlib.sha256(sys.executable.encode()).hexdigest()[:14]
    return _MARKER_DIR / f".server_install_done_{venv_hash}"


def _is_already_done() -> bool:
    return _marker_path().exists()


def _mark_done() -> None:
    _MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(
        f"python={sys.executable}\n"
        f"completed={datetime.datetime.now().isoformat()}\n"
    )

# hatchling is declared in [build-system] requires in pyproject.toml so it is
# always present when pip runs this file.  The try/except fallback stub is only
# here to keep IDEs (PyCharm, VS Code, etc.) happy when hatchling is not
# installed in the project's dev interpreter.
# Fix for your IDE: pip install hatchling
try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:
    class BuildHookInterface:  # type: ignore[no-redef]
        """Stub — replaced at build time by the real hatchling class."""
        root: str
        config: dict

        def initialize(self, version: str, build_data: dict) -> None: ...
        def finalize(self, version: str, build_data: dict, artifact_path: str) -> None: ...


class CustomBuildHook(BuildHookInterface):
    """Fires install.sh after pip resolves and installs all dependencies."""

    PLUGIN_NAME = "custom"

    # ------------------------------------------------------------------
    # initialize() is called by hatchling for BOTH wheel and editable
    # builds, which is exactly when we want install.sh to run.
    # ------------------------------------------------------------------
    def initialize(self, version: str, build_data: dict) -> None:
        # NOTE: hatchling's initialize() fires during pip's build phase where
        # PIP_BUILD_TRACKER is always set. Auto-running install.sh here is
        # unreliable — it would be blocked by _should_skip() on every run.
        #
        # The correct approach is a console script: llm-server-install
        # Registered in pyproject.toml as:
        #   llm-server-install = "local_llm_server.cli.server_install_cmd:main"
        #
        # After `pip install local-llm-server` completes, the user runs:
        #   sudo llm-server-install
        #
        # This hook simply prints that instruction so it's visible
        # immediately after pip finishes.
        sep = "═" * 64
        print(f"\n{sep}", flush=True)
        print(  "  ✅  local-llm-server installed!", flush=True)
        print(  "  →   Run setup (databases, service file, config):", flush=True)
        print(  "        sudo llm-server-install", flush=True)
        print(  "  →   Or non-interactively (CI/CD):", flush=True)
        print(  "        sudo NONINTERACTIVE=1 llm-server-install", flush=True)
        print(f"{sep}\n", flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _should_skip(self) -> bool:
        env = os.environ

        # Explicit opt-out (CI, Docker, etc.)
        if env.get("SKIP_POST_INSTALL", "").strip().lower() in ("1", "true", "yes"):
            return True

        # Recursion guard — install.sh itself called pip/python
        if env.get("LLM_SETUP_RUNNING"):
            return True

        return False

    def _run_install_sh(self) -> bool:
        here   = Path(self.root)
        script = here / "install.sh"

        if not script.exists():
            self._warn(
                f"install.sh not found at {script}\n"
                "Run manually: sudo bash install.sh"
            )
            return False

        env = {
            **os.environ,
            "LLM_SETUP_RUNNING": "1",
            "SKIP_POST_INSTALL": "1",
            # Suppress interactive prompts during pip install.
            # Users needing custom DB credentials should run
            # `sudo bash install.sh` manually afterwards.
            "NONINTERACTIVE": "1",
        }

        sep = "═" * 62
        print(f"\n{sep}", flush=True)
        print( "  🚀  local-llm-server — running install.sh",  flush=True)
        print(f"  →   sudo bash {script}",                      flush=True)
        print(f"{sep}\n",                                        flush=True)

        try:
            proc = subprocess.Popen(
                ["bash", str(script)],
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                env=env,
            )
            proc.wait()
            if proc.returncode == 0:
                print(
                    "\n  ✅  install.sh completed.\n"
                    "  →   Start: llm-server serve --config config.yaml\n",
                    flush=True,
                )
                return True
            else:
                self._warn(
                    f"install.sh exited with code {proc.returncode}.\n"
                    "Fix the issue above, then re-run: sudo bash install.sh"
                )
                return False
        except Exception as exc:
            self._warn(
                f"Could not launch install.sh: {exc}\n"
                "Run manually: sudo bash install.sh"
            )
            return False

    def _info(self, msg: str) -> None:
        print(f"\n  ℹ  {msg}\n", flush=True)

    def _warn(self, msg: str) -> None:
        print(f"\n  ⚠  {msg}\n", file=sys.stderr, flush=True)
