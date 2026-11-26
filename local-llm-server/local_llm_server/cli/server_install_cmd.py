#!/usr/bin/env python3
"""
local_llm_server/cli/server_install_cmd.py
============================================
llm-server-install — one-command post-install setup for local-llm-server.

Registered as a console script by pyproject.toml. After
`pip install local-llm-server` the user runs:

    sudo llm-server-install

This wraps install.sh, which handles:
  - Virtual environment detection
  - PostgreSQL / MySQL / SQLite database setup
  - Systemd service file installation
  - config.yaml credential writing
  - Optional TUI dependency install

Usage:
    sudo llm-server-install
    sudo llm-server-install --non-interactive
    sudo llm-server-install --config /path/to/config.yaml
    sudo llm-server-install --force
    sudo llm-server-install --dry-run
    llm-server-install --status

Environment variable overrides (passed straight to install.sh):
    VENV_DIR         virtual environment path
    CONFIG_FILE      path to config.yaml
    LOG_DIR          log directory
    SERVICE_NAME     systemd service name
    NONINTERACTIVE=1 skip all prompts
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# ── Marker (idempotency) ──────────────────────────────────────────────────────
_MARKER_DIR = Path.home() / ".config" / "local-llm"


def _marker_path() -> Path:
    venv_hash = hashlib.sha256(sys.executable.encode()).hexdigest()[:14]
    return _MARKER_DIR / f".server_install_done_{venv_hash}"


def _is_done() -> bool:
    return _marker_path().exists()


def _mark_done() -> None:
    _MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(
        f"python={sys.executable}\n"
        f"completed={datetime.datetime.now().isoformat()}\n"
    )


# ── Locate install.sh ─────────────────────────────────────────────────────────
def _find_install_sh() -> Path | None:
    override = os.environ.get("LLM_SERVER_SCRIPTS_DIR", "")
    if override:
        p = Path(override) / "install.sh"
        return p if p.exists() else None

    # Editable install — install.sh is at package root (next to pyproject.toml)
    here = Path(__file__).resolve().parents[2]   # cli/ → local_llm_server/ → root
    for candidate in [here / "install.sh", here.parent / "install.sh"]:
        if candidate.exists():
            return candidate
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="llm-server-install",
        description="Post-install setup for local-llm-server (databases, systemd, config).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo llm-server-install                          # interactive (recommended)
  sudo llm-server-install --non-interactive        # CI / headless
  sudo llm-server-install --config /etc/llm/config.yaml
  sudo llm-server-install --venv-dir /opt/myvenv
  sudo llm-server-install --force                  # re-run even if done
  sudo llm-server-install --dry-run                # preview only
  llm-server-install --status                      # check if done

Non-interactive environment variables (passed to install.sh):
  VENV_DIR         path to virtual environment
  CONFIG_FILE      path to config.yaml
  LOG_DIR          log directory path
  PG_ADMIN_USER    postgres admin user (default: postgres)
  API_DB_USER      API key store DB user
  API_DB_PASS      API key store DB password
  API_DB_NAME      API key store database name
  CONV_DB_USER     conversation store DB user
  CONV_DB_PASS     conversation store DB password
  CONV_DB_NAME     conversation store database name
        """,
    )
    ap.add_argument("--non-interactive", action="store_true",
                    help="Skip all prompts, use environment variables or defaults")
    ap.add_argument("--config",    metavar="PATH", default=None,
                    help="Path to config.yaml (overrides CONFIG_FILE env var)")
    ap.add_argument("--venv-dir",  metavar="PATH", default=None,
                    help="Virtual environment path (overrides VENV_DIR env var)")
    ap.add_argument("--log-dir",   metavar="PATH", default=None,
                    help="Log directory (overrides LOG_DIR env var)")
    ap.add_argument("--force",     action="store_true",
                    help="Re-run install.sh even if already done")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print the command that would run without executing it")
    ap.add_argument("--status",    action="store_true",
                    help="Show whether setup has been done and exit")
    args = ap.parse_args()

    # ── Status ────────────────────────────────────────────────────────────────
    if args.status:
        done = _is_done()
        print()
        print("  llm-server-install status")
        print("  " + "─" * 40)
        print(f"  Done    : {'✅ yes' if done else '❌ no — run: sudo llm-server-install'}")
        print(f"  Marker  : {_marker_path()}")
        if done:
            print(f"  Details : {_marker_path().read_text().strip()}")
        print()
        return

    # ── Already done? ─────────────────────────────────────────────────────────
    if _is_done() and not args.force:
        print()
        print("  ✅  local-llm-server is already set up.")
        print(f"      Marker: {_marker_path()}")
        print()
        print("  To re-run : sudo llm-server-install --force")
        print("  To check  : llm-server-install --status")
        print()
        return

    # ── Locate install.sh ─────────────────────────────────────────────────────
    script = _find_install_sh()
    if script is None:
        print(
            "\n  ❌  install.sh not found.\n"
            "      Set LLM_SERVER_SCRIPTS_DIR=/path/to/dir to locate it.\n"
            "      Or run manually: sudo bash local-llm-server/install.sh\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build environment ─────────────────────────────────────────────────────
    run_env = {**os.environ, "LLM_SETUP_RUNNING": "1"}

    if args.non_interactive:
        run_env["NONINTERACTIVE"] = "1"
    if args.config:
        run_env["CONFIG_FILE"] = args.config
    if args.venv_dir:
        run_env["VENV_DIR"] = args.venv_dir
    if args.log_dir:
        run_env["LOG_DIR"] = args.log_dir

    cmd = ["bash", str(script)]

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print()
        print("  🔍  Dry run — would execute:")
        print(f"      bash {script}")
        relevant = {k: v for k, v in run_env.items()
                    if k in ("NONINTERACTIVE", "CONFIG_FILE", "VENV_DIR",
                             "LOG_DIR", "LLM_SETUP_RUNNING")}
        for k, v in relevant.items():
            print(f"      {k}={v}")
        print()
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    sep = "═" * 64
    print(f"\n{sep}")
    print( "  🚀  local-llm-server — running install.sh")
    print(f"  →   sudo bash {script}")
    print(f"{sep}\n", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            env=run_env,
        )
        proc.wait()
    except KeyboardInterrupt:
        print("\n\n  ⚠  Interrupted.\n", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(
            f"\n  ❌  Could not run install.sh: {exc}\n"
            f"      Run manually: sudo bash {script}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if proc.returncode == 0:
        _mark_done()
        print()
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║   ✅  local-llm-server setup complete!                    ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
        print()
        print("  Start the service:")
        print("    sudo systemctl start local-llm-cloud")
        print("    sudo systemctl status local-llm-cloud")
        print()
        print("  Monitor:")
        print("    journalctl -u local-llm-cloud -f")
        print("    llm-monitor --config local-llm-server/config.yaml")
        print()
    else:
        print(
            f"\n  ❌  install.sh exited with code {proc.returncode}.\n"
            f"      Fix the error above then re-run: sudo llm-server-install\n",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
