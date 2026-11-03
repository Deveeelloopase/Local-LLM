#!/usr/bin/env python3
"""
local_llm_client/cli/install_cmd.py
=====================================
llm-install — one-command post-install setup for local-llm-client.

Registered as a console script by setup.py so that after
`pip install local-llm-client` the user only needs to run:

    sudo llm-install

instead of hunting for install.sh manually.

What it does:
  1. Locates install.sh (works for both editable and wheel installs)
  2. Accepts overrides for the hardcoded defaults in install.sh via
     CLI flags or environment variables
  3. Runs install.sh under bash, streaming output live
  4. Writes a marker so re-running is safe (idempotent)

Usage:
    sudo llm-install                                # use defaults
    sudo llm-install --work-dir /my/path           # override WORK_DIR
    sudo llm-install --venv-dir /my/venv           # override VENV_DIR
    sudo llm-install --server-url http://host:8080 # override SERVER_URL
    sudo llm-install --force                        # re-run even if done
    sudo llm-install --dry-run                      # print command, don't run
    llm-install --status                            # check if already done

Skip during CI/CD:
    SKIP_POST_INSTALL=1 pip install local-llm-client
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
    return _MARKER_DIR / f".client_install_done_{venv_hash}"


def _is_done() -> bool:
    return _marker_path().exists()


def _mark_done() -> None:
    _MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(
        f"python={sys.executable}\n"
        f"completed={datetime.datetime.now().isoformat()}\n"
    )


# ── Script locator ────────────────────────────────────────────────────────────
def _find_install_sh() -> Path | None:
    """Locate install.sh for both editable and wheel installs."""
    override = os.environ.get("LLM_CLIENT_SCRIPTS_DIR", "")
    if override:
        p = Path(override) / "install.sh"
        return p if p.exists() else None

    # Editable install — install.sh is next to setup.py
    here = Path(__file__).resolve().parents[2]  # local_llm_client/ → package root
    candidate = here / "install.sh"
    if candidate.exists():
        return candidate

    # Wheel install — bundled inside the package
    candidate = Path(__file__).parent.parent / "install.sh"
    if candidate.exists():
        return candidate

    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="llm-install",
        description="Post-install setup for local-llm-client (systemd service + config).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo llm-install
  sudo llm-install --work-dir /home/myuser/local-llm-cloud/local-llm-client
  sudo llm-install --venv-dir /home/myuser/myvenv
  sudo llm-install --server-url http://192.168.1.10:8080
  sudo llm-install --force          # re-run even if already done
  sudo llm-install --dry-run        # show what would run without running it
  llm-install --status              # check if setup has been done
        """,
    )
    ap.add_argument("--work-dir",    metavar="PATH",
                    help="Client working directory (overrides WORK_DIR in install.sh)")
    ap.add_argument("--venv-dir",    metavar="PATH",
                    help="Virtual environment path (overrides VENV_DIR in install.sh)")
    ap.add_argument("--server-url",  metavar="URL",
                    help="Server URL (overrides SERVER_URL in install.sh, default: http://localhost:8080)")
    ap.add_argument("--force",       action="store_true",
                    help="Re-run install.sh even if already done")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print the command that would run without executing it")
    ap.add_argument("--status",      action="store_true",
                    help="Show whether setup has been done and exit")
    args = ap.parse_args()

    # ── Status check ─────────────────────────────────────────────────────────
    if args.status:
        done = _is_done()
        print()
        print("  llm-install status")
        print("  " + "─" * 38)
        print(f"  Done    : {'✅ yes' if done else '❌ no — run: sudo llm-install'}")
        print(f"  Marker  : {_marker_path()}")
        if done:
            ts = _marker_path().read_text().strip()
            print(f"  Details : {ts}")
        print()
        return

    # ── Already done? ─────────────────────────────────────────────────────────
    if _is_done() and not args.force:
        print()
        print("  ✅  local-llm-client is already set up.")
        print(f"      Marker: {_marker_path()}")
        print()
        print("  To re-run : sudo llm-install --force")
        print("  To check  : llm-install --status")
        print()
        return

    # ── Locate install.sh ─────────────────────────────────────────────────────
    script = _find_install_sh()
    if script is None:
        print(
            "\n  ❌  install.sh not found.\n"
            "      Set LLM_CLIENT_SCRIPTS_DIR=/path/to/dir to locate it.\n"
            "      Or run manually: sudo bash local-llm-client/install.sh\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build environment ─────────────────────────────────────────────────────
    run_env = {**os.environ, "LLM_SETUP_RUNNING": "1"}

    if args.work_dir:
        run_env["WORK_DIR"] = args.work_dir
    if args.venv_dir:
        run_env["VENV_DIR"] = args.venv_dir
    if args.server_url:
        run_env["SERVER_URL"] = args.server_url

    cmd = ["bash", str(script)]

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print()
        print("  🔍  Dry run — would execute:")
        print(f"      bash {script}")
        overrides = {k: v for k, v in run_env.items()
                     if k in ("WORK_DIR", "VENV_DIR", "SERVER_URL", "LLM_SETUP_RUNNING")}
        for k, v in overrides.items():
            print(f"      {k}={v}")
        print()
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print( "  🚀  local-llm-client — running install.sh")
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
        print("\n\n  ⚠  Interrupted by user.\n", file=sys.stderr)
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
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║   ✅  local-llm-client setup complete!                  ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()
        print("  Usage:")
        print("    llm doctor                  — verify connection to server")
        print("    llm status health           — check server health")
        print("    llm chat --interactive      — start chatting")
        print()
    else:
        print(
            f"\n  ❌  install.sh exited with code {proc.returncode}.\n"
            f"      Fix the error above, then re-run: sudo llm-install\n",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
    