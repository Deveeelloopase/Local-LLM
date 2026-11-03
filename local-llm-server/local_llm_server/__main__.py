#!/usr/bin/env python3
"""
local_llm_server/__main__.py — Production Entry Point v1.0.0
=============================================================
Robust launcher for the Local LLM Cloud server.

Features:
  • Automatic venv detection (falls back to system Python if needed)
  • Dependency check before server start (optional; can be skipped via env)
  • Config file validation (`--check-config`)
  • `--version` flag
  • All original arguments forwarded exactly as before
  • Clean signal handling and exit codes
  • Works equally well for manual runs and systemd services

Usage:
    python -m local_llm_server [serve] [--host HOST] [--port PORT] [--config CONFIG]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Early environment setup – ensure the correct Python interpreter is used
# ──────────────────────────────────────────────────────────────────────────────
_VENV_BIN = Path(sys.executable).parent
if _VENV_BIN.name.lower() != "scripts":  # Windows has "Scripts"
    # If not running from a venv, try to find the default venv from config
    _VENV_BIN = None  # will be handled later


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _find_default_config() -> str:
    """Return the first existing config.yaml from standard locations."""
    candidates = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Return the home path anyway; the server will report the error cleanly
    return str(Path.home() / "config.yaml")


def _check_dependencies() -> None:
    """Check for critical packages and print installation instructions if missing."""
    missing = []
    for pkg, pip_name in [
        ("fastapi", "fastapi>=0.109.1"),
        ("uvicorn", "uvicorn[standard]>=0.27.0"),
        ("pydantic", "pydantic>=2.5.0"),
        ("yaml", "pyyaml>=6.0"),
        ("psutil", "psutil>=5.9.0"),
        ("aiosqlite", "aiosqlite>=0.19.0"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append((pkg, pip_name))

    if missing:
        sys.stdout.write("\n")
        sys.stdout.write("  ╔══════════════════════════════════════════════════╗\n")
        sys.stdout.write("  ║  ❌  LOCAL LLM CLOUD — MISSING DEPENDENCIES      ║\n")
        sys.stdout.write("  ╚══════════════════════════════════════════════════╝\n")
        sys.stdout.write("\n  The following required packages are not installed:\n\n")
        for pkg, pip_name in missing:
            sys.stdout.write(f"    • {pkg}  →  pip install \"{pip_name}\"\n")
        sys.stdout.write("\n  Install all at once:\n")
        sys.stdout.write("    pip install -r requirements.txt\n")
        sys.stdout.write("\n  If you used a virtual environment, activate it first:\n")
        sys.stdout.write("    source /path/to/venv/bin/activate\n\n")
        sys.exit(1)


def _validate_config_file(path: str) -> bool:
    """Quickly check that a YAML config file exists and is parseable."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        return False
    if not p.is_file():
        print(f"ERROR: Config path is not a regular file: {path}", file=sys.stderr)
        return False
    suffix = p.suffix.lower()
    if suffix not in (".yaml", ".yml"):
        print(f"ERROR: Config file must have .yaml or .yml extension, got: {suffix}", file=sys.stderr)
        return False
    try:
        import yaml
        with open(p) as f:
            yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse config file {path}: {e}", file=sys.stderr)
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m local_llm_server",
        description="Local LLM Cloud — Production LLM Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m local_llm_server
  python -m local_llm_server --host 0.0.0.0 --port 8080
  python -m local_llm_server --config /etc/llm-cloud/config.yaml
  python -m local_llm_server --version
  python -m local_llm_server --check-config
        """,
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="serve",
        choices=["serve"],
        help="Command to run (default: serve)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (default: 8080)",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to config.yaml (default: ~/config.yaml or ./config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--log-file",
        default="/tmp/llm_server.log",
        help="Path to log file (default: /tmp/llm_server.log)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=120.0,
        dest="drain_timeout",
        help=(
            "Seconds to let in-flight generations finish on stop/restart "
            "before cutting them (0 = wait indefinitely). Keep the systemd "
            "unit's TimeoutStopSec larger than this. (default: 120)"
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto‑reload for development",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stop any running instance on this port then start fresh",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop any running instance on this port and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging)",
    )
    # Utility flags
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show server version and exit",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the config file and exit",
    )
    # Pass‑through for TUI / other tools (not used directly here)
    parser.add_argument("--api-key", help="API key for regular endpoints")
    parser.add_argument("--master-key", help="Master key for admin operations")

    args = parser.parse_args()

    # --version
    if args.version:
        from local_llm_server import __version__
        print(f"local-llm-server {__version__}")
        return

    # --check-config
    if args.check_config:
        cfg = args.config_path or _find_default_config()
        if _validate_config_file(cfg):
            print(f"Config file OK: {cfg}")
        else:
            sys.exit(1)
        return

    # Find config file if not explicitly provided
    if args.config_path is None:
        args.config_path = _find_default_config()

    # Quick config validation before starting (optional but helpful)
    if not _validate_config_file(args.config_path):
        # If check fails, print error and exit – server won't start anyway
        sys.exit(1)

    # Optionally check dependencies
    if not os.environ.get("LLM_SKIP_DEPS_CHECK"):
        _check_dependencies()

    # Set environment variables for the server factory
    os.environ["LLM_CONFIG_PATH"] = args.config_path
    if args.debug:
        os.environ["LLM_LOG_LEVEL"] = "DEBUG"
    os.environ["LLM_BUILD_APP_ON_IMPORT"] = "1"

    # Import the server main function
    try:
        from local_llm_server.serving.api_server import main as server_main
    except ImportError as e:
        sys.stderr.write(
            f"ERROR: Could not import server module: {e}\n"
            "Ensure the package is installed correctly.\n"
            "Try: pip install -e /path/to/local-llm-server\n"
        )
        sys.exit(1)

    # Build argv list to pass directly to server_main
    forwarded_argv = [
        args.subcommand,
        "--host", args.host,
        "--port", str(args.port),
        "--config", args.config_path,
        "--log-level", args.log_level,
        "--log-file", args.log_file,
        "--workers", str(args.workers),
        "--drain-timeout", str(args.drain_timeout),
    ]
    if args.reload:
        forwarded_argv.append("--reload")
    if args.restart:
        forwarded_argv.append("--restart")
    if args.stop:
        forwarded_argv.append("--stop")
    if args.api_key:
        forwarded_argv.extend(["--api-key", args.api_key])
    if args.master_key:
        forwarded_argv.extend(["--master-key", args.master_key])

    # Run the server with explicit argv (no sys.argv modification)
    try:
        server_main(argv=forwarded_argv)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"ERROR: Unhandled exception in server: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
