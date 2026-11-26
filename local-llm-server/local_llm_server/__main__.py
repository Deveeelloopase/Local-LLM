#!/usr/bin/env python3
"""
local_llm_server/__main__.py — Production Entry Point
=============================================================
Robust launcher for the Local LLM Cloud server.

Subcommands:
  serve   (default) — start the API server
  tune              — benchmark and suggest optimizations
  bench             — run a raw inference benchmark
  doctor            — diagnose environment/config issues
  autoconfig        — show the model profiles derived from the GGUF files and hardware
  calibrate         — measure decode speed per model (the server also does this at start-up)
  ab                — A/B comparison of decode settings for one model

Features:
  • Dependency check before server start (optional; skip via env)
  • Config file validation (`--check-config`)
  • `--version` flag
  • All original arguments forwarded exactly as before for `serve`
  • Clean signal handling and exit codes
  • Works equally well for manual runs and systemd services
  • subcommand plumbing:
      * `tune` / `bench` / `doctor` delegate to their tool modules when
        installed, with a clean error + exit(1) when the module is absent
        (Drops 46/47 are deferred per plan §9 — until then, the commands
        print an actionable message and exit 1 rather than a traceback).
      * `--` pass-through so `llm-server tune -- --min-tps 5` forwards
        `--min-tps 5` verbatim to the tool.
      * Signal handlers installed so Ctrl-C during a long benchmark
        produces a clean exit(130) rather than a hard interrupt.
      * `doctor` runs WITHOUT the fastapi/uvicorn dependency check AND
        WITHOUT top-level config validation — diagnosing broken installs
        or malformed configs is precisely what it is for.

Usage:
    python -m local_llm_server [serve] [--host HOST] [--port PORT] [--config CONFIG]
    python -m local_llm_server tune   [--config CONFIG] [-- --tool-flags ...]
    python -m local_llm_server bench  [--config CONFIG] [-- --tool-flags ...]
    python -m local_llm_server doctor [--config CONFIG] [-- --tool-flags ...]
    python -m local_llm_server autoconfig [--config CONFIG] [-- plan|show|explain ID|check]
    python -m local_llm_server calibrate  [--config CONFIG] [-- --force --all --model ID]
    python -m local_llm_server ab         [--config CONFIG] -- --model ID --kv f16,q8_0

version = 1.0.0
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path


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
# Subcommand dispatch helpers
# ──────────────────────────────────────────────────────────────────────────────
# Maps subcommand → (module_path, callable_name). The callable must be
# synchronous and accept an `argv: list[str]` positional (may be empty).
# `monitoring/__main__.py::main(argv=...)` is the reference shape.
_TOOL_MODULES = {
    "tune":   ("local_llm_server.tools.tune",   "main"),
    "bench":  ("local_llm_server.tools.tune",   "bench_main"),
    "doctor": ("local_llm_server.tools.doctor", "main"),
    "autoconfig": ("local_llm_server.tools.autoconfig", "main"),
    "calibrate":  ("local_llm_server.tools.bench_decode", "main"),
    "ab":         ("local_llm_server.tools.bench_ab", "main"),
}

# Subcommands that load models or read GGUF files but never start the HTTP
# stack, so the web-framework dependency check does not apply to them.
_MODEL_TOOLS = ("autoconfig", "calibrate", "ab")


def _install_signal_handlers() -> None:
    """Turn SIGINT/SIGTERM into clean exit(128 + signum) for tool subcommands.

    Bash convention: 130 = killed by SIGINT, 143 = killed by SIGTERM.
    The default Python behaviour of letting KeyboardInterrupt propagate
    would print a traceback mid-benchmark — noisy and misleading (looks
    like a crash, not an intentional Ctrl-C).
    """
    def _clean(signum, frame):
        sys.exit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _clean)
        except (ValueError, OSError):
            # Not the main thread, or signal not supported on this OS.
            pass


def _split_passthrough(argv):
    """Split argv at the first '--'.

    Returns (head, tail). `head` is handed to argparse; `tail` is
    forwarded verbatim to the tool subcommand's own argv.
    """
    try:
        idx = argv.index("--")
    except ValueError:
        return list(argv), []
    return list(argv[:idx]), list(argv[idx + 1:])


def _run_tool(name: str, argv: list) -> int:
    """Delegate to a tool module. Returns the process exit code.

    Failure modes handled explicitly so the operator always gets a
    clear, actionable message — never a traceback that looks like a
    server crash:
      • Missing tool module  → clean msg, rc 1
      • Module exists but callable missing          → clean msg, rc 1
      • Tool called sys.exit(N)                     → propagate N
      • Tool caught KeyboardInterrupt               → rc 130
    """
    module_name, fn_name = _TOOL_MODULES[name]
    try:
        import importlib
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        sys.stderr.write(
            f"ERROR: subcommand '{name}' is not available: {exc}\n"
            f"\n"
            f"The '{name}' command delegates to "
            f"`{module_name}:{fn_name}`, which is not yet installed.\n"
            f"\n"
            f"Until then, use the individual building blocks:\n"
            f"  • Live metrics:         curl -s http://127.0.0.1:8080/v1/metrics\n"
            f"  • Endpoint diagnostics: llm-monitor check\n"
            f"  • TUI dashboard:        llm-monitor tui\n"
            f"  • Server logs:          tail -f <config.logging.file>\n"
        )
        return 1

    entry = getattr(mod, fn_name, None)
    if entry is None:
        _avail = [
            a for a in dir(mod)
            if not a.startswith("_") and callable(getattr(mod, a, None))
        ]
        sys.stderr.write(
            f"ERROR: module '{module_name}' does not export "
            f"`{fn_name}` as required by the '{name}' entrypoint.\n"
            f"Available callables: {_avail}\n"
        )
        return 1

    _install_signal_handlers()
    try:
        rc = entry(argv)
    except SystemExit as se:
        # Tools may sys.exit(); propagate the code.
        return int(se.code) if isinstance(se.code, int) else 1
    except KeyboardInterrupt:
        return 130
    return int(rc) if isinstance(rc, int) else 0


def _forward_serve_argv(args, cfg_path: str) -> list:
    """Build argv for `api_server.main()` — EXACT parity with the previous
    pre-Drop-48 behaviour for `serve`."""
    forwarded = [
        "serve",
        "--host", args.host,
        "--port", str(args.port),
        "--config", cfg_path,
        "--log-level", args.log_level,
        "--log-file", args.log_file,
        "--workers", str(args.workers),
        "--drain-timeout", str(args.drain_timeout),
    ]
    if args.reload:
        forwarded.append("--reload")
    if args.restart:
        forwarded.append("--restart")
    if args.stop:
        forwarded.append("--stop")
    if args.api_key:
        forwarded.extend(["--api-key", args.api_key])
    if args.master_key:
        forwarded.extend(["--master-key", args.master_key])
    return forwarded


def _forward_tool_argv(args, cfg_path: str, passthrough: list) -> list:
    """Build argv for the tool subcommands. Only the flags the tools
    understand — NOT `serve`-specific flags like `--host`/`--port`."""
    forwarded = ["--config", cfg_path]
    if getattr(args, "api_key", None):
        forwarded.extend(["--api-key", args.api_key])
    if getattr(args, "master_key", None):
        forwarded.extend(["--master-key", args.master_key])
    if getattr(args, "debug", False):
        forwarded.append("--debug")
    forwarded.extend(passthrough)
    return forwarded


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Split at `--` so argparse sees only the head; anything after `--` is
    # forwarded verbatim to a tool subcommand (per subcommand contract).
    head, passthrough = _split_passthrough(
        sys.argv[1:] if len(sys.argv) > 1 else []
    )

    parser = argparse.ArgumentParser(
        prog="python -m local_llm_server",
        description="Local LLM Cloud — Production LLM Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  serve    (default) — start the API server
  tune               — benchmark and suggest optimizations
  bench              — run a raw inference benchmark
  doctor             — diagnose environment/config issues
  autoconfig         — show the profiles derived from the GGUF files and hardware
  calibrate          — measure decode speed per model (also automatic at start-up)
  ab                 — A/B comparison of decode settings for one model

The `tune`, `bench`, `doctor`, `autoconfig`, `calibrate` and `ab` subcommands accept arbitrary flags after
`--` and forward them verbatim to their tool:

  python -m local_llm_server tune -- --models qwen3-8b,qwen3-4b
  python -m local_llm_server calibrate -- --force --model qwen3-8b
  python -m local_llm_server autoconfig -- explain granite-4.2-8b

For full tool flags, see `python -m local_llm_server <cmd> -- --help`.

Examples:
  python -m local_llm_server
  python -m local_llm_server --host 0.0.0.0 --port 8080
  python -m local_llm_server --config /etc/llm-cloud/config.yaml
  python -m local_llm_server --version
  python -m local_llm_server --check-config
  python -m local_llm_server tune --config config.yaml
  python -m local_llm_server doctor --config config.yaml
        """,
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="serve",
        choices=["serve", "tune", "bench", "doctor", "autoconfig", "calibrate", "ab"],
        help="Command to run (default: serve)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (serve only; default: 127.0.0.1 — pass 0.0.0.0 explicitly to listen on all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (serve only; default: 8080)",
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
        help="Log level (serve only; default: info)",
    )
    parser.add_argument(
        "--log-file",
        default="/tmp/llm_server.log",
        help="Path to log file (serve only; default: /tmp/llm_server.log)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (serve only; default: 1)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=120.0,
        dest="drain_timeout",
        help=(
            "Seconds to let in-flight generations finish on stop/restart "
            "before cutting them (serve only; 0 = wait indefinitely). Keep "
            "the systemd unit's TimeoutStopSec larger than this. "
            "(default: 120)"
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (serve only)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stop any running instance on this port then start fresh (serve only)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop any running instance on this port and exit (serve only)",
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
    # Pass-through for TUI / other tools (also forwarded to tool subcommands)
    parser.add_argument("--api-key", help="API key for regular endpoints")
    parser.add_argument("--master-key", help="Master key for admin operations")

    args = parser.parse_args(head)

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

    # Resolve config path (may or may not exist — doctor handles that case)
    if args.config_path is None:
        args.config_path = _find_default_config()

    # ── Subcommand dispatch ────────────────────────────────────────────────

    # doctor: skip config validation AND dependency check — diagnosing
    # broken configs and installs is precisely what it is for.
    if args.subcommand == "doctor":
        os.environ["LLM_CONFIG_PATH"] = args.config_path
        if args.debug:
            os.environ["LLM_LOG_LEVEL"] = "DEBUG"
        tool_argv = _forward_tool_argv(args, args.config_path, passthrough)
        sys.exit(_run_tool("doctor", tool_argv))

    # tune / bench: need a valid config and the inference stack, so they
    # take the full top-level checks.
    if args.subcommand in ("tune", "bench"):
        if not _validate_config_file(args.config_path):
            sys.exit(1)
        os.environ["LLM_CONFIG_PATH"] = args.config_path
        if args.debug:
            os.environ["LLM_LOG_LEVEL"] = "DEBUG"
        if not os.environ.get("LLM_SKIP_DEPS_CHECK"):
            _check_dependencies()
            tool_argv = _forward_tool_argv(args, args.config_path, passthrough)
            sys.exit(_run_tool(args.subcommand, tool_argv))

        # autoconfig / calibrate / ab: need a valid config but not the HTTP stack.
        if args.subcommand in _MODEL_TOOLS:
            if not _validate_config_file(args.config_path):
                sys.exit(1)
            os.environ["LLM_CONFIG_PATH"] = args.config_path
            if args.debug:
                os.environ["LLM_LOG_LEVEL"] = "DEBUG"
            tool_argv = _forward_tool_argv(args, args.config_path, passthrough)
            sys.exit(_run_tool(args.subcommand, tool_argv))

        # ── serve path ────────────────────────────────────────────────────────
        # `--` passthrough is meaningless for `serve` — warn but do not fail.
        if passthrough:
            sys.stderr.write(
            f"WARNING: arguments after '--' are ignored for the 'serve' "
            f"subcommand: {passthrough}\n"
        )

    if not _validate_config_file(args.config_path):
        sys.exit(1)

    if not os.environ.get("LLM_SKIP_DEPS_CHECK"):
        _check_dependencies()

    os.environ["LLM_CONFIG_PATH"] = args.config_path
    if args.debug:
        os.environ["LLM_LOG_LEVEL"] = "DEBUG"
    os.environ["LLM_BUILD_APP_ON_IMPORT"] = "1"

    try:
        from local_llm_server.serving.api_server import main as server_main
    except ImportError as e:
        sys.stderr.write(
            f"ERROR: Could not import server module: {e}\n"
            "Ensure the package is installed correctly.\n"
            "Try: pip install -e /path/to/local-llm-server\n"
        )
        sys.exit(1)

    forwarded_argv = _forward_serve_argv(args, args.config_path)

    try:
        server_main(argv=forwarded_argv)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"ERROR: Unhandled exception in server: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
