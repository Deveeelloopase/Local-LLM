#!/usr/bin/env python3
"""
local_llm_client/cli/__main__.py

Production‑grade entry point for `python -m local_llm_client.cli`.

This module is the single, canonical way to start the CLI when the package
is invoked via `python -m`.  It guarantees that the real ``main`` function
from ``local_llm_client.cli.main`` is called, regardless of whether the
package is installed in development mode, via pip, or run directly from a
source checkout.

Features:
- Multiple import fallbacks for resilience in any environment.
- Graceful error messages with actionable advice when imports fail.
- Coloured terminal output (Rich if available, plain text otherwise).
- Early Python version check to prevent cryptic errors.
- Exit code propagation so the shell can detect failures.
- Support for ``LLM_DEBUG`` environment variable for verbose import traces.

Usage:
    python -m local_llm_client.cli [OPTIONS] COMMAND [ARGS]...
"""

import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Python version guard
# ------------------------------------------------------------------------------
if sys.version_info < (3, 10):
    sys.exit(
        "ERROR: Local LLM Client requires Python 3.10 or later.\n"
        f"       You are running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "       Please upgrade your Python installation."
    )

# ------------------------------------------------------------------------------
# Optional coloured output
# ------------------------------------------------------------------------------
def _print_error(msg: str) -> None:
    """Print an error message, with colour if Rich is available."""
    try:
        from rich.console import Console
        Console(stderr=True).print(f"[bold red]Error:[/] {msg}")
    except ImportError:
        print(f"ERROR: {msg}", file=sys.stderr)


def _print_info(msg: str) -> None:
    """Print an informational message."""
    try:
        from rich.console import Console
        Console(stderr=True).print(f"[dim]{msg}[/]")
    except ImportError:
        print(msg, file=sys.stderr)


# ------------------------------------------------------------------------------
# Import the real main function
# ------------------------------------------------------------------------------
_main_func = None
_import_errors = []

# 1. Standard installed‑package import
try:
    from .main import main as _main_func  # type: ignore[assignment]
except ImportError as exc:
    _import_errors.append(f"Relative import failed: {exc}")

# 2. Add project root for direct execution from source
if _main_func is None:
    # Compute the project root (three levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from local_llm_client.cli.main import main as _main_func
    except ImportError as exc:
        _import_errors.append(f"Absolute import from {project_root} failed: {exc}")

# 3. Last resort – try to locate the package using pkgutil
if _main_func is None:
    try:
        import pkgutil
        import importlib
        # Attempt to load the package metadata to find its location
        spec = importlib.util.find_spec("local_llm_client")
        if spec and spec.origin:
            base = Path(spec.origin).parent
            cli_main = base / "cli" / "main.py"
            if cli_main.exists():
                # Inject the base dir into sys.path as a last resort
                if str(base.parent) not in sys.path:
                    sys.path.insert(0, str(base.parent))
                from local_llm_client.cli.main import main as _main_func
    except Exception as exc:
        _import_errors.append(f"pkgutil fallback failed: {exc}")

# ------------------------------------------------------------------------------
# If we still don't have the function, exit with a helpful message
# ------------------------------------------------------------------------------
if _main_func is None:
    _print_error("Could not import the CLI main module.")
    _print_info("We tried the following:")
    for i, err in enumerate(_import_errors, 1):
        print(f"  {i}. {err}", file=sys.stderr)
    _print_info("\nPossible solutions:")
    _print_info("  • Ensure local-llm-client is installed: pip install local-llm-client")
    _print_info("  • Run from the correct directory: cd local-llm-client && python -m local_llm_client.cli")
    _print_info("  • Set PYTHONPATH: export PYTHONPATH=/path/to/local-llm-client")
    _print_info("  • Use pip install -e . for development mode")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Show a small banner if LLM_DEBUG is set (for debugging import issues)
    if os.environ.get("LLM_DEBUG"):
        _print_info(f"Python {sys.version} on {sys.platform}")
        _print_info(f"CLI main module successfully imported from {_main_func.__module__}")

    # Run the real main function, propagating the exit code
    try:
        sys.exit(_main_func())
    except KeyboardInterrupt:
        _print_info("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        _print_error(f"Unhandled exception: {exc}")
        if os.environ.get("LLM_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)
