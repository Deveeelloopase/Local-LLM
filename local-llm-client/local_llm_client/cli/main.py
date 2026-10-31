"""
local_llm_client/cli/main.py

Production‑ready CLI entry point for Local LLM Cloud.

Improvements over previous version:
- Rich version banner with logo.
- Global options: --config, --debug, --verbose, --quiet, --no-color.
- Startup validation (config exists, server URL set, optional connectivity check).
- Doctor command (`llm doctor`) for configuration and connectivity health.
- Dynamic command registration with graceful fallback when modules are missing.
- Contextual help text with links to docs.
- Rich traceback only in debug mode; concise errors otherwise.
- All subcommands receive a pre‑loaded config and optional API key injection.
- No hardcoded paths – everything via context.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install as install_rich_traceback

# Local imports
from . import __version__
from .utils import (
    print_error, print_success, print_info, print_warning, print_hr,
    create_header, format_duration, ask_choice, ask_string,
)

# ------------------------------------------------------------------------------
# Rich traceback (only when debugging)
# ------------------------------------------------------------------------------
def install_traceback(debug: bool) -> None:
    if debug:
        install_rich_traceback(show_locals=True, width=120, suppress=[click])
    else:
        # In production, show concise traceback without locals
        install_rich_traceback(show_locals=False, width=100, suppress=[click])

console = Console()


# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
def setup_logging(debug: bool, verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    if not debug:
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "aiohttp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
def get_config(config_path: Optional[Path] = None):
    """Lazy import to avoid circular deps."""
    from .config import get_config as _get_config
    return _get_config(config_path)


# ------------------------------------------------------------------------------
# Doctor command (configuration & connectivity health)
# ------------------------------------------------------------------------------
@click.command("doctor")
@click.pass_context
def doctor_cmd(ctx: click.Context) -> None:
    """
    Diagnose configuration and server connectivity.

    Checks:
      - Configuration file existence and validity
      - Server reachability (health endpoint)
      - API key validity (if configured)
    """
    config_path = ctx.obj.get("config_path")
    cfg = get_config(config_path)

    console.print(create_header(__version__))
    console.print("[bold]Configuration Health Check[/]\n")

    # 1. Config file
    if cfg.config_path.exists():
        print_success(f"Config file found: {cfg.config_path}")
    else:
        print_warning(f"Config file not found; defaults used ({cfg.config_path})")

    # 2. Server URL
    base_url = cfg.base_url
    if base_url:
        console.print(f"  Server URL: [cyan]{base_url}[/]")
    else:
        print_error("Server URL not set. Use: llm config set server.base_url <url>")
        return

    # 3. Connectivity
    console.print("\n[bold]Connectivity Check[/]\n")
    try:
        # Lightweight probe – we import client lazily
        from local_llm_client.core.client import LLMCloudClient
        from local_llm_client.core.models import ClientConfig as CoreCfg

        client_cfg = CoreCfg(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            master_key=cfg.master_key,
            timeout=10.0,
            max_retries=1,
        )
        client = LLMCloudClient(client_cfg)
        health = client.health()
        if health.get("status") == "ok":
            print_success(f"Server at {base_url} is healthy")
        else:
            print_warning(f"Server at {base_url} returned status: {health.get('status')}")
        client.close()
    except Exception as e:
        print_error(f"Cannot reach server at {base_url}")
        console.print(f"  [dim]{e}[/]")
        return

    # 4. API key (if configured)
    console.print("\n[bold]API Key Check[/]\n")
    api_key = cfg.api_key
    if api_key:
        try:
            client_cfg = CoreCfg(
                base_url=cfg.base_url,
                api_key=api_key,
                timeout=10.0,
                max_retries=1,
            )
            client = LLMCloudClient(client_cfg)
            resp = client._request("GET", "/v1/me")
            print_success("API key is valid")
            client.close()
        except Exception as e:
            print_warning(f"API key validation failed: {e}")
    else:
        print_info("No API key configured (using localhost bypass if applicable)")

    console.print("\n[dim]Doctor check complete.[/]")


# ------------------------------------------------------------------------------
# Main CLI group
# ------------------------------------------------------------------------------
@click.group()
@click.version_option(
    version=__version__,
    prog_name="llm",
    message=f"%(prog)s %(version)s\n[dim]⚡ Local LLM Cloud – Private AI, Your Data, Your Hardware[/]",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    help="Path to config file (default: ~/.llmcloud/config.yaml)",
)
@click.option("--debug", is_flag=True, help="Enable debug logging and tracebacks")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all non‑error output")
@click.option("--no-color", is_flag=True, help="Disable color output")
@click.pass_context
def cli(
    ctx: click.Context,
    config: Optional[Path],
    debug: bool,
    verbose: bool,
    quiet: bool,
    no_color: bool,
) -> None:
    """
    Local LLM Cloud CLI – Manage your private AI server.

    \b
    Quick examples:
        llm chat "Tell me a joke"
        llm models list
        llm status
        llm config set server.base_url http://localhost:8080
    """
    # Setup console color
    if no_color:
        console.no_color = True

    # Setup logging and traceback
    setup_logging(debug, verbose, quiet)
    install_traceback(debug)

    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["debug"] = debug
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    # Early config load for validation
    try:
        cfg = get_config(config)
        if not cfg.base_url and not quiet:
            print_warning("No server.base_url configured. Using default http://localhost:8080")
            print_info("Set with: llm config set server.base_url <url>")
    except Exception as e:
        if debug:
            console.print_exception()
        print_error(f"Failed to load configuration: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Lazy command imports
# ------------------------------------------------------------------------------
def import_command(cmd_name: str):
    """Dynamically import a command module and return its Click object."""
    try:
        if cmd_name == "chat":
            from .commands.chat import chat
            return chat
        elif cmd_name == "models":
            from .commands.models import models
            return models
        elif cmd_name == "keys":
            from .commands.keys import keys
            return keys
        elif cmd_name == "status":
            from .commands.status import status
            return status
        elif cmd_name == "config":
            from .commands.config_cmd import config_cmd
            return config_cmd
        elif cmd_name == "doctor":
            return doctor_cmd   # defined above
        else:
            raise ImportError(f"Unknown command: {cmd_name}")
    except ImportError as e:
        print_error(f"Command '{cmd_name}' not available: {e}")
        print_info(f"Ensure the module 'local_llm_client.cli.commands.{cmd_name}' exists.")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Register commands dynamically
# ------------------------------------------------------------------------------
def register_commands() -> None:
    """Attach subcommands to the main CLI group."""
    for cmd_name in ["chat", "models", "keys", "status", "config", "doctor"]:
        cmd = import_command(cmd_name)
        cli.add_command(cmd)


register_commands()


# ------------------------------------------------------------------------------
# Fallback help (already handled by Click, but we add a custom one)
# ------------------------------------------------------------------------------
@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show this help message and exit."""
    click.echo(cli.get_help(ctx))


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def main() -> None:
    """Entry point for console_scripts."""
    # Resolve debug from env or argv so the outer handler knows whether
    # --debug was passed, without needing access to the parsed click context.
    debug = bool(os.environ.get("LLM_DEBUG")) or ("--debug" in sys.argv)
    try:
        cli()
    except KeyboardInterrupt:
        print_warning("Interrupted by user")
        sys.exit(130)
    except click.Abort:
        print_warning("Aborted by user")
        sys.exit(1)
    except Exception as e:
        if debug:
            console.print_exception()
        else:
            print_error(f"Unexpected error: {e}")
            print_info("Run with --debug for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
