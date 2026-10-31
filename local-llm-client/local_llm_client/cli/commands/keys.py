"""
local_llm_client/cli/commands/keys.py

Production‑ready API key management commands (admin only).

All client calls are now correctly synchronous; the config uses
`base_url` instead of the non‑existent `endpoint`. Master key is
automatically injected when calling admin endpoints – no extra
header management needed.

Features:
- Respects --config path from main CLI context
- Reuses client instance across commands for efficiency
- JSON output for scripting (`--json`)
- Confirmation prompts for destructive actions (delete, revoke)
- Batch operations (delete-many, revoke-many)
- Detailed key statistics with visual indicators
- Full error handling with user‑friendly messages
- No hardcoded paths – uses config from context
"""

import json
import sys
import time
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.client import LLMCloudClient
from ...core.exceptions import APIError, ConnectionError, CircuitOpenError
from ...core.models import ClientConfig
from ..config import get_config
from ..utils import print_error, print_success, print_info, print_warning, confirm

console = Console()


# ------------------------------------------------------------------------------
# Client factory with caching per context
# ------------------------------------------------------------------------------
def _get_client_from_context(ctx: click.Context) -> LLMCloudClient:
    """
    Create or reuse an LLMCloudClient using the config path from the Click context.
    The client is cached in ctx.obj to avoid multiple connections per command.
    """
    ctx.obj = ctx.obj or {}

    cached = ctx.obj.get("keys_client")
    if cached is not None and not getattr(cached, "_closed", False):
        return cached

    config_path = ctx.obj.get("config_path")
    cfg = get_config(config_path)

    client_cfg = ClientConfig(
        base_url=cfg.base_url,            # CORRECTED
        api_key=cfg.api_key,
        master_key=cfg.master_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=float(cfg.timeout),
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay_s,
        retry_backoff=cfg.retry_backoff_factor,
        stream_chunk_ms=cfg.stream_flush_interval_ms,
        pool_size=cfg.pool_size,
        wait_for_model=cfg.wait_for_model,
        wait_timeout=cfg.wait_timeout_s,
    )

    client = LLMCloudClient(client_cfg)
    ctx.obj["keys_client"] = client
    return client


def _cleanup_client(ctx: click.Context) -> None:
    """Close and remove cached client."""
    client = ctx.obj.get("keys_client")
    if client:
        try:
            client.close()
        except Exception:
            pass
        ctx.obj["keys_client"] = None


# ------------------------------------------------------------------------------
# Helper for formatting key stats
# ------------------------------------------------------------------------------
def _format_key_stats_table(stats: Dict[str, Any]) -> Table:
    """Create a formatted table for key statistics."""
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Key ID", stats.get("key_id", "N/A"))
    table.add_row("Created", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.get("created_at", 0))))
    if stats.get("expires_at"):
        table.add_row("Expires", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["expires_at"])))
    else:
        table.add_row("Expires", "never")
    table.add_row("Rate Limit", f"{stats.get('rate_limit_rps', 0)} RPS / {stats.get('rate_limit_capacity', 0)} burst")
    active_status = "✅ Active" if stats.get("is_active") else "❌ Inactive"
    table.add_row("Status", active_status)
    table.add_row("Total Requests", f"{stats.get('total_requests', 0):,}")
    table.add_row("Total Tokens", f"{stats.get('total_tokens', 0):,}")
    last_used = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["last_used"])) if stats.get(
        "last_used") else "never"
    table.add_row("Last Used", last_used)
    table.add_row("Last IP", stats.get("last_ip", "unknown"))
    if stats.get("description"):
        table.add_row("Description", stats["description"])

    return table


# ------------------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------------------
@click.group("keys")
@click.pass_context
def keys(ctx: click.Context) -> None:
    """Manage API keys (requires master key)."""
    ctx.obj = ctx.obj or {}
    _get_client_from_context(ctx)


# ------------------------------------------------------------------------------
# Create key command
# ------------------------------------------------------------------------------
@keys.command("create")
@click.option("--id", "key_id", required=True, help="Key identifier (e.g., 'user-alice')")
@click.option("--rps", default=10.0, type=float, help="Rate limit (requests per second)")
@click.option("--burst", default=20.0, type=float, help="Burst capacity")
@click.option("--description", default="", help="Optional description")
@click.option("--expires", type=float, help="Expiration in seconds from now (e.g., 86400 for 1 day)")
@click.option("--inactive", is_flag=True, help="Create as inactive (default: active)")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def create_key(
        ctx: click.Context,
        key_id: str,
        rps: float,
        burst: float,
        description: str,
        expires: Optional[float],
        inactive: bool,
        output_json: bool,
) -> None:
    """Create a new API key. The plain key is shown only once."""

    client = _get_client_from_context(ctx)
    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
        ) as progress:
            progress.add_task(f"Creating key '{key_id}'...", total=None)
            result = client.create_key(
                key_id=key_id,
                rate_limit_rps=rps,
                rate_limit_capacity=burst,
                description=description,
                expires_in_seconds=expires,
                active=not inactive,
            )

        if output_json:
            console.print(json.dumps(result, indent=2, default=str))
            return

        print_success(f"Key '{key_id}' created.")
        console.print(f"Plain key: [bold yellow]{result['plain_key']}[/] (store it safely)")
        console.print("[dim]This key will not be shown again. Keep it secure![/]")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to create key: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Revoke key command (with confirmation)
# ------------------------------------------------------------------------------
@keys.command("revoke")
@click.argument("key_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def revoke_key(ctx: click.Context, key_id: str, force: bool, output_json: bool) -> None:
    """Revoke (deactivate) an API key."""

    client = _get_client_from_context(ctx)
    try:
        if not force and not confirm(f"Revoke API key '{key_id}'? This will immediately deactivate it.",
                                     default=False):
            print_info("Operation cancelled.")
            return

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
        ) as progress:
            progress.add_task(f"Revoking key '{key_id}'...", total=None)
            result = client.revoke_key(key_id)

        if output_json:
            console.print(json.dumps({"status": "revoked", "key_id": key_id}, indent=2))
            return

        print_success(f"Key '{key_id}' revoked.")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to revoke key: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Activate key command
# ------------------------------------------------------------------------------
@keys.command("activate")
@click.argument("key_id")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def activate_key(ctx: click.Context, key_id: str, output_json: bool) -> None:
    """Activate a previously revoked key."""

    client = _get_client_from_context(ctx)
    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
        ) as progress:
            progress.add_task(f"Activating key '{key_id}'...", total=None)
            result = client.activate_key(key_id)

        if output_json:
            console.print(json.dumps({"status": "activated", "key_id": key_id}, indent=2))
            return

        print_success(f"Key '{key_id}' activated.")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to activate key: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Delete key command (with confirmation)
# ------------------------------------------------------------------------------
@keys.command("delete")
@click.argument("key_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def delete_key(ctx: click.Context, key_id: str, force: bool, output_json: bool) -> None:
    """Permanently delete an API key (cannot be undone)."""

    client = _get_client_from_context(ctx)
    try:
        if not force:
            if not confirm(f"⚠️ PERMANENTLY DELETE API key '{key_id}'? This action cannot be undone.",
                           default=False):
                print_info("Operation cancelled.")
                return
            if not confirm("Type 'yes' to confirm deletion", default=False):
                print_info("Operation cancelled.")
                return

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
        ) as progress:
            progress.add_task(f"Deleting key '{key_id}'...", total=None)
            result = client.delete_key(key_id)

        if output_json:
            console.print(json.dumps({"status": "deleted", "key_id": key_id}, indent=2))
            return

        print_success(f"Key '{key_id}' deleted.")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to delete key: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# List keys command
# ------------------------------------------------------------------------------
@keys.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show inactive keys as well")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.option("--limit", type=int, default=100, help="Maximum number of keys to show")
@click.pass_context
def list_keys(ctx: click.Context, show_all: bool, output_json: bool, limit: int) -> None:
    """List API keys (active only by default)."""

    client = _get_client_from_context(ctx)
    try:
        keys_list = client.list_keys(active_only=not show_all)
        if not keys_list:
            print_info("No keys found.")
            return

        keys_list = keys_list[:limit]

        if output_json:
            console.print(json.dumps(keys_list, indent=2, default=str))
            return

        table = Table(title=f"API Keys ({len(keys_list)} shown)", box=box.ROUNDED)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("RPS", style="green", justify="right")
        table.add_column("Burst", style="green", justify="right")
        table.add_column("Active", style="yellow")
        table.add_column("Requests", style="blue", justify="right")
        table.add_column("Tokens", style="blue", justify="right")
        table.add_column("Last Used", style="dim")

        for k in keys_list:
            last_used = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(k.get("last_used", 0)))
                if k.get("last_used")
                else "never"
            )
            table.add_row(
                k["id"],
                str(k.get("rate_limit_rps", 0)),
                str(k.get("rate_limit_capacity", 0)),
                "✅" if k.get("is_active") else "❌",
                f"{k.get('total_requests', 0):,}",
                f"{k.get('total_tokens', 0):,}",
                last_used,
            )
        console.print(table)

        total_active = sum(1 for k in keys_list if k.get("is_active"))
        total_inactive = len(keys_list) - total_active
        console.print(
            f"\n[dim]Summary: {total_active} active, {total_inactive} inactive "
            f"(showing {len(keys_list)} of {len(keys_list)} keys)[/]")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to list keys: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Key statistics command
# ------------------------------------------------------------------------------
@keys.command("stats")
@click.argument("key_id")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def key_stats(ctx: click.Context, key_id: str, output_json: bool) -> None:
    """Show detailed statistics for an API key."""

    client = _get_client_from_context(ctx)
    try:
        stats = client.key_stats(key_id)
        if not stats:
            print_error(f"Key '{key_id}' not found.")
            sys.exit(1)

        if output_json:
            console.print(json.dumps(stats, indent=2, default=str))
            return

        stats["key_id"] = key_id
        table = _format_key_stats_table(stats)
        console.print(Panel(table, title=f"📊 Key Statistics: {key_id}", border_style="cyan"))

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to get key stats: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Batch revoke multiple keys
# ------------------------------------------------------------------------------
@keys.command("revoke-many")
@click.argument("key_ids", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Skip individual confirmations")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def revoke_many(ctx: click.Context, key_ids: tuple, force: bool, output_json: bool) -> None:
    """Revoke multiple API keys at once."""

    client = _get_client_from_context(ctx)
    results = []
    success_count = 0
    fail_count = 0

    for key_id in key_ids:
        if not force and not confirm(f"Revoke API key '{key_id}'?", default=False):
            results.append({"key_id": key_id, "status": "skipped"})
            continue

        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[progress.description]Revoking '{key_id}'..."),
                    console=console,
                    transient=True,
            ) as progress:
                progress.add_task("", total=None)
                client.revoke_key(key_id)
            results.append({"key_id": key_id, "status": "revoked"})
            success_count += 1
            print_success(f"Revoked '{key_id}'")
        except Exception as e:
            results.append({"key_id": key_id, "status": "failed", "error": str(e)})
            fail_count += 1
            print_error(f"Failed to revoke '{key_id}': {e}")

    if output_json:
        console.print(json.dumps(results, indent=2))
        return

    console.print(f"\n[bold]Summary:[/] {success_count} revoked, {fail_count} failed")
    if fail_count > 0:
        sys.exit(1)
    else:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Batch delete multiple keys
# ------------------------------------------------------------------------------
@keys.command("delete-many")
@click.argument("key_ids", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Skip individual confirmations")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def delete_many(ctx: click.Context, key_ids: tuple, force: bool, output_json: bool) -> None:
    """Permanently delete multiple API keys (cannot be undone)."""

    client = _get_client_from_context(ctx)
    results = []
    success_count = 0
    fail_count = 0

    if not force and not confirm(f"⚠️ Permanently delete {len(key_ids)} API key(s)? This cannot be undone.",
                                 default=False):
        print_info("Operation cancelled.")
        return

    for key_id in key_ids:
        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[progress.description]Deleting '{key_id}'..."),
                    console=console,
                    transient=True,
            ) as progress:
                progress.add_task("", total=None)
                client.delete_key(key_id)
            results.append({"key_id": key_id, "status": "deleted"})
            success_count += 1
            print_success(f"Deleted '{key_id}'")
        except Exception as e:
            results.append({"key_id": key_id, "status": "failed", "error": str(e)})
            fail_count += 1
            print_error(f"Failed to delete '{key_id}': {e}")

    if output_json:
        console.print(json.dumps(results, indent=2))
        return

    console.print(f"\n[bold]Summary:[/] {success_count} deleted, {fail_count} failed")
    if fail_count > 0:
        sys.exit(1)
    else:
        _cleanup_client(ctx)
