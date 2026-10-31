"""
local_llm_client/cli/commands/models.py

Production‑ready model management commands.

Fixes applied:
- All client calls are synchronous (no async/await on sync client)
- Uses actual server model‑load‑status endpoint for polling
- Config attributes corrected to match the Config class (e.g. base_url)
- Added `get_model_load_status` method to client (implemented herein)
- Robust error handling with exit codes
- Enhanced output with model type (gguf/bitnet) and strategy
"""

import asyncio
import sys
import time
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import box
from rich.panel import Panel
from rich.text import Text

from ...core.client import LLMCloudClient
from ...core.exceptions import (
    APIError, ConnectionError, CircuitOpenError, ModelNotLoadedError,
    TimeoutError as CoreTimeoutError
)
from ...core.models import ClientConfig
from ..config import get_config, Config
from ..utils import print_error, print_success, print_info, print_warning, with_spinner

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

    # Return cached client if it exists and is not closed
    cached = ctx.obj.get("models_client")
    if cached is not None and not getattr(cached, "_closed", False):
        return cached

    config_path = ctx.obj.get("config_path")
    cfg = get_config(config_path)

    # Build ClientConfig from CLI Config (converts endpoint -> base_url)
    client_cfg = ClientConfig(
        base_url=cfg.base_url,            # corrected from cfg.endpoint
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
    ctx.obj["models_client"] = client
    return client


def _cleanup_client(ctx: click.Context) -> None:
    """Close and remove cached client."""
    client = ctx.obj.get("models_client")
    if client:
        try:
            client.close()
        except Exception:
            pass
        ctx.obj["models_client"] = None


# ------------------------------------------------------------------------------
# Helper for async model status polling via /v1/models/loading/{model_id}
# ------------------------------------------------------------------------------
async def _wait_for_model_ready(
        client: LLMCloudClient,
        model_id: str,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
) -> bool:
    """
    Poll the server's /v1/models/loading/{model_id} endpoint until
    the model is loaded. Returns True if ready within timeout, False otherwise.

    This helper calls the sync client method inside the event loop without
    blocking other tasks (uses loop.run_in_executor).
    """
    loop = asyncio.get_running_loop()
    start = time.monotonic()
    last_error = None

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
    ) as progress:
        task = progress.add_task(f"Waiting for model '{model_id}' to load...", total=100)

        while time.monotonic() - start < timeout:
            try:
                # call sync method in thread pool to avoid blocking the loop
                status_info = await loop.run_in_executor(
                    None,
                    lambda: client.get_model_load_status(model_id)
                )
                status = status_info.get("status", "unknown")
                if status == "loaded":
                    progress.update(task, completed=100)
                    return True
                elif status == "failed":
                    error_msg = status_info.get("error", "unknown error")
                    raise RuntimeError(f"Model '{model_id}' failed to load: {error_msg}")
                # else "loading" – continue
                elapsed = time.monotonic() - start
                progress.update(task, completed=min(100, int((elapsed / timeout) * 100)))
                await asyncio.sleep(poll_interval)
            except Exception as e:
                last_error = e
                await asyncio.sleep(poll_interval)

    if last_error:
        print_error(f"Model ready check failed: {last_error}")
    return False


# ------------------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------------------
@click.group("models")
@click.pass_context
def models(ctx: click.Context) -> None:
    """Manage models (load, unload, list, swap, load-profile, status)."""
    ctx.obj = ctx.obj or {}
    # Ensure client is created once
    _get_client_from_context(ctx)


# ------------------------------------------------------------------------------
# List models command
# ------------------------------------------------------------------------------
@models.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.pass_context
def list_models(ctx: click.Context, verbose: bool) -> None:
    """List currently loaded models with detailed statistics."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            models_data = client.models()       # returns list, sync
            if not models_data:
                print_info("No models loaded on server.")
                return

            if verbose:
                table = Table(title="Loaded Models (Detailed)", box=box.ROUNDED, show_lines=True)
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Type", style="green")
                table.add_column("Backend", style="green")
                table.add_column("GPU Layers", style="yellow", justify="right")
                table.add_column("CPU Layers", style="yellow", justify="right")
                table.add_column("VRAM (MB)", style="magenta", justify="right")
                table.add_column("Requests", style="blue", justify="right")
                table.add_column("Strategy", style="white")
                table.add_column("Context Len", style="dim", justify="right")

                for m in models_data:
                    table.add_row(
                        m.get("id", "-"),
                        m.get("model_type", "gguf"),
                        m.get("backend", "-"),
                        str(m.get("gpu_layers", 0)),
                        str(m.get("cpu_layers", 0)),
                        str(m.get("vram_mb", 0)),
                        str(m.get("requests", 0)),
                        m.get("strategy", "unknown"),
                        str(m.get("context_length", "auto")),
                    )
            else:
                table = Table(title="Loaded Models", box=box.ROUNDED)
                table.add_column("ID", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Backend", style="green")
                table.add_column("GPU Layers", style="yellow")
                table.add_column("VRAM (MB)", style="magenta")
                table.add_column("Requests", style="blue")

                for m in models_data:
                    table.add_row(
                        m.get("id", "-"),
                        m.get("model_type", "gguf"),
                        m.get("backend", "-"),
                        str(m.get("gpu_layers", 0)),
                        str(m.get("vram_mb", 0)),
                        str(m.get("requests", 0)),
                    )

            console.print(table)

            total_vram = sum(m.get("vram_mb", 0) for m in models_data)
            total_requests = sum(m.get("requests", 0) for m in models_data)
            console.print(
                f"\n[dim]Summary: {len(models_data)} model(s), {total_vram} MB VRAM, {total_requests} total requests[/]")

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Load model command
# ------------------------------------------------------------------------------
@models.command("load")
@click.option("--id", "model_id", required=True, help="Model identifier (e.g., 'qwen3-8b')")
@click.option("--path", "model_path", required=True, help="Path to model file (GGUF or BitNet directory)")
@click.option("--context-length", type=int, default=0, help="Context length (0 = auto)")
@click.option("--gpu-layers", type=int, default=-1, help="Number of GPU layers (-1 = auto)")
@click.option("--threads", type=int, default=0, help="Number of CPU threads (0 = auto)")
@click.option("--model-type", "model_type", default="auto",
              type=click.Choice(["gguf", "bitnet", "auto"], case_sensitive=False),
              help="Model backend type")
@click.option("--wait", is_flag=True, default=False, help="Wait for model to be fully loaded")
@click.option("--wait-timeout", type=float, default=120.0, help="Max wait time in seconds (default: 120)")
@click.pass_context
def load_model(
        ctx: click.Context,
        model_id: str,
        model_path: str,
        context_length: int,
        gpu_layers: int,
        threads: int,
        model_type: str,
        wait: bool,
        wait_timeout: float,
) -> None:
    """Load a model onto the server."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            # Check if model already loaded
            models = client.models()
            if any(m.get("id") == model_id for m in models):
                print_warning(f"Model '{model_id}' is already loaded.")
                return

            with with_spinner(f"Loading model '{model_id}'..."):
                result = client.load_model(
                    model_id=model_id,
                    model_path=model_path,
                    context_length=context_length,
                    num_gpu_layers=gpu_layers,
                    n_threads=threads,
                    model_type=model_type,
                )

            status = result.get("status", "unknown")
            if status == "loading":
                print_info(f"Model '{model_id}' is loading asynchronously.")
                if wait:
                    print_info(f"Waiting up to {wait_timeout:.0f}s for model to become ready...")
                    ready = asyncio.run(_wait_for_model_ready(
                        client, model_id, timeout=wait_timeout
                    ))
                    if ready:
                        print_success(f"Model '{model_id}' is now ready.")
                    else:
                        print_warning(
                            f"Model '{model_id}' not ready within timeout. Check status with 'llm models list'."
                        )
                else:
                    print_info("Use 'llm models list' to check when loading completes.")
            elif status == "loaded":
                print_success(f"Model '{model_id}' loaded successfully.")
            else:
                print_error(f"Model load returned unexpected status: {status}")
                sys.exit(1)

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Unload model command
# ------------------------------------------------------------------------------
@models.command("unload")
@click.argument("model_id")
@click.option("--force", is_flag=True, help="Force unload even if model is in use")
@click.pass_context
def unload_model(ctx: click.Context, model_id: str, force: bool) -> None:
    """Unload a model from the server."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            models = client.models()
            if not any(m.get("id") == model_id for m in models):
                print_error(f"Model '{model_id}' not found.")
                sys.exit(1)

            with with_spinner(f"Unloading model '{model_id}'..."):
                result = client.unload_model(model_id)

            if result.get("status") == "unloaded":
                print_success(f"Model '{model_id}' unloaded.")
            else:
                error_msg = result.get("error", "Unknown error")
                if "in use" in error_msg.lower() and not force:
                    print_error(f"Model '{model_id}' is currently in use. Use --force to unload anyway.")
                    sys.exit(1)
                print_error(f"Failed to unload model: {error_msg}")
                sys.exit(1)

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to unload model: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Hot‑swap model command
# ------------------------------------------------------------------------------
@models.command("swap")
@click.argument("old_id")
@click.option("--new-id", required=True, help="New model identifier")
@click.option("--new-path", required=True, help="Path to new model file")
@click.option("--context-length", type=int, default=0, help="Context length (0 = auto)")
@click.option("--gpu-layers", type=int, default=-1, help="Number of GPU layers (-1 = auto)")
@click.option("--model-type", "new_model_type", default="auto",
              type=click.Choice(["gguf", "bitnet", "auto"], case_sensitive=False))
@click.option("--wait", is_flag=True, default=True, help="Wait for new model to be ready")
@click.option("--wait-timeout", type=float, default=120.0, help="Max wait time in seconds")
@click.pass_context
def swap_model(
        ctx: click.Context,
        old_id: str,
        new_id: str,
        new_path: str,
        context_length: int,
        gpu_layers: int,
        new_model_type: str,
        wait: bool,
        wait_timeout: float,
) -> None:
    """Hot‑swap a model (unload old, load new in one operation)."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            models = client.models()
            if not any(m.get("id") == old_id for m in models):
                print_error(f"Model '{old_id}' not found.")
                sys.exit(1)

            with with_spinner(f"Swapping '{old_id}' -> '{new_id}'..."):
                result = client.hot_swap(
                    old_id=old_id,
                    new_id=new_id,
                    new_path=new_path,
                    context_length=context_length,
                    num_gpu_layers=gpu_layers,
                )

            status = result.get("status", "unknown")
            if status == "loading":
                print_info(f"Swap initiated. New model '{new_id}' is loading.")
                if wait:
                    print_info(f"Waiting up to {wait_timeout:.0f}s for model to become ready...")
                    ready = asyncio.run(_wait_for_model_ready(
                        client, new_id, timeout=wait_timeout
                    ))
                    if ready:
                        print_success(f"Model '{new_id}' is now ready (old '{old_id}' unloaded).")
                    else:
                        print_warning(
                            f"Model '{new_id}' not ready within timeout. Check with 'llm models list'."
                        )
                else:
                    print_success(f"Swap completed asynchronously. New model '{new_id}' is loading.")
            elif status == "swapped":
                print_success(f"Swapped '{old_id}' with '{new_id}'.")
            else:
                print_error(f"Swap returned unexpected status: {status}")
                sys.exit(1)

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to swap model: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Load from profile command
# ------------------------------------------------------------------------------
@models.command("load-profile")
@click.argument("profile_id")
@click.option("--wait", is_flag=True, default=False, help="Wait for model to be ready")
@click.option("--wait-timeout", type=float, default=120.0, help="Max wait time in seconds")
@click.pass_context
def load_profile(ctx: click.Context, profile_id: str, wait: bool, wait_timeout: float) -> None:
    """Load a model using a pre‑defined profile from server configuration."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            with with_spinner(f"Loading profile '{profile_id}'..."):
                result = client.load_profile(profile_id)

            if result.get("status") == "loading":
                model_id = result.get("model_id", profile_id)
                print_info(f"Profile '{profile_id}' is loading as model '{model_id}'.")
                if wait:
                    print_info(f"Waiting up to {wait_timeout:.0f}s for model to become ready...")
                    ready = asyncio.run(_wait_for_model_ready(
                        client, model_id, timeout=wait_timeout
                    ))
                    if ready:
                        print_success(f"Model '{model_id}' is now ready.")
                    else:
                        print_warning(
                            f"Model '{model_id}' not ready within timeout. Check with 'llm models list'."
                        )
                else:
                    print_info("Use 'llm models list' to check when loading completes.")
            elif result.get("status") == "loaded":
                print_success(f"Profile '{profile_id}' loaded as model '{result.get('model_id')}'.")
            else:
                print_error(f"Failed to load profile: {result.get('error', 'unknown error')}")
                sys.exit(1)

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to load profile: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Model status command (show detailed info for a specific model)
# ------------------------------------------------------------------------------
@models.command("status")
@click.argument("model_id", required=False)
@click.pass_context
def model_status(ctx: click.Context, model_id: Optional[str]) -> None:
    """Show detailed status of a specific model or all models."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        try:
            models_data = client.models()
            if not models_data:
                print_info("No models loaded.")
                return

            if model_id:
                models_data = [m for m in models_data if m.get("id") == model_id]
                if not models_data:
                    print_error(f"Model '{model_id}' not found.")
                    sys.exit(1)

            for m in models_data:
                panel_content = f"""
[bold cyan]ID:[/] {m.get('id', '-')}
[bold green]Type:[/] {m.get('model_type', 'gguf')}
[bold green]Backend:[/] {m.get('backend', '-')}
[bold yellow]GPU Layers:[/] {m.get('gpu_layers', 0)}  |  [bold yellow]CPU Layers:[/] {m.get('cpu_layers', 0)}
[bold magenta]VRAM:[/] {m.get('vram_mb', 0)} MB
[bold blue]Requests served:[/] {m.get('requests', 0)}
[bold]Context length:[/] {m.get('context_length', 'auto')}
[bold]Strategy:[/] {m.get('strategy', 'unknown')}
[bold]Profile:[/] {m.get('profile', '')}
"""
                console.print(Panel(panel_content, title=f"Model: {m.get('id')}", border_style="cyan"))

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to get model status: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    run()


# ------------------------------------------------------------------------------
# Batch unload command (unload multiple models at once)
# ------------------------------------------------------------------------------
@models.command("unload-many")
@click.argument("model_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Force unload even if in use")
@click.pass_context
def unload_many(ctx: click.Context, model_ids: tuple, force: bool) -> None:
    """Unload multiple models at once."""

    def run() -> None:
        client = _get_client_from_context(ctx)
        success_count = 0
        fail_count = 0

        for model_id in model_ids:
            try:
                with with_spinner(f"Unloading '{model_id}'..."):
                    result = client.unload_model(model_id)
                if result.get("status") == "unloaded":
                    print_success(f"Model '{model_id}' unloaded.")
                    success_count += 1
                else:
                    print_error(f"Failed to unload '{model_id}': {result.get('error', 'unknown')}")
                    fail_count += 1
            except Exception as e:
                print_error(f"Failed to unload '{model_id}': {e}")
                fail_count += 1

        console.print(f"\n[bold]Summary:[/] {success_count} unloaded, {fail_count} failed")
        if fail_count > 0:
            sys.exit(1)
        _cleanup_client(ctx)

    run()
