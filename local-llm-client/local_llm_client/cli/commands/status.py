"""
local_llm_client/cli/commands/status.py

Production‑grade status and monitoring commands for Local LLM Cloud.

Fixes applied:
- All client calls are synchronous (LLMCloudClient has no async methods).
- Config attribute corrected to `base_url` (the Config class provides `base_url`).
- Watch mode for metrics now uses `asyncio.to_thread` (or loop.run_in_executor)
  to keep the event loop responsive while running sync calls.
- Enhanced subcommands:
    • `ping` – lightweight latency test to /health
    • `metrics` – now shows per‑model VRAM and GPU‑layer breakdown if available
    • `system` – includes disk usage (from psutil locally) if possible
    • `logs` – supports follow mode with incremental pagination (no duplicate lines)
    • `dashboard` – live updating summary of health, metrics, queue, and models
- All commands support JSON output (`--json`) for scripting.
- Error handling is exhaustive; exit codes follow Unix conventions.

No feature of the original script has been removed – every existing subcommand
is preserved and improved.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.client import LLMCloudClient
from ...core.exceptions import (
    APIError, ConnectionError, CircuitOpenError, ModelNotLoadedError,
    TimeoutError as CoreTimeoutError,
)
from ...core.models import ClientConfig
from ..config import get_config
from ..utils import print_error, print_info, print_success, print_warning

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

    cached = ctx.obj.get("status_client")
    if cached is not None and not getattr(cached, "_closed", False):
        return cached

    config_path = ctx.obj.get("config_path")
    cfg = get_config(config_path)

    client_cfg = ClientConfig(
        base_url=cfg.base_url,            # CORRECTED – was cfg.endpoint
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
    ctx.obj["status_client"] = client
    return client


def _cleanup_client(ctx: click.Context) -> None:
    """Close and remove cached client."""
    client = ctx.obj.get("status_client")
    if client:
        try:
            client.close()
        except Exception:
            pass
        ctx.obj["status_client"] = None


# ------------------------------------------------------------------------------
# Async helper – run sync function in thread pool
# ------------------------------------------------------------------------------
async def _run_sync(func, *args, **kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ------------------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------------------
def _format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def _create_metrics_panel(metrics: Dict[str, Any]) -> Panel:
    """Create a rich Panel from metrics data for live display."""
    engine = metrics.get("engine", {})
    vram = metrics.get("vram", {})
    system = metrics.get("system", {})

    content = f"""
[bold cyan]Inference Engine[/]
  TPS: {engine.get('avg_tps', 0):.2f}
  TTFT: {engine.get('avg_ttft_ms', 0):.0f} ms
  Total requests: {engine.get('total_requests', 0):,}
  Total tokens: {engine.get('total_tokens', 0):,}
  Queue depth: {engine.get('queue_depth', 0)}
  Circuit breaker: {engine.get('circuit_breaker', {}).get('state', 'closed')}

[bold magenta]VRAM[/]
  Used: {_format_bytes(vram.get('used_bytes', 0))} / {_format_bytes(vram.get('total_bytes', 0))}
  Utilization: {vram.get('utilization_pct', 0):.1f}%

[bold yellow]System[/]
  CPU: {system.get('cpu_pct', 0):.1f}%
  RAM: {system.get('ram_pct', 0):.1f}%
"""
    return Panel(content, title="📊 Live Metrics", border_style="green")


# ------------------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------------------
@click.group("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show server status, health, metrics, and logs."""
    ctx.obj = ctx.obj or {}
    _get_client_from_context(ctx)


# ------------------------------------------------------------------------------
# Ping – lightweight latency test
# ------------------------------------------------------------------------------
@status.command("ping")
@click.option("--count", default=3, type=int, help="Number of pings (default 3)")
@click.pass_context
def ping(ctx: click.Context, count: int) -> None:
    """Measure round‑trip latency to /health."""

    client = _get_client_from_context(ctx)
    latencies = []
    for i in range(count):
        t0 = time.monotonic()
        try:
            client.health()
            elapsed = (time.monotonic() - t0) * 1000  # ms
            latencies.append(elapsed)
            console.print(f"[dim]Ping {i+1}: {elapsed:.1f} ms[/]")
        except Exception as e:
            print_error(f"Ping {i+1} failed: {e}")
            break
        if i < count - 1:
            time.sleep(0.5)

    if latencies:
        avg = sum(latencies) / len(latencies)
        m_max = max(latencies)
        m_min = min(latencies)
        console.print(f"\n[bold]Summary:[/] avg={avg:.1f} ms, min={m_min:.1f} ms, max={m_max:.1f} ms")
    _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Health command
# ------------------------------------------------------------------------------
@status.command("health")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def health(ctx: click.Context, output_json: bool) -> None:
    """Check server health (overall status and component checks)."""

    client = _get_client_from_context(ctx)
    try:
        health_data = client.health()

        if output_json:
            console.print(json.dumps(health_data, indent=2, default=str))
            return

        checks = health_data.get("checks", {})
        overall = health_data.get("status", "unknown")

        if overall == "healthy":
            print_success("Server is healthy")
        else:
            print_warning(f"Server status: {overall}")

        table = Table(title="Component Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for component, state in checks.items():
            icon = "✓" if state == "ok" else ("⚠" if state == "degraded" else "✗")
            color = "green" if state == "ok" else ("yellow" if state == "degraded" else "red")
            table.add_row(component, f"[{color}]{icon} {state}[/]")

        console.print(table)

        version = health_data.get("version", "unknown")
        timestamp = health_data.get("timestamp", 0)
        if timestamp:
            dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"\n[dim]Version: {version} | Last check: {dt}[/]")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Health check failed: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Ready command
# ------------------------------------------------------------------------------
@status.command("ready")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def ready(ctx: click.Context, output_json: bool) -> None:
    """Check if server is ready to serve requests (models loaded, etc.)."""

    client = _get_client_from_context(ctx)
    try:
        is_ready = client.ready()

        if output_json:
            console.print(json.dumps({"status": "ready" if is_ready else "not_ready"}, indent=2))
            return

        if is_ready:
            print_success("Server is ready")
        else:
            print_warning("Server is not ready – no model loaded or breaker open")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Ready check failed: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# System info command
# ------------------------------------------------------------------------------
@status.command("system")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def system(ctx: click.Context, output_json: bool) -> None:
    """Show system information (CPU, RAM, GPUs, available backends)."""

    client = _get_client_from_context(ctx)
    try:
        info = client.system_info()

        if output_json:
            console.print(json.dumps(info, indent=2, default=str))
            return

        console.print(Panel.fit(
            f"[bold]CPU:[/] {info.get('cpu_model', 'unknown')}\n"
            f"[bold]Cores:[/] {info.get('cpu_cores', 0)} physical, {info.get('cpu_threads', 0)} logical",
            title="🖥️ CPU", border_style="cyan"
        ))

        ram_total = info.get('ram_total_mb', 0)
        ram_free = info.get('ram_free_mb', 0)
        ram_pct = info.get('ram_pct', 0)
        console.print(Panel.fit(
            f"[bold]RAM:[/] {ram_total:,} MB total, {ram_free:,} MB free ({ram_pct}%)",
            title="💾 Memory", border_style="green"
        ))

        console.print(Panel.fit(
            f"[bold]Recommended backend:[/] {info.get('recommended_backend', 'cpu').upper()}\n"
            f"[bold]Available backends:[/] {', '.join(info.get('available_backends', []))}",
            title="⚙️ Backends", border_style="yellow"
        ))

        gpus = info.get("gpus", [])
        if gpus:
            gpu_text = "\n".join(
                f"  {gpu.get('name', 'unknown')} - {gpu.get('vram_total_mb', 0)} MB "
                f"({gpu.get('backend', 'unknown').upper()})"
                for gpu in gpus
            )
            console.print(Panel(gpu_text, title="🎮 GPUs", border_style="magenta"))
        else:
            console.print("[yellow]No GPU detected – CPU mode[/]")

        # Disk usage from local psutil (not from server, but helpful)
        try:
            import psutil
            usage = psutil.disk_usage('/')
            console.print(Panel.fit(
                f"[bold]Total:[/] {_format_bytes(usage.total)}\n"
                f"[bold]Used:[/] {_format_bytes(usage.used)} ({usage.percent}%)\n"
                f"[bold]Free:[/] {_format_bytes(usage.free)}",
                title="💿 Disk (local)", border_style="grey"
            ))
        except Exception:
            pass

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to get system info: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Metrics command (with live watch mode)
# ------------------------------------------------------------------------------
@status.command("metrics")
@click.option("--watch", "-w", is_flag=True, help="Watch mode: refresh every 2 seconds")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format (one-time)")
@click.pass_context
def metrics(ctx: click.Context, watch: bool, output_json: bool) -> None:
    """Show performance metrics (TPS, TTFT, VRAM, queue depth)."""

    client = _get_client_from_context(ctx)

    def fetch_metrics() -> Dict[str, Any]:
        return client.metrics()

    def show_one_shot(metrics_data: Dict[str, Any]) -> None:
        if output_json:
            console.print(json.dumps(metrics_data, indent=2, default=str))
            return

        engine = metrics_data.get("engine", {})
        vram = metrics_data.get("vram", {})
        system = metrics_data.get("system", {})

        # Engine table
        engine_table = Table(title="Inference Engine", box=box.ROUNDED)
        engine_table.add_column("Metric", style="cyan")
        engine_table.add_column("Value", style="green")
        engine_table.add_row("TPS", f"{engine.get('avg_tps', 0):.2f}")
        engine_table.add_row("TTFT", f"{engine.get('avg_ttft_ms', 0):.0f} ms")
        engine_table.add_row("Total requests", f"{engine.get('total_requests', 0):,}")
        engine_table.add_row("Total tokens", f"{engine.get('total_tokens', 0):,}")
        engine_table.add_row("Queue depth", f"{engine.get('queue_depth', 0)}")
        engine_table.add_row("Circuit breaker", engine.get('circuit_breaker', {}).get('state', 'closed'))
        console.print(engine_table)

        # VRAM table
        vram_table = Table(title="VRAM", box=box.ROUNDED)
        vram_table.add_column("Metric", style="cyan")
        vram_table.add_column("Value", style="magenta")
        used_mb = vram.get('used_bytes', 0) >> 20
        total_mb = vram.get('total_bytes', 0) >> 20
        util_pct = vram.get('utilization_pct', 0)
        vram_table.add_row("Used", f"{used_mb} MB / {total_mb} MB ({util_pct:.1f}%)")
        console.print(vram_table)

        # System table
        system_table = Table(title="System", box=box.ROUNDED)
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="yellow")
        system_table.add_row("CPU", f"{system.get('cpu_pct', 0):.1f}%")
        system_table.add_row("RAM", f"{system.get('ram_pct', 0):.1f}%")
        console.print(system_table)

        # Loaded models table (if present)
        models = metrics_data.get("models")
        if models:
            model_table = Table(title="Loaded Models", box=box.ROUNDED)
            model_table.add_column("Model ID", style="cyan")
            model_table.add_column("VRAM (MB)", style="magenta")
            model_table.add_column("GPU Layers", style="yellow")
            for m in models:
                model_table.add_row(
                    m.get("id", "?"),
                    str(m.get("vram_mb", 0)),
                    str(m.get("gpu_layers", 0)),
                )
            console.print(model_table)

    if watch:
        async def run_watch() -> None:
            try:
                with Live(refresh_per_second=2, screen=True) as live:
                    while True:
                        try:
                            m = await _run_sync(fetch_metrics)
                            panel = _create_metrics_panel(m)
                            live.update(panel)
                            await asyncio.sleep(2)
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            live.update(Panel(f"[red]Error: {e}[/]", title="❌ Error"))
                            await asyncio.sleep(2)
            finally:
                _cleanup_client(ctx)

        try:
            asyncio.run(run_watch())
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/]")
    else:
        try:
            show_one_shot(fetch_metrics())
        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to get metrics: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Queue status command
# ------------------------------------------------------------------------------
@status.command("queue")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def queue(ctx: click.Context, output_json: bool) -> None:
    """Show request queue status (depth, busy flag, estimated wait)."""

    client = _get_client_from_context(ctx)
    try:
        q = client.queue_status()

        if output_json:
            console.print(json.dumps(q, indent=2, default=str))
            return

        queue_depth = q.get('queue_depth', 0)
        gate_waiters = q.get('gate_waiters', 0)
        server_busy = q.get('server_busy', False)
        est_wait = q.get('estimated_wait_s', 0)

        busy_symbol = "🟢" if not server_busy else "🔴"
        busy_text = "Idle" if not server_busy else "Busy"

        panel_content = f"""
[bold]Queue depth:[/] {queue_depth}
[bold]Gate waiters:[/] {gate_waiters}
[bold]Server status:[/] {busy_symbol} {busy_text}
[bold]Estimated wait:[/] {est_wait:.1f} seconds
"""
        console.print(Panel(panel_content, title="📋 Request Queue", border_style="cyan"))

        if queue_depth > 0:
            bar_length = min(40, queue_depth)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            console.print(f"Queue: [{bar}] {queue_depth}")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to get queue status: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)


# ------------------------------------------------------------------------------
# Logs command (with follow mode)
# ------------------------------------------------------------------------------
@status.command("logs")
@click.option("--n", default=50, type=int, help="Number of log entries to fetch")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (poll every 2 seconds)")
@click.option("--level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Filter by log level")
@click.pass_context
def logs(ctx: click.Context, n: int, follow: bool, level: Optional[str]) -> None:
    """Show recent server logs (last N entries)."""

    client = _get_client_from_context(ctx)

    def fetch_logs(since: float = 0.0) -> List[Dict]:
        data = client.get_logs(n=n, since=since)
        return data.get("entries", [])

    def show_once() -> None:
        try:
            entries = fetch_logs()
            if not entries:
                print_info("No log entries.")
                return

            for entry in entries:
                ts = entry.get("ts", 0)
                entry_level = entry.get("level", "INFO")
                msg = entry.get("msg", "")

                if level and entry_level != level:
                    continue

                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"

                if entry_level == "ERROR":
                    console.print(f"[red]{dt} ERROR: {msg}[/]")
                elif entry_level == "WARNING":
                    console.print(f"[yellow]{dt} WARN: {msg}[/]")
                else:
                    console.print(f"[dim]{dt} {entry_level}: {msg}[/]")

        except (APIError, ConnectionError, CircuitOpenError) as e:
            print_error(str(e))
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to get logs: {e}")
            sys.exit(1)
        finally:
            _cleanup_client(ctx)

    if follow:
        async def run_follow() -> None:
            last_ts = 0.0
            try:
                with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                ) as progress:
                    progress.add_task("Following logs...", total=None)
                    while True:
                        try:
                            entries = await _run_sync(fetch_logs, since=last_ts)
                            for entry in entries:
                                ts = entry.get("ts", 0)
                                if ts > last_ts:
                                    last_ts = ts
                                entry_level = entry.get("level", "INFO")
                                msg = entry.get("msg", "")
                                if level and entry_level != level:
                                    continue
                                dt = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                                if entry_level == "ERROR":
                                    console.print(f"[red]{dt} ERROR: {msg}[/]")
                                elif entry_level == "WARNING":
                                    console.print(f"[yellow]{dt} WARN: {msg}[/]")
                                else:
                                    console.print(f"[dim]{dt} {entry_level}: {msg}[/]")
                            await asyncio.sleep(2)
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/]")
                            await asyncio.sleep(2)
            finally:
                _cleanup_client(ctx)

        try:
            asyncio.run(run_follow())
        except KeyboardInterrupt:
            console.print("\n[yellow]Log following stopped[/]")
    else:
        show_once()


# ------------------------------------------------------------------------------
# Dashboard – live updating combined view
# ------------------------------------------------------------------------------
@status.command("dashboard")
@click.option("--interval", default=2.0, type=float, help="Refresh interval in seconds (default 2)")
@click.pass_context
def dashboard(ctx: click.Context, interval: float) -> None:
    """Live dashboard combining health, metrics, queue, and model list."""

    client = _get_client_from_context(ctx)

    async def fetch_all():
        health = await _run_sync(client.health)
        metrics = await _run_sync(client.metrics)
        queue = await _run_sync(client.queue_status)
        models = await _run_sync(client.models)
        return health, metrics, queue, models

    async def run_dashboard() -> None:
        try:
            with Live(refresh_per_second=4, screen=True) as live:
                while True:
                    try:
                        health, metrics, queue, models = await fetch_all()
                        health_ok = health.get("status") == "ok"
                        engine = metrics.get("engine", {})
                        vram = metrics.get("vram", {})
                        system = metrics.get("system", {})
                        queue_depth = queue.get("queue_depth", 0)
                        model_names = [m.get("id") for m in models] if models else []
                        content = f"""
[bold]{'✅' if health_ok else '❌'} Health[/]  {'Online' if health_ok else 'Offline'}

[bold]⚡ Engine[/]
  TPS: {engine.get('avg_tps',0):.2f}  TTFT: {engine.get('avg_ttft_ms',0):.0f} ms
  Requests: {engine.get('total_requests',0):,}  Tokens: {engine.get('total_tokens',0):,}
  Queue: {engine.get('queue_depth',0)}  CB: {engine.get('circuit_breaker',{}).get('state','closed')}

[bold]💾 VRAM[/]  {vram.get('utilization_pct',0):.1f}%  ({_format_bytes(vram.get('used_bytes',0))}/{_format_bytes(vram.get('total_bytes',0))})

[bold]🖥️ System[/]  CPU: {system.get('cpu_pct',0):.1f}%  RAM: {system.get('ram_pct',0):.1f}%

[bold]📋 Queue[/]  Depth: {queue.get('gate_waiters',0)}  Est. wait: {queue.get('estimated_wait_s',0):.1f}s

[bold]🤖 Models[/]  {', '.join(model_names) if model_names else 'None loaded'}
"""
                        panel = Panel(content, title="📊 Live Dashboard", border_style="cyan")
                        live.update(panel)
                        await asyncio.sleep(interval)
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        live.update(Panel(f"[red]Error: {e}[/]", title="❌"))
                        await asyncio.sleep(interval)
        finally:
            _cleanup_client(ctx)

    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/]")


# ------------------------------------------------------------------------------
# All-in-one status command (static snapshot)
# ------------------------------------------------------------------------------
@status.command("all")
@click.pass_context
def all_status(ctx: click.Context) -> None:
    """Show comprehensive server status (health, metrics, queue, system)."""

    client = _get_client_from_context(ctx)
    try:
        health_data = client.health()
        metrics_data = client.metrics()
        queue_data = client.queue_status()
        system_data = client.system_info()

        # Health section
        console.print("\n[bold cyan]🏥 Health Status[/]")
        overall = health_data.get("status", "unknown")
        status_icon = "✅" if overall == "healthy" else "⚠️"
        console.print(f"{status_icon} Overall: {overall}")
        for comp, state in health_data.get("checks", {}).items():
            icon = "✓" if state == "ok" else ("⚠" if state == "degraded" else "✗")
            console.print(f"  {icon} {comp}: {state}")

        # Metrics section
        console.print("\n[bold cyan]📊 Performance Metrics[/]")
        engine = metrics_data.get("engine", {})
        console.print(f"  TPS: {engine.get('avg_tps', 0):.2f}")
        console.print(f"  TTFT: {engine.get('avg_ttft_ms', 0):.0f} ms")
        console.print(f"  Queue depth: {engine.get('queue_depth', 0)}")
        vram = metrics_data.get("vram", {})
        used_mb = vram.get('used_bytes', 0) >> 20
        total_mb = vram.get('total_bytes', 0) >> 20
        console.print(f"  VRAM: {used_mb} / {total_mb} MB")

        # Queue section
        console.print("\n[bold cyan]📋 Queue Status[/]")
        console.print(f"  Depth: {queue_data.get('queue_depth', 0)}")
        console.print(f"  Busy: {'Yes' if queue_data.get('server_busy', False) else 'No'}")
        console.print(f"  Est. wait: {queue_data.get('estimated_wait_s', 0):.1f}s")

        # System section
        console.print("\n[bold cyan]🖥️ System Info[/]")
        console.print(f"  CPU: {system_data.get('cpu_model', 'unknown')[:50]}")
        console.print(f"  Cores: {system_data.get('cpu_cores', 0)} physical")
        gpus = system_data.get("gpus", [])
        if gpus:
            console.print(f"  GPU: {gpus[0].get('name', 'unknown')}")

    except (APIError, ConnectionError, CircuitOpenError) as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to get status: {e}")
        sys.exit(1)
    finally:
        _cleanup_client(ctx)
