"""
local_llm_client/cli/commands/config_cmd.py

Production‑grade configuration management commands for Local LLM Cloud.

All bugs fixed:
- Replaced all references to non‑existent attributes (``cfg.endpoint``)
  with the correct ``cfg.base_url`` and ``cfg.get(key)`` for all others.
- The ``show`` command no longer tries to iterate over a non‑existent
  ``__dataclass_fields__``; it uses the schema defined here.
- Added missing ``import os`` for the ``edit`` command.
- Removed unused ``import tempfile``.
- All sensitive keys are consistently masked.
- New subcommands: ``validate``, ``backup``, ``restore``.
- Extended schema to cover every key in the default ``Config``.
- Dot‑notation get/set validated against this schema.
- JSON output always available.
"""

import json
import os
import sys
import shutil
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from ..config import get_config, reload_config, Config
from ..utils import print_success, print_error, print_info, print_warning, confirm

console = Console()

# ------------------------------------------------------------------------------
# Full schema – every key that the default Config object recognises.
# Format: key -> (type, validator_fn_or_None, description)
# ------------------------------------------------------------------------------
CONFIG_SCHEMA: Dict[str, Tuple[str, Optional[Callable[[Any], bool]], str]] = {
    # ── Server ──
    "server.base_url":      ("str",   lambda x: bool(x.strip()),          "Server endpoint URL (e.g., http://localhost:8080)"),
    "server.api_key":       ("str",   None,                              "API key for authentication (set via LLM_API_KEY env var)"),
    "server.master_key":    ("str",   None,                              "Master key for admin endpoints (set via LLM_MASTER_KEY env var)"),
    "server.verify_ssl":    ("bool",  None,                              "Verify SSL certificates (disable only for testing)"),
    "server.ca_cert":       ("str",   None,                              "Optional custom CA certificate path"),

    # ── HTTP ──
    "http.pool_size":                 ("int",   lambda x: x >= 1,          "HTTP connection pool size"),
    "http.pool_timeout_s":            ("float", lambda x: x > 0,           "Max wait to acquire a pool slot (seconds)"),
    "http.connect_timeout_s":         ("float", lambda x: x > 0,           "TCP connect timeout (seconds)"),
    "http.read_timeout_s":            ("float", lambda x: x > 0,           "Per‑chunk read timeout for streaming (seconds)"),
    "http.total_timeout_s":           ("float", lambda x: x > 0,           "Total request timeout (seconds)"),
    "http.stream_timeout_s":          ("float", lambda x: x > 0,           "Streaming keep‑alive timeout (seconds)"),
    "http.stream_chunk_timeout_s":    ("float", lambda x: x > 0,           "Timeout between stream chunks (seconds)"),
    "http.max_retries":               ("int",   lambda x: x >= 0,          "Maximum retry attempts on transient errors"),
    "http.retry_delay_s":             ("float", lambda x: x > 0,           "Initial retry delay (seconds)"),
    "http.retry_delay_max_s":         ("float", lambda x: x > 0,           "Maximum retry delay cap (seconds)"),
    "http.retry_backoff_factor":      ("float", lambda x: x > 1.0,         "Exponential backoff multiplier"),
    "http.retry_jitter":              ("float", lambda x: 0.0 <= x <= 1.0, "Random jitter fraction"),
    "http.retryable_status_codes":    ("list",  None,                      "HTTP status codes that are retryable"),
    "http.http_keep_alive":           ("bool",  None,                      "Keep TCP connections alive between requests"),
    "http.http2_enabled":             ("bool",  None,                      "Enable HTTP/2 (experimental)"),

    # ── Circuit breaker ──
    "circuit_breaker.enabled":                   ("bool",  None,             "Enable circuit breaker"),
    "circuit_breaker.failure_threshold":         ("int",   lambda x: x >= 1, "Consecutive failures to open circuit"),
    "circuit_breaker.error_rate_threshold":      ("float", lambda x: 0.0 <= x <= 1.0, "Error rate threshold to open circuit"),
    "circuit_breaker.recovery_timeout_s":        ("float", lambda x: x > 0,  "Time before trying to recover (seconds)"),
    "circuit_breaker.half_open_probe_timeout_s": ("float", lambda x: x > 0,  "Timeout for probe in half‑open state"),
    "circuit_breaker.log_state_changes":         ("bool",  None,             "Log circuit breaker state changes"),
    "circuit_breaker.log_failures":              ("bool",  None,             "Log individual failures"),

    # ── Inference ──
    "inference.model":                       ("str",   None,                     "Default model name"),
    "inference.temperature":                 ("float", lambda x: 0.0 <= x <= 2.0, "Sampling temperature (0.0‑2.0)"),
    "inference.top_p":                       ("float", lambda x: 0.0 <= x <= 1.0, "Nucleus sampling top_p"),
    "inference.top_k":                       ("int",   lambda x: x >= 1,          "Top‑K sampling (≥1)"),
    "inference.max_tokens":                  ("int",   lambda x: x >= 0,          "Maximum tokens to generate (0 = unlimited)"),
    "inference.timeout_s":                   ("float", lambda x: x > 0,           "Inference timeout (seconds)"),
    "inference.stream_enabled":              ("bool",  None,                      "Enable streaming by default"),
    "inference.stream_flush_interval_ms":    ("int",   lambda x: x > 0,           "Word buffer flush interval (ms)"),
    "inference.stream_buffer_size":          ("str",   None,                      "Buffer size for streaming ('dynamic' or bytes)"),
    "inference.system_prompt":              ("str",   None,                      "Global system prompt (optional)"),

    # ── Health check ──
    "health_check.on_init_enabled":        ("bool",  None,                      "Check server health on client init"),
    "health_check.on_init_retries":        ("int",   lambda x: x >= 0,          "Health check retry attempts"),
    "health_check.on_init_retry_delay_s":  ("float", lambda x: x > 0,           "Delay between retries (seconds)"),
    "health_check.on_init_timeout_s":      ("float", lambda x: x > 0,           "Health endpoint timeout (seconds)"),
    "health_check.wait_for_model":         ("bool",  None,                      "Wait for model to be loaded on init"),
    "health_check.wait_timeout_s":         ("float", lambda x: x > 0,           "Max wait time for model (seconds)"),
    "health_check.periodic_enabled":       ("bool",  None,                      "Enable periodic background health checks"),
    "health_check.periodic_interval_s":    ("float", lambda x: x > 0,           "Interval between periodic checks (seconds)"),
    "health_check.log_health_failures":    ("bool",  None,                      "Log health check failures"),
    "health_check.fail_open":              ("bool",  None,                      "Continue if health check fails"),

    # ── Metrics ──
    "metrics.tps_estimate_fallback":   ("float", lambda x: x > 0,           "Fallback TPS if server doesn't report"),
    "metrics.tps_cache_ttl_s":         ("float", lambda x: x > 0,           "Cache TPS for this many seconds"),
    "metrics.base_timeout_padding_s":  ("float", lambda x: x >= 0,          "Padding added to calculated timeout"),
    "metrics.min_timeout_s":           ("float", lambda x: x > 0,           "Minimum allowed timeout"),
    "metrics.token_count_method":      ("str",   lambda x: x in ("fast","exact"), "Token counting method"),
    "metrics.token_estimate_multiplier": ("float", lambda x: x > 1.0,       "Words‑to‑tokens multiplier"),

    # ── Cache ──
    "cache.config_dir":             ("str", None, "Directory for config files (default ~/.llmcloud)"),
    "cache.config_file":            ("str", None, "Configuration file name"),
    "cache.api_key_cache_ttl_s":    ("float", lambda x: x > 0, "API key cache TTL (seconds)"),
    "cache.session_dir":            ("str", None, "Directory for conversation sessions"),
    "cache.session_retention_days": ("int",  lambda x: x >= 1, "Keep sessions for N days"),
    "cache.max_sessions":           ("int",  lambda x: x >= 1, "Maximum stored sessions"),

    # ── Security ──
    "security.validate_base_url":           ("bool", None, "Validate base URL format"),
    "security.validate_model_name":         ("bool", None, "Validate model name"),
    "security.validate_api_key_format":     ("bool", None, "Validate API key format"),
    "security.api_key_allowed_in_config":   ("bool", None, "Allow API keys in config file"),
    "security.api_key_from_env_only":       ("bool", None, "Forbid API keys in config file"),
    "security.min_tls_version":             ("str",  lambda x: x in ("TLSv1.2","TLSv1.3"), "Minimum TLS version"),
    "security.hash_sensitive_logs":         ("bool", None, "Hash sensitive values in logs"),

    # ── Logging ──
    "logging.level":                 ("str",  lambda x: x in ("debug","info","warning","error"), "Log level"),
    "logging.console_enabled":       ("bool", None, "Log to stderr"),
    "logging.file_enabled":          ("bool", None, "Write logs to file"),
    "logging.file_path":             ("str",  None, "Log file path"),
    "logging.file_max_mb":           ("int",  lambda x: x >= 1, "Max size before rotation (MB)"),
    "logging.file_backups":          ("int",  lambda x: x >= 0, "Number of rotated backups"),
    "logging.format":                ("str",  None, "Log format string"),
    "logging.date_format":           ("str",  None, "Date format for timestamps"),
    "logging.use_json":              ("bool", None, "Use JSON structured logging"),
    "logging.mask_api_keys":         ("bool", None, "Mask API keys in logs"),
    "logging.mask_prompts":          ("bool", None, "Truncate long prompts in logs"),
    "logging.log_http_requests":     ("bool", None, "Log raw HTTP requests"),
    "logging.log_http_responses":    ("bool", None, "Log raw HTTP responses"),
    "logging.log_retry_attempts":    ("bool", None, "Log retry attempts"),
    "logging.log_circuit_breaker":   ("bool", None, "Log circuit breaker state changes"),

    # ── Advanced ──
    "advanced.max_workers":                 ("int",   lambda x: x >= 1, "Thread pool size for blocking ops"),
    "advanced.asyncio_debug":               ("bool",  None, "Enable asyncio debug mode"),
    "advanced.prefer_aiohttp":              ("bool",  None, "Prefer aiohttp if installed"),
    "advanced.log_aiohttp_fallback":        ("bool",  None, "Log warning when falling back to sync"),
    "advanced.urllib3_pool_connections":    ("int",   lambda x: x >= 1, "urllib3 connection pool size"),
    "advanced.urllib3_pool_maxsize":        ("int",   lambda x: x >= 1, "urllib3 max pool size"),
    "advanced.enable_c_accelerator":        ("bool",  None, "Enable C accelerator for token counting"),
    "advanced.word_boundary_detection":     ("bool",  None, "Enable smart word‑based streaming"),
    "advanced.context_manager_singleton":   ("bool",  None, "Use singleton for AdaptiveContextManager"),
    "advanced.context_window_static":       ("bool",  None, "Do not query server for context window"),
    "advanced.context_window_fallback":     ("int",   lambda x: x >= 512, "Fallback context window size (tokens)"),
    "advanced.allow_prompt_compression":    ("bool",  None, "Allow automatic prompt compression"),
    "advanced.compression_threshold_ratio": ("float", lambda x: 0.0 < x <= 1.0, "Compression threshold ratio"),
    "advanced.auto_retry_on_timeout":       ("bool",  None, "Auto‑retry on timeout"),
    "advanced.auto_fallback_to_smaller_model": ("bool", None, "Fallback to smaller model on failure"),

    # ── CLI ──
    "cli.interactive_default_stream":    ("bool",  None, "Stream by default in interactive mode"),
    "cli.interactive_history_size":      ("int",   lambda x: x >= 1, "Messages kept per conversation"),
    "cli.chat_default_timeout_s":        ("float", lambda x: x >= 0, "Default timeout for chat command"),
    "cli.chat_always_show_thinking":     ("bool",  None, "Show thinking tags in assistant output"),
    "cli.models_refresh_interval_s":     ("float", lambda x: x > 0, "Model list refresh interval (seconds)"),
    "cli.status_refresh_interval_s":     ("float", lambda x: x > 0, "Status refresh interval (seconds)"),
    "cli.status_show_queue_depth":       ("bool",  None, "Show queue depth in status output"),

    # ── Adapter ──
    "adapter.use_async":                   ("bool",  None, "Use async client internally"),
    "adapter.enable_adaptive_context":     ("bool",  None, "Enable adaptive context management"),
    "adapter.context_recovery_enabled":    ("bool",  None, "Recover context window after success"),
    "adapter.context_reduction_factor":    ("float", lambda x: 0.0 < x < 1.0, "Reduce context by factor on timeout"),
    "adapter.context_recovery_factor":     ("float", lambda x: 1.0 < x <= 2.0, "Increase context after success"),
    "adapter.enable_compression":          ("bool",  None, "Enable prompt compression"),
    "adapter.compression_target_ratio":    ("float", lambda x: 0.0 < x <= 1.0, "Target compression ratio"),
    "adapter.allow_degradation":           ("bool",  None, "Allow quality degradation"),
    "adapter.max_degradation_attempts":    ("int",   lambda x: x >= 1, "Max degradation attempts"),
    "adapter.degradation_sleep_s":         ("float", lambda x: x > 0, "Sleep between degradation attempts"),
    "adapter.track_requests":              ("bool",  None, "Track request counts in metrics"),
    "adapter.track_errors":                ("bool",  None, "Track error counts"),
    "adapter.track_tokens":                ("bool",  None, "Track token counts"),
    "adapter.report_metrics":              ("bool",  None, "Include metrics in adapter responses"),
}

# Keys that should never be printed in plaintext.
SENSITIVE_KEYS = {
    "server.api_key",
    "server.master_key",
}


def _get_config_from_context(ctx: click.Context) -> Config:
    """Retrieve Config instance using the config path from Click context."""
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    return get_config(config_path)


def _validate_value(key: str, value: Any) -> bool:
    if key not in CONFIG_SCHEMA:
        return True  # unknown keys are allowed
    _, validator, _ = CONFIG_SCHEMA[key]
    if validator is None:
        return True
    try:
        return validator(value)
    except Exception:
        return False


def _coerce_value(key: str, value: str, value_type: str) -> Any:
    """Coerce string value to the expected type."""
    if value_type == "bool":
        return value.lower() in ("true", "1", "yes", "on")
    elif value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    elif value_type == "str":
        return value
    elif value_type == "list":
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def _get_value_type(key: str) -> str:
    if key in CONFIG_SCHEMA:
        return CONFIG_SCHEMA[key][0]
    return "str"


def _mask_value(key: str, value: Any) -> str:
    if key in SENSITIVE_KEYS:
        return "***MASKED***"
    return str(value)


# ------------------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------------------
@click.group("config")
@click.pass_context
def config_cmd(ctx: click.Context) -> None:
    """Manage CLI configuration (stored in YAML file)."""
    ctx.obj = ctx.obj or {}
    ctx.obj["config_obj"] = _get_config_from_context(ctx)


# ------------------------------------------------------------------------------
# Show configuration
# ------------------------------------------------------------------------------
@config_cmd.command("show")
@click.option("--all", "show_all", is_flag=True, help="Show all settings (including nested)")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def show_config(ctx: click.Context, show_all: bool, output_json: bool) -> None:
    """Show current configuration (sensitive values hidden)."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    if output_json:
        safe = {}
        for key in CONFIG_SCHEMA:
            val = cfg.get(key)
            safe[key] = _mask_value(key, val)
        console.print(json.dumps(safe, indent=2, default=str))
        return

    if show_all:
        console.print("[bold]Full configuration:[/]")
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for key in sorted(CONFIG_SCHEMA.keys()):
            val = cfg.get(key)
            table.add_row(key, _mask_value(key, val))
        console.print(table)
    else:
        console.print("[bold]Current configuration:[/]")
        console.print(f"  Config file:     {cfg.config_path}")
        console.print(f"  Server URL:      {cfg.get('server.base_url', 'not set')}")
        console.print(f"  API Key:         {'[hidden]' if cfg.get('server.api_key') else '[not set]'}")
        console.print(f"  Master Key:      {'[hidden]' if cfg.get('server.master_key') else '[not set]'}")
        console.print(f"  Default Model:   {cfg.get('inference.model', '[not set]')}")
        console.print(f"  Temperature:     {cfg.get('inference.temperature', 0.7)}")
        console.print(f"  Max Tokens:      {cfg.get('inference.max_tokens', 0)}")
        console.print(f"  Streaming:       {cfg.get('inference.stream_enabled', True)}")
        console.print(f"  Timeout:         {cfg.get('inference.timeout_s', 1800)}s")
        console.print(f"  Max Retries:     {cfg.get('http.max_retries', 3)}")
        console.print(f"  Pool Size:       {cfg.get('http.pool_size', 50)}")


# ------------------------------------------------------------------------------
# Get a single value
# ------------------------------------------------------------------------------
@config_cmd.command("get")
@click.argument("key")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def get_config_value(ctx: click.Context, key: str, output_json: bool) -> None:
    """
    Get a specific configuration value using dot notation.

    Examples:
        llm config get server.base_url
        llm config get http.pool_size
    """
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    value = cfg.get(key)
    if value is None:
        print_error(f"Key '{key}' not found")
        sys.exit(1)

    if output_json:
        console.print(json.dumps({key: _mask_value(key, value)}, indent=2, default=str))
        return

    console.print(_mask_value(key, value))


# ------------------------------------------------------------------------------
# Set a value
# ------------------------------------------------------------------------------
@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--type", "value_type", type=click.Choice(["string", "int", "float", "bool", "list"]),
              default="string", help="Coerce value to type (auto‑detected if possible)")
@click.option("--force", "-f", is_flag=True, help="Skip validation")
@click.pass_context
def set_config(ctx: click.Context, key: str, value: str, value_type: str, force: bool) -> None:
    """
    Set a configuration value using dot notation.

    Examples:
        llm config set server.base_url http://localhost:8080
        llm config set http.pool_size 100 --type int
        llm config set inference.stream_enabled true --type bool
    """
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    # Auto‑detect type if not specified
    if value_type == "string":
        if value.lower() in ("true", "false", "yes", "no", "on", "off"):
            value_type = "bool"
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            value_type = "int"
        elif value.replace(".", "").replace("-", "").isdigit() and value.count(".") <= 1:
            value_type = "float"
        elif "," in value:
            value_type = "list"

    try:
        coerced = _coerce_value(key, value, value_type)
    except ValueError:
        print_error(f"Cannot convert '{value}' to {value_type}")
        sys.exit(1)

    if not force and not _validate_value(key, coerced):
        expected_type, _, desc = CONFIG_SCHEMA.get(key, (value_type, None, ""))
        print_error(f"Invalid value for '{key}'")
        if desc:
            print_info(f"Description: {desc}")
        if expected_type:
            print_info(f"Expected type: {expected_type}")
        sys.exit(1)

    cfg.set(key, coerced)
    cfg.save()
    print_success(f"Configuration updated: {key} = {_mask_value(key, coerced)}")


# ------------------------------------------------------------------------------
# Unset a value (revert to default)
# ------------------------------------------------------------------------------
@config_cmd.command("unset")
@click.argument("key")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def unset_config(ctx: click.Context, key: str, force: bool) -> None:
    """
    Remove a configuration value (revert to default).

    Examples:
        llm config unset server.base_url
        llm config unset server.api_key
    """
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    if cfg.get(key) is None:
        print_error(f"Key '{key}' not found")
        sys.exit(1)

    if not force and not confirm(f"Reset '{key}' to its default value?", default=False):
        print_info("Operation cancelled.")
        return

    cfg.set(key, None)
    cfg.save()
    print_success(f"Configuration key removed: {key} (will use default)")


# ------------------------------------------------------------------------------
# List all available keys
# ------------------------------------------------------------------------------
@config_cmd.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show full descriptions")
@click.pass_context
def list_config_keys(ctx: click.Context, verbose: bool) -> None:
    """List all available configuration keys with descriptions."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    table = Table(title="Configuration Keys", box=box.ROUNDED)
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Current Value", style="yellow")
    if verbose:
        table.add_column("Description", style="dim")

    for key, (typ, _, desc) in sorted(CONFIG_SCHEMA.items()):
        current = cfg.get(key)
        table.add_row(key, typ, _mask_value(key, current), desc if verbose else "")
    console.print(table)
    print_info("Use 'llm config get <key>' to see a value")
    print_info("Use 'llm config set <key> <value> --type <type>' to change")


# ------------------------------------------------------------------------------
# Validate entire configuration
# ------------------------------------------------------------------------------
@config_cmd.command("validate")
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """Check current configuration for errors or warnings."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    errors = []
    warnings = []
    for key, (typ, validator, desc) in CONFIG_SCHEMA.items():
        val = cfg.get(key)
        if val is None:
            continue  # optional
        # Type check
        if typ == "int" and not isinstance(val, int):
            errors.append(f"{key}: expected int, got {type(val).__name__}")
        elif typ == "float" and not isinstance(val, (int, float)):
            errors.append(f"{key}: expected float, got {type(val).__name__}")
        elif typ == "bool" and not isinstance(val, bool):
            errors.append(f"{key}: expected bool, got {type(val).__name__}")
        elif typ == "str" and not isinstance(val, str):
            errors.append(f"{key}: expected str, got {type(val).__name__}")
        elif typ == "list" and not isinstance(val, list):
            errors.append(f"{key}: expected list, got {type(val).__name__}")
        if validator and val is not None:
            try:
                if not validator(val):
                    errors.append(f"{key}: value {val!r} failed validation ({desc})")
            except Exception:
                warnings.append(f"{key}: could not run validator")

    if errors:
        print_error("Configuration errors found:")
        for e in errors:
            console.print(f"  [red]✗[/] {e}")
        sys.exit(1)
    elif warnings:
        print_warning("Configuration warnings:")
        for w in warnings:
            console.print(f"  [yellow]⚠[/] {w}")
        print_success("No critical errors.")
    else:
        print_success("Configuration is valid.")


# ------------------------------------------------------------------------------
# Backup configuration
# ------------------------------------------------------------------------------
@config_cmd.command("backup")
@click.option("--output", "-o", default=None, help="Output file (default: auto‑named in config dir)")
@click.pass_context
def backup_config(ctx: click.Context, output: Optional[str]) -> None:
    """Create a timestamped backup of the current configuration."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    src = cfg.config_path
    if not src.exists():
        print_error("No configuration file to backup")
        sys.exit(1)

    dest = Path(output) if output else src.parent / f"config-backup-{datetime.now():%Y%m%d_%H%M%S}.yaml"
    try:
        shutil.copy2(src, dest)
        print_success(f"Backup saved to {dest}")
    except Exception as e:
        print_error(f"Backup failed: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Restore configuration from backup
# ------------------------------------------------------------------------------
@config_cmd.command("restore")
@click.argument("backup_file", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def restore_config(ctx: click.Context, backup_file: str, force: bool) -> None:
    """Restore configuration from a backup file."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    src = Path(backup_file)
    if not src.exists():
        print_error(f"Backup file not found: {backup_file}")
        sys.exit(1)

    if not force and not confirm(f"Overwrite current configuration with {src.name}?", default=False):
        print_info("Operation cancelled.")
        return

    try:
        shutil.copy2(src, cfg.config_path)
        reload_config(cfg.config_path)
        print_success("Configuration restored and reloaded")
    except Exception as e:
        print_error(f"Restore failed: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Reload from disk
# ------------------------------------------------------------------------------
@config_cmd.command("reload")
@click.pass_context
def reload_config_cmd(ctx: click.Context) -> None:
    """Reload configuration from disk (useful after manual edits)."""
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    try:
        reload_config(config_path)
        print_success("Configuration reloaded")
    except Exception as e:
        print_error(f"Failed to reload config: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Show file path
# ------------------------------------------------------------------------------
@config_cmd.command("path")
@click.pass_context
def show_config_path(ctx: click.Context) -> None:
    """Show the configuration file path being used."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)
    console.print(str(cfg.config_path))


# ------------------------------------------------------------------------------
# Edit in external editor
# ------------------------------------------------------------------------------
@config_cmd.command("edit")
@click.pass_context
def edit_config(ctx: click.Context) -> None:
    """Open the configuration file in the default editor."""
    import subprocess

    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(cfg.config_path)], check=True)
        reload_config(cfg.config_path)
        print_success("Configuration reloaded after edit")
    except FileNotFoundError:
        print_error(f"Editor '{editor}' not found. Set $EDITOR environment variable.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to open editor: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Export (secrets excluded)
# ------------------------------------------------------------------------------
@config_cmd.command("export")
@click.argument("output_file", type=click.Path())
@click.pass_context
def export_config(ctx: click.Context, output_file: str) -> None:
    """Export current configuration to a YAML file (secrets excluded)."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    export_data = {}
    for key in sorted(CONFIG_SCHEMA.keys()):
        if key in SENSITIVE_KEYS:
            continue
        val = cfg.get(key)
        if val is not None:
            # build nested dicts from dot notation
            parts = key.split(".")
            d = export_data
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        print_success(f"Configuration exported to {output_file}")
    except Exception as e:
        print_error(f"Failed to export config: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Import (merge)
# ------------------------------------------------------------------------------
@config_cmd.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def import_config(ctx: click.Context, input_file: str) -> None:
    """Import configuration from a YAML file (merges with current)."""
    cfg = ctx.obj.get("config_obj")
    if not cfg:
        print_error("Configuration not loaded")
        sys.exit(1)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested dicts and set keys
        def _set_nested(prefix: str, node: Any) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    _set_nested(f"{prefix}.{k}" if prefix else k, v)
            else:
                cfg.set(prefix, node)

        _set_nested("", data)
        cfg.save()
        print_success(f"Configuration imported from {input_file}")
    except Exception as e:
        print_error(f"Failed to import config: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Reset to defaults
# ------------------------------------------------------------------------------
@config_cmd.command("reset")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def reset_config_cmd(ctx: click.Context, force: bool) -> None:
    """Reset configuration to factory defaults (⚠ removes all custom settings)."""
    if not force:
        if not confirm("⚠ This will reset ALL configuration to defaults. Continue?", default=False):
            print_info("Operation cancelled.")
            return
        if not confirm("Type 'yes' to confirm", default=False):
            print_info("Operation cancelled.")
            return

    config_path = ctx.obj.get("config_path") if ctx.obj else None
    if config_path and config_path.exists():
        try:
            config_path.unlink()
        except Exception as e:
            print_error(f"Failed to delete config file: {e}")
            sys.exit(1)
    reload_config(config_path)
    print_success("Configuration reset to defaults")
