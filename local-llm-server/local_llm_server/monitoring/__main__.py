#!/usr/bin/env python3
"""
local_llm_server/monitoring/__main__.py — Production Entry Point v1.0.0
=======================================================================
Universal entry point for the Local LLM Cloud monitoring ecosystem.

v1.0.0 fixes:
  • Pre‑flight now warns if health_status is not "ok".
  • Configurable pre‑flight timeout.
  • All missing env var expansions added.
  • Robust JSON output for errors.
  • Reduced code duplication (points to shared config loader – still duplicated but documented).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Path setup – ensure the package root is on sys.path
# ------------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# ------------------------------------------------------------------------------
# Temporary sys.argv context manager
# ------------------------------------------------------------------------------
@contextmanager
def temporary_argv(new_argv: List[str]):
    """Temporarily replace sys.argv, restoring it on exit."""
    original = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original


# ------------------------------------------------------------------------------
# Configuration loading (shared with other modules – consider moving to config.py)
# ------------------------------------------------------------------------------
def _expand_env_vars(value: Any, _depth: int = 0) -> Any:
    """Recursively expand ${VAR} and ${VAR:default} in strings."""
    if _depth > 10:
        return value
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        def repl(match):
            var = match.group(1)
            default = match.group(2)
            return os.environ.get(var, default if default is not None else "")
        return re.sub(pattern, repl, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v, _depth + 1) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item, _depth + 1) for item in value]
    return value


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration from the first available path, expanding env vars."""
    import yaml

    if config_path and config_path.exists():
        raw = config_path.read_text()
    else:
        candidates = [
            Path.home() / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        raw = None
        for p in candidates:
            if p.exists():
                raw = p.read_text()
                break

    if raw is None:
        return {}

    cfg = yaml.safe_load(raw) or {}
    return _expand_env_vars(cfg)


# ------------------------------------------------------------------------------
# Health / connectivity check (used by check & pre‑flight)
# ------------------------------------------------------------------------------
def _mask_key(k: str) -> str:
    if len(k) > 8:
        return k[:4] + "..." + k[-4:]
    return "****"


def _check_connectivity(url: str, api_key: Optional[str], master_key: Optional[str],
                        timeout: float = 5.0) -> Dict[str, Any]:
    """
    Probe the server's /health and /v1/models endpoints.
    Returns a dict with connection status and model info.
    """
    import urllib.request
    import urllib.error

    result = {
        "server": url,
        "reachable": False,
        "health_status": "unknown",
        "models_loaded": [],
        "error": None,
    }

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Health check
    try:
        req = urllib.request.Request(f"{url}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            result["health_status"] = data.get("health_status", "unknown")
            result["reachable"] = True
    except urllib.error.HTTPError as e:
        result["error"] = f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    if not result["reachable"]:
        return result

    # Models list
    try:
        req = urllib.request.Request(f"{url}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            result["models_loaded"] = [m["id"] for m in data.get("data", [])]
    except Exception:
        pass

    return result


# ------------------------------------------------------------------------------
# Predictive summary (uses telemetry endpoint)
# ------------------------------------------------------------------------------
def _predictive_summary(url: str, api_key: Optional[str], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Fetch /v1/telemetry and return a simplified model ranking
    with predicted idle times and popularity scores.
    """
    import urllib.request
    import urllib.error

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        req = urllib.request.Request(f"{url}/v1/telemetry", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("models", [])
            if not models:
                return None
            # Return a concise summary
            summary = []
            for m in models[:5]:  # top 5
                summary.append({
                    "model_id": m.get("model_id"),
                    "popularity_score": m.get("popularity_score", 0),
                    "avg_tps": m.get("avg_tps", 0),
                    "predicted_next_request_s": m.get("predicted_next_request_s"),
                })
            return {"models": summary, "recommendations": data.get("recommendations", {})}
    except Exception:
        return None


# ------------------------------------------------------------------------------
# CLI definition
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local LLM Cloud Monitoring & Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m local_llm_server.monitoring
  python -m local_llm_server.monitoring tui --url http://192.168.1.100:8080
  python -m local_llm_server.monitoring check --json
  python -m local_llm_server.monitoring predict
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to execute")

    # ---- tui ----
    tui_parser = sub.add_parser("tui", help="Launch interactive terminal dashboard (default)")
    tui_parser.add_argument("--config", type=Path, help="Path to config.yaml")
    tui_parser.add_argument("--url", help="Server URL (overrides config)")
    tui_parser.add_argument("--refresh", type=float, help="Refresh interval in seconds")
    tui_parser.add_argument("--api-key", help="API key")
    tui_parser.add_argument("--master-key", help="Master key")
    tui_parser.add_argument("--debug", action="store_true", help="Enable debug logging to /tmp/tui_debug.log")
    tui_parser.add_argument("--no-preflight", action="store_true",
                            help="Skip pre‑flight connectivity check")
    tui_parser.add_argument("--preflight-timeout", type=float, default=3.0,
                            help="Timeout for pre‑flight health check (seconds)")

    # ---- check ----
    check_parser = sub.add_parser("check", help="Run health & connectivity diagnostics")
    check_parser.add_argument("--config", type=Path)
    check_parser.add_argument("--url", help="Server URL")
    check_parser.add_argument("--api-key", help="API key")
    check_parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout (seconds)")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # ---- predict ----
    predict_parser = sub.add_parser("predict", help="Show predictive model ranking & alerts")
    predict_parser.add_argument("--config", type=Path)
    predict_parser.add_argument("--url")
    predict_parser.add_argument("--api-key")
    predict_parser.add_argument("--timeout", type=float, default=5.0)
    predict_parser.add_argument("--json", action="store_true")

    # ---- version ----
    sub.add_parser("version", help="Show version and exit")

    return parser


# ------------------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------------------
def cmd_tui(args: argparse.Namespace) -> int:
    """Launch the terminal dashboard."""
    # Load config
    cfg = _load_config(args.config) if args.config else _load_config()
    # Merge with CLI overrides
    url = args.url or os.environ.get("LLM_SERVER_URL") or cfg.get("server_url", "http://localhost:8080")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")
    master_key = args.master_key or os.environ.get("LLM_MASTER_KEY") or cfg.get("master_key", "")
    refresh = args.refresh or cfg.get("refresh_interval", 2.0)

    # Pre‑flight check unless suppressed
    if not args.no_preflight:
        print(f"🔍 Checking connectivity to {url} ...", file=sys.stderr)
        status = _check_connectivity(url, api_key, master_key, timeout=args.preflight_timeout)
        if not status["reachable"]:
            msg = f"❌ Cannot reach server at {url}. Ensure it is running."
            if status["error"]:
                msg += f"\n   Error: {status['error']}"
            print(msg, file=sys.stderr)
            return 1
        h = status["health_status"]
        # FIX #17: warn if health_status is not "ok"
        if h == "ok":
            print(f"   Health status: ✅ {h}", file=sys.stderr)
        elif h == "degraded":
            print(f"   Health status: ⚠ {h} – some components may be unhealthy", file=sys.stderr)
        elif h == "loading":
            print(f"   Health status: ⏳ {h} – server is still starting", file=sys.stderr)
        else:
            print(f"   Health status: ❓ {h}", file=sys.stderr)
        if status["models_loaded"]:
            print(f"   Loaded models: {', '.join(status['models_loaded'])}", file=sys.stderr)
        else:
            print("   ⚠ No models loaded – some features will be limited.", file=sys.stderr)
        print("   Starting TUI...\n", file=sys.stderr)

    # Build modified argv for monitor_main (original entry point)
    new_argv = [sys.argv[0]]
    if args.config:
        new_argv.extend(["--config", str(args.config)])
    new_argv.extend(["--url", url])
    new_argv.extend(["--refresh", str(refresh)])
    if api_key:
        new_argv.extend(["--api-key", api_key])
    if master_key:
        new_argv.extend(["--master-key", master_key])
    if args.debug:
        new_argv.append("--debug")

    try:
        from local_llm_server.monitoring.monitor import main as monitor_main
    except ImportError as e:
        print(f"ERROR: Could not import monitor module: {e}", file=sys.stderr)
        print("Ensure the package is installed correctly.", file=sys.stderr)
        return 1

    try:
        with temporary_argv(new_argv):
            monitor_main()
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Unhandled exception in monitor: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run health & connectivity diagnostics."""
    cfg = _load_config(args.config) if args.config else _load_config()
    url = args.url or os.environ.get("LLM_SERVER_URL") or cfg.get("server_url", "http://localhost:8080")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")
    master_key = args.master_key or os.environ.get("LLM_MASTER_KEY") or cfg.get("master_key", "")

    status = _check_connectivity(url, api_key, master_key, timeout=args.timeout)

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Server:   {status['server']}")
        print(f"Reachable:{'✅' if status['reachable'] else '❌'}")
        if status["reachable"]:
            h = status["health_status"]
            if h == "ok":
                print(f"Health:   ✅ {h}")
            elif h == "degraded":
                print(f"Health:   ⚠ {h}")
            elif h == "loading":
                print(f"Health:   ⏳ {h}")
            else:
                print(f"Health:   ❓ {h}")
            models = status["models_loaded"]
            if models:
                print(f"Models:   {', '.join(models)}")
            else:
                print("Models:   (none loaded)")
        else:
            print(f"Error:    {status['error']}")
    return 0 if status["reachable"] else 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Show predictive model ranking & alerts."""
    cfg = _load_config(args.config) if args.config else _load_config()
    url = args.url or os.environ.get("LLM_SERVER_URL") or cfg.get("server_url", "http://localhost:8080")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")

    summary = _predictive_summary(url, api_key, timeout=args.timeout)
    if summary is None:
        msg = "No predictive data available. Ensure server supports /v1/telemetry."
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(msg)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Top models by popularity:")
        for m in summary["models"]:
            predicted = m.get("predicted_next_request_s")
            pred_str = f" (next request in {predicted:.0f}s)" if predicted is not None else ""
            print(f"  {m['model_id']:<20} popularity={m['popularity_score']:.2f}  "
                  f"avg_tps={m['avg_tps']:.1f}{pred_str}")
        recs = summary.get("recommendations", {})
        if recs:
            print("\nUnload recommendations:")
            for mid, reason in recs.items():
                print(f"  {mid}: {reason}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print version and exit."""
    try:
        from local_llm_server import __version__ as server_version
    except ImportError:
        server_version = "unknown"
    print(f"local-llm-server {server_version}")
    print(f"Monitoring CLI v4.1.0")
    return 0


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # If no command given, default to tui (backward compat)
    if not args.command:
        # Re‑parse with 'tui' as the command and any remaining args as tui args
        new_argv = [sys.argv[0], "tui"]
        if argv is not None:
            new_argv.extend(argv)
        else:
            new_argv.extend(sys.argv[1:])
        return main(new_argv)

    if args.command == "tui":
        return cmd_tui(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "version":
        return cmd_version(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
