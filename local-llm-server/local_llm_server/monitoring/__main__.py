#!/usr/bin/env python3
"""
local_llm_server/monitoring/__main__.py — Production Entry Point
===============================================================================
Universal entry point for the Local LLM Cloud monitoring ecosystem.

version = 1.0.0
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
# Path setup — ensure the package root is on sys.path
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
# Configuration loading
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


def _load_dotenv(path: Path) -> None:
    """Minimal `.env` loader: KEY=VALUE per line, `#` comments ignored."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration from the first available path, expanding env vars.

    Discovery order:
        1. explicit --config argument
        2. $LLM_CONFIG_PATH (the systemd unit sets this)
        3. /etc/local-llm/config.yaml
        4. ~/config.yaml
        5. ./config.yaml
        6. package default
    """
    import yaml

    env_cfg = os.environ.get("LLM_CONFIG_PATH", "").strip()
    pkg_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates.extend([
        Path("/etc/local-llm/config.yaml"),
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
        pkg_dir / "config.yaml",
    ])

    raw: Optional[str] = None
    chosen_path: Optional[Path] = None
    for p in candidates:
        try:
            if p.exists():
                raw = p.read_text(encoding="utf-8")
                chosen_path = p
                break
        except OSError:
            continue

    if raw is None:
        return {}

    cfg = yaml.safe_load(raw) or {}
    cfg = _expand_env_vars(cfg)

    # Load `.env` beside the resolved config so secrets only in .env
    # (not exported to our process) become available.
    try:
        if chosen_path is not None:
            env_file = chosen_path.parent / ".env"
            if env_file.exists():
                _load_dotenv(env_file)
    except Exception:
        pass

    return cfg


# ------------------------------------------------------------------------------
# Health / connectivity check
# ------------------------------------------------------------------------------
def _mask_key(k: str) -> str:
    if len(k) > 8:
        return k[:4] + "..." + k[-4:]
    return "****"


def _check_connectivity(
    url: str, api_key: Optional[str], master_key: Optional[str],
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Probe the server's /health and /v1/models endpoints.

    A non-2xx HTTP response is NOT the same as unreachable. A 503 with a
    JSON body (pool degraded, model starting) means the server is running
    and answering — the pre-flight must NOT refuse to launch the TUI in
    that state. Only a connection-level failure is treated as
    "unreachable".

    The server sends `health_status ∈ {"ok", "degraded", "startup_failed"}`
    per the P2-18 canonical schema. Older servers sent "healthy". Both
    are accepted.
    """
    import urllib.request
    import urllib.error

    result: Dict[str, Any] = {
        "server": url,
        "reachable": False,
        "health_status": "unknown",
        "models_loaded": [],
        "error": None,
        "degraded": False,
    }

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # ── 1. Health probe ─────────────────────────────────────────────
    try:
        req = urllib.request.Request(f"{url}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            result["reachable"] = True
            result["health_status"] = (
                data.get("health_status")
                or data.get("status")
                or "unknown"
            )
    except urllib.error.HTTPError as e:
        # ANY HTTP response proves the server is up.
        result["reachable"] = True
        result["error"] = f"HTTP {e.code}: {e.reason}"
        try:
            raw = e.read().decode("utf-8", "replace")
            data = json.loads(raw) if raw else {}
            result["health_status"] = (
                data.get("health_status")
                or data.get("status")
                or f"http_{e.code}"
            )
        except Exception:
            result["health_status"] = f"http_{e.code}"
        if e.code >= 500:
            result["degraded"] = True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    if not result["reachable"]:
        return result

    # ── 2. Models list (best-effort) ────────────────────────────────
    try:
        req = urllib.request.Request(f"{url}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            data = json.loads(raw) if raw else {}
            result["models_loaded"] = [
                m.get("id", "?") for m in data.get("data", []) if m.get("id")
            ]
    except Exception:
        pass

    return result


# ------------------------------------------------------------------------------
# SLA endpoint probe (used by check + tui pre-flight)
# ------------------------------------------------------------------------------
def _probe_sla_endpoints(
    url: str, api_key: Optional[str], timeout: float = 5.0,
) -> Dict[str, Dict[str, Any]]:
    """Probe the two SLA/performance endpoints the TUI depends on.

    Returns:
        {
          "performance":     {status, detail, keys_present, missing_keys, http_status},
          "runtime_limits":  {status, detail, keys_present, missing_keys, http_status},
          "tui_ready":       bool,   # True iff both endpoints returned OK
        }

    `status` is one of:
        "ok"          — 200 with all required keys present
        "malformed"   — 200 but one or more required keys are missing
        "auth_error"  — 401 or 403 (API key required or wrong)
        "missing"     — 404 (endpoint absent — old server)
        "unreachable" — connection-level failure
        "http_NNN"    — any other HTTP status (5xx, 429, ...)

    The per-key validation is the addition. An in-version server that
    returned 200 but, say, omitted `confidence_index` after a regression
    would still pass a naive "did it 200?" check — this helper surfaces
    that as `malformed` so the operator knows the panel will be broken.
    """
    import urllib.request
    import urllib.error

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Required keys per endpoint — the exact set the TUI reads.
    # Order matters only for display; validation is set-based.
    required_keys: Dict[str, Tuple[str, ...]] = {
        "performance": ("tier", "ttft_pass", "tps_pass", "confidence_index", "per_model"),
        "runtime_limits": ("model_size_mb", "target_tps", "decode_tps", "effective_tps"),
    }
    paths: Dict[str, str] = {
        "performance": "/v1/performance",
        "runtime_limits": "/v1/runtime/limits",
    }

    out: Dict[str, Any] = {"tui_ready": False}

    for name, path in paths.items():
        entry: Dict[str, Any] = {
            "status": "unknown",
            "detail": "",
            "keys_present": [],
            "missing_keys": [],
            "http_status": None,
        }
        full_url = f"{url}{path}"
        try:
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                entry["http_status"] = resp.status
                raw = resp.read().decode("utf-8", "replace")
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    entry["status"] = "malformed"
                    entry["detail"] = "response is not JSON"
                    out[name] = entry
                    continue
                if not isinstance(data, dict):
                    entry["status"] = "malformed"
                    entry["detail"] = f"expected object, got {type(data).__name__}"
                    out[name] = entry
                    continue
                present = [k for k in required_keys[name] if k in data]
                missing = [k for k in required_keys[name] if k not in data]
                entry["keys_present"] = present
                entry["missing_keys"] = missing
                if missing:
                    entry["status"] = "malformed"
                    entry["detail"] = f"missing keys: {', '.join(missing)}"
                else:
                    entry["status"] = "ok"
                    entry["detail"] = f"{len(present)}/{len(required_keys[name])} keys"
        except urllib.error.HTTPError as e:
            entry["http_status"] = e.code
            if e.code in (401, 403):
                entry["status"] = "auth_error"
                entry["detail"] = f"HTTP {e.code} — check API key"
            elif e.code == 404:
                entry["status"] = "missing"
                entry["detail"] = "HTTP 404 — endpoint not present (old server?)"
            else:
                entry["status"] = f"http_{e.code}"
                entry["detail"] = f"HTTP {e.code}: {e.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            entry["status"] = "unreachable"
            entry["detail"] = str(e)
        except Exception as e:
            entry["status"] = "error"
            entry["detail"] = str(e)

        out[name] = entry

    out["tui_ready"] = all(
        out[name]["status"] == "ok" for name in paths.keys()
    )
    return out


def _sla_status_badge(status: str) -> str:
    """Human-readable badge for `_probe_sla_endpoints`' status field."""
    if status == "ok":
        return "✅ ok"
    if status == "missing":
        return "⚠ missing"
    if status == "auth_error":
        return "❌ auth"
    if status == "malformed":
        return "⚠ malformed"
    if status == "unreachable":
        return "❌ unreachable"
    return f"⚠ {status}"


# ------------------------------------------------------------------------------
# Predictive summary (uses /v1/telemetry)
# ------------------------------------------------------------------------------
def _predictive_summary(
    url: str, api_key: Optional[str], timeout: float = 5.0,
) -> Dict[str, Any]:
    """Fetch `/v1/telemetry` and return a simplified model ranking.

    the returned per-model dicts now include the token split
    (`thinking_tokens_total`, `cached_prompt_tokens_total`) when the
    server's `ModelProfile.to_dict()` provides them. Backward
    compatible — those keys are simply absent on older servers.

    Raises nothing. On any failure returns `{"error": <reason>, "kind": <kind>}`
    where `kind` ∈ {"unreachable", "auth_error", "missing", "malformed",
    "http_NNN", "error"}. Caller (`cmd_predict`) distinguishes these for
    the operator.
    """
    import urllib.request
    import urllib.error

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        req = urllib.request.Request(f"{url}/v1/telemetry", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            data = json.loads(raw) if raw else {}
            models = data.get("models", [])
            if not isinstance(models, list):
                return {
                    "error": f"expected 'models' to be a list, got "
                             f"{type(models).__name__}",
                    "kind": "malformed",
                }
            summary: List[Dict[str, Any]] = []
            for m in models[:5]:
                if not isinstance(m, dict):
                    continue
                entry = {
                    "model_id": m.get("model_id", "?"),
                    "popularity_score": float(m.get("popularity_score", 0) or 0),
                    "avg_tps": float(m.get("avg_tps", 0) or 0),
                    "avg_ttft_ms": float(m.get("avg_ttft_ms", 0) or 0),
                    "total_tokens": int(m.get("total_tokens", 0) or 0),
                    "predicted_next_request_s": m.get("predicted_next_request_s"),
                }
                # additions — present only on servers.
                if "thinking_tokens_total" in m:
                    entry["thinking_tokens_total"] = int(
                        m.get("thinking_tokens_total", 0) or 0
                    )
                if "cached_prompt_tokens_total" in m:
                    entry["cached_prompt_tokens_total"] = int(
                        m.get("cached_prompt_tokens_total", 0) or 0
                    )
                summary.append(entry)
            return {
                "models": summary,
                "recommendations": data.get("recommendations", {}),
            }
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return {
                "error": f"HTTP {e.code} — check API key",
                "kind": "auth_error",
            }
        if e.code == 404:
            return {
                "error": "HTTP 404 — /v1/telemetry not present (old server?)",
                "kind": "missing",
            }
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "kind": f"http_{e.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": str(e), "kind": "unreachable"}
    except Exception as e:
        return {"error": str(e), "kind": "error"}


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

Exit codes:
  tui      0 on clean exit; non-zero on unrecoverable failure
  check    0 iff the server answered /health (SLA endpoint issues are
           warnings, not failures — pipe --json output for scripting)
  predict  0 if telemetry data was retrieved; 1 on any failure
  version  0 always
        """,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="Path to config.yaml")
    common.add_argument("--url", help="Server URL")
    common.add_argument("--api-key", help="API key")
    common.add_argument(
        "--master-key",
        help="Master key (required for admin operations; same value as API key is fine for read-only panels)",
    )

    sub = parser.add_subparsers(dest="command", help="Command to execute")

    tui_parser = sub.add_parser(
        "tui",
        help="Launch interactive terminal dashboard (default)",
        parents=[common],
    )
    tui_parser.add_argument("--refresh", type=float, help="Refresh interval in seconds")
    tui_parser.add_argument("--debug", action="store_true", help="Enable debug logging to /tmp/tui_debug.log")
    tui_parser.add_argument("--no-preflight", action="store_true", help="Skip pre‑flight connectivity check")
    tui_parser.add_argument("--preflight-timeout", type=float, default=3.0,
                            help="Timeout for pre‑flight health check (seconds)")

    check_parser = sub.add_parser(
        "check",
        help="Run health & connectivity diagnostics",
        parents=[common],
    )
    check_parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout (seconds)")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")
    check_parser.add_argument(
        "--skip-sla",
        action="store_true",
        help="Skip the SLA endpoint probe (check /v1/performance and "
             "/v1/runtime/limits only if you do not care about TUI panels)",
    )

    predict_parser = sub.add_parser(
        "predict",
        help="Show predictive model ranking & alerts",
        parents=[common],
    )
    predict_parser.add_argument("--timeout", type=float, default=5.0)
    predict_parser.add_argument("--json", action="store_true")

    sub.add_parser("version", help="Show version and exit")

    return parser


# ------------------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------------------
def cmd_tui(args: argparse.Namespace) -> int:
    """Launch the terminal dashboard."""
    cfg = _load_config(args.config) if args.config else _load_config()
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
        if h in ("ok", "healthy"):
            print(f"   Health status: ✅ {h}", file=sys.stderr)
        elif h in ("degraded", "loading"):
            print(f"   Health status: ⚠ {h} – some components may be unhealthy", file=sys.stderr)
        else:
            print(f"   Health status: ❓ {h}", file=sys.stderr)
        if status["models_loaded"]:
            print(f"   Loaded models: {', '.join(status['models_loaded'])}", file=sys.stderr)
        else:
            print("   ⚠ No models loaded – some features will be limited.", file=sys.stderr)

        # soft warning — the TUI still starts if the SLA endpoints
        # are missing, but the confidence / per-model / Efficiency panels
        # will show placeholders. Make that explicit so the operator
        # isn't left wondering why the panels are empty.
        sla_probe = _probe_sla_endpoints(url, api_key, timeout=args.preflight_timeout)
        if not sla_probe["tui_ready"]:
            missing = [
                name for name in ("performance", "runtime_limits")
                if sla_probe[name]["status"] != "ok"
            ]
            print(
                f"   ⚠ SLA endpoints unavailable ({', '.join(missing)}): "
                f"confidence / per-model / Efficiency panels will show placeholders.",
                file=sys.stderr,
            )
            for name in missing:
                entry = sla_probe[name]
                print(
                    f"     • {name}: {_sla_status_badge(entry['status'])} — {entry.get('detail','')}",
                    file=sys.stderr,
                )
        print("   Starting TUI...\n", file=sys.stderr)

    # Build modified argv for monitor_main
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
    """Run health & connectivity diagnostics.

    also probes `/v1/performance` and `/v1/runtime/limits` and reports
    per-key shape validation. Exit code remains "0 iff server reachable" —
    a missing SLA endpoint is a warning, not a failure. Use `--json` and
    inspect `sla_endpoints.tui_ready` for scripting.
    """
    cfg = _load_config(args.config) if args.config else _load_config()
    url = args.url or os.environ.get("LLM_SERVER_URL") or cfg.get("server_url", "http://localhost:8080")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")
    master_key = args.master_key or os.environ.get("LLM_MASTER_KEY") or cfg.get("master_key", "")

    status = _check_connectivity(url, api_key, master_key, timeout=args.timeout)

    # Only probe SLA endpoints if the server is reachable and healthy — a
    # genuinely offline server produces a cascade of duplicate
    # "unreachable" noise that adds no information.
    sla_probe: Optional[Dict[str, Any]] = None
    if (not args.skip_sla
            and status["reachable"]
            and status["health_status"] in ("ok", "healthy", "degraded")):
        sla_probe = _probe_sla_endpoints(url, api_key, timeout=args.timeout)

    if args.json:
        payload: Dict[str, Any] = dict(status)
        if sla_probe is not None:
            payload["sla_endpoints"] = sla_probe
        print(json.dumps(payload, indent=2))
    else:
        print(f"Server:   {status['server']}")
        print(f"Reachable:{'✅' if status['reachable'] else '❌'}")
        if status["reachable"]:
            h = status["health_status"]
            if h in ("ok", "healthy"):
                print(f"Health:   ✅ {h}")
            elif h in ("degraded", "loading"):
                print(f"Health:   ⚠ {h}")
            else:
                print(f"Health:   ❓ {h}")
            models = status["models_loaded"]
            if models:
                print(f"Models:   {', '.join(models)}")
            else:
                print("Models:   (none loaded)")
        else:
            print(f"Error:    {status['error']}")

        if sla_probe is not None:
            print()
            print("SLA Endpoints:")
            for name, path in (("performance", "/v1/performance"),
                               ("runtime_limits", "/v1/runtime/limits")):
                entry = sla_probe[name]
                badge = _sla_status_badge(entry["status"])
                detail = entry.get("detail", "")
                print(f"  {path:22s} {badge:14s} {detail}")
            if not sla_probe["tui_ready"]:
                print()
                print(
                    "  ⚠ TUI panels affected: confidence, per-model, Efficiency.\n"
                    "    These require(/v1/performance) "
                    "(/v1/runtime/limits) on the server side."
                )

    return 0 if status["reachable"] else 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Show predictive model ranking & alerts.

    distinguishes four previously-indistinguishable failure modes:
      • server unreachable      → "cannot connect"
      • endpoint missing (404)  → "old server without /v1/telemetry"
      • auth error (401/403)    → "check API key"
      • malformed response      → "unexpected shape"

    """
    cfg = _load_config(args.config) if args.config else _load_config()
    url = args.url or os.environ.get("LLM_SERVER_URL") or cfg.get("server_url", "http://localhost:8080")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key", "")

    # Distinguish unreachable before trying /v1/telemetry — otherwise a
    # genuine connection failure is misreported as "no predictive data".
    conn = _check_connectivity(url, api_key, None, timeout=args.timeout)
    if not conn["reachable"]:
        msg = f"Cannot reach server at {url}"
        if conn["error"]:
            msg += f": {conn['error']}"
        if args.json:
            print(json.dumps({"error": msg, "kind": "unreachable"}))
        else:
            print(f"❌ {msg}")
        return 1

    summary = _predictive_summary(url, api_key, timeout=args.timeout)
    if "error" in summary:
        kind = summary.get("kind", "error")
        err = summary["error"]
        if args.json:
            print(json.dumps({"error": err, "kind": kind}))
        else:
            if kind == "auth_error":
                print(f"❌ Authentication failed: {err}")
                print("   Set --api-key, LLM_API_KEY, or ensure the server "
                      "has api.key configured.")
            elif kind == "missing":
                print(f"❌ Endpoint not available: {err}")
                print("   /v1/telemetry is served by local-llm-server's "
                      "monitoring/telemetry.py — update the server package.")
            elif kind == "malformed":
                print(f"❌ Malformed response: {err}")
            else:
                print(f"❌ Failed to fetch predictive data: {err}")
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        models = summary.get("models", [])
        if not models:
            print("No predictive data yet — the model hasn't been used.")
            print("Run a few inferences, then retry `predict`.")
        else:
            print("Top models by popularity:")
            # Column widths chosen for additions. Numeric columns are
            # right-aligned so a wide token count doesn't shift the layout.
            header = (
                f"  {'model':<20} {'popularity':>10} {'avg_tps':>8} "
                f"{'ttft':>8} {'tokens':>9} {'think':>8} {'cached':>8}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for m in models:
                predicted = m.get("predicted_next_request_s")
                pred_str = (
                    f"  (next ~{predicted:.0f}s)" if predicted is not None else ""
                )
                think = m.get("thinking_tokens_total")
                cached = m.get("cached_prompt_tokens_total")
                print(
                    f"  {m['model_id']:<20} "
                    f"{m['popularity_score']:>10.2f} "
                    f"{m['avg_tps']:>8.1f} "
                    f"{m['avg_ttft_ms']:>7.0f}ms "
                    f"{m['total_tokens']:>9} "
                    f"{(str(think) if think is not None else '—'):>8} "
                    f"{(str(cached) if cached is not None else '—'):>8}"
                    f"{pred_str}"
                )
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
    print(f"Monitoring CLI v1.0.0 ")
    return 0


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
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
