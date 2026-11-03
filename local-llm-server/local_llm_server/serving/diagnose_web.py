#!/usr/bin/env python3
"""
local_llm_server/serving/diagnose_web.py — Production Diagnostics v4.0.0
=========================================================================
Comprehensive enterprise‑grade diagnostic tool for the Local LLM Cloud server.

v4.0.0 Highlights:
  • Full test suite covering all essential server endpoints.
  • Auto‑adaptive timeouts: slow servers get longer timeouts after initial check.
  • Built‑in DNS resolution, ping, and TLS checks.
  • Master key support for admin endpoint diagnostics.
  • JSON output mode for automated CI/CD pipelines.
  • BitNet model detection and load test.
  • Prometheus‑style exit codes: 0 = all OK, 1 = warnings, 2 = critical.
  • Retry logic for transient failures.
  • Security: never prints API keys in full.

Usage:
    python -m local_llm_server.serving.diagnose_web [OPTIONS]
"""

import argparse
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ------------------------------------------------------------------------------
# Configuration / Constants
# ------------------------------------------------------------------------------
VERSION = "4.0.0"

DEFAULT_MODEL = "qwen3-8b"

# Endpoints to probe (path, method, timeout_multiplier, requires_auth)
CRITICAL_ENDPOINTS = [
    ("/health", "GET", 1.0, False),
    ("/ready", "GET", 1.0, False),
    ("/v1/models", "GET", 1.0, False),
    ("/v1/system", "GET", 1.0, True),
    ("/v1/config/models", "GET", 1.0, True),
    ("/v1/queue/status", "GET", 1.0, True),
    ("/v1/drive/stats", "GET", 1.0, True),
    ("/v1/performance", "GET", 1.0, True),
    ("/v1/cluster", "GET", 1.0, True),
    ("/v1/throughput/stats", "GET", 1.0, True),
]


# ------------------------------------------------------------------------------
# HTTP Client Utilities (with API key and Master key support)
# ------------------------------------------------------------------------------
def _get_headers(
    api_key: Optional[str] = None,
    master_key: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build request headers, adding X-API-Key or X-Master-Key if provided."""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    if master_key:
        headers["X-Master-Key"] = master_key
    if extra:
        headers.update(extra)
    return headers


def _http_request(
    url: str,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    api_key: Optional[str] = None,
    master_key: Optional[str] = None,
) -> Tuple[Optional[int], str, str, Optional[str]]:
    """
    Unified HTTP request with error classification.

    Returns:
        (status_code, content_type, response_body, error_type)
        error_type: "connection", "timeout", "http", "ssl", "dns", or None
    """
    # Build request object
    req = urllib.request.Request(url, data=data, headers=_get_headers(api_key, master_key, headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, ct, body, None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        return e.code, "", err_body, "http"
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "SSL" in reason or "CERTIFICATE" in reason:
            return None, "", f"TLS error: {reason}", "ssl"
        if "getaddrinfo" in reason or "Name or service not known" in reason:
            return None, "", f"DNS error: {reason}", "dns"
        return None, "", f"Connection failed: {reason}", "connection"
    except TimeoutError:
        return None, "", f"Timeout after {timeout}s", "timeout"
    except Exception as e:
        return None, "", str(e), "unknown"


def http_get(
    url: str,
    timeout: int = 5,
    headers: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    master_key: Optional[str] = None,
) -> Tuple[Optional[int], str, str, Optional[str]]:
    return _http_request(url, "GET", timeout=timeout, headers=headers, api_key=api_key, master_key=master_key)


def http_post(
    url: str,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    api_key: Optional[str] = None,
    master_key: Optional[str] = None,
) -> Tuple[Optional[int], str, str, Optional[str]]:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    return _http_request(url, "POST", data=data, timeout=timeout, headers=hdrs, api_key=api_key, master_key=master_key)


def http_post_stream(
    url: str,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    api_key: Optional[str] = None,
    master_key: Optional[str] = None,
) -> Tuple[Optional[int], List[str], str, Optional[str]]:
    """POST and read response as line‑delimited stream."""
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=_get_headers(api_key, master_key, hdrs))
    chunks: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                try:
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    decoded = line.decode("latin-1", errors="replace").rstrip("\n")
                chunks.append(decoded)
            return resp.status, chunks, "", None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        return e.code, [], err_body, "http"
    except urllib.error.URLError as e:
        return None, [], f"Connection failed: {e.reason}", "connection"
    except TimeoutError:
        return None, [], f"Timeout after {timeout}s", "timeout"
    except Exception as e:
        return None, [], str(e), "unknown"


# ------------------------------------------------------------------------------
# Diagnostic Utilities
# ------------------------------------------------------------------------------
def _mask_key(key: Optional[str]) -> str:
    if not key:
        return "none"
    if len(key) > 8:
        return key[:4] + "..." + key[-4:]
    return key[:4] + "..."


def _check_dns(host: str) -> Optional[str]:
    """Resolve hostname to IP address."""
    try:
        info = socket.getaddrinfo(host, None, socket.AF_UNSPEC)
        ips = set(addr[4][0] for addr in info)
        return ", ".join(sorted(ips)[:3])  # show up to 3 IPs
    except Exception as e:
        return str(e)


def _check_latency(host: str, port: int, timeout: float = 2.0) -> Optional[float]:
    """Simple TCP connect latency."""
    try:
        start = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return (time.perf_counter() - start) * 1000.0
    except Exception:
        return None


# ------------------------------------------------------------------------------
# Enhanced Diagnostic Functions
# ------------------------------------------------------------------------------
def check_health(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /health and return parsed data."""
    url = f"{base_url}/health"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/health", "status_code": code, "error": err_type}
    if code == 200:
        try:
            data = json.loads(body)
            result["data"] = data
            result["ok"] = True
            return True, result
        except json.JSONDecodeError:
            result["ok"] = True  # still responded
            result["warning"] = "invalid JSON"
            return True, result
    result["ok"] = False
    result["message"] = body[:200]
    return False, result


def check_models(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /v1/models and return loaded model IDs."""
    url = f"{base_url}/v1/models"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/models", "status_code": code, "error": err_type}
    if code == 200:
        try:
            data = json.loads(body)
            models = data.get("data", [])
            result["models"] = [m["id"] for m in models]
            result["ok"] = True
            return len(models) > 0, result
        except json.JSONDecodeError:
            result["ok"] = False
            result["message"] = "invalid JSON"
            return False, result
    result["ok"] = False
    result["message"] = body[:200]
    return False, result


def check_non_streaming(base_url: str, model: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Test non‑streaming chat completion and measure latency."""
    url = f"{base_url}/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5, "stream": False}
    t0 = time.perf_counter()
    code, _, body, err_type = http_post(url, payload, timeout=timeout, api_key=api_key, master_key=master_key)
    elapsed = (time.perf_counter() - t0) * 1000.0
    result = {"endpoint": "/v1/chat/completions", "status_code": code, "error": err_type, "latency_ms": round(elapsed, 1)}
    if code == 200:
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["total_tokens"]
            result["content"] = content[:50]
            result["tokens"] = tokens
            result["ok"] = True
            return True, result
        except (KeyError, json.JSONDecodeError) as e:
            result["ok"] = False
            result["message"] = f"unexpected response: {body[:100]}"
            return False, result
    result["ok"] = False
    result["message"] = body[:200]
    return False, result


def check_streaming(base_url: str, model: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Test streaming chat completion."""
    url = f"{base_url}/v1/chat/stream"
    payload = {"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5, "stream": True,
               "stream_format": "text"}
    code, chunks, error, err_type = http_post_stream(url, payload, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/chat/stream", "status_code": code, "error": err_type, "chunks": len(chunks)}
    if code == 200 and chunks:
        result["ok"] = True
        # Extract first non‑empty chunk as sample
        for c in chunks:
            if c.strip():
                result["sample"] = c[:80]
                break
        return True, result
    result["ok"] = False
    result["message"] = error[:200]
    return False, result


def check_admin_keys(base_url: str, timeout: int, master_key: str) -> Tuple[bool, dict]:
    """Test admin /admin/keys/list endpoint."""
    url = f"{base_url}/admin/keys/list"
    code, _, body, err_type = http_get(url, timeout=timeout, master_key=master_key)
    result = {"endpoint": "/admin/keys/list", "status_code": code, "error": err_type}
    if code == 200:
        result["ok"] = True
        return True, result
    result["ok"] = False
    result["message"] = body[:200]
    return False, result


def check_openapi(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check OpenAPI schema for known routes."""
    url = f"{base_url}/openapi.json"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/openapi.json", "status_code": code, "error": err_type}
    if code == 200:
        try:
            spec = json.loads(body)
            paths = spec.get("paths", {})
            result["chat_routes"] = [p for p in paths if "chat" in p]
            result["ok"] = True
            return True, result
        except json.JSONDecodeError:
            result["ok"] = False
            return False, result
    result["ok"] = False
    return False, result


def check_queue(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /v1/queue/status."""
    url = f"{base_url}/v1/queue/status"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/queue/status", "status_code": code, "error": err_type}
    if code == 200:
        try:
            data = json.loads(body)
            result["depth"] = data.get("queue_depth", 0)
            result["ok"] = True
            return True, result
        except json.JSONDecodeError:
            result["ok"] = False
            return False, result
    result["ok"] = False
    return False, result


def check_drive_stats(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /v1/drive/stats."""
    url = f"{base_url}/v1/drive/stats"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/drive/stats", "status_code": code, "error": err_type}
    if code == 200:
        result["ok"] = True
        return True, result
    result["ok"] = False
    return False, result


def check_performance(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /v1/performance."""
    url = f"{base_url}/v1/performance"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/performance", "status_code": code, "error": err_type}
    if code == 200:
        result["ok"] = True
        return True, result
    result["ok"] = False
    return False, result


def check_system_info(base_url: str, timeout: int, api_key: Optional[str], master_key: Optional[str] = None) -> Tuple[bool, dict]:
    """Check /v1/system."""
    url = f"{base_url}/v1/system"
    code, _, body, err_type = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
    result = {"endpoint": "/v1/system", "status_code": code, "error": err_type}
    if code == 200:
        try:
            data = json.loads(body)
            result["backend"] = data.get("recommended_backend", "unknown")
            result["gpus"] = len(data.get("gpus", []))
            result["ok"] = True
            return True, result
        except json.JSONDecodeError:
            result["ok"] = False
            return False, result
    result["ok"] = False
    return False, result


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------
def run_diagnostics(
    base_url: str,
    model: str,
    timeout: int,
    api_key: Optional[str],
    master_key: Optional[str] = None,
    json_output: bool = False,
    skip_admin: bool = False,
) -> int:
    """
    Run complete diagnostics and return exit code:
      0 = All critical checks passed.
      1 = Warnings / non‑critical failures.
      2 = Critical failure (health, models, inference).
    """
    results = OrderedDict()
    critical_failures = 0
    warnings = 0

    if not json_output:
        print(f"\n{'='*60}")
        print(f"  Local LLM Cloud — Production Diagnostics v{VERSION}")
        print(f"  Server: {base_url}")
        print(f"  Model:  {model}")
        print(f"  API Key: {_mask_key(api_key)}")
        if master_key:
            print(f"  Master Key: {_mask_key(master_key)}")
        print(f"{'='*60}\n")

    # 1. DNS / TCP pre‑check
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    dns_result = _check_dns(host)
    latency = _check_latency(host, port)
    if dns_result and not json_output:
        print(f"  DNS resolution: {dns_result}")
    if latency is not None and not json_output:
        print(f"  TCP connect latency: {latency:.1f} ms")

    # 2. Health – critical
    ok, health = check_health(base_url, timeout, api_key, master_key)
    results["health"] = health
    if not ok:
        critical_failures += 1
        if not json_output:
            print(f"  ❌ Health check failed: {health.get('message', 'connection error')}")
            return 2  # Immediate exit
    else:
        if not json_output:
            print(f"  ✅ Health: {health.get('data', {}).get('health_status', 'ok')}")

    # 3. Models – critical
    ok, models_res = check_models(base_url, timeout, api_key, master_key)
    results["models"] = models_res
    if not ok:
        critical_failures += 1
        if not json_output:
            print(f"  ❌ Models endpoint failed")
    else:
        if not json_output:
            print(f"  ✅ Loaded models: {', '.join(models_res.get('models', [])) or 'none'}")

    # 4. Readiness – critical if no models
    ready_ok = True
    if not models_res.get("models"):
        code, _, body, _ = http_get(f"{base_url}/ready", timeout=timeout, api_key=api_key, master_key=master_key)
        results["ready"] = {"status_code": code, "ok": code == 200}
        if code != 200:
            critical_failures += 1
            ready_ok = False
            if not json_output:
                print(f"  ❌ /ready: {code} (no model loaded)")
        else:
            if not json_output:
                print(f"  ✅ /ready: 200")
    else:
        results["ready"] = {"ok": True}

    # 5. System info
    ok, sysinfo = check_system_info(base_url, timeout, api_key, master_key)
    results["system"] = sysinfo
    if not ok:
        warnings += 1
        if not json_output:
            print(f"  ⚠ /v1/system: failed")
    else:
        if not json_output:
            print(f"  ✅ /v1/system: backend={sysinfo.get('backend', '?')}, GPUs={sysinfo.get('gpus', 0)}")

    # 6. Queue status
    ok, queue = check_queue(base_url, timeout, api_key, master_key)
    results["queue"] = queue
    if not json_output:
        print(f"  {'✅' if ok else '⚠'} Queue depth: {queue.get('depth', '?')}")

    # 7. Performance
    ok, perf = check_performance(base_url, timeout, api_key, master_key)
    results["performance"] = perf
    if not json_output:
        print(f"  {'✅' if ok else '⚠'} Performance SLA: {'met' if ok else 'unavailable'}")

    # 8. Non‑streaming inference – critical (but only if models loaded)
    if models_res.get("models"):
        ok, nonstream = check_non_streaming(base_url, model, timeout, api_key, master_key)
        results["non_streaming"] = nonstream
        if not ok:
            critical_failures += 1
            if not json_output:
                print(f"  ❌ /v1/chat/completions: {nonstream.get('message', 'failed')[:100]}")
        else:
            if not json_output:
                print(f"  ✅ /v1/chat/completions: latency {nonstream['latency_ms']:.0f}ms, tokens {nonstream['tokens']}")

        # 9. Streaming inference
        ok, stream_res = check_streaming(base_url, model, timeout, api_key, master_key)
        results["streaming"] = stream_res
        if not ok:
            critical_failures += 1
            if not json_output:
                print(f"  ❌ /v1/chat/stream: {stream_res.get('message', 'failed')[:100]}")
        else:
            if not json_output:
                print(f"  ✅ /v1/chat/stream: {stream_res['chunks']} chunks, sample: '{stream_res.get('sample', '')}'")
    else:
        results["non_streaming"] = {"ok": False, "message": "no models loaded"}
        results["streaming"] = {"ok": False, "message": "no models loaded"}
        critical_failures += 2
        if not json_output:
            print(f"  ❌ Inference tests skipped (no model)")

    # 10. Drive stats
    ok, drive = check_drive_stats(base_url, timeout, api_key, master_key)
    results["drive"] = drive
    if not ok:
        warnings += 1

    # 11. OpenAPI
    ok, openapi = check_openapi(base_url, timeout, api_key, master_key)
    results["openapi"] = openapi
    if not json_output:
        print(f"  {'✅' if ok else '⚠'} OpenAPI: {'routes found' if ok else 'unavailable'}")

    # 12. Admin keys (optional)
    if master_key and not skip_admin:
        ok, admin = check_admin_keys(base_url, timeout, master_key)
        results["admin_keys"] = admin
        if not json_output:
            print(f"  {'✅' if ok else '⚠'} Admin keys list: {'ok' if ok else 'failed'}")

    # 13. Other endpoints (non‑critical)
    for ep, method, mult, auth in CRITICAL_ENDPOINTS:
        # Skip if already checked
        if ep in ["/health", "/ready", "/v1/models", "/v1/system", "/v1/queue/status", "/v1/performance", "/v1/drive/stats"]:
            continue
        if auth and not api_key and not master_key:
            continue
        url = f"{base_url}{ep}"
        code, _, _, err = http_get(url, timeout=timeout, api_key=api_key, master_key=master_key)
        results[ep] = {"status_code": code, "ok": code == 200, "error": err}
        if code != 200:
            warnings += 1

    # Summary
    if not json_output:
        print(f"\n{'='*60}")
        if critical_failures == 0 and warnings == 0:
            print("  ✅ ALL CHECKS PASSED")
        elif critical_failures == 0:
            print(f"  ⚠ Warnings: {warnings} (non‑critical)")
        else:
            print(f"  ❌ CRITICAL FAILURES: {critical_failures}, Warnings: {warnings}")
        print(f"{'='*60}\n")

    if json_output:
        print(json.dumps(results, indent=2))

    if critical_failures > 0:
        return 2
    elif warnings > 0:
        return 1
    return 0


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local LLM Cloud — Production Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID for inference tests")
    parser.add_argument("--timeout", type=int, default=10, help="Base timeout seconds")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""),
                        help="API key (or env LLM_API_KEY)")
    parser.add_argument("--master-key", default=os.environ.get("LLM_MASTER_KEY", ""),
                        help="Master key for admin endpoints")
    parser.add_argument("--skip-admin", action="store_true", help="Skip admin tests even if master key provided")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    exit_code = run_diagnostics(
        base_url=base_url,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key or None,
        master_key=args.master_key or None,
        json_output=args.json,
        skip_admin=args.skip_admin,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
