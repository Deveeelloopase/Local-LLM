"""tools/tune.py — tuning + benchmarking for Local LLM Cloud.

Two entrypoints (both accept `argv: list[str] | None`, both return int):
  main(argv)        llm-tune     — analyse /v1/metrics, /v1/performance,
                                   /v1/runtime/limits and suggest
                                   config.yaml adjustments.
  bench_main(argv)  llm-bench    — run N streaming requests and report
                                   TTFT/TPS p50/p95/p99.

Per contract, both entrypoints:
  * accept `--config / --url / --api-key / --master-key / --debug / --json`
  * never raise — return non-zero on unreachable/broken server
  * use parse_known_args so pass-through flags never crash
    data sources produce SKIP-style reporting, not invented numbers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


# ── Config loading ──────────────────────────────────────────────────────


def _find_config(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in (
        Path.cwd() / "config.yaml",
        Path.home() / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ):
        if cand.is_file():
            return cand
    return None


def _load_config(explicit: Optional[str]) -> Dict[str, Any]:
    p = _find_config(explicit)
    if p is None or _yaml is None:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    except Exception as exc:
        sys.stderr.write(f"WARNING: could not parse {p}: {exc}\n")
        return {}


def _resolve_url_and_auth(
    cfg: Dict[str, Any], args: argparse.Namespace,
) -> tuple:
    """Return (url, api_key, master_key).
    Precedence: CLI flag > env var > config > default."""
    server_cfg = (cfg or {}).get("server", {}) or {}
    api_cfg = (cfg or {}).get("api", {}) or {}
    url = (
        getattr(args, "url", None)
        or os.environ.get("LLM_SERVER_URL")
        or f"http://{server_cfg.get('host', '127.0.0.1')}:{server_cfg.get('port', '8080')}"
    )
    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("LLM_API_KEY")
        or str(api_cfg.get("key", ""))
    )
    master_key = (
        getattr(args, "master_key", None)
        or os.environ.get("LLM_MASTER_KEY")
        or str(api_cfg.get("master_key", ""))
    )
    if api_key.startswith("${"):
        api_key = ""
    if master_key.startswith("${"):
        master_key = ""
    return url.rstrip("/"), api_key, master_key


# ── HTTP helpers ────────────────────────────────────────────────────────


def _http_get_json(
    url: str, api_key: str, *, timeout: float = 5.0,
) -> Optional[Dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code} {e.reason} for {url}\n")
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        sys.stderr.write(f"Unreachable {url}: {e}\n")
        return None


def _http_post_stream(
    url: str, api_key: str, payload: Dict[str, Any], *, timeout: float,
) -> tuple:
    """POST streaming chat, measure (ttft_s, decode_tps, tokens, err_str)."""
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t_start = time.monotonic()
        first_token_s: Optional[float] = None
        tokens = 0
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                payload_bytes = line[6:].strip()
                if payload_bytes == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload_bytes)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_s is None:
                        first_token_s = time.monotonic() - t_start
                    tokens += 1
        if first_token_s is None or tokens < 1:
            return None, None, 0, "no tokens received"
        total_s = time.monotonic() - t_start
        decode_s = total_s - first_token_s
        decode_tps = (
            (tokens - 1) / decode_s
            if decode_s > 0.05 and tokens > 1 else 0.0
        )
        return first_token_s, decode_tps, tokens, None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        return None, None, 0, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, None, 0, f"{type(e).__name__}: {e}"


def _pct(sorted_vals: List[float], pct: float) -> float:
    """Percentile — same formula as §7 invariant 17."""
    if not sorted_vals:
        return 0.0
    k = int(len(sorted_vals) * pct / 100)
    return float(sorted_vals[min(k, len(sorted_vals) - 1)])


# ── Bench loop ──────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    requests: int
    ok: int
    failed: int
    ttft_ms: List[float]
    decode_tps: List[float]
    total_tokens: int
    errors: List[str]

    def summary(self) -> Dict[str, Any]:
        ttft_sorted = sorted(self.ttft_ms)
        tps_sorted = sorted(self.decode_tps)
        return {
            "requests": self.requests,
            "ok": self.ok,
            "failed": self.failed,
            "ttft_ms_p50": round(_pct(ttft_sorted, 50), 1),
            "ttft_ms_p95": round(_pct(ttft_sorted, 95), 1),
            "ttft_ms_p99": round(_pct(ttft_sorted, 99), 1),
            "decode_tps_p50": round(_pct(tps_sorted, 50), 2),
            "decode_tps_p95": round(_pct(tps_sorted, 95), 2),
            "decode_tps_p99": round(_pct(tps_sorted, 99), 2),
            "total_tokens": self.total_tokens,
            "errors": self.errors[:5],
        }


def _run_bench(
    url: str, api_key: str, model: str,
    n_requests: int, max_tokens: int, timeout: float,
) -> BenchResult:
    ttft: List[float] = []
    tps: List[float] = []
    ok = failed = total = 0
    errors: List[str] = []
    endpoint = f"{url}/v1/chat/completions"
    for i in range(n_requests):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one short sentence."}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
        t_first, t_dec, tok, err = _http_post_stream(
            endpoint, api_key, payload, timeout=timeout,
        )
        if err:
            failed += 1
            errors.append(f"req{i}: {err}")
            continue
        ok += 1
        total += tok
        if t_first is not None:
            ttft.append(t_first * 1000.0)
        if t_dec is not None:
            tps.append(t_dec)
    return BenchResult(
        requests=n_requests, ok=ok, failed=failed,
        ttft_ms=ttft, decode_tps=tps,
        total_tokens=total, errors=errors,
    )

# ── Analysis (tune) ─────────────────────────────────────────────────────

def _reference_tps(model_id: str) -> tuple:
    """(tokens/s, source) from the model's recorded benchmark; (0.0, "") when none."""
    if not model_id:
        return 0.0, ""
    try:
        from local_llm_core.engine.hardware.hardware_profiles import (
            find_bench_record_by_model_id,
        )

        rec = find_bench_record_by_model_id(model_id) or {}
        tps = float(rec.get("decode_tps", 0.0) or 0.0)
        return (tps, "bench") if tps > 0 else (0.0, "")
    except Exception:
        return 0.0, ""

def _analyse(
    cfg: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    limits: Optional[Dict[str, Any]],
    perf: Optional[Dict[str, Any]],
) -> List[str]:
    """Return data-driven recommendations. Never fabricates: each check
    is skipped when its input is missing — absence is honest, not disguised."""
    recs: List[str] = []
    if limits is None and metrics is None and perf is None:
        return recs

    inf_cfg = (cfg or {}).get("inference", {}) or {}
    hp_cfg = (cfg or {}).get("hardware_profiles", {}).get("overrides", {}) or {}
    apu_cfg = hp_cfg.get("apu_2gb", {}) or {}

    measured_tps = 0.0
    if limits:
        measured_tps = float(limits.get("measured_tps") or 0.0)
    elif metrics:
        measured_tps = float((metrics.get("engine") or {}).get("avg_tps") or 0.0)

    # Reference speed: the benchmark recorded for the model that is loaded
    # (written automatically at start-up), else the configured target.
    served_id = str((limits or {}).get("model_id") or "")
    ref_tps, ref_src = _reference_tps(served_id)
    if ref_tps <= 0:
        cfg_target = apu_cfg.get("target_tps") or inf_cfg.get("target_tps")
        if cfg_target:
            ref_tps, ref_src = float(cfg_target), "config"

    if ref_tps > 0 and measured_tps > 0:
        if measured_tps < ref_tps * 0.75:
            if ref_src == "bench":
                recs.append(
                    f"Measured TPS ({measured_tps:.2f}) is well below the benchmark "
                    f"recorded for '{served_id}' ({ref_tps:.2f}). Check the CPU "
                    f"governor and GPU power state (llm-doctor), look for other "
                    f"processes using memory, and refresh the benchmark with the "
                    f"service stopped: llm-calibrate --force."
                )
            else:
                recs.append(
                    f"Measured TPS ({measured_tps:.2f}) is well below the configured "
                    f"target ({ref_tps:.2f}). Run llm-calibrate so the target comes "
                    f"from a measurement, or investigate throughput first (llm-bench)."
                )
        elif measured_tps > ref_tps * 1.25 and ref_src == "bench":
            recs.append(
                f"Measured TPS ({measured_tps:.2f}) exceeds the recorded benchmark "
                f"({ref_tps:.2f}) — the benchmark is stale. Refresh it with "
                f"llm-calibrate --force while the service is stopped."
            )

    if metrics:
        eng = metrics.get("engine") or {}
        ctx_used = int(eng.get("avg_new_prompt_tokens") or 0)
        ctx_total = int((limits or {}).get("context_length") or 0)
        if ctx_used > 0 and ctx_total > 0:
            usage_pct = 100 * ctx_used / ctx_total
            if usage_pct > 75:
                recs.append(
                    f"Average new prompt is {ctx_used} tokens (~{usage_pct:.0f}% "
                    f"of n_ctx={ctx_total}). If many requests hit "
                    f"model_loader.max_context_tokens, consider lowering "
                    f"max_context_tokens to reduce per-request prefill time."
                )

    if metrics:
        vram = metrics.get("vram") or {}
        vram_pct = float(vram.get("utilization_pct") or 0.0)
        if vram_pct > 92:
            recs.append(
                f"VRAM utilisation is {vram_pct:.0f}% — close to the "
                f"profile's vram_target_utilization. If you see allocation "
                f"failures, lower model_loader.vram_safety_margin_mb or "
                f"reduce offloaded layers."
            )

    if limits:
        prefill = float(limits.get("prefill_tps") or 0.0)
        decode = float(limits.get("decode_tps") or 0.0)
        if prefill > 0 and decode > 0 and prefill < decode:
            recs.append(
                f"Prefill TPS ({prefill:.2f}) < decode TPS ({decode:.2f}). "
                f"Expected for long prompts, but consider lowering "
                f"inference.chunked_prefill_chunk_max (currently "
                f"{inf_cfg.get('chunked_prefill_chunk_max', 256)}) if you "
                f"need shorter cancel latency."
            )

    if metrics:
        cache = (metrics.get("engine") or {}).get("context_cache") or {}
        hit_rate = float(cache.get("hit_rate") or 0.0)
        if 0 < hit_rate < 0.15:
            recs.append(
                f"KV context cache hit rate is {hit_rate * 100:.1f}% — "
                f"clients may not be reusing prefixes. If your workload "
                f"shares a system prompt, this is unexpected; if not, "
                f"raise inference.context_cache_bytes to retain more."
            )

    if perf:
        conf = float(perf.get("confidence_index") or 0.0)
        grade = str(perf.get("grade") or "unknown")
        if conf > 0 and conf < 50:
            recs.append(
                f"SLA confidence index is {conf:.1f}% (grade={grade}). "
                f"Recent latencies are near or above SLA targets. Inspect "
                f"/v1/performance per_model for the culprit."
            )

    return recs


# ── CLI plumbing ────────────────────────────────────────────────────────


_COMMON = argparse.ArgumentParser(add_help=False)
_COMMON.add_argument("--config", dest="config_path", default=None,
                     help="Path to config.yaml (default: auto-discover).")
_COMMON.add_argument("--url", default=None,
                     help="Server URL (default: from config or $LLM_SERVER_URL).")
_COMMON.add_argument("--api-key", default=None,
                     help="API key (default: from config or $LLM_API_KEY).")
_COMMON.add_argument("--master-key", default=None,
                     help="Master key (default: from config or $LLM_MASTER_KEY).")
_COMMON.add_argument("--json", action="store_true",
                     help="Emit machine-readable JSON instead of text.")
_COMMON.add_argument("--debug", action="store_true",
                     help="Verbose output to stderr.")
_COMMON.add_argument("--timeout", type=float, default=30.0,
                     help="Per-HTTP-request timeout in seconds (default: 30).")


def _parser_tune() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-tune",
        description="Analyse a running Local LLM Cloud server and "
                    "suggest config.yaml adjustments.",
        parents=[_COMMON],
    )
    p.add_argument("--probe", action="store_true",
                   help="Send a small warmup request to measure real "
                        "throughput before analysing.")
    p.add_argument("--model", default=None,
                   help="Model id for --probe (default: first loaded).")
    return p


def _parser_bench() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-bench",
        description="Run a small streaming benchmark against a running "
                    "Local LLM Cloud server and report TTFT/TPS.",
        parents=[_COMMON],
    )
    p.add_argument("--model", default=None,
                   help="Model id to benchmark (default: first loaded).")
    p.add_argument("--requests", type=int, default=3,
                   help="Number of benchmark requests (default: 3).")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="max_tokens per request (default: 64).")
    return p


def _resolve_model(
    url: str, api_key: str, explicit: Optional[str], timeout: float,
) -> Optional[str]:
    if explicit:
        return explicit
    data = _http_get_json(f"{url}/v1/models", api_key, timeout=timeout)
    if not data:
        return None
    for m in data.get("data") or []:
        if isinstance(m, dict) and m.get("id"):
            return str(m["id"])
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args, unknown = _parser_tune().parse_known_args(argv)
    if args.debug and unknown:
        sys.stderr.write(f"tune: ignoring unrecognised args: {unknown}\n")
    cfg = _load_config(args.config_path)
    url, api_key, _ = _resolve_url_and_auth(cfg, args)

    if args.debug:
        sys.stderr.write(f"tune: url={url}  api_key={'set' if api_key else 'empty'}\n")

    health = _http_get_json(f"{url}/health", api_key, timeout=args.timeout)
    if health is None:
        sys.stderr.write(
            f"ERROR: server unreachable at {url}/health. "
            f"Run `llm-doctor` or check that the service is up.\n"
        )
        return 1

    metrics = _http_get_json(f"{url}/v1/metrics", api_key, timeout=args.timeout)
    limits = _http_get_json(f"{url}/v1/runtime/limits", api_key, timeout=args.timeout)
    perf = _http_get_json(f"{url}/v1/performance", api_key, timeout=args.timeout)

    bench_summary: Optional[Dict[str, Any]] = None
    if args.probe:
        model = _resolve_model(url, api_key, args.model, args.timeout)
        if not model:
            sys.stderr.write("ERROR: --probe requested but no model loaded.\n")
            return 1
        sys.stderr.write(f"probe: running 1 request against '{model}'...\n")
        _pf = _run_bench(
            url, api_key, model, n_requests=1, max_tokens=32, timeout=args.timeout,
        )
        bench_summary = _pf.summary()
        if _pf.ok == 0:
            sys.stderr.write(
                f"ERROR: probe failed: {_pf.errors[0] if _pf.errors else 'unknown'}\n"
            )
            return 1
        if metrics is None:
            metrics = {}
        metrics.setdefault("engine", {})["avg_tps"] = (
            round(_pct(sorted(_pf.decode_tps), 50), 2) if _pf.decode_tps else 0.0
        )

    recs = _analyse(cfg, metrics, limits, perf)

    if args.json:
        print(json.dumps({
            "url": url,
            "health": health,
            "limits": limits,
            "metrics_available": metrics is not None,
            "performance_available": perf is not None,
            "probe": bench_summary,
            "recommendations": recs,
        }, indent=2, default=str))
        return 0

    print()
    print(f"  Local LLM Cloud — tune report for {url}")
    print("  " + "=" * 70)
    if limits:
        print(f"  tier:          {limits.get('tier', '?')}")
        print(f"  profile:       {limits.get('hardware_profile', '?')}")
        print(f"  model:         {limits.get('model_id') or '(none loaded)'}")
        print(f"  measured TPS:  {limits.get('measured_tps', 0):.2f}")
        print(f"  decode TPS:    {limits.get('decode_tps', 0):.2f}")
        print(f"  prefill TPS:   {limits.get('prefill_tps', 0):.2f}")
        print(f"  context:       {limits.get('context_length', 0)} tokens")
    else:
        print("  /v1/runtime/limits unavailable — skipping limit summary.")
    print()
    if bench_summary is not None:
        print("  Probe request")
        print("  " + "-" * 70)
        print(f"    TTFT:        {bench_summary['ttft_ms_p50']:.0f} ms")
        print(f"    decode TPS:  {bench_summary['decode_tps_p50']:.2f}")
        print(f"    tokens:      {bench_summary['total_tokens']}")
        print()
    print(f"  Recommendations ({len(recs)})")
    print("  " + "-" * 70)
    if not recs:
        print("    (no data-driven recommendations — either all values are")
        print("     within target, or insufficient data to form one.)")
    else:
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")
    print()
    return 0


def bench_main(argv: Optional[List[str]] = None) -> int:
    args, unknown = _parser_bench().parse_known_args(argv)
    if args.debug and unknown:
        sys.stderr.write(f"bench: ignoring unrecognised args: {unknown}\n")
    cfg = _load_config(args.config_path)
    url, api_key, _ = _resolve_url_and_auth(cfg, args)

    if args.debug:
        sys.stderr.write(
            f"bench: url={url} requests={args.requests} "
            f"max_tokens={args.max_tokens}\n"
        )

    model = _resolve_model(url, api_key, args.model, args.timeout)
    if not model:
        sys.stderr.write(
            f"ERROR: no model loaded on {url}. Start the server and load a "
            f"model first, or pass --model.\n"
        )
        return 1

    sys.stderr.write(
        f"bench: {args.requests} request(s) to '{model}' "
        f"(max_tokens={args.max_tokens}, timeout={args.timeout}s)\n"
    )
    _t0 = time.monotonic()
    result = _run_bench(
        url, api_key, model,
        n_requests=args.requests, max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    wall_s = time.monotonic() - _t0

    summary = result.summary()
    summary["wall_clock_s"] = round(wall_s, 2)
    summary["model"] = model

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0 if result.failed == 0 else 1

    print()
    print(f"  Local LLM Cloud — bench results ({url})")
    print("  " + "=" * 70)
    print(f"  model:       {model}")
    print(f"  requests:    {result.ok} ok / {result.failed} failed "
          f"(of {result.requests})")
    print(f"  wall clock:  {wall_s:.1f} s")
    print(f"  tokens:      {result.total_tokens}")
    print()
    if result.ok > 0:
        print(f"  TTFT:        p50={summary['ttft_ms_p50']:>6.0f} ms   "
              f"p95={summary['ttft_ms_p95']:>6.0f} ms   "
              f"p99={summary['ttft_ms_p99']:>6.0f} ms")
        print(f"  decode TPS:  p50={summary['decode_tps_p50']:>6.2f}      "
              f"p95={summary['decode_tps_p95']:>6.2f}      "
              f"p99={summary['decode_tps_p99']:>6.2f}")
    if result.errors:
        print()
        print("  Errors (first 5):")
        for e in result.errors[:5]:
            print(f"    - {e}")
    print()
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
    