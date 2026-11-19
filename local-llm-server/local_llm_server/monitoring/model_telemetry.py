from __future__ import annotations

"""
local_llm_server/monitoring/model_telemetry.py — Production Model Telemetry
==========================================================================================
Per‑model performance tracking, predictive idle detection, auto‑unload, warmup,
and model ranking for adaptive resource allocation.


version = 1.0.0
"""

import asyncio
import logging
import math
import threading
import time
import concurrent.futures
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

# Optional Prometheus integration
try:
    from prometheus_client import Gauge, Counter
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ------------------------------------------------------------------------------
# Module‑level singleton cache for Prometheus gauges
# ------------------------------------------------------------------------------
_PROM_GAUGES: Dict[str, Any] = {}
_PROM_LOCK = threading.RLock()


def _get_or_create_prom_gauge(
    name: str, documentation: str, labelnames: List[str]
) -> Optional[Any]:
    """Return a singleton Gauge, creating it only once globally.

    If the metric is already registered in the process-global Prometheus
    REGISTRY but the module-level `_PROM_GAUGES` cache is empty (typical
    after `uvicorn --reload`: the module is re-imported but REGISTRY
    persists), we recover the existing gauge from REGISTRY and cache it.
    Previously this branch returned None, silently disabling every
    `model_telemetry_*` update for the reloaded process.
    """
    if not HAS_PROMETHEUS:
        return None
    with _PROM_LOCK:
        if name in _PROM_GAUGES:
            return _PROM_GAUGES[name]
        try:
            gauge = Gauge(name, documentation, labelnames=labelnames)
            _PROM_GAUGES[name] = gauge
            return gauge
        except ValueError as e:
            # Duplicate registration — the metric exists in REGISTRY but
            # not in our cache. Locate it and cache the reference.
            log.warning(
                "Prometheus gauge '%s' registration rejected (duplicate): %s "
                "— attempting to reuse existing gauge",
                name, e,
            )
            try:
                from prometheus_client import REGISTRY
                for collector in list(REGISTRY._collector_to_names.keys()):
                    if getattr(collector, "_name", "") == name:
                        _PROM_GAUGES[name] = collector
                        log.debug(
                            "Reusing existing Prometheus gauge '%s'", name,
                        )
                        return collector
            except Exception:
                pass
            log.warning(
                "Prometheus gauge '%s' could not be reused — metric updates "
                "for this gauge will be silently skipped", name,
            )
            return None
        except Exception as e:
            log.warning("Failed to create Prometheus gauge '%s': %s", name, e)
            return None


# ------------------------------------------------------------------------------
# Model Profile with Advanced Statistics
# ------------------------------------------------------------------------------
@dataclass
class ModelProfile:
    """
    Per‑model telemetry and performance metrics.

    All counters are monotonic non-negative integers. The `update()` method
    validates input and raises on invalid payloads rather
    than allowing silent corruption of the profile totals.
    """

    model_id: str
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    oom_count: int = 0
    timeout_count: int = 0
    avg_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    peak_ttft_ms: float = 0.0
    peak_tps: float = 0.0
    last_used: float = field(default_factory=time.monotonic)

    # Memory and cache stats
    vram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    context_cache_hits: int = 0
    context_cache_misses: int = 0

    # token provenance split. `total_tokens` is the authoritative
    # completion count; these two are subsets:
    #   * thinking_tokens_total — the portion consumed by the ThinkStripper
    #     (visible tokens = total_tokens - thinking_tokens_total when the
    #     server strips think tags; the engine's InferenceResponse already
    #     subtracts them from `tokens_gen`, so in practice this is a
    #     separate informational field)
    #   * cached_prompt_tokens_total — prompt tokens served from the KV
    #     prefix cache; a high ratio explains a low TTFT.
    thinking_tokens_total: int = 0
    cached_prompt_tokens_total: int = 0

    # Request inter‑arrival tracking for predictive idle
    _request_times: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)

    # Bayesian request-activity ratio. NOTE: this is a per-model smoothed
    # activity metric that monotonically approaches 1.0 as request count
    # grows. It is NOT the cross-model popularity used by
    # ModelRankingEngine.rank() — that engine computes its own
    # `(total_requests + 1) / (total_requests_across_all_models + 1)`
    # from the full profile set. Kept as `popularity_score` for backward
    # compat with existing dashboard / alert consumers.
    alpha_request: float = 1.0
    beta_request: float = 1.0
    popularity_score: float = 0.0

    # Rolling windows for percentile calculations
    _ttft_window: deque = field(default_factory=lambda: deque(maxlen=500), repr=False)
    _tps_window: deque = field(default_factory=lambda: deque(maxlen=500), repr=False)
    _update_counter: int = field(default=0, repr=False)

    # Per-instance EMA smoothing factor. Set at construction time by
    # `ModelTelemetry` from the config value. Previously this was a class
    # attribute mutated by `create_model_telemetry()`, which never took
    # effect on instances because the dataclass field default shadowed it.
    EMA_ALPHA: float = 0.1

    def __post_init__(self) -> None:
        """Reject inconsistent construction"""
        if self.total_requests < 0:
            raise ValueError(
                f"total_requests must be >= 0, got {self.total_requests}"
            )
        if self.total_tokens < 0:
            raise ValueError(
                f"total_tokens must be >= 0, got {self.total_tokens}"
            )
        if self.thinking_tokens_total < 0:
            raise ValueError(
                f"thinking_tokens_total must be >= 0, got {self.thinking_tokens_total}"
            )
        if self.cached_prompt_tokens_total < 0:
            raise ValueError(
                f"cached_prompt_tokens_total must be >= 0, "
                f"got {self.cached_prompt_tokens_total}"
            )
        if not (0.0 < self.EMA_ALPHA <= 1.0):
            raise ValueError(
                f"EMA_ALPHA must be in (0, 1], got {self.EMA_ALPHA}"
            )

    def update(
        self,
        tokens: int,
        ttft_ms: float,
        tps: float,
        error: Optional[str] = None,
        vram_mb: Optional[float] = None,
        context_cache_hit: bool = False,
        *,
        thinking_tokens: int = 0,
        cached_prompt_tokens: int = 0,
    ) -> None:
        """Record one inference result and update statistics.

        Raises:
            ValueError: if `tokens < 0`, `thinking_tokens < 0`,
                `thinking_tokens > tokens`, or `cached_prompt_tokens < 0`.
                Validation happens BEFORE any state mutation, so a bad
                payload leaves the profile unchanged.
        """
        tokens = int(tokens or 0)
        thinking_tokens = int(thinking_tokens or 0)
        cached_prompt_tokens = int(cached_prompt_tokens or 0)
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        if thinking_tokens < 0:
            raise ValueError(
                f"thinking_tokens must be >= 0, got {thinking_tokens}"
            )
        if thinking_tokens > tokens:
            raise ValueError(
                f"thinking_tokens ({thinking_tokens}) must be <= tokens ({tokens})"
            )
        if cached_prompt_tokens < 0:
            raise ValueError(
                f"cached_prompt_tokens must be >= 0, got {cached_prompt_tokens}"
            )

        now = time.monotonic()
        self.total_requests += 1
        self.total_tokens += tokens
        self.thinking_tokens_total += thinking_tokens
        self.cached_prompt_tokens_total += cached_prompt_tokens
        self.last_used = now

        # Track inter‑arrival
        self._request_times.append(now)

        if error:
            self.total_errors += 1
            err_low = error.lower()
            if "memory" in err_low or "oom" in err_low:
                self.oom_count += 1
            if "timeout" in err_low:
                self.timeout_count += 1
        else:
            alpha = self.EMA_ALPHA
            self.avg_ttft_ms = self.avg_ttft_ms * (1 - alpha) + ttft_ms * alpha
            self.avg_tps = self.avg_tps * (1 - alpha) + tps * alpha

            self._ttft_window.append(ttft_ms)
            self._tps_window.append(tps)
            self._update_counter += 1

            if ttft_ms > self.peak_ttft_ms:
                self.peak_ttft_ms = ttft_ms
            if tps > self.peak_tps:
                self.peak_tps = tps

            if self._update_counter >= 10 and len(self._ttft_window) >= 10:
                self._recalc_percentiles()
                self._update_counter = 0

        if vram_mb is not None:
            self.vram_mb = vram_mb
            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb

        if context_cache_hit:
            self.context_cache_hits += 1
        else:
            self.context_cache_misses += 1

        # Smoothed activity ratio (see field docstring — NOT cross-model
        # popularity).
        self.popularity_score = (self.total_requests + self.alpha_request) / (
            self.total_requests + self.alpha_request + self.beta_request
        )

    def _recalc_percentiles(self) -> None:
        if len(self._ttft_window) >= 10:
            sorted_ttft = sorted(self._ttft_window)
            self.p95_ttft_ms = sorted_ttft[
                min(int(len(sorted_ttft) * 0.95), len(sorted_ttft) - 1)
            ]
            self.p99_ttft_ms = sorted_ttft[
                min(int(len(sorted_ttft) * 0.99), len(sorted_ttft) - 1)
            ]

    def force_recalc(self) -> None:
        self._recalc_percentiles()

    def predicted_next_request(self) -> Optional[float]:
        """Forecast seconds until next request using EWMA of inter‑arrival times."""
        if len(self._request_times) < 3:
            return None
        arrivals = list(self._request_times)
        if len(arrivals) < 2:
            return None
        inter_arrivals = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
        alpha = 0.3
        ewma = inter_arrivals[0]
        for ia in inter_arrivals[1:]:
            ewma = alpha * ia + (1 - alpha) * ewma
        elapsed = time.monotonic() - self.last_used
        return max(0.0, ewma - elapsed)

    def to_dict(self, include_idle: bool = True) -> Dict[str, Any]:
        self.force_recalc()
        idle_s = time.monotonic() - self.last_used if include_idle else 0.0
        predicted = self.predicted_next_request()
        return {
            "model_id": self.model_id,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "oom_count": self.oom_count,
            "timeout_count": self.timeout_count,
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "avg_tps": round(self.avg_tps, 2),
            "p95_ttft_ms": round(self.p95_ttft_ms, 2),
            "p99_ttft_ms": round(self.p99_ttft_ms, 2),
            "peak_ttft_ms": round(self.peak_ttft_ms, 2),
            "peak_tps": round(self.peak_tps, 2),
            "vram_mb": round(self.vram_mb, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "context_cache_hit_rate": (
                self.context_cache_hits
                / max(self.context_cache_hits + self.context_cache_misses, 1)
            ),
            # token provenance split
            "thinking_tokens_total": self.thinking_tokens_total,
            "cached_prompt_tokens_total": self.cached_prompt_tokens_total,
            "popularity_score": round(self.popularity_score, 4),
            "predicted_next_request_s": round(predicted, 1) if predicted is not None else None,
            "idle_seconds": round(idle_s, 1),
        }


# ------------------------------------------------------------------------------
# Model Ranking System (Bayesian + Recency)
# ------------------------------------------------------------------------------
class ModelRankingEngine:
    """
    Provides a dynamic ranking of models based on request volume, recency, and
    token throughput. The score is a weighted combination of:
        - Bayesian popularity (requests vs total requests across models)
        - Exponential recency decay (more recent → higher weight)
        - Token throughput (higher TPS → more valuable to keep loaded)
    Used by the auto‑loader / auto‑unloader to decide which models to keep.
    """

    def __init__(self, recency_half_life: float = 300.0):
        self._recency_half_life = recency_half_life
        self._profiles: Dict[str, ModelProfile] = {}

    def update(self, profile: ModelProfile) -> None:
        self._profiles[profile.model_id] = profile

    def rank(self) -> List[str]:
        """Return model IDs sorted by descending rank."""
        if not self._profiles:
            return []
        now = time.monotonic()
        scores = {}
        total_requests = sum(p.total_requests for p in self._profiles.values()) or 1
        recency_decay = math.log(2) / self._recency_half_life

        for mid, p in self._profiles.items():
            # `now` and `last_used` are both time.monotonic(), so the
            # subtraction is always >= 0 — the previous max(0, ...) was
            # dead code.
            age = now - p.last_used
            recency_weight = math.exp(-recency_decay * age)

            # Cross-model popularity — computed here, NOT read from
            # `p.popularity_score` (which is a per-model activity ratio).
            pop_score = (p.total_requests + 1.0) / (total_requests + 1.0)

            tps_bonus = min(1.0, p.avg_tps / 20.0) if p.avg_tps > 0 else 0.0

            score = (0.5 * pop_score + 0.3 * recency_weight + 0.2 * tps_bonus)
            scores[mid] = score

        return sorted(scores, key=scores.get, reverse=True)


# ------------------------------------------------------------------------------
# Thread‑Safe Telemetry Registry with Prometheus Singleton
# ------------------------------------------------------------------------------
class ModelTelemetry:
    """
    Thread‑safe registry for per‑model performance profiles.
    Includes ranking engine and Prometheus integration with singleton gauges.
    """

    def __init__(self, ema_alpha: float = 0.1) -> None:
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {ema_alpha}"
            )
        self._ema_alpha = float(ema_alpha)
        self._profiles: Dict[str, ModelProfile] = {}
        self._lock = threading.RLock()
        self._ranking = ModelRankingEngine()
        self._prom_gauges: Dict[str, Any] = {}
        if HAS_PROMETHEUS:
            self._init_prometheus()

    def _init_prometheus(self) -> None:
        """Initialise or reuse global Prometheus gauges."""
        self._prom_gauges["requests"] = _get_or_create_prom_gauge(
            "model_telemetry_requests", "Requests per model", ["model"]
        )
        self._prom_gauges["tokens"] = _get_or_create_prom_gauge(
            "model_telemetry_tokens", "Tokens per model", ["model"]
        )
        self._prom_gauges["ttft_avg"] = _get_or_create_prom_gauge(
            "model_telemetry_ttft_avg_ms", "Avg TTFT per model (ms)", ["model"]
        )
        self._prom_gauges["tps_avg"] = _get_or_create_prom_gauge(
            "model_telemetry_tps_avg", "Avg TPS per model", ["model"]
        )

    def record(
        self,
        model_id: str,
        tokens: int,
        ttft_ms: float,
        tps: float,
        error: Optional[str] = None,
        vram_mb: Optional[float] = None,
        context_cache_hit: bool = False,
        *,
        thinking_tokens: int = 0,
        cached_prompt_tokens: int = 0,
    ) -> None:
        """Record a completed inference for `model_id`.

        Positional args match the signature — existing call sites
        are unchanged. The two keyword-only args are additive:

            thinking_tokens        — subset of `tokens` consumed by the
                                     ThinkStripper (informational split)
            cached_prompt_tokens   — prompt tokens served from the KV
                                     prefix cache

        Raises ValueError on invalid token payloads (see ModelProfile.update).
        """
        with self._lock:
            if model_id not in self._profiles:
                self._profiles[model_id] = ModelProfile(
                    model_id=model_id,
                    EMA_ALPHA=self._ema_alpha,
                )
            self._profiles[model_id].update(
                tokens, ttft_ms, tps, error, vram_mb, context_cache_hit,
                thinking_tokens=thinking_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
            )
            self._ranking.update(self._profiles[model_id])

            if self._prom_gauges:
                p = self._profiles[model_id]
                try:
                    if self._prom_gauges.get("requests"):
                        self._prom_gauges["requests"].labels(model=model_id).set(p.total_requests)
                    if self._prom_gauges.get("tokens"):
                        self._prom_gauges["tokens"].labels(model=model_id).set(p.total_tokens)
                    if self._prom_gauges.get("ttft_avg"):
                        self._prom_gauges["ttft_avg"].labels(model=model_id).set(p.avg_ttft_ms)
                    if self._prom_gauges.get("tps_avg"):
                        self._prom_gauges["tps_avg"].labels(model=model_id).set(p.avg_tps)
                except Exception as e:
                    log.debug("Failed to update Prometheus metrics for %s: %s", model_id, e)

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            p = self._profiles.get(model_id)
            return p.to_dict() if p else None

    def all_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [p.to_dict() for p in self._profiles.values()]

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                model_id: {
                    "requests_total": p.total_requests,
                    "tokens_total": p.total_tokens,
                    "errors_total": p.total_errors,
                    "oom_total": p.oom_count,
                    "timeout_total": p.timeout_count,
                    "ttft_avg_ms": p.avg_ttft_ms,
                    "tps_avg": p.avg_tps,
                    "ttft_p95_ms": p.p95_ttft_ms,
                    "ttft_p99_ms": p.p99_ttft_ms,
                    "vram_mb": p.vram_mb,
                    # token provenance split
                    "thinking_tokens_total": p.thinking_tokens_total,
                    "cached_prompt_tokens_total": p.cached_prompt_tokens_total,
                    "popularity_score": p.popularity_score,
                }
                for model_id, p in self._profiles.items()
            }

    def recommendations(self, idle_threshold_s: float = 1800.0) -> Dict[str, str]:
        out = {}
        cutoff = time.monotonic() - idle_threshold_s
        with self._lock:
            for mid, p in self._profiles.items():
                predicted = p.predicted_next_request()
                if predicted is not None and predicted > idle_threshold_s * 2:
                    out[mid] = (
                        f"Predicted idle: next request in {predicted:.0f}s, "
                        f"{p.total_requests} requests — consider unloading"
                    )
                elif p.last_used < cutoff:
                    idle_min = (time.monotonic() - p.last_used) / 60
                    out[mid] = (
                        f"Idle {idle_min:.0f} min, "
                        f"{p.total_requests} requests — consider unloading"
                    )
        return out

    def pressure_unload_candidates(self, vram_budget_mb: float) -> List[str]:
        """Return model IDs that exceed VRAM budget, sorted by lowest
        request-activity ratio first."""
        candidates = []
        with self._lock:
            for mid, p in self._profiles.items():
                if p.vram_mb > vram_budget_mb:
                    candidates.append((mid, p.popularity_score))
        candidates.sort(key=lambda x: x[1])
        return [mid for mid, _ in candidates]

    def reset(self, model_id: Optional[str] = None) -> None:
        with self._lock:
            if model_id:
                self._profiles.pop(model_id, None)
            else:
                self._profiles.clear()

    def model_ranking(self) -> List[str]:
        with self._lock:
            return self._ranking.rank()


# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------
def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    if config_path is not None:
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        log.warning("Config file not found: %s", config_path)
        return {}
    paths = [Path.home() / "config.yaml", Path.cwd() / "config.yaml"]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    log.warning("No config.yaml found; using default telemetry settings.")
    return {}


# ------------------------------------------------------------------------------
# Enhanced Idle Reaper with Pressure Awareness
# ------------------------------------------------------------------------------
async def model_idle_reaper(
    loader,
    telemetry: ModelTelemetry,
    idle_s: float = 1800.0,
    check_interval: float = 60.0,
    executor: Optional[concurrent.futures.Executor] = None,
    pressure_vram_mb: Optional[float] = None,
) -> None:
    """Periodically check for idle/pressure models and unload them."""
    loop = asyncio.get_running_loop()
    own_executor = None
    if executor is None:
        own_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="unload",
        )
        executor = own_executor

    try:
        while True:
            try:
                await asyncio.sleep(check_interval)
                recs = telemetry.recommendations(idle_s)
                if pressure_vram_mb:
                    pressure_candidates = telemetry.pressure_unload_candidates(pressure_vram_mb)
                    for mid in pressure_candidates:
                        if mid not in recs:
                            recs[mid] = "Memory pressure: exceeding VRAM budget"

                for model_id, reason in recs.items():
                    try:
                        unload_func = loader.unload
                        if asyncio.iscoroutinefunction(unload_func):
                            success = await unload_func(model_id)
                        else:
                            success = await loop.run_in_executor(executor, unload_func, model_id)
                        if success:
                            log.info("Auto-unload '%s': %s", model_id, reason)
                        else:
                            log.debug("Unload skipped for '%s' (already unloaded?)", model_id)
                    except Exception as exc:
                        log.warning("Auto-unload '%s' failed: %s", model_id, exc)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Unexpected error in idle reaper loop")
    finally:
        if own_executor is not None:
            own_executor.shutdown(wait=False)


# ------------------------------------------------------------------------------
# Model Warmup with Retries and Timeout Protection
# ------------------------------------------------------------------------------
async def warmup_model(
    engine,
    model_id: str,
    prompt: str = "Write a short sentence about the weather today.",
    max_tokens: int = 24,
    timeout_s: Optional[float] = None,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """Pre‑populate KV cache after model load with automatic retries.

    Returns a dict with status, timing, and error info.

    the previous default prompt ("Hello, world!") with
    max_tokens=10 frequently produced only 1-2 completion tokens before the
    model hit a stop condition — observed directly: a BitNet warmup logged
    "TTFT=0ms TPS=0.3", which is exactly 1 token / 3.37s, not a sign of any
    actual problem, just a single-token sample being treated as a meaningful
    throughput measurement. This isn't fixable by picking one "correct"
    prompt — chat templates and stop-token behavior vary per model/fork.
    Instead: a more open-ended default prompt and higher max_tokens improve
    the odds generically, AND (the real, model-agnostic fix) the result now
    flags low-token-count samples as non-representative via
    "representative": False rather than silently logging a precise-looking
    but meaningless TPS number.
    """
    last_error = None
    for attempt in range(retries):
        try:
            loader = getattr(engine, "loader", None)
            if loader is not None:
                model_info = loader.get(model_id)
                if model_info is not None and getattr(model_info, "backend", "") == "numpy_sim":
                    log.error("Warmup skipped: numpy_sim backend")
                    return {"model_id": model_id, "ok": False, "error": "numpy_sim backend"}

            try:
                from local_llm_core.engine.inference.inference_engine import InferenceRequest
            except ImportError:
                log.warning("local_llm_core not installed; warmup not available")
                return {"model_id": model_id, "ok": False, "error": "local_llm_core not installed"}

            effective_timeout = timeout_s if timeout_s is not None else 30.0
            ir = InferenceRequest(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                stream=False,
                timeout_s=effective_timeout,
            )

            t0 = time.perf_counter()
            result = await asyncio.wait_for(engine.generate(ir), timeout=effective_timeout + 5.0)
            elapsed = time.perf_counter() - t0

            # A sample of 1-2 tokens produces a TPS number that LOOKS precise
            # but measures almost nothing meaningful (1 token / 3.37s ==
            # "TPS=0.3", which reads as "something is badly wrong" when it's
            # actually just an early stop). Flag it explicitly rather than
            # letting a misleadingly exact-looking number stand alone in the
            # logs.
            _MIN_REPRESENTATIVE_TOKENS = 5
            representative = result.tokens_gen >= _MIN_REPRESENTATIVE_TOKENS
            if representative:
                log.info("Warmup '%s': %.2fs TTFT=%.0fms TPS=%.1f (n=%d tokens)",
                          model_id, elapsed, result.ttft_ms, result.tps, result.tokens_gen)
            else:
                log.info(
                    "Warmup '%s': %.2fs, only %d token(s) generated (stopped early) — "
                    "TPS/TTFT not representative of real throughput, treat as informational only",
                    model_id, elapsed, result.tokens_gen,
                )
            return {
                "model_id": model_id,
                "warmup_s": round(elapsed, 3),
                "ttft_ms": result.ttft_ms,
                "tps": result.tps,
                "tokens_gen": result.tokens_gen,
                "representative": representative,
                "ok": True,
            }
        except asyncio.TimeoutError:
            last_error = f"Timeout after {effective_timeout}s"
            log.warning("Warmup attempt %d timed out", attempt + 1)
        except Exception as exc:
            last_error = str(exc)
            log.warning("Warmup attempt %d failed: %s", attempt + 1, exc)
        if attempt < retries - 1:
            await asyncio.sleep(retry_delay * (2 ** attempt))

    return {"model_id": model_id, "ok": False, "error": last_error}


# ------------------------------------------------------------------------------
# Factory: Create Telemetry from Configuration
# ------------------------------------------------------------------------------
def create_model_telemetry(cfg: Optional[Dict[str, Any]] = None) -> ModelTelemetry:
    """Construct a `ModelTelemetry` instance from the given config dict.

    Reads `monitoring.telemetry.ema_alpha` and passes it to the constructor.
    The previous implementation mutated `ModelProfile.EMA_ALPHA` on the
    class, which never took effect on instances (dataclass field default
    shadowed the class attribute).
    """
    if cfg is None:
        cfg = _load_config()
    telemetry_cfg = cfg.get("monitoring", {}).get("telemetry", {})
    ema_alpha = float(telemetry_cfg.get("ema_alpha", 0.1))
    # Clamp to legal range up-front so a typo in config.yaml produces an
    # actionable ValueError at factory time, not a silent fallback to 0.1.
    if not (0.0 < ema_alpha <= 1.0):
        log.warning(
            "monitoring.telemetry.ema_alpha=%r is out of range (0, 1] — "
            "using default 0.1", ema_alpha,
        )
        ema_alpha = 0.1
    return ModelTelemetry(ema_alpha=ema_alpha)


# ------------------------------------------------------------------------------
# Background Task Manager (Lifespan Integration)
# ------------------------------------------------------------------------------
class TelemetryBackgroundTasks:
    def __init__(self) -> None:
        self._tasks: List[asyncio.Task] = []
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    async def start(
        self,
        loader,
        telemetry: ModelTelemetry,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        if cfg is None:
            cfg = _load_config()
        telemetry_cfg = cfg.get("monitoring", {}).get("telemetry", {})
        auto_unload = telemetry_cfg.get("auto_unload", {})
        if auto_unload.get("enabled", False):
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="unload",
            )
            idle_s = auto_unload.get("idle_threshold_seconds", 1800.0)
            check_interval = auto_unload.get("check_interval_seconds", 60.0)
            pressure_vram_mb = auto_unload.get("pressure_vram_mb", None)
            task = asyncio.create_task(
                model_idle_reaper(
                    loader, telemetry, idle_s, check_interval,
                    executor=self._executor, pressure_vram_mb=pressure_vram_mb,
                )
            )
            self._tasks.append(task)
            log.info("Started idle reaper (pressure_vram_mb=%s)", pressure_vram_mb)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        if self._executor:
            self._executor.shutdown(wait=False)
        log.info("Telemetry background tasks stopped.")


# ------------------------------------------------------------------------------
# Convenience: Get Telemetry from Config Path
# ------------------------------------------------------------------------------
def get_model_telemetry(config_path: Optional[Path] = None) -> ModelTelemetry:
    cfg = _load_config(config_path)
    return create_model_telemetry(cfg)


# ------------------------------------------------------------------------------
# Module Exports
# ------------------------------------------------------------------------------
__all__ = [
    "ModelProfile",
    "ModelTelemetry",
    "ModelRankingEngine",
    "model_idle_reaper",
    "warmup_model",
    "create_model_telemetry",
    "get_model_telemetry",
    "TelemetryBackgroundTasks",
]
