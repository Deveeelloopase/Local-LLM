from __future__ import annotations

"""
local_llm_server/monitoring/metrics_collector.py — Production Metrics Collector 
====================================================================================================
Collects system and inference metrics, optionally exposes Prometheus endpoint.


version = 1.0.0
"""

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import yaml

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------
def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, with environment variable overrides."""
    cfg: Dict[str, Any] = {}

    if config_path is not None:
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            log.warning("Config file not found: %s", config_path)
    else:
        paths = [Path.home() / "config.yaml", Path.cwd() / "config.yaml"]
        for p in paths:
            if p.exists():
                with open(p, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                break
        else:
            log.warning("No config.yaml found; using default metrics settings.")

    env_prefix = "LLM_METRICS_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            cfg_key = key[len(env_prefix):].lower()
            try:
                cfg.setdefault("monitoring", {}).setdefault("metrics", {})[cfg_key] = int(value)
            except ValueError:
                try:
                    cfg.setdefault("monitoring", {}).setdefault("metrics", {})[cfg_key] = float(value)
                except ValueError:
                    lower_val = value.lower()
                    if lower_val in ("true", "yes", "1", "on"):
                        cfg.setdefault("monitoring", {}).setdefault("metrics", {})[cfg_key] = True
                    elif lower_val in ("false", "no", "0", "off"):
                        cfg.setdefault("monitoring", {}).setdefault("metrics", {})[cfg_key] = False
                    else:
                        cfg.setdefault("monitoring", {}).setdefault("metrics", {})[cfg_key] = value

    return cfg


# ------------------------------------------------------------------------------
# Percentile helper (identical formula to pool.py's _percentile)
# ------------------------------------------------------------------------------
def _percentile(sorted_data: List[float], pct: float) -> float:
    """Index-safe percentile from a pre-sorted list. Empty → 0.0.

    Uses the same formula as `SharedWeightProcessPool._percentile` so
    metrics_collector's p50/p95 for global TTFT/TPS match the values
    `pool.stats()` reports — a consumer that branches on `stats_source`
    sees consistent numbers across both surfaces.
    """
    if not sorted_data:
        return 0.0
    k = int(len(sorted_data) * pct / 100)
    return float(sorted_data[min(k, len(sorted_data) - 1)])


# ------------------------------------------------------------------------------
# No-Op Prometheus Metrics (Prevents AttributeError)
# ------------------------------------------------------------------------------
class _NoOpMetric:
    """No-op metric that silently accepts calls."""
    def labels(self, *args, **kwargs):
        return self
    def inc(self, amount=1):
        pass
    def set(self, value):
        pass
    def observe(self, value):
        pass


# ------------------------------------------------------------------------------
# Sample Data Class
# ------------------------------------------------------------------------------
@dataclass
class Sample:
    ts: float
    cpu_pct: float
    ram_pct: float
    vram_pct: float
    gpu_temp_c: Optional[float]
    gpu_util_pct: Optional[float]
    gpu_power_w: Optional[float]
    req_rate: float
    avg_latency_ms: float
    avg_tps: float
    vram_fragmentation_pct: float = 0.0
    adaptive_interval_factor: float = 1.0
    # `Any` (was `float`) so the enriched per_model dict — which now carries
    # `tokens_total: int` and `sample_count: int` alongside the float means
    # and percentiles — fits without a lie in the type hint.
    per_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # additions
    tokens_total: int = 0
    errors_total: int = 0
    ttft_ms_p50: float = 0.0
    ttft_ms_p95: float = 0.0
    tps_p50: float = 0.0
    tps_p95: float = 0.0


# ------------------------------------------------------------------------------
# Thermal state dataclass
# ------------------------------------------------------------------------------
@dataclass
class ThermalState:
    """Shared, thread-safe thermal state holder."""
    throttling: bool = False
    temperature_c: float = 0.0
    timestamp: float = 0.0


# ------------------------------------------------------------------------------
# Online Statistics (Welford's algorithm + Exponential Smoothing)
# ------------------------------------------------------------------------------
class RunningStats:
    """Welford's online algorithm for mean, variance, and exponential smoothed mean."""
    def __init__(self, alpha: float = 0.2):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.ewma = 0.0
        self.alpha = alpha

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.count == 1:
            self.ewma = value
        else:
            self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

    @property
    def variance(self) -> float:
        return self.m2 / max(self.count - 1, 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5


class HoltWintersStats:
    """Double exponential smoothing (Holt's linear trend) for forecasting."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level: Optional[float] = None
        self.trend: float = 0.0
        self.last_forecast: Optional[float] = None
        self.mse: float = 1.0

    def update(self, value: float) -> float:
        if self.level is None:
            self.level = value
            self.last_forecast = value
            return 0.0
        last_level = self.level
        self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        forecast = self.level + self.trend
        error = value - forecast
        self.mse = 0.9 * self.mse + 0.1 * (error ** 2)
        self.last_forecast = forecast
        return error

    @property
    def prediction(self) -> float:
        return self.last_forecast if self.last_forecast is not None else 0.0

    @property
    def prediction_std(self) -> float:
        return math.sqrt(self.mse)


class PredictiveAnomalyDetector:
    """Uses Holt‑Winters to forecast metrics and alerts on unexpected deviation."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, k: float = 3.0):
        self._models: Dict[str, HoltWintersStats] = {}
        self._alpha = alpha
        self._beta = beta
        self._k = k
        self._min_samples = 20

    def check(self, metric_name: str, value: float, current_time: float) -> Optional[str]:
        model = self._models.get(metric_name)
        if model is None:
            model = HoltWintersStats(self._alpha, self._beta)
            self._models[metric_name] = model
        err = model.update(value)
        if model.level is not None and model.last_forecast is not None:
            if abs(err) > self._k * model.prediction_std:
                return (f"Anomaly predicted for {metric_name}: value={value:.2f} "
                        f"(expected {model.prediction:.2f} ± {model.prediction_std:.2f})")
        return None


# ------------------------------------------------------------------------------
# Metrics File Exporter (async‑friendly)
# ------------------------------------------------------------------------------
class MetricsFileExporter:
    """Writes each Sample as JSON line to a file, using a background thread."""
    def __init__(self, file_path: str, flush_interval: float = 1.0):
        self._path = Path(file_path)
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def enqueue(self, sample: Sample) -> None:
        with self._lock:
            self._queue.append(sample)

    def _writer(self) -> None:
        while not self._stop.is_set():
            batch = []
            with self._lock:
                while self._queue:
                    batch.append(self._queue.popleft())
            if batch:
                try:
                    with open(self._path, "a", encoding="utf-8") as f:
                        for s in batch:
                            f.write(json.dumps(self._sample_to_dict(s)) + "\n")
                except Exception as e:
                    log.error("Metrics file export error: %s", e)
            self._stop.wait(self._flush_interval)

    @staticmethod
    def _sample_to_dict(s: Sample) -> Dict[str, Any]:
        return {
            "ts": s.ts,
            "cpu_pct": s.cpu_pct,
            "ram_pct": s.ram_pct,
            "vram_pct": s.vram_pct,
            "gpu_temp_c": s.gpu_temp_c,
            "gpu_util_pct": s.gpu_util_pct,
            "gpu_power_w": s.gpu_power_w,
            "req_rate": s.req_rate,
            "avg_latency_ms": s.avg_latency_ms,
            "avg_tps": s.avg_tps,
            "vram_fragmentation_pct": s.vram_fragmentation_pct,
            "tokens_total": s.tokens_total,
            "errors_total": s.errors_total,
            "ttft_ms_p50": s.ttft_ms_p50,
            "ttft_ms_p95": s.ttft_ms_p95,
            "tps_p50": s.tps_p50,
            "tps_p95": s.tps_p95,
        }

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


# ------------------------------------------------------------------------------
# Metrics Collector
# ------------------------------------------------------------------------------
class MetricsCollector:
    """
    Collects and aggregates system and inference metrics with advanced anomaly detection.

    Config options (from config.yaml -> monitoring.metrics):
        sample_interval_sec: float = 5.0
        history_size: int = 1200
        latency_window_size: int = 500          # maxlen for global TTFT/TPS deques
        per_model_window_size: int = 500        # maxlen for per-model TTFT/TPS deques
        max_per_model_entries: int = 200        # LFU cap for per-model state
        prometheus_enabled: bool = False
        prometheus_port: int = 9090
        anomaly_detection: bool = True
        ...
    """

    _started_servers: set = set()
    _gpu_cache: Dict[str, Tuple[float, Any]] = {}
    _gpu_cache_ttl = 5.0

    # `shutil.which("nvidia-smi")` walks every entry on `$PATH` and calls
    # `os.access` on each candidate — ~1 ms per call on a 20-entry PATH.
    # Called 4+ times per 5 s sample tick from the four `_get_gpu_*`
    # static helpers, that is a measurable fraction of the collector
    # thread's time on hosts without NVIDIA (the vast majority of
    # ticks on this deployment). Cache the answer once per process.
    _nvidia_smi_path: Optional[str] = None
    _nvidia_smi_checked: bool = False

    @classmethod
    def _which_nvidia_smi(cls) -> Optional[str]:
        if not cls._nvidia_smi_checked:
            cls._nvidia_smi_path = shutil.which("nvidia-smi")
            cls._nvidia_smi_checked = True
        return cls._nvidia_smi_path

    def __init__(
        self,
        sample_interval: float = 5.0,
        history_size: int = 1200,
        latency_window_size: int = 500,
        per_model_window_size: int = 500,
        max_per_model_entries: int = 200,
        prometheus_enabled: bool = False,
        prometheus_port: int = 9090,
        anomaly_detection: bool = True,
        anomaly_latency_z_threshold: float = 5.0,
        anomaly_temp_threshold_c: float = 85.0,
        anomaly_min_gpu_util_for_temp_alert: float = 10.0,
        gpu_temp_max_valid: float = 110.0,
        gpu_cache_ttl: float = 5.0,
        adaptive_sampling: bool = True,
        adaptive_sampling_max_factor: float = 4.0,
        adaptive_sampling_cpu_threshold: float = 80.0,
        file_export_enabled: bool = False,
        file_export_path: str = "/tmp/llm_metrics.jsonl",
    ) -> None:
        self._base_interval = sample_interval
        self._history: deque[Sample] = deque(maxlen=history_size)
        self._lock = threading.RLock()

        # Rolling counters for request rate
        self._req_count = 0
        self._last_req_ts = time.monotonic()

        # Global latency/tps windows. Configurable size now (was hardcoded 500).
        self._latencies: deque[float] = deque(maxlen=latency_window_size)
        self._tps_samples: deque[float] = deque(maxlen=latency_window_size)

        # aggregate token / error counters
        self._total_tokens: int = 0
        self._total_errors: int = 0

        # Per‑model telemetry: deques for percentile computation.
        self._per_model_ttft_window: Dict[str, deque] = {}
        self._per_model_tps_window: Dict[str, deque] = {}
        self._per_model_tokens: Dict[str, int] = {}
        self._per_model_window_size = max(10, int(per_model_window_size))
        self._max_per_model_entries = max(1, int(max_per_model_entries))

        self._per_model_lock = threading.RLock()
        self._per_model_request_count: Dict[str, int] = {}
        # Insertion-order index for deterministic FIFO tie-breaking inside
        # `_enforce_per_model_lru`. Without this, ties (equal request count)
        # would break by Python's set-iteration order, which depends on
        # SipHash of the string keys with a per-process random seed and is
        # therefore non-deterministic. Discovered while testing:
        # `assert "model-0" not in pms3` failed because some *other* tied
        # model was evicted first.
        self._per_model_order: Dict[str, int] = {}
        self._per_model_order_seq: int = 0

        # Anomaly detection settings
        self._anomaly_enabled = anomaly_detection
        self._anomaly_z_threshold = anomaly_latency_z_threshold
        self._anomaly_temp_threshold = anomaly_temp_threshold_c
        self._anomaly_min_gpu_util = anomaly_min_gpu_util_for_temp_alert
        self._gpu_temp_max_valid = gpu_temp_max_valid

        self._latency_stats = RunningStats()
        self._predictor = PredictiveAnomalyDetector(alpha=0.3, beta=0.1, k=3.0)
        self._tps_predictor = PredictiveAnomalyDetector(alpha=0.3, beta=0.1, k=3.0)

        # Adaptive sampling
        self._adaptive_sampling = adaptive_sampling
        self._adaptive_max_factor = adaptive_sampling_max_factor
        self._adaptive_cpu_threshold = adaptive_sampling_cpu_threshold
        self._current_interval = sample_interval

        MetricsCollector._gpu_cache_ttl = gpu_cache_ttl

        # Prometheus setup
        self._prom_available = False
        self._prom: Dict[str, Any] = {}
        if prometheus_enabled:
            self._prom_available = self._init_prometheus(port=prometheus_port)

        self._engine_stats_fn: Optional[Callable[[], Dict[str, Any]]] = None

        self._file_exporter: Optional[MetricsFileExporter] = None
        if file_export_enabled:
            self._file_exporter = MetricsFileExporter(file_export_path)

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu_query")

        # Thermal state (shared with api_server by reference — see property docstring)
        self._thermal_state = ThermalState()
        self._thermal_lock = threading.Lock()

        # Background sampler thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------------------
    # Prometheus Initialization
    # --------------------------------------------------------------------------
    def _init_prometheus(self, port: int = 9090) -> bool:
        """Initialize Prometheus metrics and start HTTP server if needed."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, start_http_server
        except ImportError:
            log.debug("Prometheus client not installed")
            return False

        def _safe_metric(metric_type, name: str, doc: str, *args, **kwargs):
            try:
                return metric_type(name, doc, *args, **kwargs)
            except ValueError:
                log.debug("Duplicate Prometheus metric: %s", name)
                return _NoOpMetric()
            except Exception as e:
                log.warning("Failed to create Prometheus metric %s: %s", name, e)
                return _NoOpMetric()

        self._prom = {
            "cpu": _safe_metric(Gauge, "llm_cpu_percent", "CPU utilization %"),
            "ram": _safe_metric(Gauge, "llm_ram_percent", "RAM utilization %"),
            "vram": _safe_metric(Gauge, "llm_vram_percent", "VRAM utilization %"),
            "gpu_temp": _safe_metric(Gauge, "llm_gpu_temperature_c", "GPU temperature °C"),
            "gpu_util": _safe_metric(Gauge, "llm_gpu_utilization_percent", "GPU utilization %"),
            "gpu_power": _safe_metric(Gauge, "llm_gpu_power_watts", "GPU power draw (W)"),
            "req_rate": _safe_metric(Gauge, "llm_request_rate", "Requests per second"),
            "ttft": _safe_metric(Histogram, "llm_ttft_ms", "Time to first token ms",
                                 buckets=[5, 20, 50, 100, 250, 500, 1000, 2000, 5000]),
            "tps": _safe_metric(Histogram, "llm_tokens_per_second", "Token throughput",
                                buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200]),
            "errors": _safe_metric(Counter, "llm_errors_total", "Inference errors", ["reason"]),
            "oom": _safe_metric(Counter, "llm_oom_total", "OOM events"),
            "cache": _safe_metric(Counter, "llm_ctx_cache_hits_total", "Context-cache hits"),
            "req": _safe_metric(Counter, "llm_requests_total", "Total requests"),
            "evictions": _safe_metric(Counter, "llm_vram_evictions_total", "VRAM LRU evictions"),
            "cb": _safe_metric(Gauge, "llm_circuit_open", "Circuit breaker (1=open)"),
            "queue": _safe_metric(Gauge, "llm_queue_depth", "Request queue depth"),
            "ws": _safe_metric(Gauge, "llm_ws_clients", "Active WebSocket log clients"),
            "defrag": _safe_metric(Counter, "llm_vram_defrag_runs", "VRAM defrag runs"),
            "models": _safe_metric(Gauge, "llm_loaded_models", "Loaded model count"),
            "vram_fragmentation": _safe_metric(Gauge, "llm_vram_fragmentation_pct",
                                               "VRAM fragmentation percentage"),
            "per_model_ttft": _safe_metric(Gauge, "llm_per_model_ttft_ms", "Per model TTFT",
                                           ["model"]),
            "per_model_tps": _safe_metric(Gauge, "llm_per_model_tps", "Per model TPS",
                                          ["model"]),
            # expose per-model p95 via Prometheus gauges too. Prometheus
            # histograms compute quantiles server-side, but only if the model
            # label is set on the histogram. The gauge is cheap and lets a
            # Grafana panel read `llm_per_model_ttft_p95_ms{model="qwen3-8b"}`
            # without enabling histogram quantile estimation.
            "per_model_ttft_p95": _safe_metric(Gauge, "llm_per_model_ttft_p95_ms",
                                               "Per model TTFT p95", ["model"]),
            "per_model_tps_p95": _safe_metric(Gauge, "llm_per_model_tps_p95",
                                              "Per model TPS p95", ["model"]),
        }

        start_key = (port, os.getpid())
        if start_key not in MetricsCollector._started_servers:
            try:
                start_http_server(port)
                MetricsCollector._started_servers.add(start_key)
                log.info("Prometheus metrics available on :%d/metrics", port)
            except Exception as e:
                log.warning("Failed to start Prometheus HTTP server: %s", e)
                return False
        else:
            log.debug("Prometheus already started on port %d for this process", port)
        return True

    def _register_model(self, model_id: str) -> None:
        """Assign an insertion-order index to `model_id` on first sight.

        Idempotent. Called from both `_bump_request_count()` (models seen
        via `record_request` / `record_completion`) and `_per_model_record()`
        (models seen via `record_latency(model_id=...)` without a preceding
        request bump). Ensures every model tracked in any per-model dict has
        an entry in `_per_model_order`, so `_enforce_per_model_lru()` can
        sort by `(request_count, insertion_index)` with a deterministic
        tie-break.

        Thread-safe (takes `_per_model_lock`, which is an RLock so nesting
        inside `_bump_request_count`'s own lock block is fine).
        """
        with self._per_model_lock:
            if model_id not in self._per_model_order:
                self._per_model_order[model_id] = self._per_model_order_seq
                self._per_model_order_seq += 1

    # --------------------------------------------------------------------------
    # Public Recording API
    # --------------------------------------------------------------------------
    def _bump_request_count(self, model_id: str) -> None:
        """Bump request counters and (when model_id given) per-model tracking.

        Called by `record_request()` and `record_completion()`. NOT called
        by `record_latency()` — avoids double-counting when callers use the
        legacy `record_request() + record_latency()` pair.
        """
        with self._lock:
            self._req_count += 1
        if self._prom_available:
            metric = self._prom.get("req")
            if metric:
                try:
                    metric.inc()
                except Exception:
                    pass
        if model_id:
            self._register_model(model_id)
            with self._per_model_lock:
                self._per_model_request_count[model_id] = (
                    self._per_model_request_count.get(model_id, 0) + 1
                )
            self._enforce_per_model_lru()

    def record_request(self, model_id: str = "") -> None:
        """Count a request. Bumps the rate counter and per-model tracking.

        Use `record_completion()` instead when you also have latency + token
        data — it combines this call with `record_latency()` into one atomic
        operation and prevents a caller from forgetting one of the two.
        """
        self._bump_request_count(model_id)

    def record_latency(
        self,
        ttft_ms: float,
        tps: float,
        model_id: str = "",
        *,
        tokens: int = 0,
    ) -> None:
        """Record one request's latency and (optionally) token count.

        Args:
            ttft_ms:  Time to first token, in milliseconds.
            tps:      Decode tokens per second (decode-only, not end-to-end).
            model_id: When non-empty, records per-model windows
                      (`_per_model_ttft_window` / `_per_model_tps_window`).
                      When empty, only the global windows are updated.
            tokens:   Authoritative completion-token count from the engine
                      or pool worker. Adds to `_total_tokens` and (when
                      `model_id` is provided) `_per_model_tokens[model_id]`.

        Note on request counting: `record_latency()` does NOT bump the
        request counter — only `record_request()` and `record_completion()`
        do. This preserves backward compatibility with the legacy pattern
        `record_request(model_id) + record_latency(ttft, tps)` used by
        api_server today, which would otherwise double-count.
        """
        tokens_int = int(tokens or 0)
        with self._lock:
            self._latencies.append(ttft_ms)
            self._tps_samples.append(tps)
            self._total_tokens += tokens_int
            if self._anomaly_enabled:
                self._latency_stats.update(ttft_ms)
        if model_id:
            self._per_model_record(model_id, ttft_ms, tps, tokens_int)
        self._update_prometheus_latency(ttft_ms, tps, model_id)

    def record_completion(
        self,
        model_id: str,
        ttft_ms: float,
        tps: float,
        tokens: int = 0,
        status: str = "success",
    ) -> None:
        """Record a completed request in one call — canonical entry point.

        Combines:
          * request count bump (for rate calc),
          * latency recording (global + per-model),
          * token accumulation (global + per-model),
          * error counter (when `status != "success"`).

        Args:
            model_id: The model that served this request.
            ttft_ms:  Time to first token, in milliseconds.
            tps:      Decode tokens per second (decode-only).
            tokens:   Authoritative completion-token count.
            status:   "success" (default) or any error label
                      ("error", "timeout", "circuit_open", ...). Any non-
                      "success" value increments both the Prometheus
                      `llm_errors_total{reason=...}` counter and the
                      `_total_errors` aggregate.

        Existing callers of `record_request() + record_latency()` still work
        unchanged; new code should prefer this method.
        """
        self._bump_request_count(model_id)
        self.record_latency(ttft_ms, tps, model_id=model_id, tokens=tokens)
        if status != "success":
            self.record_error(status)

    def record_error(self, reason: str = "unknown") -> None:
        """Record an inference error. Increments `_total_errors` + Prometheus."""
        with self._lock:
            self._total_errors += 1
        if self._prom_available:
            errors_metric = self._prom.get("errors")
            if errors_metric:
                try:
                    errors_metric.labels(reason=reason).inc()
                except Exception as e:
                    log.debug("Prometheus error inc error: %s", e)

    def record_oom(self) -> None:
        if self._prom_available:
            metric = self._prom.get("oom")
            if metric:
                try:
                    metric.inc()
                except Exception:
                    pass

    def record_cache_hit(self) -> None:
        if self._prom_available:
            metric = self._prom.get("cache")
            if metric:
                try:
                    metric.inc()
                except Exception:
                    pass

    def set_websocket_clients(self, count: int) -> None:
        if self._prom_available:
            metric = self._prom.get("ws")
            if metric:
                try:
                    metric.set(count)
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _per_model_record(
        self, model_id: str, ttft_ms: float, tps: float, tokens: int
    ) -> None:
        """Append one sample to the per-model windows. Does NOT bump request count.

        Thread-safe. Registers the model in `_per_model_order` on first sight
        so FIFO tie-breaking in `_enforce_per_model_lru` is deterministic.
        Enforces the LFU/FIFO cap only when a NEW model is added — an
        existing model's window append cannot grow the tracked set, so the
        extra `_enforce_per_model_lru` call would be pure overhead.
        """
        self._register_model(model_id)
        created_new = False
        with self._per_model_lock:
            if model_id not in self._per_model_ttft_window:
                self._per_model_ttft_window[model_id] = deque(
                    maxlen=self._per_model_window_size
                )
                self._per_model_tps_window[model_id] = deque(
                    maxlen=self._per_model_window_size
                )
                created_new = True
            self._per_model_ttft_window[model_id].append(ttft_ms)
            self._per_model_tps_window[model_id].append(tps)
            self._per_model_tokens[model_id] = (
                self._per_model_tokens.get(model_id, 0) + int(tokens or 0)
            )
        if created_new:
            self._enforce_per_model_lru()

    def _enforce_per_model_lru(self) -> None:
        """Evict the least-used models when the tracked set exceeds cap.

        Policy: **LFU with FIFO tie-break.** Ascending by request count
        first; ties broken by insertion order (the model seen *first* is
        evicted *first*). Deterministic across runs — the previous version
        sorted a set of model IDs by request count only, so equal counts
        broke by SipHash iteration order (non-deterministic per process).
        That's why `assert "model-0" not in pms3` failed despite all 10
        models having identical request counts.

        Tracked set is the UNION of `_per_model_request_count` and
        `_per_model_ttft_window` keys — a model can appear in either
        without the other:
          • record_request(m) only               → req_counts has m
          • record_latency(..., model_id=m) only → ttft_window has m
          • record_completion(m, ...)            → both have m

        Fast path (both dicts under cap) is O(1). Slow path is
        O(n log n) with n ≤ 2 × max_per_model_entries.
        """
        with self._per_model_lock:
            cap = self._max_per_model_entries
            if (len(self._per_model_request_count) <= cap
                    and len(self._per_model_ttft_window) <= cap):
                return
            tracked = set(self._per_model_request_count) | set(
                self._per_model_ttft_window
            )
            if len(tracked) <= cap:
                return
            # Deterministic ordering: primary key = request count (asc),
            # secondary key = insertion index (asc). Both are plain dict
            # lookups — the O(n log n) sort is over n IDs, not over sets
            # with unspecified iteration order.
            ranked = sorted(
                tracked,
                key=lambda m: (
                    self._per_model_request_count.get(m, 0),
                    self._per_model_order.get(m, float("inf")),
                ),
            )
            excess = len(tracked) - cap
            evicted = 0
            for mid in ranked[:excess]:
                self._per_model_request_count.pop(mid, None)
                self._per_model_ttft_window.pop(mid, None)
                self._per_model_tps_window.pop(mid, None)
                self._per_model_tokens.pop(mid, None)
                self._per_model_order.pop(mid, None)
                evicted += 1
            if evicted:
                log.info(
                    "MetricsCollector: LFU/FIFO-evicted %d per-model entries "
                    "(cap=%d)", evicted, cap,
                )

    def _update_prometheus_latency(
        self, ttft_ms: float, tps: float, model_id: str
    ) -> None:
        """Fire global + per-model Prometheus observations."""
        if not self._prom_available:
            return
        ttft_metric = self._prom.get("ttft")
        if ttft_metric:
            try:
                ttft_metric.observe(ttft_ms)
            except Exception as e:
                log.debug("Prometheus ttft observe error: %s", e)
        tps_metric = self._prom.get("tps")
        if tps_metric:
            try:
                tps_metric.observe(tps)
            except Exception as e:
                log.debug("Prometheus tps observe error: %s", e)
        if model_id:
            pttft = self._prom.get("per_model_ttft")
            ptps = self._prom.get("per_model_tps")
            if pttft:
                try:
                    pttft.labels(model=model_id).set(ttft_ms)
                except Exception:
                    pass
            if ptps:
                try:
                    ptps.labels(model=model_id).set(tps)
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # Per‑model telemetry retrieval
    # --------------------------------------------------------------------------
    def per_model_stats(self, max_models: int = 100) -> Dict[str, Dict[str, Any]]:
        """Return per-model TTFT / TPS stats with percentile windows.

        Each entry includes p50/p95 computed from the per-model
        deques, plus per-model token totals:

            {
              "avg_ttft_ms":  float,   # rolling mean of TTFT samples
              "ttft_ms_p50":  float,
              "ttft_ms_p95":  float,
              "avg_tps":      float,   # rolling mean of TPS samples
              "tps_p50":      float,
              "tps_p95":      float,
              "tokens_total": int,     # cumulative tokens for this model
              "sample_count": int,     # len of the TTFT window
            }

        Only models that have actual latency samples in
        `_per_model_ttft_window` are included. A caller that invokes
        `record_request(m)` without a subsequent per-model
        `record_latency(..., model_id=m)` — or the higher-level
        `record_completion(m, ...)` — counts the request but produces no
        samples, so there is nothing to report here. Switch to
        `record_completion()` or pass `model_id=` to `record_latency()` to
        populate this dict.

        `max_models` further limits the returned dict size (top-N by
        request count, ties broken by insertion order) on top of the LFU
        cap enforced by `_enforce_per_model_lru()`.

        `avg_ttft_ms` / `avg_tps` are ROLLING means over the current
        window — not lifetime averages. A model that was slow 5000
        requests ago and fast now shows as fast today. Matches
        `pool.stats()`'s `decode_tps`/`avg_tps` semantics.
        """
        with self._per_model_lock:
            # base the output on models that have actual stats
            # windows. A model that got `record_request(m)` but never a
            # per-model `record_latency(..., model_id=m)` has zero samples
            # to report — showing it with all-zero fields was misleading.
            stats_models = set(self._per_model_ttft_window.keys())
            req_counts = dict(self._per_model_request_count)
            ttft_windows = {
                k: list(v) for k, v in self._per_model_ttft_window.items()
            }
            tps_windows = {
                k: list(v) for k, v in self._per_model_tps_window.items()
            }
            token_totals = dict(self._per_model_tokens)

        # Rank by request count descending; ties broken by insertion
        # order via stable sort (matches _enforce_per_model_lru's ranking
        # direction so the top-N here aligns with what survives eviction).
        ranked_ids = sorted(
            stats_models,
            key=lambda m: req_counts.get(m, 0),
            reverse=True,
        )[: max(1, int(max_models))]

        result: Dict[str, Dict[str, Any]] = {}
        for mid in ranked_ids:
            ttft_list = ttft_windows.get(mid, [])
            tps_list = tps_windows.get(mid, [])
            sorted_ttft = sorted(ttft_list)
            sorted_tps = sorted(tps_list)
            result[mid] = {
                "avg_ttft_ms": round(
                    sum(ttft_list) / max(len(ttft_list), 1), 2
                ) if ttft_list else 0.0,
                "ttft_ms_p50": round(_percentile(sorted_ttft, 50), 2),
                "ttft_ms_p95": round(_percentile(sorted_ttft, 95), 2),
                "avg_tps": round(
                    sum(tps_list) / max(len(tps_list), 1), 2
                ) if tps_list else 0.0,
                "tps_p50": round(_percentile(sorted_tps, 50), 2),
                "tps_p95": round(_percentile(sorted_tps, 95), 2),
                "tokens_total": int(token_totals.get(mid, 0)),
                "sample_count": len(ttft_list),
            }
        return result

    # --------------------------------------------------------------------------
    # Background Sampling Loop (adaptive interval)
    # --------------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._take_sample()
            except Exception as e:
                log.debug("Metrics sample error: %s", e)
            wait_time = max(self._base_interval, self._current_interval)
            self._stop.wait(wait_time)

    def _take_sample(self) -> None:
        # Collect GPU metrics asynchronously via thread pool
        future_vram = self._executor.submit(self._get_vram_pct)
        future_temp = self._executor.submit(self._get_gpu_temp)
        future_util = self._executor.submit(self._get_gpu_utilization)
        future_power = self._executor.submit(self._get_gpu_power)
        future_frag = self._executor.submit(self._get_vram_fragmentation_pct)

        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        ram = vm.percent

        try:
            vram = future_vram.result(timeout=2.0)
        except Exception:
            vram = 0.0
        try:
            temp = future_temp.result(timeout=2.0)
        except Exception:
            temp = None
        try:
            gpu_util = future_util.result(timeout=2.0)
        except Exception:
            gpu_util = None
        try:
            gpu_power = future_power.result(timeout=2.0)
        except Exception:
            gpu_power = None
        try:
            frag = future_frag.result(timeout=2.0)
        except Exception:
            frag = 0.0

        now = time.monotonic()

        # Snapshot every counter we need in one lock block. Percentile
        # computation is done here (not per-record) — the deque maxlen is
        # small (500 default) and this runs once per sample tick, so the
        # cost is negligible and we avoid sorting on every request.
        with self._lock:
            elapsed = now - self._last_req_ts
            rate = self._req_count / max(elapsed, 0.001)
            self._req_count = 0
            self._last_req_ts = now

            avg_lat = sum(self._latencies) / max(len(self._latencies), 1)
            avg_tps = sum(self._tps_samples) / max(len(self._tps_samples), 1)

            _sorted_lat = sorted(self._latencies)
            _sorted_tps = sorted(self._tps_samples)
            ttft_p50 = _percentile(_sorted_lat, 50)
            ttft_p95 = _percentile(_sorted_lat, 95)
            tps_p50 = _percentile(_sorted_tps, 50)
            tps_p95 = _percentile(_sorted_tps, 95)
            tokens_total = self._total_tokens
            errors_total = self._total_errors

        # Adaptive sampling: if CPU or RAM high, increase interval (but never below base)
        if self._adaptive_sampling:
            if cpu >= self._adaptive_cpu_threshold or ram >= 90:
                factor = min(self._adaptive_max_factor, 1.0 + (max(cpu, ram) - 80) / 20.0)
                self._current_interval = self._base_interval * factor
            else:
                self._current_interval = self._base_interval
        else:
            self._current_interval = self._base_interval

        sample = Sample(
            ts=now,
            cpu_pct=cpu,
            ram_pct=ram,
            vram_pct=vram,
            gpu_temp_c=temp,
            gpu_util_pct=gpu_util,
            gpu_power_w=gpu_power,
            req_rate=rate,
            avg_latency_ms=avg_lat,
            avg_tps=avg_tps,
            vram_fragmentation_pct=frag,
            adaptive_interval_factor=self._current_interval / self._base_interval,
            per_model=self.per_model_stats(),
            tokens_total=tokens_total,
            errors_total=errors_total,
            ttft_ms_p50=ttft_p50,
            ttft_ms_p95=ttft_p95,
            tps_p50=tps_p50,
            tps_p95=tps_p95,
        )

        with self._lock:
            self._history.append(sample)

        self._update_prometheus(sample)
        self._update_thermal_state(sample)
        if self._anomaly_enabled:
            self._detect_anomalies(sample)
        if self._file_exporter:
            self._file_exporter.enqueue(sample)

    def _update_prometheus(self, sample: Sample) -> None:
        if not self._prom_available:
            return

        updates = [
            ("cpu", sample.cpu_pct),
            ("ram", sample.ram_pct),
            ("vram", sample.vram_pct),
            ("req_rate", sample.req_rate),
            ("vram_fragmentation", sample.vram_fragmentation_pct),
        ]
        for key, value in updates:
            metric = self._prom.get(key)
            if metric:
                try:
                    metric.set(value)
                except Exception:
                    pass

        if sample.gpu_temp_c is not None:
            metric = self._prom.get("gpu_temp")
            if metric:
                try:
                    metric.set(sample.gpu_temp_c)
                except Exception:
                    pass

        if sample.gpu_util_pct is not None:
            metric = self._prom.get("gpu_util")
            if metric:
                try:
                    metric.set(sample.gpu_util_pct)
                except Exception:
                    pass

        if sample.gpu_power_w is not None:
            metric = self._prom.get("gpu_power")
            if metric:
                try:
                    metric.set(sample.gpu_power_w)
                except Exception:
                    pass

        # push per-model gauges. Cheap: iterates only the models
        # that had activity in the last window. Any exception in a single
        # metric update is swallowed (a metrics bug must not break sampling).
        if sample.per_model:
            pttft_p95 = self._prom.get("per_model_ttft_p95")
            ptps_p95 = self._prom.get("per_model_tps_p95")
            for mid, entry in sample.per_model.items():
                if pttft_p95 is not None:
                    try:
                        pttft_p95.labels(model=mid).set(
                            float(entry.get("ttft_ms_p95", 0.0))
                        )
                    except Exception:
                        pass
                if ptps_p95 is not None:
                    try:
                        ptps_p95.labels(model=mid).set(
                            float(entry.get("tps_p95", 0.0))
                        )
                    except Exception:
                        pass

        if self._engine_stats_fn:
            try:
                es = self._engine_stats_fn()
                queue_metric = self._prom.get("queue")
                if queue_metric:
                    queue_metric.set(es.get("queue", {}).get("pending", 0))
                models_metric = self._prom.get("models")
                if models_metric:
                    models_metric.set(len(es.get("loaded_models", [])))
                cb_metric = self._prom.get("cb")
                if cb_metric:
                    cb_state = es.get("circuit_breaker", {}).get("state")
                    cb_metric.set(1 if cb_state == "open" else 0)
            except Exception as e:
                log.debug("Prometheus engine stats update error: %s", e)

    # --------------------------------------------------------------------------
    # GPU Helpers
    # --------------------------------------------------------------------------
    @classmethod
    def _cached_gpu_query(cls, vendor: str, fetcher, *args) -> Any:
        now = time.monotonic()
        if vendor in cls._gpu_cache:
            ts, data = cls._gpu_cache[vendor]
            if now - ts < cls._gpu_cache_ttl:
                return data
        data = fetcher(*args)
        cls._gpu_cache[vendor] = (now, data)
        return data

    @staticmethod
    def _get_vram_pct_nvidia() -> float:
        if MetricsCollector._which_nvidia_smi() is None:
            return 0.0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    used = float(parts[0].strip())
                    total = float(parts[1].strip())
                    if total > 0:
                        return 100.0 * used / total
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _get_vram_pct_torch() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                alloc = torch.cuda.memory_allocated(0)
                return 100.0 * alloc / props.total_memory
        except Exception:
            pass
        return 0.0

    def _get_vram_pct(self) -> float:
        vram = self._cached_gpu_query("nvidia_vram", self._get_vram_pct_nvidia)
        if vram > 0:
            return vram
        vram = self._cached_gpu_query("torch_vram", self._get_vram_pct_torch)
        if vram > 0:
            return vram
        try:
            for card in sorted(Path("/sys/class/drm").glob("card*")):
                total_f = card / "device" / "mem_info_vram_total"
                used_f = card / "device" / "mem_info_vram_used"
                if total_f.exists() and used_f.exists():
                    try:
                        total = int(total_f.read_text().strip())
                        used = int(used_f.read_text().strip())
                        if total > 0:
                            return 100.0 * used / total
                    except (PermissionError, OSError):
                        continue
        except Exception:
            pass
        try:
            for card in sorted(Path("/sys/class/drm").glob("card*")):
                total_f = card / "device" / "local_mem_total"
                used_f = card / "device" / "local_mem_used"
                if total_f.exists() and used_f.exists():
                    try:
                        total = int(total_f.read_text().strip())
                        used = int(used_f.read_text().strip())
                        if total > 0:
                            return 100.0 * used / total
                    except (PermissionError, OSError):
                        continue
        except Exception:
            pass
        return 0.0

    def _get_vram_fragmentation_pct(self) -> float:
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    parts = [float(x) for x in result.stdout.strip().split(",")]
                    if len(parts) == 3:
                        total, used, free = parts
                        if total > 0:
                            frag = (total - used - free) / total
                            return max(0.0, min(100.0, frag * 100.0))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _get_gpu_temp_nvidia() -> Optional[float]:
        if MetricsCollector._which_nvidia_smi() is None:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                temp = float(result.stdout.strip().splitlines()[0])
                if 0 < temp < 110:
                    return temp
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_temp_amd() -> Optional[float]:
        if shutil.which("rocm-smi") is None:
            return None
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                for v in data.values():
                    temp = v.get("Temperature (Sensor edge) (C)")
                    if temp and 0 < temp < 110:
                        return float(temp)
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_temp_sysfs() -> Optional[float]:
        try:
            for card in sorted(Path("/sys/class/drm").glob("card*")):
                hwmon_dir = card / "device" / "hwmon"
                if not hwmon_dir.exists():
                    continue
                for hwmon in hwmon_dir.iterdir():
                    temp_file = hwmon / "temp1_input"
                    if temp_file.exists():
                        try:
                            temp = int(temp_file.read_text().strip()) / 1000.0
                            if 0 < temp < 110:
                                return temp
                        except (PermissionError, OSError):
                            continue
        except Exception:
            pass
        return None

    def _get_gpu_temp(self) -> Optional[float]:
        # sysfs hwmon is a plain file read: no fork/exec from a process that
        # has torch/scipy/pandas mapped (every subprocess spawn is expensive
        # there, and these samples run every few seconds).
        temp = self._cached_gpu_query("sysfs_temp", self._get_gpu_temp_sysfs)
        if temp is not None:
            return temp
        temp = self._cached_gpu_query("nvidia_temp", self._get_gpu_temp_nvidia)
        if temp is not None:
            return temp
        return self._cached_gpu_query("amd_temp", self._get_gpu_temp_amd)

    @staticmethod
    def _get_gpu_util_nvidia() -> Optional[float]:
        if MetricsCollector._which_nvidia_smi() is None:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_util_amd() -> Optional[float]:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--json"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                for v in data.values():
                    util = v.get("GPU use (%)")
                    if util is not None:
                        return float(util)
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_util_sysfs() -> Optional[float]:
        try:
            for card in sorted(Path("/sys/class/drm").glob("card*")):
                util_file = card / "device" / "gpu_busy_percent"
                if util_file.exists():
                    try:
                        return float(util_file.read_text().strip())
                    except (PermissionError, OSError):
                        continue
        except Exception:
            pass
        return None

    def _get_gpu_utilization(self) -> Optional[float]:
        util = self._cached_gpu_query("nvidia_util", self._get_gpu_util_nvidia)
        if util is not None:
            return util
        util = self._cached_gpu_query("amd_util", self._get_gpu_util_amd)
        if util is not None:
            return util
        return self._cached_gpu_query("sysfs_util", self._get_gpu_util_sysfs)

    @staticmethod
    def _get_gpu_power_nvidia() -> Optional[float]:
        if MetricsCollector._which_nvidia_smi() is None:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0].replace(" W", ""))
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_power_amd() -> Optional[float]:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                for v in data.values():
                    power = v.get("Average Graphics Package Power (W)")
                    if power is not None:
                        return float(power)
        except Exception:
            pass
        return None

    def _get_gpu_power(self) -> Optional[float]:
        power = self._cached_gpu_query("nvidia_power", self._get_gpu_power_nvidia)
        if power is not None:
            return power
        return self._cached_gpu_query("amd_power", self._get_gpu_power_amd)

    # --------------------------------------------------------------------------
    # Thermal state update — unchanged
    # --------------------------------------------------------------------------
    def _update_thermal_state(self, s: Sample) -> None:
        """Mutate the shared thermal state in place from a sample.

        Called unconditionally on every sampling tick — NOT gated on
        anomaly_detection. See the property docstring below.
        """
        with self._thermal_lock:
            if s.gpu_temp_c is not None:
                self._thermal_state.temperature_c = s.gpu_temp_c
                self._thermal_state.timestamp = s.ts
                _busy = s.gpu_util_pct is None or s.gpu_util_pct > self._anomaly_min_gpu_util
                if s.gpu_temp_c > self._anomaly_temp_threshold and _busy:
                    self._thermal_state.throttling = True
                elif (s.gpu_temp_c < self._anomaly_temp_threshold - 5.0) or not _busy:
                    self._thermal_state.throttling = False
                # else: inside the 5 °C hysteresis band -> keep the previous state
            else:
                self._thermal_state.throttling = False

    # --------------------------------------------------------------------------
    # Anomaly Detection — unchanged
    # --------------------------------------------------------------------------
    def _detect_anomalies(self, s: Sample) -> None:
        with self._lock:
            count = self._latency_stats.count
            if count >= 30:
                mean = self._latency_stats.mean
                std = self._latency_stats.std
                if std < 1.0:
                    std = 1.0
                z = (s.avg_latency_ms - mean) / std
                if z > self._anomaly_z_threshold:
                    log.warning(
                        "ANOMALY: latency spike %.0fms (z=%.1f, baseline=%.0f±%.0f)",
                        s.avg_latency_ms, z, mean, std
                    )

        alert = self._predictor.check("avg_latency_ms", s.avg_latency_ms, s.ts)
        if alert:
            log.warning("PREDICTIVE: %s", alert)
        alert = self._tps_predictor.check("avg_tps", s.avg_tps, s.ts)
        if alert:
            log.warning("PREDICTIVE: %s", alert)

        if s.gpu_temp_c and s.gpu_temp_c > self._anomaly_temp_threshold:
            if s.gpu_util_pct is None or s.gpu_util_pct > self._anomaly_min_gpu_util:
                log.warning(
                    "ANOMALY: GPU temperature %.1f°C (util=%.1f%%)",
                    s.gpu_temp_c,
                    s.gpu_util_pct or 0.0,
                )

    # --------------------------------------------------------------------------
    # Thermal state property
    # --------------------------------------------------------------------------
    @property
    def thermal_state(self) -> ThermalState:
        """Return the LIVE thermal-state object (NOT a copy).

        Mutations in the sampling thread are immediately visible to every
        reader (api_server._check_thermal_throttle, engine._scheduler).
        """
        return self._thermal_state

    # --------------------------------------------------------------------------
    # Engine Stats Attachment
    # --------------------------------------------------------------------------
    def attach_engine(self, stats_fn: Callable[[], Dict[str, Any]]) -> None:
        self._engine_stats_fn = stats_fn

    # --------------------------------------------------------------------------
    # Snapshot (sync and async) – thread‑safe
    # --------------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the latest sample.

        additions in the return dict (all safe to consume; existing
        keys unchanged):
            ttft_ms_p50, ttft_ms_p95, tps_p50, tps_p95 — global percentiles
            tokens_total, errors_total                  — aggregate counters
            per_model                                   — enriched per-model dict

        The empty-history fallback (first ~5s after startup, before the
        first sampling tick) now also computes p50/p95 from the live deques
        (which may already have data even when no Sample has been stored).
        """
        # Compute per_model BEFORE taking _lock — per_model_stats() takes
        # _per_model_lock internally, and we must not nest _lock around it
        # (the sampling loop takes them in this same order, so nesting the
        # other way would risk a deadlock if a future change adds a
        # _per_model_lock -> _lock path).
        _pms = self.per_model_stats()

        with self._lock:
            if not self._history:
                vm = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)
                ram = vm.percent
                _sorted_lat = sorted(self._latencies)
                _sorted_tps = sorted(self._tps_samples)
                return {
                    "cpu_pct": cpu,
                    "ram_pct": ram,
                    "vram_pct": 0.0,
                    "gpu_temp_c": None,
                    "gpu_util_pct": None,
                    "gpu_power_w": None,
                    "req_rate": 0.0,
                    "avg_latency_ms": (
                        sum(self._latencies) / max(len(self._latencies), 1)
                    ),
                    "avg_tps": (
                        sum(self._tps_samples) / max(len(self._tps_samples), 1)
                    ),
                    "samples": 0,
                    "warming_up": True,
                    "vram_fragmentation_pct": 0.0,
                    "per_model": _pms,
                    # live-deque percentiles + aggregate counters
                    "ttft_ms_p50": _percentile(_sorted_lat, 50),
                    "ttft_ms_p95": _percentile(_sorted_lat, 95),
                    "tps_p50": _percentile(_sorted_tps, 50),
                    "tps_p95": _percentile(_sorted_tps, 95),
                    "tokens_total": self._total_tokens,
                    "errors_total": self._total_errors,
                }
            recent = self._history[-1]
            return {
                "cpu_pct": recent.cpu_pct,
                "ram_pct": recent.ram_pct,
                "vram_pct": recent.vram_pct,
                "gpu_temp_c": recent.gpu_temp_c,
                "gpu_util_pct": recent.gpu_util_pct,
                "gpu_power_w": recent.gpu_power_w,
                "req_rate": recent.req_rate,
                "avg_latency_ms": recent.avg_latency_ms,
                "avg_tps": recent.avg_tps,
                "samples": len(self._history),
                "vram_fragmentation_pct": recent.vram_fragmentation_pct,
                "per_model": recent.per_model,
                "ttft_ms_p50": recent.ttft_ms_p50,
                "ttft_ms_p95": recent.ttft_ms_p95,
                "tps_p50": recent.tps_p50,
                "tps_p95": recent.tps_p95,
                "tokens_total": recent.tokens_total,
                "errors_total": recent.errors_total,
            }

    async def async_snapshot(self) -> Dict[str, Any]:
        """Return snapshot from the event loop without blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.snapshot)

    # --------------------------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------------------------
    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._executor.shutdown(wait=False)
        if self._file_exporter:
            self._file_exporter.shutdown()
        log.debug("Metrics collector shut down")


# ------------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------------
def create_metrics_collector(cfg: Optional[Dict[str, Any]] = None) -> MetricsCollector:
    if cfg is None:
        cfg = _load_config()

    metrics_cfg = cfg.get("monitoring", {}).get("metrics", {})
    gpu_cfg = cfg.get("gpu_analyzer", {})
    threshold = gpu_cfg.get("thermal_throttle_temp_c")
    if threshold is None:
        threshold = metrics_cfg.get("anomaly_temp_threshold_c", 85.0)
    else:
        old_threshold = metrics_cfg.get("anomaly_temp_threshold_c")
        if old_threshold is not None and abs(old_threshold - threshold) > 0.1:
            log.warning(
                "Both gpu_analyzer.thermal_throttle_temp_c (%.1f) and "
                "monitoring.metrics.anomaly_temp_threshold_c (%.1f) are set. "
                "Using gpu_analyzer value as single source of truth.",
                threshold, old_threshold
            )

    return MetricsCollector(
        sample_interval=metrics_cfg.get("sample_interval_sec", 5.0),
        history_size=metrics_cfg.get("history_size", 1200),
        latency_window_size=metrics_cfg.get("latency_window_size", 500),
        per_model_window_size=metrics_cfg.get("per_model_window_size", 500),
        max_per_model_entries=metrics_cfg.get("max_per_model_entries", 200),
        prometheus_enabled=metrics_cfg.get("prometheus_enabled", False),
        prometheus_port=metrics_cfg.get("prometheus_port", 9090),
        anomaly_detection=metrics_cfg.get("anomaly_detection", True),
        anomaly_latency_z_threshold=metrics_cfg.get("anomaly_latency_z_threshold", 5.0),
        anomaly_temp_threshold_c=threshold,
        anomaly_min_gpu_util_for_temp_alert=metrics_cfg.get("anomaly_min_gpu_util_for_temp_alert", 10.0),
        gpu_temp_max_valid=metrics_cfg.get("gpu_temp_max_valid", 110.0),
        gpu_cache_ttl=metrics_cfg.get("gpu_cache_ttl_sec", 5.0),
        adaptive_sampling=metrics_cfg.get("adaptive_sampling", True),
        adaptive_sampling_max_factor=metrics_cfg.get("adaptive_sampling_max_factor", 4.0),
        adaptive_sampling_cpu_threshold=metrics_cfg.get("adaptive_sampling_cpu_threshold", 80.0),
        file_export_enabled=metrics_cfg.get("file_export_enabled", False),
        file_export_path=metrics_cfg.get("file_export_path", "/tmp/llm_metrics.jsonl"),
    )


def get_metrics_collector(config_path: Optional[Path] = None) -> MetricsCollector:
    cfg = _load_config(config_path)
    return create_metrics_collector(cfg)


# ------------------------------------------------------------------------------
# Legacy MetricSnapshot (Compatibility)
# ------------------------------------------------------------------------------
@dataclass
class MetricSnapshot:
    timestamp: float
    vram_used_bytes: int
    vram_total_bytes: int
    ram_used_bytes: int
    ram_total_bytes: int
    gpu_temperature_c: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    gpu_power_w: Optional[float] = None
    tokens_per_second: float = 0.0
    active_requests: int = 0
    queue_depth: int = 0

    @property
    def vram_utilization_pct(self) -> float:
        return 100 * self.vram_used_bytes / max(self.vram_total_bytes, 1)

    @property
    def ram_utilization_pct(self) -> float:
        return 100 * self.ram_used_bytes / max(self.ram_total_bytes, 1)


__all__ = [
    "MetricsCollector",
    "MetricSnapshot",
    "ThermalState",
    "create_metrics_collector",
    "get_metrics_collector",
]
