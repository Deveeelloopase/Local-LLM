from __future__ import annotations

"""
local_llm_server/monitoring/metrics_collector.py — Production Metrics Collector v1.0.0
========================================================================================
Collects system and inference metrics, optionally exposes Prometheus endpoint.

  
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
# Configuration Loading (with environment variable fallback)
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

    # Apply environment variable overrides for critical settings
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
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)


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
    """
    Double exponential smoothing (Holt's linear trend) for forecasting.
    Used for anomaly prediction on latency and TPS.
    (Note: duplicated in telemetry.py – future refactoring should move to a shared stats module.)
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level: Optional[float] = None
        self.trend: float = 0.0
        self.last_forecast: Optional[float] = None
        self.mse: float = 1.0  # running MSE for adaptive thresholds

    def update(self, value: float) -> float:
        """Update with new observation and return one-step-ahead forecast error."""
        if self.level is None:
            self.level = value
            self.last_forecast = value
            return 0.0
        last_level = self.level
        self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        forecast = self.level + self.trend
        error = value - forecast
        # Update MSE
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
    """
    Uses Holt‑Winters to forecast metrics and alerts on unexpected deviation.
    (Note: duplicated in telemetry.py – future refactoring should move to a shared stats module.)
    """
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
            # Only alert if enough history and error is significant
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
        prometheus_enabled: bool = False
        prometheus_port: int = 9090
        anomaly_detection: bool = True
        anomaly_latency_z_threshold: float = 5.0
        anomaly_temp_threshold_c: float = 85.0
        anomaly_min_gpu_util_for_temp_alert: float = 10.0
        gpu_temp_max_valid: float = 110.0
        gpu_cache_ttl_sec: float = 5.0
        adaptive_sampling: bool = True
        adaptive_sampling_max_factor: float = 4.0
        adaptive_sampling_cpu_threshold: float = 80.0
        file_export_enabled: bool = False
        file_export_path: str = "/tmp/llm_metrics.jsonl"
    """

    _started_servers: set = set()
    _gpu_cache: Dict[str, Tuple[float, Any]] = {}
    _gpu_cache_ttl = 5.0

    def __init__(
        self,
        sample_interval: float = 5.0,
        history_size: int = 1200,
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
        self._latencies: deque[float] = deque(maxlen=500)
        self._tps_samples: deque[float] = deque(maxlen=500)

        # Per‑model telemetry
        self._per_model_ttft: Dict[str, RunningStats] = {}
        self._per_model_tps: Dict[str, RunningStats] = {}
        self._per_model_lock = threading.Lock()
        self._per_model_request_count: Dict[str, int] = {}  # for LRU cleanup

        # Anomaly detection settings
        self._anomaly_enabled = anomaly_detection
        self._anomaly_z_threshold = anomaly_latency_z_threshold
        self._anomaly_temp_threshold = anomaly_temp_threshold_c
        self._anomaly_min_gpu_util = anomaly_min_gpu_util_for_temp_alert
        self._gpu_temp_max_valid = gpu_temp_max_valid

        # Classic Welford stats for latency (legacy)
        self._latency_stats = RunningStats()
        # Predictive anomaly detector for latency and TPS
        self._predictor = PredictiveAnomalyDetector(alpha=0.3, beta=0.1, k=3.0)
        self._tps_predictor = PredictiveAnomalyDetector(alpha=0.3, beta=0.1, k=3.0)

        # Adaptive sampling
        self._adaptive_sampling = adaptive_sampling
        self._adaptive_max_factor = adaptive_sampling_max_factor
        self._adaptive_cpu_threshold = adaptive_sampling_cpu_threshold
        self._current_interval = sample_interval

        # GPU cache TTL
        MetricsCollector._gpu_cache_ttl = gpu_cache_ttl

        # Prometheus setup
        self._prom_available = False
        self._prom: Dict[str, Any] = {}
        if prometheus_enabled:
            self._prom_available = self._init_prometheus(port=prometheus_port)

        # Optional engine stats callback (attached later)
        self._engine_stats_fn: Optional[Callable[[], Dict[str, Any]]] = None

        # File exporter
        self._file_exporter: Optional[MetricsFileExporter] = None
        if file_export_enabled:
            self._file_exporter = MetricsFileExporter(file_export_path)

        # Thread pool for non-blocking GPU queries
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu_query")

        # Thermal state (shared with api_server)
        self._thermal_state = ThermalState()
        self._thermal_lock = threading.Lock()

        # Background sampler thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # --------------------------------------------------------------------------
    # Prometheus Initialization (with no-op fallback)
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

    # --------------------------------------------------------------------------
    # Public Recording API
    # --------------------------------------------------------------------------
    def record_request(self, model_id: str = "") -> None:
        with self._lock:
            self._req_count += 1
        if self._prom_available:
            metric = self._prom.get("req")
            if metric:
                try:
                    metric.inc()
                except Exception as e:
                    log.debug("Prometheus req inc error: %s", e)
        # Track per-model request count for LRU
        if model_id:
            with self._per_model_lock:
                self._per_model_request_count[model_id] = self._per_model_request_count.get(model_id, 0) + 1

    def record_latency(self, ttft_ms: float, tps: float, model_id: str = "") -> None:
        with self._lock:
            self._latencies.append(ttft_ms)
            self._tps_samples.append(tps)
            if self._anomaly_enabled:
                self._latency_stats.update(ttft_ms)
        # Per‑model stats
        if model_id:
            with self._per_model_lock:
                if model_id not in self._per_model_ttft:
                    self._per_model_ttft[model_id] = RunningStats()
                if model_id not in self._per_model_tps:
                    self._per_model_tps[model_id] = RunningStats()
                self._per_model_ttft[model_id].update(ttft_ms)
                self._per_model_tps[model_id].update(tps)
            if self._prom_available:
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
        if self._prom_available:
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

    def record_error(self, reason: str = "unknown") -> None:
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
    # Per‑model telemetry retrieval (with LRU cleanup)
    # --------------------------------------------------------------------------
    def per_model_stats(self, max_models: int = 100) -> Dict[str, Dict[str, float]]:
        """Return per‑model average TTFT and TPS, limited to top `max_models` by request count."""
        with self._per_model_lock:
            # Sort models by request count (descending) and keep top N
            sorted_models = sorted(self._per_model_request_count.items(),
                                   key=lambda x: x[1], reverse=True)
            top_models = [mid for mid, _ in sorted_models[:max_models]]
            result = {}
            for model_id in top_models:
                ttft_stats = self._per_model_ttft.get(model_id)
                tps_stats = self._per_model_tps.get(model_id)
                if ttft_stats and ttft_stats.count > 0:
                    result[model_id] = {
                        "avg_ttft_ms": round(ttft_stats.mean, 2),
                        "avg_tps": round(tps_stats.mean, 2) if tps_stats else 0.0,
                    }
            # Also include any model that has stats but not in top (just in case)
            for model_id, ttft_stats in self._per_model_ttft.items():
                if model_id not in result and ttft_stats.count > 0:
                    result[model_id] = {
                        "avg_ttft_ms": round(ttft_stats.mean, 2),
                        "avg_tps": round(self._per_model_tps.get(model_id, RunningStats()).mean, 2),
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
            # Wait based on adaptive interval (never go below base_interval)
            wait_time = max(self._base_interval, self._current_interval)
            self._stop.wait(wait_time)

    def _take_sample(self) -> None:
        # Collect GPU metrics asynchronously via thread pool
        future_vram = self._executor.submit(self._get_vram_pct)
        future_temp = self._executor.submit(self._get_gpu_temp)
        future_util = self._executor.submit(self._get_gpu_utilization)
        future_power = self._executor.submit(self._get_gpu_power)
        future_frag = self._executor.submit(self._get_vram_fragmentation_pct)

        # Collect CPU/RAM synchronously (fast)
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        ram = vm.percent

        # Wait for GPU results with timeout
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

        with self._lock:
            elapsed = now - self._last_req_ts
            rate = self._req_count / max(elapsed, 0.001)
            self._req_count = 0
            self._last_req_ts = now

            avg_lat = sum(self._latencies) / max(len(self._latencies), 1)
            avg_tps = sum(self._tps_samples) / max(len(self._tps_samples), 1)

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
        )

        with self._lock:
            self._history.append(sample)

        self._update_prometheus(sample)
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
    # GPU Helpers (Cached, Multi‑Vendor) with improved fragmentation
    # --------------------------------------------------------------------------
    @classmethod
    def _cached_gpu_query(cls, vendor: str, fetcher, *args) -> Any:
        """Cache GPU queries for a short TTL."""
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
        if not shutil.which("nvidia-smi"):
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
        # AMD / Intel sysfs
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
        """Estimate VRAM fragmentation using nvidia-smi or fallback."""
        # NVIDIA method: compute (total - used - free) / total
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
                            # Fragmentation: memory that is neither used nor free (e.g., reserved, cache)
                            frag = (total - used - free) / total
                            return max(0.0, min(100.0, frag * 100.0))
            except Exception:
                pass
        # Fallback: use used/total ratio as proxy (no fragmentation info)
        # Just return 0.0 to avoid misleading alerts.
        return 0.0

    @staticmethod
    def _get_gpu_temp_nvidia() -> Optional[float]:
        if not shutil.which("nvidia-smi"):
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
        temp = self._cached_gpu_query("nvidia_temp", self._get_gpu_temp_nvidia)
        if temp is not None:
            return temp
        temp = self._cached_gpu_query("amd_temp", self._get_gpu_temp_amd)
        if temp is not None:
            return temp
        return self._cached_gpu_query("sysfs_temp", self._get_gpu_temp_sysfs)

    @staticmethod
    def _get_gpu_util_nvidia() -> Optional[float]:
        if not shutil.which("nvidia-smi"):
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
        if not shutil.which("nvidia-smi"):
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
    # Anomaly Detection (Classic + Predictive) + 🔥 THERMAL STATE UPDATE
    # --------------------------------------------------------------------------
    def _detect_anomalies(self, s: Sample) -> None:
        # Classic z‑score on latency
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

        # Predictive anomaly on latency and TPS
        alert = self._predictor.check("avg_latency_ms", s.avg_latency_ms, s.ts)
        if alert:
            log.warning("PREDICTIVE: %s", alert)
        alert = self._tps_predictor.check("avg_tps", s.avg_tps, s.ts)
        if alert:
            log.warning("PREDICTIVE: %s", alert)

        # Temperature threshold with utilisation check
        # 🔥 ISSUE 1 FIX: Update thermal state and set throttling flag
        with self._thermal_lock:
            if s.gpu_temp_c is not None:
                self._thermal_state.temperature_c = s.gpu_temp_c
                self._thermal_state.timestamp = s.ts
                if s.gpu_temp_c > self._anomaly_temp_threshold:
                    # Only throttle if GPU is actually being used (or we don't know)
                    if s.gpu_util_pct is None or s.gpu_util_pct > self._anomaly_min_gpu_util:
                        self._thermal_state.throttling = True
                    else:
                        # High temperature but low utilisation: maybe a sensor artefact; don't throttle
                        self._thermal_state.throttling = False
                else:
                    self._thermal_state.throttling = False
            else:
                # No temperature reading — assume safe
                self._thermal_state.throttling = False

        # Log warning (still the old behaviour)
        if s.gpu_temp_c and s.gpu_temp_c > self._anomaly_temp_threshold:
            if s.gpu_util_pct is None or s.gpu_util_pct > self._anomaly_min_gpu_util:
                log.warning(
                    "ANOMALY: GPU temperature %.1f°C (util=%.1f%%)",
                    s.gpu_temp_c,
                    s.gpu_util_pct or 0.0
                )

    # --------------------------------------------------------------------------
    # Expose thermal_state property (thread-safe read)
    # --------------------------------------------------------------------------
    @property
    def thermal_state(self) -> ThermalState:
        """Return a copy of the current thermal state (thread-safe)."""
        with self._thermal_lock:
            # Return a copy to avoid mutation from outside
            return ThermalState(
                throttling=self._thermal_state.throttling,
                temperature_c=self._thermal_state.temperature_c,
                timestamp=self._thermal_state.timestamp,
            )

    # --------------------------------------------------------------------------
    # Engine Stats Attachment
    # --------------------------------------------------------------------------
    def attach_engine(self, stats_fn: Callable[[], Dict[str, Any]]) -> None:
        """Attach a function that returns current engine stats."""
        self._engine_stats_fn = stats_fn

    # --------------------------------------------------------------------------
    # Snapshot (sync and async) – thread‑safe
    # --------------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                # fallback to live data
                vm = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)
                ram = vm.percent
                return {
                    "cpu_pct": cpu,
                    "ram_pct": ram,
                    "vram_pct": 0.0,
                    "gpu_temp_c": None,
                    "gpu_util_pct": None,
                    "gpu_power_w": None,
                    "req_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_tps": 0.0,
                    "samples": 0,
                    "warming_up": True,
                    "vram_fragmentation_pct": 0.0,
                    "per_model": {},
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
    # 🔥 ISSUE 1 FIX: Prefer gpu_analyzer.thermal_throttle_temp_c as the single source of truth
    gpu_cfg = cfg.get("gpu_analyzer", {})
    threshold = gpu_cfg.get("thermal_throttle_temp_c")
    if threshold is None:
        # Fallback to old monitoring.metrics.anomaly_temp_threshold_c if present
        threshold = metrics_cfg.get("anomaly_temp_threshold_c", 85.0)
    else:
        # If we have the new config, use it and log a warning if the old one is different
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
        prometheus_enabled=metrics_cfg.get("prometheus_enabled", False),
        prometheus_port=metrics_cfg.get("prometheus_port", 9090),
        anomaly_detection=metrics_cfg.get("anomaly_detection", True),
        anomaly_latency_z_threshold=metrics_cfg.get("anomaly_latency_z_threshold", 5.0),
        anomaly_temp_threshold_c=threshold,  # 🔥 Use the unified threshold
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
