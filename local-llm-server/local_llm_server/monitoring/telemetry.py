from __future__ import annotations

"""
local_llm_server/monitoring/telemetry.py — Production Telemetry & Prometheus v4.1.0
====================================================================================
Real‑time telemetry, predictive performance analytics, adaptive anomaly detection,
and Prometheus exporter with novel algorithms for enterprise‑grade observability.

v4.1.0 fixes:
  • GPU metrics collected in thread pool (non‑blocking async).
  • Thread‑safe GPU cache.
  • Adaptive sampling now uses correct RAM percentage.
  • All previously reported bugs (#20, #22) addressed.
"""

import asyncio
import json
import logging
import math
import shutil
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------
def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is not None:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error("Failed to load config %s: %s", config_path, e)
        else:
            logger.warning("Config file not found: %s", config_path)
        return {}

    paths = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error("Failed to load config %s: %s", p, e)
    logger.warning("No config.yaml found; using default telemetry settings.")
    return {}


# ------------------------------------------------------------------------------
# Optional Prometheus Integration with No‑Op Fallback
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - Prometheus export disabled")


class _NoOpMetric:
    """No‑operation metric that silently accepts any calls."""
    def labels(self, *args, **kwargs):
        return self
    def inc(self, amount=1):
        pass
    def set(self, value):
        pass
    def observe(self, value):
        pass
    def time(self):
        return self


class PrometheusMetrics:
    """Prometheus metric definitions with duplicate‑safe registration and no‑op fallback."""

    def __init__(self, namespace: str = "llm_cloud"):
        if not PROMETHEUS_AVAILABLE:
            self._noop = True
            return
        self._noop = False
        ns = namespace

        def _safe_metric(metric_type, name, doc, labels=None, **kwargs):
            try:
                return metric_type(name, doc, labels or [], **kwargs)
            except ValueError:
                logger.debug("Duplicate Prometheus metric: %s", name)
                return _NoOpMetric()
            except Exception as e:
                logger.warning("Failed to create Prometheus metric %s: %s", name, e)
                return _NoOpMetric()

        self.requests_total = _safe_metric(
            Counter, f"{ns}_requests_total", "Total inference requests", ["model", "status"]
        )
        self.request_duration = _safe_metric(
            Histogram,
            f"{ns}_request_duration_seconds",
            "Request duration in seconds",
            ["model"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        self.time_to_first_token = _safe_metric(
            Histogram,
            f"{ns}_ttft_milliseconds",
            "Time to first token in ms",
            ["model"],
            buckets=[50, 100, 200, 500, 1000, 2000, 5000],
        )
        self.tokens_per_second = _safe_metric(
            Gauge, f"{ns}_tokens_per_second", "Current generation speed", ["model"]
        )
        self.active_requests = _safe_metric(
            Gauge, f"{ns}_active_requests", "Number of active inference requests"
        )
        self.gpu_vram_used_bytes = _safe_metric(
            Gauge, f"{ns}_gpu_vram_used_bytes", "GPU VRAM used in bytes", ["gpu_index"]
        )
        self.gpu_temperature_celsius = _safe_metric(
            Gauge, f"{ns}_gpu_temperature_celsius", "GPU temperature in Celsius", ["gpu_index"]
        )
        self.gpu_utilization_pct = _safe_metric(
            Gauge, f"{ns}_gpu_utilization_percent", "GPU compute utilization percentage", ["gpu_index"]
        )
        self.gpu_power_watts = _safe_metric(
            Gauge, f"{ns}_gpu_power_watts", "GPU power draw in watts", ["gpu_index"]
        )
        self.model_loaded = _safe_metric(
            Gauge, f"{ns}_model_loaded", "Whether a model is loaded", ["model_id"]
        )
        self.vram_fragmentation = _safe_metric(
            Gauge, f"{ns}_vram_fragmentation_ratio", "VRAM fragmentation ratio 0-1"
        )
        self.adaptive_interval = _safe_metric(
            Gauge, f"{ns}_adaptive_interval_sec", "Current collection interval in sec"
        )


# ------------------------------------------------------------------------------
# Metric Types
# ------------------------------------------------------------------------------
@dataclass
class RequestMetric:
    request_id: str
    model_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tps: float
    status: str  # "success" | "error" | "timeout"
    timestamp: float = field(default_factory=time.time)
    context_cache_hit: bool = False
    vram_mb: float = 0.0


@dataclass
class ResourceSnapshot:
    timestamp: float
    gpu_vram_used_mb: Dict[int, int]
    gpu_vram_free_mb: Dict[int, int]
    gpu_util_pct: Dict[int, float]
    gpu_temp_c: Dict[int, float]
    gpu_power_w: Dict[int, float]
    ram_used_gb: float
    ram_free_gb: float
    cpu_util_pct: float
    active_requests: int
    vram_fragmentation_pct: float = 0.0
    adaptive_interval_factor: float = 1.0
    per_model_ttft_avg: Dict[str, float] = field(default_factory=dict)
    per_model_tps_avg: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Predictive Anomaly Detector (Holt‑Winters based) – duplicated with metrics_collector
# TODO: move to monitoring/stats.py in a future refactoring
# ------------------------------------------------------------------------------
class HoltWintersStats:
    """Double exponential smoothing for predictive anomaly detection."""
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
        self._last_alert: Dict[str, float] = {}

    def check(self, metric_name: str, value: float, ts: float) -> Optional[str]:
        model = self._models.get(metric_name)
        if model is None:
            model = HoltWintersStats(self._alpha, self._beta)
            self._models[metric_name] = model
        err = model.update(value)
        if model.level is not None and model.last_forecast is not None:
            if abs(err) > self._k * model.prediction_std:
                last = self._last_alert.get(metric_name, 0)
                if ts - last < 60:
                    return None
                self._last_alert[metric_name] = ts
                return (f"Predictive anomaly for {metric_name}: value={value:.2f} "
                        f"(expected {model.prediction:.2f} ± {model.prediction_std:.2f})")
        return None


# ------------------------------------------------------------------------------
# Classic Anomaly Detector (Z‑Score) with Enhanced Throttling
# ------------------------------------------------------------------------------
class AnomalyDetector:
    """Statistical anomaly detection using rolling Z‑score with adaptive cooldown."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._alerts: deque = deque(maxlen=1000)
        self.z_threshold = z_threshold
        self.min_samples = 30
        self._last_alert_time: Dict[str, float] = {}
        self._alert_count: Dict[str, int] = defaultdict(int)

    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        window = self._windows[metric_name]
        window.append(value)
        if len(window) < self.min_samples:
            return None

        mean = statistics.mean(window)
        stdev = statistics.stdev(window)
        if stdev < 1e-9:
            return None

        z_score = abs(value - mean) / stdev
        if z_score > self.z_threshold:
            now = time.time()
            base_cooldown = 60
            count = self._alert_count[metric_name]
            cooldown = base_cooldown + count * 30
            last = self._last_alert_time.get(metric_name, 0)
            if now - last < cooldown:
                return None
            self._last_alert_time[metric_name] = now
            self._alert_count[metric_name] += 1

            alert = {
                "metric": metric_name,
                "value": value,
                "mean": round(mean, 3),
                "z_score": round(z_score, 2),
                "timestamp": now,
                "severity": "critical" if z_score > self.z_threshold * 1.5 else "warning",
            }
            if labels:
                alert["labels"] = labels
            self._alerts.append(alert)
            logger.warning(
                "ANOMALY [%s]: %s=%.2f (z=%.1f, mean=%.2f)",
                alert["severity"], metric_name, value, z_score, mean
            )
            return alert
        self._alert_count[metric_name] = 0
        return None

    def get_recent_alerts(self, max_age_s: float = 300) -> List[Dict[str, Any]]:
        cutoff = time.time() - max_age_s
        return [a for a in self._alerts if a["timestamp"] > cutoff]


# ------------------------------------------------------------------------------
# Telemetry File Exporter (async‑safe)
# ------------------------------------------------------------------------------
class TelemetryFileExporter:
    """Writes ResourceSnapshots as JSON lines to a file in a background thread."""
    def __init__(self, file_path: str, flush_interval: float = 1.0):
        self._path = Path(file_path)
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def enqueue(self, snapshot: ResourceSnapshot) -> None:
        with self._lock:
            self._queue.append(snapshot)

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
                            f.write(json.dumps(self._snapshot_to_dict(s)) + "\n")
                except Exception as e:
                    logger.error("Telemetry file export error: %s", e)
            self._stop.wait(self._flush_interval)

    @staticmethod
    def _snapshot_to_dict(s: ResourceSnapshot) -> Dict[str, Any]:
        return {
            "timestamp": s.timestamp,
            "gpu_vram_used_mb": s.gpu_vram_used_mb,
            "gpu_vram_free_mb": s.gpu_vram_free_mb,
            "gpu_util_pct": s.gpu_util_pct,
            "gpu_temp_c": s.gpu_temp_c,
            "gpu_power_w": s.gpu_power_w,
            "ram_used_gb": s.ram_used_gb,
            "ram_free_gb": s.ram_free_gb,
            "cpu_util_pct": s.cpu_util_pct,
            "active_requests": s.active_requests,
            "vram_fragmentation_pct": s.vram_fragmentation_pct,
        }

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


# ------------------------------------------------------------------------------
# GPU Helper Functions (Multi‑Vendor) with Thread‑Safe Caching
# ------------------------------------------------------------------------------
_GPU_CACHE: Dict[str, Tuple[float, Any]] = {}
_GPU_CACHE_TTL = 5.0  # seconds
_GPU_CACHE_LOCK = threading.RLock()


def _cached_gpu_metrics(vendor: str, fetcher):
    """Cache GPU metrics for a short TTL, thread‑safe."""
    now = time.monotonic()
    with _GPU_CACHE_LOCK:
        if vendor in _GPU_CACHE:
            ts, data = _GPU_CACHE[vendor]
            if now - ts < _GPU_CACHE_TTL:
                return data
    data = fetcher()
    with _GPU_CACHE_LOCK:
        _GPU_CACHE[vendor] = (now, data)
    return data


def _get_gpu_metrics_nvidia() -> List[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        import subprocess
        import xml.etree.ElementTree as ET
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        try:
            root = ET.fromstring(result.stdout)
        except ET.ParseError as e:
            logger.warning("nvidia-smi XML parse error: %s", e)
            return []
        gpus = []
        for gpu in root.findall("gpu"):
            idx = int(gpu.find("minor_number").text)
            util = float(gpu.find("utilization/gpu_util").text.replace(" %", ""))
            mem = gpu.find("fb_memory_usage")
            used = int(mem.find("used").text.replace(" MiB", ""))
            total = int(mem.find("total").text.replace(" MiB", ""))
            free = int(mem.find("free").text.replace(" MiB", "")) if mem.find("free") is not None else 0
            temp = float(gpu.find("temperature/gpu_temp").text.replace(" C", ""))
            power = float(gpu.find("power_readings/power_draw").text.replace(" W", ""))
            gpus.append({
                "vendor": "nvidia",
                "index": idx,
                "vram_used_mb": used,
                "vram_free_mb": total - used if total - used >= 0 else 0,
                "vram_total_mb": total,
                "vram_free_reported_mb": free,
                "util_pct": util,
                "temp_c": temp,
                "power_w": power,
            })
        return gpus
    except Exception as e:
        logger.debug("nvidia-smi failed: %s", e)
        return []


def _get_gpu_metrics_amd() -> List[Dict[str, Any]]:
    gpus = []
    try:
        for card in sorted(Path("/sys/class/drm").glob("card*")):
            idx = int(card.name[4:]) if card.name[4:].isdigit() else 0
            total_f = card / "device" / "mem_info_vram_total"
            used_f = card / "device" / "mem_info_vram_used"
            if not total_f.exists():
                total_f = card / "device" / "local_mem_total"
                used_f = card / "device" / "local_mem_used"
            util_f = card / "device" / "gpu_busy_percent"
            temp_f = None
            hwmon = card / "device" / "hwmon"
            if hwmon.exists():
                for hw in hwmon.iterdir():
                    tf = hw / "temp1_input"
                    if tf.exists():
                        temp_f = tf
                        break
            if total_f.exists() and used_f.exists():
                total = int(total_f.read_text().strip())
                used = int(used_f.read_text().strip())
                util = float(util_f.read_text().strip()) if util_f.exists() else 0.0
                temp = int(temp_f.read_text().strip()) / 1000.0 if temp_f and temp_f.exists() else 0.0
                gpus.append({
                    "vendor": "amd",
                    "index": idx,
                    "vram_used_mb": used // (1024 * 1024),
                    "vram_free_mb": (total - used) // (1024 * 1024),
                    "vram_total_mb": total // (1024 * 1024),
                    "vram_free_reported_mb": 0,
                    "util_pct": util,
                    "temp_c": temp,
                    "power_w": 0.0,
                })
        return gpus
    except Exception as e:
        logger.debug("AMD GPU detection failed: %s", e)
        return []


def _get_gpu_metrics_intel() -> List[Dict[str, Any]]:
    gpus = []
    try:
        for card in sorted(Path("/sys/class/drm").glob("card*")):
            idx = int(card.name[4:]) if card.name[4:].isdigit() else 0
            total_f = card / "device" / "local_mem_total"
            used_f = card / "device" / "local_mem_used"
            util_f = card / "device" / "gpu_busy_percent"
            temp_f = None
            hwmon = card / "device" / "hwmon"
            if hwmon.exists():
                for hw in hwmon.iterdir():
                    tf = hw / "temp1_input"
                    if tf.exists():
                        temp_f = tf
                        break
            if total_f.exists() and used_f.exists():
                total = int(total_f.read_text().strip())
                used = int(used_f.read_text().strip())
                util = float(util_f.read_text().strip()) if util_f.exists() else 0.0
                temp = int(temp_f.read_text().strip()) / 1000.0 if temp_f and temp_f.exists() else 0.0
                gpus.append({
                    "vendor": "intel",
                    "index": idx,
                    "vram_used_mb": used // (1024 * 1024),
                    "vram_free_mb": (total - used) // (1024 * 1024),
                    "vram_total_mb": total // (1024 * 1024),
                    "vram_free_reported_mb": 0,
                    "util_pct": util,
                    "temp_c": temp,
                    "power_w": 0.0,
                })
        return gpus
    except Exception as e:
        logger.debug("Intel GPU detection failed: %s", e)
        return []


def _get_all_gpu_metrics() -> List[Dict[str, Any]]:
    gpus = []
    gpus.extend(_cached_gpu_metrics("nvidia", _get_gpu_metrics_nvidia))
    gpus.extend(_cached_gpu_metrics("amd", _get_gpu_metrics_amd))
    gpus.extend(_cached_gpu_metrics("intel", _get_gpu_metrics_intel))
    return gpus


def _compute_vram_fragmentation_pct(gpus: List[Dict[str, Any]]) -> float:
    total = 0.0
    fragmented = 0.0
    for gpu in gpus:
        if gpu["vendor"] == "nvidia":
            used = gpu["vram_used_mb"]
            free_reported = gpu.get("vram_free_reported_mb", 0)
            total_mb = gpu.get("vram_total_mb", used + free_reported)
            if total_mb > 0:
                frag = total_mb - used - free_reported
                fragmented += max(0.0, frag)
            total += total_mb
    if total > 0:
        return (fragmented / total) * 100.0
    return 0.0


# ------------------------------------------------------------------------------
# Adaptive Sampling Controller
# ------------------------------------------------------------------------------
class AdaptiveSampler:
    def __init__(self, base_interval: float, cpu_threshold: float = 80.0, max_factor: float = 4.0):
        self.base_interval = base_interval
        self.cpu_threshold = cpu_threshold
        self.max_factor = max_factor
        self.current_interval = base_interval

    def adjust(self, cpu_pct: float, ram_pct: float) -> float:
        if cpu_pct >= self.cpu_threshold or ram_pct >= 90:
            factor = min(self.max_factor, 1.0 + (max(cpu_pct, ram_pct) - 80) / 20.0)
            self.current_interval = self.base_interval * factor
        else:
            self.current_interval = self.base_interval
        return self.current_interval

    @property
    def factor(self) -> float:
        return self.current_interval / self.base_interval if self.base_interval > 0 else 1.0


# ------------------------------------------------------------------------------
# Telemetry Collector (v4.1.0) – non‑blocking GPU collection
# ------------------------------------------------------------------------------
class TelemetryCollector:
    """
    Central telemetry collection with:
    - Real‑time resource monitoring (multi‑GPU).
    - Request performance tracking.
    - Classical + Predictive anomaly detection.
    - Prometheus export (optional).
    - Adaptive sampling under load.
    - File export for offline analysis.
    - Per‑model performance integration.
    - **Non‑blocking GPU queries** (run in thread pool).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = _load_config()

        telemetry_cfg = config.get("monitoring", {}).get("telemetry", {})
        self._base_collect_interval = telemetry_cfg.get("collect_interval_sec", 5.0)
        self._history_requests = telemetry_cfg.get("history_requests", 10000)
        self._history_resources = telemetry_cfg.get("history_resources", 1440)

        prometheus_enabled = telemetry_cfg.get("prometheus_enabled", False)
        prometheus_port = telemetry_cfg.get("prometheus_port", 9090)

        self._request_history: deque[RequestMetric] = deque(maxlen=self._history_requests)
        self._resource_history: deque[ResourceSnapshot] = deque(maxlen=self._history_resources)
        self._active_requests: Dict[str, float] = {}
        self._active_requests_lock = asyncio.Lock()

        self._prom: Optional[PrometheusMetrics] = None
        if PROMETHEUS_AVAILABLE and prometheus_enabled:
            self._prom = PrometheusMetrics()
            try:
                start_http_server(prometheus_port)
                logger.info("Prometheus metrics at :%d/metrics", prometheus_port)
            except Exception as e:
                logger.warning("Prometheus server failed: %s", e)

        self._anomaly = AnomalyDetector(
            window_size=telemetry_cfg.get("anomaly_window_size", 100),
            z_threshold=telemetry_cfg.get("anomaly_z_threshold", 3.0),
        )
        self._predictive = PredictiveAnomalyDetector(
            alpha=telemetry_cfg.get("predictive_alpha", 0.3),
            beta=telemetry_cfg.get("predictive_beta", 0.1),
            k=telemetry_cfg.get("predictive_k", 3.0),
        )

        self._thresholds = telemetry_cfg.get("thresholds", {
            "vram_utilization": 0.98,
            "ram_utilization": 0.95,
            "gpu_temp_c": 85,
            "latency_p99_ms": 5000,
        })
        self._min_gpu_util_for_temp_alert = telemetry_cfg.get("anomaly_min_gpu_util_for_temp_alert", 10.0)

        # Adaptive sampling
        self._sampler = AdaptiveSampler(
            base_interval=self._base_collect_interval,
            cpu_threshold=telemetry_cfg.get("adaptive_cpu_threshold", 80.0),
            max_factor=telemetry_cfg.get("adaptive_max_factor", 4.0),
        )

        # File export
        self._file_exporter: Optional[TelemetryFileExporter] = None
        export_enabled = telemetry_cfg.get("file_export_enabled", False)
        if export_enabled:
            export_path = telemetry_cfg.get("file_export_path", "/tmp/llm_telemetry.jsonl")
            self._file_exporter = TelemetryFileExporter(export_path)

        # Per‑model telemetry (lightweight integration)
        self._per_model_ttft: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._per_model_tps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._per_model_lock = asyncio.Lock()

        self._lock = asyncio.Lock()
        self._collect_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

        # Thread pool for GPU queries (non‑blocking)
        self._executor = threading.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_telemetry")

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._collect_task = asyncio.create_task(self._collect_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_requests())
        logger.info("Telemetry collection started (base interval=%.1fs)", self._base_collect_interval)

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._collect_task:
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass
            self._collect_task = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        if self._file_exporter:
            self._file_exporter.shutdown()
        self._executor.shutdown(wait=False)
        logger.info("Telemetry collection stopped")

    async def _cleanup_stale_requests(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                cutoff = time.time() - 300
                async with self._active_requests_lock:
                    stale = [rid for rid, start in self._active_requests.items() if start < cutoff]
                    for rid in stale:
                        del self._active_requests[rid]
                    if stale:
                        logger.debug("Cleaned up %d stale active requests", len(stale))
                        if self._prom and self._prom.active_requests:
                            self._prom.active_requests.set(len(self._active_requests))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Stale request cleanup error: %s", e)

    async def request_started(self, request_id: str) -> None:
        async with self._active_requests_lock:
            self._active_requests[request_id] = time.time()
        if self._prom:
            self._prom.active_requests.set(len(self._active_requests))

    async def request_completed(self, metric: RequestMetric) -> None:
        async with self._active_requests_lock:
            self._active_requests.pop(metric.request_id, None)
        self._request_history.append(metric)

        # Update per‑model stats
        async with self._per_model_lock:
            self._per_model_ttft[metric.model_id].append(metric.ttft_ms)
            self._per_model_tps[metric.model_id].append(metric.tps)

        if self._prom:
            self._prom.requests_total.labels(model=metric.model_id, status=metric.status).inc()
            self._prom.request_duration.labels(model=metric.model_id).observe(metric.total_ms / 1000.0)
            self._prom.time_to_first_token.labels(model=metric.model_id).observe(metric.ttft_ms)
            self._prom.tokens_per_second.labels(model=metric.model_id).set(metric.tps)
            self._prom.active_requests.set(len(self._active_requests))

        self._anomaly.record("ttft_ms", metric.ttft_ms, {"model": metric.model_id})
        self._anomaly.record("tps", metric.tps, {"model": metric.model_id})
        self._anomaly.record("total_ms", metric.total_ms, {"model": metric.model_id})
        self._predictive.check("ttft_ms", metric.ttft_ms, time.time())
        self._predictive.check("tps", metric.tps, time.time())

    async def _collect_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._started:
            try:
                # Run GPU metric collection in a thread pool (FIX #22)
                all_gpus = await loop.run_in_executor(self._executor, _get_all_gpu_metrics)
                cpu_pct = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                total_ram_gb = vm.total / (1024**3)
                ram_used_gb = vm.used / (1024**3)
                ram_pct = (ram_used_gb / total_ram_gb) * 100 if total_ram_gb > 0 else 0

                # Process GPU data
                gpu_vram_used: Dict[int, int] = {}
                gpu_vram_free: Dict[int, int] = {}
                gpu_util: Dict[int, float] = {}
                gpu_temp: Dict[int, float] = {}
                gpu_power: Dict[int, float] = {}

                for gpu in all_gpus:
                    idx = gpu["index"]
                    gpu_vram_used[idx] = gpu["vram_used_mb"]
                    gpu_vram_free[idx] = gpu["vram_free_mb"]
                    gpu_util[idx] = gpu["util_pct"]
                    gpu_temp[idx] = gpu["temp_c"]
                    gpu_power[idx] = gpu["power_w"]

                    if self._prom:
                        self._prom.gpu_vram_used_bytes.labels(gpu_index=idx).set(gpu["vram_used_mb"] * 1024 * 1024)
                        self._prom.gpu_temperature_celsius.labels(gpu_index=idx).set(gpu["temp_c"])
                        self._prom.gpu_utilization_pct.labels(gpu_index=idx).set(gpu["util_pct"])
                        self._prom.gpu_power_watts.labels(gpu_index=idx).set(gpu["power_w"])

                    self._anomaly.record(f"gpu_{idx}_temp", gpu["temp_c"])
                    self._anomaly.record(f"gpu_{idx}_util", gpu["util_pct"])
                    self._predictive.check(f"gpu_{idx}_temp", gpu["temp_c"], time.time())

                frag_pct = _compute_vram_fragmentation_pct(all_gpus)
                if self._prom:
                    self._prom.vram_fragmentation.set(frag_pct / 100.0)

                # Per‑model averages
                async with self._per_model_lock:
                    per_model_ttft_avg = {mid: statistics.mean(ttfts) if ttfts else 0.0
                                          for mid, ttfts in self._per_model_ttft.items()}
                    per_model_tps_avg = {mid: statistics.mean(tpss) if tpss else 0.0
                                         for mid, tpss in self._per_model_tps.items()}

                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    gpu_vram_used_mb=gpu_vram_used,
                    gpu_vram_free_mb=gpu_vram_free,
                    gpu_util_pct=gpu_util,
                    gpu_temp_c=gpu_temp,
                    gpu_power_w=gpu_power,
                    ram_used_gb=ram_used_gb,
                    ram_free_gb=vm.available / (1024**3),
                    cpu_util_pct=cpu_pct,
                    active_requests=len(self._active_requests),
                    vram_fragmentation_pct=frag_pct,
                    adaptive_interval_factor=self._sampler.factor,
                    per_model_ttft_avg=per_model_ttft_avg,
                    per_model_tps_avg=per_model_tps_avg,
                )

                async with self._lock:
                    self._resource_history.append(snapshot)

                self._check_thresholds(snapshot, total_ram_gb)
                if self._file_exporter:
                    self._file_exporter.enqueue(snapshot)

                # Adaptive interval
                interval = self._sampler.adjust(cpu_pct, ram_pct)
                if self._prom and self._prom.adaptive_interval:
                    self._prom.adaptive_interval.set(interval)
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Resource collection error: %s", e)
                await asyncio.sleep(self._base_collect_interval)

    def _check_thresholds(self, snapshot: ResourceSnapshot, total_ram_gb: float) -> None:
        if total_ram_gb > 0:
            ram_ratio = snapshot.ram_used_gb / total_ram_gb
            if ram_ratio > self._thresholds["ram_utilization"]:
                logger.warning("HIGH RAM: %.1f%% utilized", ram_ratio * 100)

        for gpu_idx, temp in snapshot.gpu_temp_c.items():
            util = snapshot.gpu_util_pct.get(gpu_idx, 0)
            if temp > self._thresholds["gpu_temp_c"] and util > self._min_gpu_util_for_temp_alert:
                logger.error("GPU %s OVERHEATING: %.1f°C (util=%.1f%%)", gpu_idx, temp, util)

    # --------------------------------------------------------------------------
    # Analysis & Reporting
    # --------------------------------------------------------------------------
    def get_performance_report(self, window_minutes: int = 60) -> Dict[str, Any]:
        cutoff = time.time() - window_minutes * 60
        recent = [m for m in self._request_history if m.timestamp > cutoff]
        if not recent:
            return {"window_minutes": window_minutes, "requests": 0}

        ttfts = [m.ttft_ms for m in recent]
        tpss = [m.tps for m in recent]
        total = [m.total_ms for m in recent]
        errors = sum(1 for m in recent if m.status != "success")

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = int(len(data) * p / 100)
            return sorted(data)[min(k, len(data) - 1)]

        by_model: Dict[str, List[RequestMetric]] = defaultdict(list)
        for m in recent:
            by_model[m.model_id].append(m)

        return {
            "window_minutes": window_minutes,
            "requests": len(recent),
            "errors": errors,
            "error_rate": f"{errors / len(recent):.1%}" if recent else "0%",
            "ttft_ms": {
                "p50": round(percentile(ttfts, 50), 1),
                "p95": round(percentile(ttfts, 95), 1),
                "p99": round(percentile(ttfts, 99), 1),
                "avg": round(statistics.mean(ttfts), 1) if ttfts else 0,
            },
            "tps": {
                "avg": round(statistics.mean(tpss), 1) if tpss else 0,
                "p50": round(percentile(tpss, 50), 1),
                "min": round(min(tpss), 1) if tpss else 0,
                "max": round(max(tpss), 1) if tpss else 0,
            },
            "total_ms": {
                "p50": round(percentile(total, 50), 1),
                "p99": round(percentile(total, 99), 1),
            },
            "by_model": {
                mid: {
                    "requests": len(ms),
                    "avg_tps": round(statistics.mean(m.tps for m in ms), 1),
                    "avg_ttft_ms": round(statistics.mean(m.ttft_ms for m in ms), 1),
                }
                for mid, ms in by_model.items()
            },
            "anomalies": self._anomaly.get_recent_alerts(window_minutes * 60),
        }

    def get_current_resources(self) -> Optional[ResourceSnapshot]:
        if self._resource_history:
            return self._resource_history[-1]
        return None

    def get_throughput_trend(self, bins: int = 12) -> List[Dict[str, Any]]:
        if not self._request_history:
            return []
        now = time.time()
        span = 3600
        bin_size = span / bins
        result = []
        for i in range(bins):
            t_start = now - span + i * bin_size
            t_end = t_start + bin_size
            count = sum(1 for m in self._request_history if t_start <= m.timestamp < t_end)
            avg_tps = (
                statistics.mean([m.tps for m in self._request_history if t_start <= m.timestamp < t_end])
                if count > 0 else 0.0
            )
            result.append({
                "t": round(t_start - now, 0),
                "rps": round(count / bin_size, 2),
                "tps": round(avg_tps, 1),
            })
        return result


# ------------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------------
def create_telemetry_collector(cfg: Optional[Dict[str, Any]] = None) -> TelemetryCollector:
    return TelemetryCollector(cfg)


def get_telemetry_collector(config_path: Optional[Path] = None) -> TelemetryCollector:
    cfg = _load_config(config_path)
    return TelemetryCollector(cfg)


# ------------------------------------------------------------------------------
# Standalone Monitoring Daemon (Optional)
# ------------------------------------------------------------------------------
async def _monitor_loop(collector: TelemetryCollector, interval: int = 30) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        from rich import box
    except ImportError:
        logger.error("Rich not installed; cannot run TUI monitor.")
        return

    console = Console()
    await collector.start()

    with Live(console=console, refresh_per_second=0.5) as live:
        while True:
            report = collector.get_performance_report(window_minutes=5)
            snapshot = collector.get_current_resources()
            table = Table(title="LLM Cloud Monitor", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Requests (5m)", str(report.get("requests", 0)))
            table.add_row("Error rate", report.get("error_rate", "N/A"))
            table.add_row("TTFT p50", f"{report.get('ttft_ms', {}).get('p50', 0)}ms")
            table.add_row("TTFT p99", f"{report.get('ttft_ms', {}).get('p99', 0)}ms")
            table.add_row("Avg TPS", str(report.get("tps", {}).get("avg", 0)))
            if snapshot:
                for gpu_idx, temp in snapshot.gpu_temp_c.items():
                    table.add_row(f"GPU {gpu_idx} Temp", f"{temp}°C")
                table.add_row("RAM Used", f"{snapshot.ram_used_gb:.1f}GB")
                table.add_row("CPU", f"{snapshot.cpu_util_pct:.0f}%")
            live.update(table)
            await asyncio.sleep(interval)


def main() -> None:
    import click

    @click.command()
    @click.option("--prometheus-port", default=9090, type=int)
    @click.option("--interval", default=30, type=int)
    @click.option("--config", "config_path", type=Path, help="Path to config.yaml")
    def run(prometheus_port: int, interval: int, config_path: Optional[Path]) -> None:
        cfg = _load_config(config_path)
        cfg.setdefault("monitoring", {}).setdefault("telemetry", {})
        cfg["monitoring"]["telemetry"]["prometheus_port"] = prometheus_port
        cfg["monitoring"]["telemetry"]["prometheus_enabled"] = True
        collector = TelemetryCollector(cfg)
        asyncio.run(_monitor_loop(collector, interval))

    run()


if __name__ == "__main__":
    main()


__all__ = [
    "TelemetryCollector",
    "RequestMetric",
    "ResourceSnapshot",
    "create_telemetry_collector",
    "get_telemetry_collector",
]
