"""
local_llm_server/monitoring/__init__.py — Monitoring & Observability v1.0.0
=============================================================================
Public interface for all monitoring, telemetry, and performance analysis components.

  - get_monitoring_components now respects config flags (enabled/disabled).
  - MonitoringManager starts metrics collector automatically.
  - Added config validation for component creation.
  - Reduced duplication (still points to shared helpers, but documented).
  - All previous features retained.

Exports:
    - MetricsCollector, MetricSnapshot
    - TelemetryCollector, RequestMetric, ResourceSnapshot, PredictiveAnomalyDetector,
      TelemetryFileExporter, AdaptiveSampler
    - ModelTelemetry, ModelProfile, ModelRankingEngine
    - PerformanceAnalyzer, PerformanceHistory, SLAResult, PerformanceGrade,
      PredictiveSLAViolation, HardwareTierSuggester
    - Factories: create_metrics_collector, get_metrics_collector,
                 create_telemetry_collector, get_telemetry_collector,
                 create_model_telemetry, get_model_telemetry,
                 create_performance_analyzer, get_performance_analyzer
    - get_monitoring_components, shutdown_monitoring_components
    - MonitoringManager (for FastAPI lifespan)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import yaml

# ------------------------------------------------------------------------------
# Package metadata
# ------------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "Devellopase"

# ------------------------------------------------------------------------------
# Logger (no handler added – let application configure)
# ------------------------------------------------------------------------------
_log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration loading (shared with other modules – TODO: move to config.py)
# ------------------------------------------------------------------------------
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_PATH: Optional[Path] = None


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
    """Load configuration from YAML file (cached per path)."""
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH

    if config_path == _CONFIG_CACHE_PATH and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    cfg: Dict[str, Any] = {}
    if config_path is not None:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                _log.error("Failed to load config %s: %s", config_path, e)
        else:
            _log.warning("Config file not found: %s", config_path)
    else:
        paths = [
            Path.home() / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for p in paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    config_path = p
                    break
                except Exception as e:
                    _log.error("Failed to load config %s: %s", p, e)
        else:
            _log.warning("No config.yaml found; using default monitoring settings.")

    cfg = _expand_env_vars(cfg)
    _CONFIG_CACHE = cfg
    _CONFIG_CACHE_PATH = config_path
    return cfg


# ------------------------------------------------------------------------------
# Lazy imports – entire modules cached to avoid repeated attribute access costs
# ------------------------------------------------------------------------------
_MODULE_CACHE: Dict[str, Any] = {}


def _import_module(name: str) -> Any:
    """Import and cache a monitoring submodule."""
    if name not in _MODULE_CACHE:
        if name == "metrics_collector":
            from . import metrics_collector as mod
        elif name == "telemetry":
            from . import telemetry as mod
        elif name == "model_telemetry":
            from . import model_telemetry as mod
        elif name == "performance_analyzer":
            from . import performance_analyzer as mod
        else:
            raise ImportError(f"Unknown monitoring submodule: {name}")
        _MODULE_CACHE[name] = mod
    return _MODULE_CACHE[name]


# ------------------------------------------------------------------------------
# Dynamic attribute access (lazy and cached)
# ------------------------------------------------------------------------------
def __getattr__(name: str):
    """Lazy load module attributes, caching after first access."""
    # Metrics Collector
    if name in ("MetricsCollector", "MetricSnapshot", "create_metrics_collector", "get_metrics_collector"):
        mod = _import_module("metrics_collector")
        return getattr(mod, name)

    # Telemetry Collector
    if name in ("TelemetryCollector", "RequestMetric", "ResourceSnapshot",
                "create_telemetry_collector", "get_telemetry_collector",
                "PredictiveAnomalyDetector", "TelemetryFileExporter", "AdaptiveSampler"):
        mod = _import_module("telemetry")
        return getattr(mod, name)

    # Model Telemetry
    if name in ("ModelTelemetry", "ModelProfile", "ModelRankingEngine",
                "create_model_telemetry", "get_model_telemetry",
                "model_idle_reaper", "warmup_model", "TelemetryBackgroundTasks"):
        mod = _import_module("model_telemetry")
        return getattr(mod, name)

    # Performance Analyzer
    if name in ("PerformanceAnalyzer", "PerformanceHistory", "SLAResult",
                "PerformanceGrade", "create_performance_analyzer",
                "get_performance_analyzer", "PredictiveSLAViolation",
                "HardwareTierSuggester", "evaluate_sla"):
        mod = _import_module("performance_analyzer")
        return getattr(mod, name)

    raise AttributeError(f"module 'local_llm_server.monitoring' has no attribute {name!r}")


# ------------------------------------------------------------------------------
# Exports (for documentation and IDE discovery)
# ------------------------------------------------------------------------------
__all__ = [
    # Metrics
    "MetricsCollector",
    "MetricSnapshot",
    "create_metrics_collector",
    "get_metrics_collector",
    # Telemetry
    "TelemetryCollector",
    "RequestMetric",
    "ResourceSnapshot",
    "PredictiveAnomalyDetector",
    "TelemetryFileExporter",
    "AdaptiveSampler",
    "create_telemetry_collector",
    "get_telemetry_collector",
    # Model Telemetry
    "ModelTelemetry",
    "ModelProfile",
    "ModelRankingEngine",
    "model_idle_reaper",
    "warmup_model",
    "TelemetryBackgroundTasks",
    "create_model_telemetry",
    "get_model_telemetry",
    # Performance
    "PerformanceAnalyzer",
    "PerformanceHistory",
    "SLAResult",
    "PerformanceGrade",
    "PredictiveSLAViolation",
    "HardwareTierSuggester",
    "evaluate_sla",
    "create_performance_analyzer",
    "get_performance_analyzer",
    # High‑level
    "get_monitoring_components",
    "shutdown_monitoring_components",
    "MonitoringManager",
]


# ------------------------------------------------------------------------------
# Type stubs for IDE support (conditionally defined)
# ------------------------------------------------------------------------------
if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector, MetricSnapshot
    from .telemetry import (
        TelemetryCollector, RequestMetric, ResourceSnapshot,
        PredictiveAnomalyDetector, TelemetryFileExporter, AdaptiveSampler
    )
    from .model_telemetry import (
        ModelTelemetry, ModelProfile, ModelRankingEngine,
        model_idle_reaper, warmup_model, TelemetryBackgroundTasks
    )
    from .performance_analyzer import (
        PerformanceAnalyzer, PerformanceHistory, SLAResult, PerformanceGrade,
        PredictiveSLAViolation, HardwareTierSuggester, evaluate_sla
    )


# ------------------------------------------------------------------------------
# High‑level initializer for all monitoring components (respects config flags)
# ------------------------------------------------------------------------------
async def get_monitoring_components(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Optional["MetricsCollector"],
    Optional["TelemetryCollector"],
    Optional["ModelTelemetry"],
    Optional["PerformanceAnalyzer"],
]:
    """
    Initialize all monitoring components from configuration.
    Components that are disabled in config will be returned as None.

    Args:
        config_path: Optional explicit path to config.yaml. Ignored if `config` is supplied.
        config: Pre‑loaded configuration dictionary. If provided, overrides file loading.

    Returns:
        Tuple of (metrics_collector, telemetry_collector, model_telemetry, performance_analyzer).
        Some components may be None if disabled in config.
    """
    if config is None:
        cfg = _load_config(config_path)
    else:
        cfg = config

    # Check component enablement flags
    monitoring_cfg = cfg.get("monitoring", {})
    metrics_enabled = monitoring_cfg.get("metrics", {}).get("enabled", True)
    telemetry_enabled = monitoring_cfg.get("telemetry", {}).get("enabled", True)
    model_telemetry_enabled = monitoring_cfg.get("model_telemetry", {}).get("enabled", True)
    performance_enabled = monitoring_cfg.get("performance", {}).get("enabled", True)

    mod_mc = _import_module("metrics_collector")
    mod_tc = _import_module("telemetry")
    mod_mt = _import_module("model_telemetry")
    mod_pa = _import_module("performance_analyzer")

    metrics_collector = mod_mc.create_metrics_collector(cfg) if metrics_enabled else None
    telemetry_collector = mod_tc.create_telemetry_collector(cfg) if telemetry_enabled else None
    model_telemetry = mod_mt.create_model_telemetry(cfg) if model_telemetry_enabled else None
    performance_analyzer = mod_pa.create_performance_analyzer(cfg) if performance_enabled else None

    _log.info(
        "Monitoring components initialized: metrics=%s, telemetry=%s, model_telemetry=%s, performance=%s",
        metrics_collector is not None,
        telemetry_collector is not None,
        model_telemetry is not None,
        performance_analyzer is not None,
    )
    return metrics_collector, telemetry_collector, model_telemetry, performance_analyzer


async def shutdown_monitoring_components(
    metrics_collector: Optional[Any] = None,
    telemetry_collector: Optional[Any] = None,
    model_telemetry: Optional[Any] = None,
    performance_analyzer: Optional[Any] = None,
) -> None:
    """
    Gracefully shut down monitoring components.

    Args:
        metrics_collector: MetricsCollector instance.
        telemetry_collector: TelemetryCollector instance.
        model_telemetry: ModelTelemetry instance (no‑op shut down).
        performance_analyzer: PerformanceAnalyzer instance (no‑op shut down, kept for symmetry).
    """
    if telemetry_collector is not None and hasattr(telemetry_collector, "stop"):
        await telemetry_collector.stop()
        _log.debug("TelemetryCollector stopped")

    if metrics_collector is not None and hasattr(metrics_collector, "shutdown"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, metrics_collector.shutdown)
        _log.debug("MetricsCollector shut down")

    # ModelTelemetry and PerformanceAnalyzer do not require explicit cleanup
    _log.info("Monitoring components shut down")


# ------------------------------------------------------------------------------
# MonitoringManager – async context manager for FastAPI lifespan
# ------------------------------------------------------------------------------
class MonitoringManager:
    """
    Async context manager that initializes telemetry, metrics, and optional
    background tasks on entry, then gracefully shuts everything down on exit.

    Usage:
        async with MonitoringManager(config_path="config.yaml") as components:
            # components.metrics, components.telemetry, etc.
            app.state.metrics = components.metrics
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._config_path = config_path
        self._config = config
        self._metrics: Optional[Any] = None
        self._telemetry: Optional[Any] = None
        self._model_telemetry: Optional[Any] = None
        self._performance: Optional[Any] = None

    async def __aenter__(self) -> "MonitoringManager":
        self._metrics, self._telemetry, self._model_telemetry, self._performance = \
            await get_monitoring_components(self._config_path, config=self._config)

        # Start telemetry collector if present
        if self._telemetry:
            await self._telemetry.start()

        # Metrics collector starts its own thread automatically – nothing to do.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await shutdown_monitoring_components(
            self._metrics, self._telemetry, self._model_telemetry, self._performance
        )

    @property
    def metrics(self):
        return self._metrics

    @property
    def telemetry(self):
        return self._telemetry

    @property
    def model_telemetry(self):
        return self._model_telemetry

    @property
    def performance(self):
        return self._performance


# ------------------------------------------------------------------------------
# Module initialization logging
# ------------------------------------------------------------------------------
_log.debug("monitoring package loaded (v%s)", __version__)
