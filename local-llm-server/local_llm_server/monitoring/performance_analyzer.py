from __future__ import annotations

"""
local_llm_server/monitoring/performance_analyzer.py — Production SLA Analysis
============================================================================================
Evaluates inference performance against configurable SLA targets, predicts future
violations using linear regression, suggests hardware upgrades, and provides
actionable recommendations.


version = 1.0.0
"""

import copy
import logging
import time
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Configuration Loading with Mtime Invalidation
# ------------------------------------------------------------------------------
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_PATH: Optional[Path] = None
_CONFIG_CACHE_MTIME: float = 0.0
_CONFIG_CACHE_LOCK = threading.RLock()
_CONFIG_TTL_SECONDS: Optional[float] = None  # set to e.g. 60 to force refresh


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with mtime-based invalidation."""
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH, _CONFIG_CACHE_MTIME

    if config_path is None:
        candidates = [Path.home() / "config.yaml", Path.cwd() / "config.yaml"]
        for p in candidates:
            if p.exists():
                config_path = p
                break
        if config_path is None:
            log.warning("No config.yaml found; using default SLA targets.")
            return {}

    with _CONFIG_CACHE_LOCK:
        need_reload = False
        if config_path == _CONFIG_CACHE_PATH and _CONFIG_CACHE is not None:
            if _CONFIG_TTL_SECONDS is not None:
                age = time.time() - _CONFIG_CACHE_MTIME
                if age > _CONFIG_TTL_SECONDS:
                    need_reload = True
            else:
                try:
                    current_mtime = config_path.stat().st_mtime
                    if current_mtime != _CONFIG_CACHE_MTIME:
                        need_reload = True
                except OSError:
                    need_reload = True
        else:
            need_reload = True

        if not need_reload:
            return _CONFIG_CACHE

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            _CONFIG_CACHE = cfg
            _CONFIG_CACHE_PATH = config_path

            try:
                _CONFIG_CACHE_MTIME = config_path.stat().st_mtime
            except OSError:
                _CONFIG_CACHE_MTIME = time.time()
            # Only log INFO when a reload was driven by a genuine
            # mtime change (not the first-ever load).
            if need_reload and _CONFIG_CACHE_PATH is not None:
                log.info("Configuration loaded from %s", config_path)
        except Exception as e:
            log.error("Failed to load config %s: %s", config_path, e)
            if _CONFIG_CACHE is not None:
                return _CONFIG_CACHE
            return {}

        return _CONFIG_CACHE


def _clear_config_cache() -> None:
    """Clear the cached configuration (useful for testing)."""
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH, _CONFIG_CACHE_MTIME
    with _CONFIG_CACHE_LOCK:
        _CONFIG_CACHE = None
        _CONFIG_CACHE_PATH = None
        _CONFIG_CACHE_MTIME = 0.0


# ------------------------------------------------------------------------------
# SLA Tier Definitions (realistic defaults)
# ------------------------------------------------------------------------------
DEFAULT_SLA_TARGETS = {
    "2gb":  {"ttft_ms": 5000,  "tps": 10.0, "description": "Integrated/APU (2GB VRAM)"},
    "4gb":  {"ttft_ms": 2500,  "tps": 20.0, "description": "Entry discrete (4GB VRAM)"},
    "8gb":  {"ttft_ms": 1200,  "tps": 50.0, "description": "Mid-range discrete (8GB VRAM)"},
    "16gb": {"ttft_ms": 600,   "tps": 70.0, "description": "High-end discrete (16GB VRAM)"},
    "24gb": {"ttft_ms": 400,   "tps": 90.0, "description": "Pro/Workstation (24GB+ VRAM)"},
}


class PerformanceGrade(Enum):
    """Overall performance grade based on SLA compliance."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class SLAResult:
    tier: str
    ttft_pass: bool
    tps_pass: bool
    ttft_ms: float
    tps: float
    ttft_target_ms: float
    tps_target: float
    grade: PerformanceGrade = PerformanceGrade.UNKNOWN

    def __post_init__(self):
        if self.ttft_pass and self.tps_pass:
            ttft_ratio = self.ttft_target_ms / max(self.ttft_ms, 0.001)
            tps_ratio = self.tps / max(self.tps_target, 0.001)
            if ttft_ratio > 1.2 and tps_ratio > 1.2:
                self.grade = PerformanceGrade.EXCELLENT
            else:
                self.grade = PerformanceGrade.GOOD
        elif self.ttft_pass or self.tps_pass:
            self.grade = PerformanceGrade.FAIR
        else:
            self.grade = PerformanceGrade.POOR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "ttft_pass": self.ttft_pass,
            "tps_pass": self.tps_pass,
            "ttft_ms": round(self.ttft_ms, 2),
            "tps": round(self.tps, 2),
            "ttft_target_ms": self.ttft_target_ms,
            "tps_target": self.tps_target,
            "grade": self.grade.value,
        }


# ------------------------------------------------------------------------------
# Performance History Tracker with optional SQLite persistence
# ------------------------------------------------------------------------------
@dataclass
class PerformanceSample:
    """A single performance measurement."""
    timestamp: float
    model_id: str
    ttft_ms: float
    tps: float
    tokens: int
    vram_used_gb: float
    model_size_gb: float


class PerformanceHistory:
    """Tracks historical performance for trend analysis, with optional SQLite storage."""

    def __init__(self, max_samples: int = 1000, db_path: Optional[str] = None):
        self._samples: deque[PerformanceSample] = deque(maxlen=max_samples)
        self._by_model: Dict[str, deque[PerformanceSample]] = {}
        self._db_path = db_path
        if db_path:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        model_id TEXT,
                        ttft_ms REAL,
                        tps REAL,
                        tokens INTEGER,
                        vram_used_gb REAL,
                        model_size_gb REAL
                    )
                """)
        except Exception as e:
            log.error("Failed to initialize performance history DB: %s", e)

    def _load_from_db(self) -> None:
        if not self._db_path:
            return
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                rows = conn.execute(
                    "SELECT timestamp, model_id, ttft_ms, tps, tokens, "
                    "vram_used_gb, model_size_gb "
                    "FROM performance_history ORDER BY timestamp ASC"
                ).fetchall()
            for row in rows:
                sample = PerformanceSample(*row)
                self.add(sample, persist=False)
            if rows:
                log.info("Loaded %d performance samples from %s", len(rows), self._db_path)
        except Exception as e:
            log.warning("Failed to load history from DB: %s", e)

    def add(self, sample: PerformanceSample, persist: bool = True) -> None:
        self._samples.append(sample)
        if sample.model_id not in self._by_model:
            self._by_model[sample.model_id] = deque(maxlen=500)
        self._by_model[sample.model_id].append(sample)

        if persist and self._db_path:
            try:
                with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                    conn.execute(
                        "INSERT INTO performance_history "
                        "(timestamp, model_id, ttft_ms, tps, tokens, "
                        "vram_used_gb, model_size_gb) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sample.timestamp, sample.model_id, sample.ttft_ms,
                         sample.tps, sample.tokens, sample.vram_used_gb,
                         sample.model_size_gb)
                    )
            except Exception as e:
                log.warning("Failed to persist performance sample: %s", e)

    def get_recent(self, model_id: Optional[str] = None, n: int = 30) -> List[PerformanceSample]:
        if model_id:
            return list(self._by_model.get(model_id, []))[-n:]
        return list(self._samples)[-n:]

    def get_model_ids(self) -> List[str]:
        """Return the model IDs that have recorded samples, in insertion order.

        Used by `PerformanceAnalyzer._build_per_model_summary()` to
        enumerate per-model stats without reaching into the private
        `_by_model` dict.
        """
        return list(self._by_model.keys())

    def get_trend(self, model_id: str, metric: str = "tps") -> Optional[float]:
        samples = list(self._by_model.get(model_id, []))
        if len(samples) < 5:
            return None
        t0 = samples[0].timestamp
        x = [s.timestamp - t0 for s in samples]
        y = [getattr(s, metric) for s in samples]
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((x[i] - mean_x) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        return num / den

    def get_per_sample_trend(self, model_id: str, metric: str = "tps") -> Optional[float]:
        samples = list(self._by_model.get(model_id, []))
        if len(samples) < 5:
            return None
        n = len(samples)
        x = list(range(n))
        y = [getattr(s, metric) for s in samples]
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((x[i] - mean_x) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def clear(self, model_id: Optional[str] = None) -> None:
        if model_id:
            self._by_model.pop(model_id, None)
        else:
            self._samples.clear()
            self._by_model.clear()

    def prune_db(self, keep_days: int = 30) -> None:
        """Delete samples older than keep_days from SQLite."""
        if not self._db_path:
            return
        cutoff = time.time() - keep_days * 86400
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                conn.execute(
                    "DELETE FROM performance_history WHERE timestamp < ?",
                    (cutoff,),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to prune performance history DB: %s", e)


# ------------------------------------------------------------------------------
# Predictive SLA Violation Detection
# ------------------------------------------------------------------------------
class PredictiveSLAViolation:
    """Uses linear regression on historical data to predict when a metric
    (TTFT or TPS) will cross its SLA threshold."""

    @staticmethod
    def predict_violation_time(
        samples: List[PerformanceSample], metric: str,
        threshold: float, increasing_is_bad: bool = True,
    ) -> Optional[float]:
        """Return the future timestamp when the metric is expected to breach
        the threshold, or None if no breach is predicted in the foreseeable future."""
        if len(samples) < 5:
            return None
        t0 = samples[0].timestamp
        x = [s.timestamp - t0 for s in samples]
        y = [getattr(s, metric) for s in samples]
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((x[i] - mean_x) ** 2 for i in range(n))
        if den == 0:
            return None
        slope = num / den
        intercept = mean_y - slope * mean_x

        if increasing_is_bad:
            if slope <= 0:
                return None
            time_from_base = (threshold - intercept) / slope
        else:
            if slope >= 0:
                return None
            time_from_base = (threshold - intercept) / slope

        if time_from_base <= 0:
            return None
        predicted_ts = t0 + time_from_base
        return predicted_ts

    @staticmethod
    def violation_summary(
        samples: List[PerformanceSample],
        ttft_target: float, tps_target: float,
    ) -> Dict[str, Any]:
        summary = {}
        ttft_violation = PredictiveSLAViolation.predict_violation_time(
            samples, "ttft_ms", ttft_target, increasing_is_bad=True
        )
        tps_violation = PredictiveSLAViolation.predict_violation_time(
            samples, "tps", tps_target, increasing_is_bad=False
        )
        now = time.time()
        if ttft_violation:
            summary["ttft_violation_in_s"] = round(max(0, ttft_violation - now))
        if tps_violation:
            summary["tps_violation_in_s"] = round(max(0, tps_violation - now))
        return summary


# ------------------------------------------------------------------------------
# Hardware Tier Suggestion
# ------------------------------------------------------------------------------
class HardwareTierSuggester:
    """Analyzes historical performance data and suggests upgrading to a higher
    VRAM tier if sustained utilisation exceeds a threshold."""

    def __init__(self, sla_targets: Dict[str, Dict[str, Any]]):
        self._sla_targets = sla_targets
        self._tiers: List[Tuple[float, str]] = []
        for tier_name, targets in sla_targets.items():
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", tier_name.lower())
            if match:
                self._tiers.append((float(match.group(1)), tier_name))
        self._tiers.sort(key=lambda x: x[0])

    def suggest(self, current_tier: str, avg_vram_used_gb: float,
                total_vram_gb: float, avg_ttft_ms: float,
                avg_tps: float) -> Optional[str]:
        """If average VRAM usage > 90% of total and TTFT/TPS are below the
        next tier's targets, recommend the next higher tier."""
        current_tier_idx = -1
        for i, (_, name) in enumerate(self._tiers):
            if name == current_tier:
                current_tier_idx = i
                break
        if current_tier_idx < 0 or current_tier_idx >= len(self._tiers) - 1:
            return None

        next_tier = self._tiers[current_tier_idx + 1]
        next_tier_name = next_tier[1]
        next_tier_targets = self._sla_targets.get(next_tier_name)
        if not next_tier_targets:
            return None

        if total_vram_gb > 0 and (avg_vram_used_gb / total_vram_gb) > 0.9:
            if (avg_ttft_ms > next_tier_targets["ttft_ms"]
                    or avg_tps < next_tier_targets["tps"]):
                return next_tier_name
        return None


# ------------------------------------------------------------------------------
# Performance Analyzer
# ------------------------------------------------------------------------------
class PerformanceAnalyzer:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Args:
            config:      explicit config dict (takes precedence, freezes config)
            config_path: path to reload config from later (if config not provided)
        """
        self._config_path = config_path
        if config is not None:
            self._config = config
            self._config_frozen = True
        else:
            self._config = None
            self._config_frozen = False
        # Confidence state must persist across config reloads; initialize
        # here, NOT in _init_from_config (which re-runs on every reload).
        self._confidence_history: deque = deque(maxlen=100)
        self._confidence_ema = 0.0
        # history is populated by _init_from_config; may be preserved across
        # reloads when the db_path is unchanged.
        self.history: Optional[PerformanceHistory] = None
        self._load_config_from_path()
        self._init_from_config()

    def _load_config_from_path(self) -> None:
        """Load (or reload) configuration from file if not frozen."""
        if self._config_frozen:
            return
        loaded = _load_config(self._config_path)
        if loaded:
            self._config = loaded
        else:
            self._config = {}

    def _init_from_config(self) -> None:
        """(Re)initialize config-derived state. Preserves accumulated history
        and confidence EMA across reloads when the DB path is unchanged —
        otherwise a mid-process config edit wipes the sample window, which
        caused the TUI's per-model panel to silently empty on any edit."""
        perf_cfg = (
            self._config.get("monitoring", {}).get("performance", {})
            if self._config else {}
        )

        # was `DEFAULT_SLA_TARGETS.copy()` — shallow copy shared the
        # inner tier dicts, so a config-provided override permanently
        # mutated the module default. Use deepcopy so the module constant
        # is immutable across instances.
        self.sla_targets = copy.deepcopy(DEFAULT_SLA_TARGETS)
        cfg_targets = perf_cfg.get("sla_targets", {})
        for tier, targets in cfg_targets.items():
            if tier in self.sla_targets:
                self.sla_targets[tier].update(targets)
            else:
                self.sla_targets[tier] = targets

        # Preserve history across reloads when the db_path hasn't changed.
        # First call (from __init__): self.history is None → allocate fresh.
        # Reload call (from _ensure_fresh_config): self.history exists.
        _new_db_path = perf_cfg.get("history_db_path", None)
        _existing_db_path = getattr(self.history, "_db_path", None) if self.history else None
        if self.history is None or _existing_db_path != _new_db_path:
            self.history = PerformanceHistory(
                max_samples=perf_cfg.get("history_size", 1000),
                db_path=_new_db_path,
            )
        # NOTE: _confidence_history and _confidence_ema are initialized in
        # __init__ and NOT touched here — they must survive config reload.

        self.trend_window = perf_cfg.get("trend_window", 30)
        self.recommendation_threshold = perf_cfg.get(
            "recommendation_threshold",
            {"ttft_degradation_pct": 20, "tps_degradation_pct": 20},
        )
        self._suggester = HardwareTierSuggester(self.sla_targets)

    def reload_config(self) -> None:
        """Force reload configuration from file (if not frozen)."""
        if self._config_frozen:
            log.warning("PerformanceAnalyzer config is frozen; reload ignored.")
            return
        self._load_config_from_path()
        self._init_from_config()
        log.info("PerformanceAnalyzer configuration reloaded.")

    def _ensure_fresh_config(self) -> None:
        """If not frozen, check if config file has changed and reload."""
        if self._config_frozen:
            return
        old_config = self._config
        self._load_config_from_path()
        if self._config != old_config:
            self._init_from_config()

    def determine_tier(self, vram_gb: float,
                       model_size_gb: Optional[float] = None) -> str:
        tier_requirements: List[Tuple[float, str]] = []
        for tier_name in self.sla_targets:
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", tier_name.lower())
            if match:
                req_gb = float(match.group(1))
                tier_requirements.append((req_gb, tier_name))
        if not tier_requirements:
            return "2gb"
        tier_requirements.sort(key=lambda x: x[0])
        selected = tier_requirements[0][1]
        for req_gb, tier_name in tier_requirements:
            if vram_gb >= req_gb:
                selected = tier_name
            else:
                break
        if model_size_gb is not None and model_size_gb > vram_gb:
            for req_gb, tier_name in reversed(tier_requirements):
                if req_gb <= vram_gb and req_gb >= model_size_gb * 0.5:
                    selected = tier_name
                    break
            else:
                selected = tier_requirements[0][1]
        return selected

    def evaluate(
        self,
        vram_gb: float,
        ttft_ms: float,
        tps: float,
        model_id: Optional[str] = None,
        model_size_gb: Optional[float] = None,
        tokens: int = 0,
    ) -> SLAResult:
        """Evaluate against SLA targets; record sample and update confidence.

        Returns `SLAResult` — unchanged. Callers that want the flat
        `/v1/performance` dict (with `confidence_index` and `per_model`)
        should use `get_sla_report()` instead.
        """
        self._ensure_fresh_config()

        tier = self.determine_tier(vram_gb, model_size_gb)
        targets = self.sla_targets.get(tier, self.sla_targets["2gb"])
        result = SLAResult(
            tier=tier,
            ttft_pass=ttft_ms <= targets["ttft_ms"],
            tps_pass=tps >= targets["tps"],
            ttft_ms=ttft_ms,
            tps=tps,
            ttft_target_ms=targets["ttft_ms"],
            tps_target=targets["tps"],
        )
        if model_id:
            sample = PerformanceSample(
                timestamp=time.time(),
                model_id=model_id,
                ttft_ms=ttft_ms,
                tps=tps,
                tokens=tokens,
                vram_used_gb=vram_gb,
                model_size_gb=model_size_gb or 0.0,
            )
            self.history.add(sample)

        # Update confidence index (0–100): higher = better SLA headroom.
        ttft_ratio = min(1.0, ttft_ms / max(targets["ttft_ms"], 1))
        tps_ratio = min(1.0, targets["tps"] / max(tps, 0.01))
        health = (1.0 - ttft_ratio) * 50 + tps_ratio * 50
        alpha = 0.1
        self._confidence_ema = self._confidence_ema * (1 - alpha) + health * alpha
        self._confidence_history.append(health)

        return result

    def confidence_index(self) -> float:
        """Return current EMA of health score 0-100."""
        return round(self._confidence_ema, 1)

    def _build_per_model_summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-model avg TTFT / TPS breakdown for `/v1/performance`.

        Empty dict when no samples have been recorded. Populated only by
        callers that pass `model_id` to `evaluate()` — the sample is added
        to `history` at that point, and this method summarizes it.
        """
        if self.history is None:
            return {}
        summary: Dict[str, Dict[str, Any]] = {}
        for mid in self.history.get_model_ids():
            samples = self.history.get_recent(mid, n=self.trend_window)
            if not samples:
                continue
            ttfts = [s.ttft_ms for s in samples]
            tpss = [s.tps for s in samples]
            summary[mid] = {
                "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1),
                "avg_tps": round(sum(tpss) / len(tpss), 1),
                "sample_count": len(samples),
            }
        return summary

    def get_sla_report(
        self,
        vram_gb: float,
        ttft_ms: float,
        tps: float,
        model_id: Optional[str] = None,
        model_size_gb: Optional[float] = None,
        tokens: int = 0,
    ) -> Dict[str, Any]:
        """— full `/v1/performance` report, TUI-ready.

        Returns a flat dict with every `SLAResult.to_dict()` field PLUS:
            confidence_index — 0–100 float, from `confidence_index()`
            per_model        — {model_id: {avg_ttft_ms, avg_tps, sample_count}}

        All keys are always present. `per_model` is `{}` when no model has
        recorded samples yet (currently the default when the caller does not
        pass `model_id` — see `evaluate_sla`'s docstring).
        """
        eval_result = self.evaluate(
            vram_gb=vram_gb, ttft_ms=ttft_ms, tps=tps,
            model_id=model_id, model_size_gb=model_size_gb, tokens=tokens,
        )
        report = eval_result.to_dict()
        report["confidence_index"] = self.confidence_index()
        report["per_model"] = self._build_per_model_summary()
        return report

    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        samples = self.history.get_recent(model_id, n=100)
        if not samples:
            return {"available": False}
        ttfts = [s.ttft_ms for s in samples]
        tpss = [s.tps for s in samples]
        ttft_p95 = None
        if len(ttfts) >= 20:
            ttft_p95 = sorted(ttfts)[int(len(ttfts) * 0.95)]
        tps_p95 = None
        if len(tpss) >= 20:
            tps_p95 = sorted(tpss)[int(len(tpss) * 0.95)]
        return {
            "available": True,
            "sample_count": len(samples),
            "ttft_avg_ms": sum(ttfts) / len(ttfts),
            "ttft_p95_ms": ttft_p95,
            "tps_avg": sum(tpss) / len(tpss),
            "tps_p95": tps_p95,
            "trend_ttft": self.history.get_trend(model_id, "ttft_ms"),
            "trend_tps": self.history.get_trend(model_id, "tps"),
        }

    def get_recommendations(self, model_id: Optional[str] = None) -> List[str]:
        self._ensure_fresh_config()

        if model_id is None:
            all_samples = self.history.get_recent(n=self.trend_window)
            if not all_samples:
                return ["Insufficient data for recommendations; run more inferences."]
            from collections import Counter
            model_counts = Counter(s.model_id for s in all_samples)
            most_common = model_counts.most_common(1)
            if most_common:
                model_id = most_common[0][0]
            else:
                return ["Insufficient data for recommendations; run more inferences."]

        samples = self.history.get_recent(model_id, n=self.trend_window)
        if len(samples) < 10:
            return ["Insufficient data for recommendations; run more inferences."]

        recs = []
        mid = len(samples) // 2
        baseline = samples[:mid]
        recent = samples[mid:]

        avg_ttft_base = sum(s.ttft_ms for s in baseline) / len(baseline) if baseline else 0
        avg_ttft_recent = sum(s.ttft_ms for s in recent) / len(recent) if recent else 0
        avg_tps_base = sum(s.tps for s in baseline) / len(baseline) if baseline else 0
        avg_tps_recent = sum(s.tps for s in recent) / len(recent) if recent else 0

        ttft_change_pct = (
            ((avg_ttft_recent - avg_ttft_base) / avg_ttft_base) * 100
            if avg_ttft_base > 0 else 0
        )
        tps_change_pct = (
            ((avg_tps_recent - avg_tps_base) / avg_tps_base) * 100
            if avg_tps_base > 0 else 0
        )

        threshold = self.recommendation_threshold

        if ttft_change_pct > threshold.get("ttft_degradation_pct", 20):
            recs.append(
                f"TTFT increased by {ttft_change_pct:.1f}% recently. "
                "Check for GPU thermal throttling or increased system load."
            )
        if tps_change_pct < -threshold.get("tps_degradation_pct", 20):
            recs.append(
                f"TPS decreased by {-tps_change_pct:.1f}% recently. "
                "Consider checking VRAM fragmentation or batch size."
            )

        latest = samples[-1]
        tier = self.determine_tier(latest.vram_used_gb, latest.model_size_gb)
        targets = self.sla_targets.get(tier, self.sla_targets["2gb"])
        violations = PredictiveSLAViolation.violation_summary(
            samples, targets["ttft_ms"], targets["tps"]
        )
        if "ttft_violation_in_s" in violations:
            recs.append(
                f"TTFT predicted to breach SLA in {violations['ttft_violation_in_s']}s "
                f"({targets['ttft_ms']}ms target). Consider proactive scaling."
            )
        if "tps_violation_in_s" in violations:
            recs.append(
                f"TPS predicted to breach SLA in {violations['tps_violation_in_s']}s "
                f"({targets['tps']} target). Consider proactive scaling."
            )

        avg_vram = sum(s.vram_used_gb for s in samples) / len(samples)
        total_vram = latest.vram_used_gb
        suggestion = self._suggester.suggest(
            tier, avg_vram, total_vram, avg_ttft_recent, avg_tps_recent,
        )
        if suggestion:
            recs.append(
                f"Consider upgrading hardware to tier '{suggestion}' for better SLA compliance."
            )

        if latest.ttft_ms > targets["ttft_ms"] * 1.5:
            recs.append(
                f"TTFT ({latest.ttft_ms:.0f}ms) is >50% above target ({targets['ttft_ms']}ms). "
                "Consider using a smaller model or enabling speculative decoding."
            )
        if latest.tps < targets["tps"] * 0.5:
            recs.append(
                f"TPS ({latest.tps:.1f}) is <50% below target ({targets['tps']}). "
                "Check if you're using a quantized model or consider a GPU with "
                "more memory bandwidth."
            )

        return recs if recs else ["Performance is stable and within expectations."]

    def generate_report(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_fresh_config()

        samples = (
            self.history.get_recent(model_id, n=100) if model_id
            else self.history.get_recent(n=100)
        )
        if not samples:
            return {"status": "no_data"}

        latest = samples[-1]
        tier = self.determine_tier(latest.vram_used_gb, latest.model_size_gb)
        eval_result = self.evaluate(
            vram_gb=latest.vram_used_gb,
            ttft_ms=latest.ttft_ms,
            tps=latest.tps,
            model_id=model_id,
            model_size_gb=latest.model_size_gb,
        )
        summary = self.get_model_performance_summary(model_id) if model_id else {}
        recommendations = self.get_recommendations(model_id)
        targets = self.sla_targets.get(tier, self.sla_targets["2gb"])
        violations = PredictiveSLAViolation.violation_summary(
            samples, targets["ttft_ms"], targets["tps"]
        )

        return {
            "timestamp": time.time(),
            "model_id": model_id,
            "latest_sample": {
                "ttft_ms": latest.ttft_ms,
                "tps": latest.tps,
                "vram_used_gb": latest.vram_used_gb,
                "tokens": latest.tokens,
            },
            "sla_evaluation": eval_result.to_dict(),
            "summary": summary,
            "recommendations": recommendations,
            "sample_count": len(samples),
            "confidence_index": self.confidence_index(),
            "predicted_violations": violations,
        }


# ------------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------------
def create_performance_analyzer(cfg: Optional[Dict[str, Any]] = None) -> PerformanceAnalyzer:
    return PerformanceAnalyzer(config=cfg)


def get_performance_analyzer(config_path: Optional[Path] = None) -> PerformanceAnalyzer:
    return PerformanceAnalyzer(config_path=config_path)


# ------------------------------------------------------------------------------
# Legacy entry point — now returns Dict[str, Any]
# ------------------------------------------------------------------------------
_LEGACY_ANALYZER: Optional[PerformanceAnalyzer] = None
_LEGACY_ANALYZER_LOCK = threading.Lock()


def evaluate_sla(
    vram_gb: float,
    ttft_ms: float,
    tps: float,
    *,
    model_id: Optional[str] = None,
    model_size_gb: Optional[float] = None,
    tokens: int = 0,
) -> Dict[str, Any]:
    """Legacy entry point for `/v1/performance`. Returns a flat dict.

    Returns `SLAResult.to_dict()` fields PLUS two TUI-needed fields that
    `to_dict()` never contained:
        confidence_index — 0–100, from `PerformanceAnalyzer.confidence_index()`
        per_model        — {model_id: {avg_ttft_ms, avg_tps, sample_count}}

    The TUI's `_panel_performance` has always read these keys but
    `SLAResult.to_dict()` never included them, so the confidence gauge and
    per-model breakdown panel never rendered. This is the fix.

    Return type changed from `SLAResult` to `Dict[str, Any]`. The only
    in-tree caller — `api_server.py`'s `/v1/performance` — already handles
    both via `isinstance(sla, dict)` (returns dict directly) and
    `hasattr(sla, "to_dict")` (calls it). With the new return, the first
    branch fires and the TUI receives the enriched shape.

    `model_id` is optional. When supplied, `evaluate()` records the sample
    in `PerformanceHistory`, so subsequent `per_model` reflects it. When
    omitted, `per_model` is `{}` — honest about the fact that no per-model
    data is available. now will pass `model_id` from the current
    inference to populate this.
    """
    global _LEGACY_ANALYZER
    with _LEGACY_ANALYZER_LOCK:
        if _LEGACY_ANALYZER is None:
            _LEGACY_ANALYZER = PerformanceAnalyzer()
        return _LEGACY_ANALYZER.get_sla_report(
            vram_gb=vram_gb,
            ttft_ms=ttft_ms,
            tps=tps,
            model_id=model_id,
            model_size_gb=model_size_gb,
            tokens=tokens,
        )


# ------------------------------------------------------------------------------
# Module Exports
# ------------------------------------------------------------------------------
__all__ = [
    "PerformanceAnalyzer",
    "PerformanceHistory",
    "SLAResult",
    "PerformanceGrade",
    "evaluate_sla",
    "create_performance_analyzer",
    "get_performance_analyzer",
]
