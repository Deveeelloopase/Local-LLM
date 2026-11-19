from __future__ import annotations

"""
engine/inference/tensor_parallel.py — Production‑Ready Multi‑GPU Tensor Splitting
=================================================================================
v1.0.0 – Config‑injected, bandwidth‑aware auto strategy, extensible planning.

Splits model layers across multiple GPUs when a single GPU lacks sufficient VRAM.
Supports:
- auto          : proportional to (free VRAM × bandwidth) – maximises throughput
- even          : equal split across all GPUs
- manual        : user‑provided float ratios
- disabled      : force single‑GPU even if multiple available

All parameters are taken from the `multi_gpu` configuration section, passed in via
the `multi_gpu_config` argument.  No file I/O.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional: import the Pydantic model for validation (if available)
# -----------------------------------------------------------------------------
try:
    from local_llm_core.engine.utils.config_loader import MultiGPUConfig as MultiGPUConfigModel
except ImportError:
    MultiGPUConfigModel = None


# -----------------------------------------------------------------------------
# Strategy identifiers
# -----------------------------------------------------------------------------
class SplitStrategy:
    AUTO = "auto"
    EVEN = "even"
    MANUAL = "manual"
    DISABLED = "disabled"


# -----------------------------------------------------------------------------
# Tensor Split Plan
# -----------------------------------------------------------------------------
@dataclass
class TensorSplitPlan:
    """Describes how to split a model across multiple GPUs."""

    gpu_ids: List[int]          # list of GPU device indices (0‑based)
    split_ratios: List[float]   # VRAM ratio for each GPU (sums to 1.0)
    total_vram_mb: int          # total VRAM available across selected GPUs
    n_gpu_layers: int           # total layers on GPU (-1 = auto)
    strategy: str               # strategy used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llama_cpp_split(self) -> List[float]:
        return self.split_ratios


# -----------------------------------------------------------------------------
# Cache for computed plans (thread‑safe)
# -----------------------------------------------------------------------------
_plan_cache: Dict[str, TensorSplitPlan] = {}
_cache_lock = threading.Lock()


def _cache_signature(
    gpu_infos: List[Any],
    model_size_mb: int,
    config_snapshot: str,
) -> str:
    """Build a cache key that includes GPU IDs, model size, and config fingerprint."""
    gpu_ids = sorted(
        [getattr(g, "uuid", f"gpu{i}") for i, g in enumerate(gpu_infos)]
    )
    return f"{'|'.join(gpu_ids)}|{model_size_mb}|{config_snapshot}"


def _config_fingerprint(cfg: Dict[str, Any]) -> str:
    """Create a deterministic string from the relevant config keys."""
    # Only include keys that affect the plan
    relevant = {
        k: cfg.get(k)
        for k in ("strategy", "manual_ratios", "exclude_gpu_ids", "min_free_vram_mb", "bandwidth_weight")
    }
    return str(sorted(relevant.items()))


# -----------------------------------------------------------------------------
# GPU Compatibility
# -----------------------------------------------------------------------------
def _are_gpus_compatible(gpu_infos: List[Any]) -> bool:
    if len(gpu_infos) <= 1:
        return True
    vendors = {getattr(g, "vendor", None) for g in gpu_infos}
    if len(vendors) > 1:
        log.warning("Mixed GPU vendors – tensor split may be unstable")
        return False
    backends = {getattr(g, "backend", "unknown") for g in gpu_infos}
    if len(backends) > 1:
        log.warning("Mixed backends: %s", backends)
        return False
    return True


# -----------------------------------------------------------------------------
# Bandwidth estimation helper (bytes/sec)
# -----------------------------------------------------------------------------
def _estimate_bandwidth(gpu: Any) -> float:
    """
    Return an approximate memory bandwidth in GB/s for weighting.
    If unknown, return a conservative default.
    """
    # Try to extract from common metadata
    if hasattr(gpu, "bandwidth_gb_s"):
        return gpu.bandwidth_gb_s
    # Fallback: use vendor + VRAM size heuristics
    vendor = str(getattr(gpu, "vendor", "")).lower()
    vram_mb = getattr(gpu, "vram_total_mb", 0)
    if "nvidia" in vendor:
        if vram_mb > 20000:
            return 900.0  # H100/A100
        if vram_mb > 10000:
            return 600.0  # A6000
        return 300.0      # mid-range
    if "amd" in vendor:
        return 500.0
    # CPU fallback
    return 100.0


# -----------------------------------------------------------------------------
# Strategy implementations (pure functions returning ratios)
# -----------------------------------------------------------------------------
def _strategy_auto(
    gpu_list: List[Any],
    cfg: Dict[str, Any],
) -> List[float]:
    """
    Proportional to (free_vram_mb * bandwidth_weight).
    The `bandwidth_weight` parameter (0.0–1.0) blends between pure VRAM
    and bandwidth importance.  Default 0.5.
    """
    bw_w = cfg.get("bandwidth_weight", 0.5)
    scores = []
    for g in gpu_list:
        free_vram = getattr(g, "vram_free_mb", 0)
        bw = _estimate_bandwidth(g)
        # Scale bandwidth to roughly comparable magnitude
        # Normalise bandwidth to a per-GB basis
        score = free_vram * (1.0 - bw_w) + free_vram * (bw / 1000.0) * bw_w
        scores.append(max(score, 0.0))
    total = sum(scores)
    if total <= 0:
        # fallback to even
        return [1.0 / len(gpu_list)] * len(gpu_list)
    return [s / total for s in scores]


def _strategy_even(gpu_list: List[Any], cfg: Dict[str, Any]) -> List[float]:
    return [1.0 / len(gpu_list)] * len(gpu_list)


def _strategy_manual(gpu_list: List[Any], cfg: Dict[str, Any]) -> List[float]:
    ratios = cfg.get("manual_ratios", [])
    if not ratios or len(ratios) != len(gpu_list):
        raise ValueError("manual_ratios length must match GPU count")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("manual_ratios sum must be > 0")
    return [r / total for r in ratios]


# -----------------------------------------------------------------------------
# Core planning function
# -----------------------------------------------------------------------------
def compute_tensor_split(
    gpu_infos: List[Any],
    model_size_mb: int,
    force_refresh: bool = False,
    multi_gpu_config: Optional[Dict[str, Any]] = None,
) -> Optional[TensorSplitPlan]:
    """
    Compute optimal tensor split ratios across multiple GPUs.

    Args:
        gpu_infos: List of GPUInfo objects from hardware_detector.
        model_size_mb: Approximate model size in MB.
        force_refresh: If True, bypass cache and recompute.
        multi_gpu_config: Dict representing the `multi_gpu` config section.
                          If None, sensible defaults are used (auto, no exclusions).

    Returns:
        TensorSplitPlan if multi‑GPU is beneficial and enabled, else None.
    """
    # --- Normalise config -----------------------------------------------------
    if multi_gpu_config is None:
        multi_gpu_config = {}
    # Optionally validate with pydantic if available
    if MultiGPUConfigModel is not None and isinstance(multi_gpu_config, dict):
        try:
            validated = MultiGPUConfigModel(**multi_gpu_config)
            multi_gpu_config = validated.model_dump()
        except Exception as exc:
            log.warning("MultiGPU config validation failed: %s; using defaults", exc)
            multi_gpu_config = {}

    cfg = multi_gpu_config  # shorthand

    enabled = cfg.get("enabled", True)
    if not enabled:
        log.info("Multi‑GPU tensor splitting disabled")
        return None

    strategy = cfg.get("strategy", SplitStrategy.AUTO)
    if strategy == SplitStrategy.DISABLED:
        log.info("Multi‑GPU tensor splitting explicitly disabled")
        return None

    # --- GPU filtering --------------------------------------------------------
    supported = [
        g for g in gpu_infos
        if getattr(g, "supported", False) and getattr(g, "vram_free_mb", 0) > 0
    ]
    if len(supported) <= 1:
        log.debug("Only %d supported GPU(s); no split needed", len(supported))
        return None

    if not _are_gpus_compatible(supported):
        log.warning("GPUs incompatible; falling back to single GPU")
        return None

    # --- Config fingerprint for cache -----------------------------------------
    fingerprint = _config_fingerprint(cfg)
    cache_key = _cache_signature(supported, model_size_mb, fingerprint)

    with _cache_lock:
        if not force_refresh and cache_key in _plan_cache:
            log.debug("Returning cached tensor split plan")
            return _plan_cache[cache_key]

    # --- Apply exclusions -----------------------------------------------------
    exclude_ids = cfg.get("exclude_gpu_ids", [])
    if exclude_ids:
        supported = [
            g for i, g in enumerate(supported) if i not in exclude_ids
        ]
        if len(supported) <= 1:
            log.info("After excluding GPUs %s, only %d remain; no split", exclude_ids, len(supported))
            return None

    # --- Minimum free VRAM ----------------------------------------------------
    min_free_mb = cfg.get("min_free_vram_mb", 512)
    supported = [g for g in supported if g.vram_free_mb >= min_free_mb]
    if len(supported) <= 1:
        log.info("Not enough GPUs with ≥%d MB free VRAM", min_free_mb)
        return None

    # --- Compute ratios -------------------------------------------------------
    total_free = sum(g.vram_free_mb for g in supported)
    gpu_ids = [i for i, _ in enumerate(supported)]  # index mapping if exclusions were applied

    strategy_map: Dict[str, Callable[[List[Any], Dict[str, Any]], List[float]]] = {
        SplitStrategy.AUTO: _strategy_auto,
        SplitStrategy.EVEN: _strategy_even,
        SplitStrategy.MANUAL: _strategy_manual,
    }

    try:
        ratios = strategy_map[strategy](supported, cfg)
    except Exception as e:
        log.error("Strategy '%s' failed: %s; falling back to auto", strategy, e)
        strategy = SplitStrategy.AUTO
        ratios = _strategy_auto(supported, {})

    # --- Numerical safety -----------------------------------------------------
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        log.error("All ratios zero – falling back to even split")
        ratios = [1.0 / len(supported)] * len(supported)
    else:
        ratios = [r / total_ratio for r in ratios]

    plan = TensorSplitPlan(
        gpu_ids=gpu_ids,
        split_ratios=ratios,
        total_vram_mb=total_free,
        n_gpu_layers=-1,  # auto
        strategy=strategy,
        metadata={
            "gpu_names": [getattr(g, "name", "unknown") for g in supported],
            "vram_free_mb": [g.vram_free_mb for g in supported],
            "bandwidth_weights": [round(_estimate_bandwidth(g), 1) for g in supported],
        },
    )

    with _cache_lock:
        _plan_cache[cache_key] = plan

    return plan


# -----------------------------------------------------------------------------
# Utility: Estimate if splitting is beneficial
# -----------------------------------------------------------------------------
def should_split(
    gpu_infos: List[Any],
    model_size_mb: int,
    multi_gpu_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Quick heuristic: splitting is worthwhile only when the model does NOT fit
    in the largest single GPU's free VRAM.
    """
    plan = compute_tensor_split(gpu_infos, model_size_mb, multi_gpu_config=multi_gpu_config)
    if plan is None:
        return False
    largest_free = max(
        (g.vram_free_mb for g in gpu_infos if getattr(g, "supported", False)),
        default=0,
    )
    if model_size_mb <= largest_free:
        log.info("Model fits in single GPU (%dMB ≤ %dMB); splitting not recommended",
                 model_size_mb, largest_free)
        return False
    return True


# -----------------------------------------------------------------------------
# Cache maintenance
# -----------------------------------------------------------------------------
def clear_split_cache() -> None:
    with _cache_lock:
        _plan_cache.clear()
        log.debug("Tensor split plan cache cleared")
