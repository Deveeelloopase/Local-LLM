from __future__ import annotations

"""
memory/hybrid_cache.py — Production‑Ready Hybrid RAM+NVMe Layer Cache
======================================================================
v1.0.0 – Config injected via constructor; no filesystem reads.

Manages model layers across RAM and NVMe with automatic promotion/demotion
based on access frequency. Hot layers stay in RAM; cold layers move to NVMe.

All parameters are taken from the `config` dict (or a `HybridCacheConfig`
model) or explicit constructor arguments. Sensible defaults are used when
config is omitted.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Prometheus Metrics (optional)
# -----------------------------------------------------------------------------
def _setup_metrics():
    try:
        from prometheus_client import Counter, Gauge
        return {
            "ram_layers": Gauge("llm_hybrid_ram_layers", "Layers cached in RAM"),
            "nvme_layers": Gauge("llm_hybrid_nvme_layers", "Layers cached on NVMe"),
            "ram_used_mb": Gauge("llm_hybrid_ram_used_mb", "RAM used by layer cache"),
            "nvme_used_mb": Gauge("llm_hybrid_nvme_used_mb", "NVMe used by layer cache"),
            "promotions": Counter("llm_hybrid_promotions_total", "Layers promoted to RAM"),
            "demotions": Counter("llm_hybrid_demotions_total", "Layers demoted to NVMe"),
        }
    except ImportError:
        return {}


_METRICS = _setup_metrics()


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class LayerLocation:
    """Metadata for a single model layer."""
    layer_id: int
    size_bytes: int
    location: str  # "ram" | "nvme"
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    # EWMA of access frequency (accesses per second)
    ewma_freq: float = 0.0


# -----------------------------------------------------------------------------
# HybridCache
# -----------------------------------------------------------------------------
class HybridCache:
    """
    Manages model layers across RAM + NVMe with automatic promotion/demotion.
    """

    def __init__(
        self,
        ram_mgr: Any,  # memory.ram_optimizer.RAMOptimizer
        swap_mgr: Any,  # memory.swap_handler.SwapHandler
        ram_budget_mb: Optional[int] = None,
        nvme_budget_mb: Optional[int] = None,
        promote_threshold_freq: Optional[float] = None,  # accesses per minute
        demote_threshold_freq: Optional[float] = None,   # accesses per minute
        rebalance_interval_s: Optional[float] = None,
        ram_pressure_threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ram = ram_mgr
        self._swap = swap_mgr

        # ---- normalise config ------------------------------------------------
        cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import HybridCacheConfig as HybridCacheConfigModel
                if isinstance(config, dict):
                    cfg_model = HybridCacheConfigModel(**config.get("memory", {}))
                else:
                    cfg_model = HybridCacheConfigModel(**config.get("memory", {}))
                cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("HybridCache config validation failed: %s; using defaults", exc)
                cfg = config.get("memory", {}) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        # Use explicit kwargs, then config keys, then hardcoded defaults
        self._ram_budget_mb = ram_budget_mb if ram_budget_mb is not None else cfg.get("ram_budget_mb", 512)
        self._nvme_budget_mb = nvme_budget_mb if nvme_budget_mb is not None else cfg.get("nvme_budget_mb", 5000)

        promote_per_min = promote_threshold_freq if promote_threshold_freq is not None else cfg.get("promote_threshold_per_min", 2.0)
        demote_per_min = demote_threshold_freq if demote_threshold_freq is not None else cfg.get("demote_threshold_per_min", 0.5)
        self._promote_threshold = promote_per_min / 60.0
        self._demote_threshold = demote_per_min / 60.0

        self._rebalance_interval = rebalance_interval_s if rebalance_interval_s is not None else cfg.get("rebalance_interval_s", 60.0)
        self._ram_pressure_threshold = ram_pressure_threshold if ram_pressure_threshold is not None else cfg.get("ram_pressure_threshold", 0.90)
        self._ewma_alpha = cfg.get("ewma_alpha", 0.1)

        self._layers: Dict[int, LayerLocation] = {}
        self._ram_used_mb = 0
        self._nvme_used_mb = 0
        self._lock = asyncio.Lock()

        # Background rebalance task
        self._stop_event = asyncio.Event()
        self._rebalance_task: Optional[asyncio.Task] = None

        log.info(
            "HybridCache: ram_budget=%dMB nvme_budget=%dMB promote_thresh=%.1f/min demote_thresh=%.1f/min",
            self._ram_budget_mb,
            self._nvme_budget_mb,
            promote_per_min,
            demote_per_min,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Start background rebalance task."""
        if self._rebalance_task is None:
            self._rebalance_task = asyncio.create_task(self._rebalance_loop(), name="hybrid_cache_rebalance")

    async def shutdown(self) -> None:
        """Stop background task and clear state."""
        self._stop_event.set()
        if self._rebalance_task:
            self._rebalance_task.cancel()
            try:
                await self._rebalance_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            self._layers.clear()
            self._ram_used_mb = 0
            self._nvme_used_mb = 0
        log.info("HybridCache shutdown complete")

    async def _rebalance_loop(self) -> None:
        """Periodically rebalance layers between RAM and NVMe."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._rebalance_interval)
                await self.rebalance()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("Rebalance loop error: %s", exc)

    # -------------------------------------------------------------------------
    # Public API (unchanged logic, uses instance attrs only)
    # -------------------------------------------------------------------------
    async def store_layer(
        self,
        layer_id: int,
        data: bytes,
        prefer_ram: bool = True,
    ) -> bool:
        """Store a layer, preferring RAM if possible."""
        async with self._lock:
            size_mb = len(data) >> 20

            if prefer_ram and (self._ram_used_mb + size_mb) <= self._ram_budget_mb:
                if self._ram.store(f"layer_{layer_id}", data):
                    self._layers[layer_id] = LayerLocation(
                        layer_id=layer_id,
                        size_bytes=len(data),
                        location="ram",
                    )
                    self._ram_used_mb += size_mb
                    self._update_metrics()
                    log.debug("Layer %d: stored in RAM (%d MB)", layer_id, size_mb)
                    return True

            if (self._nvme_used_mb + size_mb) <= self._nvme_budget_mb:
                ok = await self._swap.write(f"layer_{layer_id}", data)
                if ok:
                    self._layers[layer_id] = LayerLocation(
                        layer_id=layer_id,
                        size_bytes=len(data),
                        location="nvme",
                    )
                    self._nvme_used_mb += size_mb
                    self._update_metrics()
                    log.debug("Layer %d: stored on NVMe (%d MB)", layer_id, size_mb)
                    return True

            log.error("Layer %d: no storage available", layer_id)
            return False

    async def get_layer(self, layer_id: int) -> Optional[bytes]:
        """Retrieve a layer, recording access for LRU and promotion/demotion."""
        async with self._lock:
            if layer_id not in self._layers:
                return None
            loc = self._layers[layer_id]
            now = time.monotonic()
            elapsed = now - loc.last_accessed
            loc.last_accessed = now
            loc.access_count += 1

            if elapsed > 0:
                instant_freq = 1.0 / elapsed
                loc.ewma_freq = (
                    loc.ewma_freq * (1 - self._ewma_alpha) + instant_freq * self._ewma_alpha
                )
            else:
                loc.ewma_freq = max(loc.ewma_freq, 1.0)

            if loc.location == "ram":
                data = self._ram.load(f"layer_{layer_id}")
            else:
                data = await self._swap.read(f"layer_{layer_id}")

            if loc.location == "nvme" and loc.ewma_freq >= self._promote_threshold:
                await self._promote_layer(layer_id, loc, data)

            return data

    async def _promote_layer(self, layer_id: int, loc: LayerLocation, data: bytes) -> bool:
        """Promote a layer from NVMe to RAM."""
        size_mb = loc.size_bytes >> 20
        if self._ram_used_mb + size_mb > self._ram_budget_mb:
            await self._evict_coldest_ram(size_mb)

        if self._ram_used_mb + size_mb <= self._ram_budget_mb:
            if self._ram.store(f"layer_{layer_id}", data):
                await self._swap.delete(f"layer_{layer_id}")
                self._nvme_used_mb -= size_mb
                loc.location = "ram"
                self._ram_used_mb += size_mb
                if _METRICS.get("promotions"):
                    _METRICS["promotions"].inc()
                log.info("Promoted layer %d from NVMe → RAM", layer_id)
                self._update_metrics()
                return True
        return False

    async def _evict_coldest_ram(self, needed_mb: int) -> None:
        """Evict coldest RAM layers until needed space is freed."""
        ram_layers = [loc for loc in self._layers.values() if loc.location == "ram"]
        ram_layers.sort(key=lambda loc: loc.ewma_freq)

        freed = 0
        for loc in ram_layers:
            if freed >= needed_mb:
                break
            data = self._ram.load(f"layer_{loc.layer_id}")
            if data and await self._swap.write(f"layer_{loc.layer_id}", data):
                self._ram.evict(f"layer_{loc.layer_id}")
                size_mb = loc.size_bytes >> 20
                self._ram_used_mb -= size_mb
                self._nvme_used_mb += size_mb
                loc.location = "nvme"
                freed += size_mb
                if _METRICS.get("demotions"):
                    _METRICS["demotions"].inc()
                log.debug("Evicted layer %d from RAM → NVMe", loc.layer_id)

    async def rebalance(self) -> int:
        """Demote cold RAM layers and promote hot NVMe layers."""
        async with self._lock:
            actions = 0

            ram_pressure = (self._ram_used_mb / self._ram_budget_mb) if self._ram_budget_mb > 0 else 0
            if ram_pressure > self._ram_pressure_threshold:
                ram_layers = [loc for loc in self._layers.values() if loc.location == "ram"]
                for loc in ram_layers:
                    if loc.ewma_freq < self._demote_threshold:
                        data = self._ram.load(f"layer_{loc.layer_id}")
                        if data and await self._swap.write(f"layer_{loc.layer_id}", data):
                            self._ram.evict(f"layer_{loc.layer_id}")
                            size_mb = loc.size_bytes >> 20
                            self._ram_used_mb -= size_mb
                            self._nvme_used_mb += size_mb
                            loc.location = "nvme"
                            actions += 1
                            if _METRICS.get("demotions"):
                                _METRICS["demotions"].inc()
                            log.info("Demoted layer %d from RAM → NVMe", loc.layer_id)

            nvme_layers = [loc for loc in self._layers.values() if loc.location == "nvme"]
            for loc in nvme_layers:
                if loc.ewma_freq >= self._promote_threshold:
                    data = await self._swap.read(f"layer_{loc.layer_id}")
                    if data:
                        success = await self._promote_layer(loc.layer_id, loc, data)
                        if success:
                            actions += 1

            self._update_metrics()
            return actions

    def _update_metrics(self) -> None:
        if _METRICS.get("ram_layers"):
            _METRICS["ram_layers"].set(sum(1 for loc in self._layers.values() if loc.location == "ram"))
        if _METRICS.get("nvme_layers"):
            _METRICS["nvme_layers"].set(sum(1 for loc in self._layers.values() if loc.location == "nvme"))
        if _METRICS.get("ram_used_mb"):
            _METRICS["ram_used_mb"].set(self._ram_used_mb)
        if _METRICS.get("nvme_used_mb"):
            _METRICS["nvme_used_mb"].set(self._nvme_used_mb)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "layers_in_ram": sum(1 for loc in self._layers.values() if loc.location == "ram"),
            "layers_on_nvme": sum(1 for loc in self._layers.values() if loc.location == "nvme"),
            "ram_used_mb": self._ram_used_mb,
            "ram_budget_mb": self._ram_budget_mb,
            "nvme_used_mb": self._nvme_used_mb,
            "nvme_budget_mb": self._nvme_budget_mb,
        }
