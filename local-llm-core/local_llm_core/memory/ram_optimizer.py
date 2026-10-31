from __future__ import annotations

"""
memory/ram_optimizer.py — Production‑Ready RAM Optimizer
========================================================
v1.0.0 – Config injected via constructor; no filesystem reads.

Two‑level RAM cache (hot/warm) with LZ4 compression, async prefetch
with bounded concurrency, mmap LRU, and GC‑triggered pressure relief.

All parameters are taken from the `config` dict (or a `RAMConfig`
model).  Sensible defaults are used when config is omitted.

"""

import asyncio
import gc
import logging
import mmap
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LZ4 availability
# -----------------------------------------------------------------------------
try:
    import lz4.frame as _lz4
    _LZ4 = True
except ImportError:
    _LZ4 = False
    log.warning("lz4 not available — RAM compression disabled")

# tracemalloc support (optional)
try:
    import tracemalloc
    _TRACEMALLOC = True
except ImportError:
    _TRACEMALLOC = False
    log.warning("tracemalloc not available – memory profiling disabled")


# -----------------------------------------------------------------------------
# RAMOptimizer
# -----------------------------------------------------------------------------
class RAMOptimizer:
    """
    Two‑level RAM cache:
      hot  — decompressed bytes, O(1) access
      warm — LZ4‑compressed bytes, promoted to hot on access

    Thread‑safe. Async prefetch with bounded concurrency.
    mmap LRU with configurable FD limit.

    Added optional tracemalloc profiling.
    """

    # Hardcoded fallback defaults (used when no config key present)
    _DEFAULT_MAX_MMAPS = 64
    _DEFAULT_MAX_PREFETCH = 16

    def __init__(
        self,
        max_ram_mb: int = 0,
        compression: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ---- normalise config ------------------------------------------------
        mem_cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import RAMConfig
                # Accept either a plain dict or a Pydantic model
                if isinstance(config, dict):
                    cfg_model = RAMConfig(**config.get("memory", {}))
                else:
                    cfg_model = RAMConfig(**config.get("memory", {}))
                mem_cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("RAM config validation failed: %s; using defaults", exc)
                mem_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        # Determine max RAM budget
        if max_ram_mb == 0:
            max_ram_mb = mem_cfg.get("ram_max_mb", 0)
        if max_ram_mb == 0:
            # Auto‑size: use 70% of available system RAM
            max_ram_mb = int(psutil.virtual_memory().available * 0.70 / (1024**2))
        self.max_bytes = max_ram_mb * 1024 * 1024

        self._compress = compression and _LZ4
        self._ratio_thr = mem_cfg.get("ram_compress_ratio_threshold", 0.85)
        self._max_mmaps = mem_cfg.get("ram_max_mmaps", self._DEFAULT_MAX_MMAPS)
        self._max_prefetch = mem_cfg.get(
            "ram_max_prefetch_concurrent", self._DEFAULT_MAX_PREFETCH
        )

        self._hot: OrderedDict[str, bytes] = OrderedDict()
        self._warm: OrderedDict[str, bytes] = OrderedDict()
        self._hot_bytes = 0
        self._warm_bytes = 0
        self._lock = threading.RLock()

        # LRU mmap cache (path → (file_handle, mmap_object))
        self._mmaps: OrderedDict[str, Tuple] = OrderedDict()

        # tracemalloc profiling state
        self._tracemalloc_enabled = mem_cfg.get("tracemalloc_enabled", False)
        self._tracemalloc_interval_s = mem_cfg.get("tracemalloc_interval_s", 300.0)
        self._tracemalloc_snapshot = None
        self._tracemalloc_task: Optional[asyncio.Task] = None
        if self._tracemalloc_enabled and _TRACEMALLOC:
            tracemalloc.start()
            self._tracemalloc_snapshot = tracemalloc.take_snapshot()
            log.info("RAMOptimizer: tracemalloc profiling enabled (interval=%.1fs)",
                     self._tracemalloc_interval_s)

        log.info(
            "RAMOptimizer: max=%d MB lz4=%s max_mmaps=%d max_prefetch=%d",
            max_ram_mb,
            self._compress,
            self._max_mmaps,
            self._max_prefetch,
        )

    # -------------------------------------------------------------------------
    # Store / Load (unchanged logic – uses instance attrs only)
    # -------------------------------------------------------------------------
    def store(self, key: str, data: bytes, hot: bool = True) -> bool:
        with self._lock:
            if key in self._hot or key in self._warm:
                self.evict(key)
            needed = len(data)
            if needed > self.max_bytes:
                return False
            self._make_room(needed)
            if (not hot) and self._compress:
                compressed = _lz4.compress(data, compression_level=0)
                if len(compressed) < needed * self._ratio_thr:
                    self._warm[key] = compressed
                    self._warm_bytes += len(compressed)
                    return True
            self._hot[key] = data
            self._hot_bytes += needed
            return True

    def load(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key in self._hot:
                self._hot.move_to_end(key)
                return self._hot[key]
            if key in self._warm:
                compressed = self._warm.pop(key)
                self._warm_bytes -= len(compressed)
                if self._compress and _LZ4:
                    try:
                        data = _lz4.decompress(compressed)
                    except Exception:
                        log.warning(
                            "RAMOptimizer: lz4 decompress failed for '%s' — dropping", key
                        )
                        return None
                else:
                    data = compressed
                self._hot[key] = data
                self._hot.move_to_end(key)
                self._hot_bytes += len(data)
                return data
        return None

    def evict(self, key: str) -> bool:
        with self._lock:
            if key in self._hot:
                self._hot_bytes -= len(self._hot.pop(key))
                return True
            if key in self._warm:
                self._warm_bytes -= len(self._warm.pop(key))
                return True
        return False

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._hot or key in self._warm

    # -------------------------------------------------------------------------
    # Async Prefetch
    # -------------------------------------------------------------------------
    async def prefetch(
        self,
        keys: list[str],
        loader: Callable[[str], bytes],
        max_concurrent: int = 0,
    ) -> Tuple[int, int]:
        limit = max_concurrent or self._max_prefetch
        sem = asyncio.Semaphore(limit)
        loop = asyncio.get_running_loop()
        errors = 0

        async def _one(k: str) -> bool:
            nonlocal errors
            if self.contains(k):
                return True
            async with sem:
                try:
                    data = await loop.run_in_executor(None, loader, k)
                    if data:
                        return self.store(k, data, hot=False)
                    return False
                except Exception:
                    errors += 1
                    log.warning(
                        "RAMOptimizer.prefetch: loader failed for '%s'",
                        k,
                        exc_info=True,
                    )
                    return False

        results = await asyncio.gather(*[_one(k) for k in keys], return_exceptions=False)
        n_ok = sum(1 for r in results if r is True)
        if errors:
            log.warning("RAMOptimizer.prefetch: %d/%d keys failed", errors, len(keys))
        return n_ok, errors

    # -------------------------------------------------------------------------
    # mmap Helpers (LRU FD limit)
    # -------------------------------------------------------------------------
    def mmap_open(self, path: str) -> Optional[mmap.mmap]:
        with self._lock:
            if path in self._mmaps:
                self._mmaps.move_to_end(path)
                return self._mmaps[path][1]

            p = Path(path)
            if not p.exists():
                return None

            while len(self._mmaps) >= self._max_mmaps:
                lru_path, (lru_fh, lru_mm) = self._mmaps.popitem(last=False)
                try:
                    lru_mm.close()
                    lru_fh.close()
                    log.info("RAMOptimizer: mmap LRU evicted: %s", lru_path)
                except Exception:
                    pass

            try:
                fh = open(path, "rb")
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                self._mmaps[path] = (fh, mm)
                self._mmaps.move_to_end(path)
                log.info(
                    "RAMOptimizer: mmap opened: %s (%d MB)",
                    path,
                    p.stat().st_size >> 20,
                )
                return mm
            except Exception:
                log.warning("RAMOptimizer: mmap_open failed for %s", path, exc_info=True)
                return None

    def mmap_close(self, path: str) -> None:
        with self._lock:
            entry = self._mmaps.pop(path, None)
            if entry:
                try:
                    entry[1].close()
                    entry[0].close()
                except Exception:
                    pass

    def close_all_mmaps(self) -> int:
        with self._lock:
            count = len(self._mmaps)
            for path in list(self._mmaps.keys()):
                fh, mm = self._mmaps.pop(path)
                try:
                    mm.close()
                    fh.close()
                except Exception:
                    pass
            log.info("RAMOptimizer: closed %d mmap handles", count)
            return count

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        self.stop_profiling()
        self.close_all_mmaps()
        with self._lock:
            self._hot.clear()
            self._warm.clear()
            self._hot_bytes = 0
            self._warm_bytes = 0
        log.info("RAMOptimizer shutdown complete")

    # -------------------------------------------------------------------------
    # System Helpers
    # -------------------------------------------------------------------------
    def available_system_bytes(self) -> int:
        return psutil.virtual_memory().available

    def system_utilization(self) -> float:
        return psutil.virtual_memory().percent / 100.0

    # 🔥 ISSUE 4 FIX: removed duplicate gc.collect()
    def pressure_relief(self) -> int:
        before = psutil.virtual_memory().used
        gc.collect()
        freed = max(0, before - psutil.virtual_memory().used)
        if freed:
            log.info("RAMOptimizer: GC freed ~%d MB", freed >> 20)
        return freed

    def safe_allocation_bytes(self, target_utilization: float = 0.70) -> int:
        vm = psutil.virtual_memory()
        return max(0, int(vm.total * target_utilization) - vm.used)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hot_bytes + self._warm_bytes
            vm = psutil.virtual_memory()
            return {
                "max_mb": self.max_bytes >> 20,
                "hot_mb": self._hot_bytes >> 20,
                "warm_mb": self._warm_bytes >> 20,
                "used_mb": total >> 20,
                "free_mb": (self.max_bytes - total) >> 20,
                "utilization": round(total / max(self.max_bytes, 1) * 100, 2),
                "hot_keys": len(self._hot),
                "warm_keys": len(self._warm),
                "mmapped_files": len(self._mmaps),
                "mmap_limit": self._max_mmaps,
                "system_ram_pct": vm.percent,
                "tracemalloc": {
                    "enabled": self._tracemalloc_enabled and _TRACEMALLOC,
                    "interval_s": self._tracemalloc_interval_s,
                } if self._tracemalloc_enabled else None,
            }

    # -------------------------------------------------------------------------
    # tracemalloc profiling harness
    # -------------------------------------------------------------------------
    async def start_profiling(self) -> None:
        """Start the background tracemalloc diff task."""
        if not self._tracemalloc_enabled or not _TRACEMALLOC:
            return
        if self._tracemalloc_task is not None and not self._tracemalloc_task.done():
            return
        self._tracemalloc_task = asyncio.create_task(self._profiling_loop(), name="tracemalloc")

    def stop_profiling(self) -> None:
        """Cancel the tracemalloc task. Sync-friendly.

        Shutdown runs from a synchronous context (RAMOptimizer.shutdown is
        called from api_server's lifespan finally block, which iterates a
        generic loop over managers that may or may not be async). The task
        body already handles CancelledError cleanly — cancel() schedules
        that unwind on the event loop, and dropping the reference here
        releases our strong handle on it. We do not await the completion
        because shutdown must not be a suspension point for a background
        diagnostics task that is off by default.
        """
        if self._tracemalloc_task:
            self._tracemalloc_task.cancel()
            self._tracemalloc_task = None

    async def _profiling_loop(self) -> None:
        """Periodically log tracemalloc memory diffs."""
        if not _TRACEMALLOC:
            return
        while True:
            try:
                await asyncio.sleep(self._tracemalloc_interval_s)
                if self._tracemalloc_snapshot is None:
                    self._tracemalloc_snapshot = tracemalloc.take_snapshot()
                    continue
                current = tracemalloc.take_snapshot()
                top_stats = current.compare_to(self._tracemalloc_snapshot, 'lineno')
                log.info("=== tracemalloc diff (last %.1fs) ===",
                         self._tracemalloc_interval_s)
                for stat in top_stats[:10]:
                    log.info("  %s", stat)
                self._tracemalloc_snapshot = current
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("tracemalloc diff error: %s", exc)

    # -------------------------------------------------------------------------
    # Private Helpers (unchanged)
    # -------------------------------------------------------------------------
    def _make_room(self, needed: int) -> None:
        while self._hot_bytes + self._warm_bytes + needed > self.max_bytes:
            if self._hot:
                key, data = self._hot.popitem(last=False)
                self._hot_bytes -= len(data)
                if self._compress and _LZ4:
                    try:
                        c = _lz4.compress(data)
                        self._warm[key] = c
                        self._warm_bytes += len(c)
                    except Exception:
                        log.warning(
                            "RAMOptimizer: lz4 compress failed on eviction for '%s'", key
                        )
                elif self._warm_bytes + len(data) <= self.max_bytes:
                    self._warm[key] = data
                    self._warm_bytes += len(data)
            elif self._warm:
                _, data = self._warm.popitem(last=False)
                self._warm_bytes -= len(data)
            else:
                break
