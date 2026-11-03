from __future__ import annotations

"""
engine/inference/async_weight_pipe.py — Production‑Ready Async Weight Pipeline
===============================================================================
v1.0.0 – Config‑injected, fully library‑safe, enhanced error resilience.

Background prefetcher that uses mmap and madvise(MADV_WILLNEED) to pre‑fault
transformer layer weights into the page cache before they are needed.
Reduces inference latency by eliminating page‑fault stalls during layer execution.

All configuration is passed in via the `config` parameter (a dict or
AsyncWeightPipelineConfig Pydantic model).  No file I/O.


"""

import mmap
import os
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# madvise constants
MADV_WILLNEED = 3
MADV_SEQUENTIAL = 2
MADV_DONTNEED = 4

_PAGE_SIZE = 4096


# -----------------------------------------------------------------------------
# Layer Region Dataclass (unchanged)
# -----------------------------------------------------------------------------
@dataclass
class LayerRegion:
    """Byte range in a GGUF file corresponding to one transformer layer."""
    layer_idx: int
    start_offset: int
    end_offset: int
    size_bytes: int

    def page_aligned(self) -> tuple[int, int]:
        start = (self.start_offset // _PAGE_SIZE) * _PAGE_SIZE
        end = ((self.end_offset + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
        return start, end - start


# -----------------------------------------------------------------------------
# Layer Region Estimation (now configurable)
# -----------------------------------------------------------------------------
def estimate_layer_regions(
    model_path: str,
    header_estimate_percent: float = 0.5,      # percent of file size
    max_header_bytes: int = 2 * 1024 * 1024,   # 2 MB
) -> List[LayerRegion]:
    """
    Estimate per‑layer byte ranges in a GGUF file.

    Args:
        model_path: Path to the GGUF model file.
        header_estimate_percent: Percentage of file size to treat as header (0-100).
        max_header_bytes: Hard limit for header estimate.

    Returns:
        List of LayerRegion objects, one per transformer layer.
    """
    path = Path(model_path)
    file_sz = path.stat().st_size

    try:
        from local_llm_core.engine.pipelines.layer_streamer import read_gguf_meta
        meta = read_gguf_meta(path)
        n_layers = int(meta.get("n_layer", 32))
    except Exception:
        n_layers = 32
        log.debug("Could not read n_layer from GGUF; assuming %d layers", n_layers)

    header_est = min(int(file_sz * header_estimate_percent / 100.0), max_header_bytes)
    weight_sz = file_sz - header_est
    layer_sz = weight_sz // n_layers

    regions = []
    for i in range(n_layers):
        start = header_est + i * layer_sz
        end = start + layer_sz
        regions.append(LayerRegion(
            layer_idx=i,
            start_offset=start,
            end_offset=min(end, file_sz),
            size_bytes=min(layer_sz, file_sz - start),
        ))
    log.info("LayerRegions: %d layers in %s (avg %.1f MB/layer)",
             n_layers, path.name, layer_sz / 1024 / 1024)
    return regions


# -----------------------------------------------------------------------------
# Async Weight Pipeline (config‑injected)
# -----------------------------------------------------------------------------
class AsyncWeightPipeline:
    """
    Background prefetcher using madvise(MADV_WILLNEED) for upcoming layers.

    Configuration may be passed as a dict or AsyncWeightPipelineConfig object.
    If omitted, sensible defaults are used.

    Usage:
        pipeline = AsyncWeightPipeline(model_path)
        pipeline.start()
        for layer_idx in range(n_layers):
            pipeline.advance(layer_idx)
            compute_layer(layer_idx)
        pipeline.stop()
    """

    def __init__(
        self,
        model_path: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        prefetch_depth: Optional[int] = None,
        release_behind: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._path = model_path

        # Normalise configuration source
        if config is None:
            config = {}
        # Optional Pydantic validation (if the model is present)
        try:
            from local_llm_core.engine.utils.config_loader import AsyncWeightPipelineConfig
            if isinstance(config, dict):
                config = AsyncWeightPipelineConfig(**config).model_dump()
        except Exception:
            pass

        # Extract parameters with precedence: explicit kwargs > config > hardcoded defaults
        self._depth = (
            prefetch_depth if prefetch_depth is not None
            else config.get("prefetch_depth", 3)
        )
        self._release = (
            release_behind if release_behind is not None
            else config.get("release_behind", 2)
        )
        self._enabled = (
            enabled if enabled is not None
            else config.get("enabled", True)
        )
        # Header estimation params (used inside start())
        self._header_est_pct = config.get("header_estimate_percent", 0.5)
        self._max_header_bytes = config.get("max_header_bytes", 2 * 1024 * 1024)
        self._idle_sleep = config.get("idle_sleep_ms", 5) / 1000.0
        self._thread_join_timeout = config.get("thread_join_timeout_s", 2.0)

        # Platform check
        if os.name != "posix":
            self._enabled = False
            log.info("AsyncWeightPipeline: disabled (non‑POSIX platform)")

        self._regions: List[LayerRegion] = []
        self._mm: Optional[mmap.mmap] = None
        self._fh = None
        self._lock = threading.Lock()
        self._current = -1
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._queue: List[int] = []      # prefetch/release commands

        # Stats
        self._prefetch_count = 0
        self._release_count = 0
        self._start_ts = 0.0

        if not self._enabled:
            log.info("AsyncWeightPipeline: disabled by configuration")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """Open the model file and start the prefetch background thread."""
        if not self._enabled:
            return

        fh = None
        mm = None
        try:
            # Estimate layer regions (using our configurable header params)
            self._regions = estimate_layer_regions(
                self._path,
                header_estimate_percent=self._header_est_pct,
                max_header_bytes=self._max_header_bytes,
            )
            fh = open(self._path, "rb")

            _mmap_kwargs: Dict[str, Any] = {"access": mmap.ACCESS_READ}
            if hasattr(mmap, "MAP_SHARED"):
                _mmap_kwargs["flags"] = mmap.MAP_SHARED

            mm = mmap.mmap(fh.fileno(), 0, **_mmap_kwargs)
            _madvise(mm, MADV_SEQUENTIAL)

            # Commit resources
            self._fh = fh
            self._mm = mm

            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._prefetch_loop,
                daemon=True,
                name="weight_pipe",
            )
            self._thread.start()
            self._start_ts = time.monotonic()

            log.info("AsyncWeightPipeline: started (regions=%d, depth=%d, release=%d)",
                     len(self._regions), self._depth, self._release)

        except Exception as exc:
            # Cleanup on failure
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            log.error("AsyncWeightPipeline.start() failed: %s", exc)
            raise

    def stop(self) -> None:
        """Stop the prefetch thread and close the mmap."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._thread_join_timeout)
        self._cleanup_resources()
        elapsed = time.monotonic() - self._start_ts if self._start_ts else 0
        log.info("AsyncWeightPipeline: stopped. prefetched=%d, released=%d, uptime=%.1fs",
                 self._prefetch_count, self._release_count, elapsed)

    def _cleanup_resources(self) -> None:
        try:
            if self._mm:
                self._mm.close()
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
        self._mm = None
        self._fh = None

    # -------------------------------------------------------------------------
    # Runtime Control
    # -------------------------------------------------------------------------
    def advance(self, current_layer: int) -> None:
        """
        Signal that computation has moved to layer `current_layer`.
        Triggers prefetch for upcoming layers and release of past layers.
        """
        if not self._enabled or not self._regions:
            return

        with self._lock:
            self._current = current_layer

            # Prefetch upcoming layers
            for d in range(1, self._depth + 1):
                target = current_layer + d
                if 0 <= target < len(self._regions) and target not in self._queue:
                    self._queue.append(target)

            # Release layers far behind
            if self._release > 0:
                behind = current_layer - self._release
                if 0 <= behind < len(self._regions):
                    # Negative tag indicates release
                    release_tag = -(behind + 1)
                    if release_tag not in self._queue:
                        self._queue.append(release_tag)

    # -------------------------------------------------------------------------
    # Background Thread
    # -------------------------------------------------------------------------
    def _prefetch_loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                work = list(self._queue)
                self._queue.clear()

            for item in work:
                if item >= 0:
                    self._do_prefetch(item)
                else:
                    self._do_release(-item - 1)

            if not work:
                self._stop_evt.wait(timeout=self._idle_sleep)

    # -------------------------------------------------------------------------
    # madvise Operations
    # -------------------------------------------------------------------------
    def _do_prefetch(self, layer_idx: int) -> None:
        if not self._mm or layer_idx >= len(self._regions):
            return
        region = self._regions[layer_idx]
        pg_start, pg_len = region.page_aligned()
        if pg_len == 0:
            return
        try:
            _madvise_range_or_obj(self._mm, pg_start, pg_len, MADV_WILLNEED)
            self._prefetch_count += 1
            log.debug("Prefetch layer %d: %d MB", layer_idx, pg_len >> 20)
        except Exception as exc:
            log.debug("Prefetch layer %d failed: %s", layer_idx, exc)

    def _do_release(self, layer_idx: int) -> None:
        if not self._mm or layer_idx >= len(self._regions):
            return
        region = self._regions[layer_idx]
        pg_start, pg_len = region.page_aligned()
        if pg_len == 0:
            return
        try:
            _madvise_range_or_obj(self._mm, pg_start, pg_len, MADV_DONTNEED)
            self._release_count += 1
        except Exception as exc:
            log.debug("Release layer %d failed: %s", layer_idx, exc)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "prefetch_depth": self._depth,
                "release_behind": self._release,
                "current_layer": self._current,
                "prefetch_count": self._prefetch_count,
                "release_count": self._release_count,
                "n_layers": len(self._regions),
                "queue_depth": len(self._queue),
            }


# -----------------------------------------------------------------------------
# madvise Helpers
# -----------------------------------------------------------------------------
def _madvise_range_or_obj(mm_obj, offset: int, length: int, advice: int) -> None:
    """Apply madvise to a sub‑range, trying object method first, then ctypes."""
    if hasattr(mm_obj, "madvise"):
        try:
            mm_obj.madvise(advice, offset, length)
            return
        except TypeError:
            try:
                mm_obj.madvise(advice)
                return
            except Exception:
                pass
    _madvise_range(mm_obj, offset, length, advice)


def _madvise(mm_obj, advice: int) -> None:
    """Apply madvise to the entire mmap."""
    if hasattr(mm_obj, "madvise"):
        try:
            mm_obj.madvise(advice)
            return
        except Exception:
            pass
    _madvise_range(mm_obj, 0, mm_obj.size(), advice)


def _madvise_range(mm_obj, offset: int, length: int, advice: int) -> None:
    """Fallback using ctypes."""
    if os.name != "posix":
        return
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int

        buf = ctypes.c_char.from_buffer(mm_obj)
        base = ctypes.addressof(buf)
        addr = base + offset
        ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice))
        if ret != 0:
            raise OSError(ctypes.get_errno(), "madvise failed")
    except Exception as e:
        log.debug("madvise fallback failed: %s", e)


# -----------------------------------------------------------------------------
# Model‑Level Pipeline Manager (bounded LRU)
# -----------------------------------------------------------------------------
class ModelWeightPipelineManager:
    """
    Manages one AsyncWeightPipeline per loaded model, with bounded LRU eviction.

    Created by ModelLoader after each successful load.
    Destroyed when the model is unloaded.

    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None) -> None:
        self._pipelines: Dict[str, AsyncWeightPipeline] = {}
        self._access_order: List[str] = []   # LRU order (most recent at end)
        self._lock = threading.Lock()
        self._default_config = default_config or {}
        self._max_pipelines = self._default_config.get("max_pipelines", 16)

    def register(
        self,
        model_id: str,
        model_path: str,
        prefetch_depth: Optional[int] = None,
        release_behind: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncWeightPipeline:
        """Create and start a pipeline for a newly loaded model."""
        with self._lock:
            # If already exists, replace it
            old = self._pipelines.get(model_id)
            if old:
                old.stop()
                # Remove from order if present
                if model_id in self._access_order:
                    self._access_order.remove(model_id)

            # Merge provided config with manager defaults
            merged = {**self._default_config}
            if config:
                merged.update(config)

            pipe = AsyncWeightPipeline(
                model_path=model_path,
                prefetch_depth=prefetch_depth,
                release_behind=release_behind,
                config=merged,
            )
            pipe.start()
            self._pipelines[model_id] = pipe
            self._access_order.append(model_id)

            # Enforce max pipelines: evict LRU if needed
            while len(self._pipelines) > self._max_pipelines:
                lru_id = self._access_order.pop(0)
                lru_pipe = self._pipelines.pop(lru_id, None)
                if lru_pipe:
                    lru_pipe.stop()
                    log.info("ModelWeightPipelineManager: evicted LRU pipeline for '%s'", lru_id)

            return pipe

    def get(self, model_id: str) -> Optional[AsyncWeightPipeline]:
        with self._lock:
            pipe = self._pipelines.get(model_id)
            if pipe and model_id in self._access_order:
                # Move to end (most recent)
                self._access_order.remove(model_id)
                self._access_order.append(model_id)
            return pipe

    def unregister(self, model_id: str) -> None:
        with self._lock:
            pipe = self._pipelines.pop(model_id, None)
            if model_id in self._access_order:
                self._access_order.remove(model_id)
        if pipe:
            pipe.stop()

    def advance_all(self, model_id: str, layer_idx: int) -> None:
        pipe = self.get(model_id)
        if pipe:
            pipe.advance(layer_idx)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pipelines": {mid: pipe.stats() for mid, pipe in self._pipelines.items()},
                "max_pipelines": self._max_pipelines,
                "active_count": len(self._pipelines),
            }

    def shutdown(self) -> None:
        with self._lock:
            pipes = list(self._pipelines.values())
            self._pipelines.clear()
            self._access_order.clear()
        for p in pipes:
            try:
                p.stop()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Global Singleton
# -----------------------------------------------------------------------------
_pipeline_manager: Optional[ModelWeightPipelineManager] = None


def get_pipeline_manager(
    default_config: Optional[Dict[str, Any]] = None,
) -> ModelWeightPipelineManager:
    """Return the global ModelWeightPipelineManager singleton."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = ModelWeightPipelineManager(default_config)
    elif default_config is not None:
        # Merge new defaults into existing manager (optional)
        _pipeline_manager._default_config.update(default_config)
    return _pipeline_manager
