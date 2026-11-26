from __future__ import annotations

"""
memory/vram_manager.py — Production‑Ready VRAM Manager
======================================================
v1.0.0 – Config injected via constructor; no filesystem reads.

Manages GPU VRAM with:
- LRU cache and pinning
- Fragmentation tracking and background defragmentation
- Multi‑GPU tensor partitioning
- Graceful fallback for CPU‑only environments

All parameters are derived from the `config` dict (or a `VRAMConfig`
Pydantic model). Sensible defaults are used when config is omitted.

version = 1.0.0
"""

import gc
import logging
import threading
import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
@dataclass
class TensorBlock:
    key: str
    size_bytes: int
    data: Any  # torch.Tensor or bytes
    pinned: bool = False
    last_used: float = field(default_factory=time.monotonic)
    access_count: int = 0
    layer_idx: Optional[int] = None


@dataclass
class AllocationStats:
    total_bytes: int = 0
    used_bytes: int = 0
    fragmented_bytes: int = 0
    pinned_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    defrag_runs: int = 0
    defrag_bytes_recovered: int = 0

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.used_bytes

    @property
    def utilization_pct(self) -> float:
        return 100.0 * self.used_bytes / max(self.total_bytes, 1)

    @property
    def fragmentation_ratio(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.fragmented_bytes / self.total_bytes


# -----------------------------------------------------------------------------
# VRAM Manager
# -----------------------------------------------------------------------------
class VRAMManager:
    """
    Manages VRAM allocations with LRU cache, defragmentation, and prefetching.

    All settings are taken from the `config` dict (or `VRAMConfig` model).
    If omitted, the defaults below are used.
    """

    def __init__(
        self,
        total_vram_bytes: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ---- normalise config ------------------------------------------------
        mem_cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import VRAMConfig
                if isinstance(config, dict):
                    cfg_model = VRAMConfig(**config.get("memory", {}))
                else:
                    cfg_model = VRAMConfig(**config.get("memory", {}))
                mem_cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("VRAM config validation failed: %s; using defaults", exc)
                mem_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        self.total_vram_bytes = total_vram_bytes
        total_gb = total_vram_bytes / 1e9

        self.target_utilization = mem_cfg.get("vram_target_utilization", 0.95)
        defrag_threshold = mem_cfg.get("vram_defrag_threshold")
        if defrag_threshold is None:
            defrag_threshold = max(0.10, min(0.30, 20 / max(total_gb, 1)))
        self.defrag_threshold = defrag_threshold

        self._history_window = mem_cfg.get("vram_history_window", 256)
        self._bg_defrag_interval = mem_cfg.get("vram_defrag_check_interval_s", 60.0)
        self._allocation_retries = mem_cfg.get("vram_allocation_retries", 3)

        self._cuda_available = self._detect_cuda()
        self._mps_available = self._detect_mps()

        log.info(
            "VRAMManager: %.1f GB total, target util=%.1f%%, defrag threshold=%.1f%%, "
            "bg interval=%.1fs, cuda=%s, mps=%s",
            total_gb,
            self.target_utilization * 100,
            self.defrag_threshold * 100,
            self._bg_defrag_interval,
            self._cuda_available,
            self._mps_available,
        )

        self._cache: OrderedDict[str, TensorBlock] = OrderedDict()
        self._stats = AllocationStats(total_bytes=total_vram_bytes)
        self._history: deque[str] = deque(maxlen=self._history_window)
        self._lock = threading.RLock()

        # Background defragmentation thread
        self._stop_bg = threading.Event()
        # `defragment_gpu_memory` is called once per request (from
        # `cleanup_after_generation` in api_server) and was logging
        # "No GPU backend available; skipping defrag" at WARNING every
        # time on this CPU/Vulkan deployment. One INFO line at first
        # call is sufficient to tell the operator that VRAM defrag is a
        # no-op here; subsequent calls are silent.
        self._gpu_unavailable_logged = False
        self._bg_thread = threading.Thread(
            target=self._bg_defrag_loop,
            daemon=True,
            name="vram_defrag",
        )
        self._bg_thread.start()

    @staticmethod
    def _detect_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _detect_mps() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False

    # -------------------------------------------------------------------------
    # Public Allocation Interface
    # -------------------------------------------------------------------------
    def allocate(
        self,
        key: str,
        data: Any,
        size_bytes: int,
        pin: bool = False,
        layer_idx: Optional[int] = None,
    ) -> bool:
        with self._lock:
            if key in self._cache:
                block = self._cache[key]
                self._cache.move_to_end(key)
                block.last_used = time.monotonic()
                block.access_count += 1
                self._stats.cache_hits += 1
                return True

            self._stats.cache_misses += 1

            for attempt in range(self._allocation_retries):
                if (
                    self._stats.used_bytes + size_bytes
                    <= self.total_vram_bytes * self.target_utilization
                ):
                    break

                if not self._evict_lru():
                    log.warning(
                        "VRAM full — cannot allocate %d bytes for '%s' (attempt %d)",
                        size_bytes,
                        key,
                        attempt + 1,
                    )
                    if attempt == self._allocation_retries - 1:
                        return False
                    time.sleep(0.01)
                else:
                    continue
            else:
                return False

            block = TensorBlock(
                key=key,
                size_bytes=size_bytes,
                data=data,
                pinned=pin,
                layer_idx=layer_idx,
                last_used=time.monotonic(),
            )
            self._cache[key] = block
            self._stats.used_bytes += size_bytes
            if pin:
                self._stats.pinned_bytes += size_bytes
            self._history.append(key)
            log.info("Allocated '%s' (%d bytes, total used: %d bytes)", key, size_bytes, self._stats.used_bytes)
            return True

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            block = self._cache[key]
            self._cache.move_to_end(key)
            block.last_used = time.monotonic()
            block.access_count += 1
            self._stats.cache_hits += 1
            self._history.append(key)
            return block.data

    def free(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            block = self._cache.pop(key)
            self._stats.used_bytes = max(0, self._stats.used_bytes - block.size_bytes)
            if block.pinned:
                self._stats.pinned_bytes = max(
                    0, self._stats.pinned_bytes - block.size_bytes
                )
            self._release_tensor(block.data)
            log.info("Freed '%s' (%d bytes)", key, block.size_bytes)
            return True

    def pin(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                block = self._cache[key]
                if not block.pinned:
                    block.pinned = True
                    self._stats.pinned_bytes += block.size_bytes
                return True
            return False

    def unpin(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                block = self._cache[key]
                if block.pinned:
                    block.pinned = False
                    self._stats.pinned_bytes = max(
                        0, self._stats.pinned_bytes - block.size_bytes
                    )
                return True
            return False

    # -------------------------------------------------------------------------
    # Defragmentation (Cross‑Platform)
    # -------------------------------------------------------------------------
    def defragment_gpu_memory(self) -> int:
        if not self._cuda_available and not self._mps_available:
            if not self._gpu_unavailable_logged:
                log.info(
                    "VRAM defragmentation is a no-op on this hardware "
                    "(no CUDA / no MPS backend); skipping defrag calls "
                    "silently from now on"
                )
                self._gpu_unavailable_logged = True
            return 0

        backend = "cuda" if self._cuda_available else "mps"
        log.info("Starting VRAM defragmentation on %s backend …", backend.upper())

        before_frag = self._measure_fragmentation()

        with self._lock:
            try:
                import torch

                device_mod = torch.cuda if backend == "cuda" else torch.mps

                evictable = []
                for block in self._cache.values():
                    if block.pinned:
                        continue
                    if not hasattr(block.data, "device"):
                        continue
                    if backend == "cuda" and block.data.device.type != "cuda":
                        continue
                    if backend == "mps" and block.data.device.type != "mps":
                        continue
                    evictable.append((block.key, block.data))

                if not evictable:
                    log.warning("No evictable blocks for defrag")
                    return 0

                cpu_copies = {}
                for key, data in evictable:
                    try:
                        cpu_copies[key] = data.cpu().detach()
                    except Exception as e:
                        log.warning("CPU offload failed for %s: %s", key, e)

                device_mod.empty_cache()

                failed_keys = []
                for key, cpu_data in cpu_copies.items():
                    try:
                        if key in self._cache:
                            if backend == "cuda":
                                self._cache[key].data = cpu_data.cuda()
                            else:
                                self._cache[key].data = cpu_data.to("mps")
                    except RuntimeError as e:
                        log.error("Failed to reload %s to GPU: %s", key, e)
                        failed_keys.append(key)

                device_mod.empty_cache()

                if failed_keys:
                    log.warning(
                        "Defrag: %d tensors could not be moved back to GPU; kept on CPU",
                        len(failed_keys),
                    )

            except Exception as exc:
                log.error("Defragmentation error: %s", exc, exc_info=True)
                return 0

            after_frag = self._measure_fragmentation()
            recovered = max(0, before_frag - after_frag)

            self._stats.defrag_runs += 1
            self._stats.defrag_bytes_recovered += recovered

        log.info("Defragmentation complete: recovered ~%d bytes", recovered)
        return recovered

    def _measure_fragmentation(self) -> int:
        if not self._cuda_available:
            return 0
        try:
            import torch
            mem = torch.cuda.memory_stats()
            return int(mem.get("inactive_split_bytes.all.current", 0))
        except Exception:
            return 0

    def _evict_lru(self) -> bool:
        for key, block in list(self._cache.items()):
            if not block.pinned:
                self._cache.pop(key, None)
                self._stats.used_bytes = max(
                    0, self._stats.used_bytes - block.size_bytes
                )
                self._stats.evictions += 1
                self._release_tensor(block.data)
                log.info("Evicted tensor '%s' (%d bytes)", key, block.size_bytes)
                return True
        return False

    # -------------------------------------------------------------------------
    # Predictive Loading
    # -------------------------------------------------------------------------
    def predict_next_layers(self, current_key: str, top_k: int = 4) -> List[str]:
        if len(self._history) < 2:
            return []
        history = list(self._history)
        bigrams: Dict[str, Dict[str, int]] = {}
        for i in range(len(history) - 1):
            a, b = history[i], history[i + 1]
            bigrams.setdefault(a, {})
            bigrams[a][b] = bigrams[a].get(b, 0) + 1
        candidates = bigrams.get(current_key, {})
        if not candidates:
            freq: Dict[str, int] = {}
            for k in history:
                freq[k] = freq.get(k, 0) + 1
            candidates = freq
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        return [k for k, _ in sorted_cands[:top_k] if k != current_key]

    # -------------------------------------------------------------------------
    # Multi‑GPU Tensor Partitioning
    # -------------------------------------------------------------------------
    def partition_tensor(
        self, tensor: Any, n_devices: int, strategy: str = "even"
    ) -> List[Any]:
        if not self._cuda_available:
            return [tensor]
        try:
            import torch
            if n_devices <= 1:
                return [tensor]
            if strategy == "even":
                return list(torch.chunk(tensor, n_devices, dim=0))
            elif strategy == "weighted":
                weights = self._vram_weights(n_devices)
                sizes = [max(1, int(w * tensor.size(0))) for w in weights]
                sizes[-1] = tensor.size(0) - sum(sizes[:-1])
                return list(torch.split(tensor, sizes, dim=0))
        except ImportError:
            pass
        return [tensor]

    def _vram_weights(self, n: int) -> List[float]:
        try:
            import torch
            frees = []
            for i in range(min(n, torch.cuda.device_count())):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                frees.append(props.total_memory - allocated)
            total = sum(frees) or 1
            return [f / total for f in frees]
        except Exception:
            return [1.0 / n] * n

    # -------------------------------------------------------------------------
    # Background Defragmentation
    # -------------------------------------------------------------------------
    def _bg_defrag_loop(self) -> None:
        interval = self._bg_defrag_interval
        log.info("Background defrag thread started (interval=%.1fs)", interval)
        while not self._stop_bg.wait(interval):
            try:
                if not self._cuda_available:
                    continue
                frag = self._measure_fragmentation()
                if self.total_vram_bytes > 0:
                    frag_ratio = frag / self.total_vram_bytes
                    if frag_ratio > self.defrag_threshold:
                        log.info(
                            "BG defrag triggered (frag=%.1f%%, threshold=%.1f%%)",
                            frag_ratio * 100,
                            self.defrag_threshold * 100,
                        )
                        self.defragment_gpu_memory()
            except Exception as exc:
                log.error("BG defrag loop error: %s", exc, exc_info=True)
        log.info("Background defrag thread stopped")

    def shutdown(self) -> None:
        log.info("Shutting down VRAMManager...")
        self._stop_bg.set()
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=5.0)
            if self._bg_thread.is_alive():
                log.warning(
                    "VRAMManager: background defrag thread did not stop after 5s. "
                    "It may still be running (daemon=True)."
                )
        with self._lock:
            for block in self._cache.values():
                try:
                    self._release_tensor(block.data)
                except Exception as e:
                    log.error("Error releasing tensor during shutdown: %s", e)
            self._cache.clear()
        log.info("VRAMManager shutdown complete")

    @staticmethod
    def _release_tensor(data: Any) -> None:
        try:
            del data
            gc.collect()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Stats: return a snapshot, not the live object
    # -------------------------------------------------------------------------
    @property
    def stats(self) -> AllocationStats:
        """Return a point-in-time snapshot of allocation statistics.

        the previous implementation returned the LIVE
        `_stats` object. Callers that read `vram.stats` — the metrics
        collector's sampling thread, the memory pressure monitor's
        enforcement path, the /v1/metrics route — were reading a
        moving target and could observe a mid-update state, and any
        caller holding a reference could mutate the manager's internal
        accounting by accident.

        This property takes `self._lock` and returns a fresh
        `AllocationStats` with all fields copied. Every consumer gets a
        coherent snapshot; mutating the returned object is safe (it's
        a throwaway copy).

        Computed properties (`free_bytes`, `utilization_pct`,
        `fragmentation_ratio`) work identically on the snapshot because
        they only read from the copied fields.
        """
        with self._lock:
            s = self._stats
            return AllocationStats(
                total_bytes=s.total_bytes,
                used_bytes=s.used_bytes,
                fragmented_bytes=s.fragmented_bytes,
                pinned_bytes=s.pinned_bytes,
                cache_hits=s.cache_hits,
                cache_misses=s.cache_misses,
                evictions=s.evictions,
                defrag_runs=s.defrag_runs,
                defrag_bytes_recovered=s.defrag_bytes_recovered,
            )

    def __repr__(self) -> str:
        # take the lock so the repr never reads a mid-update _stats.
        s = self.stats
        return (
            f"VRAMManager({s.used_bytes/1e9:.2f}/{s.total_bytes/1e9:.2f} GB, "
            f"util={s.utilization_pct:.1f}%, "
            f"hits={s.cache_hits}, evictions={s.evictions})"
        )
    