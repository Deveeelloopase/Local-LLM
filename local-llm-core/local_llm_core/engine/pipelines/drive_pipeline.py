from __future__ import annotations

"""
engine/pipelines/drive_pipeline.py — Production‑Ready NVMe Drive‑Backed Inference Pipeline
===========================================================================================
– Config injected; no hard‑coded constants, no filesystem reads.

All components accept an optional `config` dict (or a `DrivePipelineConfig`
Pydantic model) at construction time.  Defaults are provided for every
parameter.  This makes the module fully library‑safe and testable in
isolation.

version = 1.0.0
"""

import asyncio
import ctypes
import hashlib
import logging
import mmap
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ─── Global Configuration Constants (Backward Compatibility) ───
# These are re-exported from config_loader.py and used by external
# modules that import from drive_pipeline for convenience. Keep them for
# API stability; the classes below read the real values from `config`.
TARGET_VRAM_PCT = 0.95
TARGET_RAM_PCT = 0.75
TARGET_TPS = 8.0
CPU_THREAD_RATIO = 0.50


# --------------------------------------------------------------------------
# madvise / fadvise helpers (platform‑aware)
# --------------------------------------------------------------------------
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4


def _madvise(mm: mmap.mmap, advice: int) -> bool:
    """Call madvise() with robust fallback to ctypes."""
    if os.name != "posix":
        return False
    if hasattr(mm, "madvise"):
        try:
            mm.madvise(advice)
            return True
        except Exception as exc:
            log.warning("mmap.madvise(%d) failed: %s", advice, exc)
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        buf = ctypes.c_char.from_buffer(mm)
        addr = ctypes.addressof(buf)
        ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(mm.size()), ctypes.c_int(advice))
        if ret != 0:
            log.info("madvise(%d) returned %d errno=%d", advice, ret, ctypes.get_errno())
            return False
        return True
    except Exception as exc:
        log.warning("madvise ctypes fallback failed: %s", exc)
    return False


def advise_model_file(model_path: str, mm: Optional[mmap.mmap] = None) -> None:
    """Apply POSIX_FADV_SEQUENTIAL + WILLNEED hints."""
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        libc.posix_fadvise.restype = ctypes.c_int
        path = Path(model_path)
        size = path.stat().st_size
        with open(model_path, "rb") as fh:
            fd = fh.fileno()
            r1 = libc.posix_fadvise(fd, ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int(2))
            r2 = libc.posix_fadvise(fd, ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int(3))
        if r1 != 0 or r2 != 0:
            log.info("posix_fadvise partial failure: r1=%d r2=%d", r1, r2)
        else:
            log.info("posix_fadvise SEQUENTIAL+WILLNEED: %s (%.1f GB)", path.name, size / 1024**3)
        if mm is not None:
            _madvise(mm, MADV_SEQUENTIAL)
            _madvise(mm, MADV_WILLNEED)
    except Exception as exc:
        log.warning("advise_model_file skipped: %s", exc)


# --------------------------------------------------------------------------
# NVMeConfig — quick-start dataclass that can be built from a config dict
# --------------------------------------------------------------------------
@dataclass
class NVMeConfig:
    """Configuration for NVMe‑backed inference."""
    model_dir: str = "./models"
    swap_dir: str = "/tmp/llm_swap"
    prefetch_depth: int = 3
    layer_cache_bytes: int = 2 * 1024 ** 3
    kv_cache_bytes: int = 256 * 1024 ** 2
    use_madvise: bool = True
    use_direct_io: bool = False
    cpu_thread_limit: int = 0
    target_tps: float = 8.0

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "NVMeConfig":
        """Create NVMeConfig from a drive_pipeline config dict with validation."""
        cfg = config or {}

        model_dir = str(cfg.get("model_dir", "./models"))
        swap_dir = str(cfg.get("swap_dir", "/tmp/llm_swap"))

        prefetch_depth = max(1, int(cfg.get("prefetch_depth", 3)))
        if prefetch_depth > 32:
            log.warning("prefetch_depth=%d capped to 32", prefetch_depth)
            prefetch_depth = 32

        layer_cache_bytes = max(256 * 1024 * 1024, int(cfg.get("layer_cache_bytes", 2 * 1024 ** 3)))
        if layer_cache_bytes > 16 * 1024 ** 3:
            log.warning("layer_cache_bytes=%d capped to 16GB", layer_cache_bytes >> 30)
            layer_cache_bytes = 16 * 1024 ** 3

        kv_cache_bytes = max(64 * 1024 * 1024, int(cfg.get("kv_cache_bytes", 256 * 1024 ** 2)))
        if kv_cache_bytes > layer_cache_bytes:
            log.warning("kv_cache_bytes capped to layer_cache_bytes")
            kv_cache_bytes = layer_cache_bytes

        cpu_thread_limit = max(0, int(cfg.get("cpu_thread_limit", 0)))

        target_tps = max(1.0, float(cfg.get("target_tps", 8.0)))
        if target_tps > 1000.0:
            target_tps = 1000.0

        return cls(
            model_dir=model_dir,
            swap_dir=swap_dir,
            prefetch_depth=prefetch_depth,
            layer_cache_bytes=layer_cache_bytes,
            kv_cache_bytes=kv_cache_bytes,
            use_madvise=bool(cfg.get("use_madvise", True)),
            use_direct_io=bool(cfg.get("use_direct_io", False)),
            cpu_thread_limit=cpu_thread_limit,
            target_tps=target_tps,
        )


# --------------------------------------------------------------------------
# Context Cache (KV prefix) — with efensive state_bytes guard
# --------------------------------------------------------------------------
class ContextCache:
    """LRU KV‑prefix cache for deterministic prompts.

    defense: `put()` refuses entries whose `state_bytes`
    is below `min_state_bytes`. Callers MUST pass the real KV state size
    — typically tens of MB for even a small prefix. Passes below the
    threshold are almost always the response-text size (a caller bug),
    and accepting them makes the LRU cache believe it holds kilobytes
    while it actually holds tens of megabytes → memory bloat / OOM.

    Config keys (under `drive_pipeline.context_cache`):
        max_bytes:       cache size cap (default 256 MiB)
        prefix_len:      prompt-token prefix used for the cache key (512)
        min_state_bytes: reject entries smaller than this (1 MiB)
    """

    def __init__(
        self,
        max_bytes: Optional[int] = None,
        prefix_len: Optional[int] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        cc_cfg = cfg.get("context_cache", {})
        self._max = (
            max_bytes if max_bytes is not None
            else int(cc_cfg.get("max_bytes", 256 * 1024 * 1024))
        )
        self._prefix_len = (
            prefix_len if prefix_len is not None
            else int(cc_cfg.get("prefix_len", 512))
        )
        # reject implausibly small state sizes. A KV state for any
        # real model at any real prefix is > 1 MB.
        self._min_state_bytes = int(
            cc_cfg.get("min_state_bytes", 1024 * 1024)
        )
        self._used = 0
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, prompt: str, model_id: str = "", temperature: float = 0.0) -> str:
        if self._prefix_len > 0 and len(prompt) > self._prefix_len:
            trimmed = prompt[:self._prefix_len]
        else:
            trimmed = prompt
        full_hash = hashlib.sha256(trimmed.encode()).hexdigest()[:16]
        raw = f"{model_id}:{temperature:.2f}:{full_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model_id: str = "", temperature: float = 0.0) -> Optional[Any]:
        key = self._make_key(prompt, model_id, temperature)
        with self._lock:
            if key in self._cache:
                state, size, _ = self._cache[key]
                self._cache.move_to_end(key)
                self._cache[key] = (state, size, time.monotonic())
                self._hits += 1
                return state
            self._misses += 1
            return None

    def put(
            self,
            prompt: str,
            state: Any,
            state_bytes: Optional[int] = None,
            model_id: str = "",
            temperature: float = 0.0,
    ) -> None:
        """Store a KV state for `prompt`.

        rejects `state_bytes < min_state_bytes`. Callers must
        pass the real KV state size. The rejection is logged so the
        caller bug is visible; the caller's subsequent `get()` becomes
        a normal miss (safe default).

        `state_bytes=None` or `<= 0` triggers the previous behavior of
        using `sys.getsizeof(state)`. Note that for a str/int/bytes
        object this returns the *Python object* size, not the KV state
        size — but for a real state object (whatever shape the caller
        passes) it's the best available. Callers that can compute the
        real size should always pass it explicitly.
        """
        key = self._make_key(prompt, model_id, temperature)
        if state_bytes is None or state_bytes <= 0:
            try:
                import sys
                state_bytes = sys.getsizeof(state)
            except Exception:
                state_bytes = 1024

        # reject implausibly-small entries. This catches the common
        # caller bug of passing response-text size instead of KV-state
        # size. See module docstring §2.
        if state_bytes < self._min_state_bytes:
            log.warning(
                "ContextCache.put: rejecting entry for prompt %r "
                "(model_id=%r) — state_bytes=%d is below min_state_bytes=%d. "
                "Caller must pass the real KV state size (typically tens of "
                "MB). Silent under-accounting of KV memory causes LRU bloat "
                "and OOM. This entry will NOT be cached; subsequent get() "
                "returns a miss.",
                prompt[:64], model_id, state_bytes, self._min_state_bytes,
            )
            return

        max_entry = self._max // 4
        if state_bytes > max_entry:
            log.warning("State size %d exceeds max entry %d, skipping", state_bytes, max_entry)
            return
        with self._lock:
            if key in self._cache:
                _, old, _ = self._cache[key]
                self._used -= old
            while self._used + state_bytes > self._max and self._cache:
                _, (_, sz, _) = self._cache.popitem(last=False)
                self._used -= sz
            self._cache[key] = (state, state_bytes, time.monotonic())
            self._cache.move_to_end(key)
            self._used += state_bytes

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "used_mb": self._used >> 20,
                "max_mb": self._max >> 20,
                "hit_rate": round(self._hits / max(total, 1), 3),
                "hits": self._hits,
                "misses": self._misses,
                "prefix_len": self._prefix_len,
                "min_state_bytes": self._min_state_bytes,
            }


# --------------------------------------------------------------------------
# Layer Prefetcher
# --------------------------------------------------------------------------
class LayerPrefetcher:
    """Background NVMe prefetch for upcoming layers."""

    def __init__(
        self,
        model_path: str,
        n_layers: int,
        layer_size_bytes: int,
        depth: Optional[int] = None,
        max_cache_bytes: Optional[int] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        lp_cfg = cfg.get("layer_prefetcher", {})
        self._path = Path(model_path)
        self._n_layers = n_layers
        self._layer_size = layer_size_bytes
        self._depth = depth if depth is not None else lp_cfg.get("prefetch_depth", 3)
        self._max_cache = max_cache_bytes if max_cache_bytes is not None else lp_cfg.get("max_cache_bytes", 2 * 1024 ** 3)
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_used = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=lp_cfg.get("max_workers", 2), thread_name_prefix="nvme_pf")
        self._shutdown = False
        self._current = 0

    def set_current_layer(self, layer: int) -> None:
        self._current = layer
        for i in range(1, self._depth + 1):
            t = layer + i
            if t < self._n_layers:
                self._executor.submit(self._fetch, t)

    def _fetch(self, idx: int) -> None:
        with self._lock:
            if idx in self._cache:
                return
        try:
            with open(self._path, "rb") as f:
                f.seek(idx * self._layer_size)
                data = f.read(self._layer_size)
            with self._lock:
                while self._cache_used + len(data) > self._max_cache and self._cache:
                    _, old = self._cache.popitem(last=False)
                    self._cache_used -= len(old)
                self._cache[idx] = data
                self._cache_used += len(data)
        except Exception as exc:
            log.warning("Prefetch layer %d: %s", idx, exc)

    def get_cached(self, idx: int) -> Optional[bytes]:
        with self._lock:
            return self._cache.get(idx)

    def evict(self, idx: int) -> None:
        with self._lock:
            d = self._cache.pop(idx, None)
            if d:
                self._cache_used -= len(d)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_layers": len(self._cache),
                "cache_used_mb": self._cache_used >> 20,
                "current_layer": self._current,
            }

    def shutdown(self) -> None:
        if self._shutdown:
            return
        log.info("LayerPrefetcher: shutting down executor...")
        self._executor.shutdown(wait=True)
        self._shutdown = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# --------------------------------------------------------------------------
# CPU Thread Throttle
# --------------------------------------------------------------------------
class CPUThrottler:
    """Limits BLAS threads to physical cores × ratio."""

    def __init__(
        self,
        thread_limit: Optional[int] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        phys = self._physical_cores()
        cfg = config or {}
        ct_cfg = cfg.get("cpu_throttler", {})
        ratio = ct_cfg.get("thread_ratio", 0.50)
        if thread_limit is None:
            thread_limit = ct_cfg.get("thread_limit", 0)
        if thread_limit == 0:
            thread_limit = max(1, int(phys * ratio))
        self.thread_limit = thread_limit
        log.info("CPUThrottler: %d physical cores → %d threads", phys, thread_limit)

    @staticmethod
    def _physical_cores() -> int:
        try:
            import psutil
            return psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
        except Exception:
            return os.cpu_count() or 4

    def apply_to_llama_kwargs(self, kwargs: dict) -> dict:
        kwargs["n_threads"] = self.thread_limit
        kwargs["n_threads_batch"] = min(self.thread_limit * 2, os.cpu_count() or 8)
        return kwargs

    def set_openmp_threads(self) -> None:
        t = str(self.thread_limit)
        for var in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[var] = t
        log.info("All BLAS thread env vars → %s", t)


# --------------------------------------------------------------------------
# Concurrency Token Manager
# --------------------------------------------------------------------------
class ConcurrencyTokenManager:
    def __init__(
        self,
        initial: Optional[int] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        conc_cfg = cfg.get("concurrency", {})
        cap = initial if initial is not None else conc_cfg.get("initial_capacity", 1)
        self._capacity = cap
        self._available = cap
        self._lock: Optional[asyncio.Lock] = None
        self._not_empty: Optional[asyncio.Condition] = None
        self._initialized = False
        self._init_in_progress = False

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        while self._init_in_progress:
            await asyncio.sleep(0.001)
        if self._initialized:
            return
        try:
            self._init_in_progress = True
            self._lock = asyncio.Lock()
            self._not_empty = asyncio.Condition(self._lock)
            self._initialized = True
        finally:
            self._init_in_progress = False

    async def acquire(self) -> None:
        await self._ensure_init()
        async with self._not_empty:
            while self._available <= 0:
                await self._not_empty.wait()
            self._available -= 1

    async def release(self) -> None:
        await self._ensure_init()
        async with self._not_empty:
            self._available = min(self._available + 1, self._capacity)
            self._not_empty.notify()

    async def resize(self, new_capacity: int) -> None:
        await self._ensure_init()
        async with self._not_empty:
            delta = new_capacity - self._capacity
            self._capacity = new_capacity
            self._available += delta
            if delta > 0:
                self._not_empty.notify_all()

    @property
    def capacity(self) -> int:
        return self._capacity


# --------------------------------------------------------------------------
# Throughput Controller (AIMD) — prefill-aware TPS
# --------------------------------------------------------------------------
class ThroughputController:
    """AIMD batch‑size controller targeting a TPS goal.

    `record(tokens, prefill_ms=None)` optionally captures
    each request's prefill duration. When EVERY sample in the AIMD window
    carries `prefill_ms`, `current_tps()` returns **decode-only** TPS
    (tokens / (elapsed − sum(prefill))). When any sample lacks prefill
    info, it falls back to **end-to-end** TPS. This prevents AIMD from
    shrinking the batch to `min_batch` on workloads where prefill
    dominates wall time (typical on this hardware: prefill 40–70% of
    total on short responses).
    """

    def __init__(
        self,
        target_tps: Optional[float] = None,
        min_batch: Optional[int] = None,
        max_batch: Optional[int] = None,
        window_sec: Optional[float] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        tc_cfg = cfg.get("throughput_controller", {})
        self.target = target_tps if target_tps is not None else tc_cfg.get("target_tps", 8.0)
        self.min_b = min_batch if min_batch is not None else tc_cfg.get("min_batch", 1)
        self.max_b = max_batch if max_batch is not None else tc_cfg.get("max_batch", 32)
        self.window = window_sec if window_sec is not None else tc_cfg.get("window_sec", 300.0)
        self._batch = self.min_b
        # each sample is (timestamp, tokens, prefill_s_or_None).
        self._tokens: deque = deque()
        self._lock = threading.Lock()
        self._last_adj = time.monotonic()

    def record(self, tokens: int, prefill_ms: Optional[float] = None) -> None:
        """Record one completed request's tokens (and optional prefill time).

        Args:
            tokens: Completion tokens for this request.
            prefill_ms: Optional prefill duration in milliseconds. When
                provided for EVERY sample in the AIMD window,
                `current_tps()` computes decode-only TPS. When any
                sample is missing it, `current_tps()` falls back to
                end-to-end TPS. Passing it is strongly encouraged for
                AIMD accuracy on prefill-dominated workloads.
        """
        now = time.monotonic()
        cutoff = now - self.window
        prefill_s = (float(prefill_ms) / 1000.0) if prefill_ms is not None else None
        with self._lock:
            self._tokens.append((now, int(tokens), prefill_s))
            while self._tokens and self._tokens[0][0] <= cutoff:
                self._tokens.popleft()

    def current_tps(self) -> float:
        """Decode-only TPS if the whole window carries prefill_ms, else E2E."""
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            recent = [(t, n, p) for t, n, p in self._tokens if t > cutoff]
        if len(recent) < 2:
            return 0.0
        elapsed = recent[-1][0] - recent[0][0]
        if elapsed <= 0.001:
            return 0.0
        total_tokens = sum(n for _, n, _ in recent)
        # Decode-only requires ALL samples to carry prefill_ms — a partial
        # window would under-subtract prefill and over-report decode TPS.
        if all(p is not None for _, _, p in recent):
            total_prefill_s = sum(p for _, _, p in recent)
            decode_s = elapsed - total_prefill_s
            if decode_s > 0.05:
                return total_tokens / decode_s
        return total_tokens / elapsed

    def end_to_end_tps(self) -> float:
        """End-to-end TPS (tokens / total elapsed including prefill).

        Same value `current_tps()` returned. Exposed
        separately so callers that specifically want end-to-end TPS (e.g.
        for a user-facing "wall-clock speed" display) can read it without
        ambiguity.
        """
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            recent = [(t, n) for t, n, _ in self._tokens if t > cutoff]
        if len(recent) < 2:
            return 0.0
        elapsed = recent[-1][0] - recent[0][0]
        if elapsed <= 0:
            return 0.0
        return sum(n for _, n in recent) / elapsed

    def suggest_batch_size(self) -> int:
        now = time.monotonic()
        adj_interval = 5.0
        if now - self._last_adj < adj_interval:
            return self._batch
        tps = self.current_tps()
        if tps == 0:
            return self._batch
        with self._lock:
            high_margin = 1.1
            low_margin = 0.85
            if tps > self.target * high_margin:
                self._batch = min(self.max_b, self._batch + 1)
            elif tps < self.target * low_margin:
                self._batch = max(self.min_b, int(self._batch * 0.75))
            self._last_adj = now
            return self._batch

    async def adjust_concurrency(self, token_mgr: ConcurrencyTokenManager) -> None:
        new_cap = self.suggest_batch_size()
        if new_cap != token_mgr.capacity:
            await token_mgr.resize(new_cap)
            log.info(
                "AIMD: concurrency → %d (tps=%.1f target=%.1f)",
                new_cap, self.current_tps(), self.target,
            )

    def stats(self) -> Dict[str, Any]:
        tps = self.current_tps()
        e2e_tps = self.end_to_end_tps()
        return {
            "current_tps": round(tps, 2),
            "end_to_end_tps": round(e2e_tps, 2),
            "target_tps": self.target,
            "batch_size": self._batch,
            "tps_ratio": round(tps / max(self.target, 1), 3),
            "on_target": tps >= self.target * 0.90,
        }


# --------------------------------------------------------------------------
# NVMe Model Advisor
# --------------------------------------------------------------------------
class NVMeModelAdvisor:
    """posix_fadvise hints applied before model load."""

    @staticmethod
    def advise(
        model_path: str,
        ram_free_mb: int,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        adv_cfg = cfg.get("nvme_advisor", {})
        ram_threshold = adv_cfg.get("ram_threshold", 0.70)
        path = Path(model_path)
        if not path.exists():
            return
        size_mb = path.stat().st_size >> 20
        log.info("NVMe advisor: model=%dMB  ram_free=%dMB", size_mb, ram_free_mb)
        if os.name != "posix":
            return
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            libc.posix_fadvise.argtypes = [
                ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int
            ]
            libc.posix_fadvise.restype = ctypes.c_int
            with open(model_path, "rb") as fh:
                fd = fh.fileno()
                size = path.stat().st_size
                libc.posix_fadvise(fd, 0, size, 2)  # SEQUENTIAL always
                # WILLNEED pre-caches the *whole* file into page cache.
                # When any meaningful fraction of the model is offloaded
                # to GPU-addressable memory (VRAM on a dGPU, or GTT on an
                # APU), the page-cache copy is dead weight: the driver
                # holds its own pinned copy. On an APU that dead copy is
                # also competing with the pinned GTT copy for the same
                # single-channel bus. The only case where WILLNEED is a
                # net win is CPU-only inference with ample RAM.
                #
                # Heuristic: only pre-cache when the model is small
                # relative to free RAM. 0.4 is conservative: it leaves
                # 60% of free RAM for KV cache, scores buffer, and the
                # Postgres + OS working set.
                if size_mb <= ram_free_mb * 0.4:
                    libc.posix_fadvise(fd, 0, size, 3)  # WILLNEED
                    log.info(
                        "NVMe: WILLNEED — pre-caching %dMB (fits 40%% of free RAM)",
                        size_mb,
                    )
                else:
                    log.info(
                        "NVMe: SEQUENTIAL only (model=%dMB > 40%% of free RAM=%dMB — "
                        "WILLNEED would duplicate weights the GPU may already hold)",
                        size_mb, ram_free_mb,
                    )
        except Exception as exc:
            log.warning("NVMe advisor skipped: %s", exc)

    @staticmethod
    def set_sysctl_for_nvme(config: Optional[Dict[str, Any]] = None) -> None:
        """host-wide sysctl writes must go through a
        write-ahead journal, not
        through this advisor. Leaving as a no-op preserves the API for
        any caller that still holds a reference.
        """
        log.info(
            "NVMeModelAdvisor.set_sysctl_for_nvme: no-op (sysctl writes "
            "must go through the tuning ledger; host defaults left as-is)"
        )


# --------------------------------------------------------------------------
# Memory Pressure Monitor (swap‑aware) — config-derived thresholds
# --------------------------------------------------------------------------
class MemoryPressureMonitor:
    """Monitors VRAM/RAM/swap and triggers defrag/GC when thresholds exceeded.

    `SWAP_CRITICAL_MB`, `SWAP_CRITICAL_PCT`, and
    `SWAP_LOG_REMIND_S` now read from `memory_pressure.*` config keys.
    Class-level values kept as defaults for backward compat. The
    background `ThreadPoolExecutor` gains a `shutdown()` method so
    `api_server` can close it cleanly.
    """

    # Class-level defaults. Instance attributes (config-derived) shadow
    # these when `__init__` reads the config. External code reading
    # `MemoryPressureMonitor.SWAP_CRITICAL_MB` still sees the class value;
    # instance reads (`mp.SWAP_CRITICAL_MB`) see the config-derived value.
    SWAP_CRITICAL_MB = 4096.0
    SWAP_CRITICAL_PCT = 70.0
    SWAP_LOG_REMIND_S = 300.0

    def __init__(
        self,
        vram_manager,
        ram_manager,
        swap_manager,
        vram_target: Optional[float] = None,
        ram_target: Optional[float] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._vram = vram_manager
        self._ram = ram_manager
        self._swap = swap_manager
        cfg = config or {}
        mp_cfg = cfg.get("memory_pressure", {})
        self._vram_tgt = vram_target if vram_target is not None else mp_cfg.get("vram_target_pct", 0.95)
        self._ram_tgt = ram_target if ram_target is not None else mp_cfg.get("ram_target_pct", 0.75)
        # swap thresholds from config with class-level defaults.
        self.SWAP_CRITICAL_MB = float(
            mp_cfg.get("swap_critical_mb", self.SWAP_CRITICAL_MB)
        )
        self.SWAP_CRITICAL_PCT = float(
            mp_cfg.get("swap_critical_pct", self.SWAP_CRITICAL_PCT)
        )
        self.SWAP_LOG_REMIND_S = float(
            mp_cfg.get("swap_log_remind_s", self.SWAP_LOG_REMIND_S)
        )
        self._lock = threading.Lock()
        self._actions = 0
        self._check_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=mp_cfg.get("max_workers", 1),
            thread_name_prefix="mem_pressure",
        )
        self._swap_was_critical = False
        self._last_swap_log_ts = 0.0

    def _collect_stats(self):
        try:
            import psutil
            ram_pct = psutil.virtual_memory().percent / 100.0
        except Exception:
            ram_pct = 0.0

        try:
            import psutil
            sw = psutil.swap_memory()
            swap_used_mb = sw.used / (1024 * 1024)
            swap_pct = sw.percent
        except Exception:
            swap_used_mb, swap_pct = 0.0, 0.0

        vram_s = self._vram.stats if hasattr(self._vram, "stats") else None
        vram_pct = 0.0

        if vram_s:
            if hasattr(vram_s, "utilization_pct"):
                try:
                    pct_val = float(vram_s.utilization_pct)
                    vram_pct = pct_val / 100.0 if pct_val > 1.0 else pct_val
                except Exception:
                    vram_pct = 0.0
            elif hasattr(vram_s, "total_bytes"):
                total = max(vram_s.total_bytes, 1)
                vram_pct = vram_s.used_bytes / total
        return vram_pct, ram_pct, swap_used_mb, swap_pct

    def _swap_critical(self, swap_used_mb: float, swap_pct: float) -> bool:
        # Both thresholds must be crossed for the warning to fire. The
        # absolute floor handles tiny-swap systems where a low percentage
        # is still a lot of memory; the percentage ceiling handles large
        # (12GB+) swap systems where a stable ~2-3% is expected and not
        # alarming. ANDing them means the warning only appears when the
        # state is genuinely serious by both measures.
        return (
            swap_used_mb > self.SWAP_CRITICAL_MB
            and swap_pct > self.SWAP_CRITICAL_PCT
        )

    def _should_log_swap_state(self, critical: bool) -> bool:
        """True on ok->critical / critical->ok transitions, and at most
        every `SWAP_LOG_REMIND_S` while sustained — never every single poll."""
        now = time.monotonic()
        transitioned = critical != self._swap_was_critical
        due_for_reminder = (
            critical
            and (now - self._last_swap_log_ts) >= self.SWAP_LOG_REMIND_S
        )
        self._swap_was_critical = critical
        if transitioned or due_for_reminder:
            self._last_swap_log_ts = now
            return True
        return False

    def check_and_enforce(self) -> Dict[str, Any]:
        if not self._check_lock.acquire(blocking=False):
            vram_pct, ram_pct, swap_used_mb, swap_pct = self._collect_stats()
            return self._build_result(vram_pct, ram_pct, swap_used_mb, swap_pct)
        try:
            vram_pct, ram_pct, swap_used_mb, swap_pct = self._collect_stats()
            swap_critical = self._swap_critical(swap_used_mb, swap_pct)
            if vram_pct > self._vram_tgt or ram_pct > self._ram_tgt or swap_critical:
                self._executor.submit(self._do_enforcement, vram_pct, ram_pct, swap_used_mb, swap_pct)
            return self._build_result(vram_pct, ram_pct, swap_used_mb, swap_pct)
        finally:
            self._check_lock.release()

    def _do_enforcement(self, vram_pct: float, ram_pct: float,
                         swap_used_mb: float = 0.0, swap_pct: float = 0.0) -> None:
        try:
            if vram_pct > self._vram_tgt:
                freed = self._vram.defragment_gpu_memory()
                if freed:
                    with self._lock:
                        self._actions += 1
                    log.info("MemPressure: defrag freed %d MB", freed >> 20)

            if ram_pct > self._ram_tgt:
                freed = self._ram.pressure_relief()
                if freed:
                    with self._lock:
                        self._actions += 1
                    log.info("MemPressure: GC freed %d MB", freed >> 20)

            critical = self._swap_critical(swap_used_mb, swap_pct)
            if critical and self._should_log_swap_state(True):
                log.warning(
                    "MemPressure: SWAP CRITICAL — %.0fMB (%.1f%%) in active "
                    "use. Generation latency/reliability will degrade.", swap_used_mb, swap_pct
                )
            elif not critical and self._should_log_swap_state(False):
                log.info("MemPressure: swap pressure relieved (%.0fMB, %.1f%%).",
                         swap_used_mb, swap_pct)

        except Exception as e:
            log.error("Enforcement error: %s", e)

    async def check_and_enforce_async(self) -> Dict[str, Any]:
        if not self._check_lock.acquire(blocking=False):
            vram_pct, ram_pct, swap_used_mb, swap_pct = self._collect_stats()
            return self._build_result(vram_pct, ram_pct, swap_used_mb, swap_pct)
        try:
            vram_pct, ram_pct, swap_used_mb, swap_pct = self._collect_stats()
            loop = asyncio.get_running_loop()
            actions = []
            if vram_pct > self._vram_tgt:
                freed = await loop.run_in_executor(self._executor, self._vram.defragment_gpu_memory)
                if freed:
                    actions.append(f"defrag:{freed>>20}MB")
            if ram_pct > self._ram_tgt:
                freed = await loop.run_in_executor(self._executor, self._ram.pressure_relief)
                if freed:
                    actions.append(f"gc:{freed>>20}MB")
            critical = self._swap_critical(swap_used_mb, swap_pct)
            if critical and self._should_log_swap_state(True):
                log.warning(
                    "MemPressure[async]: SWAP CRITICAL — %.0fMB (%.1f%%) in active "
                    "use. Generation latency/reliability will degrade.", swap_used_mb, swap_pct
                )
            elif not critical and self._should_log_swap_state(False):
                log.info("MemPressure[async]: swap pressure relieved (%.0fMB, %.1f%%).",
                         swap_used_mb, swap_pct)
            if actions:
                with self._lock:
                    self._actions += 1
                log.info("MemPressure[%d]: vram=%.1f%% ram=%.1f%% swap=%.0fMB(%.1f%%)  %s",
                         self._actions, vram_pct*100, ram_pct*100, swap_used_mb, swap_pct, " ".join(actions))
            return self._build_result(vram_pct, ram_pct, swap_used_mb, swap_pct)
        finally:
            self._check_lock.release()

    def _build_result(self, vram_pct: float, ram_pct: float,
                       swap_used_mb: float = 0.0, swap_pct: float = 0.0) -> Dict[str, Any]:
        return {
            "vram_pct": round(vram_pct * 100, 1),
            "ram_pct": round(ram_pct * 100, 1),
            "vram_target": self._vram_tgt * 100,
            "ram_target": self._ram_tgt * 100,
            "vram_ok": vram_pct <= self._vram_tgt,
            "ram_ok": ram_pct <= self._ram_tgt,
            "swap_used_mb": round(swap_used_mb, 1),
            "swap_pct": round(swap_pct, 1),
            "swap_ok": not self._swap_critical(swap_used_mb, swap_pct),
            "actions_taken": self._actions,
        }

    def stats(self) -> Dict[str, Any]:
        vram_pct, ram_pct, swap_used_mb, swap_pct = self._collect_stats()
        return self._build_result(vram_pct, ram_pct, swap_used_mb, swap_pct)

    def shutdown(self) -> None:
        """Close the background executor.

        Safe to call multiple times. Called by api_server's shutdown
        loop — before this method existed, the executor
        leaked for the lifetime of the process.
        """
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
