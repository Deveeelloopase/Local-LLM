from __future__ import annotations

"""
engine/inference/throughput_optimizer.py — Production‑Ready Throughput Optimizer
===============================================================================
 Library‑only: configuration injected, no filesystem access.

Orchestrates multiple throughput optimizations:
   1. Speculative decoding (draft‑target, GGUF or BitNet aware).
   2. NVMe‑backed KV cache for prompt prefixes.
   3. Async weight prefetch pipeline.
   4. Priority‑aware slot scheduling — available for external callers
      who want fine‑grained control over slot allocation. The default
      `generate_stream()` path dispatches tokens directly and does NOT
      use the scheduler; callers that need the scheduler must submit
      `SlotRequest` objects themselves.

version = 1.0.0
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

# Pydantic config model (will be imported from central config_loader)
try:
    from local_llm_core.engine.utils.config_loader import (
        SpeculativeConfig,
        ThroughputOptimizerConfig,
        NVMeKVCacheConfig,
    )
except ImportError:
    # Fallback for development – the real library will always have these.
    # @dataclass with default_factory avoids shared mutable
    # class-level defaults; __post_init__ parses nested dicts into
    # sub-configs (matches the real Pydantic model's behaviour).
    @dataclass
    class SpeculativeConfig:
        enabled: bool = True
        draft_model_id: str = ""
        target_model_id: str = ""
        k: int = 8
        top_p: float = 0.9
        min_k: int = 3
        max_k: int = 16
        adaptive_k: bool = True
        draft_timeout_s: float = 30.0
        verify_timeout_s: float = 60.0
        max_fallback_retries: int = 5
        enable_exact_verification: bool = False
        acceptance_target_high: float = 0.75
        acceptance_target_low: float = 0.50

    @dataclass
    class NVMeKVCacheConfig:
        enabled: bool = True
        dir: str = "/tmp/llm_kv_cache"
        max_size_gb: float = 10.0
        prefix_len: int = 512
        max_entries: int = 0

    @dataclass
    class ThroughputOptimizerConfig:
        enabled: bool = True
        max_batch_slots: int = 8
        queue_depth: int = 64
        kv_cache_priority_boost_enabled: bool = True
        kv_cache_priority_boost: int = 10
        speculative: Union[SpeculativeConfig, Dict[str, Any]] = field(
            default_factory=SpeculativeConfig
        )
        kv_cache: Optional[Union[NVMeKVCacheConfig, Dict[str, Any]]] = None

        def __post_init__(self) -> None:
            if isinstance(self.speculative, dict):
                self.speculative = SpeculativeConfig(**self.speculative)
            if isinstance(self.kv_cache, dict):
                self.kv_cache = NVMeKVCacheConfig(**self.kv_cache)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SlotRequest Dataclass
# -----------------------------------------------------------------------------
@dataclass
class SlotRequest:
    """Description of one request in the continuous batching pool."""
    request_id: str
    model_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop: List[str]
    tokens_gen: int = 0
    start_ts: float = field(default_factory=time.perf_counter)
    ttft_ms: float = 0.0
    queue: "asyncio.Queue" = field(default_factory=lambda: asyncio.Queue(maxsize=512))
    done: bool = False
    priority: int = 0
    kv_cache_hit: bool = False


# -----------------------------------------------------------------------------
# Continuous Batch Scheduler
# -----------------------------------------------------------------------------
class ContinuousBatchScheduler:
    """
    Priority‑based continuous batching scheduler.

    Requests are inserted in priority order: `(-effective_priority,
    insertion_seq)`. `effective_priority` = caller's `priority` plus
    `kv_boost_amount` if `kv_cache_hit`. `insertion_seq` is a monotonic
    integer (— previously `start_ts` was used; `time.perf_counter()`
    values can collide on fast machines).

    The scheduler keeps up to `max_slots` active requests and queues the
    rest. When a slot frees, the highest‑priority waiting request fills
    it immediately.

    NOTE: the scheduler is available for external callers who
    want fine‑grained slot control. `ThroughputOptimizer.generate_stream()`
    dispatches one request at a time and does NOT use the scheduler.
    """

    def __init__(
        self,
        max_slots: int,
        queue_depth: int,
        enable_kv_priority_boost: bool = True,
        kv_boost_amount: int = 10,
    ) -> None:
        self._max_slots = max_slots
        self._queue_depth = queue_depth
        self._enable_kv_boost = enable_kv_priority_boost
        self._kv_boost_amount = kv_boost_amount

        self._active: List[SlotRequest] = []
        self._waiting: deque[SlotRequest] = deque()
        self._lock = threading.Lock()
        # deterministic tie-break for equal-priority requests.
        self._insertion_seq: int = 0
        self._req_priority: Dict[str, int] = {}
        self._req_insertion: Dict[str, int] = {}

        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._total_batch_ops = 0
        self._start_ts = time.monotonic()

    def _effective_priority(self, req: SlotRequest) -> int:
        if self._enable_kv_boost and req.kv_cache_hit:
            return req.priority + self._kv_boost_amount
        return req.priority

    def submit(self, req: SlotRequest) -> bool:
        """Add a request. Returns False if the queue is full."""
        with self._lock:
            if len(self._waiting) + len(self._active) >= self._queue_depth:
                return False

            eff_prio = self._effective_priority(req)
            self._insertion_seq += 1
            seq = self._insertion_seq
            self._req_priority[req.request_id] = eff_prio
            self._req_insertion[req.request_id] = seq

            # sort key uses effective priority (with KV boost),
            # not the original priority.
            key = (-eff_prio, seq)
            idx = 0
            for i, r in enumerate(self._waiting):
                r_prio = self._req_priority.get(r.request_id, r.priority)
                r_seq = self._req_insertion.get(r.request_id, 0)
                if (-r_prio, r_seq) > key:
                    break
                idx = i + 1
            self._waiting.insert(idx, req)
            self._total_requests += 1
            self._fill_slots_locked()
            return True

    def _fill_slots_locked(self) -> None:
        """Move waiting requests into active slots while capacity exists."""
        moved = 0
        while len(self._active) < self._max_slots and self._waiting:
            req = self._waiting.popleft()
            self._active.append(req)
            moved += 1
        if moved:
            self._total_batch_ops += 1

    def on_token_generated(self, req: SlotRequest, token: str) -> None:
        """Called when a token is generated for a slot.

        `_total_tokens` increments only when the token is
        actually queued (not after `req.done`).
        """
        if req.done:
            return
        req.tokens_gen += 1
        if req.tokens_gen == 1:
            req.ttft_ms = (time.perf_counter() - req.start_ts) * 1000
        try:
            req.queue.put_nowait(token)
        except asyncio.QueueFull:
            pass  # Slow consumer; drop token to avoid stalling batch
        with self._lock:
            self._total_tokens += 1

    def on_request_done(self, req: SlotRequest) -> None:
        """Called when a request finishes."""
        req.done = True
        try:
            req.queue.put_nowait(None)  # Sentinel to consumer
        except asyncio.QueueFull:
            pass
        with self._lock:
            if req in self._active:
                self._active.remove(req)
            else:
                # previously silently ignored. A request finishing
                # without having been dispatched (or finished twice)
                # indicates a caller bug.
                log.warning(
                    "on_request_done: request %s not in active slots "
                    "(double-close or never-dispatched)",
                    req.request_id[:8],
                )
            self._req_priority.pop(req.request_id, None)
            self._req_insertion.pop(req.request_id, None)
            self._fill_slots_locked()

    def resize(self, new_max_slots: int) -> None:
        """Adjust max concurrent slots (used by AIMD)."""
        with self._lock:
            self._max_slots = max(1, new_max_slots)
            self._fill_slots_locked()

    def active_slots(self) -> List[SlotRequest]:
        """Return a snapshot of currently active slots."""
        with self._lock:
            return list(self._active)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self._start_ts
            return {
                "active_slots": len(self._active),
                "waiting_requests": len(self._waiting),
                "max_slots": self._max_slots,
                "queue_depth": self._queue_depth,
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "server_tps": round(self._total_tokens / max(elapsed, 1e-9), 2),
                "batch_ops": self._total_batch_ops,
                "avg_batch_size": round(
                    self._total_tokens / max(self._total_batch_ops, 1), 2,
                ),
            }


# -----------------------------------------------------------------------------
# Throughput Optimizer (Main Class)
# -----------------------------------------------------------------------------
class ThroughputOptimizer:
    """
    Orchestrates all throughput optimizations.

    Accepts a configuration dict (or `ThroughputOptimizerConfig`) via the
    `config` parameter. All settings are derived from that config; no
    hard‑coded values or external file reads.

    Components (each optional; `enabled=False` disables all):
      • SpeculativeEngine — when `speculative.enabled` and both draft &
        target models are loaded.
      • NVMeKVCache — when `kv_cache.enabled`.
      • AsyncWeightPipeline — global singleton, best effort.
      • ContinuousBatchScheduler — available for external callers;
        `generate_stream()` does NOT use it (single-request dispatch).

    legacy kwargs (`max_batch=`, `spec_k=`, `kv_cache_dir=`,
    `kv_cache_max_gb=`) are accepted for backward compat with the
    api_server callsite. They're merged into `config` if `config` isn't
    provided. New callers should pass `config={...}`.
    """

    def __init__(
        self,
        engine,  # InferenceEngine instance
        *,
        config: Optional[Dict[str, Any]] = None,
        draft_model_id: Optional[str] = None,
        target_model_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        # legacy kwargs for api_server backward compat.
        max_batch: Optional[int] = None,
        spec_k: Optional[int] = None,
        kv_cache_dir: Optional[str] = None,
        kv_cache_max_gb: Optional[float] = None,
    ) -> None:
        # Normalize config; merge legacy kwargs into config
        if config is None and (max_batch is not None or spec_k is not None
                               or kv_cache_dir is not None
                               or kv_cache_max_gb is not None):
            config = {}
            if max_batch is not None:
                config["max_batch_slots"] = int(max_batch)
            if spec_k is not None:
                config.setdefault("speculative", {})["k"] = int(spec_k)
            if kv_cache_dir is not None or kv_cache_max_gb is not None:
                kv_cfg = config.setdefault("kv_cache", {})
                if kv_cache_dir is not None:
                    kv_cfg["dir"] = str(kv_cache_dir)
                if kv_cache_max_gb is not None:
                    kv_cfg["max_size_gb"] = float(kv_cache_max_gb)
        if config is None:
            config = {}

        if isinstance(config, ThroughputOptimizerConfig):
            self._cfg_dict = config.model_dump()
            self._cfg_obj = config
        else:
            self._cfg_dict = config
            try:
                self._cfg_obj = ThroughputOptimizerConfig(**config)
            except Exception as exc:
                log.warning(
                    "Could not validate throughput_optimizer config; using defaults. %s",
                    exc,
                )
                self._cfg_obj = ThroughputOptimizerConfig()

        self._engine = engine
        self._draft_id = draft_model_id or self._cfg_obj.speculative.draft_model_id
        self._target_id = target_model_id or self._cfg_obj.speculative.target_model_id
        self._enabled = enabled if enabled is not None else self._cfg_obj.enabled
        # track background KV save threads
        self._kv_save_threads: Set[threading.Thread] = set()

        if not self._enabled:
            log.info("ThroughputOptimizer disabled by configuration")
            self._scheduler: Optional[ContinuousBatchScheduler] = None
            self._spec_engine: Optional[object] = None
            self._kv_cache: Optional[object] = None
            self._pipe_mgr: Optional[object] = None
            return

        # 1. Continuous Batching Scheduler (available for external callers)
        max_slots = self._cfg_obj.max_batch_slots
        queue_depth = self._cfg_obj.queue_depth
        kv_boost = self._cfg_obj.kv_cache_priority_boost_enabled
        kv_boost_amount = self._cfg_obj.kv_cache_priority_boost
        self._scheduler = ContinuousBatchScheduler(
            max_slots=max_slots,
            queue_depth=queue_depth,
            enable_kv_priority_boost=kv_boost,
            kv_boost_amount=kv_boost_amount,
        )

        # 2. Speculative Decoding
        self._spec_engine = None
        spec_cfg = self._cfg_obj.speculative
        if spec_cfg.enabled and self._draft_id and self._target_id:
            try:
                from local_llm_core.engine.gguf.speculative_engine import SpeculativeEngine
                self._spec_engine = SpeculativeEngine(
                    engine=engine,
                    config=spec_cfg,
                    draft_model_id=self._draft_id,
                    target_model_id=self._target_id,
                )
                log.info(
                    "ThroughputOptimizer: speculative decoding enabled "
                    "(draft=%s, k=%d)",
                    self._draft_id, spec_cfg.k,
                )
            except Exception as exc:
                log.warning("Speculative engine initialization failed: %s", exc)

        # 3. NVMe KV Cache
        kv_cfg = self._cfg_obj.kv_cache
        self._kv_cache = None
        if kv_cfg and kv_cfg.enabled:
            try:
                from local_llm_core.engine import NVMeKVCache
                self._kv_cache = NVMeKVCache(
                    cache_dir=kv_cfg.dir,
                    max_gb=kv_cfg.max_size_gb,
                    prefix_len=kv_cfg.prefix_len,
                    enabled=True,
                )
                log.info("ThroughputOptimizer: NVMe KV cache enabled at %s", kv_cfg.dir)
            except Exception as exc:
                log.warning("NVMe KV cache initialization failed: %s", exc)

        # 4. Async Weight Pipeline (global singleton)
        try:
            from local_llm_core.engine.inference.async_weight_pipe import get_pipeline_manager
            self._pipe_mgr = get_pipeline_manager()
        except Exception as exc:
            log.warning("Async weight pipeline unavailable: %s", exc)
            self._pipe_mgr = None

        log.info(
            "ThroughputOptimizer initialized: max_slots=%d, queue_depth=%d, "
            "spec=%s, kv_cache=%s",
            max_slots, queue_depth,
            "enabled" if self._spec_engine else "disabled",
            "enabled" if self._kv_cache else "disabled",
        )

    # -------------------------------------------------------------------------
    # Public Stream API
    # -------------------------------------------------------------------------
    async def generate_stream(self, request) -> AsyncIterator[str]:
        """Stream tokens for one request.


          • Draft/target lookup now resolves each model INDEPENDENTLY via
            `_resolve_loaded()` — previous code only fell back to loader
            when the DRAFT was missing, so target-only-missing was
            silently skipped.
          • NVMe KV cache restore happens BEFORE generation (previously
            it happened after — no benefit).
        """
        if not self._enabled:
            async for tok in self._engine._stream_tokens(request):
                yield tok
            return

        # NVMe KV cache restore (BEFORE generation; may benefit prefill)
        if self._kv_cache is not None:
            model = self._resolve_loaded(request.model_id)
            if model is not None and hasattr(model, "llm"):
                kv_state = self._kv_cache.get(request.model_id, request.prompt)
                if kv_state is not None:
                    log.debug(
                        "KV cache hit for %s (%.1f KB)",
                        request.model_id, len(kv_state) / 1024,
                    )
                    if hasattr(model.llm, "load_state"):
                        try:
                            model.llm.load_state(kv_state)
                            request.kv_cache_hit = True
                        except Exception as exc:
                            log.debug("KV state restore failed: %s", exc)

        # Try speculative decoding if both models are loaded
        if self._spec_engine is not None:
            draft_loaded = self._resolve_loaded(self._draft_id)
            target_loaded = self._resolve_loaded(self._target_id)
            if draft_loaded is not None and target_loaded is not None:
                async for tok in self._spec_engine.generate_stream(request):
                    yield tok
                self._save_kv_state_background(request)
                return

        # Standard generation
        async for tok in self._engine._stream_tokens(request):
            yield tok
        self._save_kv_state_background(request)

    def _resolve_loaded(self, model_id: str) -> Optional[Any]:
        """Return the loaded model for `model_id`.

        Checks `_unified_models` first, then `loader.get()`. each
        model ID is resolved INDEPENDENTLY — previous code only fell
        back to loader when the draft was missing.
        """
        if not model_id:
            return None
        if hasattr(self._engine, "_unified_models"):
            m = self._engine._unified_models.get(model_id)
            if m is not None:
                return m
        if hasattr(self._engine, "loader"):
            return self._engine.loader.get(model_id)
        return None

    def _save_kv_state_background(self, request) -> None:
        """Spawn a background thread to persist KV state.

        tracked in `_kv_save_threads` set so the thread can't be
        GC'd before running; failures log at WARNING.
        """
        if not self._kv_cache:
            return

        def _save():
            try:
                model = self._resolve_loaded(request.model_id)
                if model and hasattr(model, "llm") and hasattr(model.llm, "save_state"):
                    state_bytes = model.llm.save_state()
                    if state_bytes:
                        self._kv_cache.put(
                            request.model_id, request.prompt, state_bytes,
                        )
            except Exception as exc:
                log.warning(
                    "KV state save failed for %s: %s",
                    request.model_id, exc,
                )
            finally:
                self._kv_save_threads.discard(threading.current_thread())

        t = threading.Thread(target=_save, daemon=True, name="kv_save")
        self._kv_save_threads.add(t)
        t.start()

    # -------------------------------------------------------------------------
    # Scheduler Integration (external callers)
    # -------------------------------------------------------------------------
    def submit_to_scheduler(self, req: SlotRequest) -> bool:
        if self._scheduler is None:
            return False
        return self._scheduler.submit(req)

    def get_active_slots(self) -> List[SlotRequest]:
        if self._scheduler is None:
            return []
        return self._scheduler.active_slots()

    def on_token_generated(self, req: SlotRequest, token: str) -> None:
        if self._scheduler:
            self._scheduler.on_token_generated(req, token)

    def on_request_done(self, req: SlotRequest) -> None:
        if self._scheduler:
            self._scheduler.on_request_done(req)

    # -------------------------------------------------------------------------
    # Statistics (defensive)
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of all sub-component stats.

        each sub-component's `.stats()` is wrapped in try/except so
        one broken component doesn't take down the whole report.
        """
        base: Dict[str, Any] = {"enabled": self._enabled}
        if getattr(self, "_scheduler", None):
            try:
                base["scheduler"] = self._scheduler.stats()
            except Exception as exc:
                base["scheduler"] = {"error": str(exc)}
        if getattr(self, "_kv_cache", None):
            try:
                base["kv_cache"] = self._kv_cache.stats()
            except Exception as exc:
                base["kv_cache"] = {"error": str(exc)}
        if getattr(self, "_pipe_mgr", None):
            try:
                base["pipelines"] = self._pipe_mgr.stats()
            except Exception as exc:
                base["pipelines"] = {"error": str(exc)}
        if getattr(self, "_spec_engine", None):
            try:
                base["speculative"] = self._spec_engine.stats()
            except Exception as exc:
                base["speculative"] = {"error": str(exc)}
        return base
