from __future__ import annotations

# engine/inference/throughput_optimizer.py — Production‑Ready Throughput Optimizer
# ===============================================================================
# v1.0.0 – Library‑only: configuration injected, no filesystem access.
#
# Orchestrates multiple throughput optimizations:
#   1. Continuous batching with priority scheduling (always active).
#   2. Speculative decoding (draft‑target, GGUF or BitNet aware).
#   3. NVMe‑backed KV cache for prompt prefixes.
#   4. Async weight prefetch pipeline.
#
# All parameters are injected via the `config` argument (a dict or
# ThroughputOptimizerConfig Pydantic model).  No file reads.
# ===============================================================================

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

# Pydantic config model (will be imported from central config_loader)
try:
    from local_llm_core.engine.utils.config_loader import (
        SpeculativeConfig,
        ThroughputOptimizerConfig,
        NVMeKVCacheConfig,
    )
except ImportError:
    # Fallback for development – the real library will always have these
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

    class ThroughputOptimizerConfig:
        enabled: bool = True
        max_batch_slots: int = 8
        queue_depth: int = 64
        kv_cache_priority_boost_enabled: bool = True
        kv_cache_priority_boost: int = 10
        speculative: SpeculativeConfig = SpeculativeConfig()
        kv_cache: NVMeKVCacheConfig = None

    class NVMeKVCacheConfig:
        enabled: bool = True
        dir: str = "/tmp/llm_kv_cache"
        max_size_gb: float = 10.0
        prefix_len: int = 512
        max_entries: int = 0

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SlotRequest Dataclass (mutable – runtime fields are updated by scheduler)
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
    # Internal mutable state (updated externally via scheduler)
    tokens_gen: int = 0
    start_ts: float = field(default_factory=time.perf_counter)
    ttft_ms: float = 0.0
    queue: "asyncio.Queue" = field(default_factory=lambda: asyncio.Queue(maxsize=512))
    done: bool = False
    priority: int = 0
    kv_cache_hit: bool = False


# -----------------------------------------------------------------------------
# Continuous Batch Scheduler (now fully configurable; no globals)
# -----------------------------------------------------------------------------
class ContinuousBatchScheduler:
    """
    Priority‑based continuous batching scheduler.

    Requests are inserted in priority order: ( -priority, arrival_time ).
    The scheduler keeps up to `max_slots` active requests and queues the rest.
    When a slot frees, the highest‑priority waiting request fills it immediately.

    v2.1 – Dynamic slot resizing (AIMD‑style).
    v3.0 – All configuration injected; priority boost amount is now an instance attribute.
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

        # Stats (thread‑safe via lock, but counters are simple)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_batch_ops = 0
        self._start_ts = time.monotonic()

    def submit(self, req: SlotRequest) -> bool:
        """
        Add a request to the scheduler.
        Returns False if the queue is full (backpressure).
        """
        with self._lock:
            if len(self._waiting) + len(self._active) >= self._queue_depth:
                return False

            effective_priority = req.priority
            if self._enable_kv_boost and req.kv_cache_hit:
                effective_priority += self._kv_boost_amount
                log.debug("KV cache hit boost: +%d → %d",
                          self._kv_boost_amount, effective_priority)

            # Insert in priority order (stable by arrival time)
            key = (-effective_priority, req.start_ts)
            idx = 0
            for i, r in enumerate(self._waiting):
                if (-r.priority, r.start_ts) > key:
                    break
                idx = i + 1
            self._waiting.insert(idx, req)
            self._total_requests += 1
            self._fill_slots_locked()
            return True

    def _fill_slots_locked(self) -> None:
        """Move waiting requests into active slots while capacity exists."""
        while len(self._active) < self._max_slots and self._waiting:
            req = self._waiting.popleft()
            self._active.append(req)

    def on_token_generated(self, req: SlotRequest, token: str) -> None:
        """Called by inference engine when a token is generated for a slot."""
        if not req.done:  # defensive
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
                "avg_batch_size": round(self._total_tokens / max(self._total_batch_ops, 1), 2),
            }


# -----------------------------------------------------------------------------
# Throughput Optimizer (Main Class) – config‑injected, robust
# -----------------------------------------------------------------------------
class ThroughputOptimizer:
    """
    Orchestrates all throughput optimizations.

    Accepts a configuration dict (or ThroughputOptimizerConfig Pydantic model)
    via the `config` parameter.  All settings are derived from that config;
    no hard‑coded values or external file reads.

    Components:
    - ContinuousBatchScheduler
    - SpeculativeEngine (from GGUF/BitNet)
    - NVMeKVCache
    - AsyncWeightPipeline (global singleton)
    """

    def __init__(
        self,
        engine,  # InferenceEngine instance
        *,
        config: Optional[Dict[str, Any]] = None,
        draft_model_id: Optional[str] = None,
        target_model_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        # Normalize config: accept dict or pydantic object
        if config is None:
            config = {}
        if isinstance(config, ThroughputOptimizerConfig):
            # Convert pydantic to dict for easy lookup; we'll also use the object directly where needed
            self._cfg_dict = config.model_dump()
            self._cfg_obj = config
        else:
            self._cfg_dict = config
            # Build a ThroughputOptimizerConfig for validation and easy attribute access
            try:
                self._cfg_obj = ThroughputOptimizerConfig(**config)
            except Exception as exc:
                log.warning("Could not validate throughput_optimizer config; using defaults. %s", exc)
                self._cfg_obj = ThroughputOptimizerConfig()

        self._engine = engine
        self._draft_id = draft_model_id or self._cfg_obj.speculative.draft_model_id
        self._target_id = target_model_id or self._cfg_obj.speculative.target_model_id
        self._enabled = enabled if enabled is not None else self._cfg_obj.enabled

        if not self._enabled:
            log.info("ThroughputOptimizer disabled by configuration")
            self._scheduler: Optional[ContinuousBatchScheduler] = None
            self._spec_engine: Optional[object] = None
            self._kv_cache: Optional[object] = None
            self._pipe_mgr: Optional[object] = None
            return

        # -----------------------------------------------------------------
        # 1. Continuous Batching Scheduler
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # 2. Speculative Decoding
        # -----------------------------------------------------------------
        self._spec_engine = None
        spec_cfg = self._cfg_obj.speculative  # Pydantic model
        if spec_cfg.enabled and self._draft_id and self._target_id:
            try:
                from local_llm_core.engine.gguf.speculative_engine import SpeculativeEngine

                # Build SpeculativeEngine with the required config object
                self._spec_engine = SpeculativeEngine(
                    engine=engine,
                    config=spec_cfg,                     # the central SpeculativeConfig
                    draft_model_id=self._draft_id,
                    target_model_id=self._target_id,
                )
                log.info("ThroughputOptimizer: speculative decoding enabled (draft=%s, k=%d)",
                         self._draft_id, spec_cfg.k)
            except Exception as exc:
                log.warning("Speculative engine initialization failed: %s", exc)

        # -----------------------------------------------------------------
        # 3. NVMe KV Cache
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # 4. Async Weight Pipeline (global singleton)
        # -----------------------------------------------------------------
        try:
            from local_llm_core.engine.inference.async_weight_pipe import get_pipeline_manager
            self._pipe_mgr = get_pipeline_manager()
        except Exception as exc:
            log.warning("Async weight pipeline unavailable: %s", exc)
            self._pipe_mgr = None

        log.info(
            "ThroughputOptimizer initialized: max_slots=%d, queue_depth=%d, spec=%s, kv_cache=%s",
            max_slots, queue_depth,
            "enabled" if self._spec_engine else "disabled",
            "enabled" if self._kv_cache else "disabled",
        )

    # -------------------------------------------------------------------------
    # Public Stream API (unchanged logic, but safely uses instance attributes)
    # -------------------------------------------------------------------------
    async def generate_stream(self, request) -> AsyncIterator[str]:
        if not self._enabled:
            async for tok in self._engine._stream_tokens(request):
                yield tok
            return

        # Unified model lookup
        model = None
        if hasattr(self._engine, "_unified_models"):
            model = self._engine._unified_models.get(request.model_id)
        if model is None and hasattr(self._engine, "loader"):
            model = self._engine.loader.get(request.model_id)

        # NVMe KV cache restore
        if self._kv_cache and model is not None and hasattr(model, "llm"):
            kv_state = self._kv_cache.get(request.model_id, request.prompt)
            if kv_state is not None:
                log.debug("KV cache hit for %s (%.1f KB)", request.model_id, len(kv_state) / 1024)
                if hasattr(model.llm, "load_state"):
                    try:
                        model.llm.load_state(kv_state)
                        log.debug("KV state restored from NVMe cache")
                        request.kv_cache_hit = True
                    except Exception as exc:
                        log.debug("KV state restore failed: %s", exc)

        # Speculative decoding if available and models loaded
        if self._spec_engine is not None:
            draft_loaded = target_loaded = None
            if hasattr(self._engine, "_unified_models"):
                draft_loaded = self._engine._unified_models.get(self._draft_id)
                target_loaded = self._engine._unified_models.get(self._target_id)
            if draft_loaded is None and hasattr(self._engine, "loader"):
                draft_loaded = self._engine.loader.get(self._draft_id)
                target_loaded = self._engine.loader.get(self._target_id)

            if draft_loaded and target_loaded:
                async for tok in self._spec_engine.generate_stream(request):
                    yield tok
                self._save_kv_state_background(request)
                return

        # Standard generation
        async for tok in self._engine._stream_tokens(request):
            yield tok
        self._save_kv_state_background(request)

    def _save_kv_state_background(self, request) -> None:
        if not self._kv_cache:
            return

        def _save():
            try:
                model = None
                if hasattr(self._engine, "_unified_models"):
                    model = self._engine._unified_models.get(request.model_id)
                if model is None and hasattr(self._engine, "loader"):
                    model = self._engine.loader.get(request.model_id)
                if model and hasattr(model, "llm") and hasattr(model.llm, "save_state"):
                    state_bytes = model.llm.save_state()
                    if state_bytes:
                        self._kv_cache.put(request.model_id, request.prompt, state_bytes)
            except Exception as exc:
                log.debug("KV state save failed: %s", exc)

        threading.Thread(target=_save, daemon=True, name="kv_save").start()

    # -------------------------------------------------------------------------
    # Scheduler Integration (unchanged)
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
    # Statistics (adds more detail)
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        base = {
            "enabled": self._enabled,
        }
        if self._scheduler:
            base["scheduler"] = self._scheduler.stats()
        if self._kv_cache:
            base["kv_cache"] = self._kv_cache.stats()
        if self._pipe_mgr:
            base["pipelines"] = self._pipe_mgr.stats()
        if self._spec_engine:
            base["speculative"] = self._spec_engine.stats()
        return base
