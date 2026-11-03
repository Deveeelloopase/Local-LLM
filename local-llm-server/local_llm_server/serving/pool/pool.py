"""
pool.py — public façade for the shared-weight process pool.

Exposes the same streaming API the routes already consume from
InferenceEngine.generate_stream(ir), so integration is one line per route.

Design invariants:
  * Workers are single-slot.
  * Requests are routed to an IDLE worker when one exists; otherwise queued
    in a strict global FIFO that any worker pops from when it goes idle.
  * Admission refuses when RAM > refuse_pct.
  * Cancellation propagates to the worker.
  * Aggregate stats match InferenceEngine.stats() key names used by the API.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from .handle import WorkerHandle
from .router import CAASRouter, CAASWeights, IdleCandidate
from .supervisor import WorkerSupervisor

log = logging.getLogger(__name__)


class PoolUnavailable(Exception):
    """Raised when a request cannot be admitted due to pool state (P1-5).

    Distinct from a generic RuntimeError so the API layer can convert
    each case to the correct HTTP status + Retry-After. Codes:

      * ``model_starting``        — no live workers yet, but the pool is
                                    not in a degraded state (workers are
                                    still spawning/warming). Retry in ~5s.
      * ``pool_degraded``         — every tracked worker slot is degraded
                                    (repeated respawn failures). Retry in
                                    ~30s while checking server logs.
      * ``queue_deadline_exceeded`` — the request's own wall-clock budget
                                    for queue waiting expired. Retry-After
                                    is 0: retrying with the same budget
                                    will hit the same wall.

    The api_server.py exception handler for this class (delivered in that
    file's Patch 8) returns HTTP 503 with a ``Retry-After`` header and a
    structured JSON body ``{"error": {"status", "code", "message",
    "retry_after"}}`` so clients can distinguish cases and decide policy
    instead of blanket-retrying.
    """
    def __init__(
        self,
        code: str,
        retry_after_s: float = 5.0,
        message: Optional[str] = None,
    ) -> None:
        self.code = code
        self.retry_after_s = float(retry_after_s)
        super().__init__(message or f"Pool unavailable: {code}")


@dataclass
class _PendingRequest:
    payload: Dict[str, Any]
    request_id: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=4096))
    cancelled: bool = False
    # True once the caller has consumed a terminal event (done /
    # cancelled / error). If False at generator exit, the request was
    # abandoned and needs cleanup (FIFO removal, worker cancel).
    finished: bool = False
    # absolute monotonic ts when queue wait must give up. 0 = no
    # deadline (caller did not pass a timeout_s and the engine_config
    # default is 0).
    deadline_ts: float = 0.0
    # retry counter. Capped at 1 so a single pre-first-token
    # worker failure is retried once on another (or respawned) worker;
    # post-first-token failures never retry (client has partial text).
    retries: int = 0

class SharedWeightProcessPool:
    """
    Public API:
        await pool.start()
        async for token in pool.generate_stream(ir): ...
        await pool.cancel(request_id)
        pool.stats() -> dict
        await pool.shutdown()
    """

    def __init__(
        self,
        *,
        load_spec: Dict[str, Any],
        router_weights: CAASWeights,
        initial_workers: int,
        max_workers: int,
        ram_refuse_pct: float,
        global_queue_max: int,
        heartbeat_s: float,
        heartbeat_timeout_s: float = 15.0,
        respawn_backoff: Optional[List[float]] = None,
        max_respawns_per_5min: int = 3,
        python_exe: Optional[str] = None,
    ) -> None:
        self._spec = load_spec
        self._router = CAASRouter(router_weights)

        # ── Worker count clamp (P0-3) ──────────────────────────────────────
        # The ledger `max_workers` is a HARD ceiling. `initial_workers`
        # is a request that may be lower but never higher. The previous
        # code took max(initial, max), which silently OVERRODE the ledger
        # — the committed log shows `MemoryLedger → max_workers=1`
        # followed by `SharedWeightProcessPool started: workers=2 max=2`.
        self._max = max(1, int(max_workers))
        requested = int(initial_workers)
        if requested <= 0:
            # 0 = auto: use the ceiling. The caller (api_server.py) is
            # responsible for translating "0 = auto" in config.yaml into
            # a positive integer before constructing the pool; when it
            # does, `requested` will be > 0. If it slips through as 0,
            # this fallback uses the ceiling — matches "as many as fit".
            requested = self._max
        self._initial = max(1, min(requested, self._max))
        if requested > self._max:
            log.warning(
                "pool: requested initial_workers=%d clamped to ledger "
                "ceiling %d (config.max_workers or MemoryLedger).",
                requested, self._max,
            )

        self._ram_refuse_pct = float(ram_refuse_pct)
        self._queue_max = int(global_queue_max)

        self._supervisor = WorkerSupervisor(
            load_spec=load_spec,
            python_exe=python_exe,
            heartbeat_s=heartbeat_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
            respawn_backoff=respawn_backoff,
            max_respawns_per_5min=max_respawns_per_5min,
        )
        # P1-5: wake the dispatch loop when a worker becomes idle.
        self._supervisor.on_worker_ready = self._on_worker_ready

        self._fifo: deque[_PendingRequest] = deque()
        self._fifo_lock = asyncio.Lock()
        self._dispatching = asyncio.Event()
        self._dispatch_task: Optional[asyncio.Task] = None
        self._started = False

        # Aggregate telemetry
        self._total_requests = 0
        self._total_tokens = 0
        self._decode_tps_samples: deque = deque(maxlen=500)
        self._ttft_samples: deque = deque(maxlen=500)
        self._queue_depth = 0
        self._rejected_overloaded = 0

        self._prefill_durations_ms: deque = deque(maxlen=500)
        self._prefill_prompt_lens: deque = deque(maxlen=500)

        self._dispatched: Dict[str, _PendingRequest] = {}
        self._dispatched_lock = asyncio.Lock()

    def _on_worker_ready(self, wid: int) -> None:
        """Supervisor callback: a worker just became idle."""
        # Wake the dispatch loop so any queued request fires immediately.
        self._dispatching.set()


    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        # Start the monitor BEFORE spawning, so worker 0 is monitored from
        # the moment it exists. Previously the monitor started after
        # `ensure_min_workers` returned, so worker 0 sat unmonitored for
        # 17.9 s (log: `02:42:12 worker 0 spawned` →
        #      `02:42:30 worker 0 missed heartbeat 17.9s`).
        await self._supervisor.start_monitor()
        await self._supervisor.ensure_min_workers(self._initial)
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="pool_dispatch",
        )
        self._started = True
        log.info(
            "SharedWeightProcessPool started: workers=%d max=%d queue_max=%d",
            self._initial, self._max, self._queue_max,
        )

    async def shutdown(self) -> None:
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        # Cancel any pending FIFO entries
        async with self._fifo_lock:
            for p in self._fifo:
                try:
                    p.queue.put_nowait(("error", "server shutting down"))
                except asyncio.QueueFull:
                    pass
            self._fifo.clear()
            self._queue_depth = 0
        # clear the dispatched map. Any in-flight _run_on_worker
        # tasks will error when the supervisor drains workers; their
        # _final_cleanup calls become no-ops against an empty map.
        async with self._dispatched_lock:
            self._dispatched.clear()
        await self._supervisor.shutdown()

    # ── public API ────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    # Segmented / resumable generation — REQUIRED for /v1/chat/resume
    # to work when the pool is the active token source.
    #
    # The pool workers run InferenceEngine internally, so context-window
    # sliding is already handled there. What the pool must supply is the
    # checkpoint callbacks — periodic saves so a mid-stream crash leaves a
    # resumable checkpoint in conv_store.
    #
    # Contract for checkpoint.accumulated_text (matches _do_persist in
    # api_server.py, which is what /v1/chat/resume reads on the next call):
    #   accumulated_text = GENERATED text only (prompt excluded).
    #   original_prompt  = passed separately by the caller.
    #   resume_prompt    = original_prompt + accumulated_text
    # ══════════════════════════════════════════════════════════════════════

    async def generate_segmented(
        self,
        request: Any,
        checkpoint_callback: Any,
        segment_size: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Pool-mode segmented generation.

        Emits checkpoints every ``segment_size`` tokens and on interruption
        so the parent's /v1/chat/resume can pick up where this left off.
        The actual token production is identical to generate_stream — the
        worker handles context sliding internally.
        """
        from local_llm_core.engine.inference.inference_engine import (
            GenerationCheckpoint,
        )

        seg = int(segment_size or 0) or int(
            self._spec.get("engine_config", {}).get("segment_size", 512) or 512
        )
        full_text = ""
        token_count = 0
        segment_number = 0
        completed = False

        try:
            async for token in self.generate_stream(request):
                full_text += token
                token_count += 1
                yield token
                if token_count > 0 and token_count % seg == 0:
                    segment_number += 1
                    try:
                        await checkpoint_callback(GenerationCheckpoint(
                            request_id=request.request_id,
                            model_id=request.model_id,
                            accumulated_text=full_text,   # generated only
                            tokens_generated=token_count,
                            segment_number=segment_number,
                            status="in_progress",
                        ))
                    except Exception as exc:
                        log.warning(
                            "pool checkpoint save failed (mid-stream, %d tok): %s",
                            token_count, exc,
                        )
            completed = True
        except (asyncio.CancelledError, GeneratorExit):
            # Starlette closes the outer stream via aclose() which
            # raises GeneratorExit — same treatment as CancelledError:
            # save an interrupted checkpoint so /resume works. Not caught
            # by the Exception clause below, which is why this must
            # explicitly list both types.
            try:
                await checkpoint_callback(GenerationCheckpoint(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    accumulated_text=full_text,
                    tokens_generated=token_count,
                    segment_number=segment_number,
                    status="interrupted",
                ))
            except Exception as exc:
                log.warning("pool interruption checkpoint failed: %s", exc)
            raise
        except Exception as exc:
            # Worker died / model error: same — save for resume.
            try:
                await checkpoint_callback(GenerationCheckpoint(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    accumulated_text=full_text,
                    tokens_generated=token_count,
                    segment_number=segment_number,
                    status="interrupted",
                ))
            except Exception as _cpe:
                log.warning("pool error checkpoint failed: %s", _cpe)
            raise
        finally:
            if completed:
                try:
                    await checkpoint_callback(GenerationCheckpoint(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        accumulated_text=full_text,
                        tokens_generated=token_count,
                        segment_number=segment_number + 1,
                        status="complete",
                    ))
                except Exception as exc:
                    log.warning("pool final checkpoint failed: %s", exc)

    async def resume_generation(
        self,
        checkpoint: Any,
        checkpoint_callback: Any,
        extra_tokens: int = 4096,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> AsyncIterator[str]:
        """Resume a previously interrupted generation via the pool.

        ``checkpoint.accumulated_text`` is the fully-templated prompt
        (original template + text generated so far). It becomes the new
        request's prompt — exactly as the engine does it.
        """
        from local_llm_core.engine.inference.inference_engine import (
            InferenceRequest,
        )
        resume_request = InferenceRequest(
            prompt=checkpoint.accumulated_text,
            model_id=checkpoint.model_id,
            max_tokens=extra_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
            request_id=f"{checkpoint.request_id}_resume",
        )
        async for token in self.generate_segmented(
            resume_request, checkpoint_callback,
        ):
            yield token

    # ── Compatibility accessors used by _compute_runtime_limits ────────────
    # These let the parent's existing limits/telemetry code keep working
    # when ctx["token_source"] is the pool instead of the engine.

    @property
    def _cfg(self) -> Dict[str, Any]:
        return self._spec.get("engine_config", {}) or {}

    def _model_n_ctx(self, model_id: str) -> int:
        lk = self._spec.get("load_kwargs", {}) or {}
        return int(lk.get("context_length", 0) or 0)

    def clear_progress(self, root_id: str) -> None:
        # Workers own progress; parent has nothing to clear. Keep the API
        # so api_server._stream_segmented's finally-block call is a no-op
        # instead of an AttributeError.
        return None

    def get_progress(self, root_id: str) -> Dict[str, Any]:
        # Best-effort: report the aggregate queue depth so the heartbeat
        # frame has something meaningful to say. Never raises.
        try:
            return {
                "phase": "working" if self._queue_depth > 0 else "queued",
                "done": 0,
                "total": 0,
                "reused": 0,
            }
        except Exception:
            return {}

    async def generate(self, ir: Any) -> Any:
        """Drop-in replacement for InferenceEngine.generate(ir)."""
        from local_llm_core.engine.inference.inference_engine import InferenceResponse
        start = time.monotonic()
        toks: list[str] = []
        first: float = 0.0
        async for tok in self.generate_stream(ir):
            if not toks:
                first = time.monotonic() - start
            toks.append(tok)
        text = "".join(toks)
        elapsed = time.monotonic() - start
        n = len(toks)
        decode_s = max(0.0, elapsed - first)
        decode_tps = (n - 1) / decode_s if decode_s > 0.05 and n > 1 else 0.0
        return InferenceResponse(
            request_id=ir.request_id,
            text=text,
            tokens_gen=n,
            prompt_tokens=max(1, len(ir.prompt.split())),
            ttft_ms=round(first * 1000.0, 2),
            total_time_s=round(elapsed, 4),
            tps=round(decode_tps, 2),
            finish_reason=(
                "stop" if n < getattr(ir, "effective_max_tokens", 2**31 - 1)
                else "length"
            ),
        )

    # ── Compatibility shims so _compute_runtime_limits works unchanged ──
    def prefill_tps(self) -> float:
        """Estimate prefill throughput from TTFT samples and prompt sizes.

        The pool does not see per-worker prefill events (the worker's
        PrefillTracker lives inside the subprocess), so we approximate
        from what the parent observes:

            t_first_token - t_dispatch ≈ prefill_time
            len(prompt) / prefill_time ≈ tok/s

        The first-token timestamp is recorded by generate_stream() when
        the first chunk arrives; the dispatch timestamp is captured
        inside _PendingRequest.sent_ts. For a multi-worker pool, this
        mixes samples from all workers — the memory_ledger guarantees
        workers see similar hardware, so the aggregate is meaningful.

        Returns 0.0 when no completed requests with measurable prefill
        exist yet, so callers can distinguish "not measured" from
        "measured and slow".
        """
        try:
            # _ttft_samples holds time-to-first-byte in ms, not the
            # prefill-only duration. But the second component — how
            # long the worker actually spent on prefill — is captured by
            # _prefill_durations_ms (populated in _run_on_worker).
            vals = [v for v in self._prefill_durations_ms if v > 0]
            if not vals:
                return 0.0
            avg_ms = sum(vals) / len(vals)
            if avg_ms <= 0:
                return 0.0
            # _prefill_prompt_lens holds the corresponding prompt token
            # counts (approximate, characters / 3.5). Return avg tps.
            lens = [v for v in self._prefill_prompt_lens if v > 0]
            if not lens:
                return 0.0
            avg_len = sum(lens) / len(lens)
            tps = avg_len / (avg_ms / 1000.0)
            return float(tps) if tps > 0 else 0.0
        except Exception:
            return 0.0

    def decode_tps(self) -> float:
        return self.stats().get("decode_tps", 0.0)

    def avg_new_prompt_tokens(self) -> int:
        return 200

    def get_loaded_model_ids(self) -> list:
        """Return the configured model_id, but only when a live worker exists.

        A configured model with zero live workers is NOT loaded. This is the
        truthfulness rule that api_server._get_all_models relies on: a lie
        here means `/v1/models` reports a model as ready while every chat
        request 503s. The previous unconditional return was the last piece
        that let pool-mode "model exists" disagree with "model works".
        """
        if not self._spec:
            return []
        if not self._supervisor.live_handles():
            return []
        return [self._spec.get("model_id", "")]

    async def generate_stream(self, ir: Any) -> AsyncIterator[str]:
        """
        Accepts an InferenceRequest (any object exposing the fields below).
        Yields tokens. On error, raises.

        admission control: refuses immediately when no worker is
        live, with a machine-readable code so the API layer can return a
        structured 503 + Retry-After. Before this, a request could be
        enqueued onto a pool with zero live workers and hang silently
        until the client gave up — the "silent stall" class of bug that
        masked worker-failure loops from the operator.

        queue deadline: bounds the FIFO wait separately from the
        worker-side generation timeouts. Default 300s (configurable via
        ``inference.queue_wait_timeout_s``); also capped by the request's
        own ``timeout_s`` when that is positive. Prevents unbounded
        waiting behind a healthy-but-slow worker chain.

        cancel-on-abandon: if the caller exits the generator before
        a terminal event is consumed (client disconnect, task cancel,
        aclose from Starlette), the ``finally`` marks the request
        cancelled and schedules ``_final_cleanup`` — which removes it
        from FIFO/dispatched and sends an explicit cancel frame to
        workers so the single-slot worker is freed immediately rather
        than continuing to generate for a client that is gone.
        """
        # ── admission control ──────────────────────────────────────
        if not self._supervisor.live_handles():
            # Distinguish "starting up" from "actively degraded". If
            # _handles is empty the pool has not spawned any worker yet
            # (or all were cleared during shutdown) — that is "starting",
            # not "degraded". all_degraded() returns True on empty
            # _handles, which would mislabel this case as pool_degraded.
            _has_any_handles = bool(getattr(self._supervisor, "_handles", {}))
            if _has_any_handles and self._supervisor.all_degraded():
                raise PoolUnavailable(
                    "pool_degraded",
                    retry_after_s=30.0,
                    message=(
                        "All pool workers are degraded (repeated respawn "
                        "failures). Check server logs."
                    ),
                )
            raise PoolUnavailable(
                "model_starting",
                retry_after_s=5.0,
                message="Model is starting; a worker has not yet become ready.",
            )

        import psutil
        if psutil.virtual_memory().percent > self._ram_refuse_pct:
            raise RuntimeError(
                f"server RAM at {psutil.virtual_memory().percent:.0f}% — "
                f"refusing new request to prevent crash"
            )

        # ── queue deadline ────────────────────────────────────────
        _queue_wait_s = float(
            (self._spec.get("engine_config") or {}).get(
                "queue_wait_timeout_s", 300.0
            ) or 300.0
        )
        _req_timeout = float(getattr(ir, "timeout_s", 0.0) or 0.0)
        if _req_timeout > 0:
            _queue_wait_s = min(_queue_wait_s, _req_timeout)
        deadline_ts = (
            (time.monotonic() + _queue_wait_s) if _queue_wait_s > 0 else 0.0
        )

        async with self._fifo_lock:
            if len(self._fifo) >= self._queue_max:
                self._rejected_overloaded += 1
                raise RuntimeError("pool global queue full — server overloaded")
            pending = _PendingRequest(
                payload=self._serialize(ir),
                request_id=ir.request_id,
                deadline_ts=deadline_ts,
            )
            self._fifo.append(pending)
            self._queue_depth = len(self._fifo)
        self._dispatching.set()

        start_ts = time.monotonic()
        first_token_ts: Optional[float] = None
        token_count = 0
        _dispatched = False
        try:
            while True:
                # deadline only applies BEFORE dispatch. Once a
                # worker has taken the request off the FIFO, worker-side
                # stall detection governs liveness — a slow-but-healthy
                # prefill must not be killed here.
                if deadline_ts > 0 and not _dispatched:
                    _remaining = deadline_ts - time.monotonic()
                    if _remaining <= 0:
                        pending.finished = True
                        raise PoolUnavailable(
                            "queue_deadline_exceeded",
                            retry_after_s=0.0,
                            message=(
                                f"Request queue deadline exceeded after "
                                f"{_queue_wait_s:.0f}s (pool is saturated)"
                            ),
                        )
                    try:
                        kind, data = await asyncio.wait_for(
                            pending.queue.get(), timeout=_remaining,
                        )
                    except asyncio.TimeoutError:
                        pending.finished = True
                        raise PoolUnavailable(
                            "queue_deadline_exceeded",
                            retry_after_s=0.0,
                            message=(
                                f"Request queue deadline exceeded after "
                                f"{_queue_wait_s:.0f}s (pool is saturated)"
                            ),
                        )
                else:
                    kind, data = await pending.queue.get()

                if kind == "chunk":
                    if first_token_ts is None:
                        first_token_ts = time.monotonic()
                        self._ttft_samples.append((first_token_ts - start_ts) * 1000.0)
                    token_count += 1
                    self._total_tokens += 1
                    yield data
                elif kind == "done":
                    pending.finished = True
                    return
                elif kind == "cancelled":
                    # Emit the canonical marker so the API layer's
                    # _do_persist path checkpoints rather than saving a
                    # truncated reply as a completed conversation.
                    from local_llm_core.engine.inference.inference_engine import (
                        _make_interruption_marker,
                    )
                    pending.finished = True
                    yield _make_interruption_marker(token_count)
                    return
                elif kind == "error":
                    pending.finished = True
                    raise RuntimeError(data)
                elif kind == "started":
                    # Worker took the request off the FIFO
                    _dispatched = True
                    self._queue_depth = len(self._fifo)
        finally:
            self._total_requests += 1
            if first_token_ts is not None:
                decode_s = time.monotonic() - first_token_ts
                if decode_s > 0.05 and token_count >= 2:
                    self._decode_tps_samples.append((token_count - 1) / decode_s)
            # mark abandoned if no terminal event was consumed,
            # then schedule background cleanup. Fire-and-forget so the
            # generator exits promptly even during shutdown.
            if not pending.finished:
                pending.cancelled = True
            try:
                asyncio.create_task(self._final_cleanup(pending))
            except RuntimeError:
                # No running loop (generator GC during interpreter
                # shutdown). Nothing to clean up at that point anyway.
                pass

    async def cancel(self, request_id: str) -> None:
        async with self._fifo_lock:
            for p in list(self._fifo):
                if p.request_id == request_id:
                    p.cancelled = True
                    self._fifo.remove(p)
                    # Defensive: the pending queue can be full if the
                    # caller abandoned without draining (rare, but the
                    # QueueFull would crash cancel() if we did not guard
                    # it). The delivery of the cancelled event is best-
                    # effort — _final_cleanup runs regardless and the
                    # worker will be cancelled via handle.cancel below.
                    try:
                        p.queue.put_nowait(("cancelled", None))
                    except asyncio.QueueFull:
                        log.debug(
                            "cancel(%s): pending queue full, "
                            "relying on worker cancel frame",
                            request_id[:8],
                        )
                    self._queue_depth = len(self._fifo)
                    return
        # Not in FIFO — either dispatched to a worker or already finished.
        async with self._dispatched_lock:
            hit = self._dispatched.get(request_id)
            if hit is not None:
                hit.cancelled = True
        for h in self._supervisor.live_handles():
            try:
                await h.cancel(request_id)
            except Exception:
                # Writer torn down (worker dying) — the supervisor's
                # kill path handles the process side.
                pass

    def stats(self) -> Dict[str, Any]:
        """Return dict matching InferenceEngine.stats() key names for route compat."""
        decode_avg = (
            sum(self._decode_tps_samples) / len(self._decode_tps_samples)
            if self._decode_tps_samples else 0.0
        )
        ttft_avg = (
            sum(self._ttft_samples) / len(self._ttft_samples)
            if self._ttft_samples else 0.0
        )
        workers = [
            {
                "id": h.worker_id,
                "alive": h.status().alive,
                "busy": h.status().busy,
                "resident_prefix_len": h.status().resident_prefix_len,
            }
            for h in self._supervisor.live_handles()
        ]
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "decode_tps": round(decode_avg, 2),
            "avg_tps": round(decode_avg, 2),  # alias used by _compute_runtime_limits
            "avg_ttft_ms": round(ttft_avg, 2),
            "queue_depth": self._queue_depth,
            "workers": workers,
            "worker_count": len(workers),
            "max_workers": self._max,
            "rejected_overloaded": self._rejected_overloaded,
            "mode": "process_pool",
        }

    # ── dispatch loop ─────────────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                await self._dispatching.wait()
                self._dispatching.clear()
                await self._dispatch_available()
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("pool dispatch error")
                await asyncio.sleep(0.1)

    async def _dispatch_available(self) -> None:
        while True:
            async with self._fifo_lock:
                if not self._fifo:
                    return

                # drop FIFO entries whose deadline has already
                # passed BEFORE we spend time computing a routing
                # decision for them. This is a best-effort sweep (the
                # request's own `generate_stream` wait is still bounded
                # by the same deadline, so the request will raise either
                # way). Doing it here avoids the scenario where an
                # expired request sits at the front of the FIFO, blocks
                # a live worker from being assigned to the next fresh
                # request, and only fails when its caller eventually
                # times out independently.
                _now = time.monotonic()
                _kept: deque = deque()
                _expired_count = 0
                for _p in self._fifo:
                    if _p.deadline_ts > 0 and _p.deadline_ts < _now:
                        _expired_count += 1
                        try:
                            _p.queue.put_nowait((
                                "error",
                                "queue deadline exceeded before dispatch",
                            ))
                        except asyncio.QueueFull:
                            pass
                    else:
                        _kept.append(_p)
                if _expired_count:
                    self._fifo = _kept
                    self._queue_depth = len(self._fifo)
                    log.info(
                        "pool: dropped %d expired request(s) from FIFO",
                        _expired_count,
                    )
                    if not self._fifo:
                        return

                idle = self._idle_handles()
                if not idle:
                    return

    async def _run_on_worker(
            self, handle: WorkerHandle, pending: _PendingRequest,
    ) -> None:
        pending.queue.put_nowait(("started", None))
        _submit_ts = time.monotonic()
        _first_chunk_ts: Optional[float] = None
        # Approximate prompt length for prefill_tps() estimator
        _prompt_chars = len(pending.payload.get("prompt", "") or "")
        _prompt_tokens_approx = max(1, int(_prompt_chars / 3.5))
        _retry_needed = False
        try:
            async for token in handle.submit(pending.payload):
                if _first_chunk_ts is None:
                    _first_chunk_ts = time.monotonic()
                if pending.cancelled:
                    pending.queue.put_nowait(("cancelled", None))
                    return
                pending.queue.put_nowait(("chunk", token))
            if pending.cancelled:
                pending.queue.put_nowait(("cancelled", None))
            else:
                pending.queue.put_nowait(("done", None))
        except asyncio.CancelledError:
            pending.queue.put_nowait(("cancelled", None))
            raise
        except Exception as exc:
            log.warning(
                "worker %d failed request %s: %s",
                handle.worker_id, pending.request_id, exc,
            )
            # P1-5: retry-once when the failure was BEFORE any token was
            # emitted. The request can be safely replayed on another (or
            # respawned) worker because the client has seen nothing yet.
            # Post-first-token failures never retry: the client has
            # already consumed part of the answer and cannot be handed a
            # fresh stream without visible duplication.
            #
            # With _max == 1 (single-worker deployment) this is a no-op:
            # the retry lands back on the same worker and likely fails
            # again on its second attempt (bounded by pending.retries<1).
            # It exists for multi-worker and post-respawn recovery.
            if (
                _first_chunk_ts is None
                and not pending.cancelled
                and pending.retries < 1
            ):
                _retry_needed = True
            else:
                pending.queue.put_nowait(("error", str(exc)))
        finally:
            # record prefill duration for prefill_tps() estimation.
            if _first_chunk_ts is not None:
                _prefill_ms = (_first_chunk_ts - _submit_ts) * 1000.0
                if _prefill_ms > 0:
                    self._prefill_durations_ms.append(_prefill_ms)
                    self._prefill_prompt_lens.append(_prompt_tokens_approx)

        # NOTE: do NOT pop _dispatched here. If this attempt is about
        # to be retried, a concurrent cancel() must still find the
        # request in _dispatched during the retry window (otherwise
        # the cancel would miss both _fifo and _dispatched and the
        # request would run to completion despite the client being
        # gone). _dispatched is cleaned up by generate_stream's
        # finally via _final_cleanup once the caller has exited.
        if _retry_needed:
            pending.retries += 1
            log.info(
                "retrying request %s (attempt %d/2, no tokens emitted)",
                pending.request_id[:8], pending.retries + 1,
            )
            async with self._fifo_lock:
                # Re-check cancelled under lock — the caller may have
                # abandoned between the outer check and here.
                if pending.cancelled:
                    pending.queue.put_nowait(("cancelled", None))
                    return
                self._fifo.appendleft(pending)
                self._queue_depth = len(self._fifo)
            self._dispatching.set()
            return  # dispatch wakeup already done; skip below
        if self._fifo:
            self._dispatching.set()

    async def _final_cleanup(self, pending: _PendingRequest) -> None:
        """tear down a request whose generator has exited.

        Runs as a background task (spawned from ``generate_stream``'s
        ``finally``) so the generator's exit path never blocks on lock
        acquisition or worker I/O. Operations:

          1. Remove from ``_dispatched`` — always, whether the request
             completed normally or was abandoned. On retry attempts the
             entry stayed present so ``cancel()`` could still find it
             during the retry window; only now is it safe to drop.
          2. If the request was abandoned (no terminal event consumed),
             send an explicit ``cancel`` frame to every live worker.
             Handles that were never running this rid ignore the frame
             harmlessly — the worker's ``cancel`` branch only acts when
             ``current_rid == rid``.

        Idempotent by construction: popping a missing key is a no-op,
        and cancel frames for a rid the worker does not track are
        silently ignored on both sides.
        """
        async with self._dispatched_lock:
            self._dispatched.pop(pending.request_id, None)

        if not pending.finished:
            # Abandoned: caller exited before a terminal event arrived.
            for h in self._supervisor.live_handles():
                try:
                    await h.cancel(pending.request_id)
                except Exception:
                    # Writer may have been torn down mid-cleanup; the
                    # worker's own process-tree kill via supervisor will
                    # reap it. Nothing else to do here.
                    pass

    # ── helpers ────────────────────────────────────────────────────────────
    def _idle_handles(self) -> List[WorkerHandle]:
        return [h for h in self._supervisor.live_handles() if h.is_idle()]

    def _peek_token_ids(self, pending: _PendingRequest) -> List[int]:
        """Return [] — the CAAS affinity score is disabled by design.

        The parent process does NOT hold the tokenizer (it is owned by the
        worker, which owns the model). The previous implementation returned
        `list(range(n_approx))` — an integer sequence with no relationship
        to the prompt's real token ids — which made the affinity comparison
        against the worker's real `resident_prefix_head` meaningless. The
        result was not wrong (prefix_gain stayed at 0), but it silently
        wasted the CAAS pass and its log lines read as if affinity was
        working.

        Returning [] is honest: no affinity signal, so the router falls
        through to the staleness tiebreaker (worker-just-used wins ties,
        which reduces cache churn). At 1 worker — which the memory ledger
        clamps to on this host — the choice is moot; at N workers, having
        a real signal requires holding the tokenizer in the parent, which
        is a separate architectural change.
        """
        return []

    def _serialize(self, ir: Any) -> Dict[str, Any]:
        return {
            "request_id": ir.request_id,
            "model_id": ir.model_id,
            "prompt": ir.prompt,
            "max_tokens": int(getattr(ir, "max_tokens", 0) or 0),
            "temperature": float(getattr(ir, "temperature", 0.7)),
            "top_p": float(getattr(ir, "top_p", 0.95)),
            "top_k": int(getattr(ir, "top_k", 40)),
            "stop": list(getattr(ir, "stop", []) or []),
            "timeout_s": float(getattr(ir, "timeout_s", 0.0) or 0.0),
            "root_id": getattr(ir, "root_id", "") or "",
            "grammar": getattr(ir, "grammar", None),
            "response_format": getattr(ir, "response_format", None),
            # propagate strip_think (defaults to True, matching
            # InferenceRequest's default). Worker then honours it.
            "strip_think": bool(getattr(ir, "strip_think", True)),
        }
