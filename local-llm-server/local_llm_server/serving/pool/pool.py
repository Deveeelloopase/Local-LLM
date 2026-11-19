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

version = 1.0.0
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from .handle import WorkerHandle
from .router import CAASRouter, CAASWeights, IdleCandidate
from .supervisor import WorkerSupervisor

log = logging.getLogger(__name__)


class PoolUnavailable(Exception):
    """Raised when a request cannot be admitted due to pool state.

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

    The api_server.py exception handler for this class returns HTTP 503 with a ``Retry-After`` header and a
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
    # The worker's authoritative token count for this request, read
    # from `handle.last_token_count` on completion (done / cancelled /
    # error frame). 0 means "not reported" (old worker) — fall back to
    # per-chunk counting. Reset to 0 at submit time.
    token_count: int = 0
    # Worker ID that served this request. Set on dispatch. Used in
    # `stats()['last_request']` and in the `_out` dict returned to the
    # caller (`_stream_segmented`).
    worker_id: int = -1
    # Real prompt token count, computed once on first dispatch
    # via `_count_prompt_tokens()`. Guarded by `if not pending.prompt_tokens`
    # in `_run_on_worker` so retries do not double-count.
    prompt_tokens: int = 0
    # Middleware-assigned request id, distinct from `request_id` (which
    # may carry a `_resume` or `_segN` suffix for internal segments).
    # Used by `get_progress(root_id)` to look up the specific request's
    # FIFO position for the SSE heartbeat status frame.
    root_id: str = ""


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

        # ── Worker count clamp ──────────────────────────────────────
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
        # wake the dispatch loop when a worker becomes idle.
        self._supervisor.on_worker_ready = self._on_worker_ready

        self._fifo: deque[_PendingRequest] = deque()
        self._fifo_lock = asyncio.Lock()
        self._dispatching = asyncio.Event()
        self._dispatch_task: Optional[asyncio.Task] = None
        self._started = False

        # workers that have been handed a pending request but whose
        # `_run_on_worker` task has not yet executed its first step. Between
        # `create_task(...)` and that first step the worker's `is_idle()`
        # still returns True; without this set the next dispatch pass can
        # hand the same worker a second request and hit the worker's
        # "busy" error. Closed under `_fifo_lock`; see `_dispatch_available`
        # and `_idle_handles`.
        self._reserved: set[int] = set()

        # strong references for every `_run_on_worker` task so the
        # CPython GC cannot collect an in-flight coroutine mid-stream.
        # `asyncio.create_task` alone does NOT keep a strong reference.
        self._run_tasks: set[asyncio.Task] = set()

        # Aggregate telemetry
        self._total_requests = 0
        # Legacy per-chunk counter. Kept and reconciled on completion
        # so a mid-stream read still shows progress. See `_record_request_complete`.
        self._total_tokens = 0
        self._decode_tps_samples: deque = deque(maxlen=500)
        self._ttft_samples: deque = deque(maxlen=500)
        self._queue_depth = 0
        self._rejected_overloaded = 0

        # Prefill measurement — real prompt tokens now, not chars/3.5
        self._prefill_durations_ms: deque = deque(maxlen=500)
        self._prefill_prompt_lens: deque = deque(maxlen=500)

        # ── aggregate counters ──────────────────────────────
        # Authoritative totals (used by stats()). Never used as a
        # substitute for `_total_tokens` when a caller wants mid-stream
        # visibility — that's why both exist.
        self._prompt_tokens_total: int = 0
        self._completion_tokens_total: int = 0
        # Bumped every time the fallback (`len(prompt.split())`) is used
        # because the vocab-only tokenizer could not be loaded. Exposed in
        # stats() so the operator knows when prompt_tokens is heuristic.
        self._prompt_tokens_fallback_used: int = 0
        # Last-completed request snapshot for stats()['last_request'].
        self._last_request_snapshot: Optional[Dict[str, Any]] = None

        # ── vocab-only tokenizer (lazy, thread-safe) ────────
        # Loaded from self._spec["model_path"] on first dispatch. Identical
        # approach to api_server._get_or_load_tokenizer — a
        # `Llama(model_path=..., vocab_only=True)` reads just the GGUF
        # header, no weights, no VRAM. ~100ms once per pool lifetime.
        self._vocab_tokenizer = None
        self._vocab_tokenizer_lock = asyncio.Lock()
        self._vocab_tokenizer_loaded = False

        # Per-request total wall-clock duration samples. Feeds the
        # proactive 429 estimate in `generate_stream`: before enqueueing
        # a request, we sum(queue_depth) × mean(this) and compare to the
        # client's timeout. If the estimate exceeds the timeout, we
        # reject immediately rather than making the client wait.
        self._request_durations: deque = deque(maxlen=100)

        # root_id → pending. Populated at enqueue, cleared in
        # `_final_cleanup`. `get_progress(root_id)` reads from this to
        # surface the specific request's queue position to the heartbeat
        # status frame; without it, the web UI's "queued (position N)"
        # cannot distinguish this request from the aggregate FIFO depth.
        self._root_to_request: Dict[str, _PendingRequest] = {}
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
            # cancel any in-flight `_run_on_worker` tasks so their
            # `finally` blocks run and release reservations before we clear
            # shared state.
            for t in list(self._run_tasks):
                t.cancel()
            if self._run_tasks:
                await asyncio.gather(*self._run_tasks, return_exceptions=True)
                self._run_tasks.clear()
            self._reserved.clear()
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

    # ── vocab-only tokenizer (lazy, thread-safe) ────────────────────
    async def _ensure_vocab_tokenizer(self) -> None:
        """Lazy-load a vocab-only tokenizer from self._spec["model_path"].

        Same approach as api_server._get_or_load_tokenizer: a
        `Llama(model_path=..., vocab_only=True)` reads only the GGUF
        header, no weights, no VRAM. Runs once per pool lifetime; on
        failure, sets `_vocab_tokenizer_loaded = True` so we do not
        retry every request (that would be a performance disaster).

        Called from `_run_on_worker` before `handle.submit()`. Async
        because `Llama(...)` blocks ~100ms on first load — off the
        event loop via `to_thread`.
        """
        if self._vocab_tokenizer_loaded:
            return
        async with self._vocab_tokenizer_lock:
            if self._vocab_tokenizer_loaded:
                return
            model_path = (self._spec or {}).get("model_path", "") or ""
            if not model_path:
                log.warning(
                    "pool: no model_path in spec — prompt_tokens will fall "
                    "back to len(prompt.split()). Configure "
                    "models.profiles[*].path to enable real token counts."
                )
                self._vocab_tokenizer_loaded = True
                return
            try:
                exists = await asyncio.to_thread(Path(model_path).is_file)
                if not exists:
                    log.warning(
                        "pool: model_path does not exist: %s — prompt_tokens "
                        "will fall back to len(prompt.split())",
                        model_path,
                    )
                    self._vocab_tokenizer_loaded = True
                    return
                from llama_cpp import Llama  # type: ignore
                llm = await asyncio.to_thread(
                    lambda: Llama(
                        model_path=model_path,
                        vocab_only=True,
                        verbose=False,
                    )
                )
                self._vocab_tokenizer = llm.tokenize
                log.info(
                    "pool: loaded vocab-only tokenizer for %s "
                    "(real prompt token counting enabled)",
                    model_path,
                )
            except Exception as exc:
                log.warning(
                    "pool: vocab-only tokenizer load failed: %s — "
                    "prompt_tokens will fall back to len(prompt.split())",
                    exc,
                )
            finally:
                self._vocab_tokenizer_loaded = True

    def _count_prompt_tokens(self, prompt: str) -> int:
        """real prompt token count via the vocab-only tokenizer.

        Falls back to `len(prompt.split())` only when the tokenizer
        could not be loaded. The fallback increments
        `_prompt_tokens_fallback_used` so `stats()` can surface it.
        """
        if not prompt:
            return 0
        if self._vocab_tokenizer is not None:
            try:
                ids = self._vocab_tokenizer(
                    prompt.encode("utf-8", "surrogatepass")
                )
                return len(ids)
            except Exception as exc:
                log.debug(
                    "pool: tokenizer call raised: %s — using word count", exc,
                )
        self._prompt_tokens_fallback_used += 1
        return max(1, len(prompt.split()))

    # ── stats helpers ───────────────────────────────────────────────
    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Index-safe percentile from a pre-sorted list. Empty → 0.0."""
        if not sorted_data:
            return 0.0
        k = int(len(sorted_data) * pct / 100)
        return float(sorted_data[min(k, len(sorted_data) - 1)])

    def _record_request_complete(
        self,
        pending: "_PendingRequest",
        token_count: int,
        first_token_ts: Optional[float],
    ) -> None:
        """reconcile counters and update last_request snapshot.

        Called from `generate_stream`'s done/cancelled handlers. Also
        safe to call from the finally block for abandoned requests — the
        caller is responsible for not double-counting.

        Reconciling `_total_tokens`:
          During the stream, `_total_tokens` was incremented once per
          chunk (`token_count`). If the worker reported a different
          authoritative count (`pending.token_count`), adjust by the
          delta. This keeps the legacy field consistent with the
          authoritative `_completion_tokens_total` without touching the
          per-chunk mid-stream value.
        """
        effective = pending.token_count if pending.token_count > 0 else token_count
        _delta = effective - token_count
        if _delta != 0:
            self._total_tokens += _delta
        self._completion_tokens_total += effective
        # prompt_tokens was added on dispatch (in `_run_on_worker`), do
        # not add it here.
        # Decode-TPS sample. Uses the effective count so a partially-
        # stripped stream still produces a representative TPS value.
        if first_token_ts is not None and effective >= 2:
            decode_s = time.monotonic() - first_token_ts
            if decode_s > 0.05:
                self._decode_tps_samples.append((effective - 1) / decode_s)
        self._last_request_snapshot = {
            "request_id": pending.request_id,
            "worker_id": int(pending.worker_id),
            "prompt_tokens": int(pending.prompt_tokens),
            "completion_tokens": int(effective),
            "chunks_seen": int(token_count),
            "worker_reported": pending.token_count > 0,
        }

    async def _avg_prompt_tokens_value(self) -> int:
        """ average prompt tokens per completed request."""
        if self._total_requests == 0:
            return 200
        return max(1, int(self._prompt_tokens_total / self._total_requests))

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
        *,
        _out: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Pool-mode segmented generation.

        Emits checkpoints every ``segment_size`` tokens and on interruption
        so the parent's /v1/chat/resume can pick up where this left off.
        The actual token production is identical to generate_stream — the
        worker handles context sliding internally.

        _out` (keyword-only, optional) is forwarded to
        `generate_stream`, so the caller can read the worker's
        authoritative token count after completion:
          `_out["token_count"]`, `_out["worker_id"]`, `_out["prompt_tokens"]`.
        Existing callers that omit `_out` are unaffected.
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
            async for token in self.generate_stream(request, _out=_out):
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
        """Per-request progress, keyed on root_id (§6).

        Returns `{phase, position, done, total, reused}`. `phase` is
        one of:
          * "queued"   — in the FIFO; `position` is 1-based index
          * "working"  — dispatched to a worker or just completed
          * (empty)    — root_id unknown to this pool (e.g. a health
                         probe passing the wrong key)

        The heartbeat status frame emitted by the API layer reads
        `position` to display "queued (position N)" in the web UI and
        TUI. Falls back to `{}` on any error so a heartbeat cannot
        crash because of a pool bug.
        """
        try:
            pending = self._root_to_request.get(root_id)
            if pending is None or pending.finished:
                return {
                    "phase": "working", "position": 0,
                    "done": 0, "total": 0, "reused": 0,
                }
            try:
                pos = list(self._fifo).index(pending)
                return {
                    "phase": "queued", "position": pos + 1,
                    "done": 0, "total": 0, "reused": 0,
                }
            except ValueError:
                # Not in FIFO — dispatched to a worker (or just dispatched).
                return {
                    "phase": "working", "position": 0,
                    "done": 0, "total": 0, "reused": 0,
                }
        except Exception:
            return {}

    async def generate(self, ir: Any) -> Any:
        """Drop-in replacement for InferenceEngine.generate(ir).

        uses the worker's authoritative token
        count and real prompt token count when available, instead of the
        chunk count and `len(prompt.split())`.
        """
        from local_llm_core.engine.inference.inference_engine import InferenceResponse
        start = time.monotonic()
        toks: list[str] = []
        first: float = 0.0
        _out: Dict[str, Any] = {}
        async for tok in self.generate_stream(ir, _out=_out):
            if not toks:
                first = time.monotonic() - start
            toks.append(tok)
        text = "".join(toks)
        elapsed = time.monotonic() - start
        # prefer the worker's authoritative count, fall back to chunk
        # count for old workers.
        n = int(_out.get("token_count") or 0) or len(toks)
        # real prompt token count from the tokenizer.
        prompt_tokens = int(_out.get("prompt_tokens") or 0) \
                        or self._count_prompt_tokens(str(getattr(ir, "prompt", "") or ""))
        decode_s = max(0.0, elapsed - first)
        decode_tps = (n - 1) / decode_s if decode_s > 0.05 and n > 1 else 0.0
        return InferenceResponse(
            request_id=ir.request_id,
            text=text,
            tokens_gen=n,
            prompt_tokens=prompt_tokens,
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
        """prefill throughput from real prompt token counts.

        Previously the numerator was `max(1, int(_prompt_chars / 3.5))` —
        a character/3.5 word heuristic, not a real token count. It is now
        `pending.prompt_tokens`, the real vocab-tokenizer count from
        Denominator (`_prefill_durations_ms`) is unchanged: the
        measured submit-to-first-chunk delta from `_run_on_worker`.

        Returns 0.0 when no completed requests with measurable prefill
        exist yet, so callers can distinguish "not measured" from
        "measured and slow".
        """
        try:
            vals = [v for v in self._prefill_durations_ms if v > 0]
            if not vals:
                return 0.0
            avg_ms = sum(vals) / len(vals)
            if avg_ms <= 0:
                return 0.0
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
        """average prompt tokens per request (real data).

        Previously returned the hardcoded 200 — a value that misled the
        client-side latency estimator into assuming a fixed
        prompt size. Now returns `_prompt_tokens_total / _total_requests`
        when available. Note this is *total* prompt tokens, not just
        cache-miss tokens; the pool has no visibility into cache hits
        (they happen inside the worker's llama.cpp context). Use as an
        upper bound on the real prompt size.

        Returns 200 only when zero requests have completed — matching the
        InferenceEngine fallback so callers don't need to branch.
        """
        if self._total_requests == 0:
            return 200
        try:
            return max(1, int(self._prompt_tokens_total / self._total_requests))
        except Exception:
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

    async def generate_stream(
        self,
        ir: Any,
        *,
        _out: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
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

        `_out` is a keyword-only optional dict. When
        provided and the request completes normally (done or cancelled),
        it is filled with `{token_count, worker_id, prompt_tokens,
        chunks_seen}`. This is how the caller obtains the worker's authoritative token
        count without having to count chunks itself.
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
            raise PoolUnavailable(
                "memory_pressure",
                retry_after_s=15.0,
                message=(
                    f"server RAM at {psutil.virtual_memory().percent:.0f}% — "
                    f"refusing new request to prevent crash"
                ),
            )

        # ── queue deadline ────────────────────────────────────────
        _queue_wait_s = float(
            (self._spec.get("engine_config") or {}).get(
                "queue_wait_timeout_s", 300.0
            ) or 300.0
        )
        _req_timeout = float(getattr(ir, "timeout_s", 0.0) or 0.0)
        if _req_timeout > 0:
            # ── Proactive capacity rejection (429) ────────────────────
            # If the client set a timeout AND the queue is deep enough that
            # (queue_depth × mean_request_time) exceeds it, reject now with
            # 429 + Retry-After. The alternative — enqueue, wait, then time
            # out at the deadline — wastes a FIFO slot for the full timeout
            # and delivers a slower failure with no better information.
            #
            # The estimate is conservative: mean over the last 100 successful
            # requests, multiplied by queue depth. First-ever request (empty
            # `_request_durations`) skips this check — no data to estimate
            # from, so let the existing `queue_deadline_exceeded` path
            # govern.
            if _req_timeout > 0 and len(self._fifo) > 0 and self._request_durations:
                try:
                    _avg_s = sum(self._request_durations) / len(self._request_durations)
                except Exception:
                    _avg_s = 0.0
                if _avg_s > 0.0:
                    _est_wait_s = len(self._fifo) * _avg_s
                    if _est_wait_s > _req_timeout:
                        self._rejected_overloaded += 1
                        _retry_s = float(min(_req_timeout, 30.0))
                        log.info(
                            "pool: proactive rejection — est_wait=%.0fs > "
                            "client_timeout=%.0fs (queue=%d, avg_req=%.1fs)",
                            _est_wait_s, _req_timeout,
                            len(self._fifo), _avg_s,
                        )
                        raise PoolUnavailable(
                            "queue_pressure",
                            retry_after_s=_retry_s,
                            message=(
                                f"Queue wait estimate {_est_wait_s:.0f}s exceeds "
                                f"client timeout {_req_timeout:.0f}s "
                                f"({len(self._fifo)} request(s) ahead)"
                            ),
                        )
            _queue_wait_s = min(_queue_wait_s, _req_timeout)
        deadline_ts = (
            (time.monotonic() + _queue_wait_s) if _queue_wait_s > 0 else 0.0
        )

        async with self._fifo_lock:
            if len(self._fifo) >= self._queue_max:
                self._rejected_overloaded += 1
                raise PoolUnavailable(
                    "queue_full",
                    retry_after_s=10.0,
                    message="pool global queue full — server overloaded",
                )
            pending = _PendingRequest(
                payload=self._serialize(ir),
                request_id=ir.request_id,
                # root_id matches what the API layer passes to
                # `_hb_status_fn` — either the explicit root_id from the
                # caller or the request_id as fallback.
                root_id=(getattr(ir, "root_id", "") or ir.request_id),
                deadline_ts=deadline_ts,
            )
            self._fifo.append(pending)
            self._queue_depth = len(self._fifo)
            self._root_to_request[pending.root_id] = pending
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
                    # Legacy per-chunk increment. Reconciled to the worker's
                    # authoritative count on `done` (see
                    # `_record_request_complete`). Kept here so mid-stream
                    # stats show progress.
                    self._total_tokens += 1
                    yield data
                elif kind == "done":
                    pending.finished = True
                    # record the authoritative completion.
                    self._record_request_complete(
                        pending, token_count, first_token_ts,
                    )
                    # expose the count to the caller.
                    if _out is not None:
                        _out["token_count"] = int(
                            pending.token_count or token_count
                        )
                        _out["worker_id"] = int(pending.worker_id)
                        _out["prompt_tokens"] = int(pending.prompt_tokens)
                        _out["chunks_seen"] = int(token_count)
                    return
                elif kind == "cancelled":
                    # Emit the canonical marker so the API layer's
                    # _do_persist path checkpoints rather than saving a
                    # truncated reply as a completed conversation.
                    from local_llm_core.engine.inference.inference_engine import (
                        _make_interruption_marker,
                    )
                    pending.finished = True
                    # same reconciliation as `done`.
                    self._record_request_complete(
                        pending, token_count, first_token_ts,
                    )
                    if _out is not None:
                        _out["token_count"] = int(
                            pending.token_count or token_count
                        )
                        _out["worker_id"] = int(pending.worker_id)
                        _out["prompt_tokens"] = int(pending.prompt_tokens)
                        _out["chunks_seen"] = int(token_count)
                        _out["cancelled"] = True
                    yield _make_interruption_marker(
                        pending.token_count or token_count
                    )
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
            # Record the wall-clock duration for the 429 estimate. Only
            # complete-successful requests count — a failure at t=0 would
            # pull the mean toward zero and make the estimate always
            # pass, defeating the purpose.
            _dur_s = time.monotonic() - start_ts
            if _dur_s > 0.5 and token_count > 0:
                self._request_durations.append(_dur_s)
            # only record aggregate stats if the request
            # completed normally (done or cancelled) — those branches
            # already called `_record_request_complete`. For abandoned
            # requests (client disconnect) the finally runs while
            # pending.token_count is still 0 (worker never reported);
            # record the chunk count as best-effort so the aggregate
            # totals aren't silently under-counted.
            if not pending.finished and first_token_ts is not None:
                _decode_s = time.monotonic() - first_token_ts
                if _decode_s > 0.05 and token_count >= 2:
                    self._decode_tps_samples.append(
                        (token_count - 1) / _decode_s
                    )
                self._completion_tokens_total += token_count
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

    def preflight(self, ir: Any) -> None:
        """Synchronous admission check for HTTP routes (never blocks, never
        mutates pool state).

        Call it BEFORE returning a ``StreamingResponse``. Once the first body
        byte is sent the status line is already ``200``, so a PoolUnavailable
        raised later by ``generate_stream`` can only be reported inside the
        body. Raising here lets the app's PoolUnavailable handler answer with a
        real 503/429 + ``Retry-After``.
        """
        if not self._supervisor.live_handles():
            _has_any = bool(getattr(self._supervisor, "_handles", {}))
            if _has_any and self._supervisor.all_degraded():
                raise PoolUnavailable(
                    "pool_degraded", retry_after_s=30.0,
                    message="All pool workers are degraded. Check server logs.",
                )
            raise PoolUnavailable(
                "model_starting", retry_after_s=5.0,
                message="Model is starting; a worker has not yet become ready.",
            )
        if len(self._fifo) >= self._queue_max:
            raise PoolUnavailable(
                "queue_full", retry_after_s=10.0,
                message="pool global queue full — server overloaded",
            )
        _req_timeout = float(getattr(ir, "timeout_s", 0.0) or 0.0)
        if _req_timeout > 0 and self._fifo and self._request_durations:
            _avg_s = sum(self._request_durations) / len(self._request_durations)
            _est_wait_s = len(self._fifo) * _avg_s
            if _avg_s > 0.0 and _est_wait_s > _req_timeout:
                raise PoolUnavailable(
                    "queue_pressure",
                    retry_after_s=float(min(_req_timeout, 30.0)),
                    message=(
                        f"Queue wait estimate {_est_wait_s:.0f}s exceeds "
                        f"client timeout {_req_timeout:.0f}s "
                        f"({len(self._fifo)} request(s) ahead)"
                    ),
                )

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
        """Return dict matching InferenceEngine.stats() key names for route compat.

        enriched shape. All *existing* keys are preserved
        with the same names/semantics; the new keys are additive.

        New keys:
          decode_tps_p50 / decode_tps_p95
              Percentiles from `_decode_tps_samples` (last 500 requests).
          ttft_ms_p50 / ttft_ms_p95
              Percentiles from `_ttft_samples`.
          prompt_tokens_total
              Authoritative sum of real prompt tokens. Only
              complete when `_prompt_tokens_fallback_used == 0`.
          completion_tokens_total
              Authoritative sum of worker-reported completion tokens
              (reconciled on done/cancelled).
          prompt_tokens_fallback_used
              Number of requests where `len(prompt.split())` was used
              instead of the vocab tokenizer. Non-zero means prompt-token
              stats are heuristic.
          avg_new_prompt_tokens
              Same value as `avg_new_prompt_tokens()` method.
          last_request
              Snapshot of the last completed request:
              `{request_id, worker_id, prompt_tokens, completion_tokens,
                chunks_seen, worker_reported}`. `worker_reported=False`
              means the worker didn't send the `tokens` field — the
              completion_tokens value is the chunk count.

        `workers[]` is now `supervisor.worker_stats_snapshot()` — richer
        shape, includes `load_info`, `degraded`,
        `heartbeat_age_s`, `last_token_count`, `death_reason`. The old
        4-field shape is a strict subset, so any consumer reading
        `workers[*].id` still works after adding a top-level `id` alias.
        """
        decode_avg = (
            sum(self._decode_tps_samples) / len(self._decode_tps_samples)
            if self._decode_tps_samples else 0.0
        )
        ttft_avg = (
            sum(self._ttft_samples) / len(self._ttft_samples)
            if self._ttft_samples else 0.0
        )
        # percentiles. Sorting the deques is O(n log n), n<=500,
        # called by monitoring every few seconds — negligible.
        _decode_sorted = sorted(self._decode_tps_samples)
        _ttft_sorted = sorted(self._ttft_samples)
        decode_p50 = self._percentile(_decode_sorted, 50)
        decode_p95 = self._percentile(_decode_sorted, 95)
        ttft_p50 = self._percentile(_ttft_sorted, 50)
        ttft_p95 = self._percentile(_ttft_sorted, 95)
        # per-worker snapshot from the supervisor.
        try:
            worker_snapshots = self._supervisor.worker_stats_snapshot()
        except Exception:
            worker_snapshots = []
        # Add `id` alias for backward compat with the old 4-field worker
        # dict, whose key was `id` not `worker_id`.
        workers_public: List[Dict[str, Any]] = []
        for w in worker_snapshots:
            w = dict(w)   # local copy so we don't mutate the snapshot
            w.setdefault("id", w.get("worker_id", -1))
            workers_public.append(w)
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "decode_tps": round(decode_avg, 2),
            "avg_tps": round(decode_avg, 2),  # alias used by _compute_runtime_limits
            "avg_ttft_ms": round(ttft_avg, 2),
            "queue_depth": self._queue_depth,
            "workers": workers_public,
            "worker_count": sum(1 for w in workers_public if w.get("alive")),
            "max_workers": self._max,
            "rejected_overloaded": self._rejected_overloaded,
            "mode": "process_pool",
            "decode_tps_p50": round(decode_p50, 2),
            "decode_tps_p95": round(decode_p95, 2),
            "ttft_ms_p50": round(ttft_p50, 2),
            "ttft_ms_p95": round(ttft_p95, 2),
            "prompt_tokens_total": int(self._prompt_tokens_total),
            "completion_tokens_total": int(self._completion_tokens_total),
            "prompt_tokens_fallback_used": int(self._prompt_tokens_fallback_used),
            "avg_new_prompt_tokens": self.avg_new_prompt_tokens(),
            "last_request": (
                dict(self._last_request_snapshot)
                if self._last_request_snapshot else None
            ),
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
        """
        Drain the FIFO until no idle worker or no queued request remains.

        Invariants (this method is the ONLY place FIFO entries transition
        to dispatched):

          * Every loop iteration MUST either make progress (pop a request,
            dispatch a task) or return. No spin, ever.
          * Every iteration that makes progress MUST `await asyncio.sleep(0)`
            after releasing `_fifo_lock` so the event loop can service the
            systemd watchdog beat, uvicorn sockets and monitor pollers.
            Without this, a queued request with an idle worker hangs the
            loop and the watchdog kills the service 60 s later.
          * A worker can only be dispatched once per idle-cycle. Between
            `create_task(...)` and the task's first step, `_idle_handles()`
            would report the worker still idle; `_reserved` closes that
            window.
        """
        while True:
            pending: Optional[_PendingRequest] = None
            target: Optional[WorkerHandle] = None

            async with self._fifo_lock:
                if not self._fifo:
                    return

                # Drop FIFO entries whose deadline has already passed before
                # spending routing work on them. Best-effort sweep — the
                # request's own `generate_stream` wait is still bounded by
                # the same deadline.
                _now = time.monotonic()
                _kept: deque = deque()
                _expired_count = 0
                for _p in self._fifo:
                    if _p.cancelled or (_p.deadline_ts > 0 and _p.deadline_ts < _now):
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

                head = self._fifo[0]
                candidates = []
                for h in idle:
                    st = h.status()
                    candidates.append(IdleCandidate(
                        worker_id=h.worker_id,
                        resident_prefix_head=st.resident_prefix_head,
                        last_ready_ts=st.last_ready_ts,
                    ))
                pick = self._router.pick(
                    request_token_ids=self._peek_token_ids(head),
                    candidates=candidates,
                )
                if pick is None:
                    # No routable worker this pass. Return without consuming
                    # the FIFO head; the next dispatch wake (triggered by
                    # _dispatching.set() from _on_worker_ready or the
                    # dispatch loop's other callers) will retry with fresh
                    # state.
                    return

                target = next(
                    (h for h in idle if h.worker_id == pick.worker_id),
                    None,
                )
                if target is None:
                    # Router named a worker not present in `idle` — the
                    # router and `_idle_handles()` disagree (should never
                    # happen; `candidates` was built from `idle`). Bail
                    # without consuming; the next pass will retry.
                    log.warning(
                        "pool router picked unknown worker_id=%s; "
                        "skipping dispatch this pass",
                        getattr(pick, "worker_id", "?"),
                    )
                    return

                pending = self._fifo.popleft()
                self._queue_depth = len(self._fifo)
                self._reserved.add(target.worker_id)
                async with self._dispatched_lock:
                    self._dispatched[pending.request_id] = pending

            # ── released _fifo_lock; safe to await below ────────────────
            task = asyncio.create_task(
                self._run_on_worker(target, pending),
                name=f"pool-run-{pending.request_id[:8]}",
            )
            self._run_tasks.add(task)
            task.add_done_callback(self._run_tasks.discard)
            # ALWAYS yield after progress so the event loop can run.
            await asyncio.sleep(0)

    async def _run_on_worker(
            self, handle: WorkerHandle, pending: _PendingRequest,
    ) -> None:
        # Tell generate_stream() the request has been picked up so its
        # queue-deadline check flips off and worker-side stall detection
        # takes over.
        pending.queue.put_nowait(("started", None))
        pending.worker_id = handle.worker_id
        _submit_ts = time.monotonic()
        _first_chunk_ts: Optional[float] = None
        _chunk_count = 0

        # ── real prompt token count, once per request ─────────────
        # Guarded by `if not pending.prompt_tokens` so a retry (same
        # pending object re-queued in the retry path below) does not
        # double-count. The count is added to `_prompt_tokens_total`
        # exactly once, so per-key and total aggregates are correct.
        if not pending.prompt_tokens:
            await self._ensure_vocab_tokenizer()
            _prompt = pending.payload.get("prompt", "") or ""
            pending.prompt_tokens = self._count_prompt_tokens(_prompt)
            self._prompt_tokens_total += pending.prompt_tokens

        _retry_needed = False
        try:
            async for token in handle.submit(pending.payload):
                if _first_chunk_ts is None:
                    _first_chunk_ts = time.monotonic()
                if pending.cancelled:
                    pending.queue.put_nowait(("cancelled", None))
                    return
                _chunk_count += 1
                pending.queue.put_nowait(("chunk", token))
            # read the worker's authoritative token count from the
            # handle. Set by `handle._dispatch` on done/cancelled/error.
            # 0 means "not reported" (old worker) — leave pending.token_count
            # at 0 so generate_stream falls back to chunk count.
            try:
                _worker_tokens = int(getattr(handle, "last_token_count", 0) or 0)
            except (TypeError, ValueError):
                _worker_tokens = 0
            pending.token_count = _worker_tokens if _worker_tokens > 0 else 0
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
            # Try to read the worker's partial token count even on error.
            try:
                _worker_tokens = int(getattr(handle, "last_token_count", 0) or 0)
            except (TypeError, ValueError):
                _worker_tokens = 0
            pending.token_count = _worker_tokens if _worker_tokens > 0 else 0
            # Retry-once when the failure was BEFORE any token was emitted.
            # Post-first-token failures never retry: the client has
            # already consumed part of the answer and cannot be handed a
            # fresh stream without visible duplication.
            if (
                _first_chunk_ts is None
                and not pending.cancelled
                and pending.retries < 1
            ):
                _retry_needed = True
            else:
                pending.queue.put_nowait(("error", str(exc)))
        finally:
            # release the dispatch reservation synchronously — before
            # any further awaits — so a concurrent dispatch pass can
            # immediately re-use this worker once it is actually idle again.
            self._reserved.discard(handle.worker_id)

            # record prefill duration with the REAL prompt token
            # count as the numerator (used by `prefill_tps()`).
            if _first_chunk_ts is not None and pending.prompt_tokens > 0:
                _prefill_ms = (_first_chunk_ts - _submit_ts) * 1000.0
                if _prefill_ms > 0:
                    self._prefill_durations_ms.append(_prefill_ms)
                    self._prefill_prompt_lens.append(int(pending.prompt_tokens))

        # NOTE: do NOT pop _dispatched here. If this attempt is about to be
        # retried, a concurrent cancel() must still find the request in
        # _dispatched during the retry window. _dispatched is cleaned up
        # by generate_stream's finally via _final_cleanup.
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
            # Also drop the root→pending mapping. A late heartbeat between
            # here and the next request on the same root_id would otherwise
            # report a stale "queued" position.
            self._root_to_request.pop(pending.root_id, None)

        # A request that is still sitting in the FIFO when its caller leaves
        # (client disconnect / Stop button / reload / queue deadline) must be
        # removed NOW. Otherwise the dispatcher later hands it to a worker,
        # which then generates a full answer nobody will read. (Locks are taken
        # one after the other, never nested, so this cannot deadlock with the
        # dispatcher, which nests _dispatched_lock inside _fifo_lock.)
        async with self._fifo_lock:
            if any(p is pending for p in self._fifo):
                self._fifo = deque(p for p in self._fifo if p is not pending)
                self._queue_depth = len(self._fifo)

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
        """
        Return live handles that are idle AND not already claimed by a
        dispatch in flight.

        `_reserved` closes the window between `asyncio.create_task(_run_on_worker)`
        and that task's first step, during which the worker's `is_idle()`
        still returns True. Without the filter, the next dispatch pass could
        hand the same worker a second request and hit the worker's
        "busy" error (or worse, silently interleave two streams).
        """
        return [
            h for h in self._supervisor.live_handles()
            if h.is_idle() and h.worker_id not in self._reserved
        ]

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
