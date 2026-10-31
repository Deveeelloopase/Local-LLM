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


@dataclass
class _PendingRequest:
    payload: Dict[str, Any]
    request_id: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=4096))
    cancelled: bool = False


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
        python_exe: Optional[str] = None,
    ) -> None:
        self._spec = load_spec
        self._router = CAASRouter(router_weights)
        self._initial = max(1, int(initial_workers))
        self._max = max(self._initial, int(max_workers))
        self._ram_refuse_pct = float(ram_refuse_pct)
        self._queue_max = int(global_queue_max)

        self._supervisor = WorkerSupervisor(
            load_spec=load_spec,
            python_exe=python_exe,
            heartbeat_s=heartbeat_s,
        )
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

        # Prefill-side telemetry (populated in _run_on_worker). avg_tps
        # uses TTFT for the whole request; prefill_tps() isolates the
        # prefill cost specifically.
        self._prefill_durations_ms: deque = deque(maxlen=500)
        self._prefill_prompt_lens: deque = deque(maxlen=500)

        self._dispatched: Dict[str, _PendingRequest] = {}
        self._dispatched_lock = asyncio.Lock()


    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._supervisor.ensure_min_workers(self._initial)
        await self._supervisor.start_monitor()
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
                p.queue.put_nowait(("error", "server shutting down"))
            self._fifo.clear()
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
        return [self._spec.get("model_id", "")] if self._spec else []

    async def generate_stream(self, ir: Any) -> AsyncIterator[str]:
        """
        Accepts an InferenceRequest (any object exposing the fields below).
        Yields tokens. On error, raises.
        """
        import psutil
        if psutil.virtual_memory().percent > self._ram_refuse_pct:
            raise RuntimeError(
                f"server RAM at {psutil.virtual_memory().percent:.0f}% — "
                f"refusing new request to prevent crash"
            )

        async with self._fifo_lock:
            if len(self._fifo) >= self._queue_max:
                self._rejected_overloaded += 1
                raise RuntimeError("pool global queue full — server overloaded")
            pending = _PendingRequest(
                payload=self._serialize(ir),
                request_id=ir.request_id,
            )
            self._fifo.append(pending)
            self._queue_depth = len(self._fifo)
        self._dispatching.set()

        start_ts = time.monotonic()
        first_token_ts: Optional[float] = None
        token_count = 0
        try:
            while True:
                kind, data = await pending.queue.get()
                if kind == "chunk":
                    if first_token_ts is None:
                        first_token_ts = time.monotonic()
                        self._ttft_samples.append((first_token_ts - start_ts) * 1000.0)
                    token_count += 1
                    self._total_tokens += 1
                    yield data
                elif kind == "done":
                    return
                elif kind == "cancelled":
                    # Emit the canonical marker so the API layer's
                    # _do_persist path checkpoints rather than saving a
                    # truncated reply as a completed conversation.
                    from local_llm_core.engine.inference.inference_engine import (
                        _make_interruption_marker,
                    )
                    yield _make_interruption_marker(token_count)
                    return
                elif kind == "error":
                    raise RuntimeError(data)
                elif kind == "started":
                    # Worker took the request off the FIFO
                    self._queue_depth = len(self._fifo)
        finally:
            self._total_requests += 1
            if first_token_ts is not None:
                decode_s = time.monotonic() - first_token_ts
                if decode_s > 0.05 and token_count >= 2:
                    self._decode_tps_samples.append((token_count - 1) / decode_s)

    async def cancel(self, request_id: str) -> None:
        async with self._fifo_lock:
            for p in list(self._fifo):
                if p.request_id == request_id:
                    p.cancelled = True
                    self._fifo.remove(p)
                    p.queue.put_nowait(("cancelled", None))
                    self._queue_depth = len(self._fifo)
                    return
        # Not in FIFO — either dispatched to a worker or already finished.
        async with self._dispatched_lock:
            hit = self._dispatched.get(request_id)
            if hit is not None:
                hit.cancelled = True
        for h in self._supervisor.live_handles():
            await h.cancel(request_id)

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
                idle = self._idle_handles()
                if not idle:
                    return
                # Router picks based on prefix affinity
                request_token_ids = self._peek_token_ids(self._fifo[0])
                candidate = self._router.pick(
                    request_token_ids=request_token_ids,
                    candidates=[
                        IdleCandidate(
                            worker_id=h.worker_id,
                            resident_prefix_head=h.status().resident_prefix_head,
                            last_ready_ts=h.status().last_ready_ts,
                        )
                        for h in idle
                    ],
                )
                if candidate is None:
                    return
                # Pop the request and send it
                pending = self._fifo.popleft()
                self._queue_depth = len(self._fifo)
                target = next(h for h in idle if h.worker_id == candidate.worker_id)
                async with self._dispatched_lock:
                    self._dispatched[pending.request_id] = pending
            # Fire and forget — the worker yields to pending.queue
            asyncio.create_task(self._run_on_worker(target, pending))
            # Yield so the task can mark its worker busy before we loop back
            # and re-check _idle_handles(). Without this, the same idle worker
            # is selected twice in one dispatch pass and the second request
            # fails with "already busy".
            await asyncio.sleep(0)

    async def _run_on_worker(
            self, handle: WorkerHandle, pending: _PendingRequest,
    ) -> None:
        pending.queue.put_nowait(("started", None))
        _submit_ts = time.monotonic()
        _first_chunk_ts: Optional[float] = None
        # Approximate prompt length for prefill_tps() estimator
        _prompt_chars = len(pending.payload.get("prompt", "") or "")
        _prompt_tokens_approx = max(1, int(_prompt_chars / 3.5))
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
            pending.queue.put_nowait(("error", str(exc)))
        finally:
            # record prefill duration for prefill_tps() estimation.
            if _first_chunk_ts is not None:
                _prefill_ms = (_first_chunk_ts - _submit_ts) * 1000.0
                if _prefill_ms > 0:
                    self._prefill_durations_ms.append(_prefill_ms)
                    self._prefill_prompt_lens.append(_prompt_tokens_approx)
            async with self._dispatched_lock:
                self._dispatched.pop(pending.request_id, None)
            if self._fifo:
                self._dispatching.set()

    # ── helpers ────────────────────────────────────────────────────────────

    def _idle_handles(self) -> List[WorkerHandle]:
        return [h for h in self._supervisor.live_handles() if h.is_idle()]

    def _peek_token_ids(self, pending: _PendingRequest) -> List[int]:
        """Best-effort: the parent does not tokenize; approximate by characters.

        Using the resident prefix head (which is token IDs) against the
        prompt's *characters* under-counts the shared prefix, so the affinity
        score is conservative. That's fine — CAAS still correctly prefers
        the worker with the largest genuinely-shared prefix; it simply
        refuses to over-claim affinity it cannot verify.
        """
        prompt = pending.payload.get("prompt", "")
        # Approximate mapping char → token count using 3.5 chars/token.
        # Return a pseudo-sequence where each element encodes ~3.5 chars.
        n_approx = max(0, int(len(prompt) / 3.5))
        return list(range(n_approx))

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
