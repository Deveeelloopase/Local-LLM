"""
batched_engine.py — Continuous batching for llama-cpp-python 0.3.x.

Verified primitives:
  * llm._ctx.decode(LlamaBatch)     — batched forward pass, returns None
  * LlamaBatch.add_sequence(tokens, seq_id, logits_all)
  * llm._ctx.get_logits()           — flat float32, n_vocab per seq-with-logits
  * llm._ctx.kv_cache_seq_rm(seq_id, -1, -1) → bool
  * llm._ctx.kv_cache_seq_cp        — for future prefix reuse
  * LlamaContext(*, model=, params=, verbose=)  — rebuild works

Architecture:
  BatchedInferenceEngine owns the Llama instance exclusively while active.
  Callers use generate_stream(req) — same signature as InferenceEngine.
  A single asyncio scheduler task drives one decode() per step across all
  active sequences. Per-seq asyncio.Queue carries detokenized deltas.

Design points (v1):
  * One context, one decode per step, N active sequences.
  * Chunked prefill: new prompts are prefilled in ≤(n_batch/(n_seq_max+1))
    token chunks so a 4K prompt doesn't block other decoders.
  * KV eviction on completion: kv_cache_seq_rm(seq_id, -1, -1).
  * Sampling: pure-Python (temperature, top_k, top_p, repetition penalty).
  * Detokenization: incremental via full-tail re-detok (O(n²) safe for
    typical chat lengths; optimise later if needed).
  * Stop: EOS token, max_tokens, caller cancel. Stop strings and loop
    guard come in v1 too (simple substring check).
  * Decode runs in a thread executor so the event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Low-level batch manipulation — verified against llama-cpp-python 0.3.36
# ═══════════════════════════════════════════════════════════════════════════

def _batch_add_token(
    batch: Any, token: int, pos: int, seq_id: int, logits: bool,
) -> None:
    """Append one token at explicit (pos, seq_id) to a LlamaBatch.

    Bypasses LlamaBatch.add_sequence() because that method does not accept
    an explicit pos — it derives pos internally, which is unsafe when mixing
    sequences with different prompt offsets in one batch. Direct ctypes
    manipulation of the underlying llama_batch struct is the only correct
    way to place tokens at absolute positions per seq.

    Verified: raw llama_batch exposes .token[i], .pos[i], .n_seq_id[i],
    .seq_id[i][0], .logits[i], and .n_tokens (settable).
    """
    raw = batch.batch
    i = raw.n_tokens
    raw.token[i] = token
    raw.pos[i] = pos
    raw.n_seq_id[i] = 1
    raw.seq_id[i][0] = seq_id
    raw.logits[i] = 1 if logits else 0
    raw.n_tokens = i + 1


# ═══════════════════════════════════════════════════════════════════════════
# Sampling — pure Python, per-seq independent
# ═══════════════════════════════════════════════════════════════════════════

def _sample_logits(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated_tokens: List[int],
    rng: np.random.Generator,
) -> int:
    """Sample one token from a logit vector.

    Applies (in order): repetition penalty, temperature, top-k, top-p,
    softmax, categorical sample.

    Greedy when temperature <= 1e-6 (returns argmax). Otherwise stochastic.
    All operations are in-place-ish on a float64 copy — no mutation of the
    caller's buffer.
    """
    logits = logits.astype(np.float64, copy=True)

    # Repetition penalty: boost logits of unseen tokens, suppress seen ones
    if repetition_penalty != 1.0 and generated_tokens:
        for tok in set(generated_tokens):
            if 0 <= tok < logits.size:
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty

    if temperature <= 1e-6:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf

    logits_max = float(logits.max())
    if logits_max == -np.inf:
        return 0
    probs = np.exp(logits - logits_max)
    denom = probs.sum()
    if denom <= 0.0 or not np.isfinite(denom):
        return int(np.argmax(logits))
    probs /= denom

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        keep = sorted_idx[:cutoff]
        mask = np.ones_like(probs, dtype=bool)
        mask[keep] = False
        probs[mask] = 0.0
        s = probs.sum()
        if s <= 0:
            return int(np.argmax(logits))
        probs /= s

    return int(rng.choice(probs.size, p=probs))


# ═══════════════════════════════════════════════════════════════════════════
# Per-sequence state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _SequenceState:
    seq_id: int
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int          # <=0 means unlimited (bounded by context)
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop_strings: List[str]
    timeout_s: float

    # Progress
    prefill_offset: int = 0
    last_token: Optional[int] = None
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    prev_detok_text: str = ""

    # Lifecycle
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=512))
    finished: bool = False
    cancelled: bool = False
    error: Optional[str] = None

    # Timestamps (for stall detection and stats)
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    enqueued_ts: float = field(default_factory=time.monotonic)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_offset >= self.prompt_len

    @property
    def total_len(self) -> int:
        return self.prefill_offset

    @property
    def gen_count(self) -> int:
        return len(self.generated_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Batched inference engine
# ═══════════════════════════════════════════════════════════════════════════

class BatchedInferenceEngine:
    """
    Continuous batching scheduler on top of a single Llama instance.

    Owns the Llama's context exclusively while active. All methods that
    call into llama.cpp run in a thread executor so the event loop stays
    responsive.

    Usage:
        engine = BatchedInferenceEngine(llm, n_seq_max=4)
        await engine.start()
        async for tok in engine.generate_stream(req):
            ...
        await engine.shutdown()
    """

    def __init__(
        self,
        llm: Any,
        *,
        n_seq_max: int = 4,
        kv_unified: bool = True,
        prefill_chunk_size: Optional[int] = None,
        inter_token_stall_s: float = 120.0,
        prefill_stall_s: float = 600.0,
        stats_window: int = 256,
        thermal_state: Optional[Any] = None,
        adaptive_n_seq_max: bool = False,
        adaptive_target_ttft_ms: int = 5000,
        adaptive_min_n_seq_max: int = 1,
        adaptive_max_n_seq_max: Optional[int] = None,
        adaptive_check_interval_s: float = 60.0,
        thermal_override_enabled: bool = True,
    ) -> None:
        from llama_cpp import llama_cpp as lc
        from llama_cpp._internals import LlamaContext

        self.llm = llm
        self._n_seq_max = int(n_seq_max)
        self._initial_n_seq_max = int(n_seq_max)
        self._kv_unified = bool(kv_unified)
        self._inter_token_stall_s = float(inter_token_stall_s)
        self._prefill_stall_s = float(prefill_stall_s)

        # ── Adaptive n_seq_max ────────────────────────────────────────
        # Cannot exceed _initial_n_seq_max — the context was built with that
        # value at construction. Adaptive can shrink to adaptive_min and grow
        # back toward initial, but never beyond it.
        self._thermal_state = thermal_state
        self._adaptive_enabled = bool(adaptive_n_seq_max)
        self._adaptive_target_ttft_ms = int(adaptive_target_ttft_ms)
        self._adaptive_min = max(1, int(adaptive_min_n_seq_max))
        _amax = adaptive_max_n_seq_max
        self._adaptive_max = (
            min(int(_amax), self._initial_n_seq_max)
            if _amax is not None else self._initial_n_seq_max
        )
        self._adaptive_interval_s = float(adaptive_check_interval_s)
        self._thermal_override_enabled = bool(thermal_override_enabled)
        self._adaptive_task: Optional[asyncio.Task] = None

        # Pull real hardware limits from the loaded context.
        # llama-cpp-python 0.3.36 quirk: many of these are METHODS, not
        # properties. ctx.n_ctx(), model.n_vocab(), llm.token_eos() are all
        # callable. context_params.* are plain C struct int fields (safe).
        # Wrap each read so a future version that flips a method to a
        # property doesn't break us.
        def _read_int(obj: Any, name: str, fallback: int = 0) -> int:
            v = getattr(obj, name, None)
            if v is None:
                return fallback
            if callable(v):
                try:
                    v = v()
                except Exception:
                    return fallback
            try:
                return int(v)
            except (TypeError, ValueError):
                return fallback

        self._n_ctx = _read_int(llm._ctx, "n_ctx") or \
                      _read_int(llm.context_params, "n_ctx", 2048)
        self._n_batch = _read_int(llm.context_params, "n_batch", 512)
        self._n_ubatch = _read_int(llm.context_params, "n_ubatch", 256)
        self._n_vocab = _read_int(llm._model, "n_vocab", 32000)
        self._eos_token = _read_int(llm, "token_eos", -1)

        # P2-2 companion: n_seq_max MUST be < n_batch so _run_step's
        # prefill budget (n_batch - n_active) is always >= 1. If
        # n_seq_max >= n_batch, phase 1 would force prefill_budget to 1
        # via max() but phase 2 would then emit up to n_seq_max decode
        # tokens, overflowing n_batch by construction.
        # Realistic configs have n_batch=512-2048 and n_seq_max=2-8;
        # this only fires for pathological settings (n_batch<8 or
        # n_seq_max absurdly large).
        if self._n_seq_max >= self._n_batch:
            _clamped = max(1, self._n_batch - 1)
            log.warning(
                "BatchedInferenceEngine: n_seq_max=%d >= n_batch=%d — "
                "clamping to %d (n_seq_max must be < n_batch so decode "
                "reservation leaves room for at least one prefill token)",
                self._n_seq_max, self._n_batch, _clamped,
            )
            self._n_seq_max = _clamped
            self._initial_n_seq_max = _clamped

        # Prefill chunk sizing. Must leave room for decode tokens from other
        # seqs in the same step; every active seq contributes at most 1
        # decode token, so reserve n_seq_max slots.
        if prefill_chunk_size is None:
            prefill_chunk_size = max(
                32, (self._n_batch - self._n_seq_max) // max(1, self._n_seq_max),
            )
        self._prefill_chunk = min(int(prefill_chunk_size), self._n_batch // 2)
        log.info(
            "BatchedInferenceEngine: n_seq_max=%d n_ctx=%d n_batch=%d "
            "n_vocab=%d eos=%d prefill_chunk=%d",
            self._n_seq_max, self._n_ctx, self._n_batch, self._n_vocab,
            self._eos_token, self._prefill_chunk,
        )

        # Rebuild context with multi-seq params (verified works, preserves
        # llm.generate())
        self._rebuild_context(lc, LlamaContext)

        # Scheduler state
        self._active: Dict[int, _SequenceState] = {}     # seq_id → state
        self._pending: Deque[_SequenceState] = deque()   # awaiting admission
        self._free_ids: List[int] = list(range(self._n_seq_max))
        self._new_request_event = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # RNG
        self._rng = np.random.default_rng()

        # Stats
        self._stats_window = int(stats_window)
        self._ttft_samples: Deque[float] = deque(maxlen=stats_window)
        self._decode_tps_samples: Deque[float] = deque(maxlen=stats_window)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_evictions = 0
        self._scheduler_errors = 0
        self._step_count = 0

    # ── Context rebuild ─────────────────────────────────────────────

    def _rebuild_context(self, lc: Any, LlamaContext: Any) -> None:
        """Swap llm._ctx with a context that has n_seq_max>1."""
        if self.llm.context_params.n_seq_max >= self._n_seq_max:
            return

        new_params = lc.llama_context_default_params()
        for f, _ in new_params._fields_:
            if f in ("n_seq_max", "kv_unified", "n_outputs_max",
                     "n_outputs_max_per_seq"):
                continue
            try:
                setattr(new_params, f, getattr(self.llm.context_params, f))
            except (AttributeError, TypeError):
                pass
        new_params.n_seq_max = self._n_seq_max
        new_params.kv_unified = self._kv_unified
        new_params.n_outputs_max = self._n_seq_max
        new_params.n_outputs_max_per_seq = 1

        new_ctx = LlamaContext(
            model=self.llm._model, params=new_params, verbose=False,
        )
        old_ctx = self.llm._ctx
        self.llm._ctx = new_ctx
        self.llm.context_params = new_params
        self.llm._desired_n_seq_max = self._n_seq_max
        try:
            del old_ctx
        except Exception:
            pass
        self.llm.n_tokens = 0

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(
                self._scheduler_loop(), name="batched_engine_scheduler",
            )
        if (self._adaptive_enabled
                and (self._adaptive_task is None or self._adaptive_task.done())):
            self._adaptive_task = asyncio.create_task(
                self._adaptive_loop(), name="batched_engine_adaptive",
            )

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        self._new_request_event.set()
        if self._adaptive_task and not self._adaptive_task.done():
            self._adaptive_task.cancel()
            try:
                await self._adaptive_task
            except asyncio.CancelledError:
                pass
            self._adaptive_task = None
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        # Drain any stragglers
        for seq in list(self._active.values()):
            self._fail_seq(seq, "engine shutdown")
        for seq in list(self._pending):
            self._fail_seq(seq, "engine shutdown")
        self._pending.clear()

    # ── Public API ──────────────────────────────────────────────────

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        """Stream tokens for one request.

        req: InferenceRequest-like object with attributes:
            request_id, prompt, model_id, max_tokens, temperature,
            top_p, top_k, stop, timeout_s.
        """
        rid = str(getattr(req, "request_id", "") or f"req_{id(req)}")
        prompt = str(getattr(req, "prompt", "") or "")
        max_tokens = int(getattr(req, "max_tokens", 0) or 0)
        temperature = float(getattr(req, "temperature", 0.7))
        top_p = float(getattr(req, "top_p", 0.95))
        top_k = int(getattr(req, "top_k", 40))
        stop_list = list(getattr(req, "stop", []) or [])
        timeout_s = float(getattr(req, "timeout_s", 0.0) or 0.0)

        if not prompt:
            raise ValueError("empty prompt")

        # Tokenize (blocking; but llama tokenizer is microseconds)
        try:
            prompt_tokens = list(self.llm.tokenize(prompt.encode("utf-8", "ignore")))
        except Exception as exc:
            raise RuntimeError(f"tokenize failed: {exc}") from exc

        if not prompt_tokens:
            raise ValueError("empty prompt after tokenization")

        seq = _SequenceState(
            seq_id=-1,   # assigned on admission
            request_id=rid,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,   # conservative default; could be wired
            stop_strings=stop_list,
            timeout_s=timeout_s,
        )

        self._pending.append(seq)
        self._new_request_event.set()
        self._total_requests += 1

        try:
            while True:
                # Wait for a chunk from the scheduler
                try:
                    item = await seq.queue.get()
                except asyncio.CancelledError:
                    seq.cancelled = True
                    self._new_request_event.set()
                    raise
                if item is None:
                    # Scheduler signalled completion
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Ensure the scheduler knows we're gone
            if not seq.finished:
                seq.cancelled = True
                self._new_request_event.set()
            # Free the seq_id if it was assigned
            if seq.seq_id >= 0 and seq.seq_id in self._active:
                self._active.pop(seq.seq_id, None)
                try:
                    self._free_ids.append(seq.seq_id)
                except Exception:
                    pass

    # ── Scheduler ───────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                # 1. Admit pending requests
                self._admit_pending()

                # 2. If nothing active and nothing pending, wait
                if not self._active and not self._pending:
                    self._new_request_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._new_request_event.wait(), timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # 3. If still nothing active (pending but no free seq_id),
                #    wait briefly
                if not self._active:
                    try:
                        await asyncio.wait_for(
                            self._new_request_event.wait(), timeout=0.1,
                        )
                    except asyncio.TimeoutError:
                        pass
                    self._new_request_event.clear()
                    continue

                # 4. Run one step
                await self._run_step()

                # Yield control every step so incoming requests/cancels can
                # be processed before the next decode
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("BatchedInferenceEngine scheduler crashed")
            self._scheduler_errors += 1

    def _admit_pending(self) -> None:
        """Move pending requests into active with a fresh seq_id."""
        while self._pending and self._free_ids:
            seq = self._pending.popleft()
            if seq.cancelled or seq.finished:
                continue
            seq.seq_id = self._free_ids.pop(0)
            self._active[seq.seq_id] = seq
            log.info(
                "admit seq_id=%d rid=%s prompt_len=%d",
                seq.seq_id, seq.request_id[:8], seq.prompt_len,
            )

    @staticmethod
    def _coerce_logits(raw: Any, expected_n: int) -> Optional[np.ndarray]:
        """Normalize a logits return value to a flat float32 numpy array.

        Handles all forms llama-cpp-python has produced across versions:
          * List[float] / tuple               — direct numpy
          * np.ndarray                        — direct
          * LP_c_float (ctypes pointer)       — cast via numpy ctypeslib
          * NULL pointer                      — returns None (logged)

        A NULL pointer from get_logits()/get_logits_ith() means llama.cpp
        did not compute logits for the requested slot. The caller must
        treat None as "this slot failed" and either fall back to the
        whole-buffer path or fail the seq.
        """
        import ctypes as _ct
        if raw is None:
            return None
        try:
            if isinstance(raw, np.ndarray):
                return raw.astype(np.float32, copy=False).ravel()
            if isinstance(raw, (list, tuple)):
                return np.asarray(raw, dtype=np.float32).ravel()
            if isinstance(raw, _ct._Pointer) or hasattr(raw, "contents"):
                # ctypes cast of a NULL pointer raises ValueError("NULL
                # pointer access"), so guard explicitly.
                try:
                    arr_type = _ct.c_float * int(expected_n)
                    ptr = _ct.cast(raw, _ct.POINTER(arr_type))
                    return np.ctypeslib.as_array(ptr.contents).astype(
                        np.float32, copy=False,
                    ).ravel()
                except (ValueError, TypeError) as exc:
                    log.error(
                        "_coerce_logits: pointer cast failed (NULL or wrong "
                        "type): %s", exc,
                    )
                    return None
            log.error(
                "_coerce_logits: unexpected type %s repr=%r",
                type(raw).__name__, raw,
            )
            return None
        except Exception as exc:
            log.error("_coerce_logits failed: %s", exc)
            return None

    async def _run_step(self) -> None:
        """One scheduler iteration: build batch, decode, sample, evict."""
        # Evict finished/cancelled seqs before building (kv_cache_seq_rm)
        self._evict_finished()

        if not self._active:
            return

        # Build batch entries. Two-phase allocation guarantees that active
        # decode streams are never starved by a long prefill. Prior single-
        # loop implementation interleaved prefill and decode; a prefill seq
        # that consumed the full prefill_budget caused every later decode
        # seq to `break` before emitting its next token — turning the batch
        # into "prefill only" for that step and stalling live streams.
        #
        # Phase 1 allocates up to (n_batch - n_reserved) tokens across all
        # prefill seqs. Phase 2 then adds exactly one decode token per seq
        # with a last_token, using the n_reserved slots guaranteed by the
        # budget split. Total entries <= n_batch by construction (proved by
        # n_seq_max < n_batch clamp in __init__).
        entries: List[tuple] = []
        logits_seqs: List[tuple] = []
        logits_index = 0

        n_reserved = len(self._active)      # 1 decode token max per active seq
        prefill_budget = max(1, self._n_batch - n_reserved)

        # ── Phase 1: prefill chunks ─────────────────────────────────
        # Snapshot self._active.values() so a hypothetical concurrent
        # mutation (which cannot happen today — no awaits inside this
        # function — but is defensively cheap) would raise cleanly.
        for seq in list(self._active.values()):
            if seq.finished or seq.cancelled or seq.prefill_done:
                continue
            remaining = seq.prompt_len - seq.prefill_offset
            take = min(remaining, self._prefill_chunk, prefill_budget)
            if take <= 0:
                continue
            chunk = seq.prompt_tokens[
                seq.prefill_offset:seq.prefill_offset + take
            ]
            base_pos = seq.prefill_offset
            seq_will_finish_prefill = (take == remaining)
            last_emitted_slot: Optional[int] = None

            for i, tok in enumerate(chunk):
                is_last = (i == take - 1)
                wants_logits = is_last and seq_will_finish_prefill
                pos = base_pos + i
                entries.append((seq, tok, pos, wants_logits))
                if wants_logits:
                    last_emitted_slot = logits_index
                    logits_index += 1

            if seq_will_finish_prefill and last_emitted_slot is not None:
                logits_seqs.append((seq, last_emitted_slot))

            seq.prefill_offset += take
            prefill_budget -= take

        # ── Phase 2: decode ─────────────────────────────────────────
        # No budget gate here — n_reserved was reserved exactly for this,
        # and the number of seqs with last_token is <= len(self._active)
        # == n_reserved. So total entries cannot exceed n_batch.
        for seq in list(self._active.values()):
            if seq.finished or seq.cancelled:
                continue
            if not seq.prefill_done:
                continue
            if seq.last_token is None:
                # Seq finished prefill in a PRIOR step but never sampled
                # (would only happen if sampling failed, which routes
                # through _fail_seq → _evict_seq; this branch is
                # unreachable in practice but defensive against a future
                # code path that clears last_token).
                continue
            entries.append((seq, seq.last_token, seq.prefill_offset, True))
            logits_seqs.append((seq, logits_index))
            logits_index += 1
            seq.prefill_offset += 1

        if not entries:
            return

        # Build the LlamaBatch
        from llama_cpp._internals import LlamaBatch
        batch = LlamaBatch(
            n_tokens=len(entries), embd=0, n_seq_max=self._n_seq_max,
        )
        for seq, tok, pos, logits in entries:
            _batch_add_token(batch, tok, pos, seq.seq_id, logits)

        # Decode (blocking C call → executor so event loop stays alive)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.llm._ctx.decode, batch)
        except Exception as exc:
            self._handle_decode_error(exc, entries)
            return

        self._step_count += 1

        n_logits_seq = len(logits_seqs)
        if n_logits_seq == 0:
            return

        n_vocab = self._n_vocab
        expected_floats = n_logits_seq * n_vocab

        # ── Read whole logits buffer ────────────────────────────────
        # get_logits() returns a ctypes LP_c_float pointer (verified on
        # this build). The buffer is n_outputs * n_vocab floats, where
        # n_outputs = count of logits=True entries in the batch, in
        # batch order. Slicing by output index is unambiguous; per-slot
        # get_logits_ith has version-dependent semantics (output index
        # vs batch position) and has proven unreliable here.
        raw_whole = None
        try:
            raw_whole = self.llm._ctx.get_logits()
        except Exception as exc:
            log.error("get_logits() raised: %s", exc)
            self._handle_decode_error(exc, entries)
            return

        whole = self._coerce_logits(raw_whole, expected_floats)
        if whole is None:
            log.error(
                "get_logits() returned unreadable buffer "
                "(type=%s repr=%r expected=%d floats). "
                "Sampling impossible this step.",
                type(raw_whole).__name__, raw_whole, expected_floats,
            )
            # Diagnostic: also probe get_logits_ith to help future debugging
            try:
                for i in range(min(2, n_logits_seq)):
                    r2 = self.llm._ctx.get_logits_ith(i)
                    log.error(
                        "  diagnostic get_logits_ith(%d) type=%s repr=%r",
                        i, type(r2).__name__, r2,
                    )
            except Exception as e2:
                log.error("  diagnostic get_logits_ith failed: %s", e2)
            for seq, _ in logits_seqs:
                self._fail_seq(seq, "get_logits() unreadable")
            return

        if whole.size < expected_floats:
            log.error(
                "get_logits() buffer too short: got %d floats, expected %d "
                "(n_vocab=%d, logits_slots=%d)",
                whole.size, expected_floats, n_vocab, n_logits_seq,
            )
            for seq, _ in logits_seqs:
                self._fail_seq(seq, "logits buffer too short")
            return

        # ── Sample per seq ──────────────────────────────────────────
        for seq, output_idx in logits_seqs:
            if seq.finished or seq.cancelled:
                continue

            lo = output_idx * n_vocab
            hi = lo + n_vocab
            slice_ = whole[lo:hi]
            if slice_.size < n_vocab:
                log.error(
                    "slice for slot %d is %d floats (n_vocab=%d)",
                    output_idx, slice_.size, n_vocab,
                )
                self._fail_seq(seq, f"short logits slice slot {output_idx}")
                continue

            try:
                next_tok = _sample_logits(
                    slice_,
                    temperature=seq.temperature,
                    top_k=seq.top_k,
                    top_p=seq.top_p,
                    repetition_penalty=seq.repetition_penalty,
                    generated_tokens=seq.generated_tokens,
                    rng=self._rng,
                )
            except Exception as exc:
                log.error("sample failed seq=%d: %s", seq.seq_id, exc)
                self._fail_seq(seq, f"sample failed: {exc}")
                continue

            seq.last_token = next_tok
            seq.generated_tokens.append(next_tok)
            seq.last_token_ts = time.monotonic()
            self._total_tokens += 1

            if seq.first_token_ts is None:
                seq.first_token_ts = seq.last_token_ts
                self._ttft_samples.append(
                    seq.first_token_ts - seq.enqueued_ts,
                )

            # Detokenize incrementally and emit delta
            self._emit_detok_delta(seq)

            # Stop conditions
            if self._should_stop(seq, next_tok):
                self._finish_seq(seq)

    # ── Detokenization ──────────────────────────────────────────────

    def _emit_detok_delta(self, seq: _SequenceState) -> None:
        """Re-detokenize the full generated tail, emit only the delta."""
        try:
            raw_bytes = self.llm.detokenize(seq.generated_tokens, special=False)
            full_text = raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            log.warning(
                "detokenize failed seq=%d: %s — using raw token id string",
                seq.seq_id, exc,
            )
            return

        delta = full_text[len(seq.prev_detok_text):]
        seq.prev_detok_text = full_text
        seq.generated_text = full_text

        if not delta:
            return

        # Stop-string check on the full accumulated text — handled in caller
        try:
            seq.queue.put_nowait(delta)
        except asyncio.QueueFull:
            log.warning(
                "seq=%d queue full, dropping delta (client too slow)",
                seq.seq_id,
            )

    def _should_stop(self, seq: _SequenceState, next_tok: int) -> bool:
        """True when this seq should be evicted after this token."""
        # EOS
        if next_tok == self._eos_token:
            return True
        # Max tokens
        if seq.max_tokens > 0 and seq.gen_count >= seq.max_tokens:
            return True
        # Context exhaustion
        if seq.total_len >= self._n_ctx - 4:
            return True
        # Stop strings
        if seq.stop_strings:
            for s in seq.stop_strings:
                if s and s in seq.generated_text:
                    return True
        # Stall detection
        now = time.monotonic()
        if seq.timeout_s > 0 and (now - seq.enqueued_ts) > seq.timeout_s:
            return True
        if seq.last_token_ts is not None and \
                (now - seq.last_token_ts) > self._inter_token_stall_s:
            return True
        return False

    # ── Eviction ────────────────────────────────────────────────────

    def _evict_finished(self) -> None:
        """Remove finished/cancelled seqs from the active map and free KV."""
        to_evict = [
            seq for seq in self._active.values()
            if seq.finished or seq.cancelled
        ]
        for seq in to_evict:
            self._evict_seq(seq)

    def _evict_seq(self, seq: _SequenceState) -> None:
        """Free this seq's KV blocks and reuse its seq_id."""
        if seq.seq_id < 0:
            return
        try:
            self.llm._ctx.kv_cache_seq_rm(seq.seq_id, -1, -1)
        except Exception as exc:
            log.warning(
                "kv_cache_seq_rm(seq_id=%d) failed: %s", seq.seq_id, exc,
            )
        self._active.pop(seq.seq_id, None)
        # Only return the slot to _free_ids if it's within the CURRENT
        # admission window. If n_seq_max shrank while this seq was running,
        # its id may exceed the new limit — in that case we let the id go
        # (llama.cpp context still has _initial_n_seq_max slots, but we
        # don't hand out ids >= _n_seq_max).
        if seq.seq_id < self._n_seq_max and seq.seq_id not in self._free_ids:
            self._free_ids.append(seq.seq_id)
        self._total_evictions += 1
        log.info(
            "evict seq_id=%d rid=%s gen=%d", seq.seq_id, seq.request_id[:8],
            seq.gen_count,
        )

    def _finish_seq(self, seq: _SequenceState) -> None:
        """Mark a seq as naturally finished and signal its consumer."""
        if seq.finished:
            return
        seq.finished = True
        try:
            seq.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        # Compute decode TPS sample
        if seq.first_token_ts is not None and seq.gen_count > 1:
            dt = seq.last_token_ts - seq.first_token_ts
            if dt > 0.05:
                self._decode_tps_samples.append((seq.gen_count - 1) / dt)

    def _fail_seq(self, seq: _SequenceState, reason: str) -> None:
        if seq.finished or seq.cancelled:
            return
        seq.error = reason
        seq.finished = True
        try:
            seq.queue.put_nowait(RuntimeError(reason))
        except asyncio.QueueFull:
            pass
        if seq.seq_id >= 0 and seq.seq_id in self._active:
            self._evict_seq(seq)

    def _handle_decode_error(self, exc: BaseException, entries: List[tuple]) -> None:
        """Recover from a decode() failure.

        v1 strategy: log, fail every seq in this step, try to clear the
        whole KV cache. This is crude but safe — the alternative is
        attempting to identify the culprit seq, which can hang if the
        context state is corrupted.

        The likely cause of a decode failure is n_batch overflow (a seq
        with a very long prompt asked for a chunk that exceeded the
        budget). This should be rare given our chunk sizing.
        """
        log.error("decode() failed: %s", exc)
        self._scheduler_errors += 1
        # Fail all seqs from this step
        failed_ids = set()
        for seq, *_ in entries:
            if seq.seq_id not in failed_ids:
                failed_ids.add(seq.seq_id)
                self._fail_seq(seq, f"decode failed: {exc}")
        # Attempt to reset KV to a clean state
        try:
            self.llm._ctx.kv_cache_clear()
        except Exception as clear_exc:
            log.error("kv_cache_clear failed after decode error: %s", clear_exc)


    async def _adaptive_loop(self) -> None:
        """Grow/shrink admission capacity based on TTFT and thermal state.

        Every adaptive_check_interval_s:
          * if thermal state reports throttling → halve n_seq_max (floor at min)
          * elif enough TTFT samples:
              - avg TTFT < 0.8 * target → n_seq_max += 1 (cap at max)
              - avg TTFT > 1.2 * target → n_seq_max -= 1 (floor at min)

        Changing n_seq_max NEVER evicts running sequences — it only limits
        which seq_ids can be handed to new admissions. Existing active seqs
        keep their slots until they finish naturally.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._adaptive_interval_s)
                if self._shutdown_event.is_set():
                    return

                current = self._n_seq_max

                # Thermal override wins
                thermal_throttling = False
                if self._thermal_state is not None:
                    try:
                        thermal_throttling = bool(
                            getattr(self._thermal_state, "throttling", False)
                        )
                    except Exception:
                        thermal_throttling = False

                if thermal_throttling and self._thermal_override_enabled:
                    new_n = max(self._adaptive_min, current // 2)
                    if new_n != current:
                        self._apply_n_seq_max(new_n)
                        log.warning(
                            "Adaptive: thermal throttle → n_seq_max %d→%d",
                            current, new_n,
                        )
                    continue

                # TTFT-driven growth/shrink — require enough samples
                if len(self._ttft_samples) < 8:
                    continue
                recent = list(self._ttft_samples)[-20:]
                avg_ttft_ms = 1000.0 * (sum(recent) / len(recent))

                lo_thr = self._adaptive_target_ttft_ms * 0.80
                hi_thr = self._adaptive_target_ttft_ms * 1.20

                if avg_ttft_ms < lo_thr and current < self._adaptive_max:
                    self._apply_n_seq_max(min(current + 1, self._adaptive_max))
                    log.info(
                        "Adaptive: TTFT %.0fms < %.0fms → n_seq_max %d→%d",
                        avg_ttft_ms, lo_thr, current, self._n_seq_max,
                    )
                elif avg_ttft_ms > hi_thr and current > self._adaptive_min:
                    self._apply_n_seq_max(max(current - 1, self._adaptive_min))
                    log.warning(
                        "Adaptive: TTFT %.0fms > %.0fms → n_seq_max %d→%d",
                        avg_ttft_ms, hi_thr, current, self._n_seq_max,
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("Adaptive loop error: %s", exc)

    def _apply_n_seq_max(self, new_n: int) -> None:
        """Change admission capacity without touching running sequences.

        Why this is safe to do at runtime:
          * llama.cpp's context was built with n_seq_max = _initial_n_seq_max,
            so seq_ids in [0, _initial_n_seq_max) are always valid.
          * _n_seq_max is only consulted when picking a free seq_id for a
            new admission. Shrinking it below _initial just means fewer ids
            are handed out.
          * An active seq with id >= new_n keeps running normally — its
            slot is reclaimed by _evict_seq when it finishes, at which
            point the id is checked against the CURRENT _n_seq_max before
            being returned to _free_ids.
        """
        new_n = max(1, min(int(new_n), self._initial_n_seq_max))
        if new_n == self._n_seq_max:
            return
        old_n = self._n_seq_max
        self._n_seq_max = new_n
        active_ids = set(self._active.keys())
        self._free_ids = [i for i in range(new_n) if i not in active_ids]
        log.info(
            "n_seq_max adjusted %d→%d (active=%d, free_ids=%s)",
            old_n, new_n, len(active_ids), self._free_ids,
        )

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        ttfts = list(self._ttft_samples)
        tps_samples = list(self._decode_tps_samples)
        return {
            "engine_version": "1.0.0",
            "mode": "continuous_batching",
            "n_seq_max": self._n_seq_max,
            "n_ctx": self._n_ctx,
            "n_batch": self._n_batch,
            "prefill_chunk": self._prefill_chunk,
            "active_seqs": len(self._active),
            "pending_seqs": len(self._pending),
            "free_seq_ids": len(self._free_ids),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_evictions": self._total_evictions,
            "scheduler_errors": self._scheduler_errors,
            "step_count": self._step_count,
            "avg_ttft_ms": round(1000.0 * (sum(ttfts) / len(ttfts)), 1) if ttfts else 0.0,
            "avg_decode_tps": round(sum(tps_samples) / len(tps_samples), 2) if tps_samples else 0.0,
            "eos_token": self._eos_token,
            "n_vocab": self._n_vocab,
            "initial_n_seq_max": self._initial_n_seq_max,
            "adaptive_enabled": self._adaptive_enabled,
            "adaptive_min": self._adaptive_min,
            "adaptive_max": self._adaptive_max,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone test — run with: python -m local_llm_core.engine.inference.batched_engine
# ═══════════════════════════════════════════════════════════════════════════

async def _standalone_test() -> None:
    """Submit 4 concurrent requests, verify all complete.

    This is the acceptance test for Step 3 — it exercises the full path
    without needing the pool, worker, or any FastAPI routes.
    """
    import os
    import sys

    from llama_cpp import Llama
    from local_llm_core.engine.inference.inference_engine import InferenceRequest

    model = os.environ.get(
        "BATCH_TEST_MODEL",
        "/home/deve-oops/models/Qwen3-4B-Q4_K_M.gguf",
    )

    print(f"Loading {model}...")
    llm = Llama(
        model_path=model, n_ctx=2048, n_batch=512, n_ubatch=256,
        n_gpu_layers=0, n_threads=4, n_threads_batch=4,
        flash_attn=False, verbose=False,
    )

    engine = BatchedInferenceEngine(llm, n_seq_max=4, prefill_chunk_size=128)
    await engine.start()

    prompts = [
        "What is 2+2? Answer in one word.",
        "Name the capital of France.",
        "What colour is the sky on a clear day?",
        "Is water wet?",
    ]

    async def one(i: int, prompt: str) -> str:
        req = InferenceRequest(
            request_id=f"test-{i}",
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            timeout_s=120.0,
        )
        parts: List[str] = []
        async for tok in engine.generate_stream(req):
            parts.append(tok)
        return "".join(parts).strip()

    t0 = time.monotonic()
    results = await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])
    elapsed = time.monotonic() - t0

    print(f"\n=== Results ({elapsed:.2f}s wall) ===")
    for i, (p, r) in enumerate(zip(prompts, results)):
        print(f"\n[{i}] {p!r}\n    → {r!r}")

    print("\n=== Stats ===")
    for k, v in engine.stats().items():
        print(f"  {k}: {v}")

    await engine.shutdown()
    try:
        llm.close()
    except Exception:
        pass


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_standalone_test())
