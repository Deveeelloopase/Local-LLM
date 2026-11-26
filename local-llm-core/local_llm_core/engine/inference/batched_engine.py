"""
batched_engine.py — Continuous batching for llama-cpp-python 0.3.x.

Verified primitives:
  * llm._ctx.decode(LlamaBatch)     — batched forward pass, returns None
  * LlamaBatch.add_sequence(tokens, seq_id, logits_all)
  * llm._ctx.get_logits()           — flat float32, n_vocab per seq-with-logits
  * llm._ctx.kv_cache_seq_rm(seq_id, -1, -1) → bool
  * llm._ctx.kv_cache_seq_cp(src, dst, -1, -1) — for prefix reuse
  * LlamaContext(*, model=, params=, verbose=)  — rebuild works

Owner-of-context invariant
==========================
When continuous batching is enabled, this engine is the ONLY code path
that touches llama_decode. InferenceEngine passes its `_decode_lock` in
so the direct path (if a caller bypasses us) still serialises against
this engine. The lock is held per decode call, not per request, so
multiple sequences interleave inside one forward pass but two forward
passes never overlap on one context.

Scheduler loop
==============
The loop MUST survive exceptions. The previous version logged and
stopped on any error — every later request hung forever. Now each step
runs in a try/except: a failed step fails only the sequences it touched,
backs off (0.5s→4s), and the loop restarts. `scheduler_alive` in stats()
is the readiness signal.

version = 1.0.0
"""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import logging
import os
import threading
import time
import gc
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# — crash-loop sentinel
# ═══════════════════════════════════════════════════════════════════════════
# Written immediately BEFORE LlamaContext construction and removed
# immediately AFTER success. If the next process start sees this file, it
# knows the previous process died inside the C constructor (which is
# uncatchable from Python) and can fall back to the single-sequence path
# instead of looping on Restart=on-failure.
#
# InferenceEngine._ensure_batched_engine imports _SENTINEL / _sentinel_key
# from this module — do not move them.
from local_llm_core.engine.inference.ctx_guard import (
    sentinel_path as _sentinel_path,
)

# Resolved with the same priority order as inference_engine._default_state_dir()
# ($LLM_STATE_DIR -> /var/lib/local-llm -> XDG), so the writer (here) and the
# reader (InferenceEngine._ensure_batched_engine) can never disagree.
_SENTINEL = _sentinel_path()


def _sentinel_key(model_path: Any, n_ctx: int, n_seq_max: int,
                  kv_unified: bool) -> Dict[str, Any]:
    """Stable fingerprint of a context-build request. A different key
    invalidates a stale `.crashed` marker so a fixed config isn't blocked."""
    return {
        "model": str(model_path or ""),
        "n_ctx": int(n_ctx),
        "n_seq_max": int(n_seq_max),
        "kv_unified": bool(kv_unified),
    }


# ═══════════════════════════════════════════════════════════════════════════
# — retryable decode error
# ═══════════════════════════════════════════════════════════════════════════
class DecodeError(RuntimeError):
    """Raised when llama_decode fails before the user has received any
    output. The outer generate_stream() retries this once; other errors
    (ValueError, RuntimeError from tokenize, etc.) do not retry."""


# ═══════════════════════════════════════════════════════════════════════════
# Low-level batch manipulation — verified against llama-cpp-python 0.3.36
# ═══════════════════════════════════════════════════════════════════════════

def _batch_add_token(
    batch: Any, token: int, pos: int, seq_id: int, logits: bool,
) -> None:
    """Append one token at explicit (pos, seq_id) to a LlamaBatch.

    Bypasses LlamaBatch.add_sequence() because that derives pos internally
    and cannot handle mixed per-seq offsets. Verified: raw llama_batch
    exposes .token[i], .pos[i], .n_seq_id[i], .seq_id[i][0], .logits[i],
    and settable .n_tokens.
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
# Sampling
# ═══════════════════════════════════════════════════════════════════════════

def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix of two token lists.

    Pure-Python loop for short inputs, vectorised for long ones, so a
    4k-token history costs microseconds rather than a Python-level scan."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    if n < 64:
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i
    diff = np.flatnonzero(
        np.asarray(a[:n], dtype=np.int64) != np.asarray(b[:n], dtype=np.int64)
    )
    return int(diff[0]) if diff.size else n

def _sample_logits(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    generated_tokens: List[int],
    rng: np.random.Generator,
) -> int:
    """Sample one token.

    Greedy when temperature <= 1e-6. Otherwise: penalties → temperature →
    top-k (partition, not sort) → top-p → categorical.

    All logits arithmetic stays in float32 until softmax; the final probs
    vector is float32 too. At 152k vocab, float64 costs ~1.2 MB extra per
    step for no accuracy gain at this precision.
    """
    if temperature <= 1e-6:
        logits = logits.astype(np.float32, copy=True)
        _apply_penalties(
            logits, repetition_penalty, presence_penalty, frequency_penalty,
            generated_tokens,
        )
        return int(np.argmax(logits))

    # float32 in-place copy — cheaper than the old float64
    work = logits.astype(np.float32, copy=True)
    _apply_penalties(
        work, repetition_penalty, presence_penalty, frequency_penalty,
        generated_tokens,
    )

    work = work / float(temperature)

    # Top-k via partition — O(n) instead of O(n log n) for a 152k array.
    if top_k > 0 and top_k < work.size:
        kth = np.partition(work, -top_k)[-top_k]
        work[work < kth] = -np.inf

    m = float(work.max())
    if m == -np.inf:
        return 0
    probs = np.exp(work - m)
    denom = float(probs.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return int(np.argmax(work))
    probs /= denom

    if 0.0 < top_p < 1.0:
        # argsort is fine here — it operates on `probs` which we just
        # produced, and it's the only reliable way to get cumulative
        # order. The win was killing the per-step float64 copies.
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(csum, top_p)) + 1
        keep = order[:cutoff]
        mask = np.ones_like(probs, dtype=bool)
        mask[keep] = False
        probs[mask] = 0.0
        s = float(probs.sum())
        if s <= 0:
            return int(np.argmax(work))
        probs /= s

    return int(rng.choice(probs.size, p=probs))


def _apply_penalties(
    logits: np.ndarray,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    generated_tokens: List[int],
) -> None:
    """Apply llama.cpp-style penalties in place, float32.

    Order matters and matches llama.cpp's sampler chain: repetition first
    (multiply/divide logit), then presence (add per unique token), then
    frequency (add per occurrence count).
    """
    if not generated_tokens:
        return
    if repetition_penalty != 1.0:
        for tok in set(generated_tokens):
            if 0 <= tok < logits.size:
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty
    if presence_penalty != 0.0 or frequency_penalty != 0.0:
        # Frequency map for this sample
        freq: Dict[int, int] = {}
        for tok in generated_tokens:
            freq[tok] = freq.get(tok, 0) + 1
        for tok, cnt in freq.items():
            if 0 <= tok < logits.size:
                logits[tok] -= presence_penalty + frequency_penalty * cnt


# ═══════════════════════════════════════════════════════════════════════════
# Per-sequence state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _SequenceState:
    seq_id: int
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    stop_strings: List[str]
    timeout_s: float

    prefill_offset: int = 0
    reused_tokens: int = 0          # prompt tokens served from retained KV
    last_token: Optional[int] = None
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""

    # Incremental detok — one per seq. Replaces the O(n²) full re-detok.
    _utf8_dec: Any = field(default=None, repr=False)
    # Tail buffer for stop-string checking. We hold back the last
    # (max_stop_len - 1) chars of `generated_text` from the client until
    # they either complete a stop string or the gen ends.
    _emit_upto: int = 0

    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=4096))
    finished: bool = False
    cancelled: bool = False
    error: Optional[str] = None

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
# Prefix cache entry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PrefixEntry:
    """A KV snapshot held in a reserved seq_id.

    Reserved seq_ids are kept out of the live allocation pool:
    the reserve range is [n_seq_max, n_seq_max + _PF_RESERVED).
    """
    seq_id: int
    key: bytes
    token_len: int
    last_used: float
    hits: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Batched engine
# ═══════════════════════════════════════════════════════════════════════════

class BatchedInferenceEngine:
    """Continuous batching scheduler on a single Llama instance.

    Callers: `generate_stream(req)`, `generate(req)`, `generate_segmented`,
    `resume_generation`. All decode work is serialised through
    `self._decode_lock` (shared with the parent InferenceEngine if one is
    passed in).
    """

    # Default for the prefix cache. Configurable per-deployment via
    # `inference.continuous_batching.pf_enabled`. Every deployment that
    # runs on tight VRAM should leave it off; larger machines
    # can enable it.
    PF_ENABLED: bool = False
    _PF_RESERVED: int = 2
    _PF_CAPACITY: int = 16

    # ── KV prefix reuse (slot-level) ──────────────────────────────────
    # After a request finishes, its slot keeps the KV for
    # (prompt + generated) instead of being wiped. The next request is
    # routed to the free slot with the longest common token prefix and only
    # prefills the tail. Multi-turn chat re-sends the whole history every
    # turn, so this removes almost all of the re-prefill cost.
    # Per-deployment switch: `inference.continuous_batching.prefix_reuse`.
    PREFIX_REUSE: bool = True
    _PREFIX_REUSE_MIN: int = 8          # below this the trim isn't worth it
    _RETAIN_FRAC_MULTI: float = 0.5     # share of n_ctx idle slots may hold
                                        # when n_seq_max > 1 (shared pool)

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
        decode_lock: Optional[threading.Lock] = None,
        parent_engine: Optional[Any] = None,
        pf_enabled: Optional[bool] = None,
        prefix_reuse: Optional[bool] = None,
    ) -> None:
        # Effective PF-enabled: explicit kwarg > class default. Kept on the
        # instance so callers can flip it without editing source.
        self._pf_enabled = (
            bool(pf_enabled) if pf_enabled is not None else self.PF_ENABLED
        )
        # Prefix reuse state. `_slot_tokens[sid]` is the exact token list whose
        # KV is resident in FREE slot `sid` (positions 0..len-1). A slot that is
        # active, or whose contents are unknown, has no entry.
        self._prefix_reuse = (
            bool(prefix_reuse) if prefix_reuse is not None else self.PREFIX_REUSE
        )
        self._slot_tokens: Dict[int, List[int]] = {}
        self._slot_last_used: Dict[int, float] = {}
        self._reuse_hits = 0
        self._reuse_tokens_saved = 0
        self._reuse_trim_failures = 0

        from llama_cpp import llama_cpp as lc
        from llama_cpp._internals import LlamaContext

        self.llm = llm
        self._initial_n_seq_max = int(n_seq_max)
        self._n_seq_max = int(n_seq_max)
        self._kv_unified = bool(kv_unified)
        self._inter_token_stall_s = float(inter_token_stall_s)
        self._prefill_stall_s = float(prefill_stall_s)
        self._parent = parent_engine  # InferenceEngine, for stats forwarding

        # ── Shared decode lock ─────────────────────────────────────────
        # If the caller owns one, use it. Otherwise create our own. The
        # invariant is: EVERY call that reaches llama_decode runs while
        # holding this lock. `_run_step` and `_rebuild_context` do.
        self._decode_lock = decode_lock if decode_lock is not None else threading.Lock()

        # ── Adaptive capacity (unchanged from prior version) ───────────
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

        # ── Real hardware limits ──────────────────────────────────────
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
        # ── Model geometry for the rebuild feasibility check ────────
        # Conservative defaults, refined from the model if exposed.
        # Used by `_check_rebuild_feasible` to decide whether a second
        # LlamaContext can fit alongside the current one without
        # triggering ggml_abort.
        self._n_layer = _read_int(llm._model, "n_layer", 32)
        self._n_embd = _read_int(llm._model, "n_embd", 4096)
        _n_head = _read_int(llm._model, "n_head", 32)
        self._n_kv_heads = _read_int(
            llm._model, "n_kv_heads", _n_head,
        ) or _n_head
        self._head_dim = max(1, self._n_embd // max(1, _n_head))
        # Bytes-per-element for the KV quant the context was built with.
        try:
            _kvq = str(getattr(llm.context_params, "type_k", 1))
            self._kv_bpe = {"0": 4.0, "1": 2.0, "8": 1.0, "2": 0.5}.get(_kvq, 2.0)
        except Exception:
            self._kv_bpe = 2.0
        # Newer llama.cpp: token_is_eog covers <|im_end|>, <|endoftext|>, EOS.
        self._has_token_is_eog = callable(getattr(llm, "token_is_eog", None))

        if self._n_seq_max >= self._n_batch:
            _clamped = max(1, self._n_batch - 1)
            log.warning(
                "BatchedInferenceEngine: n_seq_max=%d >= n_batch=%d — clamping to %d",
                self._n_seq_max, self._n_batch, _clamped,
            )
            self._n_seq_max = _clamped
            self._initial_n_seq_max = _clamped

        if prefill_chunk_size is None:
            prefill_chunk_size = max(
                32, (self._n_batch - self._n_seq_max) // max(1, self._n_seq_max),
            )
        self._prefill_chunk = min(int(prefill_chunk_size), self._n_batch // 2)

        # Reserve extra seq slots for prefix-cache snapshots, if enabled.
        _extra = self._PF_RESERVED if self._pf_enabled else 0
        _ctx_n_seq_max = self._initial_n_seq_max + _extra

        log.info(
            "BatchedInferenceEngine: n_seq_max=%d (ctx=%d, pf_reserved=%d) "
            "n_ctx=%d n_batch=%d n_vocab=%d eos=%d prefill_chunk=%d",
            self._n_seq_max, _ctx_n_seq_max, _extra,
            self._n_ctx, self._n_batch, self._n_vocab,
            self._eos_token, self._prefill_chunk,
        )

        # ── Rebuild context under lock ────────────────────────────────
        # The rebuild allocates a new LlamaContext. Two different Llama
        # instances must never share a context, and the swap must be
        # atomic with respect to any concurrent decoder.
        self._rebuild_context(lc, LlamaContext, _ctx_n_seq_max)

        # ── Scheduler state ───────────────────────────────────────────
        self._active: Dict[int, _SequenceState] = {}
        self._pending: Deque[_SequenceState] = deque()
        self._free_ids: List[int] = list(range(self._n_seq_max))
        # defense-in-depth dirty-slot set. A slot lands
        # here if kv_cache_seq_rm failed during eviction. _run_step
        # retries the clear before the slot is used for a fresh seq.
        self._dirty_slots: set[int] = set()
        self._new_request_event = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Prefix cache. Reserved seq IDs live above the live range.
        self._pf_cache: "OrderedDict[bytes, _PrefixEntry]" = OrderedDict()
        self._pf_free_ids: List[int] = list(
            range(self._n_seq_max, self._n_seq_max + (_extra or 0))
        )

        self._rng = np.random.default_rng()

        # ── Stats ────────────────────────────────────────────────────
        self._stats_window = int(stats_window)
        self._ttft_samples: Deque[float] = deque(maxlen=stats_window)
        self._decode_tps_samples: Deque[float] = deque(maxlen=stats_window)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_evictions = 0
        self._scheduler_errors = 0
        self._step_count = 0
        # Live rate — 3-5 s sliding window of tokens generated. Used by
        # /v1/metrics as `engine.live_tps` (§5.2). Not a per-request
        # average, and not zero before the first completion.
        self._live_window: Deque[Tuple[float, int]] = deque(maxlen=512)
        self._live_tps: float = 0.0
        self._live_tps_last_calc: float = 0.0

    # ── Context rebuild ─────────────────────────────────────────────
    def _rebuild_context(
        self, lc: Any, LlamaContext: Any, ctx_n_seq_max: int,
    ) -> None:
        """Swap `llm._ctx` for a context sized for batching + PF reserve.

        Two invariants the previous version violated:

          1. **Free before allocate.** The old context is dropped
             *before* the new one is constructed. The previous code
             created `new_ctx` while `old_ctx` was still live, so peak
             VRAM was `weights + 2×KV + 2×compute_buffers`. On a 2 GB
             APU that peak does not fit and llama.cpp calls `ggml_abort`
             — a C-level abort Python cannot catch.

          2. **Refuse when it will not fit.** Before touching anything,
             the feasibility check estimates peak memory and refuses
             with a `RuntimeError` when it exceeds available RAM or the
             device budget. The caller catches this and falls back to
             the single-sequence path.

        — crash-loop sentinel: a file is written to
        `$LLM_STATE_DIR/ctx_rebuild.sentinel` immediately before the
        new `LlamaContext` is constructed and removed immediately after
        success. If this process dies inside the C constructor, the
        next process start finds the file and the parent
        (InferenceEngine._ensure_batched_engine) disables batching
        instead of looping on Restart=on-failure. See the module
        docstring for _SENTINEL / _sentinel_key.

        — the early-return now considers kv_unified. On this
        Vega-APU box (n_seq_max=1) the extra comparison is skipped
        because unified vs non-unified is functionally identical for a
        single sequence — rebuilding just to flip the flag would
        trigger the exact abort we are trying to prevent.

        Called from `__init__` only. `__init__` itself runs in an
        executor (see `InferenceEngine._ensure_batched_engine`) because
        `llama_init_from_model` can take 10+ s when memory is tight,
        and running it on the event loop trips the systemd watchdog.

        Raises `RuntimeError` with a human-readable reason when the
        rebuild is unsafe. A Python exception is recoverable; the
        C-level abort is not.
        """
        # ── early return, considering kv_unified ───────────
        _loaded_n_seq_max = self.llm.context_params.n_seq_max
        if _loaded_n_seq_max >= ctx_n_seq_max:
            # At n_seq_max=1, unified vs non-unified is functionally
            # identical (a single KV cache either way). Only bother
            # rebuilding for kv_unified when the flag would actually
            # change concurrency behaviour.
            if ctx_n_seq_max <= 1:
                return
            _loaded_kv_unified = bool(getattr(
                self.llm.context_params, "kv_unified", False,
            ))
            if _loaded_kv_unified == self._kv_unified:
                return

        # ── Step 1: feasibility check (may raise) ─────────────────
        _refuse = self._check_rebuild_feasible(ctx_n_seq_max)
        if _refuse is not None:
            raise RuntimeError(f"_rebuild_context refused: {_refuse}")

        # ── Step 2: build new context params ──────────────────────
        new_params = lc.llama_context_default_params()
        for f, _ in new_params._fields_:
            if f in (
                "n_seq_max", "kv_unified", "n_outputs_max",
                "n_outputs_max_per_seq",
            ):
                continue
            try:
                setattr(new_params, f, getattr(self.llm.context_params, f))
            except (AttributeError, TypeError):
                pass
        new_params.n_seq_max = ctx_n_seq_max
        new_params.kv_unified = self._kv_unified
        new_params.n_outputs_max = ctx_n_seq_max
        new_params.n_outputs_max_per_seq = 1

        with self._decode_lock:
            # ── Step 3: free old context FIRST ────────────────────
            _old_params = self.llm.context_params
            _old_ctx = self.llm._ctx

            # Drop our reference, then release the C resources explicitly.
            # `del` alone would leak here: LlamaContext.__del__ calls
            # `llama_free`, which does not run until the last Python
            # reference is gone *and* CPython deallocates. Setting to
            # None + explicit close ensures the C memory is freed before
            # the new allocation happens.
            try:
                self.llm._ctx = None
            except Exception:
                pass
            try:
                if hasattr(_old_ctx, "close"):
                    _old_ctx.close()
            except Exception:
                pass
            try:
                del _old_ctx
            except Exception:
                pass
            _old_ctx = None
            gc.collect()

            # ── Step 4: try new context (P0-4 Fix 1 sentinel) ─────
            try:
                # Write sentinel BEFORE the C ctor. If llama.cpp aborts
                # inside LlamaContext(...) the file survives the
                # process death and the next start sees it.
                try:
                    _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
                    _SENTINEL.write_text(json.dumps({
                        **_sentinel_key(
                            getattr(self.llm, "model_path", ""),
                            self._n_ctx,
                            ctx_n_seq_max,
                            self._kv_unified,
                        ),
                        "ts": time.time(),
                        "pid": os.getpid(),
                    }))
                except Exception as _se:
                    log.warning(
                        "_rebuild_context: could not write sentinel: %s",
                        _se,
                    )

                new_ctx = LlamaContext(
                    model=self.llm._model, params=new_params, verbose=False,
                )

                # Success — remove the sentinel so a healthy restart
                # isn't falsely flagged as crashed.
                try:
                    _SENTINEL.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as exc:
                # ── Step 5: rollback to old context ────────────────
                log.error(
                    "_rebuild_context: new context construction failed: %s "
                    "— attempting rollback to old context", exc,
                )
                try:
                    restore_params = lc.llama_context_default_params()
                    for f, _ in restore_params._fields_:
                        if f in (
                            "n_seq_max", "kv_unified", "n_outputs_max",
                            "n_outputs_max_per_seq",
                        ):
                            continue
                        try:
                            setattr(
                                restore_params, f, getattr(_old_params, f),
                            )
                        except (AttributeError, TypeError):
                            pass
                    restore_params.n_seq_max = _old_params.n_seq_max
                    restore_params.kv_unified = getattr(
                        _old_params, "kv_unified", True,
                    )
                    restore_params.n_outputs_max = getattr(
                        _old_params, "n_outputs_max", 1,
                    )
                    restore_params.n_outputs_max_per_seq = 1
                    restored = LlamaContext(
                        model=self.llm._model,
                        params=restore_params, verbose=False,
                    )
                    self.llm._ctx = restored
                    self.llm.context_params = restore_params
                    # Rollback succeeded — clear sentinel so we don't
                    # falsely flag a healthy next start.
                    try:
                        _SENTINEL.unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception as restore_exc:
                    log.critical(
                        "_rebuild_context: rollback failed too (%s) — "
                        "engine is now context-less and unusable until "
                        "restart", restore_exc,
                    )
                raise RuntimeError(
                    f"LlamaContext construction failed: {exc}"
                ) from exc

            self.llm._ctx = new_ctx
            self.llm.context_params = new_params
            self.llm._desired_n_seq_max = ctx_n_seq_max
            self.llm.n_tokens = 0
            if hasattr(self.llm, "input_ids"):
                try:
                    self.llm.input_ids = (
                        self.llm._input_ids.__class__(self.llm._input_ids)
                    )
                except Exception:
                    pass

    def _check_rebuild_feasible(
        self, ctx_n_seq_max: int,
    ) -> Optional[str]:
        """Return a refusal reason if unsafe, else None.

        Two limits apply, and either can refuse:

          1. **System RAM.** The process must be able to fit
             `weights + new_ctx + margin` in available memory.
          2. **Device budget.** The KV cache and compute buffers
             live on the GPU; on an APU that means carve-out + GTT
             (or whatever ModelLoader._profile.vram_budget_mb reports),
             on a dGPU the free VRAM. The old check only saw (1),
             which is why it passed on this Vega-APU box right before
             `ggml_abort`.

        Estimates used:

          * weights_mb — model file size (mmap-backed; competes with
            the new context for resident pages).
          * KV — 2(K+V) × n_layer × n_kv_heads × head_dim × n_ctx × bpe.
            Multiplied by ctx_n_seq_max when kv_unified is False.
          * compute — n_ctx × n_embd × 4 × 4 (activations, attention
            output, FFN intermediate) — rough but conservative.
          * output — n_seq_max × n_vocab × 4.

        Never raises. If psutil is missing the RAM check is skipped; if
        llm.vram_total_mb is unset the device check is skipped. Either
        is a best-effort gate — the sentinel catches what
        we don't see.
        """
        try:
            import psutil
        except ImportError:
            psutil = None  # type: ignore

        # Weights size from the model file if exposed.
        _weights_mb = 0
        _mp = (
            getattr(self.llm, "model_path", None)
            or getattr(self.llm, "_model_path", None)
        )
        if _mp:
            try:
                _p = Path(str(_mp))
                if _p.is_file():
                    _weights_mb = int(_p.stat().st_size / (1024 * 1024))
            except Exception:
                pass

        _n_ctx = self._n_ctx
        _kv_seqs = 1 if self._kv_unified else int(ctx_n_seq_max)
        _kv_bytes = int(
            2 * self._n_layer * self._n_kv_heads * self._head_dim
            * _n_ctx * self._kv_bpe * _kv_seqs
        )
        _kv_mb = int(_kv_bytes / (1024 * 1024))

        _compute_mb = int(
            _n_ctx * self._n_embd * 4 * 4 / (1024 * 1024)
        )
        _output_mb = int(
            ctx_n_seq_max * self._n_vocab * 4 / (1024 * 1024)
        )
        _new_ctx_mb = _kv_mb + _compute_mb + _output_mb
        _margin_mb = 512
        _peak_mb = _weights_mb + _new_ctx_mb + _margin_mb

        # ── Limit 1: system RAM ──────────────────────────────────────
        _free_mb = 0
        if psutil is not None:
            try:
                _free_mb = int(
                    psutil.virtual_memory().available / (1024 * 1024)
                )
            except Exception:
                _free_mb = 0

        if _free_mb > 0 and _peak_mb > _free_mb:
            return (
                f"peak≈{_peak_mb} MB (weights {_weights_mb} + "
                f"new_ctx {_new_ctx_mb} [kv {_kv_mb} + compute "
                f"{_compute_mb} + output {_output_mb}] + margin "
                f"{_margin_mb}) exceeds available RAM {_free_mb} MB. "
                f"ggml_abort would occur otherwise. Options: lower "
                f"num_gpu_layers, lower n_seq_max, disable PF cache, "
                f"or add RAM."
            )

        # ── Limit 2: device budget ───────────────────────────────────
        # InferenceEngine sets llm.vram_total_mb from
        # ModelLoader._profile.vram_budget_mb(_vram_total_mb()). On an
        # APU that is carve-out + GTT; on a dGPU free VRAM. 0 means
        # "not set", which we treat as "no device limit known".
        _device_budget_mb = int(getattr(self.llm, "vram_total_mb", 0) or 0)
        if _device_budget_mb > 0:
            # Weights are resident in device-addressable memory when
            # offloaded; n_gpu_layers=36 means all of them here. Use
            # the conservative weights + ctx sum so we never approve a
            # rebuild that would push the device heap over.
            _device_need_mb = _weights_mb + _new_ctx_mb
            if _device_need_mb > _device_budget_mb:
                return (
                    f"device budget≈{_device_budget_mb} MB "
                    f"(carve-out + GTT / free VRAM) cannot hold "
                    f"weights {_weights_mb} + new_ctx {_new_ctx_mb} "
                    f"= {_device_need_mb} MB. ggml_abort would occur "
                    f"otherwise. Options: lower num_gpu_layers, lower "
                    f"n_seq_max, disable PF cache."
                )

        log.info(
            "_rebuild_context feasibility: peak=%d MB (weights=%d "
            "new_ctx=%d kv=%d compute=%d output=%d kv_unified=%s "
            "kv_seqs=%d) ram_free=%d device_budget=%d — allowed",
            _peak_mb, _weights_mb, _new_ctx_mb, _kv_mb, _compute_mb,
            _output_mb, self._kv_unified, _kv_seqs,
            _free_mb, _device_budget_mb,
        )
        return None

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
        for seq in list(self._active.values()):
            self._fail_seq(seq, "engine shutdown")
        for seq in list(self._pending):
            self._fail_seq(seq, "engine shutdown")
        self._pending.clear()

    @property
    def scheduler_alive(self) -> bool:
        return (
            self._scheduler_task is not None
            and not self._scheduler_task.done()
        )

    # ── Public streaming API ────────────────────────────────────────
    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        """Public entry point. Retries once when llama_decode fails
        before any token has been emitted
        this should be rare, but a transient -1 (GTT pressure, brief
        OOM) shouldn't surface to the user when the stream is still
        empty. Mid-stream failures are not retried (the client already
        has partial content); they propagate so P0-2's error frame
        reaches the client."""
        emitted = False
        for attempt in (1, 2):
            try:
                async for tok in self._generate_stream_once(req):
                    emitted = True
                    yield tok
                return
            except DecodeError as exc:
                if emitted or attempt == 2:
                    raise
                log.warning(
                    "decode failed before first token "
                    "(rid=%s): %s — KV was reset, retrying once",
                    str(getattr(req, "request_id", ""))[:8], exc,
                )

    async def _generate_stream_once(self, req: Any) -> AsyncIterator[str]:
        """Single-attempt token stream for one request. Signature matches
        the InferenceEngine's `generate_stream` so callers are source-
        compatible."""
        rid = str(getattr(req, "request_id", "") or f"req_{id(req)}")
        prompt = str(getattr(req, "prompt", "") or "")
        if not prompt:
            raise ValueError("empty prompt")

        # Tokenize: match `Llama.create_completion` and chunked_prefill.
        try:
            prompt_tokens = list(self.llm.tokenize(
                prompt.encode("utf-8", "ignore"),
                add_bos=False,
                special=True,
            ))
        except Exception as exc:
            raise RuntimeError(f"tokenize failed: {exc}") from exc

        if not prompt_tokens:
            raise ValueError("empty prompt after tokenization")

        seq = _SequenceState(
            seq_id=-1,
            request_id=rid,
            prompt_tokens=prompt_tokens,
            max_tokens=int(getattr(req, "max_tokens", 0) or 0),
            temperature=float(getattr(req, "temperature", 0.7)),
            top_p=float(getattr(req, "top_p", 0.95)),
            top_k=int(getattr(req, "top_k", 40)),
            repetition_penalty=float(getattr(req, "repetition_penalty", 1.0) or 1.0),
            presence_penalty=float(getattr(req, "presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(getattr(req, "frequency_penalty", 0.0) or 0.0),
            stop_strings=list(getattr(req, "stop", []) or []),
            timeout_s=float(getattr(req, "timeout_s", 0.0) or 0.0),
        )
        seq._utf8_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

        self._pending.append(seq)
        self._new_request_event.set()
        self._total_requests += 1

        try:
            while True:
                try:
                    item = await seq.queue.get()
                except asyncio.CancelledError:
                    seq.cancelled = True
                    self._new_request_event.set()
                    raise
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Do NOT pop from _active / return the slot here. The
            # scheduler's _evict_finished() -> _evict_seq() is the
            # single place that runs kv_cache_seq_rm and frees the
            # slot. Popping here skipped the clear, leaving the slot
            # dirty for the next request — which then decoded at pos 0
            # on stale KV and returned -1, producing the alternating
            # "every 2nd request empty" failure.
            if not seq.finished:
                seq.cancelled = True
            self._new_request_event.set()

    async def generate(self, ir: Any) -> Any:
        """Non-streaming convenience — mirrors InferenceEngine.generate."""
        from local_llm_core.engine.inference.inference_engine import InferenceResponse
        t0 = time.perf_counter()
        first_at = None
        parts: List[str] = []
        async for tok in self.generate_stream(ir):
            if first_at is None:
                first_at = time.perf_counter()
            parts.append(tok)
        elapsed = time.perf_counter() - t0
        ttft_ms = (first_at - t0) * 1000.0 if first_at else 0.0
        n = len(parts)
        decode_s = max(0.0, elapsed - (ttft_ms / 1000.0))
        decode_tps = (n - 1) / decode_s if decode_s > 0.05 and n > 1 else 0.0
        prompt_tokens = len(getattr(ir, "prompt", "").split())
        try:
            # Cheap: we already have the ids from _pending; do not retokenize.
            prompt_tokens = max(1, len(self.llm.tokenize(
                str(getattr(ir, "prompt", "")).encode("utf-8", "ignore"),
                add_bos=False, special=True,
            )))
        except Exception:
            pass
        return InferenceResponse(
            request_id=str(getattr(ir, "request_id", "")),
            text="".join(parts),
            tokens_gen=n,
            prompt_tokens=prompt_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_time_s=round(elapsed, 4),
            tps=round(decode_tps, 2),
            finish_reason="stop",
        )

    # ── Segmented / resume — thin wrappers over generate_stream ─────
    # These exist so InferenceEngine.generate_segmented can dispatch
    # here unconditionally when batching is enabled (§1.2).

    async def generate_segmented(
        self,
        request: Any,
        checkpoint_callback: Any,
        segment_size: Optional[int] = None,
    ) -> AsyncIterator[str]:
        from local_llm_core.engine.inference.inference_engine import (
            GenerationCheckpoint,
        )
        seg = int(segment_size or 512)
        full = ""
        n = 0
        segno = 0
        try:
            async for tok in self.generate_stream(request):
                full += tok
                n += 1
                yield tok
                if n > 0 and n % seg == 0:
                    segno += 1
                    try:
                        await checkpoint_callback(GenerationCheckpoint(
                            request_id=getattr(request, "request_id", ""),
                            model_id=getattr(request, "model_id", ""),
                            accumulated_text=full,
                            tokens_generated=n,
                            segment_number=segno,
                            status="in_progress",
                        ))
                    except Exception as exc:
                        log.warning("checkpoint save failed: %s", exc)
            try:
                await checkpoint_callback(GenerationCheckpoint(
                    request_id=getattr(request, "request_id", ""),
                    model_id=getattr(request, "model_id", ""),
                    accumulated_text=full,
                    tokens_generated=n,
                    segment_number=segno + 1,
                    status="complete",
                ))
            except Exception as exc:
                log.warning("final checkpoint save failed: %s", exc)
        except (asyncio.CancelledError, GeneratorExit):
            try:
                await checkpoint_callback(GenerationCheckpoint(
                    request_id=getattr(request, "request_id", ""),
                    model_id=getattr(request, "model_id", ""),
                    accumulated_text=full,
                    tokens_generated=n,
                    segment_number=segno,
                    status="interrupted",
                ))
            except Exception as exc:
                log.warning("interruption checkpoint failed: %s", exc)
            raise

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
        from local_llm_core.engine.inference.inference_engine import (
            InferenceRequest,
        )
        r = InferenceRequest(
            prompt=checkpoint.accumulated_text,
            model_id=checkpoint.model_id,
            max_tokens=extra_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
            request_id=f"{checkpoint.request_id}_resume",
        )
        async for tok in self.generate_segmented(r, checkpoint_callback):
            yield tok

    # ── Scheduler loop ──────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        """Drive one decode step per iteration.

        MUST survive exceptions: a failed step fails only the seqs it
        touched, and the loop continues. The previous version logged and
        stopped, hanging every later request forever.
        """
        backoff = 0.0
        while not self._shutdown_event.is_set():
            try:
                self._admit_pending()

                if not self._active and not self._pending:
                    self._new_request_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._new_request_event.wait(), timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                if not self._active:
                    try:
                        await asyncio.wait_for(
                            self._new_request_event.wait(), timeout=0.1,
                        )
                    except asyncio.TimeoutError:
                        pass
                    self._new_request_event.clear()
                    continue

                await self._run_step()
                await asyncio.sleep(0)
                backoff = 0.0
            except asyncio.CancelledError:
                return
            except Exception:
                self._scheduler_errors += 1
                backoff = min(4.0, max(0.5, backoff * 2 or 0.5))
                log.exception(
                    "BatchedInferenceEngine scheduler step failed "
                    "(errors=%d, backoff=%.1fs) — continuing",
                    self._scheduler_errors, backoff,
                )
                await asyncio.sleep(backoff)

    def _admit_pending(self) -> None:
        while self._pending and self._free_ids:
            seq = self._pending.popleft()
            if seq.cancelled or seq.finished:
                continue
            sid, reuse = self._claim_slot(seq)
            seq.seq_id = sid
            seq.prefill_offset = reuse
            seq.reused_tokens = reuse
            self._active[sid] = seq
            if reuse:
                self._reuse_hits += 1
                self._reuse_tokens_saved += reuse
            log.info(
                "admit seq_id=%d rid=%s prompt_len=%d reused=%d",
                sid, seq.request_id[:8], seq.prompt_len, reuse,
            )

    # ── KV prefix reuse ────────────────────────────────────────────
    def _claim_slot(self, seq: _SequenceState) -> Tuple[int, int]:
        """Pick a free slot for `seq`; return (slot_id, reused_prefix_len).

        Ranking (max wins):
          1. longest common token prefix with the slot's retained KV,
          2. when nothing matches: prefer an EMPTY slot so other slots keep
             their caches for future prefix hits,
          3. then the least-recently-used cached slot.
        A slot whose reuse is 0 is left untouched here: `_run_step` wipes it
        (prefill_offset == 0 -> "fresh") before its first decode."""
        if not self._prefix_reuse or not self._slot_tokens:
            sid = self._free_ids.pop(0)
            self._slot_tokens.pop(sid, None)
            return sid, 0

        best_sid = -1
        best_key: Optional[Tuple[int, int, float]] = None
        best_common = 0
        for sid in self._free_ids:
            cached = self._slot_tokens.get(sid)
            common = _common_prefix_len(cached, seq.prompt_tokens) if cached else 0
            key = (
                common,
                0 if cached else 1,
                -self._slot_last_used.get(sid, 0.0),
            )
            if best_key is None or key > best_key:
                best_sid, best_key, best_common = sid, key, common

        self._free_ids.remove(best_sid)
        cached = self._slot_tokens.pop(best_sid, None)
        # Always re-decode at least the last prompt token: that is what
        # produces the logits for the first generated token.
        reuse = min(best_common, seq.prompt_len - 1)
        if cached and reuse >= self._PREFIX_REUSE_MIN:
            if self._trim_slot(best_sid, reuse):
                return best_sid, reuse
        return best_sid, 0

    def _trim_slot(self, sid: int, keep: int) -> bool:
        """Drop KV positions >= `keep` from slot `sid`. False -> caller must
        treat the slot as dirty (it is queued for a full wipe here).

        llama.cpp returns False when a partial removal is impossible
        (recurrent / SWA memory). Removing a whole sequence never fails."""
        ok = False
        if self._decode_lock.acquire(blocking=False):
            try:
                ok = bool(self.llm._ctx.kv_cache_seq_rm(sid, keep, -1))
            except Exception as exc:
                log.warning("kv_cache_seq_rm(seq=%d, %d..) trim: %s", sid, keep, exc)
            finally:
                self._decode_lock.release()
        if not ok:
            self._reuse_trim_failures += 1
            self._dirty_slots.add(sid)
        return ok

    def _retain_budget(self) -> int:
        if self._n_seq_max <= 1:
            return self._n_ctx
        return max(1, int(self._n_ctx * self._RETAIN_FRAC_MULTI))

    def _drop_slot_cache(self, sid: int) -> None:
        self._slot_tokens.pop(sid, None)
        self._slot_last_used.pop(sid, None)
        try:
            self.llm._ctx.kv_cache_seq_rm(sid, -1, -1)
        except Exception as exc:
            log.warning("kv_cache_seq_rm(seq=%d) drop: %s", sid, exc)
            self._dirty_slots.add(sid)

    def _enforce_retain_budget(self) -> None:
        """Keep idle slots from starving active sequences of KV cells:
        evict least-recently-used retained caches until under budget."""
        budget = self._retain_budget()
        total = sum(len(v) for v in self._slot_tokens.values())
        while total > budget and self._slot_tokens:
            victim = min(
                self._slot_tokens,
                key=lambda s: self._slot_last_used.get(s, 0.0),
            )
            total -= len(self._slot_tokens[victim])
            self._drop_slot_cache(victim)

    @staticmethod
    def _coerce_logits(raw: Any, expected_n: int) -> Optional[np.ndarray]:
        import ctypes as _ct
        if raw is None:
            return None
        try:
            if isinstance(raw, np.ndarray):
                return raw.astype(np.float32, copy=False).ravel()
            if isinstance(raw, (list, tuple)):
                return np.asarray(raw, dtype=np.float32).ravel()
            if isinstance(raw, _ct._Pointer) or hasattr(raw, "contents"):
                try:
                    arr_type = _ct.c_float * int(expected_n)
                    ptr = _ct.cast(raw, _ct.POINTER(arr_type))
                    return np.ctypeslib.as_array(ptr.contents).astype(
                        np.float32, copy=False,
                    ).ravel()
                except (ValueError, TypeError):
                    return None
            return None
        except Exception:
            return None

    async def _run_step(self) -> None:
        """One iteration: prefill + decode in one batch, sample, emit."""
        self._evict_finished()
        if not self._active:
            return

        # ── capture dirty slots before Phase 1 ────────
        # A slot is "fresh" when it has never decoded anything for the
        # current seq (prefill_offset == 0). Its KV may still hold data
        # from the previous occupant if that occupancy was freed by a
        # path that didn't call kv_cache_seq_rm (defense in depth — Fix
        # A removes the one such path we know of). Slots whose clear
        # failed during eviction are also included.
        fresh_ids = {
            s.seq_id for s in self._active.values()
            if not (s.finished or s.cancelled) and s.prefill_offset == 0
        }
        stale_ids = fresh_ids | self._dirty_slots
        self._dirty_slots.clear()

        entries: List[tuple] = []
        logits_seqs: List[tuple] = []
        logits_index = 0

        n_reserved = len(self._active)
        prefill_budget = max(1, self._n_batch - n_reserved)

        # ── Phase 1: prefill chunks ──────────────────────────────────
        for seq in list(self._active.values()):
            if seq.finished or seq.cancelled or seq.prefill_done:
                continue
            remaining = seq.prompt_len - seq.prefill_offset
            take = min(remaining, self._prefill_chunk, prefill_budget)
            if take <= 0:
                continue
            chunk = seq.prompt_tokens[seq.prefill_offset:seq.prefill_offset + take]
            base_pos = seq.prefill_offset
            seq_will_finish = (take == remaining)
            last_slot: Optional[int] = None
            for i, tok in enumerate(chunk):
                is_last = (i == take - 1)
                wants_logits = is_last and seq_will_finish
                pos = base_pos + i
                entries.append((seq, tok, pos, wants_logits))
                if wants_logits:
                    last_slot = logits_index
                    logits_index += 1
            if seq_will_finish and last_slot is not None:
                logits_seqs.append((seq, last_slot))
            seq.prefill_offset += take
            prefill_budget -= take

        # ── Phase 2: one decode token per active seq ──────────────────
        for seq in list(self._active.values()):
            if seq.finished or seq.cancelled or not seq.prefill_done:
                continue
            if seq.last_token is None:
                continue
            entries.append((seq, seq.last_token, seq.prefill_offset, True))
            logits_seqs.append((seq, logits_index))
            logits_index += 1
            seq.prefill_offset += 1

        if not entries:
            return

        # prefill_offset was already advanced above. If batch construction
        # fails, KV and offsets disagree -> same recovery as a decode failure
        # (fail everything, wipe KV, drop retained caches).
        try:
            from llama_cpp._internals import LlamaBatch
            batch = LlamaBatch(
                n_tokens=len(entries), embd=0, n_seq_max=self._initial_n_seq_max,
            )
            for seq, tok, pos, logits in entries:
                _batch_add_token(batch, tok, pos, seq.seq_id, logits)
        except Exception as exc:
            self._handle_decode_error(exc, entries)
            return

        # ── Decode, under the shared lock, in an executor ────────────
        loop = asyncio.get_running_loop()

        _failed_clears: List[int] = []

        def _locked_decode() -> None:
            with self._decode_lock:
                # clear every slot that's about to decode
                # its first token. Slots whose clear failed are re-
                # marked so the next step retries.
                for sid in stale_ids:
                    try:
                        self.llm._ctx.kv_cache_seq_rm(sid, -1, -1)
                    except Exception as e:
                        log.warning(
                            "kv_cache_seq_rm(seq=%d) pre-decode: %s",
                            sid, e,
                        )
                        _failed_clears.append(sid)
                self.llm._ctx.decode(batch)

        try:
            await loop.run_in_executor(None, _locked_decode)
        except Exception as exc:
            self._handle_decode_error(exc, entries)
            return

        for sid in _failed_clears:
            self._dirty_slots.add(sid)

        self._step_count += 1
        if not logits_seqs:
            return

        # ── Read whole logits buffer ─────────────────────────────────
        n_logits = len(logits_seqs)
        n_vocab = self._n_vocab
        expected = n_logits * n_vocab
        try:
            raw = self.llm._ctx.get_logits()
        except Exception as exc:
            self._handle_decode_error(exc, entries)
            return
        whole = self._coerce_logits(raw, expected)
        if whole is None or whole.size < expected:
            log.error(
                "get_logits() unreadable or short (size=%s expected=%d)",
                None if whole is None else whole.size, expected,
            )
            for seq, _ in logits_seqs:
                self._fail_seq(seq, "logits unavailable")
            return

        # ── Sample per seq ───────────────────────────────────────────
        for seq, out_idx in logits_seqs:
            if seq.finished or seq.cancelled:
                continue
            lo = out_idx * n_vocab
            hi = lo + n_vocab
            slice_ = whole[lo:hi]
            if slice_.size < n_vocab:
                self._fail_seq(seq, "short logits slice")
                continue
            try:
                next_tok = _sample_logits(
                    slice_,
                    temperature=seq.temperature,
                    top_k=seq.top_k,
                    top_p=seq.top_p,
                    repetition_penalty=seq.repetition_penalty,
                    presence_penalty=seq.presence_penalty,
                    frequency_penalty=seq.frequency_penalty,
                    generated_tokens=seq.generated_tokens,
                    rng=self._rng,
                )
            except Exception as exc:
                self._fail_seq(seq, f"sample failed: {exc}")
                continue

            seq.last_token = next_tok
            seq.generated_tokens.append(next_tok)
            seq.last_token_ts = time.monotonic()
            self._total_tokens += 1
            self._live_window.append((seq.last_token_ts, 1))

            if seq.first_token_ts is None:
                seq.first_token_ts = seq.last_token_ts
                self._ttft_samples.append(
                    seq.first_token_ts - seq.enqueued_ts,
                )

            self._emit_detok_delta(seq)

            if self._should_stop(seq, next_tok):
                self._finish_seq(seq)

    # ── Detokenization ──────────────────────────────────────────────

    def _emit_detok_delta(self, seq: _SequenceState) -> None:
        """Emit the delta since the last emit, honouring stop-string tail.

        Uses the per-seq incremental UTF-8 decoder so emoji and CJK pass
        through correctly even when a token is a partial codepoint. Holds
        back `tail_hold` characters so a stop string that spans tokens
        never leaks to the client.
        """
        try:
            raw = self.llm.detokenize([seq.generated_tokens[-1]], special=False)
        except Exception as exc:
            log.warning("detok failed seq=%d: %s", seq.seq_id, exc)
            return

        delta = seq._utf8_dec.decode(raw)
        if not delta:
            return
        seq.generated_text += delta

        # How many trailing characters must we withhold?
        if seq.stop_strings:
            max_stop = max((len(s) for s in seq.stop_strings if s), default=0)
            hold = max(0, max_stop - 1)
        else:
            hold = 0

        end = len(seq.generated_text) - hold
        if end <= seq._emit_upto:
            return
        chunk = seq.generated_text[seq._emit_upto:end]
        seq._emit_upto = end
        if not chunk:
            return
        try:
            seq.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            log.warning(
                "seq=%d queue full — client too slow, dropping delta",
                seq.seq_id,
            )

    def _flush_emit_tail(self, seq: _SequenceState) -> None:
        """Emit any withheld tail once the generation is decided to be over."""
        chunk = seq.generated_text[seq._emit_upto:]
        seq._emit_upto = len(seq.generated_text)
        if chunk:
            try:
                seq.queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def _should_stop(self, seq: _SequenceState, next_tok: int) -> bool:
        # EOG covers EOS + <|im_end|> + <|endoftext|>
        if self._has_token_is_eog:
            try:
                if self.llm.token_is_eog(next_tok):
                    return True
            except Exception:
                pass
        if next_tok == self._eos_token:
            return True
        if seq.max_tokens > 0 and seq.gen_count >= seq.max_tokens:
            return True
        if seq.total_len >= self._n_ctx - 4:
            return True
        if seq.stop_strings:
            for s in seq.stop_strings:
                if s and s in seq.generated_text:
                    return True
        now = time.monotonic()
        if seq.timeout_s > 0 and (now - seq.enqueued_ts) > seq.timeout_s:
            return True
        if (seq.last_token_ts is not None
                and (now - seq.last_token_ts) > self._inter_token_stall_s):
            return True
        return False

    # ── Eviction / completion ───────────────────────────────────────

    def _evict_finished(self) -> None:
        for seq in [s for s in self._active.values() if s.finished or s.cancelled]:
            self._evict_seq(seq)

    def _evict_seq(self, seq: _SequenceState) -> None:
        if seq.seq_id < 0:
            return
        sid = seq.seq_id

        # Exactly the tokens whose KV is resident: positions 0..prefill_offset-1.
        # (The last sampled token has not been decoded yet, so it is excluded
        # automatically because prefill_offset only counts decoded positions.)
        retained = 0
        if self._prefix_reuse and seq.error is None and sid < self._n_seq_max:
            resident = (seq.prompt_tokens + seq.generated_tokens)[:seq.prefill_offset]
            if resident and len(resident) <= self._retain_budget():
                self._slot_tokens[sid] = resident
                self._slot_last_used[sid] = time.monotonic()
                retained = len(resident)

        if not retained:
            # Failed / cancelled-before-anything / reuse disabled: wipe the slot.
            self._slot_tokens.pop(sid, None)
            _cleared = False
            try:
                self.llm._ctx.kv_cache_seq_rm(sid, -1, -1)
                _cleared = True
            except Exception as exc:
                log.warning("kv_cache_seq_rm(seq=%d): %s", sid, exc)
            if not _cleared and sid < self._n_seq_max:
                # Clear failed: queue the slot for a retry before it is reused.
                self._dirty_slots.add(sid)

        self._active.pop(sid, None)
        if sid < self._n_seq_max and sid not in self._free_ids:
            self._free_ids.append(sid)
        self._total_evictions += 1
        if retained:
            self._enforce_retain_budget()
        log.info(
            "evict seq=%d rid=%s gen=%d retained=%d",
            sid, seq.request_id[:8], seq.gen_count, retained,
        )

    def _finish_seq(self, seq: _SequenceState) -> None:
        if seq.finished:
            return
        seq.finished = True
        self._flush_emit_tail(seq)
        try:
            seq.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if seq.first_token_ts is not None and seq.gen_count > 1:
            dt = seq.last_token_ts - seq.first_token_ts
            if dt > 0.05:
                tps = (seq.gen_count - 1) / dt
                self._decode_tps_samples.append(tps)
                # Forward to parent for /v1/metrics and monitor
                if self._parent is not None:
                    try:
                        with self._parent._lk:
                            self._parent._decode_tps_samples.append(tps)
                            self._parent._total_requests += 1
                            self._parent._total_tokens += seq.gen_count
                            if seq.first_token_ts:
                                ttft_ms = (seq.first_token_ts - seq.enqueued_ts) * 1000.0
                                self._parent._latencies.append(ttft_ms)
                    except Exception:
                        pass

    def _fail_seq(
        self, seq: _SequenceState, reason: str,
        exc_type: type = RuntimeError,
    ) -> None:
        if seq.finished or seq.cancelled:
            return
        seq.error = reason
        seq.finished = True
        try:
            seq.queue.put_nowait(exc_type(reason))
        except asyncio.QueueFull:
            pass
        if seq.seq_id >= 0 and seq.seq_id in self._active:
            self._evict_seq(seq)

    def _handle_decode_error(
        self, exc: BaseException, entries: List[tuple],
    ) -> None:
        """Recover from decode() failure.

        Fail EVERY active and pending seq with a `DecodeError` so
        generate_stream's retry wrapper can distinguish a transient
        decode failure (retry once) from other errors (no retry).
        Then clear the KV cache so any future request starts clean.
        """
        log.error("decode() failed: %s — failing all active seqs", exc)
        self._scheduler_errors += 1
        for seq in list(self._active.values()):
            self._fail_seq(seq, f"decode failed: {exc}", DecodeError)
        for seq in list(self._pending):
            self._fail_seq(
                seq, f"decode failed upstream: {exc}", DecodeError,
            )
        self._pending.clear()
        try:
            with self._decode_lock:
                self.llm._ctx.kv_cache_clear()
        except Exception as clear_exc:
            log.error("kv_cache_clear after decode error: %s", clear_exc)
        # Invalidate PF cache — the snapshots now reference cleared KV.
        self._pf_cache.clear()
        # Retained slot caches reference KV that no longer exists.
        self._slot_tokens.clear()
        self._slot_last_used.clear()
        self._dirty_slots.clear()

    # ── Adaptive controller (unchanged behaviour) ────────────────────

    async def _adaptive_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._adaptive_interval_s)
                if self._shutdown_event.is_set():
                    return
                current = self._n_seq_max
                thermal_throttling = False
                if self._thermal_state is not None:
                    try:
                        thermal_throttling = bool(
                            getattr(self._thermal_state, "throttling", False)
                        )
                    except Exception:
                        pass
                if thermal_throttling and self._thermal_override_enabled:
                    new_n = max(self._adaptive_min, current // 2)
                    if new_n != current:
                        self._apply_n_seq_max(new_n)
                        log.warning(
                            "Adaptive: thermal throttle → n_seq_max %d→%d",
                            current, new_n,
                        )
                    continue
                if len(self._ttft_samples) < 8:
                    continue
                recent = list(self._ttft_samples)[-20:]
                avg_ttft_ms = 1000.0 * (sum(recent) / len(recent))
                lo = self._adaptive_target_ttft_ms * 0.80
                hi = self._adaptive_target_ttft_ms * 1.20
                if avg_ttft_ms < lo and current < self._adaptive_max:
                    self._apply_n_seq_max(min(current + 1, self._adaptive_max))
                elif avg_ttft_ms > hi and current > self._adaptive_min:
                    self._apply_n_seq_max(max(current - 1, self._adaptive_min))
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("Adaptive loop error: %s", exc)

    def _apply_n_seq_max(self, new_n: int) -> None:
        new_n = max(1, min(int(new_n), self._initial_n_seq_max))
        if new_n == self._n_seq_max:
            return
        self._n_seq_max = new_n
        active_ids = set(self._active.keys())
        self._free_ids = [i for i in range(new_n) if i not in active_ids]

    # ── Live rate calculation ───────────────────────────────────────

    def _compute_live_tps(self) -> float:
        now = time.monotonic()
        if now - self._live_tps_last_calc < 0.5:
            return self._live_tps
        while self._live_window and now - self._live_window[0][0] > 5.0:
            self._live_window.popleft()
        if not self._live_window:
            self._live_tps = 0.0
        else:
            span = max(0.5, now - self._live_window[0][0])
            total = sum(c for _, c in self._live_window)
            self._live_tps = total / span
        self._live_tps_last_calc = now
        return self._live_tps

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        ttfts = list(self._ttft_samples)
        tps_s = list(self._decode_tps_samples)
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
            "scheduler_alive": self.scheduler_alive,
            "step_count": self._step_count,
            "avg_ttft_ms": round(1000.0 * (sum(ttfts) / len(ttfts)), 1) if ttfts else 0.0,
            "avg_decode_tps": round(sum(tps_s) / len(tps_s), 2) if tps_s else 0.0,
            "decode_tps": round(sum(tps_s) / len(tps_s), 2) if tps_s else 0.0,
            "avg_tps": round(sum(tps_s) / len(tps_s), 2) if tps_s else 0.0,
            "live_tps": round(self._compute_live_tps(), 2),
            "eos_token": self._eos_token,
            "n_vocab": self._n_vocab,
            "initial_n_seq_max": self._initial_n_seq_max,
            "adaptive_enabled": self._adaptive_enabled,
            "adaptive_min": self._adaptive_min,
            "adaptive_max": self._adaptive_max,
            "pf_cache_entries": len(self._pf_cache),
            "pf_cache_enabled": self._pf_enabled,
            "prefix_reuse_enabled": self._prefix_reuse,
            "prefix_reuse_hits": self._reuse_hits,
            "prefix_reuse_tokens_saved": self._reuse_tokens_saved,
            "prefix_reuse_trim_failures": self._reuse_trim_failures,
            "prefix_retained_slots": len(self._slot_tokens),
        }
