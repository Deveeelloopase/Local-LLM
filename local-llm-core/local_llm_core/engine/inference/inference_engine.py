from __future__ import annotations

# engine/inference/inference_engine.py — Production Inference Engine  v1.0.0
# ===========================================================================
# Unified GGUF + BitNet dispatcher.  Config‑injected, library‑safe.
# ===========================================================================

import asyncio
import atexit
import gc
import heapq
import inspect
import logging
import os
import threading
import time
import uuid
import json
import contextlib
from pathlib import Path
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable

from local_llm_core.engine.scheduling.scheduler import Scheduler

from local_llm_core.engine.inference.chunked_prefill import chunked_prefill
from local_llm_core.engine.inference.stream_guards import (
    UNLIMITED_TOKENS, RepetitionGuard, LoopGuardV2, SolicitationGuard,
    ThinkStripper, resolve_max_tokens, slide_window, strip_think_text,
)

log = logging.getLogger(__name__)

def _default_state_dir() -> Path:
    """State directory for pid, crash log, swap, kv-cache, and similar.

    Priority:
      1. $LLM_STATE_DIR (explicit override)
      2. /var/lib/local-llm (systemd StateDirectory=local-llm)
      3. $XDG_STATE_HOME/local-llm, defaulting to ~/.local/state/local-llm

    Creates the dir with mode 0700 if it doesn't exist. (L-06)
    """
    explicit = os.environ.get("LLM_STATE_DIR", "").strip()
    if explicit:
        p = Path(explicit)
    else:
        var_lib = Path("/var/lib/local-llm")
        chosen: Optional[Path] = None
        try:
            var_lib.mkdir(parents=True, exist_ok=True)
            if os.access(var_lib, os.W_OK | os.R_OK):
                chosen = var_lib
        except (OSError, PermissionError):
            chosen = None
        if chosen is None:
            xdg = Path(
                os.environ.get("XDG_STATE_HOME")
                or (Path.home() / ".local" / "state")
            )
            chosen = xdg / "local-llm"
        p = chosen
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p

# ── Apply SPECTRE at module‑import time (gguf TL1/TL2 for BitNet) ───────────
try:
    from local_llm_core.compatibility.transformers_compat import StateFileReader as _SFR
    _SFR.apply_spectre_pythonpath()
    del _SFR
except Exception:
    pass

# =============================================================================
# Prometheus metrics — module‑level singleton
# =============================================================================
def _make_prom() -> Dict[str, Any]:
    try:
        from prometheus_client import Counter, Gauge, Histogram

        def _h(name, doc, buckets):
            try: return Histogram(name, doc, buckets=buckets)
            except ValueError: return None

        def _c(name, doc, labels=None):
            try: return Counter(name, doc, labels or [])
            except ValueError: return None

        def _g(name, doc):
            try: return Gauge(name, doc)
            except ValueError: return None

        prom = {
            "ttft":   _h("llm_ttft_ms", "TTFT ms",
                         (5, 20, 50, 100, 250, 500, 1000, 2000, 5000)),
            "tps":    _h("llm_tps", "Tokens/sec",
                         (0.5, 1, 2, 5, 10, 20, 50, 100, 200)),
            "errors": _c("llm_errors_total", "Inference errors", ["reason"]),
            "oom":    _c("llm_oom_total", "OOM events"),
            "cache":  _c("llm_ctx_cache_hits", "Context cache hits"),
            "req":    _c("llm_requests_total", "Requests"),
            "cb":     _g("llm_circuit_open", "1=open 0=closed"),
        }
        return {k: v for k, v in prom.items() if v is not None}
    except Exception:
        return {}

# Module‑level singleton — not reset per‑instance
_PROM: Dict[str, Any] = _make_prom()

_DONE = object()
_PROGRESS = object()   # producer→consumer keep-alive sentinel (resets stall clock)
_FINISH_REASON_STOP = "stop"
_FINISH_REASON_LENGTH = "length"
_FINISH_REASON_ERROR = "error"

# Canonical structured markers — shared by the engine and the client UI.
# The `⚡` prefix makes accidental model output collisions effectively
# impossible (it does not appear in any tokenizer's default chat vocab).
_TRUNCATION_MARKER_FMT = "\n\n[⚡TRUNCATED:limit-reached:tokens={n}]"
_INTERRUPTION_MARKER_FMT = "\n\n[⚡INTERRUPTED:client-cancel:tokens={n}]"
_STALL_MARKER_FMT = "\n\n[⚡STALLED:phase={phase}:tokens={n}]"

def _make_truncation_marker(tokens: int) -> str:
    return _TRUNCATION_MARKER_FMT.format(n=int(tokens))

def _make_interruption_marker(tokens: int) -> str:
    return _INTERRUPTION_MARKER_FMT.format(n=int(tokens))

def _make_stall_marker(phase: str, tokens: int) -> str:
    return _STALL_MARKER_FMT.format(phase=phase, n=int(tokens))

class StreamStallError(Exception):
    """Raised when a stream stalls (no first token, or gap between tokens)."""
    def __init__(self, phase: str, elapsed_s: float,
                 budget_s: float, partial_tokens: int) -> None:
        super().__init__(
            f"Stream stalled during {phase}: {elapsed_s:.1f}s without progress "
            f"(budget {budget_s:.1f}s, {partial_tokens} tokens received)"
        )
        self.phase = phase
        self.elapsed_s = elapsed_s
        self.budget_s = budget_s
        self.partial_tokens = partial_tokens


class PrefillTracker:
    """EWMA of prefill throughput, used to size the first-token timeout.

    Deliberately an EWMA — hardware behaviour here is dominated by linear
    factors (prompt length, cache state), so one scalar is the right
    abstraction. alpha=0.05 gives a ~14-observation half-life, smoothing
    GC/swap jitter while still tracking real degradation.

    The EWMA is persisted to ~/.cache/local-llm/perf.json so a restart
    doesn't reset to the (wrong) seed value. The audit's L-02 finding:
    the old seed of 15 tps was ~3× the real measured value (4.5–5.4
    tps), so the first-token budget for a cold prefill was 3× too
    tight and the server fell back to the 120 s floor.
    """
    def __init__(
        self,
        alpha: float,
        seed_tps: float = 4.0,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._alpha = alpha
        self._tps = seed_tps
        self._n = 0
        self._persist_path = persist_path
        if persist_path is not None:
            self._load_persisted()

    def _load_persisted(self) -> None:
        try:
            import json
            p = self._persist_path
            if p is None or not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            tps = float(data.get("tps", 0))
            n = int(data.get("n", 0))
            if 0.1 < tps < 500.0 and n > 0:
                self._tps = tps
                self._n = n
                log.info(
                    "PrefillTracker: loaded persisted EWMA tps=%.2f (n=%d) from %s",
                    tps, n, p,
                )
        except Exception as exc:
            log.warning("PrefillTracker: could not load %s: %s", self._persist_path, exc)

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        try:
            import json
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"tps": self._tps, "n": self._n}),
                encoding="utf-8",
            )
            tmp.replace(self._persist_path)
        except Exception as exc:
            log.warning("PrefillTracker: persist failed: %s", exc)

    def update(self, prompt_tokens: int, elapsed_s: float) -> None:
        if prompt_tokens <= 0 or elapsed_s <= 0:
            return
        sample = prompt_tokens / elapsed_s
        # A KV-prefix-cache hit returns the first token in ~30 ms, so
        # prompt_tokens / elapsed_s can look like 100,000+ tps. Feeding
        # that into the EWMA makes the tracker believe this box prefills
        # at hundreds of tps, which shrinks the per-request stall budget
        # to the 120 s floor and fires StreamStallError on the very next
        # real (cache-miss) prompt. Drop samples above 500 tps — that is
        # far above any CPU/hybrid prefill we will ever see and clearly a
        # cache artefact.
        if sample > 500.0:
            return
        if self._n == 0:
            self._tps = sample
        else:
            self._tps = (1.0 - self._alpha) * self._tps + self._alpha * sample
        self._n += 1
        # Persist every N observations; not every token, to avoid SSD
        # wear. 10 observations is ~30-60 s of real traffic on this box.
        if self._n % 10 == 0:
            self._persist()

    @property
    def tps(self) -> float:
        return self._tps

    @property
    def has_observations(self) -> bool:
        return self._n > 0

# =============================================================================
# Helpers
# =============================================================================
def _strip_think(text: str) -> str:
    """Strip <think>…</think> blocks — manual scan, ReDoS‑safe."""
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start + 7)
        if end == -1:
            text = text[:start].strip()
            break
        text = text[:start] + text[end + 8:]
    return text.strip()


def _is_bitnet_wrapper(obj: Any) -> bool:
    """
    Duck‑type check for BitNetInferenceWrapper.
    Avoids isinstance() which breaks when the class is imported from
    different paths (e.g. direct import vs. engine.bitnet import).
    """
    return (
        callable(getattr(obj, "agenerate", None))
        and hasattr(obj, "backend_info")
        and hasattr(obj, "shutdown")
    )


def _is_native_async_backend(llm: Any) -> bool:
    """
    Duck‑type check for a backend that exposes a REAL async‑generator
    agenerate() method (e.g. BitNetCppBackend, BitNetRunInferenceBackend).
    """
    return inspect.isasyncgenfunction(getattr(llm, "agenerate", None))

class LeanLlama:
    """
    Version-probed wrapper: returns a Llama subclass that skips the scores
    copy in save_state/load_state, IFF the installed llama-cpp-python
    performs that copy.

    The scores buffer is n_tokens × n_vocab × 4 bytes. At 1,407 tokens
    and a 152k vocab that is 855 MB per cached prefix. LlamaRAMCache
    counts only `llama_state_size` (~110 MB for the same state), so the
    "128 MB" cache silently holds ~965 MB per entry.

    After this fix, save_state returns LlamaState.scores as an empty
    float32 array. `_scores` (the underlying C buffer) is untouched —
    it is only refreshed by eval(), and every code path that samples
    calls eval() first. `logprobs` would break, but this server does
    not expose logprobs.

    Version safety:
      * `probe_llama_cpp()` inspects the installed save_state source.
        If it does NOT contain self._scores.copy(), this returns the
        stock class — no override.
      * LlamaState.__init__ parameter names are read at class creation
        so a future version that adds/removes fields (e.g. drops
        `seed`) still works.
    """

    _cache: dict = {}

    @classmethod
    def get(cls, base_cls):
        key = id(base_cls)
        if key in cls._cache:
            return cls._cache[key]

        try:
            from local_llm_core.accel.capabilities import probe_llama_cpp
            caps = probe_llama_cpp()
        except Exception:
            cls._cache[key] = base_cls
            return base_cls

        if not caps.save_state_copies_scores:
            # Installed version is either fixed upstream or old enough
            # that the bug never existed. Use it as-is.
            cls._cache[key] = base_cls
            return base_cls

        # Import once, at class-creation time
        import ctypes as _ctypes
        import inspect as _inspect
        import importlib as _importlib
        import numpy as _np
        from llama_cpp import llama_cpp as _lc

        # ── Locate LlamaState across versions / forks ────────────────────
        # The class has lived in at least four places across llama-cpp-python
        # versions and forks:
        #   * 0.3.36+       : llama_cpp.llama  (defined inline as NamedTuple)
        #   * some earlier  : llama_cpp.llama_cpp
        #   * re-exported   : llama_cpp.llama_types  (in some builds)
        #   * top level     : llama_cpp (rare)
        # Try every known location; fall back to introspecting the return
        # annotation of Llama.save_state. Return stock Llama if we cannot
        # find it — LeanLlama then degrades to the safe no-op.
        _LlamaState = None
        for _mod_name, _attr in (
            ("llama_cpp.llama",         "LlamaState"),
            ("llama_cpp.llama_cpp",     "LlamaState"),
            ("llama_cpp.llama_types",   "LlamaState"),
            ("llama_cpp",               "LlamaState"),
        ):
            try:
                _m = _importlib.import_module(_mod_name)
                _candidate = getattr(_m, _attr, None)
                if _candidate is not None:
                    _LlamaState = _candidate
                    break
            except Exception:
                continue

        if _LlamaState is None:
            # Introspect the annotation on Llama.save_state — most versions
            # declare `-> LlamaState` as a forward ref resolved against the
            # llama module.
            try:
                _ann = base_cls.save_state.__annotations__.get("return")
                if isinstance(_ann, str):
                    _llama_mod = _importlib.import_module("llama_cpp.llama")
                    _LlamaState = getattr(_llama_mod, _ann, None)
                elif callable(_ann):
                    _LlamaState = _ann
            except Exception:
                pass

        if _LlamaState is None:
            log.warning(
                "LeanLlama: LlamaState not found in any known location "
                "(llama_cpp.llama / llama_cpp.llama_cpp / llama_cpp.llama_types / "
                "llama_cpp, nor via save_state's return annotation). "
                "Falling back to stock Llama — save_state will copy scores. "
                "Upgrade llama-cpp-python or report the LlamaState location."
            )
            cls._cache[key] = base_cls
            return base_cls

        try:
            _ls_params = frozenset(_inspect.signature(_LlamaState.__init__).parameters)
        except (ValueError, TypeError):
            _ls_params = frozenset({"scores", "input_ids", "n_tokens",
                                    "llama_state", "llama_state_size", "seed"})

        class _Lean(base_cls):
            def save_state(self):
                state_size = _lc.llama_state_get_size(self._ctx.ctx)
                llama_state = (_ctypes.c_uint8 * int(state_size))()
                n_bytes = _lc.llama_state_get_data(
                    self._ctx.ctx, llama_state, state_size,
                )
                if int(n_bytes) > int(state_size):
                    raise RuntimeError("Failed to copy llama state data")
                llama_state_compact = (_ctypes.c_uint8 * int(n_bytes))()
                _ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))

                kwargs = {
                    "scores": _np.empty(0, dtype=_np.float32),
                    "input_ids": self.input_ids.copy(),
                    "n_tokens": self.n_tokens,
                    "llama_state": bytes(llama_state_compact),
                    "llama_state_size": n_bytes,
                }
                if "seed" in _ls_params:
                    kwargs["seed"] = self._seed
                return _LlamaState(**kwargs)

            def load_state(self, state):
                # Skip the two scores-restore lines from stock load_state.
                # _requires_eval = True guarantees the next eval() repopulates
                # _scores before any sampling reads it. Safe for any sampler.
                self.input_ids = state.input_ids.copy()
                self.n_tokens = state.n_tokens
                self._requires_eval = True
                if hasattr(state, "seed"):
                    self._seed = state.seed
                state_size = state.llama_state_size
                LLamaStateArrayType = _ctypes.c_uint8 * state_size
                llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)
                if (
                    _lc.llama_state_set_data(self._ctx.ctx, llama_state, state_size)
                    != state_size
                ):
                    raise RuntimeError("Failed to set llama state data")

        _Lean.__name__ = "LeanLlama"
        _Lean.__qualname__ = "LeanLlama"
        cls._cache[key] = _Lean
        log.info(
            "LeanLlama active: stripping scores from save_state "
            "(llama-cpp-python %s, saves n_tokens×vocab×4 B per cached prefix)",
            caps.version,
        )
        return _Lean

def _normalise_format(fmt: str) -> str:
    """Normalise model format aliases to canonical names."""
    fmt = fmt.lower().strip()
    _aliases = {
        "gguf": "gguf",
        "llama": "gguf",
        "llama_cpp": "gguf",
        "llamacpp": "gguf",
        "bitnet": "bitnet",
        "bit_net": "bitnet",
        "transformers": "transformers",
        "hf": "transformers",
        "huggingface": "transformers",
        "safetensors": "transformers",
    }
    return _aliases.get(fmt, fmt)

def _read_llm_n_ctx(llm) -> int:
    """Return the loaded context length of a llama-cpp-like object.

    `llama_cpp.Llama` exposes `n_ctx` and `n_ctx_train` as **methods**, not
    attributes. A naive `getattr(llm, "n_ctx", 0)` returns a bound method
    (truthy) which then fails `int()`, silently falling through to a
    hard-coded default and producing bogus headroom math. This helper
    accepts both shapes and returns 0 if nothing usable was found so the
    caller can apply its own default explicitly.
    """
    for src in (llm, getattr(llm, "ctx", None)):
        if src is None:
            continue
        for attr in ("n_ctx", "_n_ctx"):
            v = getattr(src, attr, None)
            if v is None:
                continue
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                return iv
    return 0

# =============================================================================
# BoundedThreadPoolExecutor
# =============================================================================
class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with a hard cap on pending work items."""

    def __init__(self, max_workers: int, max_queue: int = 256,
                 thread_name_prefix: str = "bounded") -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._sem   = threading.Semaphore(max_queue)
        self._depth = 0
        self._lk    = threading.Lock()

    def submit(self, fn, *args, **kwargs) -> Future:
        if not self._sem.acquire(blocking=False):
            raise RuntimeError("Executor queue full — server overloaded, retry shortly")
        with self._lk:
            self._depth += 1

        def _wrap():
            try:
                return fn(*args, **kwargs)
            finally:
                self._sem.release()
                with self._lk:
                    self._depth = max(0, self._depth - 1)

        return super().submit(_wrap)

    def queue_stats(self) -> dict:
        with self._lk:
            return {"pending": self._depth, "capacity": self._sem._value}


# =============================================================================
# AsyncTokenBuffer — thread‑safe blocking‑producer → async‑consumer bridge
# =============================================================================
class AsyncTokenBuffer:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        maxsize: int = 4096,
        timeout_s: float = 0.0,
        *,
        first_token_budget_s: float = 120.0,
        inter_token_stall_s: float = 60.0,
    ) -> None:
        self._loop    = loop
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.cancel   = threading.Event()
        # Normalize: None / negative means "no overall deadline".
        if timeout_s is None:
            timeout_s = 0.0
        self._timeout = float(timeout_s)
        self.future: Optional[Future] = None
        self._first_token_budget = first_token_budget_s
        self._inter_token_stall  = inter_token_stall_s

    def sync_put(self, item, put_timeout: float = 30.0) -> None:
        if self.cancel.is_set():
            raise RuntimeError("Consumer cancelled — aborting producer")
        for attempt in range(8):
            try:
                self._q.put_nowait(item)
                return
            except asyncio.QueueFull:
                if self.cancel.is_set():
                    raise RuntimeError("Consumer cancelled — aborting producer")
                if attempt < 7:
                    time.sleep(0.0002)
                    continue
                break
        if self.cancel.is_set():
            raise RuntimeError("Consumer cancelled — aborting producer")
        try:
            asyncio.run_coroutine_threadsafe(
                self._q.put(item), self._loop
            ).result(timeout=put_timeout)
        except Exception as exc:
            raise RuntimeError(f"sync_put failed: {exc}") from exc

    async def __aiter__(self) -> AsyncIterator[str]:
        timeout = self._timeout
        unlimited = timeout is None or timeout <= 0
        deadline = float("inf") if unlimited else time.monotonic() + timeout
        start_ts  = time.monotonic()
        last_item = start_ts
        got_first = False
        emitted   = 0
        try:
            while True:
                now = time.monotonic()

                if not unlimited and now >= deadline:
                    self.cancel.set()
                    raise StreamStallError(
                        phase="overall_timeout",
                        elapsed_s=now - start_ts,
                        budget_s=timeout,
                        partial_tokens=emitted,
                    )

                budget = self._first_token_budget if not got_first else self._inter_token_stall
                phase_label = "prefill" if not got_first else "generation"
                idle = now - last_item
                if idle >= budget:
                    log.warning(
                        "Stream %s stall: %.1fs without progress (budget %.1fs, %d tok so far)",
                        phase_label, idle, budget, emitted,
                    )
                    self.cancel.set()
                    raise StreamStallError(
                        phase=phase_label,
                        elapsed_s=idle,
                        budget_s=budget,
                        partial_tokens=emitted,
                    )

                if self.future and self.future.done():
                    exc = self.future.exception()
                    # Drain anything the producer queued before it
                    # finished.  Previously the producer could enqueue
                    # tokens AND _DONE and then return; the consumer's
                    # `break` here silently dropped the tokens.
                    while not self._q.empty():
                        _it = self._q.get_nowait()
                        if _it is _DONE or _it is _PROGRESS:
                            continue
                        if isinstance(_it, Exception):
                            raise _it
                        emitted += 1
                        yield _it
                    if exc:
                        raise exc
                    break

                try:
                    item = await asyncio.wait_for(
                        self._q.get(),
                        timeout=min(2.0, max(budget - idle, 0.1)) if not unlimited else 2.0,
                    )
                    last_item = time.monotonic()
                    if item is _PROGRESS:
                        continue           # prefill chunk done; alive, not a token
                    if not got_first:
                        got_first = True
                except asyncio.TimeoutError:
                    continue

                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                emitted += 1
                yield item
        finally:
            self.cancel.set()
            if self.future and not self.future.done():
                # don't block the caller. Mid-prefill C
                # calls cannot be interrupted from Python, so waiting here
                # just delays releasing the slot. Instead, spawn a
                # background task that reports when the producer exits;
                # the caller returns immediately and the slot's
                # request_id can be reclaimed.
                _fut = self.future

                # Bind both `fut` and the cancel timestamp as default args
                # so the closure captures the *values* at definition time.
                # Reading `_fut_cancel_ts` from the enclosing scope would
                # work today because it's assigned immediately below, but
                # any future edit that reassigns or moves that assignment
                # would silently change the elapsed-time report. Default
                # args make the capture semantics explicit.
                async def _bg_wait(
                        fut=_fut,
                        cancel_ts=time.monotonic(),
                ) -> None:
                    try:
                        import concurrent.futures as _cf
                        loop = asyncio.get_running_loop()
                        _, not_done = await loop.run_in_executor(
                            None,
                            lambda: _cf.wait([fut], timeout=30.0),
                        )
                        if not_done:
                            log.warning(
                                "Producer thread still running 30s after "
                                "cancel — slot may remain busy until the "
                                "current llama.cpp call returns."
                            )
                        else:
                            log.info(
                                "Producer thread exited %.1fs after cancel.",
                                time.monotonic() - cancel_ts,
                            )
                    except Exception:
                        pass

                try:
                    asyncio.get_running_loop().create_task(
                        _bg_wait(), name="token_buf_wait",
                    )
                except RuntimeError:
                    # No running loop (GC-time close) — nothing to do.
                    pass

# =============================================================================
# CircuitBreaker
# =============================================================================
class CircuitBreaker:
    """Sliding‑window circuit breaker with single‑probe half‑open state."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self._threshold     = failure_threshold
        self._recovery      = recovery_timeout
        self._state         = "closed"
        self._failures      = 0
        self._opened        = 0.0
        self._probe_in_flight = False
        self._lk            = threading.Lock()

    def can_proceed(self) -> bool:
        with self._lk:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened >= self._recovery:
                    if not self._probe_in_flight:
                        self._state = "half-open"
                        self._probe_in_flight = True
                        log.info("CircuitBreaker → HALF‑OPEN (single probe)")
                        return True
                    return False
                return False
            if self._state == "half-open":
                return self._probe_in_flight
            return False

    def record_success(self) -> None:
        with self._lk:
            if self._state == "half-open" and self._probe_in_flight:
                self._probe_in_flight = False
                self._state = "closed"
                self._failures = 0
                log.info("CircuitBreaker → CLOSED (probe succeeded)")
                if _PROM.get("cb"):
                    _PROM["cb"].set(0)
            elif self._state == "closed":
                self._failures = 0

    def record_failure(self) -> None:
        with self._lk:
            if self._state == "half-open":
                self._probe_in_flight = False
                self._state = "open"
                self._opened = time.monotonic()
                log.error("CircuitBreaker → OPEN (probe failed)")
                if _PROM.get("cb"):
                    _PROM["cb"].set(1)
                return
            if self._state == "closed":
                self._failures += 1
                if self._failures >= self._threshold:
                    self._state = "open"
                    self._opened = time.monotonic()
                    log.error("CircuitBreaker → OPEN (%d failures)", self._failures)
                    if _PROM.get("cb"):
                        _PROM["cb"].set(1)

    def state_name(self) -> str:
        """Return the current breaker state as a plain string.

        Read-only, unlike ``can_proceed()`` which transitions open →
        half-open when the recovery window has elapsed. Safe to call from
        pre-flight capacity checks without perturbing breaker
        behaviour. Values: "closed", "open", "half-open".
        """
        with self._lk:
            return self._state

    def stats(self) -> dict:
        with self._lk:
            return {
                "state":       self._state,
                "failures":    self._failures,
                "threshold":   self._threshold,
                "recovery_secs": round(
                    max(0.0, self._recovery - (time.monotonic() - self._opened)), 1
                ) if self._state == "open" else 0.0,
                "probe_in_flight": self._probe_in_flight,
            }


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================
class TokenBucket:
    def __init__(self, rps: float = 10.0) -> None:
        self.rps     = rps
        self._tokens = rps
        self._last   = time.monotonic()
        self._lk     = threading.Lock()

    def take(self) -> bool:
        with self._lk:
            now     = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.rps, self._tokens + self.rps * elapsed)
            self._last   = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class RateLimiter:
    def __init__(self, default_rps: float = 10.0, max_keys: int = 1000) -> None:
        self._rps      = default_rps
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lk       = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lk:
            if key not in self._buckets:
                if len(self._buckets) >= self._max_keys:
                    self._buckets.popitem(last=False)
                self._buckets[key] = TokenBucket(self._rps)
            else:
                self._buckets.move_to_end(key)
            return self._buckets[key].take()

    def set_rps(self, key: str, rps: float) -> None:
        with self._lk:
            self._buckets[key] = TokenBucket(rps)
            self._buckets.move_to_end(key)

    def stats(self) -> dict:
        with self._lk:
            return {"keys": len(self._buckets), "default_rps": self._rps,
                    "max_keys": self._max_keys}


# =============================================================================
# AutoTuner
# =============================================================================
class AutoTuner:
    def __init__(self, engine: "InferenceEngine", interval_s: float = 300.0,
                 max_batch: int = 8) -> None:
        self._engine   = engine
        self._interval = interval_s
        self._max_batch = max_batch
        self._history: deque = deque(maxlen=20)
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        try:
            self._task = asyncio.get_running_loop().create_task(
                self._loop(), name="auto_tuner"
            )
        except RuntimeError:
            pass  # no running loop yet — called from sync context

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._tune()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("AutoTuner: %s", exc)

    async def _tune(self) -> None:
        s   = self._engine.stats()
        tps = s.get("avg_tps", 0.0)
        bat = s.get("batch_size", self._engine.max_batch)
        self._history.append({"t": time.monotonic(), "tps": tps, "batch": bat})
        if len(self._history) < 4:
            return
        pts = list(self._history)[-10:]
        if len(pts) < 2:
            return
        t0  = pts[0]["t"]
        xs  = [p["t"] - t0 for p in pts]
        ys  = [p["tps"] for p in pts]
        if len(set(xs)) < 2:
            return
        n   = len(xs); sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
        denom = n*sxx - sx*sx
        if abs(denom) < 1e-6:
            return
        slope = ((n*sxy - sx*sy) / denom) * 60
        new_b = bat
        if slope < -0.5 and bat > 1:
            new_b = bat - 1
            log.info("AutoTuner: TPS declining → batch %d→%d", bat, new_b)
        elif slope > 0.5 and bat < self._max_batch:
            new_b = bat + 1
            log.info("AutoTuner: TPS rising → batch %d→%d", bat, new_b)
        if new_b != bat and self._engine._token_mgr:
            await self._engine._token_mgr.resize(new_b)


# =============================================================================
# NGramCache
# =============================================================================
class NGramCache:
    def __init__(self, n: int = 3, max_size: int = 10_000) -> None:
        self._n   = n
        self._max = max_size
        self._cache: OrderedDict = OrderedDict()
        self._hits   = 0
        self._misses = 0
        self._lk     = threading.Lock()

    def observe(self, tokens: list) -> None:
        if len(tokens) < self._n:
            return
        with self._lk:
            for i in range(len(tokens) - self._n + 1):
                key = tuple(tokens[i:i + self._n])
                self._cache[key] = self._cache.pop(key, 0) + 1
                if len(self._cache) > self._max:
                    self._cache.popitem(last=False)

    def lookup(self, context: list) -> Optional[str]:
        if len(context) < self._n - 1:
            return None
        key = tuple(context[-(self._n - 1):])
        with self._lk:
            best_key = None
            best_cnt = 0
            for k, cnt in self._cache.items():
                if k[:self._n - 1] == key and cnt > best_cnt:
                    best_cnt = cnt
                    best_key = k
            if best_key:
                self._hits += 1
                return best_key[-1]
        self._misses += 1
        return None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# =============================================================================
# RequestQueue
# =============================================================================
class RequestQueue:
    def __init__(self, maxsize: int = 256) -> None:
        self._max  = maxsize
        self._heap: list = []
        self._lk   = threading.Lock()

    def push(self, req: "InferenceRequest") -> None:
        with self._lk:
            if len(self._heap) >= self._max:
                raise RuntimeError("Request queue full — server overloaded")
            heapq.heappush(self._heap, req)

    def pop(self) -> Optional["InferenceRequest"]:
        with self._lk:
            return heapq.heappop(self._heap) if self._heap else None

    def stats(self) -> dict:
        with self._lk:
            return {"size": len(self._heap), "capacity": self._max}


# =============================================================================
# InferenceRequest / InferenceResponse
# =============================================================================
@dataclass
class InferenceRequest:
    prompt:     str   = ""
    model_id:   str   = ""
    # 0 = unlimited (generate until EOS / stop / repetition-guard / cancel).
    # Any positive value is honoured exactly. See effective_max_tokens.
    max_tokens: int   = 0
    temperature: float = 0.7
    top_p:      float = 0.95
    top_k:      int   = 40
    stop:       list  = field(default_factory=list)
    stream:     bool  = True
    priority:   int   = 0
    timeout_s:  float = 600.0
    # Engine-level <think> stripping. True by default so no caller ever
    # leaks reasoning tags. Set False only when the caller (e.g. the Web
    # UI's Thinking ON mode) explicitly wants the raw reasoning stream.
    strip_think: bool = True
    # Id of the user-facing request this (possibly segment) request belongs
    # to.  Progress/heartbeat state is keyed on it.  Empty → request_id.
    root_id: str = ""
    request_id: str   = field(default_factory=lambda: str(uuid.uuid4()))
    _arrival:   float = field(default_factory=time.monotonic, repr=False)
    # ── Constrained decoding (optional) ─────────────────────────────────
    # Either a raw GBNF string (fast path) or a JSON-schema dict (slower
    # — goes through LlamaGrammar.from_json_schema once, cached).
    # Both are consumed by `_llama_stream`; other backends ignore them.
    grammar: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None

    def __lt__(self, other: "InferenceRequest") -> bool:
        if self.priority != other.priority:
            return self.priority > other.priority
        return self._arrival < other._arrival

    @property
    def effective_max_tokens(self) -> int:
        # <=0 / None → UNLIMITED_TOKENS (2**31-1). Any positive integer is
        # returned unchanged. The engine still bounds each backend call by
        # available context headroom, and slides the window on overflow, so
        # this value is a *logical* ceiling, not a physical one.
        return resolve_max_tokens(self.max_tokens)


@dataclass
class InferenceResponse:
    request_id:   str
    text:         str
    tokens_gen:   int
    prompt_tokens: int
    ttft_ms:      float
    total_time_s: float
    tps:          float
    finish_reason: str  = "stop"
    context_cache_hit:    bool = False
    model_thinking_active: bool = False
    thinking_tokens:      int  = 0
    has_thinking:         bool = False
    thinking_type:        str  = ""


# =============================================================================
# Checkpoint dataclass (for segmented generation)
# =============================================================================
@dataclass
class GenerationCheckpoint:
    request_id: str
    model_id: str
    accumulated_text: str
    tokens_generated: int
    segment_number: int
    status: str  # "in_progress", "interrupted", "complete"
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# =============================================================================
# InferenceEngine — Unified GGUF + BitNet dispatcher
# =============================================================================
class InferenceEngine:
    """
    Production inference engine routing GGUF and BitNet models through a
    single unified model map.  Config‑injected, library‑safe.
    """

    def __init__(
        self,
        model_loader,
        *,
        config: Optional[Dict[str, Any]] = None,
        max_batch_size: Optional[int] = None,
        target_tps:     Optional[float] = None,
        bitnet_config:  Optional[Dict[str, Any]] = None,
        bitnet_profile_configured: bool = False,
        thermal_state: Optional[Any] = None,
        bitnet_enabled: bool = True,
        # kernels_enabled: controls whether UniversalKernelManager loads
        # and monkey-patches PyTorch ops with HIP/CUDA kernels.
        #
        # on this deployment (gfx902 Picasso/Raven, GGUF inference
        # via llama.cpp) the patched kernels are not used at all — the
        # llama.cpp path doesn't go through PyTorch. They are extra native
        # code loaded into the process for no measurable benefit, on a GPU
        # that is not an officially supported ROCm target. Any native
        # crash from this path lands in /var/lib/local-llm/crash.log
        # (faulthandler target), NOT in the server log — a silent failure
        # mode. So this defaults to False here.
        #
        # The `kernels.enabled: false` config key was already present but
        # the manager was being initialised unconditionally. This kwarg
        # makes that config key actually take effect.
        kernels_enabled: bool = False,
    ) -> None:

        # ── Config normalisation ─────────────────────────────────────────
        if config is None:
            config = {}
        try:
            from local_llm_core.engine.utils.config_loader import InferenceConfig
            cfg_model = (
                InferenceConfig(**config) if isinstance(config, dict) else config
            )
            self._cfg = cfg_model.model_dump()
        except Exception as exc:
            log.warning("Config validation failed: %s; using defaults", exc)
            from local_llm_core.engine.utils.config_loader import InferenceConfig
            self._cfg = InferenceConfig().model_dump()

        # ── Slot lifecycle config ────────────────────────────────────
        self._cfg.setdefault("slot_timeout_s", 300)
        self._cfg.setdefault("slot_health_check_interval_s", 60)

        # ── Core ─────────────────────────────────────────────────────────
        self.loader    = model_loader
        self.max_batch = max_batch_size or self._cfg["max_batch_size"]
        self._target_tps = target_tps or self._cfg["target_tps"]

        self.rate_limiter = RateLimiter(
            default_rps=self._cfg["rate_limit_rps"],
            max_keys=self._cfg["rate_limit_max_keys"],
        )

        # ── BitNet loader ──────────────────────────────────────────────
        from local_llm_core.engine.utils.config_loader import BitNetSystemConfig
        bitnet_config = bitnet_config or {}
        bitnet_sys = BitNetSystemConfig(**bitnet_config)

        self._bitnet_state_model_id: Optional[str] = None
        self._bitnet_profile_configured: bool = bitnet_profile_configured
        self._bitnet_enabled: bool = bitnet_enabled

        self._bitnet_loader = self._init_bitnet_loader(bitnet_sys)

        # ── Unified model map ─────────────────────────────────────────────
        self._unified_models: Dict[str, Any] = {}
        self._models_lock    = threading.RLock()
        self._lifecycle_manager: Optional[Any] = None

        # ── Async state ───────────────────────────────────────────────────
        self._stop_event = asyncio.Event()
        self._aimd_task: Optional[asyncio.Task] = None
        self._lk         = threading.RLock()

        # ── Counters ──────────────────────────────────────────────────────
        self._total_requests = 0
        self._total_tokens   = 0
        self._total_errors   = 0
        self._oom_count      = 0
        self._latencies:    deque = deque(maxlen=self._cfg["metrics_history_size"])
        self._tps_samples:  deque = deque(maxlen=self._cfg["metrics_history_size"])
        # separate decode-only samples. `_tps_samples` records
        # end-to-end tps (prefill + decode); the audit showed that at
        # qwen3-8b's measured 4.5 tok/s prefill, end-to-end tps reads
        # as 2.4–2.75 (decode-only is 2.4–2.75; end-to-end divides by
        # (prefill + decode) so it's artificially low by 30-40%).
        # `_decode_tps_samples` records the honest decode-only number.
        self._decode_tps_samples: deque = deque(maxlen=self._cfg["metrics_history_size"])
        # New prompt tokens per call (excludes KV-cache hits). Used by
        # the client-side latency estimator (F-06) which currently assumes
        # 1200 tokens per prefill; the real number is ~184 after the
        # first call thanks to KV prefix caching.
        self._avg_new_prompt_samples: deque = deque(maxlen=self._cfg["metrics_history_size"])

        # ── Drive pipeline ────────────────────────────────────────────────
        try:
            from local_llm_core.engine.pipelines.drive_pipeline import (
                ConcurrencyTokenManager, ContextCache, ThroughputController,
            )
            self._ctx_cache  = ContextCache(
                max_bytes    = self._cfg["context_cache_bytes"],
                prefix_len   = self._cfg["context_cache_prefix_len"],
            )
            self._tput_ctrl  = ThroughputController(
                target_tps   = self._target_tps,
                min_batch    = self._cfg["aimd_min_batch"],
                max_batch    = self.max_batch,
            )
            self._token_mgr  = ConcurrencyTokenManager(initial=self.max_batch)
            self._drive_wired = True
            log.info("Drive pipeline active")
        except Exception as exc:
            self._ctx_cache  = None
            self._tput_ctrl  = None
            self._token_mgr  = None
            self._drive_wired = False
            log.warning("Drive pipeline unavailable: %s", exc)

        _perf_path_str = self._cfg.get("prefill_persist_path") or ""
        _perf_path = (
            Path(_perf_path_str) if _perf_path_str
            else Path.home() / ".cache" / "local-llm" / "perf.json"
        )
        self._prefill_tracker = PrefillTracker(
            alpha=float(self._cfg.get("prefill_ewma_alpha", 0.05)),
            seed_tps=float(self._cfg.get("prefill_seed_tps", 4.0)),
            persist_path=_perf_path,
        )
        # root_id → {"phase", "done", "total", "reused", "t"}.
        # Read by the API layer to emit heartbeat frames.  Written from
        # producer threads; single-key dict assignment is atomic under the GIL.
        self._progress: Dict[str, dict] = {}
        self._ngram = NGramCache(
            n=self._cfg["ngram_n"], max_size=self._cfg["ngram_max_size"]
        )
        # ── Grammar cache ─────────────────────────────────────────────────
        # Compiled LlamaGrammar objects are immutable and reusable. On a
        # server receiving the same schema repeatedly (JSON tool calls,
        # classification prompts, etc.) this removes almost all parsing
        # overhead. Bounded LRU so a misbehaving client can't OOM the box.
        self._grammar_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._grammar_cache_max: int = 64
        self._grammar_cache_lock = threading.Lock()
        self._queue   = RequestQueue(maxsize=self._cfg["request_queue_maxsize"])
        self._inflight = 0

        self._breaker = CircuitBreaker(
            failure_threshold = self._cfg["circuit_breaker_failure_threshold"],
            recovery_timeout  = self._cfg["circuit_breaker_recovery_s"],
        )
        self._tuner = AutoTuner(
            self,
            interval_s = self._cfg["autotune_interval_s"],
            max_batch  = self.max_batch,
        )

        # ── Thread pool ───────────────────────────────────────────────────
        cpu = os.cpu_count() or 4
        n_parallel_cap = self._cfg.get("n_parallel")
        if n_parallel_cap:
            workers = max(2, min(cpu, int(n_parallel_cap)))
            log.info(
                "Executor workers=%d (n_parallel_cap=%s, floor=2) — "
                "ensures one stuck thread cannot starve the pool",
                workers, n_parallel_cap,
            )
        else:
            workers = min(cpu, max(self.max_batch, self._cfg["min_executor_workers"]))

        self._executor = BoundedThreadPoolExecutor(
            max_workers        = workers,
            max_queue          = self._cfg["executor_max_queue"],
            thread_name_prefix = "llm_inf",
        )

        # ── Async gates ───────────────────────────────────────────────────
        self._inference_gate = asyncio.Semaphore(1)
        self._slot_semaphore = asyncio.Semaphore(self.max_batch)
        self._slot_holders: Dict[int, Dict[str, Any]] = {}
        self._slot_lock = asyncio.Lock()
        self._slot_health_task: Optional[asyncio.Task] = None

        self._semaphore      = asyncio.Semaphore(self.max_batch)  # kept for compatibility
        self._gate_waiters   = 0
        self._gate_lock      = asyncio.Lock()

        # ── Scheduler ─────────────────────────────────────────────────
        # The Scheduler reads its config keys at top level, not from a
        # nested "inference" dict. The previous code wrapped everything
        # under {"inference": {...}}, so SchedulerConfig validation
        # failed silently and every AIMD parameter fell back to its
        # hardcoded default (target_tps=7.0). See B7 in the diagnosis.
        _profile = getattr(model_loader, "_profile", {}) or {}
        _sched_cfg = {
            "aimd_target_tps":  float(self._target_tps),
            "aimd_min_batch":   int(self._cfg.get("aimd_min_batch", 1)),
            "aimd_max_batch":   int(self.max_batch),
            "aimd_interval_s":  float(self._cfg.get("aimd_interval_s", 5.0)),
            "aimd_hysteresis":  int(self._cfg.get("aimd_hysteresis", 2)),
            "aimd_high_margin": float(self._cfg.get("aimd_high_margin", 0.95)),
            "aimd_low_margin":  float(self._cfg.get("aimd_low_margin", 0.85)),
        }
        self._scheduler = Scheduler(
            profile=_profile,
            config=_sched_cfg,
            thermal_state=thermal_state,
        )
        # Assert the scheduler actually got the configured value.
        try:
            _sched_target = float(getattr(self._scheduler, "_aimd_target_tps", 0.0))
            if abs(_sched_target - float(self._target_tps)) > 0.01:
                log.error(
                    "Scheduler target_tps mismatch: configured=%.2f "
                    "scheduler=%.2f — config nesting is still wrong",
                    float(self._target_tps), _sched_target,
                )
        except Exception:
            pass

        # ── Kernel manager ────────────────────────────────────────────────
        # Only initialise if the caller asked for it (kernels_enabled) —
        # see the kwarg docstring for the rationale. When off, the log
        # line "Custom kernels backend activated: hip" and
        # "PyTorch ops patched with custom kernels" will NOT
        # appear on startup, and the process carries no extra native code.
        self._kernel_manager = None
        if kernels_enabled:
            try:
                from local_llm_core.engine.kernels.manager import UniversalKernelManager
                self._kernel_manager = UniversalKernelManager()
                self._kernel_manager.initialize(
                    gpu_infos=getattr(model_loader, "_gpus", None)
                )
                self._kernel_manager.patch_pytorch()
            except Exception as exc:
                self._kernel_manager = None
                log.warning("Kernel manager unavailable: %s", exc)
        else:
            log.info(
                "UniversalKernelManager disabled (kernels.enabled=false) — "
                "GGUF path does not use PyTorch kernels"
            )

        # ── Faulthandler ────────────────────────────────────────────────
        self._faulthandler_fd = None
        self._setup_faulthandler()

        # ── Auto‑register BitNet from state file ──────────────────────
        self._auto_register_bitnet_from_state()

        # track running tasks for cancellation
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._running_tasks_lock = asyncio.Lock()

        # ── Segmentation config ──────────────────────────────────────────
        self._segment_size = self._cfg.get("segment_size", 512)
        self._health_check_interval_segments = self._cfg.get("health_check_interval_segments", 1)
        self._max_segments = self._cfg.get("max_segments", 200)

        log.info("InferenceEngine v1.0.0 initialised (GGUF + BitNet, unified)")

        # Retained as None for backward compat with any external code
        # that still does `getattr(engine, '_batch_scheduler', None)`.
        # Neither field is read anywhere in this class. Real concurrency
        # is provided by the shared-weight process pool at the server
        # level, not here.
        self._batch_scheduler = None
        self._cb_cfg = None

        # ── Continuous batching (opt-in) ──────────────────────────
        # There has never been a local_llm_core.engine.inference.continuous_batching
        # module. llama-cpp-python does not expose a public multi-request
        # API for it either (its Llama object is single-request by design).
        # Real concurrency is provided by the shared-weight process pool in
        # local-llm-server/serving/pool — see concurrency.mode: process_pool
        # in config.yaml. This block is preserved only to give a clear
        # message if an operator re-enables the (unimplemented) config flag.
        _cb_cfg = (self._cfg.get("continuous_batching") or {})
        if _cb_cfg.get("enabled", False):
            log.warning(
                "inference.continuous_batching.enabled=true is IGNORED — "
                "continuous batching is not implemented in llama-cpp-python "
                "and no local module provides it. Real concurrency is handled "
                "by concurrency.mode: process_pool at the server level "
                "(local-llm-server/serving/pool). Set "
                "inference.continuous_batching.enabled: false to silence."
            )

    # ── BitNet loader initialization ──────────────────────────
    def _init_bitnet_loader(self, bitnet_sys: Any) -> Any:
        from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader

        if not self._bitnet_enabled:
            log.info("BitNet disabled by config — skipping BitNet loader")
            return None

        if self._bitnet_profile_configured:
            log.info(
                "BitNetLoader: BitNet profile configured — skipping "
                "from_state_file() (state file is a zero‑config fallback "
                "only; not needed when a profile is present)."
            )
            return BitNetLoader(config=bitnet_sys)

        try:
            loader = BitNetLoader.from_state_file()
            log.info("BitNetLoader: initialised from state file (zero‑config)")
            return loader
        except (FileNotFoundError, ValueError) as exc:
            _default_id = getattr(bitnet_sys, "default_model_id", None) or "bitnet-gguf"
            log.warning(
                "BitNet state file not available (%s). "
                "Using direct‑path GGUF loading if a 'bitnet-gguf' profile is configured. "
                "Run `local-llm-setup` to enable the zero‑config bitnet.cpp path.",
                exc,
            )
        except Exception as exc:
            log.warning(
                "BitNetLoader.from_state_file() unexpected error: %s; "
                "falling back to manual config",
                exc,
            )
        return BitNetLoader(config=bitnet_sys)

    # ── Auto‑register BitNet from state file ─────────────────
    def _auto_register_bitnet_from_state(self) -> None:
        if not self._bitnet_enabled:
            log.warning("_auto_register_bitnet_from_state: BitNet disabled — skipping")
            return

        if self._bitnet_profile_configured:
            log.info(
                "_auto_register_bitnet_from_state: a BitNet profile is "
                "already configured with autoload:true — skipping state‑file "
                "fallback registration entirely to prevent phantom duplicate."
            )
            return

        try:
            from local_llm_core.compatibility.transformers_compat import StateFileReader
            state = StateFileReader.load()
            if not state or not state.get("verify_passed"):
                return
            model_name = state.get("model_name") or "bitnet-state"
            state_gguf = str(state.get("gguf_path") or "")

            with self._models_lock:
                if model_name in self._unified_models:
                    return

                for m in self._unified_models.values():
                    if getattr(m, "model_type", None) == "bitnet":
                        log.info(
                            "_auto_register_bitnet_from_state: BitNet model "
                            "already in _unified_models — skipping '%s'",
                            model_name,
                        )
                        return

                if state_gguf:
                    def _norm(p: str) -> str:
                        if not p: return ""
                        try:
                            from pathlib import Path
                            return str(Path(p).expanduser().resolve())
                        except Exception:
                            return str(p)

                    norm_state = _norm(state_gguf)
                    for m in self._unified_models.values():
                        cfg = getattr(m, "config", None)
                        mp  = _norm(getattr(cfg, "model_path", "") or "")
                        if mp and mp == norm_state:
                            log.info(
                                "_auto_register_bitnet_from_state: state "
                                "gguf_path matches loaded model path — skipping '%s'",
                                model_name,
                            )
                            return

            log.info(
                "Auto‑registering BitNet model from state file: %s", model_name
            )
            self._bitnet_state_model_id = model_name
        except Exception as exc:
            log.warning("_auto_register_bitnet_from_state: %s", exc)

    # ── Faulthandler setup ───────────────────────────────────
    def _setup_faulthandler(self) -> None:
        import faulthandler as _fh
        try:
            _fh.enable()
            _crash_log = _default_state_dir() / "crash.log"
            self._faulthandler_fd = open(_crash_log, "a")
            _fh.enable(file=self._faulthandler_fd, all_threads=True)
            atexit.register(self._close_faulthandler)
            log.info("Faulthandler enabled → %s", _crash_log)
        except Exception as exc:
            log.warning("Faulthandler setup failed: %s", exc)
            self._faulthandler_fd = None

    def _close_faulthandler(self) -> None:
        if self._faulthandler_fd is not None:
            try:
                self._faulthandler_fd.close()
            except Exception:
                pass
            self._faulthandler_fd = None

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def set_lifecycle_manager(self, manager) -> None:
        self._lifecycle_manager = manager

    def register_model(self, model_id: str, model: Any) -> None:
        with self._models_lock:
            self._unified_models[model_id] = model

    # ── Slot management ──────────────────────────────────────────────
    async def _acquire_slot(self, request_id: str,
                            timeout_s: float = 0.0) -> int:
        await self._slot_semaphore.acquire()
        try:
            for i in range(self.max_batch):
                if i not in self._slot_holders:
                    now = time.monotonic()
                    self._slot_holders[i] = {
                        "request_id": request_id,
                        "start": now,
                        "last_activity": now,
                        # Fall back to the configured slot_timeout_s when the
                        # caller didn't declare one — health check uses this
                        # as the "definitely stuck" ceiling.
                        "timeout_s": float(timeout_s) or float(
                            self._cfg["slot_timeout_s"]
                        ),
                        # Phase flag: False until the first token arrives.
                        # Health check uses this to pick prefill vs.
                        # inter-token budget.
                        "tokens_seen": False,
                    }
                    log.info("Slot %d acquired for request %s", i, request_id[:8])
                    return i
        except BaseException:
            try:
                self._slot_semaphore.release()
            except Exception:
                pass
            raise
        self._slot_semaphore.release()
        raise RuntimeError("No available slot despite semaphore")

    def _touch_slot(self, request_id: str, *, progress_only: bool = False) -> None:
        # list(): also called from producer threads (prefill progress).
        for slot, info in list(self._slot_holders.items()):
            if info.get("request_id") == request_id:
                info["last_activity"] = time.monotonic()
                # A prefill-progress touch is NOT a token and must not flip
                # tokens_seen (that would switch the health-check from the
                # prefill budget to the much shorter inter-token budget).
                if not progress_only:
                    info["tokens_seen"] = True
                return

    async def _release_slot(self, slot: int, request_id: str) -> None:
        """Return a slot to the pool and release its semaphore permit.

        Called exactly once per `_acquire_slot` via `_slot_context`'s
        `finally`. Always releases the semaphore — even if the holder
        record is already gone or belongs to a different request — so a
        partially-crashed request can never permanently leak a permit
        (which would eventually deadlock all inference).
        """
        async with self._slot_lock:
            info = self._slot_holders.pop(slot, None)
            if info is None:
                log.warning(
                    "Slot %d released by %s but no holder record present — "
                    "semaphore will still be released to prevent leak.",
                    slot, request_id[:8],
                )
            elif info.get("request_id") != request_id:
                log.warning(
                    "Slot %d released by %s but held by %s — removing "
                    "holder and releasing semaphore to prevent leak.",
                    slot, request_id[:8],
                    str(info.get("request_id", "?"))[:8],
                )
        try:
            self._slot_semaphore.release()
        except Exception as exc:
            log.warning("Slot %d semaphore release failed: %s", slot, exc)
            return
        log.info("Slot %d released for request %s", slot, request_id[:8])

    @contextlib.asynccontextmanager
    async def _slot_context(self, request_id: str, timeout_s: float = 0.0):
        slot = await self._acquire_slot(request_id, timeout_s)
        try:
            yield slot
        finally:
            await self._release_slot(slot, request_id)

    # ── Stuck‑slot health check ──────────────────────────────────────────
    async def _slot_health_check(self) -> None:
        # Track request IDs we've already warned about so a legitimately
        # long request (e.g. CLI generation of 700s at 4.5 TPS) produces
        # exactly ONE warning, not one every interval.
        warned: set[str] = set()
        inter_stall = float(self._cfg.get("inter_token_stall_s", 60.0))
        # Prefill (no tokens yet) is legitimately slow on big prompts —
        # llama.cpp can chew on a 1.5k-token prompt for 90s+ with zero
        # output and be perfectly healthy. Give prefill its own budget so
        # the check does not fire mid-prefill.
        prefill_budget = float(
            self._cfg.get("first_token_timeout_floor_s", 120.0)
        )

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._cfg["slot_health_check_interval_s"])
                now = time.monotonic()
                async with self._slot_lock:
                    active_ids = {
                        info["request_id"]
                        for info in self._slot_holders.values()
                    }
                    # Drop stale warnings for requests no longer in flight.
                    warned &= active_ids

                    for slot, info in list(self._slot_holders.items()):
                        rid = info["request_id"]
                        elapsed = now - info["start"]
                        idle = now - info.get("last_activity", info["start"])

                        # Phase-aware idle budget: prefill uses the longer
                        # first-token budget; generation uses the (shorter)
                        # inter-token stall.
                        if info.get("tokens_seen", False):
                            idle_budget = inter_stall
                            phase = "generation"
                        else:
                            idle_budget = prefill_budget
                            phase = "prefill"

                        # `elapsed > timeout_budget` flagged healthy progressing
                        # streams as STUCK ("no activity for 0.1s … held 342s"
                        # in the logs).  Real stalls are idle > budget; total
                        # deadlines are enforced by AsyncTokenBuffer/wait_for.
                        stuck = idle > idle_budget

                        if stuck and rid not in warned:
                            log.warning(
                                "⚠️ Slot %d STUCK during %s — no activity for "
                                "%.1fs (idle budget %.1fs, held %.1fs) request %s",
                                slot, phase, idle, idle_budget, elapsed, rid[:8],
                            )
                            warned.add(rid)
                        elif not stuck and elapsed > self._cfg["slot_timeout_s"]:
                            log.info(
                                "Slot %d held %.1fs (idle %.1fs, phase=%s) "
                                "request %s — long but progressing",
                                slot, elapsed, idle, phase, rid[:8],
                            )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Slot health check error: %s", exc)

    # ── Background tasks ─────────────────────────────────────────────────
    def start_background_tasks(self) -> None:
        if self._tput_ctrl and self._token_mgr:
            try:
                self._aimd_task = asyncio.get_running_loop().create_task(
                    self._aimd_loop(), name="aimd"
                )
            except RuntimeError:
                pass
        self._tuner.start()
        if hasattr(self._scheduler, "start"):
            try:
                asyncio.get_running_loop().create_task(
                    self._scheduler.start(), name="scheduler_start"
                )
                log.info("Scheduler start task enqueued")
            except RuntimeError:
                try:
                    asyncio.ensure_future(self._scheduler.start())
                    log.warning("Scheduler start scheduled via ensure_future")
                except Exception as exc:
                    log.warning("Scheduler start (ensure_future) failed: %s", exc)
            except Exception as exc:
                log.warning("Scheduler start failed: %s", exc)

        try:
            self._slot_health_task = asyncio.get_running_loop().create_task(
                self._slot_health_check(), name="slot_health"
            )
            log.info("Slot health check started")
        except RuntimeError:
            try:
                asyncio.ensure_future(self._slot_health_check())
            except Exception as exc:
                log.warning("Slot health check (ensure_future) failed: %s", exc)

    async def shutdown(self) -> None:
        self._stop_event.set()
        await self._tuner.stop()
        if self._aimd_task and not self._aimd_task.done():
            self._aimd_task.cancel()
            try:
                await self._aimd_task
            except asyncio.CancelledError:
                pass
        if hasattr(self._scheduler, "stop"):
            try:
                await self._scheduler.stop()
            except Exception:
                pass
        if self._slot_health_task and not self._slot_health_task.done():
            self._slot_health_task.cancel()
            try:
                await self._slot_health_task
            except asyncio.CancelledError:
                pass
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._executor.shutdown(wait=True, cancel_futures=False),
        )
        if self._kernel_manager and hasattr(self._kernel_manager, "shutdown"):
            self._kernel_manager.shutdown()

        self._close_faulthandler()

        log.info(
            "InferenceEngine shutdown (req=%d tok=%d err=%d oom=%d)",
            self._total_requests, self._total_tokens,
            self._total_errors,   self._oom_count,
        )

    async def _aimd_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._cfg["aimd_interval_s"])
                if self._tput_ctrl and self._token_mgr:
                    await self._tput_ctrl.adjust_concurrency(self._token_mgr)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("AIMD loop: %s", exc)

    # ── Unified model loading ───────────────────────────────
    async def load_model_from_profile(self, profile: dict) -> str:
        model_id = profile["id"]
        fmt = _normalise_format(profile.get("format", "gguf"))

        with self._models_lock:
            if model_id in self._unified_models:
                return model_id

        if fmt == "gguf":
            from local_llm_core.engine.gguf.model_loader import ModelConfig
            mc = ModelConfig(
                model_id       = model_id,
                model_path     = profile["path"],
                quantization   = profile.get("quantization", "auto"),
                context_length = profile.get("context_length", 0),
                num_gpu_layers = profile.get("num_gpu_layers", -1),
                n_threads      = profile.get("n_threads", 0),
            )
            loaded = await self.loader.load(mc)
            with self._models_lock:
                self._unified_models[model_id] = loaded

        elif fmt == "bitnet":
            if not self._bitnet_enabled:
                raise RuntimeError("BitNet is disabled by configuration")
            wrapper = await self._load_bitnet_model(model_id, profile)
            with self._models_lock:
                self._unified_models[model_id] = wrapper

        elif fmt == "transformers":
            wrapper = await self._load_transformers_model(model_id, profile)
            with self._models_lock:
                self._unified_models[model_id] = wrapper

        else:
            raise ValueError(f"Unsupported model format: {fmt!r}")

        return model_id

    async def _load_bitnet_model(self, model_id: str, profile: dict) -> Any:
        if not self._bitnet_enabled:
            raise RuntimeError("BitNet is disabled by configuration")
        if profile.get("path"):
            return await self._bitnet_loader.load(model_id)
        try:
            from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
            tmp_loader = BitNetLoader.from_state_file()
            return await tmp_loader.load(model_id)
        except Exception as exc:
            log.warning("BitNet from_state_file failed: %s — using loader.load()", exc)
            return await self._bitnet_loader.load(model_id)

    async def _load_transformers_model(self, model_id: str, profile: dict) -> Any:
        model_path = profile.get("path", model_id)
        from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
        from local_llm_core.engine.utils.config_loader import BitNetSystemConfig
        loader = BitNetLoader(config=BitNetSystemConfig())
        return await loader.load(model_path, force_reload=True)

    async def unload_model(self, model_id: str) -> bool:
        with self._models_lock:
            model = self._unified_models.pop(model_id, None)
        if not model:
            return False
        if _is_bitnet_wrapper(model):
            await model.shutdown()
        elif hasattr(model, "llm"):
            await self.loader.unload(model_id)
        else:
            log.warning("Unknown model type for %s", model_id)
        return True

    def get_loaded_model_ids(self) -> List[str]:
        with self._models_lock:
            ids = list(self._unified_models.keys())
        if hasattr(self.loader, "_models"):
            ids += [k for k in self.loader._models.keys() if k not in ids]
        return ids

    def is_model_viable(self, model_id: str) -> bool:
        with self._models_lock:
            if model_id in self._unified_models:
                return True
        if hasattr(self.loader, "get") and self.loader.get(model_id) is not None:
            return True
        return model_id == self._bitnet_state_model_id

    # ── Gate acquire/release ───────────────────────────────────────────────
    async def _acquire(self) -> int:
        async with self._gate_lock:
            pos = self._gate_waiters
            self._gate_waiters += 1
        await self._inference_gate.acquire()
        return pos

    async def _release(self) -> None:
        self._inference_gate.release()
        async with self._gate_lock:
            self._gate_waiters = max(0, self._gate_waiters - 1)

    # cancel an in‑flight request
    async def cancel_request(self, request_id: str) -> bool:
        async with self._running_tasks_lock:
            task = self._running_tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
            log.info("Cancelled in‑flight request %s", request_id[:8])
            if hasattr(self._scheduler, "cancel_request"):
                await self._scheduler.cancel_request(request_id)
            return True
        log.info("cancel_request: request %s not found in running tasks", request_id[:8])
        return False

    # ── Health check ────────────────────────────────────────────────────────
    async def _check_health(self, request: Optional[InferenceRequest] = None) -> bool:
        if hasattr(self, "_scheduler") and self._scheduler is not None:
            thermal = getattr(self._scheduler, "_thermal_state", None)
            if thermal and thermal.throttling:
                log.warning("Health check failed: thermal throttling active")
                return False

        try:
            import psutil as _psutil
            if _psutil.virtual_memory().percent >= 90.0:
                log.warning("Health check failed: RAM pressure at %.1f%%", _psutil.virtual_memory().percent)
                return False
        except ImportError:
            pass

        if request and hasattr(request, "is_disconnected"):
            if callable(request.is_disconnected) and await request.is_disconnected():
                log.info("Health check: client disconnected, stopping generation")
                return False

        return True

    # ── Segmented / resumable generation ───────────────────────────────────
    #
    # CHANGED (resume audit):
    #   The previous implementation of `generate_segmented` (and, by
    #   extension, `resume_generation`) ran entirely outside the slot
    #   manager and the inference gate. That meant:
    #     * the stuck-slot health check never saw resume requests,
    #     * concurrent resumes could stomp each other inside the same
    #       llama.cpp context,
    #     * the circuit breaker and rate limiter were bypassed,
    #     * and the RAM safety guard that protects `generate_stream` and
    #       `_generate_internal` did not apply.
    #
    #   Now the whole segmented run acquires a slot via `_slot_context`,
    #   takes the inference gate via `_acquire()`, checks the breaker and
    #   the rate limiter up front, refuses to start when RAM is above the
    #   same 90% ceiling, and records breaker success/failure on exit.
    #
    #   The two plain-text markers that used to be yielded
    #   ("[Response interrupted during segment …]" and "[Generation paused
    #   due to health issue …]") have been replaced with canonical
    #   `⚡INTERRUPTED` / `⚡STALLED` markers so the client's
    #   `parseStreamMarker` recognises them and shows a working Continue
    #   button instead of dropping the text into the reply body.
    # =====================================================================
    async def generate_segmented(
        self,
        request: InferenceRequest,
        checkpoint_callback: Callable[[GenerationCheckpoint], Any],
        segment_size: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens in segments, checkpointing after each segment.

        Each segment is a self‑contained generation call; the loop continues
        until the model signals a natural stop (EOS/stop sequence), the token
        ceiling is reached, or the segment limit is exhausted.

        Natural‑stop detection is driven by the finish_reason propagated from
        the underlying backend.  If the backend does not provide a reason,
        a fallback heuristic (segment_tokens < seg_max) is used.

        Slot/gate/breaker/rate-limit guards mirror `generate_stream` — this
        path used to bypass all of them.
        """
        if segment_size is None:
            segment_size = self._segment_size

        # ── Circuit breaker (mirrors generate_stream) ──────────────────
        if not self._breaker.can_proceed():
            with self._lk:
                self._total_errors += 1
            yield "[service unavailable — circuit breaker open]"
            return

        # ── Rate limiter (mirrors generate_stream) ─────────────────────
        rate_key = request.model_id or "default"
        if not self.rate_limiter.allow(rate_key):
            with self._lk:
                self._total_errors += 1
            yield "[rate limited]"
            return

        async with self._slot_context(
            request.request_id, request.timeout_s or 0.0,
        ) as slot:
            log.info(
                "Slot %d assigned to segmented request %s",
                slot, request.request_id[:8],
            )

            # ── RAM safety guard (mirrors generate_stream) ─────────────
            try:
                import psutil as _psutil
                _ram_pct = _psutil.virtual_memory().percent
                if _ram_pct >= 90.0:
                    log.error(
                        "Segmented %s: RAM at %.1f%% — refusing to prevent crash.",
                        request.request_id[:8], _ram_pct,
                    )
                    yield (
                        f"[Error: server RAM at {_ram_pct:.0f}% — inference "
                        "refused to prevent crash. Please retry shortly.]"
                    )
                    return
            except ImportError:
                pass

            pos = await self._acquire()
            if pos > 0:
                log.info(
                    "Segmented %s queued at pos %d",
                    request.request_id[:8], pos,
                )

            try:
                total_tokens = 0
                accumulated = request.prompt
                segment_number = 0
                checkpoint = None
                prev_accumulated_len = len(accumulated)
                _effective_max = request.effective_max_tokens
                # Operator backstop (config-driven; 0 = unlimited).
                try:
                    _hard = int(self._cfg.get("max_total_tokens", 0) or 0)
                except Exception:
                    _hard = 0
                if _hard > 0:
                    _effective_max = min(_effective_max, _hard)
                _thinks = ThinkStripper(
                    strip=bool(getattr(request, "strip_think", True))
                )
                _loop_guard = LoopGuardV2(
                    min_cover=int(self._cfg.get("loop_guard_min_cover", 800)),
                    window=int(self._cfg.get("loop_guard_window", 8000)),
                    check_every=int(self._cfg.get("loop_guard_check_every", 48)),
                    line_min_chars=int(self._cfg.get("loop_guard_line_min_chars", 40)),
                    line_repeats=int(self._cfg.get("loop_guard_line_repeats", 3)),
                    heading_min_chars=int(self._cfg.get("loop_guard_heading_min_chars", 12)),
                    heading_repeats=int(self._cfg.get("loop_guard_heading_repeats", 3)),
                    shingle_n=int(self._cfg.get("loop_guard_shingle_n", 8)),
                    shingle_window=int(self._cfg.get("loop_guard_shingle_window", 6000)),
                    shingle_ratio=float(self._cfg.get("loop_guard_shingle_ratio", 0.45)),
                    shingle_min_words=int(self._cfg.get("loop_guard_shingle_min_words", 600)),
                ) if self._cfg.get("loop_guard_enabled", True) else None
                _solicit_guard = SolicitationGuard(
                    grace_tokens=int(self._cfg.get("solicitation_grace_tokens", 64)),
                    min_arm_tokens=int(self._cfg.get("solicitation_min_arm_tokens", 256)),
                    families=self._cfg.get("solicitation_families") or None,
                ) if self._cfg.get("solicitation_guard_enabled", True) else None
                _looped = False

                # Unlimited requests (max_tokens <= 0) continue past max_segments.
                # The limit is only enforced when the caller explicitly asked
                # for a bounded response.
                _seg_cap = int(self._max_segments or 0)
                while total_tokens < _effective_max and (
                        _seg_cap <= 0 or segment_number < _seg_cap
                ):
                    # Integrity: accumulated text must not shrink
                    if len(accumulated) < prev_accumulated_len:
                        log.error(
                            "Segment continuation integrity violation: accumulated length decreased "
                            "from %d to %d at segment %d. Aborting.",
                            prev_accumulated_len, len(accumulated), segment_number
                        )
                        yield _make_stall_marker("integrity", total_tokens)
                        return

                    log.info(
                        "Segment %d starting from %d accumulated characters",
                        segment_number, len(accumulated)
                    )

                    remaining = _effective_max - total_tokens
                    seg_max = min(segment_size, remaining)

                    _n_ctx = self._model_n_ctx(request.model_id)
                    if _n_ctx:
                        seg_max = min(seg_max, max(64, _n_ctx // 8))
                        _before = len(accumulated)
                        accumulated = slide_window(
                            request.prompt,
                            accumulated[len(request.prompt):],
                            n_ctx=_n_ctx,
                            seg_max=seg_max,
                            safety=min(256, max(32, _n_ctx // 16)),
                            count_tokens=self._token_counter(request.model_id),
                        )
                        if len(accumulated) < _before:
                            # A slide is not an integrity violation; resync the
                            # baseline so the next iteration's check passes.
                            prev_accumulated_len = len(accumulated)
                            log.info(
                                "Context window slid at segment %d (n_ctx=%d)",
                                segment_number, _n_ctx,
                            )

                    seg_prompt = accumulated
                    seg_request = InferenceRequest(
                        prompt=seg_prompt,
                        model_id=request.model_id,
                        max_tokens=seg_max,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        stop=request.stop,
                        stream=True,
                        timeout_s=request.timeout_s,
                        request_id=f"{request.request_id}_seg{segment_number}",
                        root_id=(request.root_id or request.request_id),
                    )

                    seg_text = ""
                    segment_tokens = 0
                    finish_reason = None
                    # Create a future to receive the finish reason from the stream
                    finish_reason_future = asyncio.get_running_loop().create_future()

                    try:
                        # Consume the stream; the generator will set finish_reason_future when done
                        gen = self._stream_tokens(seg_request, finish_reason_future=finish_reason_future)
                        async for tok in gen:
                            # Touch the ORIGINAL request's slot — the seg request
                            # shares the same slot but has a different request_id.
                            self._touch_slot(request.request_id)
                            seg_text += tok          # raw — continuation prompts need it
                            segment_tokens += 1
                            total_tokens += 1
                            if _loop_guard is not None and _loop_guard.feed(tok):
                                _looped = True
                                log.warning(
                                    "Repetition loop (%s) in segmented gen %s "
                                    "after %d tokens — ending cleanly",
                                    _loop_guard.reason, request.request_id[:8],
                                    total_tokens,
                                )
                                break
                            if _solicit_guard is not None and _solicit_guard.feed(tok):
                                _looped = True
                                log.warning(
                                    "Self-answer solicitation in segmented gen %s "
                                    "after %d tokens — ending cleanly",
                                    request.request_id[:8], total_tokens,
                                )
                                break
                            _vis = _thinks.feed(tok)
                            if _vis:
                                yield _vis
                        # If a loop guard fired, close the generator NOW.
                        # That sets AsyncTokenBuffer.cancel, which the
                        # producer checks between chunks and exits promptly.
                        if _looped:
                            try:
                                await gen.aclose()
                            except Exception:
                                pass
                            if not finish_reason_future.done():
                                finish_reason_future.set_result(_FINISH_REASON_STOP)

                        # After the stream ends, await the future (with a short timeout)
                        try:
                            finish_reason = await asyncio.wait_for(
                                finish_reason_future, timeout=5.0
                            )
                        except asyncio.TimeoutError:
                            log.warning(
                                "Finish reason not received within timeout, "
                                "using fallback"
                            )
                            finish_reason = (
                                _FINISH_REASON_LENGTH if segment_tokens >= seg_max
                                else _FINISH_REASON_STOP
                            )
                    except asyncio.CancelledError:
                        log.info("Segment %d cancelled after %d tokens", segment_number, total_tokens)
                        # Canonical interruption marker — the client's
                        # parseStreamMarker recognises this and shows the
                        # Continue button. The previous implementation
                        # yielded a plain-text string here, which landed
                        # in the reply body as literal text.
                        yield _make_interruption_marker(total_tokens)
                        checkpoint = GenerationCheckpoint(
                            request_id=request.request_id,
                            model_id=request.model_id,
                            accumulated_text=accumulated + seg_text,
                            tokens_generated=total_tokens,
                            segment_number=segment_number,
                            status="interrupted",
                        )
                        await checkpoint_callback(checkpoint)
                        return

                    accumulated += seg_text
                    prev_accumulated_len = len(accumulated)

                    if _looped:
                        await checkpoint_callback(GenerationCheckpoint(
                            request_id=request.request_id,
                            model_id=request.model_id,
                            accumulated_text=accumulated,
                            tokens_generated=total_tokens,
                            segment_number=segment_number,
                            status="complete",
                        ))
                        break

                    if not await self._check_health(request):
                        log.warning("Health check failed after segment %d, pausing", segment_number)
                        checkpoint = GenerationCheckpoint(
                            request_id=request.request_id,
                            model_id=request.model_id,
                            accumulated_text=accumulated,
                            tokens_generated=total_tokens,
                            segment_number=segment_number,
                            status="interrupted",
                        )
                        await checkpoint_callback(checkpoint)
                        # Canonical stall marker (was plain text before).
                        yield _make_stall_marker("health", total_tokens)
                        return

                    checkpoint = GenerationCheckpoint(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        accumulated_text=accumulated,
                        tokens_generated=total_tokens,
                        segment_number=segment_number,
                        status="in_progress" if total_tokens < _effective_max else "complete",
                    )
                    await checkpoint_callback(checkpoint)

                    # ----- Critical natural‑stop check -----
                    # 1) If the backend explicitly said "stop", we break immediately.
                    if finish_reason == _FINISH_REASON_STOP:
                        log.info("Segment %d finished with natural stop → breaking loop", segment_number)
                        break

                    # 2) Fallback: if the segment produced fewer tokens than requested AND
                    #    the backend did NOT report "length", assume the model stopped
                    #    naturally (or hit a stop sequence). If the backend DID report
                    #    "length", we genuinely hit the per-segment budget; the loop's
                    #    normal continuation logic below handles moving to the next segment.
                    if segment_tokens < seg_max and finish_reason != _FINISH_REASON_LENGTH:
                        log.info(
                            "Segment %d produced fewer tokens than limit (finish_reason=%s) "
                            "→ natural stop assumed",
                            segment_number, finish_reason,
                        )
                        break

                    # 3) If we hit the token limit, we continue only if we haven't reached the global max.
                    if total_tokens >= _effective_max:
                        break

                    segment_number += 1

                _t = _thinks.flush()
                if _t:
                    yield _t

                if checkpoint and checkpoint.status == "in_progress":
                    checkpoint.status = "complete"
                    await checkpoint_callback(checkpoint)

                # Clean completion — record breaker success.
                self._breaker.record_success()
            except (asyncio.CancelledError, GeneratorExit):
                # Starlette closes segmented streams via aclose(),
                # which raises GeneratorExit — not CancelledError. The
                # previous `except asyncio.CancelledError` clause let
                # GeneratorExit fall through to `except Exception:`,
                # tripping the circuit breaker on every client
                # disconnect. Route both through the same path, but
                # don't count a disconnect as a breaker failure.
                self._breaker.record_success()  # client cancel ≠ breaker failure
                raise
            except Exception:
                # Propagate failure to the circuit breaker so the same
                # symptom that trips generate_stream also trips here.
                self._breaker.record_failure()
                raise
            finally:
                await self._release()

    async def resume_generation(
        self,
        checkpoint: GenerationCheckpoint,
        checkpoint_callback: Callable[[GenerationCheckpoint], Any],
        extra_tokens: int = 4096,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> AsyncIterator[str]:
        """
        Resume a previously interrupted generation from its checkpoint.

        CHANGED (resume audit):
          * temperature/top_p/top_k are now forwarded from the caller
            instead of being hard-coded to 0.7/0.95/40. Before this fix
            the client's temperature slider was silently ignored on
            every Continue.
          * The slot/gate/breaker/rate-limit guards now live inside
            `generate_segmented` (which this function delegates to), so
            resume requests are properly scheduled alongside fresh
            requests. Previously they bypassed all of that and could
            stomp a concurrent generation inside the same llama.cpp
            context.

        `checkpoint.accumulated_text` is the fully templated prompt
        (original chat template + text generated so far), which is
        exactly the right thing to feed back to llama.cpp for a
        continuation. No additional templating is applied here.
        """
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
        async for token in self.generate_segmented(resume_request, checkpoint_callback):
            yield token

    # ── Public API ─────────────────────────────────────────────────────────
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        task = asyncio.current_task()
        async with self._running_tasks_lock:
            self._running_tasks[request.request_id] = task

        try:
            return await self._generate_internal(request)
        finally:
            async with self._running_tasks_lock:
                self._running_tasks.pop(request.request_id, None)

    async def _generate_internal(self, request: InferenceRequest) -> InferenceResponse:
        rate_key = request.model_id or "default"
        if not self.rate_limiter.allow(rate_key):
            with self._lk:
                self._total_errors += 1
            if _PROM.get("errors"):
                _PROM["errors"].labels(reason="rate_limited").inc()
            return InferenceResponse(
                request_id=request.request_id,
                text="[rate limited — slow down requests]",
                tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                total_time_s=0.0, tps=0.0, finish_reason="rate_limited",
            )

        if not self._breaker.can_proceed():
            with self._lk:
                self._total_errors += 1
            if _PROM.get("errors"):
                _PROM["errors"].labels(reason="circuit_open").inc()
            return InferenceResponse(
                request_id=request.request_id,
                text="[service unavailable — circuit breaker open, retry in 60s]",
                tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                total_time_s=0.0, tps=0.0, finish_reason="circuit_open",
            )

        with self._lk:
            if self._inflight >= self._queue._max:
                self._total_errors += 1
                return InferenceResponse(
                    request_id=request.request_id,
                    text="[server overloaded — queue full, retry shortly]",
                    tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                    total_time_s=0.0, tps=0.0, finish_reason="overloaded",
                )
            self._inflight += 1

        try:
            import psutil as _psutil
            _ram_pct = _psutil.virtual_memory().percent
            if _ram_pct >= 90.0:
                log.error(
                    "RAM at %.1f%% — refusing inference to prevent "
                    "C‑level GGML_ASSERT crash (logits=nullptr).",
                    _ram_pct,
                )
                with self._lk:
                    self._inflight -= 1
                return InferenceResponse(
                    request_id=request.request_id,
                    text=f"[Error: server RAM at {_ram_pct:.0f}% — inference refused to prevent crash]",
                    tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                    total_time_s=0.0, tps=0.0, finish_reason="overloaded",
                )
        except ImportError:
            pass

        try:
            for attempt in range(3):
                acquired = False
                try:
                    pos = await self._acquire()
                    acquired = True
                    if pos > 0:
                        log.info("Request %s queued at pos %d",
                                 request.request_id[:8], pos)
                    timeout = (request.timeout_s if request.timeout_s > 0
                               else self._cfg["default_timeout_s"])
                    _wait = timeout if (timeout and timeout > 0) else None  # None == no deadline
                    result = await asyncio.wait_for(self._generate_inner(request), timeout=_wait)
                    self._breaker.record_success()
                    if _PROM.get("req"):
                        _PROM["req"].inc()
                    return result

                except asyncio.CancelledError:
                    log.info("Request %s cancelled", request.request_id[:8])
                    return InferenceResponse(
                        request_id=request.request_id, text="[request cancelled]",
                        tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                        total_time_s=0.0, tps=0.0, finish_reason="cancelled",
                    )
                except asyncio.TimeoutError:
                    log.warning("Request %s timed out after %.1fs",
                                request.request_id[:8], timeout or 0.0)
                    with self._lk:
                        self._total_errors += 1
                    if _PROM.get("errors"):
                        _PROM["errors"].labels(reason="timeout").inc()
                    return InferenceResponse(
                        request_id=request.request_id,
                        text=f"[timed out after {timeout:.0f}s]",
                        tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                        total_time_s=timeout, tps=0.0, finish_reason="timeout",
                    )
                except MemoryError as exc:
                    log.warning("OOM attempt %d/3: %s", attempt + 1, exc)
                    with self._lk:
                        self._oom_count += 1
                    if _PROM.get("oom"):
                        _PROM["oom"].inc()
                    gc.collect()
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
                    continue
                except Exception as exc:
                    self._breaker.record_failure()
                    with self._lk:
                        self._total_errors += 1
                    if _PROM.get("errors"):
                        _PROM["errors"].labels(reason="exception").inc()
                    log.error("generate error for %s: %s",
                              request.request_id[:8], exc, exc_info=True)
                    raise
                finally:
                    if acquired:
                        await self._release()

            return InferenceResponse(
                request_id=request.request_id, text="[failed after 3 attempts]",
                tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                total_time_s=0.0, tps=0.0, finish_reason="error",
            )
        finally:
            with self._lk:
                self._inflight = max(0, self._inflight - 1)

    async def _generate_inner(self, request: InferenceRequest) -> InferenceResponse:
        async with self._slot_context(
                request.request_id, request.timeout_s or 0.0,
        ) as slot:
            log.info("Slot %d assigned to request %s", slot, request.request_id[:8])

            t0 = time.perf_counter()
            toks: List[str] = []
            ttft = 0.0
            cache_hit = False

            if self._ctx_cache and request.temperature <= 0.01:
                cached = self._ctx_cache.get(
                    request.prompt, model_id=request.model_id,
                    temperature=request.temperature,
                )
                if cached:
                    cache_hit = True
                    toks = list(cached)
                    ttft = 0.0
                    if _PROM.get("cache"):
                        _PROM["cache"].inc()

            if not cache_hit:
                try:
                    log.info("Dispatching request %s to backend", request.request_id[:8])
                    async for tok in self._continuous_tokens(request):
                        self._touch_slot(request.request_id)
                        if not toks:
                            ttft = (time.perf_counter() - t0) * 1000
                        toks.append(tok)
                except asyncio.CancelledError:
                    elapsed = time.perf_counter() - t0
                    return InferenceResponse(
                        request_id=request.request_id, text="".join(toks),
                        tokens_gen=len(toks), prompt_tokens=len(request.prompt.split()),
                        ttft_ms=round(ttft, 2), total_time_s=round(elapsed, 4),
                        tps=0.0, finish_reason="cancelled",
                    )

            elapsed = time.perf_counter() - t0
            raw_text = "".join(toks)
            full_text = (
                strip_think_text(raw_text)
                if getattr(request, "strip_think", True)
                else raw_text
            )
            think_len = max(0, len(raw_text) - len(full_text))
            think_tok = int(think_len / max(len(raw_text), 1) * len(toks)) if think_len else 0
            n_completion = max(1, len(toks) - think_tok)

            tps = (
                min(n_completion / elapsed, 500.0)
                if not cache_hit and elapsed >= 0.5 and n_completion >= 1
                else 0.0
            )

            # honest decode-only tps. `elapsed` includes prefill, so
            # n_completion / elapsed is end-to-end (i.e. artificially low
            # when prefill dominates). `elapsed - ttft_s` is the decode
            # window; that's the number CapabilityProfile needs.
            _ttft_s = ttft / 1000.0
            _decode_s = max(0.0, elapsed - _ttft_s)
            _decode_tps = (
                n_completion / _decode_s
                if _decode_s >= 0.05 and n_completion >= 1 else 0.0
            )

            with self._lk:
                self._total_requests += 1
                self._total_tokens += n_completion
                self._latencies.append(ttft)
                if tps > 0:
                    self._tps_samples.append(tps)
                if _decode_tps > 0:
                    self._decode_tps_samples.append(_decode_tps)

            if self._tput_ctrl:
                self._tput_ctrl.record(n_completion)
            if _PROM.get("ttft"):
                _PROM["ttft"].observe(ttft)
            if _PROM.get("tps"):
                _PROM["tps"].observe(tps)

            if self._ctx_cache and request.temperature <= 0.01 and not cache_hit and full_text:
                self._ctx_cache.put(
                    request.prompt, full_text,
                    state_bytes=len(full_text.encode()),
                    model_id=request.model_id, temperature=request.temperature,
                )
            self._ngram.observe(toks)

            return InferenceResponse(
                request_id=request.request_id,
                text=full_text,
                tokens_gen=n_completion,
                prompt_tokens=max(1, len(request.prompt.split())),
                ttft_ms=round(ttft, 2),
                total_time_s=round(elapsed, 4),
                tps=round(tps, 2),
                finish_reason=("stop" if n_completion < request.effective_max_tokens else "length"),
                context_cache_hit=cache_hit,
            )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        task = asyncio.current_task()
        async with self._running_tasks_lock:
            self._running_tasks[request.request_id] = task

        async with self._slot_context(
                    request.request_id, request.timeout_s or 0.0,
            ) as slot:
            log.info("Slot %d assigned to stream request %s", slot, request.request_id[:8])

            _finish_future: asyncio.Future = asyncio.get_running_loop().create_future()

            try:
                if not self._breaker.can_proceed():
                    with self._lk:
                        self._total_errors += 1
                    yield "[service unavailable — circuit breaker open]"
                    return

                rate_key = request.model_id or "default"
                if not self.rate_limiter.allow(rate_key):
                    yield "[rate limited]"
                    return

                t0 = time.perf_counter()
                first_tok = 0.0
                token_count = 0

                pos = await self._acquire()
                if pos > 0:
                    log.info("Stream %s queued at pos %d", request.request_id[:8], pos)

                try:
                    import psutil as _psutil
                    _ram_pct = _psutil.virtual_memory().percent
                    if _ram_pct >= 90.0:
                        log.error(
                            "Stream %s: RAM at %.1f%% — refusing to prevent crash.",
                            request.request_id[:8], _ram_pct,
                        )
                        await self._release()
                        yield (
                            f"[Error: server RAM at {_ram_pct:.0f}% — inference "
                            "refused to prevent crash. Please retry shortly.]"
                        )
                        return
                except ImportError:
                    pass

                try:
                    log.info("Dispatching stream request %s to backend",
                             request.request_id[:8])
                    _thinks = ThinkStripper(
                        strip=bool(getattr(request, "strip_think", True))
                    )
                    async for raw in self._continuous_tokens(request, _finish_future):
                        # update last_activity on every token so
                        # the health check can distinguish "slow but
                        # progressing" from "silently stuck".
                        self._touch_slot(request.request_id)
                        if self._tput_ctrl:
                            self._tput_ctrl.record(1)
                        tok = _thinks.feed(raw)
                        if not tok:
                            continue
                        if token_count == 0:
                            first_tok = (time.perf_counter() - t0) * 1000
                        token_count += 1
                        yield tok
                    _tail = _thinks.flush()
                    if _tail:
                        if token_count == 0:
                            first_tok = (time.perf_counter() - t0) * 1000
                        token_count += 1
                        yield _tail
                    self._breaker.record_success()
                except asyncio.CancelledError:
                    log.info("Stream %s cancelled (%d tokens)",
                             request.request_id[:8], token_count)
                    yield _make_interruption_marker(token_count)
                    return
                except StreamStallError as _stall:
                    log.warning("Stream %s stalled: %s",
                                request.request_id[:8], _stall)
                    yield _make_stall_marker(_stall.phase, token_count)
                    return
                except Exception as exc:
                    self._breaker.record_failure()
                    log.error("Stream %s error: %s", request.request_id[:8], exc)
                    if token_count == 0:
                        yield f"[error: {exc}]"
                    return
                finally:
                    await self._release()
                    elapsed = time.perf_counter() - t0
                    tps = (min(token_count / elapsed, 500.0)
                           if elapsed >= 0.5 and token_count >= 1 else 0.0)
                    _ttft_s = first_tok / 1000.0
                    _decode_s = max(0.0, elapsed - _ttft_s)
                    _decode_tps = (
                        token_count / _decode_s
                        if _decode_s >= 0.05 and token_count >= 1 else 0.0
                    )
                    with self._lk:
                        self._total_requests += 1
                        self._total_tokens += token_count
                        if first_tok > 0:
                            self._latencies.append(first_tok)
                        if tps > 0:
                            self._tps_samples.append(tps)
                        if _decode_tps > 0:
                            self._decode_tps_samples.append(_decode_tps)

                # After a clean run, inspect finish reason.
                try:
                    fr = await asyncio.wait_for(_finish_future, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    fr = None
                if fr == _FINISH_REASON_LENGTH:
                    yield _make_truncation_marker(token_count)
            finally:
                async with self._running_tasks_lock:
                    self._running_tasks.pop(request.request_id, None)

    async def generate_batch(self, requests: list) -> list:
        results = await asyncio.gather(
            *[self.generate(r) for r in requests], return_exceptions=True
        )
        out = []
        for req, res in zip(requests, results):
            if isinstance(res, Exception):
                out.append(InferenceResponse(
                    request_id=req.request_id, text=f"[error: {res}]",
                    tokens_gen=0, prompt_tokens=0, ttft_ms=0.0,
                    total_time_s=0.0, tps=0.0, finish_reason="error",
                ))
            else:
                out.append(res)
        return out

    # ── Loaded-model introspection (used by _continuous_tokens and callers) ──
    def _loaded_model_obj(self, model_id: str):
        """Return the loaded model wrapper (LoadedModel or BitNet wrapper)."""
        with self._models_lock:
            m = self._unified_models.get(model_id)
        if m is None and hasattr(self.loader, "get"):
            m = self.loader.get(model_id)
        return m

    def _model_n_ctx(self, model_id: str) -> int:
        """Real loaded context window (0 if unknown, e.g. BitNet wrappers).

        Prefers `loaded_context_length` (which model_loader sets after the
        memory-aware planner ran), falls back to reading `n_ctx` off the
        underlying llama_cpp object via _read_llm_n_ctx.
        """
        m = self._loaded_model_obj(model_id)
        n = int(getattr(m, "loaded_context_length", 0) or 0)
        return n or _read_llm_n_ctx(getattr(m, "llm", m))

    def get_progress(self, root_id: str) -> dict:
        """Snapshot for heartbeat frames; {} if unknown."""
        p = self._progress.get(root_id)
        return dict(p) if p else {}

    def clear_progress(self, root_id: str) -> None:
        self._progress.pop(root_id, None)

    def _token_counter(self, model_id: str):
        """Return a callable(text)->int using the loaded model's tokenizer.

        Falls back to chars/3.0 when the tokenizer isn't reachable, which
        matches the client-side estimate closely enough for window sliding.
        """
        m = self._loaded_model_obj(model_id)
        llm = getattr(m, "llm", m)
        tok = getattr(llm, "tokenize", None)

        def count(text: str) -> int:
            if tok is not None:
                try:
                    return len(tok(text.encode("utf-8", "ignore"),
                                   add_bos=False, special=True))
                except Exception:
                    pass
            return max(1, int(len(text) / 3.0))

        return count

    def _get_or_compile_grammar(self, src: Any) -> Optional[Any]:
        """
        Compile-or-fetch a `LlamaGrammar` from a GBNF string or JSON schema.

        `src` is either:
          * a `str` — raw GBNF, compiled via `from_string`, or
          * a `dict` — JSON-schema, compiled via `from_json_schema`.

        Returns `None` (and logs once) if compilation fails; the caller
        must treat `None` as "no grammar" and continue unconstrained.
        """
        try:
            from llama_cpp import LlamaGrammar
        except ImportError:
            return None

        # Canonical cache key — identical schemas hash to the same entry
        # regardless of dict insertion order.
        if isinstance(src, dict):
            try:
                key = "json:" + json.dumps(src, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                return None
        else:
            key = "gbnf:" + str(src)

        with self._grammar_cache_lock:
            cached = self._grammar_cache.get(key)
            if cached is not None:
                self._grammar_cache.move_to_end(key)
                return cached

        try:
            if isinstance(src, dict):
                obj = LlamaGrammar.from_json_schema(
                    json.dumps(src, separators=(",", ":"))
                )
            else:
                obj = LlamaGrammar.from_string(str(src))
        except Exception as exc:
            log.warning("Grammar compile failed (%s): %s", key[:32], exc)
            return None

        with self._grammar_cache_lock:
            self._grammar_cache[key] = obj
            while len(self._grammar_cache) > self._grammar_cache_max:
                self._grammar_cache.popitem(last=False)
        return obj

    async def _continuous_tokens(self, request, finish_future=None):
        """Token stream that never ends at the context edge.

        Generates in segments. Before each segment, the window is slid
        (original prompt preserved verbatim; oldest generated text dropped;
        a continuation marker injected) so prompt + segment always fits
        n_ctx. Ends only on EOS/stop, an explicit max_tokens, a degenerate
        repetition loop, or client cancel.

        finish_future gets _FINISH_REASON_LENGTH only when an *explicit*
        max_tokens was reached, so callers can distinguish "user asked for
        exactly N tokens" from "model stopped naturally".
        """
        loop = asyncio.get_running_loop()
        limit = request.effective_max_tokens
        try:
            hard_cap = int(self._cfg.get("max_total_tokens", 0) or 0)
        except Exception:
            hard_cap = 0
        if hard_cap > 0:
            limit = min(limit, hard_cap)

        n_ctx = self._model_n_ctx(request.model_id)
        count = self._token_counter(request.model_id)
        try:
            cfg_safety = int(self._cfg.get("context_safety_tokens", 256))
        except Exception:
            cfg_safety = 256
        seg_cap = int(self._segment_size or 512)
        if n_ctx:
            # safety sized relative to n_ctx so it can't dominate on tiny windows
            safety = min(cfg_safety, max(32, n_ctx // 16))
            seg_cap = min(seg_cap, max(64, n_ctx // 8))
        else:
            safety = cfg_safety

        head, generated = request.prompt, ""
        total, seg_no, result = 0, 0, _FINISH_REASON_LENGTH

        _lg_enabled = bool(self._cfg.get("loop_guard_enabled", True))
        _sg_enabled = bool(self._cfg.get("solicitation_guard_enabled", True))
        guard = LoopGuardV2(
            min_cover=int(self._cfg.get("loop_guard_min_cover", 800)),
            window=int(self._cfg.get("loop_guard_window", 8000)),
            check_every=int(self._cfg.get("loop_guard_check_every", 48)),
            line_min_chars=int(self._cfg.get("loop_guard_line_min_chars", 40)),
            line_repeats=int(self._cfg.get("loop_guard_line_repeats", 3)),
            heading_min_chars=int(self._cfg.get("loop_guard_heading_min_chars", 12)),
            heading_repeats=int(self._cfg.get("loop_guard_heading_repeats", 3)),
            shingle_n=int(self._cfg.get("loop_guard_shingle_n", 8)),
            shingle_window=int(self._cfg.get("loop_guard_shingle_window", 6000)),
            shingle_ratio=float(self._cfg.get("loop_guard_shingle_ratio", 0.45)),
            shingle_min_words=int(self._cfg.get("loop_guard_shingle_min_words", 600)),
        ) if _lg_enabled else RepetitionGuard()
        solicit = SolicitationGuard(
            grace_tokens=int(self._cfg.get("solicitation_grace_tokens", 64)),
            min_arm_tokens=int(self._cfg.get("solicitation_min_arm_tokens", 256)),
            families=self._cfg.get("solicitation_families") or None,
        ) if _sg_enabled else None

        while total < limit:
            seg_max = min(seg_cap, limit - total)

            if n_ctx:
                before = len(generated)
                prompt = slide_window(
                    head, generated, n_ctx=n_ctx, seg_max=seg_max,
                    safety=safety, count_tokens=count,
                )
                generated = prompt[len(head):]
                if len(generated) < before:
                    log.info(
                        "Context window slid for %s at segment %d "
                        "(kept prompt + last %d chars; n_ctx=%d)",
                        request.request_id[:8], seg_no, len(generated), n_ctx,
                    )
            else:
                prompt = head + generated

            seg_req = InferenceRequest(
                prompt=prompt, model_id=request.model_id, max_tokens=seg_max,
                temperature=request.temperature, top_p=request.top_p,
                top_k=request.top_k, stop=request.stop, stream=True,
                timeout_s=request.timeout_s,
                strip_think=False,  # ThinkStripper runs at the caller level
                request_id=f"{request.request_id}_c{seg_no}",
                root_id=(request.root_id or request.request_id),
            )
            seg_fut = loop.create_future()
            seg_text, seg_tokens, looped = "", 0, False
            gen = self._stream_tokens(seg_req, seg_fut)
            try:
                async for tok in gen:
                    self._touch_slot(request.request_id)
                    seg_text += tok
                    seg_tokens += 1
                    total += 1
                    if guard.feed(tok):
                        looped = True
                        break
                    if solicit is not None and solicit.feed(tok):
                        looped = True
                        break
                    yield tok
            finally:
                await gen.aclose()

            try:
                fr = await asyncio.wait_for(seg_fut, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                fr = (_FINISH_REASON_LENGTH if seg_tokens >= seg_max
                      else _FINISH_REASON_STOP)

            generated += seg_text

            if looped:
                _reason = getattr(guard, "reason", "") or "periodic"
                log.warning(
                    "Repetition loop (%s) detected for %s after %d tokens — "
                    "ending cleanly",
                    _reason, request.request_id[:8], total,
                )
                result = _FINISH_REASON_STOP
                break

            if fr != _FINISH_REASON_LENGTH or seg_tokens == 0:
                result = _FINISH_REASON_STOP  # EOS / stop seq / no progress
                break

            seg_no += 1

        if finish_future is not None and not finish_future.done():
            finish_future.set_result(result)

    # ── Internal stream dispatch ───────────────────────────────────────────
    async def _stream_tokens(
        self,
        req: InferenceRequest,
        finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        """
        Internal streaming dispatcher.

        If finish_reason_future is provided, the underlying backend is expected
        to set it with a string ('stop' or 'length') when the stream ends.
        """
        with self._models_lock:
            model = self._unified_models.get(req.model_id)

        if model is None and hasattr(self.loader, "get"):
            model = self.loader.get(req.model_id)

        if model is None:
            model = await self._try_autoload_model(req.model_id)

        if model is None:
            raise RuntimeError(
                f"Model '{req.model_id}' not loaded. "
                f"Available: {self.get_loaded_model_ids()}"
            )

        from local_llm_core.engine.gguf.model_loader import LoadedModel

        if _is_bitnet_wrapper(model):
            async for tok in self._bitnet_dispatch(model, req, finish_reason_future):
                yield tok
        elif isinstance(model, LoadedModel):
            async for tok in self._dispatch(model.llm, req, finish_reason_future):
                yield tok
        elif hasattr(model, "llm"):
            async for tok in self._dispatch(model.llm, req, finish_reason_future):
                yield tok
        else:
            async for tok in self._dispatch(model, req, finish_reason_future):
                yield tok

    async def _try_autoload_model(self, model_id: str) -> Optional[Any]:
        if self._lifecycle_manager:
            try:
                log.info("Auto‑loading '%s' via lifecycle manager", model_id)
                model = await self._lifecycle_manager.load_model(model_id)
                with self._models_lock:
                    self._unified_models[model_id] = model
                return model
            except Exception as exc:
                log.warning("Lifecycle auto‑load failed for '%s': %s", model_id, exc)

        if (model_id == self._bitnet_state_model_id or self._bitnet_state_model_id) and self._bitnet_enabled:
            try:
                log.info("Auto‑loading BitNet from state file for '%s'", model_id)
                from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
                loader = BitNetLoader.from_state_file()
                wrapper = await loader.load("state_model")
                actual_id = self._bitnet_state_model_id or model_id
                with self._models_lock:
                    self._unified_models[actual_id] = wrapper
                    if actual_id != model_id:
                        self._unified_models[model_id] = wrapper
                return wrapper
            except Exception as exc:
                log.warning("BitNet state auto‑load failed for '%s': %s", model_id, exc)

        return None

    async def _dispatch(
            self,
            llm,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:

        if _is_native_async_backend(llm):
            async for t in self._bitnet_cpp_dispatch(llm, req, finish_reason_future):
                yield t
        elif callable(llm) and (hasattr(llm, "n_ctx") or hasattr(llm, "create_completion")):
            async for t in self._llama_stream(llm, req, finish_reason_future):
                yield t
        elif isinstance(llm, dict) and "model" in llm:
            async for t in self._hf_stream(llm["model"], req, finish_reason_future, llm.get("tokenizer")):
                yield t
        elif hasattr(llm, "generate") and hasattr(llm, "config"):
            async for t in self._hf_stream(llm, req, finish_reason_future):
                yield t
        elif hasattr(llm, "tokenize"):
            async for t in self._ct_stream(llm, req, finish_reason_future):
                yield t
        else:
            async for t in self._sim_stream(req, finish_reason_future):
                yield t


    async def _bitnet_cpp_dispatch(
            self,
            llm,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        from local_llm_core.engine.bitnet.bitnet_cpp_backend import BitNetCppConfig

        _BITNET_STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>"]
        caller_stop = list(req.stop) if req.stop else []
        merged_stop = list(dict.fromkeys(_BITNET_STOP_TOKENS + caller_stop))

        cfg = BitNetCppConfig(
            max_tokens=min(int(req.effective_max_tokens) or 512, 1024),
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=merged_stop,
            first_token_timeout=180.0,
        )
        try:
            token_count = 0
            async for token in llm.agenerate(
                    prompt=req.prompt, cfg=cfg, stop=merged_stop,
                    request_id=req.request_id,
            ):
                token_count += 1
                yield token
            # If we get here without exception, assume natural stop if token_count < max_tokens
            if finish_reason_future and not finish_reason_future.done():
                if token_count < cfg.max_tokens:
                    finish_reason_future.set_result(_FINISH_REASON_STOP)
                else:
                    finish_reason_future.set_result(_FINISH_REASON_LENGTH)
        except asyncio.CancelledError:
            log.info("BitNet.cpp generation cancelled: %s", req.request_id[:8])
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.cancel()
            raise
        except asyncio.TimeoutError:
            log.error("BitNet.cpp timeout: %s", req.request_id[:8])
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.set_exception(asyncio.TimeoutError())
            raise

    async def _bitnet_dispatch(
            self,
            wrapper,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        from local_llm_core.engine.bitnet.bitnet_engine import BitNetInferenceConfig
        cfg = BitNetInferenceConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            first_token_timeout=0.0,
            total_timeout=0.0,
        )
        try:
            token_count = 0
            async for token in wrapper.agenerate(
                    prompt=req.prompt, cfg=cfg, stop=req.stop,
                    request_id=req.request_id,
            ):
                token_count += 1
                yield token
            if finish_reason_future and not finish_reason_future.done():
                if token_count < cfg.max_tokens:
                    finish_reason_future.set_result(_FINISH_REASON_STOP)
                else:
                    finish_reason_future.set_result(_FINISH_REASON_LENGTH)
        except asyncio.CancelledError:
            log.info("BitNet generation cancelled: %s", req.request_id[:8])
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.cancel()
            raise
        except asyncio.TimeoutError:
            log.error("BitNet timeout: %s", req.request_id[:8])
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.set_exception(asyncio.TimeoutError())
            raise

    async def _llama_stream(
            self,
            llm,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        # AsyncTokenBuffer is STALL DETECTION ONLY — first-token budget +
        # inter-token stall. It must never carry a wall-clock total deadline:
        # prefill on this hardware legitimately takes 15+ minutes for a
        # 3k-token prompt at ~6 tok/s, and a total deadline here duplicates
        # the stall checks while misfiring before the first token arrives.
        # Total deadlines (if any) are enforced by the caller:
        #   * _generate_internal uses asyncio.wait_for(..., timeout=_wait)
        #   * streaming paths rely on stall detection + client disconnect
        _buf_timeout = 0.0
        if timeout_s := (req.timeout_s if req.timeout_s > 0
                         else float(self._cfg.get("default_timeout_s", 0.0) or 0.0)):
            log.info(
                "Request %s carries an explicit %.0fs total deadline; it will "
                "be enforced by the caller, not by AsyncTokenBuffer.",
                req.request_id[:8], timeout_s,
            )

        # Read n_ctx from the model. Prefer the loaded_context_length that
        # ModelLoader sets — it reflects the hardware-profile-capped value
        # the model was actually loaded with. Fall back to the raw
        # llama_cpp.n_ctx only if the loader didn't record one, and warn
        # loudly before falling back to the 4096 default: a wrong n_ctx
        # silently undersizes reply headroom and can truncate prompts.
        _n_ctx = self._model_n_ctx(req.model_id)
        if _n_ctx <= 0:
            _n_ctx = _read_llm_n_ctx(llm)
        if _n_ctx <= 0:
            log.warning(
                "Could not determine n_ctx for model %r — falling back to "
                "4096. Replies may be shorter than necessary and long "
                "prompts may be silently truncated. Verify the model was "
                "loaded via ModelLoader (which sets loaded_context_length).",
                req.model_id,
            )
            _n_ctx = 4096
        _prompt_est = self._token_counter(req.model_id)(req.prompt)
        try:
            _safety = min(
                int(self._cfg.get("context_safety_tokens", 256)),
                max(32, _n_ctx // 16),
            )
        except Exception:
            _safety = 256
        _max_tokens_avail = max(1, _n_ctx - _prompt_est - _safety)

        # refuse generation if headroom is insufficient, rather
        # than silently clamping to 1 token.
        _min_floor = int(self._cfg.get("min_response_tokens", 64))
        if _max_tokens_avail < _min_floor:
            log.error(
                "Refusing generation for %s: prompt (%d tok) leaves only %d "
                "tokens of headroom on this model's n_ctx=%d.",
                req.request_id[:8], _prompt_est, _max_tokens_avail, _n_ctx,
            )
            raise RuntimeError(
                f"Insufficient context headroom: {_max_tokens_avail} < "
                f"{_min_floor}. Shorten the prompt or reduce max_tokens."
            )
        _max_tokens = min(int(req.effective_max_tokens), _max_tokens_avail)

        # adaptive first-token budget from measured prefill TPS.
        _floor = float(self._cfg.get("first_token_timeout_floor_s", 120.0))
        _mult = float(self._cfg.get("first_token_timeout_multiplier", 3.0))
        _first_budget = _floor
        if self._prefill_tracker.has_observations:
            _est_prefill_s = _prompt_est / max(self._prefill_tracker.tps, 1.0)
            _first_budget = max(_floor, _est_prefill_s * _mult)
        log.info(
            "Prefill budget for %s: %.1fs (est_prompt_tokens=%d, "
            "ewma_prefill_tps=%.2f)",
            req.request_id[:8], _first_budget, _prompt_est,
            self._prefill_tracker.tps,
        )

        # ── Grammar resolution ────────────────────────────────────────────
        # Priority: raw `grammar` string > `response_format`.
        # `json_object` is converted to a minimal object schema so that the
        # downstream compiler and cache see the same shape as an explicit
        # `json_schema` request would produce. A failure here is logged
        # and downgraded to unconstrained — never fatal.
        _grammar_obj = None
        _grammar_src: Any = None
        if req.grammar:
            _grammar_src = req.grammar
        elif req.response_format:
            rtype = (req.response_format.get("type") or "").lower()
            if rtype == "json_object":
                _grammar_src = {"type": "object"}
            elif rtype == "json_schema":
                _grammar_src = (
                    req.response_format.get("json_schema", {}).get("schema")
                )
            # rtype == "text" → deliberately unconstrained
        if _grammar_src:
            _grammar_obj = self._get_or_compile_grammar(_grammar_src)
            if _grammar_obj is None:
                log.warning(
                    "Request %s asked for constrained output but grammar "
                    "compilation failed — generating unconstrained.",
                    req.request_id[:8],
                )

        buf = AsyncTokenBuffer(
            loop, maxsize=4096, timeout_s=_buf_timeout,
            first_token_budget_s=_first_budget,
            inter_token_stall_s=float(self._cfg.get("inter_token_stall_s", 60.0)),
        )

        temp = 0.0 if req.temperature <= 0.01 else max(req.temperature, 1e-7)
        # Anti-repetition sampling.  Only forwarded when we are sampling
        # (temperature > 0); under greedy decoding llama.cpp ignores them
        # anyway, but passing them causes silent drift on some builds.
        _apply_pen = bool(
            self._cfg.get("penalty_apply_only_when_sampling", True)
        ) is False or temp > 0
        _pen_kwargs: Dict[str, Any] = {}
        if _apply_pen:
            try:
                _pp = float(self._cfg.get("presence_penalty", 1.0) or 0.0)
                _fp = float(self._cfg.get("frequency_penalty", 0.2) or 0.0)
                _rp = float(self._cfg.get("repeat_penalty", 1.05) or 1.0)
            except Exception:
                _pp = _fp = 0.0
                _rp = 1.0
            if _pp:
                _pen_kwargs["presence_penalty"] = _pp
            if _fp:
                _pen_kwargs["frequency_penalty"] = _fp
            if _rp and _rp != 1.0:
                _pen_kwargs["repeat_penalty"] = _rp
        top_p = float(req.top_p) if temp > 0 else 1.0
        top_k = int(req.top_k) if temp > 0 else 1
        _stop = list(req.stop) if req.stop else []
        # </think> removed as a stop token — see write-up.

        captured_finish_reason = None
        produce_start = time.monotonic()
        first_token_ts: list = []  # mutable cell

        _extra_kwargs: Dict[str, Any] = dict(_pen_kwargs)
        if _grammar_obj is not None:
            _extra_kwargs["grammar"] = _grammar_obj

        _root = req.root_id or req.request_id
        _pf_info: dict = {}

        def _produce():
            nonlocal captured_finish_reason
            _count = 0
            try:
                # Interruptible, observable prefill.
                if self._cfg.get("chunked_prefill", True):
                    def _prog(done, total, reused):
                        self._progress[_root] = {
                            "phase": "prefill",
                            "done": done, "total": total, "reused": reused,
                            "t": time.monotonic(),
                        }
                        try:
                            self._touch_slot(_root, progress_only=True)
                        except Exception:
                            pass
                        try:
                            buf.sync_put(_PROGRESS)
                        except Exception:
                            pass

                    _tps = (self._prefill_tracker.tps
                            if self._prefill_tracker.has_observations else 0.0)
                    _cmin = int(self._cfg.get("chunked_prefill_chunk_min", 32))
                    _cmax = int(self._cfg.get("chunked_prefill_chunk_max", 256))
                    _csec = float(self._cfg.get("chunked_prefill_chunk_seconds", 3.0))
                    _mtok = int(self._cfg.get("chunked_prefill_min_tokens", 256))
                    # Chunk sizing: the old formula derived chunk from
                    # the throughput it itself was affecting, producing a
                    # self-reinforcing slow loop. Fixed floor at one
                    # `n_ubatch` (the compute-block size the backend is
                    # tuned for) and let `chunk_seconds` only bound the
                    # cancel latency — a genuinely independent knob.
                    _chunk = max(_cmin, int(_cmax))
                    # Honour cancel latency target if it is stricter than
                    # the full-preferred chunk:
                    if _tps > 0:
                        _by_latency = int(_tps * _csec)
                        if _by_latency > _cmin:
                            _chunk = min(_chunk, _by_latency)
                    _t0 = time.monotonic()
                    try:
                        _pf_info.update(chunked_prefill(
                            llm, req.prompt, buf.cancel, _prog,
                            chunk=_chunk, min_tokens=_mtok,
                        ))
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        log.warning(
                            "chunked prefill disabled for this request: %s", exc,
                        )
                    if _pf_info.get("cancelled"):
                        return
                    if _pf_info.get("evaluated", 0) >= 64:
                        self._prefill_tracker.update(
                            _pf_info["evaluated"], time.monotonic() - _t0,
                        )
                    log.info(
                        "Prefill %s: total=%s reused=%s evaluated=%s cache_hit=%s in %.1fs",
                        req.request_id[:8], _pf_info.get("total"),
                        _pf_info.get("reused"), _pf_info.get("evaluated"),
                        _pf_info.get("cache_hit", False),
                        time.monotonic() - _t0,
                    )
                    # track new-prompt tokens (cache-misses only).
                    # The client's estimator uses this instead of the
                    # prompt_budget (1200-token) assumption.
                    _ev = int(_pf_info.get("evaluated", 0) or 0)
                    if _ev > 0:
                        self._avg_new_prompt_samples.append(_ev)

                self._progress[_root] = {
                    "phase": "decode", "done": 0, "total": 0,
                    "reused": _pf_info.get("reused", 0), "t": time.monotonic(),
                }

                for chunk in llm(
                        req.prompt, max_tokens=_max_tokens, temperature=temp,
                        top_p=top_p, top_k=top_k, stop=_stop, stream=True,
                        **_extra_kwargs,
                ):
                    if buf.cancel.is_set():
                        break
                    tok = chunk["choices"][0]["text"]
                    if tok:
                        if not first_token_ts:
                            first_token_ts.append(time.monotonic())
                        buf.sync_put(tok)
                        _count += 1
                    if "finish_reason" in chunk["choices"][0]:
                        captured_finish_reason = chunk["choices"][0]["finish_reason"]
            except RuntimeError as exc:
                if "cancelled" not in str(exc).lower():
                    buf.sync_put(exc)
            except Exception as exc:
                log.error("llama.cpp error after %d tokens: %s", _count, exc)
                buf.sync_put(exc)
            finally:
                # Legacy path only: when chunked prefill ran, the tracker
                # was already fed with *evaluated* tokens (which excludes
                # KV-cache hits), so feeding it again with prompt_est would
                # inflate the EWMA on partial prefix hits.
                if first_token_ts and not _pf_info:
                    try:
                        self._prefill_tracker.update(
                            _prompt_est,
                            first_token_ts[0] - produce_start,
                        )
                    except Exception:
                        pass
                if finish_reason_future and not finish_reason_future.done():
                    if captured_finish_reason in ("stop", "eos"):
                        finish_reason_future.set_result(_FINISH_REASON_STOP)
                    elif captured_finish_reason == "length":
                        finish_reason_future.set_result(_FINISH_REASON_LENGTH)
                    else:
                        if _count < _max_tokens:
                            finish_reason_future.set_result(_FINISH_REASON_STOP)
                        else:
                            finish_reason_future.set_result(_FINISH_REASON_LENGTH)
                try:
                    buf.sync_put(_DONE)
                except Exception:
                    pass

        buf.future = self._executor.submit(_produce)
        async for tok in buf:
            yield tok

    async def _hf_stream(
            self,
            model,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
            tokenizer=None,
    ) -> AsyncIterator[str]:
        try:
            from transformers import AutoTokenizer, TextIteratorStreamer
        except ImportError:
            yield "[transformers not installed]"
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.set_result(_FINISH_REASON_ERROR)
            return

        if tokenizer is None:
            try:
                name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            except Exception as exc:
                yield f"[tokenizer error: {exc}]"
                if finish_reason_future and not finish_reason_future.done():
                    finish_reason_future.set_result(_FINISH_REASON_ERROR)
                return

        try:
            import torch
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in
                      tokenizer(req.prompt, return_tensors="pt").items()}
        except Exception as exc:
            yield f"[input error: {exc}]"
            if finish_reason_future and not finish_reason_future.done():
                finish_reason_future.set_result(_FINISH_REASON_ERROR)
            return

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        loop = asyncio.get_running_loop()
        buf = AsyncTokenBuffer(loop, maxsize=512, timeout_s=req.timeout_s or 300.0)

        def _produce():
            token_count = 0
            try:
                gen_thread = threading.Thread(
                    target=model.generate,
                    kwargs=dict(
                        **inputs,
                        max_new_tokens=req.max_tokens,
                        temperature=max(req.temperature, 1e-7),
                        top_p=req.top_p,
                        do_sample=req.temperature > 0.01,
                        streamer=streamer,
                    ),
                    daemon=True,
                )
                gen_thread.start()
                for tok in streamer:
                    if buf.cancel.is_set():
                        break
                    if tok:
                        buf.sync_put(tok)
                        token_count += 1
                gen_thread.join(timeout=30.0)
            except Exception as exc:
                buf.sync_put(exc)
            finally:
                if finish_reason_future and not finish_reason_future.done():
                    # For HF, we don't have a definitive finish reason; fallback
                    if token_count < req.max_tokens:
                        finish_reason_future.set_result(_FINISH_REASON_STOP)
                    else:
                        finish_reason_future.set_result(_FINISH_REASON_LENGTH)
                try:
                    buf.sync_put(_DONE)
                except Exception:
                    pass

        buf.future = self._executor.submit(_produce)
        async for tok in buf:
            yield tok

    async def _ct_stream(
            self,
            model,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        buf = AsyncTokenBuffer(loop, maxsize=512, timeout_s=(req.timeout_s or 0.0))

        def _produce():
            token_count = 0
            try:
                for tok in model(req.prompt, max_new_tokens=req.max_tokens,
                                 temperature=req.temperature, stream=True):
                    if buf.cancel.is_set():
                        break
                    if tok:
                        buf.sync_put(tok)
                        token_count += 1
            except RuntimeError as exc:
                if "cancelled" not in str(exc).lower():
                    buf.sync_put(exc)
            except Exception as exc:
                buf.sync_put(exc)
            finally:
                if finish_reason_future and not finish_reason_future.done():
                    if token_count < req.max_tokens:
                        finish_reason_future.set_result(_FINISH_REASON_STOP)
                    else:
                        finish_reason_future.set_result(_FINISH_REASON_LENGTH)
                try:
                    buf.sync_put(_DONE)
                except Exception:
                    pass

        buf.future = self._executor.submit(_produce)
        async for tok in buf:
            yield tok

    async def _sim_stream(
            self,
            req: InferenceRequest,
            finish_reason_future: Optional[asyncio.Future] = None,
    ) -> AsyncIterator[str]:
        log.error(
            "SIM BACKEND — model='%s' req=%s — Install llama‑cpp‑python and reload.",
            req.model_id, req.request_id[:8],
        )
        import random
        words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog."]
        rng = random.Random(hash(req.prompt) & 0xFFFF)
        for _ in range(min(req.max_tokens, 25)):
            await asyncio.sleep(0.01)
            yield " " + rng.choice(words)
        if finish_reason_future and not finish_reason_future.done():
            finish_reason_future.set_result(_FINISH_REASON_STOP)

        # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        # This method takes self._lk and called
        # self.decode_tps() and self.avg_new_prompt_tokens() INSIDE that
        # block. Both of those methods take self._lk again. With a plain
        # threading.Lock, the second acquire by the same thread blocks
        # forever — and since stats() runs on the asyncio event loop, the
        # server freezes on the first call.
        #
        # Two independent defences:
        #   1. self._lk is now a threading.RLock (see __init__), so the
        #      nested acquire by the same thread succeeds.
        #   2. The lock-taking helpers are computed BEFORE entering
        #      `with self._lk:` below, so they never nest.
        # Either alone is sufficient; keeping both means reverting one
        # cannot reintroduce the bug.
        try:
            _loaded = list(self._unified_models.keys()) or [
                k for k in (getattr(self.loader, "_models", {}) or {}).keys()
            ]
        except Exception:
            _loaded = []

        # Compute lock-taking helpers BEFORE taking self._lk.
        _decode_tps = round(self.decode_tps(), 2)
        _avg_new_prompt = self.avg_new_prompt_tokens()

        with self._lk:
            nl = len(self._latencies)
            _valid_tps = [t for t in self._tps_samples if 0 < t <= 500.0]
            _nvt = len(_valid_tps)
            base = {
                "engine_version": "1.0.0",
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_errors": self._total_errors,
                "oom_recoveries": self._oom_count,
                "avg_ttft_ms": round(sum(self._latencies) / max(nl, 1), 2),
                "avg_tps": round(sum(_valid_tps) / max(_nvt, 1), 2),
                "decode_tps": _decode_tps,
                "avg_new_prompt_tokens": _avg_new_prompt,
                "p95_ttft_ms": round(self._pct(list(self._latencies), 95), 2),
                "p95_tps": round(self._pct(_valid_tps, 95), 2),
                "ngram_hit_rate": round(self._ngram.hit_rate, 3),
                "queue": self._queue.stats(),
                "loaded_models": _loaded,
                "drive_wired": self._drive_wired,
                "batch_size": (self._token_mgr.capacity
                               if self._token_mgr else self.max_batch),
                "circuit_breaker": self._breaker.stats(),
                "executor": self._executor.queue_stats(),
                "rate_limiter": self.rate_limiter.stats(),
                "gate_waiters": self._gate_waiters,
                "inflight": self._inflight,
                "bitnet_state_model": self._bitnet_state_model_id,
                "bitnet_enabled": self._bitnet_enabled,
                "segment_size": self._segment_size,
                "slots_in_use": len(self._slot_holders) if hasattr(self, "_slot_holders") else 0,
                "slot_capacity": self.max_batch,
            }
        if self._ctx_cache:
            base["context_cache"] = self._ctx_cache.stats()
        if self._tput_ctrl:
            base["throughput"] = self._tput_ctrl.stats()
        if base.get("circuit_breaker", {}).get("state") == "open":
            base["metrics_stale"] = True
            base["metrics_note"] = "Circuit breaker OPEN — avg_tps/ttft from last closed period"
        return base

    def prefill_tps(self) -> float:
        """
        Measured prefill throughput in tokens/sec (EWMA).

        Returns 0.0 when no observations have been recorded yet, so
        callers can distinguish "not measured" from "measured and it's
        slow". `_llama_stream` updates the tracker on every cache-miss
        request, so a busy server converges within a handful of calls.

        This is the value `_compute_runtime_limits` exposes to clients;
        `amb_llm_core.estimate_deadline_s` uses it when present and
        falls back to `2 × decode_tps` when it's 0.0.
        """
        try:
            t = self._prefill_tracker
            if t is None or not t.has_observations:
                return 0.0
            v = float(t.tps)
            return v if v > 0.0 else 0.0
        except Exception:
            return 0.0

    def decode_tps(self) -> float:
        """
        Measured decode-only throughput in tokens/sec (EWMA). Excludes
        prefill time.

        This is the value `_compute_runtime_limits` should expose as
        `decode_tps`, and the value AMB-APP's CapabilityProfile should
        prefer over `effective_tps` for latency math. `effective_tps`
        is end-to-end and underreports decode speed by 30-40% on this
        hardware.

        Returns 0.0 when no decode samples have been recorded yet, so
        callers can distinguish "not measured" from "measured and slow".
        """
        try:
            with self._lk:
                if not self._decode_tps_samples:
                    return 0.0
                vals = [v for v in self._decode_tps_samples if 0 < v <= 500.0]
                if not vals:
                    return 0.0
                return float(sum(vals) / len(vals))
        except Exception:
            return 0.0

    def avg_new_prompt_tokens(self) -> int:
        """
        EWMA of new prompt tokens per call (i.e. tokens the server had to
        actually prefill — KV-cache hits excluded). Defaults to 200 when
        no observations exist yet.

        The client-side latency estimator (F-06) uses this instead of
        the prompt budget. At steady state on qwen3-8b with the server's
        prefix cache, this settles to ~180-190.
        """
        try:
            with self._lk:
                if not self._avg_new_prompt_samples:
                    return 200
                vals = list(self._avg_new_prompt_samples)
                n = len(vals)
                if n == 0:
                    return 200
                # Simple trimmed mean: drop the top/bottom 10% to reject
                # outliers (cold-start 500+, warm cache 20).
                vals.sort()
                lo = max(0, n // 10)
                hi = max(lo + 1, n - n // 10)
                window = vals[lo:hi]
                return int(sum(window) / max(1, len(window)))
        except Exception:
            return 200

    @staticmethod
    def _pct(data: list, pct: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * pct / 100), len(s) - 1)]
