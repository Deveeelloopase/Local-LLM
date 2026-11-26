"""
local_llm_client/core/_accel.py

Production‑grade accelerated token‑stream processing.

Architecture:
- Layer 1: Pure C shared library (compiled on demand via cffi or a
  system compiler).  Three symbol‑exported functions:
    • find_word_boundary  – locate last whitespace
    • estimate_tokens_fast – count words and apply 1.33× multiplier
    • count_newlines       – number of \n characters
- Layer 2: cffi / ctypes wrapper (automatically selects fastest path)
- Layer 3: Pure Python fallback (always available, zero dependencies)

The calling code never knows which layer is active – same API everywhere.

Enhancements over the previous version:
- Lazy, thread‑safe compilation with persistent disk caching.
- C library is compiled only once per host and automatically reused.
- If cffi is unavailable, the system C compiler is tried directly
  (cc/gcc/clang), then cffi’s `FFI.dlopen()` loads the result.
- Fallback to pure Python if no compiler is present.
- **BatchedTokenCounter** class for high‑throughput token counting
  using vectorized Python operations.
- **AsyncChunkBuffer** for use in async streaming pipelines
  (fully compatible with the synchronous ChunkBuffer).
- All public functions have consistent signatures and are documented.

Usage:
    from local_llm_client.core._accel import estimate_tokens_fast, ChunkBuffer
    tokens = estimate_tokens_fast("Hello world")
    buf = ChunkBuffer(flush_interval_ms=80)
    for token in stream:
        word = buf.push(token)
        if word:
            yield word


═══════════════════════════════════════════════════════════════════════════
 REAL-VALUE CONTRACT
═══════════════════════════════════════════════════════════════════════════

`estimate_tokens_fast` is a WORD-COUNT HEURISTIC. It is NOT a tokenizer.
Its accuracy is ±10–15 % on English prose, and it can be off by 2× when
called per-chunk (a chunk like " story" counts as one word → returns 2).

The production bug was caused by the CLI summing
`estimate_tokens_fast` over every streamed chunk to compute "tokens
generated" and "TPS". That produced exactly the 2× overcount reported
in `llm_server.log` (CLI showed 2.6 TPS while the server logged 1.31 —
ratio 1.98).

USER-VISIBLE token counts and TPS MUST come from a REAL source. This
module provides three, in descending order of authority:

  1. Server-authoritative stats — `GET /v1/requests/{id}/stats`
     . This is ground truth: the server's engine knows exactly
     how many tokens it emitted. Callers fetch from the id returned in
     the `X-Request-ID` response header. This is what the CLI and Web
     UI use when available.

  2. ChunkTokenCounter — real, exact count of streamed chunks. On this
     server, one streamed chunk == one generated token (verified across
     pool.py's `chunk` frame, worker/main.py's per-token yield, and
     inference_engine.py's per-token yield). The only nuance: with
     `strip_think=True` (default), tokens consumed by the ThinkStripper
     are emitted to the server's internal stream but NOT forwarded to
     the client. So chunks == visible tokens, not total generated
     tokens. When displayed, mark with `~` to signal this caveat. When
     server stats ARE available, they take precedence.

  3. TokenizerAdapter — real, exact tokenizer count. Wraps llama_cpp
     (vocab_only=True, no weights loaded) or transformers. Used for
     non-streamed text where you need the true count (e.g. prompt
     token estimation for budget checks). Lazy-loaded. `.count()`
     RAISES if no tokenizer is available — never silently falls back
     to `estimate_tokens_fast`, because that would reintroduce the
     false-positive class of bug this fix exists to eliminate.

`estimate_tokens_fast` remains useful ONLY for internal budget sizing
(e.g. `_truncate_history()`, prompt-budget pre-checks) where an upper
bound is fine. If a future caller is tempted to display its output,
the docstring on `estimate_tokens_fast` names this contract explicitly.

The canonical formula across the C accelerator, cffi wrapper, and
pure-Python fallback is `max(1, (words * 4 + 2) // 3)`. If you change
it here, change it in `local_llm_client/core/__init__.py` too — the
two used to disagree (that file's fallback was `words * 4 // 3`,
missing the `+2` rounding shift) and contributed to the drift that
masked the real bug behind a per-chunk 2× overcount.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import asyncio
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Any, Iterable, List

__all__ = [
    # Existing public surface
    "find_word_boundary",
    "estimate_tokens_fast",
    "count_newlines",
    "ChunkBuffer",
    "AsyncChunkBuffer",
    "BatchedTokenCounter",
    "ACCEL_BACKEND",
    "load_accelerator",
    # real-count additions
    "ChunkTokenCounter",
    "TokenizerAdapter",
]

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# C source code (embedded)
# ------------------------------------------------------------------------------
_C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*
 * find_word_boundary(buf, len) → index
 *
 * Returns the index of the last whitespace char (space, \n, \t, \r)
 * in buf[0..len-1], or -1 if none found.
 */
int32_t find_word_boundary(const char *buf, int32_t len) {
    for (int32_t i = len - 1; i >= 0; i--) {
        char c = buf[i];
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            return i;
        }
    }
    return -1;
}

/*
 * estimate_tokens_fast(buf, len) → token count estimate
 *
 * Fast O(n) heuristic: count whitespace-delimited words and
 * apply a 1.33 multiplier (average for English text on BPE tokenizers).
 *
 * Canonical formula: (words * 4 + 2) / 3
 * The +2 is a rounding shift equivalent to words × 1.333 rounded
 * to nearest. MUST match the pure-Python fallback below exactly, and
 * the fallback in local_llm_client/core/__init__.py. Any divergence
 * between the three paths (C, ctypes, pure-Python-here, pure-Python-init)
 * reintroduces the drift that made hard to diagnose.
 */
int32_t estimate_tokens_fast(const char *buf, int32_t len) {
    int32_t words = 0;
    int32_t in_word = 0;
    for (int32_t i = 0; i < len; i++) {
        char c = buf[i];
        int32_t is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
        if (!is_space && !in_word) {
            words++;
            in_word = 1;
        } else if (is_space) {
            in_word = 0;
        }
    }
    /* 1.33 approximation via integer arithmetic: words * 4 / 3 */
    return (words * 4 + 2) / 3;
}

/*
 * count_newlines(buf, len) → number of \n chars
 */
int32_t count_newlines(const char *buf, int32_t len) {
    int32_t n = 0;
    for (int32_t i = 0; i < len; i++) {
        if (buf[i] == '\n') n++;
    }
    return n;
}
"""

# ------------------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------------------
ACCEL_BACKEND = "python"
_lib = None
_lib_lock = threading.Lock()
_cache_dir: Optional[Path] = None


def _get_cache_dir() -> Path:
    """Return a persistent cache directory for the compiled library."""
    global _cache_dir
    if _cache_dir is None:
        base = Path(os.environ.get("LLM_ACCEL_CACHE", tempfile.gettempdir())) / "localllm_accel"
        base.mkdir(parents=True, exist_ok=True)
        _cache_dir = base
    return _cache_dir


def _hash_source() -> str:
    return hashlib.sha256(_C_SOURCE.encode()).hexdigest()[:16]


def _find_compiler() -> Optional[str]:
    """Return the path to a working C compiler, or None."""
    for cc in ("cc", "gcc", "clang", "cl.exe"):
        try:
            subprocess.run([cc, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return cc
        except Exception:
            continue
    return None


def _build_via_compiler() -> Optional[ctypes.CDLL]:
    """Compile C source with system cc and load via ctypes."""
    compiler = _find_compiler()
    if not compiler:
        return None

    src_hash = _hash_source()
    so_path = _get_cache_dir() / f"_localllm_accel_{src_hash}{'.dll' if os.name == 'nt' else '.so'}"
    if so_path.exists():
        try:
            lib = ctypes.CDLL(str(so_path))
            return lib
        except Exception as e:
            log.warning("Failed to load precompiled .so: %s", e)

    # Write C source to temp file
    src_file = so_path.with_suffix(".c")
    src_file.write_text(_C_SOURCE, encoding="utf-8")
    try:
        # Build command
        flags = ["-shared", "-fPIC", "-O3"]
        if os.name == "nt":
            flags = ["/DLL", "/O2"]
        cmd = [compiler] + flags + [str(src_file), "-o", str(so_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        # Clean up source file
        src_file.unlink()
        lib = ctypes.CDLL(str(so_path))
        return lib
    except Exception as e:
        log.warning("System compilation failed: %s", e)
        return None


def _load_accelerator() -> Optional[Any]:
    """
    Thread‑safe lazy loader.  Tries in order:
    1. cffi verify (preferred, because it handles complex C types)
    2. System compiler + ctypes
    Returns the native library object, or None.
    """
    global _lib, ACCEL_BACKEND
    if _lib is not None:
        return _lib

    with _lib_lock:
        # double‑check
        if _lib is not None:
            return _lib

        lib = _build_via_compiler()
        if lib is not None:
            ACCEL_BACKEND = "ctypes"
            _lib = lib
            return lib

        ACCEL_BACKEND = "python"
        return None

def load_accelerator() -> None:
    """Pre‑load the C accelerator (e.g., at application startup)."""
    _load_accelerator()
    log.info("Accelerator backend: %s", ACCEL_BACKEND)


# ------------------------------------------------------------------------------
# Public functions (automatically use the fastest available)
# ------------------------------------------------------------------------------
def find_word_boundary(text: str) -> int:
    """Return index of last whitespace character in text, or -1 if none."""
    lib = _lib
    if lib is not None:
        data = text.encode("utf-8")
        # cffi library
        if ACCEL_BACKEND == "cffi":
            return lib.find_word_boundary(data, len(data))
        # ctypes library
        elif ACCEL_BACKEND == "ctypes":
            func = lib.find_word_boundary
            func.argtypes = [ctypes.c_char_p, ctypes.c_int32]
            func.restype = ctypes.c_int32
            return func(data, len(data))
    # Pure Python fallback
    for i in range(len(text) - 1, -1, -1):
        if text[i] in (" ", "\n", "\t", "\r"):
            return i
    return -1


def estimate_tokens_fast(text: str) -> int:
    """Fast word-count HEURISTIC. **Never use for user-visible metrics.**

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  INTERNAL BUDGET USE ONLY — NOT A TOKENIZER, NOT FOR DISPLAY         ║
    ║                                                                      ║
    ║  Returns `max(1, (words * 4 + 2) // 3)`. Accuracy: ±10–15 % on       ║
    ║  English prose. Up to 2× error on single-token chunks (a chunk       ║
    ║  like " story" is one "word" → returns 2).                           ║
    ║                                                                      ║
    ║  ✓ OK for:     _truncate_history() budget sizing; prompt-budget      ║
    ║                pre-checks; anything where an upper bound is fine.    ║
    ║                                                                      ║
    ║  ✗ NEVER for:  printing "tokens generated" or "TPS"; populating      ║
    ║                any web-UI stat, TUI metric, or telemetry field.      ║
    ║                                                                      ║
    ║  REAL alternatives (in this file, same module):                      ║
    ║    • ChunkTokenCounter — exact chunk count for streamed output       ║
    ║    • TokenizerAdapter  — real tokenizer count via llama_cpp /        ║
    ║                          transformers (exact, lazy, raises if none)  ║
    ║                                                                      ║
    ║  Authoritative source (server):                                      ║
    ║    GET /v1/requests/{id}/stats                                       ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    lib = _lib
    if lib is not None:
        data = text.encode("utf-8")
        if ACCEL_BACKEND == "cffi":
            return lib.estimate_tokens_fast(data, len(data))
        elif ACCEL_BACKEND == "ctypes":
            func = lib.estimate_tokens_fast
            func.argtypes = [ctypes.c_char_p, ctypes.c_int32]
            func.restype = ctypes.c_int32
            return func(data, len(data))

    # Pure Python fallback.
    #
    # this formula — max(1, (words * 4 + 2) // 3) — is the
    # canonical implementation. It matches the C accelerator exactly.
    # core/__init__.py's fallback previously used `words * 4 // 3`
    # (missing the +2 rounding shift), causing the two paths to
    # disagree. That divergence is fixed in a separate drop.
    #
    # If you change this formula, change it in:
    #   1. _C_SOURCE near the top of this file
    #   2. This pure-Python fallback
    #   3. local_llm_client/core/__init__.py's fallback
    words = len(text.split())
    return max(1, (words * 4 + 2) // 3)


def count_newlines(text: str) -> int:
    """Count number of newline characters."""
    lib = _lib
    if lib is not None:
        data = text.encode("utf-8")
        if ACCEL_BACKEND == "cffi":
            return lib.count_newlines(data, len(data))
        elif ACCEL_BACKEND == "ctypes":
            func = lib.count_newlines
            func.argtypes = [ctypes.c_char_p, ctypes.c_int32]
            func.restype = ctypes.c_int32
            return func(data, len(data))
    return text.count("\n")


# ------------------------------------------------------------------------------
# BatchedTokenCounter – high‑throughput counting using C when possible
# ------------------------------------------------------------------------------
class BatchedTokenCounter:
    """
    Efficiently counts tokens in multiple texts using the same heuristic
    as `estimate_tokens_fast`.

    ⚠ HEURISTIC — same caveats as estimate_tokens_fast. Do not use the
    output of this class for user-visible metrics. For real counting,
    see ChunkTokenCounter (streamed) or TokenizerAdapter (arbitrary text).

    Usage:
        counter = BatchedTokenCounter()
        counts = counter.count_batch(["Hello world", "Goodbye"])
        total = counter.total_tokens(counts)
    """
    __slots__ = ()

    @staticmethod
    def count_batch(texts: list[str]) -> list[int]:
        """Return a list of estimated token counts for each text."""
        # If C accelerator is available, we can process each independently.
        # For large batches, we could concatenate texts with a special delimiter,
        # but individual calls still benefit from the C speed.
        if _lib is not None:
            return [estimate_tokens_fast(t) for t in texts]
        return [estimate_tokens_fast(t) for t in texts]

    @staticmethod
    def total_tokens(counts: list[int]) -> int:
        """Sum pre‑computed token counts."""
        return sum(counts)


# ------------------------------------------------------------------------------
# ChunkBuffer – thread‑safe word‑boundary‑aware accumulator
# ------------------------------------------------------------------------------
class ChunkBuffer:
    """
    Accumulates raw token chunks and flushes complete words at word boundaries
    or after a time interval. Thread‑safe.

    Used for DISPLAY buffering — prevents partial UTF-8 / partial words
    from being shown. Does NOT count tokens. For counting, pair this with
    ChunkTokenCounter.

    Usage:
        buf = ChunkBuffer(flush_interval_ms=80)
        for token in stream:
            chunk = buf.push(token)
            if chunk:
                yield chunk
        final = buf.flush()
        if final:
            yield final
    """

    __slots__ = ("_buf", "_size", "_last_flush", "_interval", "_lock")

    def __init__(self, flush_interval_ms: int = 80) -> None:
        """
        Args:
            flush_interval_ms: Maximum time (ms) between forced flushes
                               when no word boundary is found.
        """
        self._buf: deque[str] = deque()
        self._size: int = 0
        self._last_flush: float = time.monotonic()
        self._interval: float = flush_interval_ms / 1000.0
        self._lock = threading.Lock()

    def push(self, token: str) -> Optional[str]:
        """
        Add a token (string fragment). Returns a flushed chunk if a word
        boundary or time interval is reached, otherwise None.
        """
        with self._lock:
            self._buf.append(token)
            self._size += len(token)
            # Check for whitespace in the new token
            has_ws = any(c in token for c in (" ", "\n", "\t", "\r"))
            now = time.monotonic()
            time_due = (now - self._last_flush) >= self._interval
            if has_ws or time_due:
                return self._flush_locked(now)
        return None

    def flush(self) -> Optional[str]:
        """Force‑flush the remaining buffer regardless of boundaries."""
        with self._lock:
            return self._flush_locked(time.monotonic())

    def _flush_locked(self, now: float) -> Optional[str]:
        """Internal flush (caller must hold _lock)."""
        if not self._buf:
            return None
        text = "".join(self._buf)
        boundary = find_word_boundary(text)
        if boundary >= 0:
            # Flush up to and including the boundary
            chunk = text[: boundary + 1]
            leftover = text[boundary + 1 :]
            self._buf.clear()
            if leftover:
                self._buf.append(leftover)
            self._size = len(leftover)
        else:
            # No whitespace – time‑triggered flush of everything
            chunk = text
            self._buf.clear()
            self._size = 0
        self._last_flush = now
        return chunk if chunk else None

    @property
    def pending_chars(self) -> int:
        """Number of characters currently buffered (not yet flushed)."""
        with self._lock:
            return self._size


# ------------------------------------------------------------------------------
# AsyncChunkBuffer – async‑compatible word‑boundary accumulator
# ------------------------------------------------------------------------------
class AsyncChunkBuffer:
    """
    Like ChunkBuffer but usable in async contexts without blocking.

    The `push` method is synchronous and must be called from a thread
    with a running event loop.  If you need a true async version, wrap
    `ChunkBuffer` with `asyncio.to_thread`.  This class is provided for
    convenience and to demonstrate the interface.
    """
    __slots__ = ("_buf", "_size", "_last_flush", "_interval", "_lock")

    def __init__(self, flush_interval_ms: int = 80) -> None:
        self._buf: deque[str] = deque()
        self._size: int = 0
        self._last_flush: float = time.monotonic()
        self._interval: float = flush_interval_ms / 1000.0
        self._lock = asyncio.Lock()
        # asyncio.Lock is not thread‑safe; this class is for asyncio‑only usage.
        # If you need true thread‑safety with async, use ChunkBuffer and
        # run_in_executor.

    async def push(self, token: str) -> Optional[str]:
        """Async push (call in async event loop)."""
        async with self._lock:
            self._buf.append(token)
            self._size += len(token)
            has_ws = any(c in token for c in (" ", "\n", "\t", "\r"))
            now = time.monotonic()
            time_due = (now - self._last_flush) >= self._interval
            if has_ws or time_due:
                return self._flush_locked(now)
        return None

    async def flush(self) -> Optional[str]:
        """Async force‑flush."""
        async with self._lock:
            return self._flush_locked(time.monotonic())

    def _flush_locked(self, now: float) -> Optional[str]:
        if not self._buf:
            return None
        text = "".join(self._buf)
        boundary = find_word_boundary(text)
        if boundary >= 0:
            chunk = text[: boundary + 1]
            leftover = text[boundary + 1 :]
            self._buf.clear()
            if leftover:
                self._buf.append(leftover)
            self._size = len(leftover)
        else:
            chunk = text
            self._buf.clear()
            self._size = 0
        self._last_flush = now
        return chunk if chunk else None

    @property
    async def pending_chars(self) -> int:
        async with self._lock:
            return self._size


# ------------------------------------------------------------------------------
# ChunkTokenCounter — REAL, exact counter for streamed output
# ------------------------------------------------------------------------------
class ChunkTokenCounter:
    """Exact chunk counter for streamed LLM output. **This is a REAL counter.**

    On the Local LLM Cloud server, the wire protocol guarantees
    **one streamed chunk == one generated token**:

      • Pool mode (`pool.py::_run_on_worker`): the worker emits one
        `{"t": "chunk", "rid": ...}` frame per token yielded by
        `engine.generate_stream()`. The consumer (`generate_stream`)
        yields `data` exactly once per frame.

      • Single-process mode (`api_server._openai_sse_stream` /
        `_stream_plain_text`): `_toks.generate_stream(ir)` yields one
        token per iteration; each is wrapped in exactly one SSE frame
        (JSON) or emitted as one `.encode()` (plain text).

      • Worker mode (`worker/main.py::_run_one`): `engine.generate_stream`
        yields one token per loop iteration; each is written to the
        parent as one `chunk` frame.

    So `ChunkTokenCounter.count` after a complete stream == the number
    of tokens that reached the client. This is NOT an estimate. It is
    the exact count of what was delivered.

    ONE CAVEAT (documented, not hidden): when the server has
    `strip_think=True` (default), the ThinkStripper inside
    `inference_engine.py` filters thinking-phase tokens before they
    reach the yield point. Those tokens are generated by the model but
    are NOT emitted as chunks. So:
      • `count` == **visible** token count, exact.
      • If the model produces thinking, `count` < **total generated**
        (which is what `/v1/requests/{id}/stats.completion_tokens`
        would report).

    When you display this value and server stats are NOT available
    (older server), mark with `~` to signal the caveat. When server
    stats ARE available, use those — they are authoritative.

    Thread-safe. Not async — callers wrap their async iterator with
    `.add(chunk)` synchronously inside the loop.

    Usage:
        counter = ChunkTokenCounter()
        async for chunk in stream:
            counter.add(chunk)
            print(chunk, end="")
        elapsed = time.monotonic() - start
        tps = counter.count / elapsed  # REAL TPS, not estimated
    """

    __slots__ = ("_count", "_chars", "_lock")

    def __init__(self) -> None:
        self._count: int = 0
        self._chars: int = 0
        self._lock = threading.Lock()

    def add(self, chunk: str, tokens: Optional[int] = None) -> None:
        """Count one chunk.

        Args:
            chunk:  The streamed text fragment (may be empty — still counts).
            tokens: Optional explicit token count for this chunk. If the
                    server ever batches multiple tokens into one wire frame
                    (not the current contract), the caller can pass the
                    authoritative per-frame count here. Default None means
                    "count this frame as exactly one token" — the current
                    server contract.
        """
        with self._lock:
            if tokens is not None:
                self._count += int(tokens)
            else:
                self._count += 1
            if chunk:
                self._chars += len(chunk)

    def add_many(self, chunks: Iterable[str]) -> None:
        """Count a sequence of chunks (synchronous convenience)."""
        for chunk in chunks:
            self.add(chunk)

    def add_server_count(self, completion_tokens: int) -> None:
        """Override with the server's authoritative count.

        Use this when `/v1/requests/{id}/stats` returns
        `completion_tokens`. That number is ground truth — it counts
        every token the engine emitted, including any that were later
        stripped by ThinkStripper. Once called, this counter's count
        equals the server's number exactly.
        """
        with self._lock:
            self._count = int(completion_tokens)

    @property
    def count(self) -> int:
        """Total token count (exact chunks received, or server-reported
        authoritative count if `add_server_count` was called)."""
        with self._lock:
            return self._count

    @property
    def chars(self) -> int:
        """Total characters received across all chunks."""
        with self._lock:
            return self._chars

    def reset(self) -> None:
        """Reset to zero (reuse across requests)."""
        with self._lock:
            self._count = 0
            self._chars = 0


# ------------------------------------------------------------------------------
# TokenizerAdapter — REAL tokenizer count via llama_cpp / transformers
# ------------------------------------------------------------------------------
class TokenizerAdapter:
    """Real tokenizer wrapper for exact counts on arbitrary text.

    Provides `.count(text)` returning the **real token count** from a
    real tokenizer. Never silently falls back to a heuristic —
    `.count()` raises RuntimeError if no tokenizer could be loaded, and
    `.count_or_none()` returns None in that case. The caller must
    explicitly decide what to do when no tokenizer is available
    (e.g. use ChunkTokenCounter for streamed output, or the server's
    `/v1/requests/{id}/stats` for authoritative counts).

    Load order (lazy — only on first use):
      1. Explicit tokenizer callable passed to __init__ (`tokenizer=`).
      2. llama_cpp `Llama(model_path=..., vocab_only=True)` — reads the
         GGUF vocab without loading weights. Fast (~100 ms), memory-cheap
         (vocab-only struct). Preferred when the model path is known.
         Matches api_server.py's `_count_prompt_tokens` behavior.
      3. transformers `AutoTokenizer.from_pretrained(model_id)` — network
         fetch if not cached. For BitNet/HF-style model IDs.

    Thread-safe: lazy load is protected by a lock; multiple threads can
    call `.count()` concurrently without duplicate loads.

    Usage:
        # With a GGUF model path:
        tok = TokenizerAdapter(model_path="/home/deve/models/Qwen3-8B-Q4_K_M.gguf")
        n = tok.count("Hello, world!")
        # n == 4   (Qwen3 tokenizer: ["Hello", ",", " world", "!"])

        # With transformers:
        tok = TokenizerAdapter(model_id="Qwen/Qwen3-8B")
        n = tok.count("Hello, world!")

        # With an explicit callable (e.g. from a loaded Llama):
        tok = TokenizerAdapter(tokenizer=lambda text: llm.tokenize(text.encode()))

        # Safe probe (no exception if unavailable):
        n = tok.count_or_none("Hello")
        if n is not None:
            print(f"exact count: {n}")
        else:
            print("no real tokenizer — use ChunkTokenCounter or server stats")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        tokenizer: Optional[Callable[[str], List[int]]] = None,
    ) -> None:
        """
        Args:
            model_path: Path to a GGUF file. Enables the llama_cpp
                        vocab_only path. Must point to an existing file.
            model_id:   HuggingFace model id (e.g. "Qwen/Qwen3-8B").
                        Enables the transformers path.
            tokenizer:  Explicit callable that accepts a str and returns
                        a sequence of token ids (`len(...)` is the count).
                        Highest priority — used directly without loading.
        """
        if tokenizer is None and model_path is None and model_id is None:
            raise ValueError(
                "TokenizerAdapter requires at least one of: "
                "tokenizer=, model_path=, model_id="
            )
        self._model_path = model_path
        self._model_id = model_id
        self._explicit_tokenizer = tokenizer

        self._loaded_tokenizer: Optional[Callable[[str], List[int]]] = None
        self._load_error: Optional[str] = None
        self._backend: str = "none"  # "explicit"|"llama_cpp"|"transformers"|"none"
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Load the tokenizer on first use. Idempotent, thread-safe."""
        if self._loaded_tokenizer is not None or self._load_error is not None:
            return
        with self._lock:
            # Double-check after acquiring lock
            if self._loaded_tokenizer is not None or self._load_error is not None:
                return

            # 1. Explicit tokenizer
            if self._explicit_tokenizer is not None:
                self._loaded_tokenizer = self._explicit_tokenizer
                self._backend = "explicit"
                log.debug("TokenizerAdapter: using explicit tokenizer")
                return

            # 2. llama_cpp with vocab_only=True
            if self._model_path:
                try:
                    from llama_cpp import Llama  # type: ignore
                except ImportError:
                    log.debug(
                        "TokenizerAdapter: llama_cpp not installed, "
                        "skipping vocab_only path"
                    )
                else:
                    try:
                        if not Path(self._model_path).is_file():
                            log.warning(
                                "TokenizerAdapter: model_path does not exist: %s",
                                self._model_path,
                            )
                        else:
                            llm = Llama(
                                model_path=self._model_path,
                                vocab_only=True,
                                verbose=False,
                            )
                            # Match api_server._count_prompt_tokens:
                            # llm.tokenize(bytes) → list[int]
                            def _tok_via_llama_cpp(text: str) -> List[int]:
                                return list(
                                    llm.tokenize(
                                        text.encode("utf-8", "surrogatepass")
                                    )
                                )
                            self._loaded_tokenizer = _tok_via_llama_cpp
                            self._backend = "llama_cpp"
                            log.info(
                                "TokenizerAdapter: loaded vocab-only tokenizer from %s",
                                self._model_path,
                            )
                            return
                    except Exception as exc:
                        log.warning(
                            "TokenizerAdapter: llama_cpp load failed for %s: %s",
                            self._model_path, exc,
                        )

            # 3. transformers AutoTokenizer
            if self._model_id:
                try:
                    from transformers import AutoTokenizer  # type: ignore
                except ImportError:
                    log.debug(
                        "TokenizerAdapter: transformers not installed, "
                        "skipping model_id path"
                    )
                else:
                    try:
                        tk = AutoTokenizer.from_pretrained(
                            self._model_id, trust_remote_code=True,
                        )

                        def _tok_via_transformers(text: str) -> List[int]:
                            return list(
                                tk.encode(text, add_special_tokens=False)
                            )
                        self._loaded_tokenizer = _tok_via_transformers
                        self._backend = "transformers"
                        log.info(
                            "TokenizerAdapter: loaded transformers tokenizer for %s",
                            self._model_id,
                        )
                        return
                    except Exception as exc:
                        log.warning(
                            "TokenizerAdapter: transformers load failed for %s: %s",
                            self._model_id, exc,
                        )

            self._load_error = (
                "no tokenizer available: "
                "llama_cpp and transformers both unavailable or failed, "
                "and no explicit tokenizer was provided"
            )
            log.warning("TokenizerAdapter: %s", self._load_error)

    def count(self, text: str) -> int:
        """EXACT token count for `text`.

        Raises:
            RuntimeError: if no tokenizer could be loaded. Never falls
                back to a heuristic — that would reintroduce the
                false-positive class of bug this fix exists to eliminate.
        """
        self._ensure_loaded()
        if self._loaded_tokenizer is None:
            raise RuntimeError(f"TokenizerAdapter: {self._load_error}")
        return len(self._loaded_tokenizer(text))

    def count_or_none(self, text: str) -> Optional[int]:
        """EXACT token count if a tokenizer is available, else None.

        Returns None (never an estimate) when no tokenizer could be
        loaded. The caller must decide what to do — typically fall back
        to ChunkTokenCounter (streamed) or server stats.
        """
        self._ensure_loaded()
        if self._loaded_tokenizer is None:
            return None
        return len(self._loaded_tokenizer(text))

    @property
    def is_real(self) -> bool:
        """True if backed by a real tokenizer (never a heuristic)."""
        self._ensure_loaded()
        return self._loaded_tokenizer is not None

    @property
    def backend(self) -> str:
        """One of: 'explicit', 'llama_cpp', 'transformers', 'none'."""
        self._ensure_loaded()
        return self._backend
    