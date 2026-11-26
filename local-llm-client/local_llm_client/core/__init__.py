"""
local_llm_client/core/__init__.py

Production‑grade public API surface for the Local LLM Cloud client core.

This module exposes every symbol needed by both the CLI and the SDK,
including the full exception hierarchy, data models for inference results,
server status, and token metrics.  All imports are safe and lazy where
appropriate to avoid early dependency resolution.

Exports:
    - LLMCloudClient (synchronous client)
    - AsyncLLMCloudClient (asynchronous client)
    - ClientConfig (configuration dataclass)
    - GenerationResult, ServerStatus, TokenMetrics (data models)
    - ModelInfo, ModelList (model listing models)
    - ChunkBuffer, BatchedTokenCounter (accelerated stream helpers)
    - ChunkTokenCounter, TokenizerAdapter (REAL token counters)
    - LocalLLMCloudAdapter (ai_agent.py compatible adapter)
    - get_model_manager (singleton for adaptive context management)
    - AdaptiveContextManager (context window manager)
    - All exceptions (22 specific types + base LLMCloudError)
    - Legacy aliases (APIError, CircuitBreakerOpenError)
    - ACCEL_BACKEND (info about the token‑counting backend)
    - load_accelerator (function to force C acceleration)
    - __version__ (package version string)

Usage:
    from local_llm_client.core import LLMCloudClient, ClientConfig
    config = ClientConfig(base_url="http://localhost:8080")
    client = LLMCloudClient(config)
    response = client.generate("Hello")


— REAL-VALUE CONTRACT
─────────────────────────────────────────────────────────────
The exports in this module fall into two categories. Do not confuse them.

  ⓘ HEURISTIC (internal budget use only, NEVER user-visible)
      • estimate_tokens_fast   — word-count heuristic, ±10–15 % on prose,
                                 2× error on single-token chunks.
      • BatchedTokenCounter    — same heuristic in batched form.
    Use ONLY for: _truncate_history, prompt-budget pre-checks.
    NEVER for:   TPS, "tokens generated", any web-UI stat, any TUI metric.

  ⓘ REAL (safe for user-visible metrics)
      • ChunkTokenCounter      — exact count of streamed chunks. On this
                                 server, one chunk == one generated token
                                 (verified across pool.py, worker/main.py,
                                 inference_engine.py). Caveat: with
                                 strip_think=True (default), this counts
                                 VISIBLE tokens, not total generated —
                                 mark display with `~` if you have no
                                 server-side stats.
      • TokenizerAdapter       — real tokenizer (llama_cpp vocab_only or
                                 transformers). Exact count for arbitrary
                                 text. Raises if no tokenizer available —
                                 never silently degrades to a heuristic.

  ⓘ AUTHORITATIVE (server-provided)
      GET /v1/requests/{id}/stats — ground truth, includes completion_tokens
                                    that already excludes the "visible
                                    only" caveat above. Prefer this when
                                    available.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# ------------------------------------------------------------------------------
# Version
# ------------------------------------------------------------------------------
__version__ = "1.0.0"

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Clients
# ------------------------------------------------------------------------------
from .client import LLMCloudClient
from .async_client import AsyncLLMCloudClient

# ------------------------------------------------------------------------------
# Configuration & data models
# ------------------------------------------------------------------------------
from .models import (
    ClientConfig,
    GenerationResult,
    ServerStatus,
    TokenMetricsCompat as TokenMetrics,
    ModelInfo,
    ModelList,
)

# ------------------------------------------------------------------------------
# Accelerated helpers
#
# ChunkBuffer is used by clients internally; users may also construct it
# manually. The two new entries — ChunkTokenCounter and TokenizerAdapter —
# are the "real counter" additions. See the module docstring above
# for the difference between heuristic and real token counting.
# ------------------------------------------------------------------------------
try:
    from ._accel import (
        # Existing surface
        ChunkBuffer,
        BatchedTokenCounter,
        ACCEL_BACKEND,
        load_accelerator,
        estimate_tokens_fast,
        find_word_boundary,
        # real counters (never fall back to a heuristic)
        ChunkTokenCounter,
        TokenizerAdapter,
    )
except ImportError:
    # Fallback if _accel is completely unavailable.
    #
    # _accel.py is a pure-Python file that provides its own fallbacks, so
    # this branch should never fire in practice. It exists so that a
    # future packaging mistake (missing .py) degrades cleanly rather than
    # breaking every `from local_llm_client.core import ...` site in the
    # codebase.
    ChunkBuffer = None
    BatchedTokenCounter = None
    ACCEL_BACKEND = "python"

    def load_accelerator() -> None:
        pass

    def estimate_tokens_fast(text: str) -> int:
        """Word-count HEURISTIC (fallback path).

        MUST match _accel.py's fallback exactly. The previous version
        of this file used `words * 4 // 3` (missing the +2 rounding shift)
        while _accel.py used `(words * 4 + 2) // 3`. For 1-word chunks —
        exactly the per-chunk case the CLI used for TPS — the two
        disagreed by 2×: this file returned 1, _accel returned 2. Every
        code path that hit this fallback silently under-reported by 1
        token per chunk while the CLI's _accel-based path over-reported
        by 2×; the two errors partially cancelled in comparisons, making
        the real bug harder to diagnose.

        ⚠ INTERNAL BUDGET USE ONLY. Never display. For real counts use
        ChunkTokenCounter (streamed) or TokenizerAdapter (arbitrary text).
        """
        return max(1, (len(text.split()) * 4 + 2) // 3)

    def find_word_boundary(text: str) -> int:
        for i in range(len(text) - 1, -1, -1):
            if text[i] in (" ", "\n", "\t", "\r"):
                return i
        return -1

    # ── fallback stubs for the real-counter symbols ─────────────────
    # These are functionally equivalent to the _accel.py implementations
    # (they are short enough that duplicating them here is safer than
    # relying on a conditional import inside every caller). The primary
    # implementations live in _accel.py — see that module for the
    # authoritative versions with full docstrings.

    class ChunkTokenCounter:  # type: ignore[no-redef]
        """Exact chunk counter (fallback when _accel unavailable).

        Functionally identical to _accel.ChunkTokenCounter. This stub exists
        only so that `from local_llm_client.core import ChunkTokenCounter`
        does not raise ImportError in the pathological "no _accel.py" case.
        """
        __slots__ = ("_count", "_chars")

        def __init__(self) -> None:
            self._count = 0
            self._chars = 0

        def add(self, chunk: str, tokens: Optional[int] = None) -> None:
            self._count += int(tokens) if tokens is not None else 1
            if chunk:
                self._chars += len(chunk)

        def add_many(self, chunks) -> None:
            for c in chunks:
                self.add(c)

        def add_server_count(self, completion_tokens: int) -> None:
            self._count = int(completion_tokens)

        @property
        def count(self) -> int:
            return self._count

        @property
        def chars(self) -> int:
            return self._chars

        def reset(self) -> None:
            self._count = 0
            self._chars = 0

    class TokenizerAdapter:  # type: ignore[no-redef]
        """Stub tokenizer adapter (fallback when _accel unavailable).

        count() raises. count_or_none() returns None. NEVER falls back to
        a heuristic — that would reintroduce the exact class of bug this
        fix exists to eliminate. Callers must handle the unavailable case
        explicitly (use ChunkTokenCounter, server stats, or surface the
        error to the user).
        """

        def __init__(self, **kwargs) -> None:
            pass

        def count(self, text: str) -> int:
            raise RuntimeError(
                "TokenizerAdapter unavailable: _accel module failed to import. "
                "For streamed output use ChunkTokenCounter. For arbitrary "
                "text install the server's GGUF path or a transformers-compatible "
                "tokenizer."
            )

        def count_or_none(self, text: str) -> Optional[int]:
            return None

        @property
        def is_real(self) -> bool:
            return False

        @property
        def backend(self) -> str:
            return "none"

# ------------------------------------------------------------------------------
# Adapter & context management
# ------------------------------------------------------------------------------
from .adapter import (
    LocalLLMCloudAdapter,
    get_model_manager,
    AdaptiveContextManager,
)

# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------
from .exceptions import (
    LLMCloudError,
    ConnectionError,
    TimeoutError,
    SSLError,
    ProxyError,
    AuthError,
    AuthenticationExpiredError,
    InvalidAPIKeyError,
    MasterKeyRequiredError,
    ModelNotLoadedError,
    ModelAlreadyLoadedError,
    ModelTimeoutError,
    ModelBusyError,
    RateLimitError,
    QueueFullError,
    ServerError,
    ServiceUnavailableError,
    InternalServerError,
    BadGatewayError,
    BadRequestError,
    NotFoundError,
    MethodNotAllowedError,
    ConflictError,
    UnsupportedMediaTypeError,
    PayloadTooLargeError,
    TooManyRequestsError,
    StreamError,
    StreamInterruptedError,
    ConfigError,
    CircuitOpenError,
    # Legacy aliases
    APIError,
    CircuitBreakerOpenError,
    # Factory
    from_http_status,
)

# ------------------------------------------------------------------------------
# Lazy imports for optional features (if ever needed)
# ------------------------------------------------------------------------------
# The pool module is not imported by default as it duplicates functionality
# already in client.py.  If someone wants to use a standalone pool, they can
# import it directly: from local_llm_client.core.pool import ConnectionPool
# However, we expose the class reference for introspection tools.
if TYPE_CHECKING:
    from .pool import ConnectionPool, AsyncConnectionPool
else:
    ConnectionPool = None
    AsyncConnectionPool = None

# ------------------------------------------------------------------------------
# Public API surface (exhaustive)
# ------------------------------------------------------------------------------
__all__ = [
    "__version__",

    # Clients
    "LLMCloudClient",
    "AsyncLLMCloudClient",

    # Configuration & models
    "ClientConfig",
    "GenerationResult",
    "ServerStatus",
    "TokenMetrics",
    "ModelInfo",
    "ModelList",

    # Streaming & acceleration
    "ChunkBuffer",
    "BatchedTokenCounter",
    "ACCEL_BACKEND",
    "load_accelerator",
    "estimate_tokens_fast",
    "find_word_boundary",
    # real counters (never fall back to a heuristic)
    "ChunkTokenCounter",
    "TokenizerAdapter",

    # Adapter & context management
    "LocalLLMCloudAdapter",
    "get_model_manager",
    "AdaptiveContextManager",

    # Exceptions (complete hierarchy)
    "LLMCloudError",
    "ConnectionError",
    "TimeoutError",
    "SSLError",
    "ProxyError",
    "AuthError",
    "AuthenticationExpiredError",
    "InvalidAPIKeyError",
    "MasterKeyRequiredError",
    "ModelNotLoadedError",
    "ModelAlreadyLoadedError",
    "ModelTimeoutError",
    "ModelBusyError",
    "RateLimitError",
    "QueueFullError",
    "ServerError",
    "ServiceUnavailableError",
    "InternalServerError",
    "BadGatewayError",
    "BadRequestError",
    "NotFoundError",
    "MethodNotAllowedError",
    "ConflictError",
    "UnsupportedMediaTypeError",
    "PayloadTooLargeError",
    "TooManyRequestsError",
    "StreamError",
    "StreamInterruptedError",
    "ConfigError",
    "CircuitOpenError",
    # Legacy aliases
    "APIError",
    "CircuitBreakerOpenError",
    # Factory
    "from_http_status",
]
