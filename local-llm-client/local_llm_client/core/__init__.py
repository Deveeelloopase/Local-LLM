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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# ------------------------------------------------------------------------------
# Version
# ------------------------------------------------------------------------------
__version__ = "2.0.0"

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
# Accelerated helpers (ChunkBuffer is used by clients internally,
# but users may also construct it manually)
# ------------------------------------------------------------------------------
try:
    from ._accel import (
        ChunkBuffer,
        BatchedTokenCounter,
        ACCEL_BACKEND,
        load_accelerator,
        estimate_tokens_fast,
        find_word_boundary,
    )
except ImportError:
    # Fallback if _accel is completely unavailable (should never happen,
    # because the .py file itself provides pure‑Python fallbacks, but
    # we guard nonetheless)
    ChunkBuffer = None
    BatchedTokenCounter = None
    ACCEL_BACKEND = "python"
    def load_accelerator() -> None:
        pass
    def estimate_tokens_fast(text: str) -> int:
        return max(1, len(text.split()) * 4 // 3)
    def find_word_boundary(text: str) -> int:
        for i in range(len(text) - 1, -1, -1):
            if text[i] in (" ", "\n", "\t", "\r"):
                return i
        return -1

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
