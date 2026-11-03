"""
local_llm_client/__init__.py

Local LLM Cloud Client Library and SDK.

This package provides synchronous and asynchronous clients for the
Local LLM Cloud API, along with CLI tools, a web UI, and an adapter
for ai_agent.py integration.

Exports:
    - LLMCloudClient (sync client)
    - AsyncLLMCloudClient (async client)
    - ClientConfig (configuration dataclass for core clients)
    - TokenMetrics, GenerationResult, ServerStatus, ModelInfo, ModelList
    - Full exception hierarchy (22+ specific types)
    - LocalLLMCloudAdapter, get_model_manager, AdaptiveContextManager
    - HTML, get_web_ui (embedded web UI)
    - __version__
"""

__version__ = "1.0.0"  # bumped to reflect production readiness

# ------------------------------------------------------------------------------
# Core clients
# ------------------------------------------------------------------------------
from .core.client import LLMCloudClient
from .core.async_client import AsyncLLMCloudClient

# ------------------------------------------------------------------------------
# Configuration & data models
# ------------------------------------------------------------------------------
from .core.models import (
    ClientConfig,
    GenerationResult,
    ServerStatus,
    TokenMetricsCompat as TokenMetrics,
    ModelInfo,
    ModelList,
)

# ------------------------------------------------------------------------------
# Exceptions (complete hierarchy)
# ------------------------------------------------------------------------------
from .core.exceptions import (
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
# Adapter & context management (ai_agent.py integration)
# ------------------------------------------------------------------------------
from .core.adapter import (
    LocalLLMCloudAdapter,
    get_model_manager,
    AdaptiveContextManager,
)

# ------------------------------------------------------------------------------
# Web UI
# ------------------------------------------------------------------------------
try:
    from .web_ui import HTML
except ImportError as e:
    import warnings
    warnings.warn(f"Web UI module not available: {e}", ImportWarning)
    HTML = ""


def get_web_ui() -> str:
    """Return the embedded Web UI HTML."""
    return HTML


# ------------------------------------------------------------------------------
# CLI Config (optional – may be used by SDK users)
# ------------------------------------------------------------------------------
try:
    from .cli.config import Config as CLIConfig
except ImportError:
    CLIConfig = None

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
__all__ = [
    # Version
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

    # Web UI
    "HTML",
    "get_web_ui",
]
