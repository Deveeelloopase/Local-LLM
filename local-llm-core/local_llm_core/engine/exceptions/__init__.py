"""
Local LLM Cloud Exception Hierarchy
===================================
Provides a comprehensive set of typed exceptions for all layers of the inference
engine. All exceptions inherit from `LocalLLMError` and include:
- Machine-readable error codes
- Severity levels for logging/alerting
- HTTP status code mappings for API responses
- Retryability hints

Usage:
    from local_llm_core.engine.exceptions import (
        LocalLLMError,
        ModelNotFoundError,
        InferenceTimeoutError,
        exception_to_http_response,
        log_exception,
    )

    try:
        ...
    except ModelNotFoundError as e:
        # Handle gracefully
        log_exception(e, request_id="abc123", operation="model_load")
        status, body = exception_to_http_response(e)
        return JSONResponse(status_code=status, content=body)

Version: 1.0.0
"""

from .exceptions import (
    # Base
    LocalLLMError,
    ErrorSeverity,
    HTTPStatusMapping,
    # Config
    ConfigError,
    InvalidModelPathError,
    InvalidConfigValueError,
    # Hardware
    HardwareError,
    GPUNotFoundError,
    BackendNotAvailableError,
    ThermalThrottleError,
    # Model
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    IncompatibleModelError,
    CorruptModelError,
    # Inference
    InferenceError,
    SpeculativeDecodingError,
    TokenDecodingError,
    SamplingError,
    GenerationInterruptedError,
    GGMLAssertionError,
    # Memory
    MemoryExhaustedError,
    VRAMOvercommitError,
    KVCacheAllocationError,
    # Scheduling
    SchedulingError,
    QueueFullError,
    PreemptionFailedError,
    # Cache
    CacheError,
    KVCacheCorruptError,
    ContextCacheExpiredError,
    # Storage
    StorageError,
    NVMeReadError,
    SwapWriteError,
    # Rate Limit & Auth
    RateLimitError,
    AuthError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    # Timeout
    TimeoutError,
    ModelLoadTimeoutError,
    InferenceTimeoutError,
    # Server
    ServerError,
    CircuitBreakerOpenError,
    ServiceUnavailableError,
    # Utilities
    exception_to_http_response,
    log_exception,
)

__version__ = "1.0.0"

__all__ = [
    # Base
    "LocalLLMError",
    "ErrorSeverity",
    "HTTPStatusMapping",
    # Config
    "ConfigError",
    "InvalidModelPathError",
    "InvalidConfigValueError",
    # Hardware
    "HardwareError",
    "GPUNotFoundError",
    "BackendNotAvailableError",
    "ThermalThrottleError",
    # Model
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "IncompatibleModelError",
    "CorruptModelError",
    # Inference
    "InferenceError",
    "SpeculativeDecodingError"
    "TokenDecodingError",
    "SamplingError",
    "GenerationInterruptedError",
    "GGMLAssertionError",
    # Memory
    "MemoryExhaustedError",
    "VRAMOvercommitError",
    "KVCacheAllocationError",
    # Scheduling
    "SchedulingError",
    "QueueFullError",
    "PreemptionFailedError",
    # Cache
    "CacheError",
    "KVCacheCorruptError",
    "ContextCacheExpiredError",
    # Storage
    "StorageError",
    "NVMeReadError",
    "SwapWriteError",
    # Rate Limit & Auth
    "RateLimitError",
    "AuthError",
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
    # Timeout
    "TimeoutError",
    "ModelLoadTimeoutError",
    "InferenceTimeoutError",
    # Server
    "ServerError",
    "CircuitBreakerOpenError",
    "ServiceUnavailableError",
    # Utilities
    "exception_to_http_response",
    "log_exception",
    "__version__",
]
