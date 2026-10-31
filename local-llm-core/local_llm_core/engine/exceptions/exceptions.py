from __future__ import annotations

"""
engine/exceptions.py — Production Exception Hierarchy for Local LLM Cloud
==========================================================================
All exceptions inherit from LocalLLMError for unified error handling,
logging, and API responses.

Exception hierarchy:
    LocalLLMError (base)
      ├── ConfigError
      ├── HardwareError
      ├── ModelError
      ├── InferenceError
      ├── MemoryExhaustedError
      ├── SchedulingError
      ├── CacheError
      ├── StorageError
      ├── RateLimitError
      ├── AuthError
      ├── TimeoutError
      └── ServerError

Usage:
    • api_server.py — converts exceptions to HTTP status codes
    • inference_engine.py — wraps backend errors
    • model_loader.py — load failures with adaptive retry
    • middleware.py — auth/rate limit enforcement
    • log_handler.py — structured error logging with request_id

Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Exception Severity & HTTP Mapping
# -----------------------------------------------------------------------------
class ErrorSeverity(Enum):
    """Error classification for logging and alerting."""
    DEBUG = 0       # Transient, expected, recoverable
    INFO = 1        # Normal error flow, expected
    WARNING = 2     # Degraded service, investigate
    ERROR = 3       # Service broken, immediate attention needed
    CRITICAL = 4    # Process/kernels panic, restart required


@dataclass
class HTTPStatusMapping:
    """Maps exception to HTTP status code and retry-after header."""
    status_code: int
    retry_after_s: Optional[float] = None
    is_retryable: bool = False
    is_server_error: bool = False


# -----------------------------------------------------------------------------
# Base Exception Class
# -----------------------------------------------------------------------------
class LocalLLMError(Exception):
    """
    Base exception for all Local LLM Cloud errors.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier (e.g., "MODEL_LOAD_OOM")
        severity: ErrorSeverity enum for logging/alerting
        cause: Original exception (preserved for debugging)
        context: Additional context dict for logging/monitoring
    """
    # Default HTTP mapping — subclasses override
    _http_mapping = HTTPStatusMapping(
        status_code=500,
        retry_after_s=None,
        is_retryable=False,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.cause = cause
        self.context = context or {}

    @property
    def http_status(self) -> int:
        return self._http_mapping.status_code

    @property
    def retry_after_s(self) -> Optional[float]:
        return self._http_mapping.retry_after_s

    @property
    def is_retryable(self) -> bool:
        return self._http_mapping.is_retryable

    @property
    def is_server_error(self) -> bool:
        return self._http_mapping.is_server_error

    def to_dict(self) -> dict:
        """Serialize exception to dict for JSON API response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "severity": self.severity.name,
            "retryable": self.is_retryable,
            "context": self.context,
        }

    def __repr__(self) -> str:
        cause_str = f" (caused by {self.cause.__class__.__name__})" if self.cause else ""
        return f"{self.__class__.__name__}({self.error_code}: {self.message}){cause_str}"


# -----------------------------------------------------------------------------
# Configuration Errors
# -----------------------------------------------------------------------------
class ConfigError(LocalLLMError):
    """Configuration validation or parsing failure."""
    _http_mapping = HTTPStatusMapping(
        status_code=400,
        is_retryable=False,
        is_server_error=False,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            severity=ErrorSeverity.ERROR,
            cause=cause,
            context=context,
        )


class InvalidModelPathError(ConfigError):
    """Model file path is outside allowed_dir or doesn't exist."""
    def __init__(self, path: str, allowed_dir: str, reason: str = "") -> None:
        msg = f"Model path rejected: {path}. Allowed: {allowed_dir}"
        if reason:
            msg += f" ({reason})"
        super().__init__(
            msg,
            context={"path": path, "allowed_dir": allowed_dir, "reason": reason},
        )
        self.error_code = "INVALID_MODEL_PATH"


class InvalidConfigValueError(ConfigError):
    """Configuration parameter out of range or invalid type."""
    def __init__(
        self,
        param: str,
        value: Any,
        expected: str,
        reason: str = "",
    ) -> None:
        msg = f"Invalid config parameter '{param}': {value} (expected {expected})"
        if reason:
            msg += f". {reason}"
        super().__init__(
            msg,
            context={"parameter": param, "value": value, "expected": expected},
        )
        self.error_code = "INVALID_CONFIG_VALUE"


# -----------------------------------------------------------------------------
# Hardware Errors
# -----------------------------------------------------------------------------
class HardwareError(LocalLLMError):
    """GPU detection, backend unavailable, or hardware failure."""
    _http_mapping = HTTPStatusMapping(
        status_code=503,
        retry_after_s=60.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="HARDWARE_ERROR",
            severity=ErrorSeverity.CRITICAL,
            cause=cause,
            context=context,
        )


class GPUNotFoundError(HardwareError):
    """No compatible GPU detected for the preferred backend."""
    def __init__(self, preferred_backend: str, available: list) -> None:
        msg = (
            f"GPU not found for backend '{preferred_backend}'. "
            f"Available: {available or 'none'}"
        )
        super().__init__(
            msg,
            context={"preferred_backend": preferred_backend, "available": available},
        )
        self.error_code = "GPU_NOT_FOUND"


class BackendNotAvailableError(HardwareError):
    """Inference backend (llama_cpp, transformers, etc.) not installed."""
    def __init__(self, backend: str, reason: str = "") -> None:
        msg = f"Backend '{backend}' not available or failed to initialize"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={"backend": backend, "reason": reason},
        )
        self.error_code = "BACKEND_NOT_AVAILABLE"


class ThermalThrottleError(HardwareError):
    """GPU temperature exceeded throttle threshold."""
    def __init__(self, current_temp: float, throttle_temp: float) -> None:
        msg = (
            f"Thermal throttling: GPU at {current_temp:.1f}°C "
            f"(threshold: {throttle_temp}°C)"
        )
        super().__init__(
            msg,
            severity=ErrorSeverity.WARNING,
            context={
                "current_temperature": current_temp,
                "throttle_temperature": throttle_temp,
            },
        )
        self.error_code = "THERMAL_THROTTLE"


# -----------------------------------------------------------------------------
# Model Errors
# -----------------------------------------------------------------------------
class ModelError(LocalLLMError):
    """Model loading, incompatibility, or weight issues."""
    _http_mapping = HTTPStatusMapping(
        status_code=400,
        is_retryable=False,
        is_server_error=False,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="MODEL_ERROR",
            severity=ErrorSeverity.ERROR,
            cause=cause,
            context=context,
        )


class ModelNotFoundError(ModelError):
    """Model file does not exist or is not in allowed_dir."""
    def __init__(self, model_id: str, path: str) -> None:
        msg = f"Model '{model_id}' not found at {path}"
        super().__init__(
            msg,
            context={"model_id": model_id, "path": path},
        )
        self.error_code = "MODEL_NOT_FOUND"
        self._http_mapping = HTTPStatusMapping(404, is_retryable=False)


class ModelLoadError(ModelError):
    """Model loading failed (backend error, corrupt file, etc.)."""
    def __init__(
        self,
        model_id: str,
        reason: str,
        cause: Optional[Exception] = None,
    ) -> None:
        msg = f"Failed to load model '{model_id}': {reason}"
        super().__init__(
            msg,
            cause=cause,
            context={"model_id": model_id, "reason": reason},
        )
        self.error_code = "MODEL_LOAD_ERROR"


class IncompatibleModelError(ModelError):
    """Two models are incompatible (vocabulary mismatch, etc.)."""
    def __init__(
        self,
        model_a: str,
        model_b: str,
        reason: str,
    ) -> None:
        msg = (
            f"Models '{model_a}' and '{model_b}' are incompatible for this operation: {reason}"
        )
        super().__init__(
            msg,
            context={"model_a": model_a, "model_b": model_b, "reason": reason},
        )
        self.error_code = "INCOMPATIBLE_MODELS"


class CorruptModelError(ModelError):
    """Model file is corrupted or has invalid format."""
    def __init__(self, model_id: str, reason: str) -> None:
        msg = f"Model '{model_id}' is corrupt or invalid: {reason}"
        super().__init__(
            msg,
            context={"model_id": model_id, "reason": reason},
        )
        self.error_code = "CORRUPT_MODEL"


# -----------------------------------------------------------------------------
# Inference Errors
# -----------------------------------------------------------------------------
class InferenceError(LocalLLMError):
    """Generation, token decoding, or sampling failure."""
    _http_mapping = HTTPStatusMapping(
        status_code=500,
        retry_after_s=5.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="INFERENCE_ERROR",
            severity=ErrorSeverity.ERROR,
            cause=cause,
            context=context,
        )

class SpeculativeDecodingError(InferenceError):
    """Speculative decoding failed (e.g., vocab mismatch, model not loaded)."""
    def __init__(self, message: str, cause: Optional[Exception] = None, context: Optional[dict] = None) -> None:
        super().__init__(
            message,
            error_code="SPECULATIVE_DECODING_ERROR",
            severity=ErrorSeverity.ERROR,
            cause=cause,
            context=context,
        )


class TokenDecodingError(InferenceError):
    """Token ID out of vocabulary or decoding failed."""
    def __init__(self, token_id: int, vocab_size: int) -> None:
        msg = f"Token ID {token_id} out of vocabulary (size: {vocab_size})"
        super().__init__(
            msg,
            context={"token_id": token_id, "vocab_size": vocab_size},
        )
        self.error_code = "TOKEN_DECODING_ERROR"


class SamplingError(InferenceError):
    """Sampling failed (invalid logits, all-zero probs, etc.)."""
    def __init__(self, reason: str, context_extra: Optional[dict] = None) -> None:
        msg = f"Sampling error: {reason}"
        ctx = {"reason": reason}
        if context_extra:
            ctx.update(context_extra)
        super().__init__(msg, context=ctx)
        self.error_code = "SAMPLING_ERROR"


class GenerationInterruptedError(InferenceError):
    """Generation was stopped by user or timeout before completion."""
    def __init__(self, reason: str = "User cancelled") -> None:
        super().__init__(
            f"Generation interrupted: {reason}",
            severity=ErrorSeverity.INFO,
            context={"reason": reason},
        )
        self.error_code = "GENERATION_INTERRUPTED"
        self._http_mapping = HTTPStatusMapping(499, is_retryable=False)


class GGMLAssertionError(InferenceError):
    """llama.cpp GGML_ASSERT failed (SIGSEGV likely)."""
    def __init__(self, assertion: str, context_extra: Optional[dict] = None) -> None:
        msg = f"GGML assertion failed: {assertion}"
        ctx = {"assertion": assertion}
        if context_extra:
            ctx.update(context_extra)
        super().__init__(
            msg,
            severity=ErrorSeverity.CRITICAL,
            context=ctx,
        )
        self.error_code = "GGML_ASSERTION"


# -----------------------------------------------------------------------------
# Memory Errors
# -----------------------------------------------------------------------------
class MemoryExhaustedError(LocalLLMError):
    """Out of memory on GPU or system RAM."""
    _http_mapping = HTTPStatusMapping(
        status_code=507,
        retry_after_s=30.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        memory_type: str,  # "VRAM", "RAM", "SSD"
        needed_mb: int,
        available_mb: int,
        cause: Optional[Exception] = None,
    ) -> None:
        msg = f"Out of {memory_type}: need {needed_mb} MB, have {available_mb} MB"
        super().__init__(
            msg,
            error_code="OUT_OF_MEMORY",
            severity=ErrorSeverity.CRITICAL,
            cause=cause,
            context={
                "memory_type": memory_type,
                "needed_mb": needed_mb,
                "available_mb": available_mb,
            },
        )


class VRAMOvercommitError(MemoryExhaustedError):
    """VRAM budget exceeded — model won't fit."""
    def __init__(self, model_id: str, model_mb: int, budget_mb: int) -> None:
        super().__init__(
            "VRAM",
            model_mb,
            budget_mb,
            context={
                "model_id": model_id,
                "model_size_mb": model_mb,
                "vram_budget_mb": budget_mb,
            },
        )
        self.error_code = "VRAM_OVERCOMMIT"


class KVCacheAllocationError(MemoryExhaustedError):
    """KV cache allocation failed during inference."""
    def __init__(
        self,
        context_length: int,
        batch_size: int,
        reason: str = "",
    ) -> None:
        msg = (
            f"KV cache allocation failed for "
            f"context_length={context_length}, batch_size={batch_size}"
        )
        if reason:
            msg += f": {reason}"
        super().__init__(
            "VRAM",
            context_length * batch_size * 100,
            0,
            context={
                "context_length": context_length,
                "batch_size": batch_size,
                "reason": reason,
            },
        )
        self.error_code = "KV_CACHE_ALLOCATION"


# -----------------------------------------------------------------------------
# Scheduling Errors
# -----------------------------------------------------------------------------
class SchedulingError(LocalLLMError):
    """Request queuing, concurrency, or preemption failures."""
    _http_mapping = HTTPStatusMapping(
        status_code=429,
        retry_after_s=5.0,
        is_retryable=True,
        is_server_error=False,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="SCHEDULING_ERROR",
            severity=ErrorSeverity.WARNING,
            cause=cause,
            context=context,
        )


class QueueFullError(SchedulingError):
    """Scheduler queue is at capacity."""
    def __init__(self, queue_size: int, max_size: int) -> None:
        msg = f"Scheduler queue full ({queue_size}/{max_size}). Try again in 5s."
        super().__init__(
            msg,
            context={"queue_size": queue_size, "max_size": max_size},
        )
        self.error_code = "QUEUE_FULL"


class PreemptionFailedError(SchedulingError):
    """Request preemption failed (state save/restore error)."""
    def __init__(self, request_id: str, reason: str) -> None:
        msg = f"Preemption failed for request {request_id}: {reason}"
        super().__init__(
            msg,
            severity=ErrorSeverity.ERROR,
            context={"request_id": request_id, "reason": reason},
        )
        self.error_code = "PREEMPTION_FAILED"


# -----------------------------------------------------------------------------
# Cache Errors
# -----------------------------------------------------------------------------
class CacheError(LocalLLMError):
    """KV cache, context cache, or NVMe cache failure."""
    _http_mapping = HTTPStatusMapping(
        status_code=500,
        retry_after_s=5.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="CACHE_ERROR",
            severity=ErrorSeverity.WARNING,
            cause=cause,
            context=context,
        )


class KVCacheCorruptError(CacheError):
    """KV cache state is corrupted or incompatible."""
    def __init__(self, reason: str) -> None:
        super().__init__(
            f"KV cache corrupt: {reason}",
            context={"reason": reason},
        )
        self.error_code = "KV_CACHE_CORRUPT"


class ContextCacheExpiredError(CacheError):
    """Cached context prefix is no longer valid (model unloaded, etc.)."""
    def __init__(self, cache_key: str) -> None:
        super().__init__(
            f"Context cache entry expired: {cache_key}",
            severity=ErrorSeverity.INFO,
            context={"cache_key": cache_key},
        )
        self.error_code = "CONTEXT_CACHE_EXPIRED"


# -----------------------------------------------------------------------------
# Storage Errors
# -----------------------------------------------------------------------------
class StorageError(LocalLLMError):
    """NVMe, swap, or file I/O failures."""
    _http_mapping = HTTPStatusMapping(
        status_code=507,
        retry_after_s=10.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="STORAGE_ERROR",
            severity=ErrorSeverity.WARNING,
            cause=cause,
            context=context,
        )


class NVMeReadError(StorageError):
    """NVMe read operation failed."""
    def __init__(self, path: str, reason: str) -> None:
        msg = f"NVMe read error on {path}: {reason}"
        super().__init__(
            msg,
            context={"path": path, "reason": reason},
        )
        self.error_code = "NVME_READ_ERROR"


class SwapWriteError(StorageError):
    """Swap file write failed."""
    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Swap write error: {reason}",
            context={"reason": reason},
        )
        self.error_code = "SWAP_WRITE_ERROR"


# -----------------------------------------------------------------------------
# Rate Limit & Auth Errors
# -----------------------------------------------------------------------------
class RateLimitError(LocalLLMError):
    """Client exceeded rate limit."""
    _http_mapping = HTTPStatusMapping(
        status_code=429,
        retry_after_s=1.0,
        is_retryable=True,
        is_server_error=False,
    )

    def __init__(
        self,
        limit_type: str,  # "ip" or "api_key"
        limit_value: float,
        client_id: str,
    ) -> None:
        msg = (
            f"{limit_type.upper()} rate limit exceeded: "
            f"{limit_value} requests/sec. Try again later."
        )
        super().__init__(
            msg,
            error_code="RATE_LIMIT_EXCEEDED",
            severity=ErrorSeverity.INFO,
            context={
                "limit_type": limit_type,
                "limit_value": limit_value,
                "client_id": client_id,
            },
        )


class AuthError(LocalLLMError):
    """Authentication failed (invalid API key, missing credentials)."""
    _http_mapping = HTTPStatusMapping(
        status_code=403,
        is_retryable=False,
        is_server_error=False,
    )

    def __init__(self, reason: str = "Invalid or missing API key") -> None:
        super().__init__(
            reason,
            error_code="AUTH_ERROR",
            severity=ErrorSeverity.INFO,
            context={"reason": reason},
        )


class InvalidAPIKeyError(AuthError):
    """API key is invalid or revoked."""
    def __init__(self) -> None:
        super().__init__("Invalid X-API-Key header")
        self.error_code = "INVALID_API_KEY"


class MissingAPIKeyError(AuthError):
    """API key is required but missing."""
    def __init__(self) -> None:
        super().__init__("X-API-Key header required")
        self.error_code = "MISSING_API_KEY"


# -----------------------------------------------------------------------------
# Timeout Errors
# -----------------------------------------------------------------------------
class TimeoutError(LocalLLMError):
    """Request or inference timeout."""
    _http_mapping = HTTPStatusMapping(
        status_code=504,
        retry_after_s=5.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        operation: str,
        timeout_s: float,
    ) -> None:
        msg = f"{operation} timed out after {timeout_s}s"
        super().__init__(
            msg,
            error_code="TIMEOUT",
            severity=ErrorSeverity.WARNING,
            context={"operation": operation, "timeout_s": timeout_s},
        )


class ModelLoadTimeoutError(TimeoutError):
    """Model loading exceeded timeout."""
    def __init__(self, model_id: str, timeout_s: float) -> None:
        super().__init__(
            f"Model '{model_id}' load",
            timeout_s,
        )
        self.error_code = "MODEL_LOAD_TIMEOUT"


class InferenceTimeoutError(TimeoutError):
    """Inference (generation) exceeded timeout."""
    def __init__(self, timeout_s: float) -> None:
        super().__init__("Inference (generation)", timeout_s)
        self.error_code = "INFERENCE_TIMEOUT"


# -----------------------------------------------------------------------------
# Server State Errors
# -----------------------------------------------------------------------------
class ServerError(LocalLLMError):
    """Circuit breaker open, service unavailable."""
    _http_mapping = HTTPStatusMapping(
        status_code=503,
        retry_after_s=60.0,
        is_retryable=True,
        is_server_error=True,
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message,
            error_code="SERVER_ERROR",
            severity=ErrorSeverity.CRITICAL,
            cause=cause,
            context=context,
        )


class CircuitBreakerOpenError(ServerError):
    """Circuit breaker is open — service recovering."""
    def __init__(self, recovery_s: float = 60.0) -> None:
        msg = (
            f"Service temporarily unavailable due to high error rate. "
            f"Recovery in ~{recovery_s}s. Retry later."
        )
        super().__init__(
            msg,
            severity=ErrorSeverity.WARNING,
            context={"recovery_seconds": recovery_s},
        )
        self.error_code = "CIRCUIT_BREAKER_OPEN"


class ServiceUnavailableError(ServerError):
    """Server is starting up, shutting down, or otherwise unavailable."""
    def __init__(self, reason: str = "Service unavailable") -> None:
        super().__init__(
            reason,
            severity=ErrorSeverity.WARNING,
            context={"reason": reason},
        )
        self.error_code = "SERVICE_UNAVAILABLE"


# -----------------------------------------------------------------------------
# Error Handler Utilities
# -----------------------------------------------------------------------------
def exception_to_http_response(exc: Exception) -> tuple[int, dict]:
    """
    Convert any exception to (status_code, response_body) for HTTP API.

    Usage in FastAPI:
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            status, body = exception_to_http_response(exc)
            return JSONResponse(status_code=status, content=body)
    """
    if isinstance(exc, LocalLLMError):
        status = exc.http_status
        body = exc.to_dict()
        if exc.retry_after_s:
            body["retry_after_seconds"] = exc.retry_after_s
    else:
        # Unknown exception — treat as 500 server error
        status = 500
        body = {
            "error": "INTERNAL_SERVER_ERROR",
            "message": str(exc),
            "severity": ErrorSeverity.ERROR.name,
            "retryable": False,
        }
    return status, body


def log_exception(
    exc: Exception,
    request_id: str = "",
    operation: str = "",
) -> None:
    """
    Structured logging for exceptions with request_id context.

    Usage:
        try:
            ...
        except LocalLLMError as e:
            log_exception(e, request_id=req_id, operation="model_load")
    """
    if isinstance(exc, LocalLLMError):
        severity_name = exc.severity.name
        if exc.severity == ErrorSeverity.DEBUG:
            log_func = log.debug
        elif exc.severity == ErrorSeverity.INFO:
            log_func = log.info
        elif exc.severity == ErrorSeverity.WARNING:
            log_func = log.warning
        elif exc.severity == ErrorSeverity.ERROR:
            log_func = log.error
        else:  # CRITICAL
            log_func = log.critical

        log_func(
            "%s [%s] %s: %s | context=%s",
            severity_name,
            request_id or "no-id",
            operation or exc.error_code,
            exc.message,
            exc.context,
            exc_info=exc.cause,
        )
    else:
        log.error(
            "UNKNOWN_ERROR [%s] %s: %s",
            request_id or "no-id",
            operation,
            str(exc),
            exc_info=exc,
        )


# -----------------------------------------------------------------------------
# Public exports (for __init__.py)
# -----------------------------------------------------------------------------
__all__ = [
    "LocalLLMError",
    "ErrorSeverity",
    "HTTPStatusMapping",
    "ConfigError",
    "InvalidModelPathError",
    "InvalidConfigValueError",
    "HardwareError",
    "GPUNotFoundError",
    "BackendNotAvailableError",
    "ThermalThrottleError",
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "IncompatibleModelError",
    "CorruptModelError",
    "InferenceError",
    "SpeculativeDecodingError",
    "TokenDecodingError",
    "SamplingError",
    "GenerationInterruptedError",
    "GGMLAssertionError",
    "MemoryExhaustedError",
    "VRAMOvercommitError",
    "KVCacheAllocationError",
    "SchedulingError",
    "QueueFullError",
    "PreemptionFailedError",
    "CacheError",
    "KVCacheCorruptError",
    "ContextCacheExpiredError",
    "StorageError",
    "NVMeReadError",
    "SwapWriteError",
    "RateLimitError",
    "AuthError",
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
    "TimeoutError",
    "ModelLoadTimeoutError",
    "InferenceTimeoutError",
    "ServerError",
    "CircuitBreakerOpenError",
    "ServiceUnavailableError",
    "exception_to_http_response",
    "log_exception",
]
