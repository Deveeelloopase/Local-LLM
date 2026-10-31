"""
local_llm_client/core/exceptions.py

Production‑grade exception hierarchy for the Local LLM Cloud client.

Every exception inherits from LLMCloudError.  The hierarchy mirrors the
server's HTTP semantics and provides helper methods for:

- Automatic mapping to HTTP status codes
- Serialisation (to_dict / from_dict) for structured logging
- Retry‑ability indicators
- Factory method from_http_status for constructing exceptions from raw responses

All aliases for backward compatibility (APIError, CircuitBreakerOpenError) are
preserved.  No hard‑coded paths – pure exception definitions.

Usage:
    from local_llm_client.core.exceptions import AuthError, RateLimitError

    try:
        ...
    except RateLimitError as exc:
        print(f"Retry after {exc.retry_after} seconds")
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Type, Union


# ------------------------------------------------------------------------------
# Base exception
# ------------------------------------------------------------------------------
class LLMCloudError(Exception):
    """Base class for all exceptions raised by the Local LLM Cloud client."""

    http_status: ClassVar[int] = 500
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary for structured logging."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "http_status": self.http_status,
            "retryable": self.retryable,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMCloudError:
        """Reconstruct from a dictionary created by to_dict()."""
        exc_type = data.get("type", "LLMCloudError")
        for klass in _ALL_EXCEPTIONS:
            if klass.__name__ == exc_type:
                return klass(data.get("message", ""), details=data.get("details"))
        return cls(data.get("message", ""), details=data.get("details"))


# ------------------------------------------------------------------------------
# Connectivity errors
# ------------------------------------------------------------------------------
class ConnectionError(LLMCloudError):
    """Server unreachable, refused connection, or network error."""
    http_status = 502
    retryable = True


class TimeoutError(LLMCloudError):
    """Request exceeded the configured timeout (client‑side or server‑side)."""
    http_status = 504
    retryable = True


class SSLError(ConnectionError):
    """SSL/TLS verification or handshake failure."""
    http_status = 526
    retryable = False


class ProxyError(ConnectionError):
    """Proxy connection or authentication failed."""
    http_status = 502
    retryable = False


# ------------------------------------------------------------------------------
# Authentication / authorisation errors
# ------------------------------------------------------------------------------
class AuthError(LLMCloudError):
    """Authentication failed – API key missing, invalid, or rejected (HTTP 401/403)."""
    http_status = 403
    retryable = False


class AuthenticationExpiredError(AuthError):
    """API key has expired (HTTP 401 with expiry message)."""
    http_status = 401
    retryable = False


class InvalidAPIKeyError(AuthError):
    """The provided API key does not exist or is malformed."""
    http_status = 403
    retryable = False


class MasterKeyRequiredError(AuthError):
    """The operation requires a master key that was not supplied."""
    http_status = 403
    retryable = False


# ------------------------------------------------------------------------------
# Model / resource errors
# ------------------------------------------------------------------------------
class ModelNotLoadedError(LLMCloudError):
    """No model is loaded on the server (typically HTTP 503 from /ready)."""
    http_status = 503
    retryable = True  # because the model may finish loading


class ModelAlreadyLoadedError(LLMCloudError):
    """A model with the same ID is already loaded."""
    http_status = 409
    retryable = False


class ModelTimeoutError(ModelNotLoadedError):
    """The model did not finish loading within the expected time."""
    http_status = 504
    retryable = True


class ModelBusyError(LLMCloudError):
    """The model is currently processing another request (queue full)."""
    http_status = 503
    retryable = True


# ------------------------------------------------------------------------------
# Rate limiting & throttling
# ------------------------------------------------------------------------------
class RateLimitError(LLMCloudError):
    """
    Rate limit exceeded (HTTP 429).  The `retry_after` attribute indicates
    how many seconds to wait before retrying.
    """
    http_status = 429
    retryable = True

    def __init__(self, message: str, retry_after: float = 0.0, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details=details)
        self.retry_after = retry_after

    def __str__(self) -> str:
        if self.retry_after > 0:
            return f"{super().__str__()} (retry after {self.retry_after:.1f}s)"
        return super().__str__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["retry_after"] = self.retry_after
        return d


class QueueFullError(RateLimitError):
    """Server inference queue is full (HTTP 429 or 503 with 'queue' in message)."""
    http_status = 503
    retryable = True


# ------------------------------------------------------------------------------
# Server‑side errors
# ------------------------------------------------------------------------------
class ServerError(LLMCloudError):
    """Server returned an unexpected 5xx error."""
    http_status = 500
    retryable = True


class ServiceUnavailableError(ServerError):
    """Server is temporarily unavailable (HTTP 503, not model‑related)."""
    http_status = 503
    retryable = True


class InternalServerError(ServerError):
    """Generic internal server error (HTTP 500)."""
    http_status = 500
    retryable = True


class BadGatewayError(ServerError):
    """Server acting as gateway received an invalid response (HTTP 502)."""
    http_status = 502
    retryable = True


# ------------------------------------------------------------------------------
# Client‑side errors
# ------------------------------------------------------------------------------
class BadRequestError(LLMCloudError):
    """HTTP 400 – malformed request, invalid parameters, or unsupported operation."""
    http_status = 400
    retryable = False


class NotFoundError(LLMCloudError):
    """HTTP 404 – requested endpoint or resource does not exist."""
    http_status = 404
    retryable = False


class MethodNotAllowedError(LLMCloudError):
    """HTTP 405 – method not allowed for the requested endpoint."""
    http_status = 405
    retryable = False


class ConflictError(LLMCloudError):
    """HTTP 409 – request conflicts with current server state."""
    http_status = 409
    retryable = False


class UnsupportedMediaTypeError(LLMCloudError):
    """HTTP 415 – unsupported content type."""
    http_status = 415
    retryable = False


class PayloadTooLargeError(LLMCloudError):
    """HTTP 413 – request body exceeds server limit."""
    http_status = 413
    retryable = False


class TooManyRequestsError(RateLimitError):
    """HTTP 429 – generic rate limiting (non‑queue)."""
    http_status = 429
    retryable = True


# ------------------------------------------------------------------------------
# Streaming errors
# ------------------------------------------------------------------------------
class StreamError(LLMCloudError):
    """Error occurred while reading a streaming response (network or decoding)."""
    http_status = 500
    retryable = True


class StreamInterruptedError(StreamError):
    """The connection dropped *after* part of the answer was delivered.

    This is what a server restart, crash or kill looks like mid-generation.
    The client cannot transparently retry (that would print the answer twice),
    so it surfaces what it already received instead of a raw network error.

    Attributes:
        partial_text: Content already yielded to the caller before the drop.
        tokens: Rough token count of ``partial_text``.
    """
    # Own code on purpose: _STATUS_MAP below is built from http_status, and
    # reusing 500 would change what from_http_status(500) returns.
    http_status = 499
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        partial_text: str = "",
        tokens: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.partial_text = partial_text
        self.tokens = tokens


# ------------------------------------------------------------------------------
# Configuration & client‑state errors
# ------------------------------------------------------------------------------
class ConfigError(LLMCloudError):
    """Invalid client configuration (e.g., missing URL, invalid pool size)."""
    http_status = 500  # client‑side, but for logging consistency
    retryable = False


class CircuitOpenError(LLMCloudError):
    """
    Client‑side circuit breaker is open – server was recently unreachable
    or failing repeatedly. Requests are being blocked to prevent further load.
    """
    http_status = 503
    retryable = True


# ------------------------------------------------------------------------------
# Factory from HTTP status
# ------------------------------------------------------------------------------
_STATUS_MAP: Dict[int, Type[LLMCloudError]] = {}

for _cls in list(globals().values()):
    if isinstance(_cls, type) and issubclass(_cls, LLMCloudError) and hasattr(_cls, "http_status"):
        code = getattr(_cls, "http_status")
        # Prefer more specific exceptions over generic ones (e.g., QueueFullError over RateLimitError)
        if code not in _STATUS_MAP or _cls.__name__.endswith("Error"):
            _STATUS_MAP[code] = _cls

_ALL_EXCEPTIONS = list(_STATUS_MAP.values())


def from_http_status(status_code: int, message: str = "", **kwargs: Any) -> LLMCloudError:
    """
    Construct the most appropriate exception for a given HTTP status code.

    Args:
        status_code: HTTP status code from the server response.
        message: Optional error message from the response body.
        **kwargs: Additional keyword arguments passed to the exception constructor.

    Returns:
        An instance of a subclass of LLMCloudError.
    """
    exc_class = _STATUS_MAP.get(status_code, LLMCloudError)
    if exc_class is RateLimitError and "retry_after" not in kwargs:
        return exc_class(message, **kwargs)
    return exc_class(message, **kwargs)


# ------------------------------------------------------------------------------
# Backward‑compatibility aliases
# ------------------------------------------------------------------------------
APIError = AuthError
CircuitBreakerOpenError = CircuitOpenError
