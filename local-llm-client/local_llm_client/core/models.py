"""
local_llm_client/core/models.py

Production‑grade data structures for the Local LLM Cloud client.

All dataclasses include comprehensive validation in __post_init__,
serialisation helpers (to_dict / from_dict), and type annotations
that match both the server’s REST API and the internal adapter
interfaces.  No hard‑coded paths – all configuration is passed
via ClientConfig.

Enhancements in this version:
- ModelInfo now carries a ``status`` field (ready, loading, error).
- ClientConfig is hardened with additional HTTP tuning parameters,
  circuit breaker configuration, and token metrics fallback.
- Immutable dataclasses (frozen) where mutation is unnecessary.
- Convenience methods: ClientConfig.from_env(), with_overrides(),
  to_headers() for API key injection.
- Enums for ModelStatus, FinishReason, ModelType.
- Drop‑in compatibility with ai_agent.py TokenMetrics via
  TokenMetricsCompat.
- Full serialisation support for streaming, batching, and
  token budget estimation.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    OVERLOADED = "overloaded"


class ModelType(str, Enum):
    GGUF = "gguf"
    BITNET = "bitnet"
    AUTO = "auto"


class ModelStatus(str, Enum):
    """Represents the current readiness state of a model."""
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Token Metrics (compatibility with model_manager)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenMetricsCompat:
    """
    Drop‑in replacement for model_manager.TokenMetrics.

    Same fields, same repr.  This class exists to avoid importing
    model_manager directly, which would create a circular dependency.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        available_for_generation: Remaining tokens available for generation.
        recommended_max_gen: Recommended max tokens to generate.
        total_budget: Total context window size.
        remaining: Remaining tokens after generation.
        efficiency_ratio: Ratio of used tokens to budget.
        is_feasible: Whether generation is feasible within budget.
        warnings: List of warning messages.
        recommended_timeout: Recommended timeout in seconds.
        max_prompt_tokens: Maximum allowed prompt tokens.
    """
    prompt_tokens: int
    available_for_generation: int
    recommended_max_gen: int
    total_budget: int
    remaining: int
    efficiency_ratio: float
    is_feasible: bool
    warnings: List[str] = field(default_factory=list)
    recommended_timeout: float = 120.0
    max_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError(f"prompt_tokens must be >= 0, got {self.prompt_tokens}")
        if self.available_for_generation < 0:
            raise ValueError(f"available_for_generation must be >= 0, got {self.available_for_generation}")
        if self.recommended_max_gen < 0:
            raise ValueError(f"recommended_max_gen must be >= 0, got {self.recommended_max_gen}")
        if self.total_budget < 0:
            raise ValueError(f"total_budget must be >= 0, got {self.total_budget}")
        if self.remaining < 0:
            raise ValueError(f"remaining must be >= 0, got {self.remaining}")
        if self.efficiency_ratio < 0:
            raise ValueError(f"efficiency_ratio must be >= 0, got {self.efficiency_ratio}")
        if self.recommended_timeout <= 0:
            raise ValueError(f"recommended_timeout must be > 0, got {self.recommended_timeout}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "available_for_generation": self.available_for_generation,
            "recommended_max_gen": self.recommended_max_gen,
            "total_budget": self.total_budget,
            "remaining": self.remaining,
            "efficiency_ratio": self.efficiency_ratio,
            "is_feasible": self.is_feasible,
            "warnings": self.warnings.copy(),
            "recommended_timeout": self.recommended_timeout,
            "max_prompt_tokens": self.max_prompt_tokens,
        }


# -----------------------------------------------------------------------------
# Generation Result
# -----------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Returned by every generate_* method.

    Compatible with ai_agent.py metadata dicts.

    Attributes:
        text: The generated text content.
        model: Model identifier that produced the response.
        finish_reason: Reason generation stopped.
        tokens_generated: Number of tokens generated (excluding prompt).
        prompt_tokens: Number of tokens in the prompt.
        total_tokens: Total tokens processed (prompt + generated).
        ttft_ms: Time to first token in milliseconds.
        total_time_s: Total generation time in seconds.
        tps: Tokens per second (inference speed).
        queue_position: Position in queue (0 = immediate execution).
        context_cache_hit: Whether context cache was used.
        request_id: Unique request identifier from server.
        server_url: Server endpoint that served the request.
        error: Error message if finish_reason is "error".
    """
    text: str
    model: str = ""
    finish_reason: str = "stop"
    tokens_generated: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    tps: float = 0.0
    queue_position: int = 0
    context_cache_hit: bool = False
    request_id: str = ""
    server_url: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tokens_generated < 0:
            raise ValueError(f"tokens_generated must be >= 0, got {self.tokens_generated}")
        if self.prompt_tokens < 0:
            raise ValueError(f"prompt_tokens must be >= 0, got {self.prompt_tokens}")
        if self.ttft_ms < 0:
            raise ValueError(f"ttft_ms must be >= 0, got {self.ttft_ms}")
        if self.total_time_s < 0:
            raise ValueError(f"total_time_s must be >= 0, got {self.total_time_s}")
        if self.tps < 0:
            raise ValueError(f"tps must be >= 0, got {self.tps}")
        if self.queue_position < 0:
            raise ValueError(f"queue_position must be >= 0, got {self.queue_position}")

    @property
    def success(self) -> bool:
        """Return True if generation completed without error or timeout."""
        return self.finish_reason not in ("error", "timeout", "circuit_open", "overloaded")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns ai_agent.py‑compatible execution_metadata dict."""
        return {
            "original_strategy": "local_llm_cloud",
            "strategy_degraded": False,
            "compression_applied": False,
            "attempt_number": 1,
            "max_attempts": 3,
            "ttft_ms": self.ttft_ms,
            "tps": self.tps,
            "tokens_generated": self.tokens_generated,
            "queue_position": self.queue_position,
            "context_cache_hit": self.context_cache_hit,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
        }

    def to_token_metrics(self, context_window: int = 4096) -> TokenMetricsCompat:
        """Produce a TokenMetricsCompat instance with budget calculations."""
        available = max(0, context_window - self.prompt_tokens)
        return TokenMetricsCompat(
            prompt_tokens=self.prompt_tokens,
            available_for_generation=available,
            recommended_max_gen=min(available, 4096),
            total_budget=context_window,
            remaining=max(0, available - self.tokens_generated),
            efficiency_ratio=(self.tokens_generated / max(context_window, 1)),
            is_feasible=self.success,
            warnings=([] if self.success else [f"Generation failed: {self.finish_reason}"]),
            recommended_timeout=max(self.total_time_s * 2, 60.0),
            max_prompt_tokens=context_window // 2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "ttft_ms": self.ttft_ms,
            "total_time_s": self.total_time_s,
            "tps": self.tps,
            "queue_position": self.queue_position,
            "context_cache_hit": self.context_cache_hit,
            "request_id": self.request_id,
            "server_url": self.server_url,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        return cls(
            text=data.get("text", ""),
            model=data.get("model", ""),
            finish_reason=data.get("finish_reason", "stop"),
            tokens_generated=data.get("tokens_generated", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            ttft_ms=data.get("ttft_ms", 0.0),
            total_time_s=data.get("total_time_s", 0.0),
            tps=data.get("tps", 0.0),
            queue_position=data.get("queue_position", 0),
            context_cache_hit=data.get("context_cache_hit", False),
            request_id=data.get("request_id", ""),
            server_url=data.get("server_url", ""),
            error=data.get("error"),
        )

    @classmethod
    def from_chat_response(cls, response: Dict[str, Any], server_url: str = "") -> "GenerationResult":
        """Construct from the server's /v1/chat/completions response."""
        choice = (response.get("choices", [{}])[0]) if "choices" in response else {}
        message = choice.get("message", {})
        usage = response.get("usage", {})
        return cls(
            text=message.get("content", ""),
            model=response.get("model", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            tokens_generated=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            ttft_ms=response.get("ttft_ms", 0.0),
            total_time_s=response.get("total_time_s", 0.0),
            tps=response.get("tps", 0.0),
            queue_position=response.get("queue_position", 0),
            context_cache_hit=response.get("context_cache_hit", False),
            request_id=response.get("id", ""),
            server_url=server_url,
            error=response.get("error"),
        )


# -----------------------------------------------------------------------------
# Server Status
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerStatus:
    """
    Snapshot of the server's current state.

    Combines information from /health, /ready, /v1/metrics, and
    /v1/queue/status.  Not all fields will be populated on every call.

    Attributes:
        online: Whether the server is reachable.
        model_loaded: Whether a model is currently loaded.
        model_id: Identifier of the loaded model (if any).
        queue_depth: Number of requests waiting in queue.
        gate_waiters: Number of requests waiting for the gate.
        estimated_wait_s: Estimated wait time in seconds.
        avg_tps: Average tokens per second.
        avg_ttft_ms: Average time to first token in milliseconds.
        vram_pct: VRAM utilisation percentage.
        ram_pct: RAM utilisation percentage.
        checked_at: Monotonic timestamp when this status was captured.
    """
    online: bool
    model_loaded: bool
    model_id: str = ""
    queue_depth: int = 0
    gate_waiters: int = 0
    estimated_wait_s: float = 0.0
    avg_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    vram_pct: float = 0.0
    ram_pct: float = 0.0
    checked_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.queue_depth < 0:
            raise ValueError(f"queue_depth must be >= 0, got {self.queue_depth}")
        if self.gate_waiters < 0:
            raise ValueError(f"gate_waiters must be >= 0, got {self.gate_waiters}")
        if self.estimated_wait_s < 0:
            raise ValueError(f"estimated_wait_s must be >= 0, got {self.estimated_wait_s}")
        if self.avg_tps < 0:
            raise ValueError(f"avg_tps must be >= 0, got {self.avg_tps}")
        if self.avg_ttft_ms < 0:
            raise ValueError(f"avg_ttft_ms must be >= 0, got {self.avg_ttft_ms}")
        if not (0 <= self.vram_pct <= 100):
            raise ValueError(f"vram_pct must be 0-100, got {self.vram_pct}")
        if not (0 <= self.ram_pct <= 100):
            raise ValueError(f"ram_pct must be 0-100, got {self.ram_pct}")

    @property
    def is_idle(self) -> bool:
        """Return True if queue is empty and server is not busy."""
        return self.queue_depth == 0 and self.gate_waiters == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "online": self.online,
            "model_loaded": self.model_loaded,
            "model_id": self.model_id,
            "queue_depth": self.queue_depth,
            "gate_waiters": self.gate_waiters,
            "estimated_wait_s": self.estimated_wait_s,
            "avg_tps": self.avg_tps,
            "avg_ttft_ms": self.avg_ttft_ms,
            "vram_pct": self.vram_pct,
            "ram_pct": self.ram_pct,
            "checked_at": self.checked_at,
        }


# -----------------------------------------------------------------------------
# Model Information (from /v1/models) – now with status
# -----------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """
    Information about a loaded model as returned by the server.

    Mirrors the 'data' element of /v1/models response.
    """
    id: str
    model_type: str = "gguf"           # gguf or bitnet
    path: str = ""
    quantization: str = ""
    backend: str = ""
    gpu_layers: int = 0
    cpu_layers: int = 0
    vram_mb: int = 0
    requests: int = 0
    load_time_s: float = 0.0
    age_s: float = 0.0
    strategy: str = ""
    nvme_advised: bool = False
    cpu_threads: int = 0
    profile: str = ""
    status: str = "ready"              # NEW: ready, loading, error

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInfo":
        return cls(
            id=d.get("id", ""),
            model_type=d.get("model_type", "gguf"),
            path=d.get("path", ""),
            quantization=d.get("quantization", ""),
            backend=d.get("backend", ""),
            gpu_layers=d.get("gpu_layers", 0),
            cpu_layers=d.get("cpu_layers", 0),
            vram_mb=d.get("vram_mb", 0),
            requests=d.get("requests", 0),
            load_time_s=d.get("load_time_s", 0.0),
            age_s=d.get("age_s", 0.0),
            strategy=d.get("strategy", ""),
            nvme_advised=d.get("nvme_advised", False),
            cpu_threads=d.get("cpu_threads", 0),
            profile=d.get("profile", ""),
            status=d.get("status", "ready"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_type": self.model_type,
            "path": self.path,
            "quantization": self.quantization,
            "backend": self.backend,
            "gpu_layers": self.gpu_layers,
            "cpu_layers": self.cpu_layers,
            "vram_mb": self.vram_mb,
            "requests": self.requests,
            "load_time_s": self.load_time_s,
            "age_s": self.age_s,
            "strategy": self.strategy,
            "nvme_advised": self.nvme_advised,
            "cpu_threads": self.cpu_threads,
            "profile": self.profile,
            "status": self.status,
        }


@dataclass
class ModelList:
    """Wrapper for /v1/models response."""
    data: List[ModelInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelList":
        return cls(
            data=[ModelInfo.from_dict(m) for m in d.get("data", [])]
        )


# -----------------------------------------------------------------------------
# Client Configuration (centralised)
# -----------------------------------------------------------------------------

@dataclass
class ClientConfig:
    """
    Configuration for LLMCloudClient / AsyncLLMCloudClient / adapters.

    Mirrors ModelConfig from ai_agent.py so the same YAML block can construct
    either.  All values can be overridden via environment variables or
    keyword arguments.

    Attributes:
        base_url: Server endpoint (e.g., http://localhost:8080).
        api_key: API key for authentication (optional, set to empty string if none).
        master_key: Master key for admin endpoints (optional).
        model: Default model identifier.
        model_type: Default model backend (gguf, bitnet, auto).
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate (0 = unlimited).
        timeout: Total request timeout in seconds.
        max_retries: Number of retry attempts on transient errors.
        retry_delay: Initial retry delay in seconds.
        retry_backoff: Exponential backoff multiplier.
        retry_delay_max_s: Cap on retry delay (seconds).
        stream_chunk_ms: Word‑buffer flush interval in milliseconds.
        pool_size: HTTP connection pool size.
        pool_timeout: Pool acquisition timeout in seconds.
        connect_timeout_s: TCP connect timeout (seconds).
        read_timeout_s: Per‑chunk read timeout (seconds).
        health_check_on_init: Whether to check server health on initialisation.
        health_check_retries: Number of health‑check retries.
        wait_for_model: Whether to wait for a model to be loaded on init.
        wait_timeout: Maximum wait time for model load in seconds.
        tps_estimate_fallback: Fallback TPS when server metrics are unavailable.
        circuit_breaker_enabled: Whether the circuit breaker is active.
        circuit_breaker_threshold: Consecutive failures to open circuit.
        circuit_breaker_recovery: Seconds before attempting recovery.
        stream_reconnects: Times a streaming request may wait out a server
        restart and be re-sent, when nothing of the answer had arrived yet.
        server_restart_wait_s: Max seconds to wait for the server to come back
        (per reconnect) before giving up with a ConnectionError.
    """
    base_url: str = "http://localhost:8080"
    api_key: str = ""                   # empty = no auth
    master_key: str = ""
    model: str = "qwen3-8b"
    model_type: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 0
    timeout: float = 600.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_delay_max_s: float = 30.0
    stream_reconnects: int = 2
    server_restart_wait_s: float = 180.0
    stream_chunk_ms: int = 80
    pool_size: int = 50
    pool_timeout: float = 10.0
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 300.0
    health_check_on_init: bool = True
    health_check_retries: int = 3
    wait_for_model: bool = True
    wait_timeout: float = 300.0
    tps_estimate_fallback: float = 7.0
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 60.0

    # Internal cache for resolved API key (env overrides)
    _resolved_api_key: ClassVar[Optional[str]] = None

    def __post_init__(self) -> None:
        """Validate and normalise all fields."""
        if self.model and not re.match(r"^[a-zA-Z0-9_-]+$", self.model):
            raise ValueError(f"Invalid model name: {self.model} (must match [a-zA-Z0-9_-]+)")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {self.base_url}")
        self.base_url = self.base_url.rstrip("/")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be 0-2, got {self.temperature}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_delay <= 0:
            raise ValueError(f"retry_delay must be > 0, got {self.retry_delay}")
        if self.retry_backoff <= 1:
            raise ValueError(f"retry_backoff must be > 1, got {self.retry_backoff}")
        if self.retry_delay_max_s <= self.retry_delay:
            self.retry_delay_max_s = self.retry_delay * 30.0
        if self.stream_reconnects < 0:
            raise ValueError(f"stream_reconnects must be >= 0, got {self.stream_reconnects}")
        if self.server_restart_wait_s < 0:
            raise ValueError(f"server_restart_wait_s must be >= 0, got {self.server_restart_wait_s}")
        if self.stream_chunk_ms <= 0:
            raise ValueError(f"stream_chunk_ms must be > 0, got {self.stream_chunk_ms}")
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {self.pool_size}")
        if self.pool_timeout <= 0:
            raise ValueError(f"pool_timeout must be > 0, got {self.pool_timeout}")
        if self.connect_timeout_s <= 0:
            raise ValueError(f"connect_timeout_s must be > 0, got {self.connect_timeout_s}")
        if self.read_timeout_s <= 0:
            raise ValueError(f"read_timeout_s must be > 0, got {self.read_timeout_s}")
        if self.wait_timeout <= 0:
            raise ValueError(f"wait_timeout must be > 0, got {self.wait_timeout}")
        if self.tps_estimate_fallback <= 0:
            raise ValueError(f"tps_estimate_fallback must be > 0, got {self.tps_estimate_fallback}")
        if self.circuit_breaker_threshold < 1:
            raise ValueError(f"circuit_breaker_threshold must be >= 1, got {self.circuit_breaker_threshold}")
        if self.circuit_breaker_recovery <= 0:
            raise ValueError(f"circuit_breaker_recovery must be > 0, got {self.circuit_breaker_recovery}")
        if self.health_check_retries < 0:
            raise ValueError(f"health_check_retries must be >= 0, got {self.health_check_retries}")

    @property
    def is_authenticated(self) -> bool:
        """Return True if an API key or master key is provided."""
        return bool(self.api_key) or bool(self.master_key)

    def to_headers(self, admin: bool = False) -> Dict[str, str]:
        """Build HTTP headers for a request, including API key if present."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if admin and self.master_key:
            headers["X-Master-Key"] = self.master_key
        return headers

    def with_overrides(self, **kwargs) -> "ClientConfig":
        """Return a new ClientConfig with the given overrides applied."""
        # Build a dict of current values, then update with kwargs.
        current = self.to_dict()
        current.update(kwargs)
        # Remove any keys that are not valid dataclass fields.
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in current.items() if k in valid}
        return ClientConfig(**filtered)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientConfig":
        """Create a ClientConfig from a dictionary (e.g., loaded from YAML)."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration back to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_env(cls, prefix: str = "LLM_", overrides: Optional[Dict[str, Any]] = None) -> "ClientConfig":
        """Build a ClientConfig from environment variables, optionally merged with overrides."""
        cfg = cls()
        cfg.update_from_env(prefix)
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        return cfg

    def update_from_env(self, prefix: str = "LLM_") -> None:
        """
        Update configuration from environment variables.
        Maps LLM_BASE_URL → base_url, LLM_API_KEY → api_key, etc.
        """
        env_map: Dict[str, tuple] = {
            f"{prefix}BASE_URL": ("base_url", str),
            f"{prefix}API_KEY": ("api_key", str),
            f"{prefix}MASTER_KEY": ("master_key", str),
            f"{prefix}MODEL": ("model", str),
            f"{prefix}MODEL_TYPE": ("model_type", str),
            f"{prefix}TEMPERATURE": ("temperature", float),
            f"{prefix}MAX_TOKENS": ("max_tokens", int),
            f"{prefix}TIMEOUT": ("timeout", float),
            f"{prefix}MAX_RETRIES": ("max_retries", int),
            f"{prefix}RETRY_DELAY": ("retry_delay", float),
            f"{prefix}RETRY_BACKOFF": ("retry_backoff", float),
            f"{prefix}RETRY_DELAY_MAX_S": ("retry_delay_max_s", float),
            f"{prefix}STREAM_RECONNECTS": ("stream_reconnects", int),
            f"{prefix}SERVER_RESTART_WAIT_S": ("server_restart_wait_s", float),
            f"{prefix}STREAM_CHUNK_MS": ("stream_chunk_ms", int),
            f"{prefix}POOL_SIZE": ("pool_size", int),
            f"{prefix}POOL_TIMEOUT": ("pool_timeout", float),
            f"{prefix}CONNECT_TIMEOUT_S": ("connect_timeout_s", float),
            f"{prefix}READ_TIMEOUT_S": ("read_timeout_s", float),
            f"{prefix}HEALTH_CHECK_ON_INIT": ("health_check_on_init", bool),
            f"{prefix}HEALTH_CHECK_RETRIES": ("health_check_retries", int),
            f"{prefix}WAIT_FOR_MODEL": ("wait_for_model", bool),
            f"{prefix}WAIT_TIMEOUT": ("wait_timeout", float),
            f"{prefix}TPS_ESTIMATE_FALLBACK": ("tps_estimate_fallback", float),
            f"{prefix}CIRCUIT_BREAKER_ENABLED": ("circuit_breaker_enabled", bool),
            f"{prefix}CIRCUIT_BREAKER_THRESHOLD": ("circuit_breaker_threshold", int),
            f"{prefix}CIRCUIT_BREAKER_RECOVERY": ("circuit_breaker_recovery", float),
        }
        for env_var, (attr, conv) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    if conv == bool:
                        converted = value.lower() in ("true", "1", "yes", "on")
                    else:
                        converted = conv(value)
                    setattr(self, attr, converted)
                except ValueError:
                    pass  # ignore invalid conversions
        self.__post_init__()

    def __repr__(self) -> str:
        # Mask sensitive fields
        safe = self.to_dict()
        if safe.get("api_key"):
            safe["api_key"] = "***"
        if safe.get("master_key"):
            safe["master_key"] = "***"
        return f"ClientConfig({safe})"


# -----------------------------------------------------------------------------
# Request / Response Models (for typed endpoints)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatMessage:
    """Single message in a chat conversation."""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class CompletionRequest:
    """
    Parameters for a chat completion request.
    Matches the server's /v1/chat/completions body.
    """
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    stop: Optional[List[str]] = None
    stream: bool = False
    no_think: Optional[bool] = None
    stream_format: str = "json"
    thinking: bool = False
    thinking_budget_tokens: int = 0
    timeout_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop or [],
            "stream": self.stream,
        }
        if self.no_think is not None:
            d["no_think"] = self.no_think
        if self.stream_format:
            d["stream_format"] = self.stream_format
        if self.thinking:
            d["thinking"] = True
        if self.thinking_budget_tokens:
            d["thinking_budget_tokens"] = self.thinking_budget_tokens
        if self.timeout_s is not None:
            d["timeout_s"] = self.timeout_s
        return d


@dataclass(frozen=True)
class CompletionResponse:
    """
    Parsed chat completion response from the server.

    Contains convenience accessors for the first choice and usage.
    """
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    ttft_ms: float = 0.0
    tps: float = 0.0
    total_time_s: float = 0.0
    context_cache_hit: bool = False

    @property
    def text(self) -> str:
        """Return the content of the first choice."""
        if self.choices:
            return self.choices[0].get("message", {}).get("content", "")
        return ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionResponse":
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=data.get("choices", []),
            usage=data.get("usage", {}),
            ttft_ms=data.get("ttft_ms", 0.0),
            tps=data.get("tps", 0.0),
            total_time_s=data.get("total_time_s", 0.0),
            context_cache_hit=data.get("context_cache_hit", False),
        )


# -----------------------------------------------------------------------------
# Utility: Configuration Merger
# -----------------------------------------------------------------------------

class ConfigMerger:
    """
    Utility for merging multiple config sources (files, env, overrides).
    Used by the CLI and adapter layers to produce a final ClientConfig.
    """
    @staticmethod
    def merge(
        file_cfg: Optional[Dict[str, Any]],
        env_prefix: str = "LLM_",
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> ClientConfig:
        """
        Merge a file‑loaded config with environment variables and CLI overrides.

        Priority: CLI overrides > environment > file > defaults.
        """
        cfg = ClientConfig()
        if file_cfg:
            cfg = cfg.with_overrides(**{k: v for k, v in file_cfg.items() if hasattr(cfg, k)})
        cfg.update_from_env(env_prefix)
        if cli_overrides:
            cfg = cfg.with_overrides(**cli_overrides)
        return cfg
