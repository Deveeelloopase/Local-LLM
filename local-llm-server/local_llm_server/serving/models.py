"""
local_llm_server/serving/models.py — Pydantic v2 Schemas v4.0.0
================================================================
OpenAI‑compatible request/response models for chat completions,
model management, health checks, and error reporting.

v4.0.0 Highlights:
  • LoadModelRequest now includes `model_type` (gguf/bitnet/auto).
  • ChatRequest and CompletionRequest gain `repetition_penalty`, `seed`,
    and improved validation (NaN/Inf, cross‑field consistency).
  • ModelInfo includes `model_type` and optional BitNet‑specific fields.
  • All validators handle edge cases and emit actionable warnings.
  • Fully backward compatible with v3.x clients.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    OVERLOADED = "overloaded"


class RoleEnum(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SeverityEnum(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(str, Enum):
    GGUF = "gguf"
    BITNET = "bitnet"
    AUTO = "auto"


# ══════════════════════════════════════════════════════════════════════════════
# Chat Completion Models (OpenAI‑compatible)
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: RoleEnum
    content: str
    name: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "role": "user",
                "content": "What is the capital of France?"
            }]
        }
    )


class ChatRequest(BaseModel):
    """OpenAI‑compatible chat completion request (extended)."""
    model: str = Field(..., description="Model ID to use for generation")
    messages: List[ChatMessage] = Field(
        ..., min_length=1, description="Chat history (user/assistant/system messages)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0,
        description="Sampling temperature (0=greedy, 1=normal, >1=more random)"
    )
    top_p: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Nucleus sampling threshold"
    )
    top_k: int = Field(
        default=40, ge=1, le=1000,
        description="Top‑K sampling (only allow top K tokens)"
    )
    max_tokens: int = Field(
        default=0, ge=0,
        description="Maximum tokens to generate. 0 = unlimited (up to model context)."
    )
    repetition_penalty: float = Field(
        default=1.0, ge=0.0,
        description="Repetition penalty. 1.0 = no penalty, >1 penalise, <1 encourage."
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences (generation stops when one is encountered)"
    )
    stream: bool = Field(
        default=False,
        description="Stream tokens as they are generated (Server‑Sent Events)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for deterministic sampling (when supported by backend)"
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Frequency penalty (reserved, not yet supported)"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Presence penalty (reserved, not yet supported)"
    )
    user: Optional[str] = Field(
        default=None, description="Unique user identifier for rate limiting"
    )
    # Extended fields (Local LLM Cloud only)
    no_think: Optional[bool] = Field(
        default=None,
        description="Force disable thinking/reasoning mode (overrides config)"
    )
    thinking: bool = Field(
        default=False,
        description="Enable thinking/reasoning mode (Qwen models)"
    )
    thinking_budget_tokens: int = Field(
        default=0, ge=0,
        description="Max tokens for thinking (0 = auto)"
    )
    timeout_s: float = Field(
        default=0.0, ge=0.0,
        description="Per‑request timeout in seconds (0 = use server default)"
    )
    stream_format: str = Field(
        default="json",
        pattern="^(json|text)$",
        description="Stream format: 'json' for SSE, 'text' for plain text"
    )

    history_mode: str = Field(
        default="auto",
        pattern="^(auto|append|replace)$",
        description=(
            "How to merge the incoming messages with stored history. "
            "'replace' means the client owns full state (web UI). "
            "'append' means incoming is a single new message (CLI). "
            "'auto' picks append for 1 non-system message, else replace."
        ),
    )

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for seq in v:
                if not seq.strip():
                    raise ValueError("stop sequences must be non-empty strings")
        return v

    @field_validator("temperature", "top_p", "repetition_penalty")
    @classmethod
    def _validate_float_constraints(cls, v: float, info) -> float:
        if v != v:  # NaN
            raise ValueError(f"{info.field_name} cannot be NaN")
        if v == float("inf") or v == float("-inf"):
            raise ValueError(f"{info.field_name} cannot be infinite")
        return v

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("seed must be non‑negative")
        return v

    @model_validator(mode="after")
    def _validate_parameter_consistency(self) -> "ChatRequest":
        # Greedy decoding + nucleus sampling is contradictory
        if self.temperature == 0.0 and self.top_p != 1.0:
            logger.warning(
                "temperature=0 (greedy) but top_p=%.2f. top_p is effectively ignored.",
                self.top_p,
            )
        # Unlimited max_tokens without streaming can block for a very long time
        if self.max_tokens == 0 and not self.stream:
            logger.warning(
                "max_tokens=0 (unlimited) with stream=False. "
                "Ensure a reasonable timeout is set."
            )
        # Thinking mode without a budget will use server default, but warn
        if self.thinking and self.thinking_budget_tokens == 0:
            logger.warning(
                "thinking=True but thinking_budget_tokens=0. "
                "Server will use a default budget (2048)."
            )
        # Repetition penalty should be 1.0 if not intended
        if self.repetition_penalty != 1.0:
            logger.warning(
                "repetition_penalty=%.2f active. Ensure you want to modify token probabilities.",
                self.repetition_penalty,
            )
        # Seed with temperature=0 is pointless but not harmful
        if self.seed is not None and self.temperature == 0.0:
            logger.warning("seed is set but temperature=0 (greedy) – seed has no effect.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "qwen3-8b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": False
            }]
        }
    )


class ChatCompletionChoice(BaseModel):
    """One completion choice from a chat request."""
    index: int
    message: ChatMessage
    finish_reason: FinishReason
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens generated")
    total_tokens: int = Field(description="Total tokens (prompt + completion)")


class ChatResponse(BaseModel):
    id: str = Field(description="Unique request ID")
    object: str = Field(default="chat.completion")
    created: int = Field(description="Unix timestamp when response was created")
    model: str = Field(description="Model used for generation")
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None

    # Local LLM Cloud extensions
    ttft_ms: float = Field(default=0.0, description="Time to first token (ms)")
    total_time_s: float = Field(default=0.0, description="Total generation time (seconds)")
    tps: float = Field(default=0.0, description="Actual tokens/second achieved")
    context_cache_hit: bool = Field(default=False,
                                    description="Whether this request hit the KV‑prefix cache")
    queue_position: int = Field(default=0,
                                description="Position in the inference queue when submitted")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "id": "req-12345",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "qwen3-8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "4"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
                "ttft_ms": 150.5,
                "total_time_s": 2.3,
                "tps": 1.2
            }]
        }
    )


class ChatStreamChunk(BaseModel):
    id: str
    object: str = Field(default="chat.completion.chunk")
    created: int
    model: str
    choices: List[Dict[str, Any]] = Field(
        description='[{"index": 0, "delta": {"content": "token"}, "finish_reason": null}]'
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "id": "req-12345",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "qwen3-8b",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]
            }]
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Text Completion Models (legacy, non‑chat)
# ══════════════════════════════════════════════════════════════════════════════

class CompletionRequest(BaseModel):
    model: str
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    max_tokens: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = Field(default=None, ge=0)
    user: Optional[str] = None

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for seq in v:
                if not seq.strip():
                    raise ValueError("stop sequences must be non‑empty strings")
        return v

    @field_validator("temperature", "top_p", "repetition_penalty")
    @classmethod
    def _validate_float_constraints(cls, v: float, info) -> float:
        if v != v:
            raise ValueError(f"{info.field_name} cannot be NaN")
        if v == float("inf") or v == float("-inf"):
            raise ValueError(f"{info.field_name} cannot be infinite")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "qwen3-8b",
                "prompt": "The capital of France is",
                "max_tokens": 20,
                "stream": False
            }]
        }
    )


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: FinishReason
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = Field(default="text_completion")
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage

    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    tps: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Model Management Models
# ══════════════════════════════════════════════════════════════════════════════

class ModelInfo(BaseModel):
    """Information about a loaded model (GGUF or BitNet)."""
    id: str = Field(description="Model identifier")
    object: str = Field(default="model")
    owned_by: str = Field(default="local-llm-cloud")

    # Common
    path: str = Field(description="File path or directory to model weights")
    quantization: str = Field(description="Quantization type (Q4_K_M, 1bit, etc.)")
    model_type: ModelType = Field(description="Backend type: gguf or bitnet")
    backend: str = Field(description="Runtime backend (llama_cpp, transformers, etc.)")
    requests: int = Field(description="Total requests served")
    load_time_s: float = Field(description="Time to load (seconds)")
    age_s: float = Field(description="Seconds since model was loaded")

    # GGUF‑specific (optional)
    gpu_layers: Optional[int] = Field(default=None, description="Number of layers on GPU")
    cpu_layers: Optional[int] = Field(default=None, description="Number of layers on CPU")
    vram_mb: Optional[int] = Field(default=None, description="VRAM used (MB)")
    strategy: Optional[str] = Field(default=None, description="Loading strategy (cpu_only, hybrid, full_gpu)")
    nvme_advised: Optional[bool] = Field(default=None, description="NVMe prefetch applied")
    cpu_threads: Optional[int] = Field(default=None, description="CPU threads allocated")
    profile: Optional[str] = Field(default=None, description="Hardware profile used")

    # BitNet‑specific (optional)
    bits: Optional[int] = Field(default=None, description="Bit width for BitNet models (1/2/4/8)")
    num_parameters: Optional[int] = Field(default=None, description="Number of trainable parameters")
    context_length: Optional[int] = Field(default=None, description="Maximum context length")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "id": "qwen3-8b",
                "object": "model",
                "owned_by": "local-llm-cloud",
                "path": "/models/qwen3-8b-q4_k_m.gguf",
                "quantization": "Q4_K_M",
                "model_type": "gguf",
                "backend": "llama_cpp",
                "requests": 42,
                "load_time_s": 3.2,
                "age_s": 7234.5,
                "gpu_layers": 11,
                "cpu_layers": 25,
                "vram_mb": 1350,
                "strategy": "hybrid",
                "nvme_advised": True,
                "cpu_threads": 4,
                "profile": "apu_4gb"
            }, {
                "id": "bitnet-b1.58-2B",
                "object": "model",
                "owned_by": "local-llm-cloud",
                "path": "/models/bitnet-b1.58-2B-4T",
                "quantization": "1bit",
                "model_type": "bitnet",
                "backend": "transformers",
                "requests": 10,
                "load_time_s": 8.5,
                "age_s": 1200.0,
                "bits": 1,
                "num_parameters": 2_000_000_000,
                "context_length": 4096
            }]
        }
    )


class ModelList(BaseModel):
    object: str = Field(default="list")
    data: List[ModelInfo]


class LoadModelRequest(BaseModel):
    """Request to load a GGUF or BitNet model."""
    model_id: str = Field(..., description="Model identifier")
    model_path: str = Field(..., description="Path to GGUF file or BitNet directory")
    model_type: ModelType = Field(
        default=ModelType.AUTO,
        description="'gguf' for single‑file, 'bitnet' for directory, 'auto' to detect"
    )
    num_gpu_layers: int = Field(default=-1, ge=-1,
                                 description="GPU layers (-1=auto, 0=cpu_only, N=specific)")
    context_length: int = Field(default=0, ge=0, description="Context window (0=auto)")
    quantization: str = Field(default="auto", description="Quantization (auto/Q4_K_M/etc.)")
    n_threads: int = Field(default=0, ge=0, description="Number of CPU threads (0=auto)")
    priority: int = Field(default=0, ge=0, le=10, description="Load priority")
    autoload: bool = Field(default=False, description="Keep loaded across restarts")
    description: str = Field(default="", max_length=500, description="Human‑readable description")

    @field_validator("model_path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_path cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model_id": "qwen3-8b",
                "model_path": "/models/qwen3-8b-q4_k_m.gguf",
                "model_type": "gguf",
                "num_gpu_layers": -1,
                "context_length": 8192
            }]
        }
    )


class UnloadModelRequest(BaseModel):
    model_id: str = Field(..., description="Model to unload")


# ══════════════════════════════════════════════════════════════════════════════
# Health & Status Models
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'error'")
    uptime_s: float = Field(description="Seconds since server started")
    version: str = Field(description="Server version")
    loaded_models: int = Field(description="Number of models currently loaded")
    queue_depth: int = Field(description="Requests waiting in queue")
    circuit_breaker: str = Field(description="Circuit breaker state (closed/open/half-open)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "status": "ok",
                "uptime_s": 3600.5,
                "version": "1.0.0",
                "loaded_models": 1,
                "queue_depth": 0,
                "circuit_breaker": "closed"
            }]
        }
    )


class MetricsResponse(BaseModel):
    total_requests: int
    total_tokens: int
    total_errors: int
    oom_recoveries: int
    avg_ttft_ms: float
    avg_tps: float
    p95_ttft_ms: float
    p95_tps: float
    ngram_hit_rate: float
    batch_size: int
    queue_depth: int
    gate_waiters: int

    circuit_breaker_state: str
    circuit_breaker_error_rate: float
    circuit_breaker_recovery_secs: float

    context_cache_entries: Optional[int] = None
    context_cache_used_mb: Optional[int] = None
    context_cache_hit_rate: Optional[float] = None

    memory_vram_pct: float
    memory_ram_pct: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "total_requests": 1234,
                "total_tokens": 567890,
                "total_errors": 2,
                "oom_recoveries": 0,
                "avg_ttft_ms": 145.3,
                "avg_tps": 8.2,
                "p95_ttft_ms": 250.0,
                "p95_tps": 12.5,
                "ngram_hit_rate": 0.23,
                "batch_size": 1,
                "queue_depth": 0,
                "gate_waiters": 0,
                "circuit_breaker_state": "closed",
                "circuit_breaker_error_rate": 0.0,
                "circuit_breaker_recovery_secs": 0.0,
                "context_cache_entries": 5,
                "context_cache_used_mb": 128,
                "context_cache_hit_rate": 0.34,
                "memory_vram_pct": 65.5,
                "memory_ram_pct": 42.0
            }]
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Error Models
# ══════════════════════════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    error: str = Field(description="Error code (e.g. MODEL_NOT_FOUND)")
    message: str = Field(description="Human‑readable error message")
    severity: SeverityEnum = Field(description="Severity level")
    retryable: bool = Field(description="Whether the request can be retried")
    retry_after_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "error": {
                    "error": "MODEL_NOT_FOUND",
                    "message": "Model 'unknown-model' not loaded. Load it with POST /v1/models/load",
                    "severity": "warning",
                    "retryable": False
                },
                "request_id": "req-12345"
            }]
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Logging & Monitoring Models
# ══════════════════════════════════════════════════════════════════════════════

class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: LogLevelEnum = Field(description="log level (DEBUG/INFO/WARNING/ERROR)")
    logger: str = Field(description="logger name")
    message: str
    request_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class LogStream(BaseModel):
    entries: List[LogEntry]
    has_more: bool = Field(description="Whether more logs exist beyond this page")


# ══════════════════════════════════════════════════════════════════════════════
# Internal Models (Not exposed via API)
# ══════════════════════════════════════════════════════════════════════════════

class TokenMetricsCompat(BaseModel):
    tokens_generated: int
    prompt_tokens: int
    total_time_s: float
    ttft_ms: float
    tps: float
    finish_reason: FinishReason


class InternalRequest(BaseModel):
    request_id: str
    user_id: Optional[str] = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=0, ge=0)


# ══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "FinishReason",
    "RoleEnum",
    "SeverityEnum",
    "LogLevelEnum",
    "ModelType",
    # Chat
    "ChatMessage",
    "ChatRequest",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatResponse",
    "ChatStreamChunk",
    # Text Completion
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    # Models
    "ModelInfo",
    "ModelList",
    "LoadModelRequest",
    "UnloadModelRequest",
    # Health
    "HealthResponse",
    "MetricsResponse",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
    # Logs
    "LogEntry",
    "LogStream",
    # Internal
    "TokenMetricsCompat",
    "InternalRequest",
]
