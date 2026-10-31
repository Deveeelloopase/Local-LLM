"""
engine/gguf/__init__.py — Production‑Ready GGUF Inference Module
=================================================================
Exports the public API for GGUF model inference, loading, and speculative decoding.

Components:
- InferenceEngine: Async inference with concurrency control, circuit breaker, metrics.
- ModelLoader: Lifecycle management, security, hardware‑adaptive loading.
- SpeculativeEngine: Draft‑target speculative decoding for 2–4× throughput.
- Dataclasses: InferenceRequest, InferenceResponse, ModelConfig, LoadedModel.

All configuration is read from the global config.yaml (passed via constructors).
"""

__version__ = "1.0.0"

# -----------------------------------------------------------------------------
# Core inference components
# -----------------------------------------------------------------------------
from local_llm_core.engine.inference.inference_engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
    AsyncTokenBuffer,
    BoundedThreadPoolExecutor,
    CircuitBreaker,
    RateLimiter,
    TokenBucket,
    AutoTuner,
    NGramCache,
    RequestQueue,
    _strip_think,
    _PROM,
)

# -----------------------------------------------------------------------------
# Model loading components
# -----------------------------------------------------------------------------
from .model_loader import (
    ModelLoader,
    ModelConfig,
    LoadedModel,
    _LoaderBreaker,
    _BPW,
    _select_quantization,
)

# -----------------------------------------------------------------------------
# Speculative decoding
# -----------------------------------------------------------------------------
from .speculative_engine import (
    SpeculativeEngine,
    SpecStats,
    _softmax,
    _top_p_filter,
    _sample,
    _acceptance_prob,
    _corrected_sample,
)

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    # InferenceEngine
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "AsyncTokenBuffer",
    "BoundedThreadPoolExecutor",
    "CircuitBreaker",
    "RateLimiter",
    "TokenBucket",
    "AutoTuner",
    "NGramCache",
    "RequestQueue",
    "_strip_think",
    "_PROM",
    # ModelLoader
    "ModelLoader",
    "ModelConfig",
    "LoadedModel",
    "_LoaderBreaker",
    "_BPW",
    "_select_quantization",
    # SpeculativeEngine
    "SpeculativeEngine",
    "SpecStats",
    "_softmax",
    "_top_p_filter",
    "_sample",
    "_acceptance_prob",
    "_corrected_sample",
    # Version
    "__version__",
]
