from __future__ import annotations

"""
adaptive_awareness/__init__.py — Production‑Ready Inference Engine
======================================================
Lazy‑loading entry point for the Local LLM inference stack.

All public components are available via attribute access, which triggers
on‑demand imports to reduce startup time and avoid circular dependencies.

Configuration is loaded from config.yaml via the `utils` subpackage.
No hard‑coded values; everything is config‑driven.

Version: 1.0.0
"""

__version__ = "1.0.0"

import logging


log = logging.getLogger(__name__)

from ..adaptive_awareness.model_unified_env import (
    classify_gguf,          # detect if a GGUF is BitNet or standard
    route_model,            # pick the right backend for any model path
    auto_detect_backend,    # diagnostics / startup logging
    get_bitnet_state,       # read ~/.config/local-llm/bitnet_state.json
    validate_environment,   # check unified transformers constraint
    GGUFModelType,          # enum: BITNET | STANDARD | UNKNOWN
    InferenceBackend,       # enum: BITNET_CPP_SERVER | LLAMA_CPP_PYTHON | …
    AdaptiveModelRouter,    # the AQAMR class itself (advanced usage)
)

__all__ = [
    "classify_gguf", "route_model", "auto_detect_backend",
    "get_bitnet_state", "validate_environment",
    "GGUFModelType", "InferenceBackend", "AdaptiveModelRouter",
]
