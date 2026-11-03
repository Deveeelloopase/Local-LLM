"""
Compatibility layer for managing library version conflicts.
Ensures code works across multiple versions of dependencies.
"""

from local_llm_core.compatibility.transformers_compat import (
    TransformersCompatibilityManager,
    TRANSFORMERS_VERSION,
)

__all__ = [
    "TransformersCompatibilityManager",
    "TRANSFORMERS_VERSION",
]
