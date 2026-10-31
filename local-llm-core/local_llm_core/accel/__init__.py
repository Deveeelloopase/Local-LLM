"""accel — version-agnostic capability probing and memory accounting."""
__version__ = "1.0.0"
SCHEMA_VERSION = "1.0.0"

from .capabilities import LlamaCppCaps, probe_llama_cpp
from .memory_ledger import MemoryLedger, LedgerBudget
from .model_advisor import ModelAdvisor, ModelAdvice

__all__ = [
    "LlamaCppCaps", "probe_llama_cpp",
    "MemoryLedger", "LedgerBudget",
    "ModelAdvisor", "ModelAdvice",
    "__version__", "SCHEMA_VERSION",
]