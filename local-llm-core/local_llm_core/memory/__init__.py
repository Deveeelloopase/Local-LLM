"""
memory/__init__.py — Production‑Ready Memory Management
=======================================================
Three‑tier memory hierarchy: VRAM → RAM → SSD swap.

Public API:
- VRAMManager, AllocationStats, TensorBlock
- RAMOptimizer
- SwapHandler
- HybridMemoryManager, MemoryTier, HybridBlock
- HybridCache, LayerLocation
- SimpleConnectionPool

All configuration is read from config.yaml by individual modules.
No hard‑coded values; everything is config‑driven.
"""

__version__ = "1.0.0"

import logging
import os

log = logging.getLogger(__name__)

# Strict mode: if set, missing memory managers raise ImportError.
STRICT_MODE = os.environ.get("LLM_MEMORY_STRICT", "0") == "1"


def _handle_import_error(module_name: str, exc: Exception):
    if STRICT_MODE:
        raise ImportError(f"Critical memory module '{module_name}' failed to import") from exc
    log.warning("%s import failed: %s", module_name, exc)


# -----------------------------------------------------------------------------
# VRAM Manager
# -----------------------------------------------------------------------------
try:
    from .vram_manager import VRAMManager, AllocationStats, TensorBlock
except ImportError as e:
    _handle_import_error("vram_manager", e)
    VRAMManager = None
    AllocationStats = None
    TensorBlock = None

# -----------------------------------------------------------------------------
# RAM Optimizer
# -----------------------------------------------------------------------------
try:
    from .ram_optimizer import RAMOptimizer
except ImportError as e:
    _handle_import_error("ram_optimizer", e)
    RAMOptimizer = None

# -----------------------------------------------------------------------------
# Swap Handler (NVMe SSD)
# -----------------------------------------------------------------------------
try:
    from .swap_handler import SwapHandler
except ImportError as e:
    _handle_import_error("swap_handler", e)
    SwapHandler = None

# -----------------------------------------------------------------------------
# Hybrid Memory Manager (Three‑Tier Orchestrator)
# -----------------------------------------------------------------------------
try:
    from .hybrid_manager import HybridMemoryManager, MemoryTier, HybridBlock
except ImportError as e:
    _handle_import_error("hybrid_manager", e)
    HybridMemoryManager = None
    MemoryTier = None
    HybridBlock = None

# -----------------------------------------------------------------------------
# Hybrid Cache (Layer‑Specific RAM+NVMe Cache)
# -----------------------------------------------------------------------------
try:
    from .hybrid_cache import HybridCache, LayerLocation
except ImportError as e:
    _handle_import_error("hybrid_cache", e)
    HybridCache = None
    LayerLocation = None

# -----------------------------------------------------------------------------
# Connection Pool (SQLite)
# -----------------------------------------------------------------------------
try:
    from .connection_pool import SimpleConnectionPool
except ImportError as e:
    _handle_import_error("connection_pool", e)
    SimpleConnectionPool = None


def validate_imports() -> Dict[str, bool]:
    """Return a dict indicating which memory managers are available."""
    return {
        "VRAMManager": VRAMManager is not None,
        "RAMOptimizer": RAMOptimizer is not None,
        "SwapHandler": SwapHandler is not None,
        "HybridMemoryManager": HybridMemoryManager is not None,
        "HybridCache": HybridCache is not None,
        "SimpleConnectionPool": SimpleConnectionPool is not None,
    }


# -----------------------------------------------------------------------------
# Public Exports
# -----------------------------------------------------------------------------
__all__ = [
    "VRAMManager",
    "AllocationStats",
    "TensorBlock",
    "RAMOptimizer",
    "SwapHandler",
    "HybridMemoryManager",
    "MemoryTier",
    "HybridBlock",
    "HybridCache",
    "LayerLocation",
    "SimpleConnectionPool",
    "__version__",
    "validate_imports",
]
