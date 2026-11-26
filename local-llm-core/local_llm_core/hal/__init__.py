"""
hal/__init__.py — Production‑Ready Hardware Abstraction Layer
=============================================================
Lazy‑loading entry point for hardware detection, analysis, tuning, and
system power management (suspend inhibition).

Public API:
- HardwareDetector, GPUInfo, SystemInfo, GPUVendor
- HardwareProfile, detect_profile, get_profile
- GPUAnalyzer, ComputeProfile, ThermalStatus
- SystemOptimizer
- SuspendInhibitor  (new)
- _select_quantization (helper)

All configuration is read from config.yaml by individual modules.
No hard‑coded values.
"""

__version__ = "1.0.0"

import logging
import importlib
from typing import Dict

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lazy Import Machinery
# -----------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # Hardware Detector
    "HardwareDetector": ("hardware_detector", "HardwareDetector"),
    "GPUInfo": ("hardware_detector", "GPUInfo"),
    "SystemInfo": ("hardware_detector", "SystemInfo"),
    "GPUVendor": ("hardware_detector", "GPUVendor"),
    "MemoryType": ("hardware_detector", "MemoryType"),
    "_select_quantization": ("hardware_detector", "_select_quantization"),
    # Hardware Profiles
    "HardwareProfile": ("hardware_profiles", "HardwareProfile"),
    "detect_profile": ("local_llm_core.engine.hardware.hardware_profiles", "detect_profile"),
    "get_profile": ("hardware_profiles", "get_profile"),
    # GPU Analyzer
    "GPUAnalyzer": ("gpu_analyzer", "GPUAnalyzer"),
    "ComputeProfile": ("gpu_analyzer", "ComputeProfile"),
    "ThermalStatus": ("gpu_analyzer", "ThermalStatus"),
    # System Optimizer
    "SystemOptimizer": ("system_optimizer", "SystemOptimizer"),
    # Suspend Inhibitor
    "SuspendInhibitor": ("suspend_inhibitor", "SuspendInhibitor"),
}


def __getattr__(name: str):
    """Lazy import of HAL submodules with error logging."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'hal' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    except ImportError as e:
        log.exception("Failed to lazy‑import %s from %s", name, module_name)
        raise ImportError(
            f"Could not import '{name}' from '{module_name}'. "
            f"This usually means a missing system dependency (e.g., nvidia‑smi, rocm‑smi). "
            f"Original error: {e}"
        ) from e


def validate_imports() -> Dict[str, bool]:
    """Attempt to import all public symbols. Returns {name: success}."""
    results = {}
    for name in _LAZY_IMPORTS:
        try:
            __getattr__(name)
            results[name] = True
        except Exception as e:
            log.error("Validation failed for %s: %s", name, e)
            results[name] = False
    return results


# -----------------------------------------------------------------------------
# Public API Definition
# -----------------------------------------------------------------------------
__all__ = list(_LAZY_IMPORTS.keys()) + ["__version__", "validate_imports"]
