from __future__ import annotations

"""
engine/kernels/__init__.py — Kernel subsystem public API
========================================================
Exports the UniversalKernelManager and the patcher.
All components are lazily imported.
"""

__version__ = "1.0.0"

import logging

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lazy imports
# -----------------------------------------------------------------------------
_LAZY = {
    "UniversalKernelManager": ("manager", "UniversalKernelManager"),
    "install_patch": ("patcher", "install_patch"),
    "remove_patch": ("patcher", "remove_patch"),
}

def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'engine.kernels' has no attribute {name!r}")
    mod_name, attr_name = _LAZY[name]
    import importlib
    module = importlib.import_module(f".{mod_name}", __package__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr

__all__ = list(_LAZY.keys()) + ["__version__"]
