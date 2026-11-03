"""
engine/kernels/metal/loader.py
==============================
Public entry-point for the Metal compute backend.

Architecture decision
---------------------
Two Metal shaders exist in this repo:

  kernels/metal_kernels/matmul_1bit.metal   — ternary / 1-bit BitNet matmul
  engine/kernels/metal/matmul.metal         — reserved for future general-purpose
                                              (fp16, bf16, int8) Metal compute kernels

This file is the **canonical import point** wired into UniversalKernelManager.
It re-exports the production-ready 1-bit loader from kernels/metal_kernels/ so that
the manager always has a single stable import path, while the engine/kernels/metal/
directory remains the natural home for future Apple-specific Metal work.

Adding a new shader (e.g. fp16 attention):
  1. Add <name>.metal next to matmul.metal
  2. Add a loader function below (follow _load_metal_kernel pattern)
  3. Export it from this module — nothing else needs to change.

Apple Silicon targeting
-----------------------
Required at runtime (macOS only):
  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders

These are commented out in requirements.txt and listed under setup.py [apple]
extra so non-Apple CI doesn't pull them in.
"""

from __future__ import annotations

# Re-export the production 1-bit ternary matmul from the real implementation.
# UniversalKernelManager imports from HERE — not from metal_kernels directly.
from local_llm_core.kernels.metal_kernels.metal_loader import (   # noqa: F401
    _load_metal_kernel,
    metal_ternary_matmul,
    _METAL_AVAIL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Placeholder: future general-purpose Metal kernels
# (fp16 attention, RoPE embedding, softmax — compiled from matmul.metal)
# ──────────────────────────────────────────────────────────────────────────────
# from ._general_loader import metal_fp16_matmul   # uncomment when ready

__all__ = [
    "_load_metal_kernel",
    "metal_ternary_matmul",
    "_METAL_AVAIL",
]