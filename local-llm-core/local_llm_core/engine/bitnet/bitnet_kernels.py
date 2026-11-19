"""
engine/bitnet/bitnet_kernels.py — Backward‑Compatibility Alias
=============================================================
Previously contained a custom kernels manager for BitNet models.
As of local‑llm‑core v3, this has been superseded by the
UniversalKernelManager (engine.kernels.manager), which provides
automatic backend selection for CUDA, HIP, Metal, Vulkan, and CPU.

This module exists solely to keep existing code working.
New code should import UniversalKernelManager directly.

Example (old):
    from local_llm_core.engine.bitnet.bitnet_kernels import BitNetKernelManager

Example (new):
    from local_llm_core.engine.kernels.manager import UniversalKernelManager
"""

from local_llm_core.engine.kernels.manager import UniversalKernelManager

# Keep the old class name for backward compatibility
BitNetKernelManager = UniversalKernelManager

__all__ = ["BitNetKernelManager"]
