from __future__ import annotations

"""
engine/kernels/manager.py — Universal Kernel Manager
====================================================
v1.0.0 – Config injected via constructor; no filesystem access.

Lazy, optional backend imports to avoid import errors when a
hardware‑specific library is missing.  All settings come from a
`config` dict (or a `KernelsConfig` model) provided at init time.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from local_llm_core.hal import HardwareDetector
from .patcher import install_patch, remove_patch

log = logging.getLogger(__name__)

# Order to try backends when auto_select is enabled
BACKEND_ORDER = ["cuda", "hip", "metal", "vulkan"]

# Default configuration used when no config is supplied – ensures sensible defaults
_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "auto_select": True,
    "force_backend": None,
    "cpu_optimizer": {},
}


class UniversalKernelManager:
    """
    Manages loading and activation of custom compute kernels.

    Accepts a configuration dictionary (or a KernelsConfig model)
    through the `config` parameter.  The recognised keys are:

    * ``enabled``          – bool, whether to use custom kernels (default True)
    * ``auto_select``      – bool, try backends in standard order (default True)
    * ``force_backend``    – optional str, overrides auto‑detection
    * ``cpu_optimizer``    – dict, forwarded to the CPU tuning module

    If no config is given, the defaults above are used.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # ------------------------------------------------------------------
        # Normalise config – optionally validate with Pydantic
        # ------------------------------------------------------------------
        raw = config or {}
        try:
            # Prefer the central Pydantic model if available
            from local_llm_core.engine.utils.config_loader import KernelsConfig
            if isinstance(raw, KernelsConfig):
                self._cfg = raw.model_dump()
            else:
                self._cfg = KernelsConfig(**raw).model_dump()
        except Exception as exc:
            log.warning("Kernel config validation failed: %s; using defaults", exc)
            self._cfg = _DEFAULT_CONFIG.copy()
            self._cfg.update(raw)  # still allow partial overrides

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._selected_backend: Optional[str] = None
        self._backend_fn: Optional[Callable] = None
        self._patch_installed = False

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def initialize(self, gpu_infos: Optional[List[Any]] = None) -> bool:
        """
        Probe hardware and activate the best available kernel backend.

        Returns True if a backend was successfully activated, False otherwise.
        """
        if not self._cfg.get("enabled", True):
            log.info("Custom kernels disabled by configuration")
            return False

        # Obtain GPU info if not provided
        if gpu_infos is None:
            try:
                hw = HardwareDetector().detect()
                gpu_infos = hw.gpus
            except Exception as exc:
                log.warning("Hardware detection failed: %s; kernels will not be activated", exc)
                return False

        # 1. Check for a forced backend override
        forced = self._cfg.get("force_backend")
        if forced and forced in BACKEND_ORDER + ["cpu"]:
            return self._activate_backend(forced, gpu_infos)

        # 2. Auto‑select from the standard backend order
        if self._cfg.get("auto_select", True):
            for backend in BACKEND_ORDER:
                if self._activate_backend(backend, gpu_infos):
                    return True

        # 3. Apply CPU optimisations (always called, even if no GPU backend)
        self._apply_cpu_optimizer()
        return False

    # -------------------------------------------------------------------------
    # Backend activation helpers
    # -------------------------------------------------------------------------
    def _activate_backend(self, backend: str, gpu_infos: List[Any]) -> bool:
        """Load and enable a single backend if compatible hardware is present."""
        loader_map = {
            "cuda": self._load_cuda,
            "hip": self._load_hip,
            "metal": self._load_metal,
            "vulkan": self._load_vulkan,
        }
        if backend not in loader_map:
            return False
        if not self._backend_hardware_available(backend, gpu_infos):
            log.debug("Backend '%s' hardware not available", backend)
            return False

        fn = loader_map[backend]()
        if fn is not None:
            self._selected_backend = backend
            self._backend_fn = fn
            log.info("Custom kernels backend activated: %s", backend)
            self.patch_pytorch()
            return True
        return False

    # -------------------------------------------------------------------------
    # Hardware availability checks
    # -------------------------------------------------------------------------
    @staticmethod
    def _backend_hardware_available(backend: str, gpu_infos: List[Any]) -> bool:
        for gpu in gpu_infos:
            if hasattr(gpu, "backend") and gpu.backend == backend:
                return True
        if backend == "cuda" and any("NVIDIA" in str(gpu.vendor) for gpu in gpu_infos):
            return True
        if backend == "hip" and any("AMD" in str(gpu.vendor) for gpu in gpu_infos):
            return True
        if backend == "metal":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return True
            except Exception:
                pass
        if backend == "vulkan":
            try:
                import subprocess
                ret = subprocess.run(
                    ["vulkaninfo", "--summary"], capture_output=True, timeout=2
                )
                if ret.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    # -------------------------------------------------------------------------
    # Lazy backend loaders (unchanged except for formatting)
    # -------------------------------------------------------------------------
    def _load_cuda(self) -> Optional[Callable]:
        try:
            from local_llm_core.kernels.cuda_optimizer.cuda_kernels import (
                load_cuda_kernel,
                ternary_matmul,
            )
            if load_cuda_kernel() is not None:
                return ternary_matmul
        except Exception as e:
            log.warning("CUDA kernels load failed: %s", e)
        return None

    def _load_hip(self) -> Optional[Callable]:
        try:
            from local_llm_core.kernels.hip_optimizer.hip_kernels import (
                load_hip_kernel,
                hip_ternary_matmul,
            )
            if load_hip_kernel() is not None:
                return hip_ternary_matmul
        except Exception as e:
            log.warning("HIP kernels load failed: %s", e)
        return None

    def _load_metal(self) -> Optional[Callable]:
        try:
            from local_llm_core.kernels.metal_kernels.metal_loader import (
                _load_metal_kernel,
                metal_ternary_matmul,
            )
            if _load_metal_kernel():
                return metal_ternary_matmul
        except Exception as e:
            log.warning("Metal kernels load failed: %s", e)
        return None

    def _load_vulkan(self) -> Optional[Callable]:
        try:
            from local_llm_core.kernels.vulkan_optimizer.vulkan_loader import (
                vulkan_ternary_matmul,
            )
            return vulkan_ternary_matmul
        except Exception as e:
            log.warning("Vulkan kernels load failed: %s", e)
        return None

    def _apply_cpu_optimizer(self) -> None:
        try:
            from local_llm_core.kernels.cpu_optimizer.cpu_tuning import apply_from_config
            apply_from_config({"cpu_optimizer": self._cfg.get("cpu_optimizer", {})})
        except Exception as e:
            log.debug("CPU optimizer not applied: %s", e)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def get_matmul_fn(self) -> Optional[Callable]:
        """Return the active custom matmul function, or None if none is active."""
        return self._backend_fn

    def patch_pytorch(self) -> None:
        """Replace torch.matmul and F.linear with the custom kernel wrappers."""
        if not self._selected_backend or self._patch_installed:
            return
        if self._backend_fn is not None:
            install_patch(self._backend_fn)
            self._patch_installed = True
            log.info("PyTorch ops patched with %s kernels", self._selected_backend)

    def unpatch_pytorch(self) -> None:
        """Restore the original PyTorch operations."""
        if self._patch_installed:
            remove_patch()
            self._patch_installed = False

    def shutdown(self) -> None:
        """Clean up: remove the patch and release any backend resources."""
        self.unpatch_pytorch()
        log.info("UniversalKernelManager shut down")
