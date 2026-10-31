"""
local_llm_core — Production LLM Inference System
================================================
Lazy‑loading entry point for the entire inference stack.

All public components are accessible via attribute access, triggering
on‑demand imports to reduce startup time and avoid circular dependencies.

Configuration is loaded from config.yaml via the `utils` subpackage.
No hard‑coded values; everything is config‑driven.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Deveeelloopase"

import logging
import importlib

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lazy Import Machinery
# -----------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # -------------------------------------------------------------------------
    # Engine — GGUF Inference
    # -------------------------------------------------------------------------
    "InferenceEngine": ("engine", "InferenceEngine"),
    "InferenceRequest": ("engine", "InferenceRequest"),
    "InferenceResponse": ("engine", "InferenceResponse"),
    "ModelLoader": ("engine", "ModelLoader"),
    "ModelConfig": ("engine", "ModelConfig"),
    "LoadedModel": ("engine", "LoadedModel"),
    "SpeculativeEngine": ("engine", "SpeculativeEngine"),
    # -------------------------------------------------------------------------
    # Engine — BitNet (Core)
    # -------------------------------------------------------------------------
    "BitNetLoader": ("engine.bitnet", "BitNetLoader"),
    "BitNetModelConfig": ("engine.bitnet", "BitNetModelConfig"),
    "BitNetInferenceWrapper": ("engine.bitnet", "BitNetInferenceWrapper"),
    "BitNetMemoryManager": ("engine.bitnet", "BitNetMemoryManager"),
    "BitNetKernelManager": ("engine.bitnet", "BitNetKernelManager"),  # alias
    # -------------------------------------------------------------------------
    # Engine — BitNet (State & Router)
    # -------------------------------------------------------------------------
    "BitNetState": ("adaptive_awareness.model_unified_env", "BitNetState"),
    "get_bitnet_state": ("adaptive_awareness.model_unified_env", "get_bitnet_state"),
    "AdaptiveModelRouter": ("adaptive_awareness.model_unified_env", "AdaptiveModelRouter"),
    "GGUFModelType": ("adaptive_awareness.model_unified_env", "GGUFModelType"),
    "InferenceBackend": ("adaptive_awareness.model_unified_env", "InferenceBackend"),
    "classify_gguf": ("adaptive_awareness.model_unified_env", "classify_gguf"),
    "route_model": ("adaptive_awareness.model_unified_env", "route_model"),
    "auto_detect_backend": ("adaptive_awareness.model_unified_env", "auto_detect_backend"),
    "validate_environment": ("adaptive_awareness.model_unified_env", "validate_environment"),
    "EnvironmentStatus": ("adaptive_awareness.model_unified_env", "EnvironmentStatus"),
    # -------------------------------------------------------------------------
    # Engine — BitNet C++ Backend (optional)
    # -------------------------------------------------------------------------
    "BitNetCppBackend": ("engine.bitnet.bitnet_cpp_backend", "BitNetCppBackend"),
    "BitNetCppConfig": ("engine.bitnet.bitnet_cpp_backend", "BitNetCppConfig"),
    "create_best_backend": ("engine.bitnet.bitnet_cpp_backend", "create_best_backend"),
    # -------------------------------------------------------------------------
    # Engine — Exceptions
    # -------------------------------------------------------------------------
    "LocalLLMError": ("engine", "LocalLLMError"),
    "ConfigError": ("engine", "ConfigError"),
    "HardwareError": ("engine", "HardwareError"),
    "ModelError": ("engine", "ModelError"),
    "InferenceError": ("engine", "InferenceError"),
    "MemoryExhaustedError": ("engine", "MemoryExhaustedError"),
    "SchedulingError": ("engine", "SchedulingError"),
    "RateLimitError": ("engine", "RateLimitError"),
    "AuthError": ("engine", "AuthError"),
    "TimeoutError": ("engine", "TimeoutError"),
    "ServerError": ("engine", "ServerError"),
    "exception_to_http_response": ("engine", "exception_to_http_response"),
    "log_exception": ("engine", "log_exception"),
    # -------------------------------------------------------------------------
    # Engine — Hardware Profiles (authoritative)
    # -------------------------------------------------------------------------
    "HardwareProfile": ("engine", "HardwareProfile"),
    "detect_profile": ("engine", "detect_profile"),
    "get_profile": ("engine", "get_profile"),
    # -------------------------------------------------------------------------
    # Engine — Inference Optimizations
    # -------------------------------------------------------------------------
    "ThroughputOptimizer": ("engine", "ThroughputOptimizer"),
    "ContinuousBatchScheduler": ("engine", "ContinuousBatchScheduler"),
    "TensorSplitPlan": ("engine", "TensorSplitPlan"),
    "compute_tensor_split": ("engine", "compute_tensor_split"),
    "AsyncWeightPipeline": ("engine", "AsyncWeightPipeline"),
    # -------------------------------------------------------------------------
    # Engine — Pipelines
    # -------------------------------------------------------------------------
    "FLPConfig": ("engine", "FLPConfig"),
    "AdaptiveLoader": ("engine", "AdaptiveLoader"),
    "ContextCache": ("engine", "ContextCache"),
    "ThroughputController": ("engine", "ThroughputController"),
    "MemoryPressureMonitor": ("engine", "MemoryPressureMonitor"),
    # -------------------------------------------------------------------------
    # Engine — Scheduling
    # -------------------------------------------------------------------------
    "Scheduler": ("engine", "Scheduler"),
    "ScheduledRequest": ("engine", "ScheduledRequest"),
    "Priority": ("engine", "Priority"),
    # -------------------------------------------------------------------------
    # Engine — Utilities (Configuration)
    # -------------------------------------------------------------------------
    "ConfigLoader": ("engine", "ConfigLoader"),
    "LocalLLMCloudConfig": ("engine", "LocalLLMCloudConfig"),
    "load_config": ("engine", "load_config"),
    "ServerConfig": ("engine", "ServerConfig"),
    "APIConfig": ("engine", "APIConfig"),
    "HardwareConfig": ("engine", "HardwareConfig"),
    "InferenceConfig": ("engine", "InferenceConfig"),
    "ModelsConfig": ("engine", "ModelsConfig"),
    "LoggingConfig": ("engine", "LoggingConfig"),
    # -------------------------------------------------------------------------
    # Engine — Kernels (NEW)
    # -------------------------------------------------------------------------
    "UniversalKernelManager": ("engine.kernels", "UniversalKernelManager"),
    # -------------------------------------------------------------------------
    # Engine — Model Lifecycle Manager (NEW)
    # -------------------------------------------------------------------------
    "ModelLifecycleManager": ("engine.models", "ModelLifecycleManager"),
    # -------------------------------------------------------------------------
    # HAL — Hardware Abstraction Layer (detector, analyzer, optimizer)
    # -------------------------------------------------------------------------
    "HardwareDetector": ("hal", "HardwareDetector"),
    "GPUInfo": ("hal", "GPUInfo"),
    "SystemInfo": ("hal", "SystemInfo"),
    "GPUVendor": ("hal", "GPUVendor"),
    "GPUAnalyzer": ("hal", "GPUAnalyzer"),
    "SystemOptimizer": ("hal", "SystemOptimizer"),
    # -------------------------------------------------------------------------
    # Memory — Three‑Tier Memory Management
    # -------------------------------------------------------------------------
    "VRAMManager": ("memory", "VRAMManager"),
    "AllocationStats": ("memory", "AllocationStats"),
    "RAMOptimizer": ("memory", "RAMOptimizer"),
    "SwapHandler": ("memory", "SwapHandler"),
    "HybridMemoryManager": ("memory", "HybridMemoryManager"),
    "MemoryTier": ("memory", "MemoryTier"),
    "HybridCache": ("memory", "HybridCache"),
    "SimpleConnectionPool": ("memory", "SimpleConnectionPool"),
}


def __getattr__(name: str):
    """Lazy import of submodules."""
    if name not in _LAZY_IMPORTS:
        # Special case: the fast C extensions (_fast)
        if name == "_fast":
            try:
                mod = importlib.import_module("._fast", __package__)
                globals()["_fast"] = mod
                return mod
            except ImportError:
                log.warning("_fast module not available – continuing without native extensions")
                raise AttributeError(f"module 'local_llm_core' has no attribute {name!r}")
        raise AttributeError(f"module 'local_llm_core' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        attr = getattr(module, attr_name)
        globals()[name] = attr  # Cache for subsequent access
        return attr
    except ImportError as e:
        log.exception("Failed to lazy‑import %s from %s", name, module_name)
        raise ImportError(
            f"Could not import '{name}' from '{module_name}'. "
            f"This indicates a missing dependency or broken installation. "
            f"Original error: {e}"
        ) from e


def __dir__() -> list:
    """Return the list of public attributes for better IDE support."""
    return sorted(list(_LAZY_IMPORTS.keys()) + ["__version__", "__author__", "_fast"])


# -----------------------------------------------------------------------------
# Public API Definition
# -----------------------------------------------------------------------------
__all__ = list(_LAZY_IMPORTS.keys()) + ["__version__", "__author__", "_fast"]
