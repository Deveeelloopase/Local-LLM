from __future__ import annotations

"""
engine/__init__.py — Production‑Ready Inference Engine
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
import importlib
from typing import Dict

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lazy Import Machinery
# -----------------------------------------------------------------------------
_LAZY_IMPORTS = {
    #Adaptive Awareness
    "auto_detect_backend":  ("adaptive_awareness", "auto_detect_backend"),
    "classify_gguf":        ("adaptive_awareness", "classify_gguf"),
    "route_model":          ("adaptive_awareness", "route_model"),
    # BitNet
    "BitNetLoader": ("bitnet", "BitNetLoader"),
    "BitNetModelConfig": ("bitnet", "BitNetModelConfig"),
    "BitNetInferenceWrapper": ("bitnet", "BitNetInferenceWrapper"),
    "BitNetMemoryProfile": ("bitnet", "BitNetMemoryProfile"),
    "BitNetMemoryManager": ("bitnet", "BitNetMemoryManager"),
    "BitNetKernelManager": ("bitnet", "BitNetKernelManager"),
    # Exceptions
    "LocalLLMError": ("exceptions", "LocalLLMError"),
    "ErrorSeverity": ("exceptions", "ErrorSeverity"),
    "HTTPStatusMapping": ("exceptions", "HTTPStatusMapping"),
    "ConfigError": ("exceptions", "ConfigError"),
    "HardwareError": ("exceptions", "HardwareError"),
    "ModelError": ("exceptions", "ModelError"),
    "InferenceError": ("exceptions", "InferenceError"),
    "MemoryExhaustedError": ("exceptions", "MemoryExhaustedError"),
    "SchedulingError": ("exceptions", "SchedulingError"),
    "CacheError": ("exceptions", "CacheError"),
    "StorageError": ("exceptions", "StorageError"),
    "RateLimitError": ("exceptions", "RateLimitError"),
    "AuthError": ("exceptions", "AuthError"),
    "TimeoutError": ("exceptions", "TimeoutError"),
    "ServerError": ("exceptions", "ServerError"),
    "exception_to_http_response": ("exceptions", "exception_to_http_response"),
    "log_exception": ("exceptions", "log_exception"),
    # GGUF Inference Engine
    "AsyncTokenBuffer": ("gguf", "AsyncTokenBuffer"),
    "BoundedThreadPoolExecutor": ("gguf", "BoundedThreadPoolExecutor"),
    "CircuitBreaker": ("gguf", "CircuitBreaker"),
    "RateLimiter": ("gguf", "RateLimiter"),
    "TokenBucket": ("gguf", "TokenBucket"),
    "AutoTuner": ("gguf", "AutoTuner"),
    "NGramCache": ("gguf", "NGramCache"),
    "RequestQueue": ("gguf", "RequestQueue"),
    "ModelLoader": ("gguf", "ModelLoader"),
    "ModelConfig": ("gguf", "ModelConfig"),
    "LoadedModel": ("gguf", "LoadedModel"),
    "SpeculativeEngine": ("gguf", "SpeculativeEngine"),
    "SpecStats": ("gguf", "SpecStats"),
    # Hardware Profiles
    "HardwareProfile": ("hardware", "HardwareProfile"),
    "detect_profile": ("hardware", "detect_profile"),
    "get_profile": ("hardware", "get_profile"),
    "list_profile_keys": ("hardware", "list_profile_keys"),
    # Inference Optimizations
    "InferenceEngine": ("inference", "InferenceEngine"),
    "InferenceRequest": ("inference", "InferenceRequest"),
    "InferenceResponse": ("inference", "InferenceResponse"),
    "ThroughputOptimizer": ("inference", "ThroughputOptimizer"),
    "ContinuousBatchScheduler": ("inference", "ContinuousBatchScheduler"),
    "SlotRequest": ("inference", "SlotRequest"),
    "TensorSplitPlan": ("inference", "TensorSplitPlan"),
    "compute_tensor_split": ("inference", "compute_tensor_split"),
    "should_split": ("inference", "should_split"),
    "SplitStrategy": ("inference", "SplitStrategy"),
    "clear_split_cache": ("inference", "clear_split_cache"),
    "AsyncWeightPipeline": ("inference", "AsyncWeightPipeline"),
    "ModelWeightPipelineManager": ("inference", "ModelWeightPipelineManager"),
    "get_pipeline_manager": ("inference", "get_pipeline_manager"),
    "LayerRegion": ("inference", "LayerRegion"),
    "estimate_layer_regions": ("inference", "estimate_layer_regions"),
    # Model Support (NVMe KV Cache)
    "NVMeKVCache": ("models", "NVMeKVCache"),
    "KVCacheEntry": ("models", "KVCacheEntry"),
    # Pipelines (Fractional Layer Pipeline, Drive Pipeline)
    "FLPConfig": ("pipelines", "FLPConfig"),
    "AdaptiveLoader": ("pipelines", "AdaptiveLoader"),
    "BackendProbe": ("pipelines", "BackendProbe"),
    "calc_optimal_gpu_layers": ("pipelines", "calc_optimal_gpu_layers"),
    "read_gguf_meta": ("pipelines", "read_gguf_meta"),
    "ContextCache": ("pipelines", "ContextCache"),
    "LayerPrefetcher": ("pipelines", "LayerPrefetcher"),
    "CPUThrottler": ("pipelines", "CPUThrottler"),
    "ConcurrencyTokenManager": ("pipelines", "ConcurrencyTokenManager"),
    "ThroughputController": ("pipelines", "ThroughputController"),
    "NVMeModelAdvisor": ("pipelines", "NVMeModelAdvisor"),
    "MemoryPressureMonitor": ("pipelines", "MemoryPressureMonitor"),
    "NVMeConfig": ("pipelines", "NVMeConfig"),
    "advise_model_file": ("pipelines", "advise_model_file"),
    # Scheduling
    "Scheduler": ("scheduling", "Scheduler"),
    "ScheduledRequest": ("scheduling", "ScheduledRequest"),
    "Priority": ("scheduling", "Priority"),
    # Utilities (Configuration)
    "load_config": ("utils", "load_config"),
    "HardwareConfig": ("utils", "HardwareConfig"),
    "SystemConfig": ("utils", "SystemConfig"),
    "MemoryConfig": ("utils", "MemoryConfig"),
    "InferenceConfig": ("utils", "InferenceConfig"),
    "DriveConfig": ("utils", "DriveConfig"),
    "CircuitBreakerConfig": ("utils", "CircuitBreakerConfig"),
    "ModelsConfig": ("utils", "ModelsConfig"),
    "ModelProfileConfig": ("utils", "ModelProfileConfig"),
    "SpeculativeConfig": ("utils", "SpeculativeConfig"),
    "MultiGPUConfig": ("utils", "MultiGPUConfig"),
    "ThroughputOptimizerConfig": ("utils", "ThroughputOptimizerConfig"),
    "NVMeKVCacheConfig": ("utils", "NVMeKVCacheConfig"),
    "LayerStreamerConfig": ("utils", "LayerStreamerConfig"),
    "DrivePipelineConfig": ("utils", "DrivePipelineConfig"),
    "SchedulerConfig": ("utils", "SchedulerConfig"),
}


def __getattr__(name: str):
    """Lazy import of engine submodules with detailed error logging."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'engine' has no attribute {name!r}")

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


# -----------------------------------------------------------------------------
# Validation Helper (for CI / startup checks)
# -----------------------------------------------------------------------------
def validate_imports() -> Dict[str, bool]:
    """
    Attempt to import every public symbol. Returns a dict of {name: success}.
    Use in CI to catch broken lazy imports before deployment.
    """
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
