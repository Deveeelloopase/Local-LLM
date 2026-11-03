"""
engine/utils/__init__.py — Production‑Ready Utilities
=====================================================
Exports configuration loading and validation utilities.

Public API:
- ConfigLoader: YAML loader with env substitution and hot‑reload detection.
- LocalLLMCloudConfig: Master Pydantic configuration schema.
- load_config: Convenience function to load config from path.
- All sub‑config schemas for direct access when needed.

All configuration is read from config.yaml; no hard‑coded values.
"""

__version__ = "1.0.0"

import logging
from typing import Optional, TypeVar, Type

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Import all configuration classes from config_loader
# -----------------------------------------------------------------------------
try:
    from .config_loader import (
        # Core configurations
        InferenceConfig,
        ModelsConfig,
        ModelProfileConfig,
        SpeculativeConfig,
        MultiGPUConfig,
        ThroughputOptimizerConfig,
        NVMeKVCacheConfig,
        LayerStreamerConfig,
        DrivePipelineConfig,
        SchedulerConfig,
        # BitNet configurations
        BitNetSystemConfig,
        BitNetProfileConfig,
        # Additional engine configurations
        KernelsConfig,
        ModelLifecycleConfig,
        AsyncWeightPipelineConfig,
        ContextCacheConfig,
        LayerPrefetcherConfig,
        CPUThrottlerConfig,
        ConcurrencyConfig,
        ThroughputControllerConfig,
        NVMeAdvisorConfig,
        MemoryPressureConfig,
        SysctlConfig,
        PreemptionConfig,
    )
    _CONFIG_LOADER_OK = True
except ImportError as e:
    log.warning("config_loader import failed: %s – using None placeholders", e)
    _CONFIG_LOADER_OK = False
    # Placeholders to avoid NameError when accessing via __getattr__
    InferenceConfig = None
    ModelsConfig = None
    ModelProfileConfig = None
    SpeculativeConfig = None
    MultiGPUConfig = None
    ThroughputOptimizerConfig = None
    NVMeKVCacheConfig = None
    LayerStreamerConfig = None
    DrivePipelineConfig = None
    SchedulerConfig = None
    BitNetSystemConfig = None
    BitNetProfileConfig = None
    KernelsConfig = None
    ModelLifecycleConfig = None
    AsyncWeightPipelineConfig = None
    ContextCacheConfig = None
    LayerPrefetcherConfig = None
    CPUThrottlerConfig = None
    ConcurrencyConfig = None
    ThroughputControllerConfig = None
    NVMeAdvisorConfig = None
    MemoryPressureConfig = None
    SysctlConfig = None
    PreemptionConfig = None

# -----------------------------------------------------------------------------
# Optional: ConfigLoader and load_config (if available)
# -----------------------------------------------------------------------------
ConfigLoader: Optional[Type] = None
LocalLLMCloudConfig: Optional[Type] = None
load_config = None

if _CONFIG_LOADER_OK:
    try:
        from .config_loader import ConfigLoader, LocalLLMCloudConfig, load_config
    except ImportError:
        # These may be defined in a separate file; if not, keep None
        pass

# -----------------------------------------------------------------------------
# Public Exports
# -----------------------------------------------------------------------------
__all__ = [
    # Core configs
    "InferenceConfig",
    "ModelsConfig",
    "ModelProfileConfig",
    "SpeculativeConfig",
    "MultiGPUConfig",
    "ThroughputOptimizerConfig",
    "NVMeKVCacheConfig",
    "LayerStreamerConfig",
    "DrivePipelineConfig",
    "SchedulerConfig",
    # BitNet configs
    "BitNetSystemConfig",
    "BitNetProfileConfig",
    # Additional engine configs
    "KernelsConfig",
    "ModelLifecycleConfig",
    "AsyncWeightPipelineConfig",
    "ContextCacheConfig",
    "LayerPrefetcherConfig",
    "CPUThrottlerConfig",
    "ConcurrencyConfig",
    "ThroughputControllerConfig",
    "NVMeAdvisorConfig",
    "MemoryPressureConfig",
    "SysctlConfig",
    "PreemptionConfig",
    # Optional utilities
    "ConfigLoader",
    "LocalLLMCloudConfig",
    "load_config",
    "__version__",
]
