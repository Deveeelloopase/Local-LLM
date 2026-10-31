"""
engine/pipelines/__init__.py — Production‑Ready Inference Pipelines
===================================================================
Exports the core pipeline components for model loading and drive‑backed inference.

All configuration is read from config.yaml by the individual modules.
No hard‑coded values; everything is config‑driven.

Public API:
- Layer Streaming: AdaptiveLoader, FLPConfig, read_gguf_meta, calc_optimal_gpu_layers, BackendProbe
- Drive Pipeline: ContextCache, LayerPrefetcher, CPUThrottler, ConcurrencyTokenManager, ThroughputController,
  NVMeModelAdvisor, MemoryPressureMonitor, NVMeConfig, advise_model_file
"""

__version__ = "1.0.0"

# -----------------------------------------------------------------------------
# Layer Streamer (Fractional Layer Pipeline)
# -----------------------------------------------------------------------------
try:
    from .layer_streamer import (
        AdaptiveLoader,
        FLPConfig,
        read_gguf_meta,
        calc_optimal_gpu_layers,
        BackendProbe,
    )
except ImportError as e:
    AdaptiveLoader = None
    FLPConfig = None
    read_gguf_meta = None
    calc_optimal_gpu_layers = None
    BackendProbe = None
    import logging
    logging.getLogger(__name__).warning("layer_streamer import failed: %s", e)

# -----------------------------------------------------------------------------
# Drive Pipeline (NVMe‑backed acceleration)
# -----------------------------------------------------------------------------
try:
    from .drive_pipeline import (
        ContextCache,
        LayerPrefetcher,
        CPUThrottler,
        ConcurrencyTokenManager,
        ThroughputController,
        NVMeModelAdvisor,
        MemoryPressureMonitor,
        NVMeConfig,
        advise_model_file,
        TARGET_VRAM_PCT,
        TARGET_RAM_PCT,
        TARGET_TPS,
        CPU_THREAD_RATIO,
    )
except ImportError as e:
    ContextCache = None
    LayerPrefetcher = None
    CPUThrottler = None
    ConcurrencyTokenManager = None
    ThroughputController = None
    NVMeModelAdvisor = None
    MemoryPressureMonitor = None
    NVMeConfig = None
    advise_model_file = None
    TARGET_VRAM_PCT = None
    TARGET_RAM_PCT = None
    TARGET_TPS = None
    CPU_THREAD_RATIO = None
    import logging
    logging.getLogger(__name__).warning("drive_pipeline import failed: %s", e)

# -----------------------------------------------------------------------------
# Public Exports
# -----------------------------------------------------------------------------
__all__ = [
    # Layer Streamer
    "AdaptiveLoader",
    "FLPConfig",
    "read_gguf_meta",
    "calc_optimal_gpu_layers",
    "BackendProbe",
    # Drive Pipeline
    "ContextCache",
    "LayerPrefetcher",
    "CPUThrottler",
    "ConcurrencyTokenManager",
    "ThroughputController",
    "NVMeModelAdvisor",
    "MemoryPressureMonitor",
    "NVMeConfig",
    "advise_model_file",
    "TARGET_VRAM_PCT",
    "TARGET_RAM_PCT",
    "TARGET_TPS",
    "CPU_THREAD_RATIO",
    # Version
    "__version__",
]
