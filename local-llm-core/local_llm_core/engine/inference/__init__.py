"""
engine/inference/__init__.py — Production‑Ready Inference Optimizations
=======================================================================
Exports components for throughput optimization, tensor parallelism,
async weight prefetching, and speculative decoding setup.

All configuration is read from config.yaml via the respective modules.
No hard‑coded values; everything is config‑driven.

Public API:
- ThroughputOptimizer, ContinuousBatchScheduler, SlotRequest
- TensorSplitPlan, compute_tensor_split, should_split, SplitStrategy
- AsyncWeightPipeline, ModelWeightPipelineManager, get_pipeline_manager
- setup_speculative script utilities (optional, for CLI)
"""

__version__ = "1.0.0"

# -----------------------------------------------------------------------------
# Throughput Optimizer
# -----------------------------------------------------------------------------
try:
    from .throughput_optimizer import (
        ThroughputOptimizer,
        ContinuousBatchScheduler,
        SlotRequest,
    )
except ImportError as e:
    ThroughputOptimizer = None
    ContinuousBatchScheduler = None
    SlotRequest = None
    import logging
    logging.getLogger(__name__).warning("throughput_optimizer import failed: %s", e)

# -----------------------------------------------------------------------------
# Tensor Parallelism
# -----------------------------------------------------------------------------
try:
    from .tensor_parallel import (
        TensorSplitPlan,
        compute_tensor_split,
        should_split,
        SplitStrategy,
        clear_split_cache,
    )
except ImportError as e:
    TensorSplitPlan = None
    compute_tensor_split = None
    should_split = None
    SplitStrategy = None
    clear_split_cache = None
    import logging
    logging.getLogger(__name__).warning("tensor_parallel import failed: %s", e)

# -----------------------------------------------------------------------------
# Async Weight Pipeline
# -----------------------------------------------------------------------------
try:
    from .async_weight_pipe import (
        AsyncWeightPipeline,
        ModelWeightPipelineManager,
        get_pipeline_manager,
        LayerRegion,
        estimate_layer_regions,
    )
except ImportError as e:
    AsyncWeightPipeline = None
    ModelWeightPipelineManager = None
    get_pipeline_manager = None
    LayerRegion = None
    estimate_layer_regions = None
    import logging
    logging.getLogger(__name__).warning("async_weight_pipe import failed: %s", e)

# -----------------------------------------------------------------------------
# Speculative Setup Utilities (optional, for CLI)
# -----------------------------------------------------------------------------
try:
    from .setup_speculative import (
        cmd_check,
        cmd_download,
        cmd_enable,
        cmd_benchmark,
        find_config_path,
        load_config,
    )
except ImportError as e:
    cmd_check = None
    cmd_download = None
    cmd_enable = None
    cmd_benchmark = None
    find_config_path = None
    load_config = None
    import logging
    logging.getLogger(__name__).debug("setup_speculative import failed: %s", e)

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    # Throughput
    "ThroughputOptimizer",
    "ContinuousBatchScheduler",
    "SlotRequest",
    # Tensor Parallel
    "TensorSplitPlan",
    "compute_tensor_split",
    "should_split",
    "SplitStrategy",
    "clear_split_cache",
    # Async Weight Pipe
    "AsyncWeightPipeline",
    "ModelWeightPipelineManager",
    "get_pipeline_manager",
    "LayerRegion",
    "estimate_layer_regions",
    # Speculative Setup (CLI)
    "cmd_check",
    "cmd_download",
    "cmd_enable",
    "cmd_benchmark",
    "find_config_path",
    "load_config",
    # Version
    "__version__",
]