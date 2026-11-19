from __future__ import annotations

"""
hal/config_loader.py — Configuration schemas for the Hardware Abstraction Layer
================================================================================
Pydantic models that validate the `system`, `hardware`, and `gpu_analyzer`
sections of the application config.

Version 1.0.0

Used by:
- SystemOptimizer
- HardwareDetector
- GPUAnalyzer
"""


from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# System / OS Tuning
# -----------------------------------------------------------------------------

class VMConfig(BaseModel):
    """Virtual memory tunables."""
    swappiness: int = Field(default=10, ge=0, le=100)
    dirty_ratio: int = Field(default=40, ge=0, le=100)
    dirty_background_ratio: int = Field(default=5, ge=0, le=100)
    vfs_cache_pressure: int = Field(default=50, ge=0)
    overcommit_memory: int = Field(default=1, ge=0, le=2)


class HugePagesConfig(BaseModel):
    """Huge pages allocation parameters."""
    auto_pct: float = Field(default=12.0, ge=0.0, le=100.0,
                            description="Percent of RAM to allocate as huge pages")
    min_pages: int = Field(default=64, ge=1, description="Minimum huge pages")
    max_pages: int = Field(default=512, ge=1, description="Maximum huge pages")
    warn_threshold_pct: float = Field(default=15.0, ge=0.0, le=100.0,
                                      description="Percent RAM usage threshold for warning")
    info_threshold_pct: float = Field(default=10.0, ge=0.0, le=100.0,
                                      description="Percent RAM usage threshold for info log")


class SystemConfig(BaseModel):
    """OS‑level tuning configuration for the `system` section."""
    enable_huge_pages: bool = True
    huge_pages: HugePagesConfig = Field(default_factory=HugePagesConfig)

    enable_thp: bool = True
    thp_mode: str = "madvise"  # one of "always", "madvise", "never"

    enable_cpu_governor: bool = True
    cpu_governor: str = "performance"

    enable_io_scheduler: bool = True
    io_scheduler: str = "mq-deadline"

    enable_memlock: bool = True

    enable_vm_tune: bool = True
    vm: VMConfig = Field(default_factory=VMConfig)

    # Forwarded to the CPU optimizer module
    cpu_optimizer: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Hardware Detection
# -----------------------------------------------------------------------------

class HardwareConfig(BaseModel):
    """Hardware detection configuration (section `hardware`)."""
    cmd_timeout_s: int = Field(default=5, ge=1, description="Timeout for subprocess commands")
    detection_timeout_s: float = Field(default=30.0, ge=1.0,
                                       description="Global detection timeout")
    vulkan_icd_paths: List[str] = Field(
        default=[
            "/usr/share/vulkan/icd.d",
            "/etc/vulkan/icd.d",
            "/usr/lib/x86_64-linux-gnu/GL/vulkan/icd.d",
        ],
        description="Paths to search for Vulkan ICD files"
    )
    backend_order: List[str] = Field(
        default=["cuda", "rocm", "vulkan", "opencl", "mps", "numba", "numpy", "cpu"],
        description="Preferred backend order"
    )
    preferred_backend: str = Field(default="auto",
                                   description="Force a specific backend ('auto' for automatic)")


# -----------------------------------------------------------------------------
# GPU Analyzer
# -----------------------------------------------------------------------------

class GpuAnalyzerConfig(BaseModel):
    """GPU analyzer configuration (section `gpu_analyzer` or fallback to `hardware`)."""
    cmd_timeout_s: int = Field(default=8, ge=1, description="Timeout for subprocess commands")
    thermal_throttle_temp_c: float = Field(default=85.0, ge=50.0, le=120.0,
                                           description="Temperature threshold for throttling (°C)")
    default_ram_bw_gbps: float = Field(default=50.0, gt=0.0,
                                       description="Assumed RAM bandwidth if unknown")
    default_fp32_tflops: float = Field(default=10.0, gt=0.0,
                                       description="Assumed FP32 TFLOPS for unknown GPUs")
    default_mem_bw_gbps: float = Field(default=200.0, gt=0.0,
                                       description="Assumed GPU memory bandwidth if unknown")
    default_l2_cache_mb: float = Field(default=4.0, gt=0.0,
                                       description="Assumed L2 cache size if unknown")
    default_pcie_bw_gbps: float = Field(default=16.0, gt=0.0,
                                        description="Default PCIe bandwidth (GB/s)")
    default_pcie_gen: int = Field(default=4, ge=1, le=5)
    default_pcie_lanes: int = Field(default=16, ge=1)
    model_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per‑model overrides (keys: fp32_tflops, mem_bw_gbps, l2_cache_mb)"
    )

