from __future__ import annotations

"""
memory/config_loader.py — Configuration schemas for the Memory Management layer
================================================================================
Pydantic models for VRAM, RAM, Swap, Hybrid cache, SQLite, and the top‑level
`memory` section.  Used by all memory modules for optional validation.

Usage:
    # From a raw config dict (e.g., loaded from YAML)
    mem_cfg = MemoryConfig(**config.get("memory", {}))
    vram_manager = VRAMManager(total_vram_bytes, mem_cfg.vram)

    # Or pass the entire `memory` dict directly (backward‑compatible)
    vram_manager = VRAMManager(total_vram_bytes, config={"memory": mem_dict})
"""


from pydantic import BaseModel, Field
from typing import Optional


class VRAMConfig(BaseModel):
    """VRAM manager settings."""
    vram_target_utilization: float = Field(
        default=0.95, ge=0.5, le=0.99,
        description="Target fraction of VRAM to use"
    )
    vram_defrag_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Fragmentation threshold; null = auto‑scale"
    )
    vram_history_window: int = Field(default=256, ge=1)
    vram_prefetch_alpha: float = Field(default=0.1, ge=0.0, le=1.0)
    vram_defrag_check_interval_s: float = Field(default=60.0, ge=1.0)
    vram_allocation_retries: int = Field(default=3, ge=1)


class SwapConfig(BaseModel):
    """SSD swap handler settings."""
    ssd_swap_path: str = Field(default="/tmp/llm_swap", description="Directory for swap files")
    ssd_swap_max_gb: float = Field(default=32.0, ge=1.0, description="Maximum SSD swap space (GB)")
    ssd_compression_enabled: bool = True
    ssd_compression_level: int = Field(default=0, ge=0, le=9, description="LZ4 compression level (0 = fast)")


class RAMConfig(BaseModel):
    """RAM optimizer settings."""
    ram_max_mb: int = Field(default=0, ge=0, description="Max RAM budget in MB (0 = auto)")
    ram_compress_ratio_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Min compression ratio to keep as warm"
    )
    ram_max_mmaps: int = Field(default=64, ge=1)
    ram_max_prefetch_concurrent: int = Field(default=16, ge=1)


class HybridConfig(BaseModel):
    """Hybrid memory manager settings (three‑tier: VRAM‑RAM‑SSD)."""
    ssd_path: str = Field(default="/tmp/llm_hybrid", description="SSD directory for hybrid cache")
    ssd_max_gb: float = Field(default=50.0, ge=1.0, description="Maximum SSD space (GB)")
    compression_enabled: bool = True
    promotion_threshold: float = Field(
        default=0.95, ge=0.5, le=1.0,
        description="VRAM usage threshold for promoting blocks"
    )


class HybridCacheConfig(BaseModel):
    """Hybrid RAM+NVMe layer cache settings."""
    ram_budget_mb: int = Field(default=512, ge=1, description="RAM budget for layer cache (MB)")
    nvme_budget_mb: int = Field(default=5000, ge=1, description="NVMe budget for layer cache (MB)")
    promote_threshold_per_min: float = Field(default=2.0, ge=0.0, description="Access freq/min to promote to RAM")
    demote_threshold_per_min: float = Field(default=0.5, ge=0.0, description="Access freq/min below which demote to NVMe")
    rebalance_interval_s: float = Field(default=60.0, ge=1.0, description="Background rebalance interval (seconds)")
    ram_pressure_threshold: float = Field(default=0.90, ge=0.5, le=1.0, description="RAM usage ratio to trigger eviction")
    ewma_alpha: float = Field(default=0.1, ge=0.0, le=1.0, description="EWMA smoothing factor for access frequency")


class SQLiteConfig(BaseModel):
    """SQLite connection pool settings."""
    db_path: str = Field(default="/tmp/llm_keys.db", description="Path to SQLite database file")
    max_connections: int = Field(default=10, ge=1, description="Maximum concurrent connections")
    connection_timeout_s: float = Field(default=5.0, ge=0.1, description="Timeout for acquiring a connection (seconds)")


# ---------------------------------------------------------------------------
# Master Memory Configuration
# ---------------------------------------------------------------------------

class MemoryConfig(BaseModel):
    """
    Top‑level `memory` section configuration.

    All sub‑sections are optional and default to their respective defaults.
    Use this to validate a complete `memory` block, e.g.:
        MemoryConfig(**config["memory"])
    """
    vram: VRAMConfig = Field(default_factory=VRAMConfig)
    swap: SwapConfig = Field(default_factory=SwapConfig)
    ram: RAMConfig = Field(default_factory=RAMConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    hybrid_cache: HybridCacheConfig = Field(default_factory=HybridCacheConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    