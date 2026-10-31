# local_llm_core/memory/ — engine memory tiers

Manages where model data lives: VRAM, RAM, NVMe/SSD. (This is **not** the server's chat-history store — that is `local_llm_server/memory/`.)

| File | Key classes | Purpose |
|---|---|---|
| `vram_manager.py` | `VRAMManager`, `TensorBlock`, `AllocationStats` | VRAM allocation with LRU eviction, defragmentation and prefetching |
| `ram_optimizer.py` | `RAMOptimizer` | Two-level RAM cache |
| `swap_handler.py` | `SwapHandler` | Asynchronous SSD swap; keys map to flat files under a swap directory |
| `hybrid_manager.py` | `HybridMemoryManager`, `MemoryTier`, `HybridBlock` | Three-tier LRU hierarchy (VRAM → RAM → disk) for weight caching |
| `hybrid_cache.py` | `HybridCache`, `LayerLocation` | Tracks each model layer across RAM and NVMe with promotion/demotion |
| `connection_pool.py` | `SimpleConnectionPool` | Small asyncio SQLite connection pool (semaphore-based) |
| `config_loader.py` | config dataclasses | Reads the `memory` config section |
| `__init__.py` | – | Lazy re-exports |

## Config (`memory:` section)

`vram_target_utilization`, `vram_defrag_*`, `vram_history_window`, `vram_prefetch_alpha`, `vram_allocation_retries`; `ram_target_utilization`, `ram_max_mb`, `ram_compress_ratio_threshold`, `ram_max_mmaps`, `ram_max_prefetch_concurrent`; `ssd_swap_path`, `ssd_swap_max_gb`, `ssd_compression_enabled`, `ssd_compression_level`; nested `hybrid`, `hybrid_cache` and `sqlite` blocks. Typed in `config_loader.py` (`VRAMConfig`, `RAMConfig`, `SwapConfig`, `HybridConfig`, `HybridCacheConfig`, `SQLiteConfig`, `MemoryConfig`).

## Relationship to `engine/`

`engine/pipelines/` (layer streaming, drive pipeline) and `engine/models/nvme_kv_cache.py` build on these managers to decide how many layers go to the GPU and where KV data spills.
