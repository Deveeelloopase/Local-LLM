# engine/models/

State of loaded models and the KV-cache disk tier.

| File | Key classes | Purpose |
|---|---|---|
| `model_lifecycle.py` | `ModelLifecycleManager`, `ModelLifecycleInfo` | Tracks every loaded model across all backends; evicts idle models and, under memory pressure, according to the configured policy |
| `nvme_kv_cache.py` | `NVMeKVCache`, `KVCacheEntry`, `remove_temp_kv()`, `remove_temp_meta()` | Stores KV-cache entries, keyed by a prompt prefix (`prefix_len`), on NVMe |
| `__init__.py` | – | Lazy re-exports |

## Config

- `model_lifecycle:` — `max_loaded_models`, `idle_timeout_s`, `vram_target_utilization`, `ram_target_utilization`, `eviction_policy`, `auto_evict_on_pressure`, `check_interval_s`.
- `nvme_kv_cache:` — `enabled`, `dir`, `max_size_gb`, `max_entries`, `prefix_len`.
- 