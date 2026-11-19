# engine/utils/

| File | Purpose |
|---|---|
| `config_loader.py` | Typed (Pydantic) models for the engine's configuration and the loader that fills them from `config.yaml`: `LocalLLMCloudConfig` with sections such as `ModelsConfig`, `InferenceConfig`, `SchedulerConfig`, `KernelsConfig`, `ModelLifecycleConfig`, `MultiGPUConfig`, `SpeculativeConfig`, `BitNetSystemConfig`, `BitNetProfileConfig`, `NVMeKVCacheConfig`, `ThroughputOptimizerConfig`, `LayerStreamerConfig`, `DrivePipelineConfig` (and its sub-configs), `SysctlConfig`, `PreemptionConfig`, `ModelProfileConfig`; entry points `ConfigLoader` and `load_config()` |
| `__init__.py` | Re-exports |

This is the single place where defaults for engine settings are defined. Add a new setting here first, then read it from the typed object — avoid reading raw dictionaries elsewhere.
