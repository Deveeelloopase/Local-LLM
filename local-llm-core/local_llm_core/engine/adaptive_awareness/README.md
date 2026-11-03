# engine/adaptive_awareness/

Decides **how** a model file should be run, before anything is loaded.

| File | Key names | Purpose |
|---|---|---|
| `model_unified_env.py` | `GGUFTypeDetector`, `GGUFHeader`, `GGUFModelType`, `PRISMRouter`, `classify_gguf()`, `route_model()`, `InferenceBackend`, `AdaptiveModelRouter`, `UnifiedEnvironmentValidator` / `validate_environment()`, `EnvironmentStatus`, `BitNetStateManager` / `BitNetState` / `get_bitnet_state()`, `auto_detect_backend()`, `auto_detect_backend_info()` | Reads the GGUF header, classifies the file (standard quantisation vs BitNet i2_s / TL1 / TL2), checks that the environment can run it (BitNet.cpp build, Python packages) and returns the backend to use |
| `__init__.py` | – | Lazy re-exports |

"PRISM" (*Proactive Repository Intelligence for Smart Model-routing*) is the classification algorithm; `scripts/lib/prism_detect.py` is its standalone, stdlib-only copy used by the setup scripts to classify model directories. `BitNetStateManager` reads the state file written by `scripts/bitnet_setup.sh` (`~/.config/local-llm/bitnet_state.json`; the script derives the directory from `LOCAL_LLM_STATE_DIR`).

Consumers: `engine/gguf/model_loader.py` and the server's model-loading routes call `route_model()` / `classify_gguf()` to pick `ModelLoader` or `BitNetLoader`.
