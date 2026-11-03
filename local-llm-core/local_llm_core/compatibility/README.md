# local_llm_core/compatibility/

Makes BitNet checkpoints loadable across different `transformers` versions (the "transformers path" used when the BitNet.cpp backend is not chosen).

| File | Purpose |
|---|---|
| `transformers_compat.py` | `TransformersCompatibilityManager`, `BitLinearCompatibilityLayer` (`ensure_bitlinear()`, `is_bitlinear_available()`), `StateFileReader` (reads the state file written by `bitnet_setup.sh`), `BitNetModelLoader` / `load_bitnet_model()`, `TransformersBackendRouter` / `get_transformers_backend()`, `get_compatibility_report()`, helpers (`patch_docstring_validator`, `suppress_warnings`, `get_safe_tokenizer_kwargs`) |
| `__init__.py` | Re-exports |

Imported by `engine/adaptive_awareness/model_unified_env.py`, `engine/inference/inference_engine.py` and the BitNet modules (`engine/bitnet/bitnet_loader.py`, `bitnet_engine.py`).

`requirements.txt` (core) documents the one deliberate version range for this path, `transformers>=4.46.3,<5.0`, in a commented `[bitnet]` block; it is installed by the BitNet setup rather than by default.