# engine/exceptions/

One exception hierarchy for the whole system, with HTTP status mapping.

| File | Contents |
|---|---|
| `exceptions.py` | `LocalLLMError` (base) and subclasses grouped by area — config (`ConfigError`, `InvalidModelPathError`, …), hardware (`HardwareError`, `GPUNotFoundError`, `BackendNotAvailableError`, `ThermalThrottleError`), model (`ModelError`, `ModelNotFoundError`, `ModelLoadError`, `IncompatibleModelError`, `CorruptModelError`), inference (`InferenceError`, `SpeculativeDecodingError`, `TokenDecodingError`, `SamplingError`, `GenerationInterruptedError`, `GGMLAssertionError`), memory (`MemoryExhaustedError`, `VRAMOvercommitError`, `KVCacheAllocationError`, …), plus scheduling, rate-limit, auth, timeout and server errors. Also `ErrorSeverity`, `HTTPStatusMapping`, `exception_to_http_response()`, `log_exception()` |
| `__init__.py` | Re-exports |

The server uses `exception_to_http_response()` to turn an engine exception into a structured HTTP error instead of a traceback.
