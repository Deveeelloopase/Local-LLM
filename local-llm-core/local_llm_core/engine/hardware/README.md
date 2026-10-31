# engine/hardware/

Hardware **profiles**: a named set of engine defaults for a class of machine.

| File | Contents |
|---|---|
| `hardware_profiles.py` | `HardwareProfile` dataclass, `detect_profile()`, `get_profile()`, `list_profile_keys()`, `clear_detection_cache()` |
| `__init__.py` | Re-exports |

Profile keys: `apu_2gb`, `apu_4gb`, `gpu_4gb`, `gpu_6gb`, `gpu_8gb`, `gpu_12gb`, `gpu_16gb`, `gpu_24gb`, `gpu_32gb`, `gpu_48gb`, `gpu_80gb`, `gpu_192gb+`, `cpu_only`.

`detect_profile()` uses the result of `hal.HardwareDetector` to choose a key. Config (`hardware_profiles:`): `force_profile` (skip detection), `detection_thresholds`, `overrides` (per-profile field overrides such as context length, target tokens/s, KV-cache types) and `custom_profiles`.

**Overrides decide the numbers you see at runtime**, not only the defaults in code: different `config.yaml` files can give the same profile different context lengths. Check `GET /v1/runtime/limits` on a running server for the effective values.
