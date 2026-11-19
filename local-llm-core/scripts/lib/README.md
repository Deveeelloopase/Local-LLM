# scripts/lib/

Helper tools shipped with the setup scripts. They are standalone: `install.sh` makes the `.sh` files executable but does not source them.

| File | Name | Purpose |
|---|---|---|
| `atlas_probe.sh` | ATLAS | Hardware fingerprint: CPU vendor/flags/cores, GPU backend, RAM, disk, OS/arch. Prints shell variables and can write JSON |
| `forge_cache.sh` | FORGE | sha256-keyed build cache, so an unchanged BitNet.cpp build is not redone |
| `guardian_check.sh` | GUARDIAN | Pre-build checklist (write access, download tool, git, cmake ≥ 3.14, …) with pass/fail output |
| `spectre_env.sh` | SPECTRE | Resolves conflicts between PyPI `gguf` and BitNet.cpp's bundled `gguf-py` |
| `prism_detect.py` | PRISM | Classifies a model directory/GGUF (standard vs BitNet packed/bf16) using only the standard library |
| `gguf_tl_shim.py` | – | Shim that adds TL1/TL2 tensor types to `gguf`; installed into the virtualenv's `site-packages` by `install.sh` |
| `__init__.py` | – | Makes the folder a package (empty) |
