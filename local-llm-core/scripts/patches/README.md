# scripts/patches/

Pre-fixed copies of files from the upstream **BitNet.cpp** repository. `bitnet_setup.sh` copies them over the corresponding files in the cloned BitNet tree (function `_apply_file_patch`).

| File | Replaces | Why |
|---|---|---|
| `ggml-bitnet-mad.cpp` | `ggml-bitnet-mad.cpp` | `const int8_t*` fix required by clang 18+; always applied |
| `codegen_tl2.py` | `codegen_tl2.py` | Adds the shape entry for the BitNet-b1.58-2B-4T model; always applied (needed for kernel generation) |
| `convert-hf-to-gguf-bitnet.py` | `convert-hf-to-gguf-bitnet.py` | Tokenizer and packed-weight conversion fix; applied with `--apply-patch` |

When BitNet.cpp upstream changes, re-diff these files before updating them: they are full-file replacements, not diffs.
