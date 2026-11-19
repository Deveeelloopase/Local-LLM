"""Decode throughput sweep for llama-cpp-python, with bench persistence.

Usage:
    python bench_decode.py /path/to/model.gguf --ctx 4096 --tokens 48

Loads the model once per configuration, generates a fixed number of tokens
from a short prompt, and prints decode tok/s (time after the first token)
and prefill tok/s. Stop the llm-server service first so the benchmark has
the machine to itself.

On completion, the best configuration (highest `decode_tps`) is persisted
via `local_llm_core.engine.hardware.hardware_profiles.record_bench_measurement()`
to `BENCH_FILE_PATH` (default: `$LLM_STATE_DIR/bench_decode.json`, i.e.
`/var/lib/local-llm/bench_decode.json`).

On the next server start, `hardware_profiles.compute_profile()` calls
`find_bench_for_model(model_path)` and, on a match, replaces the roofline
target_tps with this measured value (`target_tps_source="bench_file"`).
Matching is by exact path, then by basename, so a repo moved from
`~/models` to `/opt/models` still hits.

Persistence is best-effort:
  • If `local_llm_core` is not importable, the sweep still runs, still
    prints the best config, and warns that persistence was skipped.
  • If the write fails (permissions, full disk), a warning is printed
    and the process exits 1 so scripts can detect it.

Skip persistence with `--no-persist`. Override the file with `$LLM_BENCH_FILE`.

Not run in the audit sandbox (no model or llama.cpp build there): run it
on your box.

Example:
    # Full sweep (12 configs × 2 repeats — ~10-15 minutes on this box)
    python bench_decode.py ~/models/Qwen3-8B-Q4_K_M.gguf

    # Quick single-config test (persists that one result)
    python bench_decode.py ~/models/Qwen3-8B-Q4_K_M.gguf \
        --layers 36 --threads 4 --flash 1

    # Sweep without persisting (e.g. exploring configs)
    python bench_decode.py ~/models/Qwen3-8B-Q4_K_M.gguf --no-persist
"""
import argparse
import gc
import itertools
import re
import statistics
import time
from pathlib import Path

from llama_cpp import Llama


PROMPT = "Write a short paragraph about why the sky is blue."

# Match common GGUF quant suffixes at end of the filename:
#   `-Q4_K_M.gguf`, `.Q8_0.gguf`, `_q4_k_m.gguf`, `.IQ2_XS.gguf`, `.F16.gguf`
_QUANT_RE = re.compile(
    r"(?i)(?:[._-])"
    r"(q[2-8]_(?:k_[sml]|k|0|1)|f16|f32|bf16|iq[1-4]_[a-z0-9]+)"
    r"(?:\.gguf)?$"
)


def _parse_quantization(model_path: str) -> str:
    """Best-effort quantization parser from the model filename.

    Returns e.g. "Q4_K_M", "Q8_0", "IQ2_XS", "" — empty string when the
    filename does not advertise a quantization (rather than guessing).
    """
    m = _QUANT_RE.search(Path(model_path).name)
    return m.group(1).upper() if m else ""


def _model_id_from_path(model_path: str) -> str:
    """Strip the quant suffix from the GGUF filename stem.

    Examples:
        Qwen3-8B-Q4_K_M.gguf       → Qwen3-8B
        qwen2.5-7b-instruct_q8_0.gguf → qwen2.5-7b-instruct
        mystery.gguf                → mystery
    """
    stem = Path(model_path).stem
    m = _QUANT_RE.search(Path(model_path).name)
    if m:
        q = m.group(1)
        for sep in ("-", "_", "."):
            suffix = sep + q
            if stem.lower().endswith(suffix.lower()):
                return stem[: -len(suffix)]
    return stem


def run(path, ctx, ngl, threads, fa, n_tokens, repeats):
    """Load once, measure prefill + decode tps over `repeats` runs.

    Returns `(median_prefill_tps, median_decode_tps)`. Raises on model
    load failure or generation error.
    """
    llm = Llama(
        model_path=path,
        n_ctx=ctx,
        n_gpu_layers=ngl,
        n_threads=threads,
        n_threads_batch=threads,
        flash_attn=fa,
        n_batch=512,
        n_ubatch=512,
        use_mmap=True,
        verbose=False,
    )
    toks = llm.tokenize(PROMPT.encode(), add_bos=False, special=True)
    prefill, decode = [], []
    for _ in range(repeats):
        llm.reset()
        t0 = time.perf_counter()
        first = None
        count = 0
        for _tok in llm.generate(toks, temp=0.0, reset=True):
            now = time.perf_counter()
            if first is None:
                first = now
            count += 1
            if count >= n_tokens:
                break
        prefill.append(len(toks) / max(first - t0, 1e-6))
        decode.append((count - 1) / max(time.perf_counter() - first, 1e-6))
    del llm
    gc.collect()
    return statistics.median(prefill), statistics.median(decode)


def _persist_best(model_path: str, best: dict, ctx: int) -> int:
    """Persist the winning config via record_bench_measurement.

    Returns an exit code:
      0 — persisted successfully, OR persistence was unavailable but the
          bench itself succeeded (never masquerade as a failure).
      1 — record_bench_measurement returned False (write actually failed).
    """
    try:
        from local_llm_core.engine.hardware.hardware_profiles import (
            record_bench_measurement,
            BENCH_FILE_PATH,
        )
    except ImportError as exc:
        print(
            f"\n  ⚠  local_llm_core not importable ({exc}); skipping persistence.\n"
            f"     Install it (`pip install -e local-llm-core`) to make the\n"
            f"     measured value available to compute_profile() on next start.",
            flush=True,
        )
        return 0

    ok = record_bench_measurement(
        model_path=model_path,
        decode_tps=best["decode_tps"],
        prefill_tps=best["prefill_tps"],
        model_id=_model_id_from_path(model_path),
        quantization=_parse_quantization(model_path),
        gpu_layers=best["ngl"],
        threads=best["threads"],
        flash_attn=bool(best["flash"]),
        ctx=ctx,
    )

    if ok:
        print(f"\n  ✓ Persisted to {BENCH_FILE_PATH}", flush=True)
        print(
            f"     hardware_profiles.compute_profile() will use\n"
            f"     decode_tps={best['decode_tps']:.2f} as target_tps on "
            f"the next server start.",
            flush=True,
        )
        return 0

    print(
        f"\n  ⚠  Failed to persist to {BENCH_FILE_PATH} "
        f"(see warning in the server log for the reason).",
        flush=True,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Decode-throughput sweep for llama-cpp-python with "
            "bench persistence for local-llm-cloud."
        ),
    )
    ap.add_argument("model", help="Path to a GGUF model file.")
    ap.add_argument("--ctx", type=int, default=4096,
                    help="Context length (default: 4096).")
    ap.add_argument("--tokens", type=int, default=48,
                    help="Tokens per run (default: 48).")
    ap.add_argument("--repeats", type=int, default=2,
                    help="Runs per config (default: 2; median reported).")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 8, 36],
                    help="GPU layer counts to sweep (default: 0 8 36).")
    ap.add_argument("--threads", type=int, nargs="+", default=[3, 4],
                    help="Thread counts to sweep (default: 3 4).")
    ap.add_argument("--flash", type=int, nargs="+", default=[0, 1],
                    help="flash_attn values to sweep, 0/1 (default: 0 1).")
    ap.add_argument(
        "--no-persist", action="store_true",
        help=(
            "Skip writing bench_decode.json. The sweep still runs and "
            "prints the best config; only the persistence step is skipped."
        ),
    )
    a = ap.parse_args()

    # Resolve once. `find_bench_for_model` matches by exact path first,
    # then by basename, so resolving here is safe even if the config.yaml
    # `path` field uses a different absolute form — the basename always
    # lines up for the same file.
    model_path = str(Path(a.model).resolve())

    results: list[dict] = []
    print(
        f"{'layers':>6} {'threads':>7} {'flash':>5} "
        f"{'prefill t/s':>12} {'decode t/s':>11}",
        flush=True,
    )

    for ngl, thr, fa in itertools.product(a.layers, a.threads, a.flash):
        try:
            p, d = run(model_path, a.ctx, ngl, thr, bool(fa), a.tokens, a.repeats)
            print(
                f"{ngl:>6} {thr:>7} {fa:>5} {p:>12.1f} {d:>11.2f}",
                flush=True,
            )
            results.append({
                "ngl": int(ngl),
                "threads": int(thr),
                "flash": int(fa),
                "prefill_tps": float(p),
                "decode_tps": float(d),
            })
        except Exception as exc:
            print(
                f"{ngl:>6} {thr:>7} {fa:>5}  failed: {exc}",
                flush=True,
            )

    if not results:
        print(
            "\n  ✗ Every configuration failed. Nothing to persist.\n"
            "    Check that the model path is correct and that the "
            "llama-cpp-python build supports the model's quantization.",
            flush=True,
        )
        return 1

    best = max(results, key=lambda r: r["decode_tps"])
    print(
        f"\n  Best: layers={best['ngl']} threads={best['threads']} "
        f"flash={best['flash']} "
        f"decode={best['decode_tps']:.2f} tok/s "
        f"prefill={best['prefill_tps']:.1f} tok/s",
        flush=True,
    )

    if a.no_persist:
        print("  (--no-persist: bench_decode.json not written)", flush=True)
        return 0

    return _persist_best(model_path, best, a.ctx)


if __name__ == "__main__":
    import sys
    sys.exit(main())
