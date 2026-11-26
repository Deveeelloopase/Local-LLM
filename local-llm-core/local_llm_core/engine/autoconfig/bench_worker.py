"""Single-configuration decode benchmark, run in its own process.

The calibrator launches ``python -m local_llm_core.engine.autoconfig.bench_worker``
once per configuration.  Running each measurement in a fresh process gives
three guarantees the in-process approach cannot:

* a configuration that aborts inside the native library (out of device memory,
  unsupported quantisation) cannot take the server down with it,
* memory is returned to the system between configurations, so every
  measurement starts from the same state,
* a hard timeout can be enforced from outside.

The worker reads a JSON *spec* from ``--spec`` (a JSON string or ``@file``),
prints diagnostics to stderr and exactly one ``RESULT {...}`` line to stdout.

Measured values
---------------
``decode_tps``   tokens/second after the first token, median of ``repeats``
                 runs, after a discarded warm-up run.
``prefill_tps``  tokens/second for a prompt of up to ``prefill_tokens`` tokens
                 on a warm model (time to first token minus one decode step).
``backend``      accelerator backend reported by the native library at load.
``offloaded``    "<layers on device>/<total layers>" as reported at load.

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict

__version__ = "1.0.0"

PROMPT = "Write a short paragraph about why the sky is blue."
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets behind "
    "the quiet hills and the river keeps flowing toward the sea. "
) * 40

#: ggml tensor type ids for the supported KV-cache types.
KV_GGML = {"f16": 1, "q8_0": 8, "q4_0": 2}


@contextmanager
def _capture_native_log():
    """Redirect file descriptor 2 so the native library's output is readable."""
    tf = tempfile.TemporaryFile(mode="w+b")
    sys.stderr.flush()
    saved = os.dup(2)
    os.dup2(tf.fileno(), 2)
    box = {"text": ""}
    try:
        yield box
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        tf.seek(0)
        box["text"] = tf.read().decode("utf-8", "replace")
        tf.close()


def detect_backend(log_text: str) -> str:
    """Name the compute backend from the native load log."""
    if "ggml_vulkan" in log_text:
        return "vulkan"
    if re.search(r"ggml_cuda_init|ROCm|HIP", log_text):
        return "hip" if re.search(r"ROCm|HIP", log_text) else "cuda"
    if "CUDA" in log_text:
        return "cuda"
    if "Metal" in log_text:
        return "metal"
    return "cpu"


def effective_gpu_layers(planned: int, n_layer: int) -> int:
    """Layer count handed to the library.

    The output matrix counts as layer ``n_layer + 1``; asking for exactly
    ``n_layer`` leaves it on the host, so a full offload requests one more.
    """
    if planned > 0 and n_layer > 0 and planned >= n_layer:
        return planned + 1
    return planned


def measure(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Load the model once and measure it.  Raises on any failure."""
    from llama_cpp import Llama

    path = str(spec["model_path"])
    ctx = int(spec.get("ctx", 4096))
    threads = int(spec.get("threads", 4))
    kv = str(spec.get("kv", "f16")).lower()
    n_tokens = int(spec.get("tokens", 32))
    repeats = int(spec.get("repeats", 2))
    kw: Dict[str, Any] = dict(
        model_path=path, n_ctx=ctx,
        n_gpu_layers=effective_gpu_layers(int(spec.get("ngl", 0)), int(spec.get("n_layer", 0))),
        n_threads=threads, n_threads_batch=int(spec.get("threads_batch", threads)),
        flash_attn=bool(spec.get("flash", False)),
        n_batch=int(spec.get("n_batch", 512)), n_ubatch=int(spec.get("n_ubatch", 512)),
        use_mmap=True, use_mlock=False, verbose=True,
    )
    if kv != "f16":
        kw["type_k"] = kw["type_v"] = KV_GGML[kv]

    t_load = time.perf_counter()
    with _capture_native_log() as cap:
        llm = Llama(**kw)
        load_s = time.perf_counter() - t_load
        try:
            toks = llm.tokenize(PROMPT.encode(), add_bos=False, special=True)
            for _ in llm.generate(toks, temp=0.0, reset=True):       # warm-up, excluded
                break
            runs = []
            for _ in range(repeats):
                llm.reset()
                first, count = None, 0
                for _tok in llm.generate(toks, temp=0.0, reset=True):
                    now = time.perf_counter()
                    first = first or now
                    count += 1
                    if count >= n_tokens:
                        break
                if count > 1 and first is not None:
                    runs.append((count - 1) / max(time.perf_counter() - first, 1e-6))
            if not runs:
                raise RuntimeError("model produced fewer than two tokens")
            decode = statistics.median(runs)

            long_toks = llm.tokenize(LONG_TEXT.encode(), add_bos=False, special=True)
            long_toks = long_toks[: max(16, min(int(spec.get("prefill_tokens", 300)), ctx // 2))]
            llm.reset()
            t0 = time.perf_counter()
            for _ in llm.generate(long_toks, temp=0.0, reset=True):
                break
            ttft = time.perf_counter() - t0
            prefill = len(long_toks) / max(ttft - 1.0 / max(decode, 1e-6), 1e-3)
        finally:
            del llm
            gc.collect()

    log_text = cap["text"]
    off = re.search(r"offloaded (\d+)/(\d+) layers", log_text)
    return {
        "decode_tps": float(decode),
        "prefill_tps": float(prefill),
        "backend": detect_backend(log_text),
        "offloaded": f"{off[1]}/{off[2]}" if off else "",
        "load_s": round(load_s, 2),
        "runs": [round(x, 3) for x in runs],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure one decode configuration.")
    ap.add_argument("--spec", required=True, help="JSON string or @path to a JSON file")
    a = ap.parse_args(argv)
    raw = a.spec
    if raw.startswith("@"):
        with open(raw[1:], "r", encoding="utf-8") as fh:
            raw = fh.read()
    spec = json.loads(raw)
    try:
        out = measure(spec)
        out["ok"] = True
    except Exception as exc:                                    # reported, never raised
        out = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("RESULT " + json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0 if out.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
