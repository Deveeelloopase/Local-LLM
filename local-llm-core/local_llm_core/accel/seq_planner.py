"""
seq_planner.py — derive ``inference.continuous_batching.n_seq_max`` from the
hardware and the model instead of pinning it in config.

Why this exists
---------------
``n_seq_max`` is the number of requests one llama.cpp context decodes in a
single forward pass.  With ``n_seq_max: 1`` the single-process engine serves
exactly one request at a time, so a second client (CLI + Web UI) waits for the
first one to finish.  On memory-bandwidth-bound hardware batching is *cheaper*
than running a second worker process: the weights are streamed once per
forward pass for all sequences, not once per process.

With ``kv_unified: true`` the KV cache is one shared pool of ``n_ctx`` tokens,
so extra sequence slots cost only small per-sequence buffers, not a second KV
cache.  What actually limits the slot count is (a) how many tokens of context
each sequence needs to be useful and (b) RAM headroom.

Pure functions, no I/O except ``resolve_n_seq_max_in_cfg`` (one ``stat`` and
one ``psutil`` call).  Never raises into the caller: on any problem it returns
1, i.e. today's behaviour.

version = 1.0.0
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

AUTO_TOKENS = ("auto", "AUTO", "Auto", None, 0, "0", "")

# Tuning constants (all overridable from config.yaml, see resolve_n_seq_max_in_cfg)
DEFAULT_MIN_CTX_PER_SEQ = 1024     # below this a chat turn does not fit
DEFAULT_PER_SEQ_OVERHEAD_MB = 64   # output buffers, sampler state, python-side state
DEFAULT_HARD_CAP = 8               # practical llama.cpp decode-batching ceiling
DEFAULT_OS_RESERVE_MB = 1024


def plan_n_seq_max(
    *,
    n_ctx: int,
    weights_mb: float,
    kv_mb: float,
    host_available_mb: float,
    cpu_physical_cores: int,
    min_ctx_per_seq: int = DEFAULT_MIN_CTX_PER_SEQ,
    per_seq_overhead_mb: float = DEFAULT_PER_SEQ_OVERHEAD_MB,
    os_reserve_mb: float = DEFAULT_OS_RESERVE_MB,
    hard_cap: int = DEFAULT_HARD_CAP,
) -> int:
    """Return the number of concurrent sequences this machine can host (>= 1).

    The result is the minimum of three independent ceilings:

    * context ceiling  – ``n_ctx // min_ctx_per_seq`` (a slot that cannot hold a
      useful conversation is not a slot);
    * memory ceiling   – RAM left after weights, the shared KV pool and the OS
      reserve, divided by the per-sequence overhead;
    * CPU ceiling      – two sequences per physical core (decode threads are
      shared, more slots than that only adds latency);

    and is clamped to ``[1, hard_cap]``.
    """
    try:
        n_ctx = int(n_ctx)
        if n_ctx <= 0:
            return 1
        by_ctx = max(1, n_ctx // max(1, int(min_ctx_per_seq)))

        headroom = (
            float(host_available_mb) - float(weights_mb) - float(kv_mb)
            - float(os_reserve_mb)
        )
        if headroom <= 0:
            by_mem = 1
        else:
            by_mem = 1 + int(headroom // max(1.0, float(per_seq_overhead_mb)))

        by_cpu = max(1, 2 * int(cpu_physical_cores or 1))
        return max(1, min(by_ctx, by_mem, by_cpu, max(1, int(hard_cap))))
    except Exception:           # defensive: planning must never take the server down
        return 1


def kv_mb_estimate(n_ctx: int, kv_bytes_per_token: float) -> float:
    return max(0.0, float(n_ctx) * float(kv_bytes_per_token)) / 1_048_576.0


def resolve_n_seq_max_in_cfg(
    cfg: Dict[str, Any],
    *,
    ram_total_mb: Optional[int] = None,
    cpu_physical: Optional[int] = None,
    kv_bytes_per_token: float = 78_336.0,
) -> Optional[int]:
    """Replace ``continuous_batching.n_seq_max: auto`` with a concrete int.

    Must run BEFORE ``ModelLoader`` / ``InferenceEngine`` are constructed so the
    load-time context and the batched engine agree on the same number (a
    mismatch forces the memory-hungry context rebuild this whole mechanism
    exists to avoid).  An explicit integer in config is left untouched.

    Returns the resolved value, or ``None`` if nothing was changed.
    """
    inf = cfg.setdefault("inference", {}) if isinstance(cfg, dict) else {}
    cb = inf.get("continuous_batching")
    if not isinstance(cb, dict) or not bool(cb.get("enabled", False)):
        return None
    raw = cb.get("n_seq_max", 1)
    if raw not in AUTO_TOKENS:
        return None                      # operator pinned it: respect that

    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / 1_048_576.0
        if ram_total_mb:
            avail_mb = min(avail_mb, float(ram_total_mb))
        cores = int(cpu_physical or psutil.cpu_count(logical=False) or 1)
    except Exception:
        cb["n_seq_max"] = 1
        return 1

    weights_mb = 0.0
    n_ctx = int(cb.get("planning_ctx", 0) or 0)
    try:
        for p in (cfg.get("models", {}) or {}).get("profiles", []) or []:
            if p.get("autoload") and "draft" not in (p.get("tags") or []):
                path = p.get("path") or ""
                if path and os.path.isfile(path):
                    weights_mb = os.stat(path).st_size / 1_048_576.0
                n_ctx = n_ctx or int(p.get("context_length") or 0)
                break
    except Exception:
        pass
    n_ctx = n_ctx or 4096

    n = plan_n_seq_max(
        n_ctx=n_ctx,
        weights_mb=weights_mb,
        kv_mb=kv_mb_estimate(n_ctx, kv_bytes_per_token),
        host_available_mb=avail_mb,
        cpu_physical_cores=cores,
        min_ctx_per_seq=int(cb.get("min_ctx_per_seq", DEFAULT_MIN_CTX_PER_SEQ)),
        per_seq_overhead_mb=float(cb.get("per_seq_overhead_mb", DEFAULT_PER_SEQ_OVERHEAD_MB)),
        os_reserve_mb=float(cb.get("os_reserve_mb", DEFAULT_OS_RESERVE_MB)),
        hard_cap=int(cb.get("adaptive_max_n_seq_max") or DEFAULT_HARD_CAP),
    )
    cb["n_seq_max"] = n
    log.info(
        "continuous_batching.n_seq_max=auto → %d  (n_ctx=%d weights=%.0fMB "
        "avail=%.0fMB cores=%d)", n, n_ctx, weights_mb, avail_mb, cores,
    )
    return n
