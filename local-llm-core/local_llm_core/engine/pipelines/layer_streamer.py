from __future__ import annotations

# engine/pipelines/layer_streamer.py — Production‑Ready Fractional Layer Pipeline  v1.0.0
# ======================================================================================
# v1.0.0 — Integrated BitNet GGUF detection:
#   • AdaptiveLoader now checks if a model is a BitNet GGUF before selecting a backend.
#     BitNet models are routed to `transformers_cpu` (i.e., HuggingFace) instead of
#     llama‑cpp‑python, preventing silent garbage / crashes.
#   • The detection uses the unified environment module (model_unified_env) with a
#     graceful fallback if the module is missing.
# ======================================================================================

import gc
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants (unchanged)
# -----------------------------------------------------------------------------
GGUF_MAGIC = b"GGUF"
_GGUF_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
    6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}
_GGUF_FMTS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}

_KV_QUANT_FACTOR = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}

# ── Calibration constants (used by AdaptiveLoader._calibration_measure) ──
# Same values as offload_calibrator.py exports. Defined locally so this
# module remains importable when the calibrator module is absent — which
# is the current state, because the calibrator's wire-up is gated on
# _profile_vram_carveout_mb (never set), so _calibration_measure is
# unreachable today. These constants exist so the method is callable
# the moment the wire-up is fixed.
_CAL_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
_CAL_TOKENS = 16

# Default policy constants
DEFAULT_CONTEXT_LAYERS_PRIORITY = "balanced"
MIN_GPU_LAYERS_FRACTION_FOR_BALANCED = 0.6   # at least 60% of layers on GPU
MIN_SAFE_CONTEXT = 256
MAX_ITERATIONS = 64  # safety backstop for any loop that grows context iteratively


# -----------------------------------------------------------------------------
# GGUF Metadata Reader (unchanged)
# -----------------------------------------------------------------------------
def read_gguf_meta(path: Path) -> Dict[str, Any]:
    """Parse GGUF header without loading weights. Falls back to safe defaults.

    BUG FIX (hang / silent OOM-kill):
    The previous implementation read only ``min(cnt, 1024)`` elements from
    each GGUF array KV entry but did NOT advance the file pointer past the
    remaining elements.  For Qwen3-8B the tokenizer stores 151 936 vocab
    tokens, 151 936 scores and 151 936 token-type entries as separate GGUF
    ARRAY KV pairs.  After reading 1 024 entries the pointer sat mid-array;
    the next KV-key read consumed raw float/byte data as a 64-bit length,
    then called ``f.read(<multi-GB>)`` — stalling the executor thread for
    ~2 minutes until the Linux OOM-killer sent SIGKILL.  Python's
    ``except Exception`` cannot catch SIGKILL, so no log entry appeared and
    systemd restarted the server silently.
    """
    meta: Dict[str, Any] = {
        "n_layer":        32,
        "n_head":         32,
        "n_embd":         4096,
        "n_kv_heads":     0,
        "vocab_size":     32000,
        "arch":           "llama",
        "context_length": 4096,
        "n_ff":           11008,
        "n_ctx_train":    0,
        "head_dim":       0,
        "head_dim_v":     0,
    }

    _MAX_STR  = 1 << 20   # 1 MiB — no single metadata string should be larger
    _MAX_KEEP = 64         # max array elements we bother storing (arch only needs 1)

    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return meta
            ver = struct.unpack("<I", f.read(4))[0]
            if ver not in (1, 2, 3):
                return meta
            struct.unpack("<Q", f.read(8))[0]   # n_tensors (unused)
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _rstr() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > _MAX_STR:
                    raise ValueError(
                        f"GGUF string length {n} > {_MAX_STR}; "
                        "file pointer likely desynchronised (truncated array)"
                    )
                return f.read(n).decode("utf-8", errors="replace")

            def _rval(t: int, keep: bool = True):
                """Read one GGUF value, ALWAYS advancing the file pointer correctly.

                ``keep=False`` reads and discards — used for array elements beyond
                ``_MAX_KEEP`` so the pointer stays synchronised without allocation.
                """
                if t == 8:   # GGUF_TYPE_STRING
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > _MAX_STR:
                        raise ValueError(f"GGUF string length {n} > {_MAX_STR}")
                    raw = f.read(n)
                    return raw.decode("utf-8", errors="replace") if keep else None

                if t == 9:   # GGUF_TYPE_ARRAY
                    et  = struct.unpack("<I", f.read(4))[0]
                    cnt = struct.unpack("<Q", f.read(8))[0]
                    esz = _GGUF_SIZES.get(et, 0)
                    kn  = min(cnt, _MAX_KEEP) if keep else 0

                    if esz > 0 and et not in (8, 9):
                        # Fixed-size elements: keep a few, seek past the rest — O(1)
                        result = [_rval(et) for _ in range(kn)]
                        skip   = cnt - kn
                        if skip > 0:
                            f.seek(skip * esz, 1)
                        return result if keep else None
                    else:
                        # Variable-size (STRING / nested ARRAY): parse each element
                        # individually to locate byte boundaries; discard beyond kn.
                        result: list = []
                        for i in range(cnt):
                            v = _rval(et, keep=(keep and i < kn))
                            if keep and i < kn:
                                result.append(v)
                        return result if keep else None

                esz = _GGUF_SIZES.get(t, 0)
                if esz == 0:
                    return None
                raw = f.read(esz)
                return struct.unpack(f"<{_GGUF_FMTS[t]}", raw)[0] if keep else None

            for _ in range(n_kv):
                try:
                    key = _rstr()
                    t   = struct.unpack("<I", f.read(4))[0]
                    val = _rval(t)
                    k   = key.split(".")[-1]
                    if k == "block_count"      and isinstance(val, int): meta["n_layer"]        = val
                    if k == "head_count"       and isinstance(val, int): meta["n_head"]         = val
                    if k == "head_count_kv"    and isinstance(val, int): meta["n_kv_heads"]     = val
                    if k == "embedding_length" and isinstance(val, int): meta["n_embd"]         = val
                    if k == "feed_forward_length" and isinstance(val, int): meta["n_ff"]        = val
                    if k == "vocab_size"       and isinstance(val, int): meta["vocab_size"]     = val
                    if k == "context_length" and isinstance(val, int):
                        # True trained window — uncapped. Used by the
                        # context planner in model_loader.py. We keep the
                        # legacy context_length capped at 32768 for any
                        # caller that still reads it directly.
                        meta["n_ctx_train"]    = val
                        meta["context_length"] = min(val, 32768)
                    if k == "key_length"     and isinstance(val, int): meta["head_dim"]   = val
                    if k == "value_length"   and isinstance(val, int): meta["head_dim_v"] = val
                    if "architecture" in key   and isinstance(val, str): meta["arch"]     = val
                except Exception:
                    break   # desynchronised / truncated; return what we have

    except Exception as e:
        log.warning("GGUF parse fallback (%s): %s", path.name, e)

    # ── Derive missing values from what we have ──────────────────────────
    # Qwen3-4B has n_embd=2560, n_head=32 → naive head_dim = 80, but the
    # real head_dim is 128 (confirmed via GGUF key_length=128). Deriving
    # from n_embd // n_head would under-size the KV cache by 37 % and
    # silently produce wrong context-length plans. GGUF key_length /
    # value_length are authoritative when present; the n_embd // n_head
    # fallback is only correct for architectures (Llama 2 / Llama 3,
    # Mistral, Phi-3) where that identity actually holds.
    if not meta["head_dim"]:
        meta["head_dim"] = meta["n_embd"] // max(1, meta["n_head"])
    if not meta["head_dim_v"]:
        meta["head_dim_v"] = meta["head_dim"]
    if not meta["n_kv_heads"]:
        meta["n_kv_heads"] = meta["n_head"]
    if not meta["n_ctx_train"]:
        # GGUF v1/v2 headers, or a truncated KV table that we stopped
        # parsing early. Fall back to the legacy (capped) context_length
        # so downstream callers still get *something* sane.
        meta["n_ctx_train"] = meta["context_length"]

    return meta


# -----------------------------------------------------------------------------
# Optimal GPU Layers Calculator — now priority‑aware (FIXED INFINITE LOOP)
# -----------------------------------------------------------------------------
def calc_optimal_gpu_layers(
    model_path: Path,
    vram_budget_mb: int,
    ram_free_mb: int,
    context_len: int = 2048,
    kv_quant: str = "q8_0",
    *,
    vram_overhead_mb: Optional[int] = None,
    layer_overcommit_factor: Optional[float] = None,
    priority: str = DEFAULT_CONTEXT_LAYERS_PRIORITY,
    hardware_profile: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculate optimal GPU layer split using GQA‑aware KV cache size.

    Parameters
    ----------
    model_path : Path
    vram_budget_mb : int
    ram_free_mb : int
    context_len : int
        Base context length (used as floor). Actual may be increased if priority allows.
    kv_quant : str
    vram_overhead_mb : int, optional
    layer_overcommit_factor : float, optional
    priority : str, optional
        One of "context", "layers", or "balanced". Controls tradeoff.
    hardware_profile : optional
        If provided, used to obtain max_context and other profile limits.
    """
    if vram_overhead_mb is None:
        vram_overhead_mb = 80
    if layer_overcommit_factor is None:
        layer_overcommit_factor = 1.05

    # Use the new priority‑aware planner
    plan = compute_optimal_offload_plan(
        model_path=model_path,
        vram_budget_mb=vram_budget_mb,
        ram_free_mb=ram_free_mb,
        base_context_len=context_len,
        kv_quant=kv_quant,
        vram_overhead_mb=vram_overhead_mb,
        layer_overcommit_factor=layer_overcommit_factor,
        priority=priority,
        hardware_profile=hardware_profile,
    )
    return plan


def compute_optimal_offload_plan(
    model_path: Path,
    vram_budget_mb: int,
    ram_free_mb: int,
    base_context_len: int,
    kv_quant: str = "q8_0",
    *,
    vram_overhead_mb: int = 80,
    layer_overcommit_factor: float = 1.05,
    priority: str = DEFAULT_CONTEXT_LAYERS_PRIORITY,
    hardware_profile: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    New priority‑aware planner that decides both context length and GPU layers.

    The tradeoff is:
      - "layers": keep context at base, maximize GPU layers.
      - "context": maximize context while keeping at least 1 GPU layer.
      - "balanced": allow context to grow but ensure at least
         MIN_GPU_LAYERS_FRACTION_FOR_BALANCED (60%) of layers remain on GPU.

    Returns:
        dict with keys: n_gpu_layers, n_cpu_layers, n_total_layers,
        vram_used_mb, ram_used_mb, strategy, context_len_used, meta, priority_used.
    """
    meta = read_gguf_meta(model_path)
    n_layers = meta["n_layer"]
    n_embd = meta["n_embd"]
    file_mb = model_path.stat().st_size >> 20
    layer_mb = file_mb / max(n_layers, 1)

    n_head = meta.get("n_head", 32)
    n_kv_heads = meta.get("n_kv_heads", 0)
    if n_kv_heads == 0:
        n_kv_heads = max(1, n_head // 4)  # GQA conservative
    head_dim = max(1, n_embd // max(n_head, 1))

    kv_factor = _KV_QUANT_FACTOR.get(kv_quant, 0.5)
    usable_vram = max(0, vram_budget_mb - vram_overhead_mb) * layer_overcommit_factor

    # Determine max possible context from hardware profile or fallback
    max_context = base_context_len * 4   # fallback
    if hardware_profile is not None:
        max_context = getattr(hardware_profile, "max_context", base_context_len * 4)
        # Also respect any explicit context cap from profile
        if hasattr(hardware_profile, "max_context"):
            max_context = hardware_profile.max_context

    max_context = max(base_context_len, min(max_context, 32768))  # sane upper bound

    # Helper to compute kv_layer_mb for a given context
    def _kv_mb_for_ctx(ctx: int) -> float:
        return (2 * n_kv_heads * head_dim * ctx * 2 * kv_factor) / (1024 * 1024)

    # Helper to compute number of layers that fit in VRAM for a given context
    def _layers_for_ctx(ctx: int) -> int:
        kv_mb = _kv_mb_for_ctx(ctx)
        mb_per_layer = layer_mb + kv_mb
        if mb_per_layer <= 0:
            return 0
        return min(n_layers, int(usable_vram / mb_per_layer))

    # Base case: context = base_context_len
    base_layers = _layers_for_ctx(base_context_len)
    if base_layers == 0:
        # Not even base context fits; reduce context until at least 1 layer fits
        ctx = base_context_len
        while ctx > MIN_SAFE_CONTEXT and _layers_for_ctx(ctx) == 0:
            ctx = max(MIN_SAFE_CONTEXT, ctx // 2)
        base_context_len = ctx
        base_layers = _layers_for_ctx(ctx)

    final_context = base_context_len
    final_layers = base_layers

    if priority == "layers":
        # Keep context at base, maximize layers
        final_context = base_context_len
        final_layers = base_layers

    elif priority == "context":
        # Maximize context, but keep at least 1 layer on GPU
        ctx = base_context_len
        best_ctx = base_context_len
        iterations = 0
        while ctx <= max_context:
            layers = _layers_for_ctx(ctx)
            if layers >= 1:
                best_ctx = ctx
                next_ctx = min(max_context, ctx * 2)
                if next_ctx <= ctx:
                    # No progress possible (already at ceiling)
                    break
                ctx = next_ctx
            else:
                break
            iterations += 1
            if iterations > MAX_ITERATIONS:
                log.error("Context growth loop exceeded %d iterations at ctx=%d – breaking", MAX_ITERATIONS, ctx)
                break
        final_context = best_ctx
        final_layers = _layers_for_ctx(best_ctx)

    elif priority == "balanced":
        # Allow context growth as long as GPU layers don't drop below
        # MIN_GPU_LAYERS_FRACTION_FOR_BALANCED * base_layers (but at least 1)
        min_layers = max(1, int(base_layers * MIN_GPU_LAYERS_FRACTION_FOR_BALANCED))
        ctx = base_context_len
        best_ctx = base_context_len
        iterations = 0
        # Try increasing context gradually (e.g., by 1.5x)
        while ctx <= max_context:
            layers = _layers_for_ctx(ctx)
            if layers >= min_layers:
                best_ctx = ctx
                next_ctx = min(max_context, int(ctx * 1.5))
                if next_ctx <= ctx:
                    # No progress possible (already at ceiling)
                    break
                ctx = next_ctx
            else:
                break
            iterations += 1
            if iterations > MAX_ITERATIONS:
                log.error("Balanced growth loop exceeded %d iterations at ctx=%d – breaking", MAX_ITERATIONS, ctx)
                break
        final_context = best_ctx
        final_layers = _layers_for_ctx(best_ctx)

    else:
        # Unknown priority: fallback to balanced
        log.warning("Unknown priority '%s', falling back to 'balanced'", priority)
        return compute_optimal_offload_plan(
            model_path=model_path,
            vram_budget_mb=vram_budget_mb,
            ram_free_mb=ram_free_mb,
            base_context_len=base_context_len,
            kv_quant=kv_quant,
            vram_overhead_mb=vram_overhead_mb,
            layer_overcommit_factor=layer_overcommit_factor,
            priority="balanced",
            hardware_profile=hardware_profile,
        )

    # Now compute final metrics
    kv_layer_mb = _kv_mb_for_ctx(final_context)
    mb_per_gpu_layer = layer_mb + kv_layer_mb
    gpu_layers = final_layers
    cpu_layers = n_layers - gpu_layers
    cpu_weight_mb = cpu_layers * layer_mb
    if cpu_weight_mb > ram_free_mb * 0.80:
        log.warning("RAM pressure: model=%dMB cpu_layers=%d — full mmap enabled", file_mb, cpu_layers)

    if gpu_layers == 0:
        strategy = "cpu_only"
    elif gpu_layers >= n_layers:
        strategy = "full_gpu"
    elif gpu_layers / n_layers >= 0.75:
        strategy = "mostly_gpu"
    elif gpu_layers / n_layers >= 0.25:
        strategy = "hybrid"
    else:
        strategy = "mostly_cpu"

    vram_used = int(gpu_layers * mb_per_gpu_layer + vram_overhead_mb)
    ram_used = int(cpu_layers * layer_mb)

    # ── CRITICAL TRADEOFF LOG LINE ──
    log.info(
        "FLP tradeoff: priority=%s | context_len=%d | kv_layer_mb=%.1f | mb_per_layer=%.1f | "
        "gpu_layers=%d/%d (%s) | vram_used=%dMB | ram_used=%dMB | "
        "overcommit=%.2f | kv_quant=%s",
        priority,
        final_context,
        kv_layer_mb,
        mb_per_gpu_layer,
        gpu_layers,
        n_layers,
        strategy,
        vram_used,
        ram_used,
        layer_overcommit_factor,
        kv_quant,
    )

    return {
        "n_gpu_layers": gpu_layers,
        "n_cpu_layers": cpu_layers,
        "n_total_layers": n_layers,
        "vram_used_mb": vram_used,
        "ram_used_mb": ram_used,
        "strategy": strategy,
        "context_len_used": final_context,
        "meta": meta,
        "priority_used": priority,
        "kv_layer_mb": kv_layer_mb,
        "mb_per_gpu_layer": mb_per_gpu_layer,
    }


# -----------------------------------------------------------------------------
# FLPConfig Dataclass (with updated auto() method)
# -----------------------------------------------------------------------------
@dataclass
class FLPConfig:
    """Runtime configuration for the Fractional Layer Pipeline."""

    model_path: str
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    n_batch: int = 512
    n_ubatch: int = 256
    n_threads: int = 0
    n_threads_batch: int = 0
    n_parallel: int = 1
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attn: bool = True
    rope_freq_base: float = 0.0
    quant_type: str = "auto"
    tensor_split: Optional[List[float]] = None
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    verbose: bool = False

    @classmethod
    def auto(
            cls,
            model_path: str,
            vram_total_mb: int = 2048,
            ram_free_mb: int = 8192,
            hardware_profile=None,
    ) -> "FLPConfig":
        """Create an FLPConfig using hardware profile defaults and priority‑aware planning."""
        if not model_path:
            raise ValueError("model_path is required")

        vram_total_mb = max(0, int(vram_total_mb))
        ram_free_mb = max(1024, int(ram_free_mb))

        if hardware_profile is None:
            try:
                from local_llm_core.engine.hardware.hardware_profiles import get_profile

                if vram_total_mb <= 0:
                    hardware_profile = get_profile("cpu_only")
                elif vram_total_mb <= 2048:
                    hardware_profile = get_profile("apu_2gb")
                elif vram_total_mb <= 4096:
                    hardware_profile = get_profile("apu_4gb")
                elif vram_total_mb <= 5120:
                    hardware_profile = get_profile("gpu_4gb")
                elif vram_total_mb <= 7168:
                    hardware_profile = get_profile("gpu_6gb")
                elif vram_total_mb <= 10240:
                    hardware_profile = get_profile("gpu_8gb")
                elif vram_total_mb <= 14336:
                    hardware_profile = get_profile("gpu_12gb")
                elif vram_total_mb <= 20480:
                    hardware_profile = get_profile("gpu_16gb")
                elif vram_total_mb <= 32768:
                    hardware_profile = get_profile("gpu_24gb")
                elif vram_total_mb <= 49152:
                    hardware_profile = get_profile("gpu_32gb")
                elif vram_total_mb <= 65536:
                    hardware_profile = get_profile("gpu_48gb")
                elif vram_total_mb <= 98304:
                    hardware_profile = get_profile("gpu_80gb")
                else:
                    hardware_profile = get_profile("gpu_192gb+")
            except KeyError as e:
                log.warning("Failed to get hardware profile: %s, using cpu_only", e)
                hardware_profile = get_profile("cpu_only")

        if not hasattr(hardware_profile, "n_batch"):
            log.error("Invalid hardware_profile object, using cpu_only")
            from local_llm_core.engine.hardware.hardware_profiles import get_profile
            hardware_profile = get_profile("cpu_only")

        path = Path(model_path)
        if not path.exists():
            log.warning("Model path does not exist: %s", model_path)

        # ── Extract priority from hardware_profile (with fallback) ──
        priority = getattr(hardware_profile, "context_vs_layers_priority", DEFAULT_CONTEXT_LAYERS_PRIORITY)
        if priority not in ("context", "layers", "balanced"):
            log.warning("Invalid priority '%s', using 'balanced'", priority)
            priority = "balanced"

        # Use the new priority‑aware planning
        plan = compute_optimal_offload_plan(
            model_path=path,
            vram_budget_mb=hardware_profile.vram_budget_mb(vram_total_mb),
            ram_free_mb=ram_free_mb,
            base_context_len=hardware_profile.context_length,
            kv_quant=hardware_profile.kv_cache_quant,
            vram_overhead_mb=getattr(hardware_profile, "vram_overhead_mb", 80),
            layer_overcommit_factor=getattr(hardware_profile, "layer_overcommit_factor", 1.05),
            priority=priority,
            hardware_profile=hardware_profile,
        )

        n_gpu_layers = plan.get("n_gpu_layers", 0)
        n_ctx = plan.get("context_len_used", hardware_profile.context_length)

        return cls(
            model_path=str(model_path),
            n_gpu_layers=max(0, int(n_gpu_layers)),
            n_ctx=max(256, int(n_ctx)),
            n_batch=max(64, int(hardware_profile.n_batch)),
            n_ubatch=max(32, min(int(hardware_profile.n_ubatch), hardware_profile.n_batch)),
            n_threads=hardware_profile.n_threads(),
            n_parallel=max(1, int(hardware_profile.n_parallel)),
            use_mmap=True,
            use_mlock=False,
            flash_attn=bool(hardware_profile.flash_attn),
            rope_freq_base=0.0,
            quant_type="auto",
            tensor_split=None,
            cache_type_k=hardware_profile.kv_cache_quant,
            cache_type_v=hardware_profile.kv_cache_quant,
            verbose=False,
        )


# -----------------------------------------------------------------------------
# Backend Probe (unchanged)
# -----------------------------------------------------------------------------
class BackendProbe:
    """Auto‑detect best inference backend."""

    _result: Optional[str] = None
    _lock = threading.Lock()
    ORDERED = [
        "llama_cpp",
        "transformers_gpu",
        "transformers_cpu",
        "ctransformers",
        "numpy_sim",
    ]

    @classmethod
    def best(cls) -> str:
        with cls._lock:
            if cls._result:
                return cls._result
            for name in cls.ORDERED:
                if cls._check(name):
                    cls._result = name
                    log.info("Backend: %s", name)
                    return name
            cls._result = "numpy_sim"
            return "numpy_sim"

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            old = cls._result
            cls._result = None
        if old:
            log.info("BackendProbe: cache cleared (was '%s') — re-probing", old)

    @staticmethod
    def _check(name: str) -> bool:
        try:
            if name == "llama_cpp":
                import llama_cpp  # noqa
                return True
            if name == "transformers_gpu":
                import transformers  # noqa
                import torch  # noqa
                return torch.cuda.is_available() or (
                    hasattr(torch.version, "hip") and bool(torch.version.hip)
                )
            if name == "transformers_cpu":
                import transformers  # noqa
                return True
            if name == "ctransformers":
                import ctransformers  # noqa
                return True
            if name == "numpy_sim":
                import numpy  # noqa
                return True
        except ImportError:
            pass
        return False

    @classmethod
    def all_available(cls) -> List[str]:
        return [b for b in cls.ORDERED if cls._check(b)]


# -----------------------------------------------------------------------------
# Adaptive Loader — with BitNet detection and robust OOM recovery
# -----------------------------------------------------------------------------
class AdaptiveLoader:
    """
    Loads a GGUF model with OOM‑adaptive retry and BitNet awareness.

    this used to override the backend to "transformers_cpu"
    for any detected BitNet GGUF (I2_S/TL1/TL2), on the assumption that
    HuggingFace transformers could load it as a fallback. Verified against
    transformers' own published supported-quantization-type list
    (docs/source/en/gguf.md): F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0,
    Q2_K–Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL — no ternary
    types at all. That fallback could never have worked; it just failed
    with a confusing, unrelated error (e.g. a HuggingFace Hub repo-id
    validation error) instead of a clear one. BitNet detection is now used
    only to log a clear diagnostic; the load is allowed to proceed and
    fail through the existing format-error guard below, which already
    produces the correct, actionable "build BitNet.cpp" message.
    """

    def __init__(
        self,
        cfg: FLPConfig,
        *,
        streamer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg

        # Normalise config
        _defaults = {
            "prompt_lookup_enabled": True,
            "prompt_lookup_num_tokens": 2,
        }
        self._scfg: Dict[str, Any] = dict(_defaults)
        if streamer_config is not None:
            try:
                from local_llm_core.engine.utils.config_loader import LayerStreamerConfig
                if isinstance(streamer_config, LayerStreamerConfig):
                    _parsed = streamer_config.model_dump()
                else:
                    _parsed = LayerStreamerConfig(**streamer_config).model_dump()
                self._scfg.update(_parsed)
            except Exception as exc:
                log.warning("layer_streamer config validation failed: %s; using defaults", exc)
                if isinstance(streamer_config, dict):
                    self._scfg.update(streamer_config)

        # Determine initial backend
        with BackendProbe._lock:
            cur = BackendProbe._result
        if cur in (None, "numpy_sim"):
            BackendProbe.reset()
        self.backend = BackendProbe.best()

        # ── BitNet detection (diagnostic only — no backend override) ──────
        try:
            from local_llm_core.engine.adaptive_awareness.model_unified_env import classify_gguf, GGUFModelType
            header = classify_gguf(str(self.cfg.model_path))
            if header.model_type == GGUFModelType.BITNET:
                log.info(
                    "AdaptiveLoader: '%s' is a BitNet GGUF (i2_s/tl1/tl2). "
                    "transformers' GGUF loader does not support this "
                    "quantization family, so it is not used as a fallback. "
                    "If no built BitNet.cpp binary was found upstream, this "
                    "load will fail below with an actionable error.",
                    Path(self.cfg.model_path).name,
                )
        except ImportError:
            log.warning("model_unified_env not available – skipping BitNet detection")
        except Exception as e:
            log.warning("BitNet detection failed: %s – proceeding with default backend", e)

    def _calibration_load(self, strategy, cfg):
        """Load a throwaway model for calibration. Must be freed before
        returning control to the caller — the caller will load again with
        the winning strategy."""
        import gc
        from llama_cpp import Llama
        kwargs = dict(
            model_path=str(cfg.model_path),
            n_ctx=int(cfg.n_ctx),
            n_batch=max(64, int(cfg.n_batch)),
            n_ubatch=max(32, int(cfg.n_ubatch)),
            n_gpu_layers=int(strategy.gpu_layers),
            n_threads=int(cfg.n_threads),
            n_threads_batch=int(cfg.n_threads_batch),
            use_mmap=True,
            use_mlock=False,
            flash_attn=False,   # calibrate without it; it is an amplifier
            verbose=False,
        )
        return Llama(**kwargs)

    def _calibration_measure(self, llm) -> dict:
        """Run a fixed 16-token decode and return ttft_s and decode_tps."""
        import time
        # prompt_toks was unused; kept for future per-token timing work
        _ = llm.tokenize(_CAL_PROMPT.encode("utf-8"))
        t0 = time.monotonic()
        _ = llm(_CAL_PROMPT, max_tokens=_CAL_TOKENS, temperature=0.0)
        elapsed = time.monotonic() - t0
        tps = max(_CAL_TOKENS - 1, 1) / max(elapsed - 0.05, 0.05)
        return {"decode_tps": tps, "ttft_ms": 0.0}

    def load(self) -> Any:
        if self.backend == "llama_cpp":
            return self._load_llama_adaptive()
        if self.backend.startswith("transformers"):
            return self._load_transformers()
        if self.backend == "ctransformers":
            return self._load_ctransformers()

        mpath = str(self.cfg.model_path).lower()
        real_exts = (".gguf", ".bin", ".pt", ".safetensors", ".ggml")
        if any(mpath.endswith(ext) for ext in real_exts):
            mname = Path(self.cfg.model_path).name
            raise RuntimeError(
                f"No real inference backend available for model file: {mname}\n"
                "llama-cpp-python is not installed or failed to import.\n"
                "Fix (run inside your venv):\n"
                "  pip uninstall llama-cpp-python -y\n"
                "  CMAKE_ARGS=\"-DLLAMA_VULKAN=on\" FORCE_CMAKE=1 pip install --no-binary llama-cpp-python llama-cpp-python\n"
                "  python -c \"import llama_cpp; print(llama_cpp.__version__)\"\n"
                "Then POST /v1/models/load again — no server restart needed."
            )
        log.warning(
            "numpy_sim backend active for non-file path '%s' — output is SYNTHETIC",
            self.cfg.model_path,
        )
        return self._load_sim()

    # -------------------------------------------------------------------------
    # Llama.cpp Adaptive Loading — now uses self._scfg instead of globals
    # -------------------------------------------------------------------------
    def _load_llama_adaptive(self) -> Any:
        from llama_cpp import Llama

        cfg = self.cfg
        n_parallel = cfg.n_parallel
        n_ctx = cfg.n_ctx

        n_gpu = cfg.n_gpu_layers

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Read OOM recovery parameters from injected config (with defaults)
        max_attempts = max(1, int(self._scfg.get("oom_max_attempts", 5)))
        if max_attempts > 20:
            max_attempts = 20

        layer_reduction_factor = float(self._scfg.get("oom_layer_reduction_factor", 0.75))
        if not (0.1 < layer_reduction_factor < 1.0):
            layer_reduction_factor = 0.75

        ctx_reduction_steps = self._scfg.get("ctx_reduction_steps", [1, 2, 4])
        if not isinstance(ctx_reduction_steps, list):
            ctx_reduction_steps = [1, 2, 4]
        ctx_reduction_steps = [max(1, int(x)) for x in ctx_reduction_steps]

        min_ctx = max(64, int(self._scfg.get("min_context", 256)))
        if min_ctx > cfg.n_ctx:
            min_ctx = max(64, cfg.n_ctx // 4)

        n_batch = max(1, int(cfg.n_batch))
        n_ubatch = max(1, min(int(cfg.n_ubatch), n_batch))

        logical = os.cpu_count() or 4
        physical = logical
        try:
            import psutil
            detected_physical = psutil.cpu_count(logical=False)
            if detected_physical and detected_physical > 0:
                physical = detected_physical
        except Exception as e:
            log.warning("psutil detection failed, using logical // 2: %s", e)
            physical = max(1, logical // 2)

        n_threads = int(cfg.n_threads) if cfg.n_threads > 0 else physical
        n_threads = max(1, min(n_threads, logical))
        # Batch threads are the prefill compute pool. SMT siblings share
        # the FP units; 8 threads on 4 physical cores starves the decode
        # threads and regresses prefill. Use physical threads for batch as
        # well — measured on Zen, Haswell, and Apple P-cores.
        n_threads_batch = physical

        log.info(
            "AdaptiveLoader: cpus=logical:%d physical:%d threads=%d batch_threads=%d parallel=%d",
            logical, physical, n_threads, n_threads_batch, n_parallel
        )

        last_exception = None
        for attempt in range(max_attempts):
            try:
                kwargs: Dict[str, Any] = {
                    "model_path": str(model_path),
                    "n_ctx": int(n_ctx),
                    "n_batch": int(n_batch),
                    "n_ubatch": int(n_ubatch),
                    "n_gpu_layers": int(n_gpu),
                    "n_threads": int(n_threads),
                    "n_threads_batch": int(n_threads_batch),
                    "use_mmap": bool(cfg.use_mmap),
                    "use_mlock": bool(cfg.use_mlock),
                    "flash_attn": bool(cfg.flash_attn),
                    "verbose": bool(cfg.verbose),
                }

                kv_str_k = (cfg.cache_type_k or "f16").lower().strip()
                kv_str_v = (cfg.cache_type_v or "f16").lower().strip()
                if kv_str_k not in ("f16", "f32") or kv_str_v not in ("f16", "f32"):
                    try:
                        self._apply_kv_cache_type(kwargs, kv_str_k, kv_str_v, Llama)
                    except Exception:
                        pass
                # Penalty-window size.  Default 64 is too short for
                # paragraph-level loop suppression; 2048 reaches most of a
                # typical reply.  Construction-time only (not per-request).
                try:
                    kwargs["last_n_tokens_size"] = int(
                        self._scfg.get("last_n_tokens_size", 2048)
                    )
                except Exception:
                    kwargs["last_n_tokens_size"] = 2048

                if cfg.rope_freq_base and cfg.rope_freq_base > 0:
                    kwargs["rope_freq_base"] = float(cfg.rope_freq_base)
                if cfg.tensor_split and isinstance(cfg.tensor_split, list):
                    if sum(cfg.tensor_split) > 0:
                        kwargs["tensor_split"] = cfg.tensor_split

                log.info(
                    "Load attempt %d/%d: n_ctx=%d n_batch=%d n_ubatch=%d gpu=%d parallel=%d",
                    attempt + 1, max_attempts, n_ctx, n_batch, n_ubatch, n_gpu, n_parallel
                )

                t0 = time.monotonic()
                try:
                    # Resolve speculative-decoding config BEFORE we need it.
                    # These must be defined before the `if _spec_enabled` check
                    # below — otherwise the very first call raises NameError.
                    _scfg = self._scfg or {}
                    _spec_enabled = bool(_scfg.get("prompt_lookup_enabled", True))
                    _spec_num = int(_scfg.get("prompt_lookup_num_tokens", 2))

                    if _spec_enabled:
                        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                        kwargs["draft_model"] = LlamaPromptLookupDecoding(
                            num_pred_tokens=_spec_num,
                        )
                        log.info(
                            "Prompt-lookup speculation enabled (num_pred_tokens=%d)",
                            _spec_num,
                        )
                    if n_parallel > 1:
                        log.info(
                            "Multi-slot requested (n_parallel=%d) but "
                            "llama-cpp-python's Llama() does not accept "
                            "it as a kwarg; running single-slot. "
                            "See continuous_batching.py for the future "
                            "multi-slot path (currently disabled).",
                            n_parallel,
                        )

                    # Use LeanLlama only if the capability probe says
                    # save_state copies scores (0.3.36 yes; future versions
                    # probed live). Falls back to stock Llama automatically.
                    try:
                        from local_llm_core.engine.inference.inference_engine import LeanLlama
                        _cls = LeanLlama.get(Llama)
                    except Exception as _le:
                        log.warning(
                            "LeanLlama unavailable (%s) — using stock Llama. "
                            "save_state will copy the scores buffer on every "
                            "cached prefix (n_tokens × vocab × 4 bytes).",
                            _le,
                        )
                        _cls = Llama
                    llm = _cls(**kwargs)
                    elapsed = time.monotonic() - t0
                    log.info(
                        "✓ Load successful in %.2fs | model=%s gpu=%d/%d ctx=%d batch=%d/%d parallel=%d",
                        elapsed, model_path.name, n_gpu, cfg.n_gpu_layers,
                        n_ctx, n_batch, n_ubatch, n_parallel
                    )

                    # ── Prefix cache (real, longest-prefix lookup) ─────
                    # The old prefix match used `llm._input_ids`, which
                    # only reflects the *currently loaded* sequence. Once
                    # any request overwrote it, agents/clients that
                    # shared a system prompt re-paid full prefill.
                    #
                    # Both LlamaRAMCache and LlamaTrieCache implement
                    # BaseLlamaCache.__getitem__, which calls
                    # _find_longest_prefix_key — i.e. they do longest
                    # shared token-prefix lookup, not exact-match dict
                    # lookup. We prefer LlamaTrieCache where it exists
                    # because it does the lookup in O(prompt length);
                    # LlamaRAMCache is O(cached_keys × key_length), which
                    # on a 4-core Ryzen is measurably slower once the
                    # cache holds dozens of prefixes.
                    try:
                        cache_bytes = max(
                            64 * 1024 * 1024,
                            int(getattr(self, "_kv_ram_budget_mb", 0) or 0)
                            * 1024 * 1024 // 2,
                        ) if getattr(self, "_kv_ram_budget_mb", 0) else 128 * 1024 * 1024

                        if not hasattr(llm, "set_cache"):
                            log.warning(
                                "llama_cpp.Llama has no set_cache() — "
                                "installed version is older than 0.3.0; "
                                "prefix reuse will fall back to live "
                                "_input_ids matching only."
                            )
                        else:
                            _cache_obj = None
                            _cache_name = "none"

                            # lamaRAMCache.
                            if _cache_obj is None:
                                try:
                                    from llama_cpp.llama_cache import LlamaRAMCache
                                    _cache_obj = LlamaRAMCache(capacity_bytes=cache_bytes)
                                    _cache_name = "LlamaRAMCache"
                                except Exception as exc:
                                    log.warning(
                                        "LlamaRAMCache construction failed "
                                        "(%s) — prefix reuse disabled.",
                                        exc,
                                    )
                                    _cache_obj = None

                            if _cache_obj is not None:
                                llm.set_cache(_cache_obj)
                                log.info(
                                    "%s attached (capacity=%d MB) — "
                                    "longest-prefix lookup active",
                                    _cache_name, cache_bytes >> 20,
                                )
                    except Exception as exc:
                        log.warning("Prefix-cache setup failed: %s", exc)

                    # ── KV-cache RAM budget (used by chunked_prefill) ─
                    # Passed to chunked_prefill so it can size the block
                    # boundary cache without re-reading MemAvailable on
                    # every call.
                    try:
                        import psutil as _ps
                        _avail_mb = int(_ps.virtual_memory().available >> 20)
                        llm._kv_ram_budget_mb = max(128, int(_avail_mb * 0.30))
                    except Exception:
                        llm._kv_ram_budget_mb = 256

                    return llm
                except TypeError as te:
                    log.warning("TypeError: %s", te)
                    llm = self._handle_typeerror(kwargs, te, Llama)
                    if llm is not None:
                        elapsed = time.monotonic() - t0
                        log.info("✓ Load successful (after kwarg adjustment) in %.2fs", elapsed)
                        return llm
                    raise
                except BaseException as load_exc:
                    if isinstance(load_exc, KeyboardInterrupt):
                        raise
                    # ── Skip context retry for hard format/compatibility errors ────────
                    _err_s = str(load_exc).lower()
                    _is_format_err = (
                        "failed to load model from file" in _err_s
                        or "invalid model" in _err_s
                        or "not a valid gguf" in _err_s
                        or "unsupported model type" in _err_s
                    )
                    if n_ctx > min_ctx and not _is_format_err:
                        llm = self._handle_loader_exception_with_context_retry(
                            kwargs, load_exc, n_ctx, min_ctx, ctx_reduction_steps, Llama
                        )
                        if llm is not None:
                            elapsed = time.monotonic() - t0
                            log.info("✓ Load successful (after context reduction) in %.2fs", elapsed)
                            return llm
                    raise

            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                last_exception = exc
                err_lower = str(exc).lower()

                _is_format_error = (
                    "failed to load model from file" in err_lower
                    or "not a valid gguf" in err_lower
                    or "invalid model" in err_lower
                    or "unsupported model type" in err_lower
                )

                is_oom = (
                    "out of memory" in err_lower
                    or "oom" in err_lower
                    or "ggml_" in err_lower
                    or "failed to allocate" in err_lower
                    or "cuda out" in err_lower
                    or "hip out" in err_lower
                    or "vulkan" in err_lower
                    or "couldn't allocate" in err_lower
                    or "not enough memory" in err_lower
                    or isinstance(exc, MemoryError)
                )

                if _is_format_error:
                    _mp_name = Path(cfg.model_path).name
                    _is_bitnet_gguf = any(
                        x in _mp_name.lower()
                        for x in ("i2_s", "tl1", "tl2", "bitnet", "ggml-model")
                    )
                    if _is_bitnet_gguf:
                        log.error(
                            "Cannot load BitNet GGUF '%s' with standard llama.cpp.\n"
                            "  BitNet i2_s / TL1 / TL2 quantization requires the BitNet.cpp\n"
                            "  executable (bitnet-cli) — NOT standard llama-cpp-python.\n"
                            "  ┌─ To fix: run `local-llm-setup` (builds bitnet.cpp from source)\n"
                            "  │          or `local-llm-setup --minimal-build` (downloads GGUF)\n"
                            "  │          then restart the service.\n"
                            "  └─ Or: switch config.yaml to `qwen3-8b: true` which uses\n"
                            "         standard llama.cpp and works without any setup.",
                            _mp_name,
                        )
                    else:
                        log.error(
                            "Model format/compatibility error (non-retryable): %s — %s",
                            _mp_name, str(exc)[:200],
                        )
                    raise

                if is_oom:
                    if n_parallel > 1:
                        n_parallel = max(1, n_parallel - 1)
                        log.warning("OOM, reducing n_parallel to %d", n_parallel)
                        gc.collect()
                        if torch_available():
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue
                    if n_gpu > 0:
                        old_gpu = n_gpu
                        n_gpu = max(0, int(n_gpu * layer_reduction_factor))
                        log.warning("OOM, reducing gpu_layers: %d → %d", old_gpu, n_gpu)
                        gc.collect()
                        if torch_available():
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue
                    if n_ctx > min_ctx:
                        old_ctx = n_ctx
                        n_ctx = max(min_ctx, n_ctx // 2)
                        log.warning("OOM, reducing n_ctx: %d → %d", old_ctx, n_ctx)
                        continue
                    log.error("OOM after exhaustive retries; cannot load model.")
                log.error("Non-OOM error: %s", str(exc)[:200])
                raise

        raise RuntimeError(
            f"Failed to load model after {max_attempts} attempts. "
            f"Last error: {str(last_exception)[:300] if last_exception else 'unknown'}"
        )

    # -------------------------------------------------------------------------
    # Helper methods (unchanged)
    # -------------------------------------------------------------------------
    def _handle_typeerror(self, kwargs: Dict[str, Any], te: TypeError, Llama) -> Optional[Any]:
        msg_lower = str(te).lower()
        removable_params = [
            ("n_ubatch", "n_ubatch not supported – removing"),
            ("type_k", "type_k not supported – removing"),
            ("type_v", "type_v not supported – removing"),
            ("rope_freq_base", "rope_freq_base not supported – removing"),
            ("tensor_split", "tensor_split not supported – removing"),
        ]
        for param, msg in removable_params:
            if param in msg_lower or f"'{param}'" in str(te):
                if param in kwargs:
                    log.warning("TypeError: %s", msg)
                    kwargs.pop(param, None)
                    try:
                        return Llama(**kwargs)
                    except TypeError:
                        continue
        log.warning("TypeError recovery: removing all optional parameters")
        essential_params = {"model_path", "n_ctx", "n_batch", "n_gpu_layers"}
        kwargs_essential = {k: v for k, v in kwargs.items() if k in essential_params}
        try:
            return Llama(**kwargs_essential)
        except Exception as e:
            log.error("Even essential-only kwargs failed: %s", e)
            return None

    def _handle_loader_exception_with_context_retry(
            self,
            kwargs: Dict[str, Any],
            exc: BaseException,
            current_ctx: int,
            min_ctx: int,
            ctx_reduction_steps: List[int],
            Llama,
    ) -> Optional[Any]:
        if current_ctx <= min_ctx:
            return None
        ctx_attempts = [current_ctx]
        for div in ctx_reduction_steps:
            reduced = max(min_ctx, current_ctx // div)
            if reduced not in ctx_attempts:
                ctx_attempts.append(reduced)
        ctx_attempts = sorted(set(ctx_attempts), reverse=True)
        if len(ctx_attempts) <= 1:
            return None
        for ctx_try in ctx_attempts[1:]:
            try:
                kwargs["n_ctx"] = int(ctx_try)
                log.warning("Context retry: n_ctx %d → %d", current_ctx, ctx_try)
                llm = Llama(**kwargs)
                log.info("✓ Loaded with reduced n_ctx=%d", ctx_try)
                return llm
            except BaseException as ctx_exc:
                if isinstance(ctx_exc, KeyboardInterrupt):
                    raise
                log.warning("Context retry n_ctx=%d failed: %s", ctx_try, str(ctx_exc)[:80])
                if ctx_try == min_ctx:
                    log.error("Failed even with min_ctx=%d", min_ctx)
                    break
        return None

    def _apply_kv_cache_type(
        self, kwargs: Dict[str, Any], kv_k: str, kv_v: str, Llama
    ) -> None:
        kv_int_map = {
            "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3,
            "q5_0": 6, "q5_1": 7, "q8_0": 8, "q8_1": 9,
            "q2_k": 10, "q3_k": 11, "q4_k": 12,
            "q5_k": 13, "q6_k": 14, "q8_k": 15, "bf16": 30,
        }
        try:
            from llama_cpp import LlamaKvCacheType as KVT
            attr_map = {
                "q8_0": ("K_Q8_0", "Q8_0"),
                "q4_0": ("K_Q4_0", "Q4_0"),
                "f16": ("K_F16", "F16"),
                "f32": ("K_F32", "F32"),
                "q4_k": ("K_Q4_K", "Q4_K"),
                "q5_k": ("K_Q5_K", "Q5_K"),
                "q6_k": ("K_Q6_K", "Q6_K"),
                "q8_k": ("K_Q8_K", "Q8_K"),
            }
            def resolve(s):
                for attr in attr_map.get(s, ()):
                    if hasattr(KVT, attr):
                        return getattr(KVT, attr)
                return None
            tv_k = resolve(kv_k)
            tv_v = resolve(kv_v)
            if tv_k is not None and tv_v is not None:
                kwargs["type_k"] = tv_k
                kwargs["type_v"] = tv_v
                return
        except (ImportError, AttributeError):
            pass
        int_k = kv_int_map.get(kv_k)
        int_v = kv_int_map.get(kv_v)
        if int_k is not None and int_v is not None:
            import inspect
            if "type_k" in inspect.signature(Llama.__init__).parameters:
                kwargs["type_k"] = int(int_k)
                kwargs["type_v"] = int(int_v)

    def _handle_loader_exception(
        self, kwargs: Dict[str, Any], exc: BaseException, ctx_attempts: List[int], Llama
    ) -> Any:
        err = str(exc).lower()
        if kwargs.get("flash_attn") and any(
            x in err for x in ("flash", "attn", "not supported", "invalid")
        ):
            log.warning("flash_attn failed — retrying without it")
            kwargs["flash_attn"] = False
            return Llama(**kwargs)

        last_exc = exc
        for ctx_try in ctx_attempts[1:]:
            try:
                kwargs["n_ctx"] = ctx_try
                log.warning("Llama() failed (%s) — retrying with n_ctx=%d", type(exc).__name__, ctx_try)
                llm = Llama(**kwargs)
                log.info("Loaded with reduced n_ctx=%d", ctx_try)
                return llm
            except BaseException as ctx_exc:
                if isinstance(ctx_exc, KeyboardInterrupt):
                    raise
                last_exc = ctx_exc
                continue
        raise last_exc

    # -------------------------------------------------------------------------
    # Other Backends (unchanged)
    # -------------------------------------------------------------------------
    def _load_transformers(self) -> Dict[str, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            import torch
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            device_map = "auto"
        except ImportError:
            bnb, device_map = None, "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            quantization_config=bnb,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True
        )
        return {"model": model, "tokenizer": tokenizer}

    def _load_ctransformers(self) -> Any:
        from transformers import AutoModelForCausalLM

        meta = read_gguf_meta(Path(self.cfg.model_path))
        return AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            model_type=meta.get("arch", "llama"),
            gpu_layers=self.cfg.n_gpu_layers,
            context_length=self.cfg.n_ctx,
        )

    def _load_sim(self) -> Dict[str, str]:
        return {"type": "sim", "path": self.cfg.model_path}


def torch_available() -> bool:
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False
