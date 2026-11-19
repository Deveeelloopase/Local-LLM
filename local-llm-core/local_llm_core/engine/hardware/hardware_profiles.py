from __future__ import annotations

# engine/hardware/hardware_profiles.py — Production‑Ready Hardware Profiles
# ==========================================================================
#  Pure library: configuration overrides injected, no file I/O
# for anything on the hot path. Bench results are read once at profile
# detection time from a small JSON file (§7.3 of the playbook).
#
# Detects system hardware and returns an optimal HardwareProfile for
# inference. Overrides are applied from a provided hardware configuration
# dict. Profile detection is cached (base profile only) to avoid repeated
# computation, while overrides are re-applied each call to honour config
# changes.
#
# ── Measuring vs. estimating target_tps ──────────────────────────────
# `target_tps` is the number the client, the TUI, and /v1/runtime/limits
# all display as "what to expect". Getting it honest is the difference
# between a monitor that shows 2.70 and a box that does 3.06, and a
# monitor that shows 3.06 and a box that does 3.06.
#
# Three sources, in priority order (also recorded in
# `HardwareProfile.target_tps_source`):
#
#   1. "bench_measured" — the operator ran tools/bench_decode.py on the
#      exact model + quantization + hardware, and persisted the result to
#      /var/lib/local-llm/bench_decode.json (or passed it explicitly as
#      `measured_decode_tps=`). This is the only number we treat as
#      authoritative.
#   2. "roofline" — computed from effective memory bandwidth ÷ bytes
#      read per token × a roofline efficiency constant. Accurate to
#      roughly ±20 % because decode-bound throughput is genuinely
#      bounded by bandwidth, but the constant (0.85) is a fudge.
#   3. "tier_default" — the tier table's aspirational number. Useful
#      only as a last resort; on a UMA APU the tier default (8.0) is
#      roughly 2.6× the actual achievable rate.
#
# Persistence format at BENCH_FILE_PATH (JSON):
#   {
#     "model_path":  "/abs/path/to/Qwen3-8B-Q4_K_M.gguf",
#     "model_id":    "qwen3-8b",
#     "quantization":"Q4_K_M",
#     "decode_tps":  3.06,
#     "prefill_tps": 3.9,
#     "gpu_layers":  36,
#     "threads":     4,
#     "flash_attn":  true,
#     "ctx":         4096,
#     "measured_at": 1731600000.0,
#     "hostname":    "deve-oops"
#   }
#
# Matching is by model file basename so the same GGUF on a different
# absolute path still hits the cache. If no matching record exists,
# compute_profile falls through to roofline / tier default with a
# clear log line stating which source it used.

# version = 1.0.0
# ==========================================================================

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bench-result file and roofline efficiency constant
# -----------------------------------------------------------------------------
# Override the location with $LLM_BENCH_FILE (systemd units already set
# $LLM_STATE_DIR=/var/lib/local-llm, so this default works out of the box).
BENCH_FILE_PATH = Path(
    os.environ.get("LLM_BENCH_FILE")
    or str(
        Path(os.environ.get("LLM_STATE_DIR", "/var/lib/local-llm"))
        / "bench_decode.json"
    )
)

# Fraction of the memory bandwidth bound that real decode achieves.
# 0.85 is conservative: the bench on this box gives ~90 % of theoretical
# after accounting for the 4.8 GB-per-token read, and the playbook's
# "0.9 × bw / model_size" gave 3.6 vs. measured 3.06 — i.e. it was ~18 %
# optimistic. 0.85 pulls the roofline estimate into the measured range.
ROOFLINE_EFFICIENCY = 0.85

# Bandwidth measurement plausibility band relative to theoretical. A
# measured value inside this band is trusted; anything outside is
# almost certainly a probe artefact (single-threaded, cache-sized
# buffer, contention during the triad) and is replaced by theory.
# 30 %: a real probe that runs single-threaded on a multi-core memory
#       bus can legitimately return ~30 % of multi-core peak.
# 150 %: some UMA APUs briefly exceed theory during short triads
#        because the probe fits in L3. Anything above 150 % is noise.
_BW_LOW_RATIO = 0.30
_BW_HIGH_RATIO = 1.50


# -----------------------------------------------------------------------------
# HardwareProfile Dataclass
# -----------------------------------------------------------------------------
@dataclass
class HardwareProfile:
    """Optimal inference configuration for a detected hardware class."""

    name: str

    # VRAM budgeting
    vram_budget_pct: float
    vram_os_reserve_mb: int

    # llama.cpp batching
    n_batch: int
    n_ubatch: int

    # Memory & context
    context_length: int
    ram_target: float
    kv_cache_quant: str

    # CPU
    cpu_thread_ratio: float

    # Throughput target
    target_tps: float

    # Fields with defaults (must come last)
    flash_attn: bool = True
    n_parallel: int = 1
    force_no_think: bool = False
    speculative_decoding: bool = False

    # Policy for context‑vs‑layers tradeoff
    context_vs_layers_priority: str = "balanced"   # "context", "layers", "balanced"

    # Whether to mlock() model weights into RAM so the OS cannot swap them out
    # under memory pressure. Most valuable for APU tiers where a large fraction
    # of the model resides in system RAM. Requires the process memlock ulimit
    # to be raised (see config.yaml notes).
    use_mlock: bool = False

    # GTT-inclusive effective VRAM total for UMA APUs. When > 0,
    # vram_budget_mb() uses this instead of the passed-in carve-out.
    # The carve-out (typically 2-4 GB) badly understates the pool the
    # iGPU can address through GTT (typically 50-75% of system RAM).
    # 0 means: use whatever the caller passes to vram_budget_mb().
    vram_effective_mb: int = 0

    # Provenance of `target_tps` — informational only, but the operator
    # and /v1/runtime/limits both benefit from knowing whether the
    # number is measured, roofline-derived, or a tier default.
    # Values: "bench_measured", "bench_file", "roofline", "tier_default",
    #         "config_override", "unknown".
    target_tps_source: str = "tier_default"

    # -------------------------------------------------------------------------
    # Computed helpers
    # -------------------------------------------------------------------------
    def vram_budget_mb(self, vram_total_mb: int) -> int:
        total = self.vram_effective_mb if self.vram_effective_mb > 0 else vram_total_mb
        raw = int(total * self.vram_budget_pct)
        return max(0, raw - self.vram_os_reserve_mb)

    def kv_budget_mb(self, vram_total_mb: int, model_vram_mb: int) -> int:
        return max(0, self.vram_budget_mb(vram_total_mb) - model_vram_mb)

    def compute_context_length(
        self,
        model_n_ctx_train: int,
        vram_total_mb: int,
        model_vram_mb: int,
        kv_cache_bytes_per_token: float = 1.0,
    ) -> int:
        """
        Return the effective context length, capped by both the model's native
        maximum and the KV‑cache memory headroom.
        """
        kv_budget_mb = self.kv_budget_mb(vram_total_mb, model_vram_mb)
        headroom_bytes = kv_budget_mb * 1024 * 1024
        tokens_from_mem = int(headroom_bytes // kv_cache_bytes_per_token) if kv_cache_bytes_per_token > 0 else 0
        effective = min(model_n_ctx_train, tokens_from_mem)
        return max(self.context_length, effective)

    def n_threads(self) -> int:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
        except (ImportError, Exception):
            physical = None
        if physical is None:
            physical = os.cpu_count()
        if physical is None or physical <= 0:
            physical = 4
            log.warning("CPU count detection failed, defaulting to %d", physical)
        threads = int(physical * self.cpu_thread_ratio)
        return max(1, threads)

    def effective_n_parallel(
        self,
        vram_total_mb: int,
        model_vram_mb: int,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        n_layers: int = 36,
    ) -> int:
        quant_bytes = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}.get(
            self.kv_cache_quant, 1.0
        )
        kv_per_slot_mb = (
            self.context_length * n_kv_heads * head_dim * 2 * quant_bytes * n_layers
        ) / (1024 ** 2)
        available_mb = self.kv_budget_mb(vram_total_mb, model_vram_mb)
        safe = max(1, int(available_mb / max(kv_per_slot_mb, 1)))
        return min(self.n_parallel, safe)

    # -------------------------------------------------------------------------
    # Merge user overrides (from a provided config dict)
    # -------------------------------------------------------------------------
    def apply_overrides(self, overrides: Dict[str, Any]) -> "HardwareProfile":
        """Return a new HardwareProfile with the given overrides applied.

        Unknown override keys are filtered with a WARNING instead of
        raising `TypeError` (a single typo in config.yaml should not
        crash profile detection). When `target_tps` is overridden but
        `target_tps_source` is not, the source is set to
        `"config_override"` so that downstream consumers (the startup
        log, /v1/runtime/limits, the TUI) can distinguish a
        user-configured value from a measured or computed one.
        """
        kwargs = {
            "name": self.name,
            "vram_budget_pct": self.vram_budget_pct,
            "vram_os_reserve_mb": self.vram_os_reserve_mb,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "context_length": self.context_length,
            "ram_target": self.ram_target,
            "kv_cache_quant": self.kv_cache_quant,
            "cpu_thread_ratio": self.cpu_thread_ratio,
            "target_tps": self.target_tps,
            "flash_attn": self.flash_attn,
            "n_parallel": self.n_parallel,
            "force_no_think": self.force_no_think,
            "speculative_decoding": self.speculative_decoding,
            "context_vs_layers_priority": self.context_vs_layers_priority,
            "use_mlock": self.use_mlock,
            "vram_effective_mb": self.vram_effective_mb,
            "target_tps_source": self.target_tps_source,
        }
        _valid_keys = set(kwargs.keys())
        _tps_in_overrides = "target_tps" in overrides
        _source_in_overrides = "target_tps_source" in overrides
        for k, v in overrides.items():
            if k in _valid_keys:
                kwargs[k] = v
            else:
                log.warning(
                    "HardwareProfile.apply_overrides: unknown override key %r "
                    "ignored (valid keys: %s). Check config.yaml for typos "
                    "under hardware_profiles.overrides / profile_overrides.",
                    k, sorted(_valid_keys),
                )
        # If the caller changed target_tps without saying where it came
        # from, assume they set it deliberately — mark provenance so
        # nobody downstream mistakes it for a measurement.
        if _tps_in_overrides and not _source_in_overrides:
            kwargs["target_tps_source"] = "config_override"
        return HardwareProfile(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vram_budget_pct": self.vram_budget_pct,
            "vram_os_reserve_mb": self.vram_os_reserve_mb,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "context_length": self.context_length,
            "ram_target": self.ram_target,
            "kv_cache_quant": self.kv_cache_quant,
            "cpu_thread_ratio": self.cpu_thread_ratio,
            "target_tps": self.target_tps,
            "target_tps_source": self.target_tps_source,
            "flash_attn": self.flash_attn,
            "n_parallel": self.n_parallel,
            "force_no_think": self.force_no_think,
            "speculative_decoding": self.speculative_decoding,
            "context_vs_layers_priority": self.context_vs_layers_priority,
            "use_mlock": self.use_mlock,
            "vram_effective_mb": self.vram_effective_mb,
        }


# -----------------------------------------------------------------------------
# Base Profile Registry
# -----------------------------------------------------------------------------
# Tier defaults. Every `target_tps` value here is an ASPIRATIONAL number
# for that hardware class, NOT a measurement. On a real box the sequence
# should be:
#   1. If /var/lib/local-llm/bench_decode.json matches the loaded model
#      → measured value wins (target_tps_source="bench_file").
#   2. Else if the caller passes measured_decode_tps=X > 0
#      → measured value wins (target_tps_source="bench_measured").
#   3. Else if compute_profile can derive the roofline (model_bytes > 0
#      and a plausible bandwidth figure)
#      → roofline value wins (target_tps_source="roofline").
#   4. Else → the tier default below (target_tps_source="tier_default").
#   5. If config.yaml sets hardware_profiles.overrides.<tier>.target_tps,
#      that wins over anything above (target_tps_source="config_override").
#
# Fields kept identical across the tier table and compute_profile():
#   * cpu_thread_ratio         APU=1.0, non-APU=0.85
#   * context_vs_layers_priority  APU="layers", non-APU="balanced"
#   * use_mlock                False everywhere; opt-in via config override
#   * speculative_decoding     False everywhere (no draft model by default)
_BASE_PROFILES: Dict[str, HardwareProfile] = {
    "apu_2gb": HardwareProfile(
        name="apu_2gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=50,
        n_batch=2048,
        n_ubatch=512,
        context_length=2048,
        ram_target=0.75,
        kv_cache_quant="q4_0",
        cpu_thread_ratio=1.0,
        target_tps=8.0,
        flash_attn=True,
        n_parallel=1,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="layers",
        use_mlock=False,
    ),
    "apu_4gb": HardwareProfile(
        name="apu_4gb",
        vram_budget_pct=0.92,
        vram_os_reserve_mb=250,
        n_batch=512,
        n_ubatch=512,
        context_length=4096,
        ram_target=0.75,
        kv_cache_quant="q8_0",
        cpu_thread_ratio=1.0,
        target_tps=18.0,
        flash_attn=True,
        n_parallel=1,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="layers",
        use_mlock=False,
    ),
    "gpu_4gb": HardwareProfile(
        name="gpu_4gb",
        vram_budget_pct=0.90,
        vram_os_reserve_mb=150,
        n_batch=512,
        n_ubatch=512,
        context_length=4096,
        ram_target=0.75,
        kv_cache_quant="q8_0",
        cpu_thread_ratio=0.85,
        target_tps=25.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_6gb": HardwareProfile(
        name="gpu_6gb",
        vram_budget_pct=0.92,
        vram_os_reserve_mb=150,
        n_batch=512,
        n_ubatch=512,
        context_length=4096,
        ram_target=0.75,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=40.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_8gb": HardwareProfile(
        name="gpu_8gb",
        vram_budget_pct=0.93,
        vram_os_reserve_mb=150,
        n_batch=1024,
        n_ubatch=512,
        context_length=8192,
        ram_target=0.75,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=60.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_12gb": HardwareProfile(
        name="gpu_12gb",
        vram_budget_pct=0.94,
        vram_os_reserve_mb=150,
        n_batch=2048,
        n_ubatch=512,
        context_length=8192,
        ram_target=0.75,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=80.0,
        flash_attn=True,
        n_parallel=3,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_16gb": HardwareProfile(
        name="gpu_16gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=200,
        n_batch=2048,
        n_ubatch=1024,
        context_length=16384,
        ram_target=0.75,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=100.0,
        flash_attn=True,
        n_parallel=4,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_24gb": HardwareProfile(
        name="gpu_24gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=200,
        n_batch=4096,
        n_ubatch=1024,
        context_length=32768,
        ram_target=0.80,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=150.0,
        flash_attn=True,
        n_parallel=8,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_32gb": HardwareProfile(
        name="gpu_32gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=300,
        n_batch=4096,
        n_ubatch=2048,
        context_length=65536,
        ram_target=0.80,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=200.0,
        flash_attn=True,
        n_parallel=8,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_48gb": HardwareProfile(
        name="gpu_48gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=300,
        n_batch=8192,
        n_ubatch=2048,
        context_length=65536,
        ram_target=0.85,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=250.0,
        flash_attn=True,
        n_parallel=16,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_80gb": HardwareProfile(
        name="gpu_80gb",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=512,
        n_batch=8192,
        n_ubatch=4096,
        context_length=131072,
        ram_target=0.85,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=350.0,
        flash_attn=True,
        n_parallel=32,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "gpu_192gb+": HardwareProfile(
        name="gpu_192gb+",
        vram_budget_pct=0.95,
        vram_os_reserve_mb=1024,
        n_batch=16384,
        n_ubatch=4096,
        context_length=131072,
        ram_target=0.90,
        kv_cache_quant="f16",
        cpu_thread_ratio=0.85,
        target_tps=600.0,
        flash_attn=True,
        n_parallel=64,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="balanced",
    ),
    "cpu_only": HardwareProfile(
        name="cpu_only",
        vram_budget_pct=0.0,
        vram_os_reserve_mb=0,
        n_batch=512,
        n_ubatch=128,
        context_length=4096,
        ram_target=0.70,
        kv_cache_quant="q8_0",
        cpu_thread_ratio=0.85,
        target_tps=3.0,
        flash_attn=False,
        n_parallel=1,
        force_no_think=False,
        speculative_decoding=False,
        context_vs_layers_priority="layers",
    ),
}

# Legacy alias
_BASE_PROFILES["gpu_24gb+"] = _BASE_PROFILES["gpu_24gb"]


# -----------------------------------------------------------------------------
# Bench-result helpers
# -----------------------------------------------------------------------------
def _load_bench_measurements(
    bench_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the persisted bench record, or None.

    Reads BENCH_FILE_PATH (or `bench_path`). Never raises — a missing or
    malformed file just returns None. Called from compute_profile and
    find_bench_for_model, so keeping it silent is important.
    """
    p = bench_path or BENCH_FILE_PATH
    try:
        if not p.is_file():
            return None
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log.info(
            "Bench file %s unreadable (%s) — falling back to roofline",
            p, exc,
        )
        return None


def find_bench_for_model(
    model_path: str,
    bench_path: Optional[Path] = None,
) -> float:
    """Return `decode_tps` from the persisted bench record for `model_path`.

    Match priority:
      1. Exact model_path string match.
      2. Basename match — lets the same GGUF on a different absolute
         path still hit the cache, which is the common case for repos
         that move between `~/models` and `/opt/models`.

    Returns 0.0 when no match exists, when the record has no `decode_tps`,
    or when the bench file is missing/unreadable.
    """
    data = _load_bench_measurements(bench_path)
    if not data:
        return 0.0
    try:
        recorded_path = str(data.get("model_path") or "")
        if recorded_path:
            if recorded_path == model_path:
                return float(data.get("decode_tps", 0.0) or 0.0)
            try:
                if Path(recorded_path).name == Path(model_path).name:
                    return float(data.get("decode_tps", 0.0) or 0.0)
            except Exception:
                pass
        # Fall through: the record has no model_path but has decode_tps.
        # Only accept this if the caller asked for the file generically
        # (empty model_path) — a specific mismatch should not silently
        # borrow a different model's measurement.
        if not recorded_path and not model_path:
            return float(data.get("decode_tps", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    return 0.0

def _bench_implied_bandwidth_gbps(
    model_path: str,
    roofline_eff: float = ROOFLINE_EFFICIENCY,
) -> float:
    """If a bench record exists for this model, derive the effective
    memory bandwidth from it:

        bench_effective_bw_gbps ≈ decode_tps × model_size_gb / roofline_eff

    For this box: 3.06 × 4.79 / 0.85 ≈ 17.3 GB/s. The parallel triad at
    startup measures 4.3 GB/s on the same hardware; the triad is
    under-reporting (numpy GIL, cache line contention, or thread startup
    cost dominates at these sizes). The bench result is the honest
    number — it is the rate the model actually achieved.

    Returns 0.0 when no bench record, no model file, or any exception.
    """
    try:
        _tps = find_bench_for_model(model_path)
        if _tps <= 0:
            return 0.0
        _p = Path(model_path)
        if not _p.is_file():
            return 0.0
        _size_gb = _p.stat().st_size / 1e9
        return float(_tps * _size_gb / max(roofline_eff, 0.01))
    except Exception:
        return 0.0

def record_bench_measurement(
    *,
    model_path: str,
    decode_tps: float,
    prefill_tps: float = 0.0,
    model_id: str = "",
    quantization: str = "",
    gpu_layers: int = 0,
    threads: int = 0,
    flash_attn: bool = False,
    ctx: int = 0,
    bench_path: Optional[Path] = None,
) -> bool:
    """Persist a bench result for later `compute_profile` reads.

    Call this from `tools/bench_decode.py` (or from the operator's shell
    after a manual bench) to make the number available to the loader on
    the next start. Format is documented at the top of this module.

    Returns True on success, False on write failure. Never raises — a
    failed bench-file write must not stop the caller.
    """
    p = bench_path or BENCH_FILE_PATH
    record = {
        "model_path": str(model_path or ""),
        "model_id": str(model_id or ""),
        "quantization": str(quantization or ""),
        "decode_tps": float(decode_tps or 0.0),
        "prefill_tps": float(prefill_tps or 0.0),
        "gpu_layers": int(gpu_layers or 0),
        "threads": int(threads or 0),
        "flash_attn": bool(flash_attn),
        "ctx": int(ctx or 0),
        "measured_at": float(time.time()),
        "hostname": socket.gethostname(),
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(p)
        log.info(
            "Persisted bench: %s → decode_tps=%.2f prefill_tps=%.2f "
            "(gpu_layers=%d threads=%d flash_attn=%s ctx=%d)",
            model_path, record["decode_tps"], record["prefill_tps"],
            record["gpu_layers"], record["threads"],
            record["flash_attn"], record["ctx"],
        )
        return True
    except Exception as exc:
        log.warning("Could not persist bench file %s: %s", p, exc)
        return False


# -----------------------------------------------------------------------------
# Profile Caching (base profile only – overrides are applied each call)
# -----------------------------------------------------------------------------
_detected_base_profile_cache: Optional[HardwareProfile] = None
_cache_lock = threading.Lock()


def _apply_global_overrides(profile: HardwareProfile, hw_config: Dict[str, Any]) -> HardwareProfile:
    """Apply global overrides from the hardware config's 'overrides' section."""
    overrides = hw_config.get("overrides", {})
    if overrides:
        log.info("Applying global hardware overrides: %s", overrides)
        return profile.apply_overrides(overrides)
    return profile


def _apply_profile_specific_overrides(
    profile: HardwareProfile, profile_key: str, hw_config: Dict[str, Any]
) -> HardwareProfile:
    """Apply per-profile overrides from hardware config's 'profile_overrides' section."""
    specific = hw_config.get("profile_overrides", {}).get(profile_key, {})
    if specific:
        log.info("Applying profile‑specific overrides for '%s': %s", profile_key, specific)
        return profile.apply_overrides(specific)
    return profile


def _apply_measured_tps(
    base: HardwareProfile, measured_decode_tps: float, source: str,
) -> HardwareProfile:
    """Return `base` with target_tps replaced by the measured value.

    Called on the tier-lookup path (detect_profile / get_profile) where
    compute_profile is not in the call chain. Sets
    `target_tps_source` so downstream consumers can distinguish a
    measurement from a default.
    """
    if measured_decode_tps <= 0:
        return base
    return base.apply_overrides({
        "target_tps": float(measured_decode_tps),
        "target_tps_source": source,
    })


# -----------------------------------------------------------------------------
# APU Detection Helpers
# -----------------------------------------------------------------------------
def _is_apu(gpu_info: Any, total_vram_mb: int) -> bool:
    mt = getattr(gpu_info, "memory_type", None)
    mtval = (mt.value if hasattr(mt, "value") else str(mt)).upper()
    if any(s in mtval for s in ("SHARED", "LPDDR", "UMADT")):
        return True
    if any(s in mtval for s in ("GDDR", "HBM")):
        return False
    name = getattr(gpu_info, "name", "").upper()
    discrete_markers = (
        "RTX ", "GTX ", "TESLA", "QUADRO", "A100", "H100", "H200",
        "L40", "A40", "A6000", "MI300", "RADEON RX"
    )
    if any(m in name for m in discrete_markers):
        return False
    vendor = getattr(gpu_info, "vendor", None)
    vname = (vendor.name if hasattr(vendor, "name") else str(vendor)).upper()
    if vname in ("AMD", "INTEL") and total_vram_mb <= 4096:
        return True
    return False


def _canonical_name(vram_mb: int, is_apu: bool, has_gpu: bool) -> str:
    """Map measured hardware to an existing tier name.

    Tier names are the *config-override* key in config.yaml (see
    hardware_profiles.overrides.<name>). Preserving them here means the
    operator's existing overrides keep working even when the profile is
    computed rather than looked up.
    """
    if not has_gpu or vram_mb == 0:
        return "cpu_only"
    if is_apu:
        return "apu_4gb" if vram_mb > 3072 else "apu_2gb"
    if vram_mb < 5120:  return "gpu_4gb"
    if vram_mb < 7168:  return "gpu_6gb"
    if vram_mb < 10240: return "gpu_8gb"
    if vram_mb < 14336: return "gpu_12gb"
    if vram_mb < 20480: return "gpu_16gb"
    if vram_mb < 32768: return "gpu_24gb"
    if vram_mb < 49152: return "gpu_32gb"
    if vram_mb < 65536: return "gpu_48gb"
    if vram_mb < 98304: return "gpu_80gb"
    return "gpu_192gb+"


# -----------------------------------------------------------------------------
# Bandwidth selection — the §7.2 item 2 fix
# -----------------------------------------------------------------------------
def _select_effective_bandwidth(system_info: Any) -> Tuple[float, str]:
    """Return (effective_bandwidth_gbps, source_description).

    The old rule — "if measured < 50 % of theoretical, discard the
    measurement and use theory" — is precisely what made this box report
    target_tps=3.6 when it does 3.06. A real bandwidth probe run
    single-threaded on a multi-core memory bus can easily read 30–40 %
    of the theoretical multi-core peak; rejecting that reading pushed
    the estimate back to the theoretical 19.2 GB/s, which is *above* the
    effective bandwidth for decode (about 16 GB/s on this box).

    New rule:
      * Both present, ratio in [0.30, 1.50] → trust the measurement.
        It reflects channel interleaving, refresh, and real contention
        that theory does not.
      * Both present, ratio outside that band → the probe is broken
        (too low) or the measurement is a cache-fit artefact (too high);
        fall back to theory, log loudly.
      * Only measurement → trust it.
      * Only theory → use it.
      * Neither → conservative 25 GB/s (about one channel DDR4-2666).

    Never raises. All extraction is defensive.
    """
    bw_measured    = float(getattr(system_info, "ram_bandwidth_gbps_measured", 0.0) or 0.0)
    bw_theoretical = float(getattr(system_info, "ram_bandwidth_gbps_theoretical", 0.0) or 0.0)

    if bw_theoretical > 0 and bw_measured > 0:
        ratio = bw_measured / bw_theoretical
        if _BW_LOW_RATIO <= ratio <= _BW_HIGH_RATIO:
            return bw_measured, f"measurement ({ratio:.0%} of theory)"
        if ratio < _BW_LOW_RATIO:
            log.warning(
                "Bandwidth probe returned %.1f GB/s = %.0f%% of theory "
                "(%.1f GB/s) — below the %.0f%% plausibility floor. "
                "The probe is almost certainly single-threaded or "
                "ran into cache contention; using theory. To fix: run "
                "the probe on all physical cores with a buffer larger "
                "than L3, several passes, median.",
                bw_measured, ratio * 100, bw_theoretical, _BW_LOW_RATIO * 100,
            )
            return bw_theoretical, f"theory (probe {ratio:.0%} < {_BW_LOW_RATIO:.0%} floor)"
        # ratio > _BW_HIGH_RATIO
        log.warning(
            "Bandwidth probe returned %.1f GB/s = %.0f%% of theory "
            "(%.1f GB/s) — above the %.0f%% plausibility ceiling. "
            "The probe buffer is probably fitting inside L3. "
            "Using theory.",
            bw_measured, ratio * 100, bw_theoretical, _BW_HIGH_RATIO * 100,
        )
        return bw_theoretical, f"theory (probe {ratio:.0%} > {_BW_HIGH_RATIO:.0%} ceiling)"

    if bw_measured > 0:
        return bw_measured, "measurement (no theory to compare)"
    if bw_theoretical > 0:
        return bw_theoretical, "theory (no measurement)"
    return 25.0, "conservative fallback (25 GB/s)"


# -----------------------------------------------------------------------------
# Profile Computation
# -----------------------------------------------------------------------------
def compute_profile(
    system_info: Any,
    model_bytes: int = 0,
    model_meta: Optional[Dict[str, Any]] = None,
    hardware_config: Optional[Dict[str, Any]] = None,
    measured_decode_tps: float = 0.0,
) -> HardwareProfile:
    """Derive the profile from *measured* hardware rather than a tier table.

    `measured_decode_tps`: a directly-measured decode rate from
    `tools/bench_decode.py` run on this exact model + quantization. When
    > 0 it becomes `target_tps` verbatim; the roofline is skipped. This
    is the only source the playbook (§7.3) says to advertise.

    If `measured_decode_tps` is 0 and `model_meta` carries a
    `model_path` (or `path`), the persisted bench record at
    BENCH_FILE_PATH is consulted. Match is by exact path, then by
    basename so a repo moved between `~/models` and `/opt/models` still
    hits.

    If neither yields a number, the roofline estimator
    (ROOFLINE_EFFICIENCY × bandwidth ÷ bytes-per-token) is used. The
    source that won is recorded in `profile.target_tps_source` for the
    startup log and /v1/runtime/limits.
    """
    gpus = [g for g in (system_info.gpus or []) if getattr(g, "supported", True)]
    vram_total_mb = sum(int(getattr(g, "vram_total_mb", 0) or 0) for g in gpus)

    is_apu = any(
        (getattr(g.memory_type, "value", str(g.memory_type)) or "").upper()
        in ("SHARED", "SHARED SYSTEM RAM", "UNIFIED MEMORY")
        for g in gpus
    )

    ram_total_mb = int(getattr(system_info, "ram_total_mb", 0) or 0)
    ram_free_mb  = int(getattr(system_info, "ram_free_mb", 0) or 0)

    # ── Bandwidth, fixed selection ──
    effective_bw_gbps, _bw_source = _select_effective_bandwidth(system_info)
    log.info("Bandwidth selection: %.1f GB/s (%s)", effective_bw_gbps, _bw_source)

    # If a bench record exists for the model we're about to serve, the
    # bench-implied bandwidth is the honest measurement — the triad
    # under-reports by 3-4× on this class of box. Use it for the
    # roofline target_tps calculation that follows.
    _bench_bw = 0.0
    _mp = (
        (model_meta or {}).get("model_path")
        or (model_meta or {}).get("path")
        or ""
    ) if model_meta else ""
    if _mp:
        _bench_bw = _bench_implied_bandwidth_gbps(str(_mp))
        if _bench_bw > 0:
            log.info(
                "Bandwidth: using bench-implied %.1f GB/s (probe said %.1f, "
                "theory %.1f) — bench is authoritative",
                _bench_bw, effective_bw_gbps,
                float(getattr(system_info, "ram_bandwidth_gbps_theoretical", 0.0) or 0.0),
            )
            effective_bw_gbps = _bench_bw
            _bw_source = f"bench_implied ({_bench_bw:.1f} GB/s)"

    llama_cpp_backends = list(getattr(system_info, "llama_cpp_backends", []) or [])
    gpu_offload_supported = bool(getattr(system_info, "llama_cpp_gpu_offload", False))
    has_accel = gpu_offload_supported and bool(vram_total_mb) and any(
        b for b in llama_cpp_backends if b != "cpu"
    )

    # ── KV-cache quant: back off one step on tight RAM ──
    if ram_total_mb and ram_total_mb < 12288:
        kv_quant = "q4_0"
    elif ram_total_mb and ram_total_mb < 24576:
        kv_quant = "q8_0"
    elif ram_total_mb and ram_total_mb < 49152:
        kv_quant = "q8_0"
    else:
        kv_quant = "f16"

    # ── Base context (KV planner refines this per-model) ──
    base_ctx = 4096
    if ram_total_mb >= 8192:  base_ctx = 8192
    if ram_total_mb >= 16384: base_ctx = 12288
    if ram_total_mb >= 32768: base_ctx = 16384
    if ram_total_mb >= 65536: base_ctx = 32768

    # ── target_tps — measured first, roofline second, tier default last ──
    #
    # Order of preference is documented at the top of the module. The
    # short version: a bench on this exact hardware beats any model of
    # this hardware, every time. Roofline is a good second because
    # decode on Q4_K_M is genuinely bandwidth-bound; the ±20 % fudge in
    # ROOFLINE_EFFICIENCY is honest about that.
    measured_tps = float(measured_decode_tps or 0.0)
    _tps_source = "bench_measured" if measured_tps > 0 else ""

    if measured_tps <= 0 and model_meta:
        _mp = str(
            model_meta.get("model_path")
            or model_meta.get("path")
            or ""
        ).strip()
        if _mp:
            _from_file = find_bench_for_model(_mp)
            if _from_file > 0:
                measured_tps = _from_file
                _tps_source = "bench_file"
                log.info(
                    "Bench record found for %s: decode_tps=%.2f (from %s)",
                    _mp, measured_tps, BENCH_FILE_PATH,
                )

    if measured_tps > 0:
        target_tps = measured_tps
        target_tps_source = _tps_source or "bench_measured"
        log.info(
            "target_tps=%.2f from %s",
            target_tps, target_tps_source,
        )
    elif model_bytes > 0:
        model_size_gb = model_bytes / 1e9
        # bytes_per_token ≈ model_size_gb for a decode-bound Q4_K_M model:
        # every generated token reads the whole weight tensor. Roofline
        # ceiling is bandwidth ÷ bytes_per_token; efficiency constant
        # accounts for KV-cache reads, activations, and the fact that a
        # perfect bandwidth-bound kernel is unattainable in practice.
        target_tps = ROOFLINE_EFFICIENCY * effective_bw_gbps / model_size_gb
        target_tps = max(1.0, min(target_tps, 1000.0))
        target_tps_source = f"roofline ({_bw_source})"
        log.info(
            "target_tps=%.2f from %s (bw=%.1f GB/s, model=%.2f GB, eff=%.2f)",
            target_tps, target_tps_source,
            effective_bw_gbps, model_size_gb, ROOFLINE_EFFICIENCY,
        )
    else:
        target_tps = 8.0 if has_accel else 3.0
        target_tps_source = "tier_default"
        log.info(
            "target_tps=%.2f from tier_default "
            "(no bench, no model size yet — will refine after load)",
            target_tps,
        )

    # ── VRAM budget ──
    # Save the carve-out value before any UMA substitution so the tier
    # name (used for config-override lookup) stays tied to physical
    # hardware class, not to the GTT-inclusive effective budget.
    _carve_out_mb = vram_total_mb
    if is_apu:
        _gtt_mb = int(getattr(system_info, "vram_gtt_total_mb", 0) or 0)
        _hw_cfg = hardware_config or {}
        _gtt_offload = bool(_hw_cfg.get("uma_gtt_offload", False))
        if _gtt_offload and _gtt_mb > 0:
            _effective_uma_mb = max(_carve_out_mb, _gtt_mb)
            vram_budget_pct = 0.95
            vram_os_reserve_mb = max(256, int(_effective_uma_mb * 0.15))
            vram_effective_mb = _effective_uma_mb
            log.info(
                "UMA APU budget (GTT OPT-IN ENABLED): carve_out=%d MB "
                "gtt=%d MB effective=%d MB reserve=%d MB",
                _carve_out_mb, _gtt_mb, _effective_uma_mb, vram_os_reserve_mb,
            )
        else:
            vram_budget_pct = 0.95
            vram_os_reserve_mb = 200
            vram_effective_mb = 0
            log.info(
                "UMA APU budget (carve-out ONLY): carve_out=%d MB "
                "gtt=%d MB (not used) reserve=%d MB",
                _carve_out_mb, _gtt_mb, vram_os_reserve_mb,
            )
    elif _carve_out_mb > 0:
        vram_budget_pct = 0.95
        vram_os_reserve_mb = 200
        vram_effective_mb = 0
    else:
        vram_budget_pct = 0.0
        vram_os_reserve_mb = 0
        vram_effective_mb = 0

    # ── Batch sizes and n_parallel ──
    _budget_basis_mb = vram_effective_mb if vram_effective_mb > 0 else _carve_out_mb
    if has_accel:
        n_batch  = 2048 if _budget_basis_mb >= 6144 else 1024
        n_ubatch = 512
        if _budget_basis_mb >= 12288:   n_parallel = 4
        elif _budget_basis_mb >= 6144:  n_parallel = 2
        else:                            n_parallel = 1
    else:
        n_batch, n_ubatch = 512, 128
        n_parallel = 1

    cpu_thread_ratio = 1.0 if is_apu else 0.85

    _hw_cfg = hardware_config or {}
    _force_no_think_cfg = _hw_cfg.get("force_no_think")
    if _force_no_think_cfg is not None:
        _force_no_think = bool(_force_no_think_cfg)
    else:
        _force_no_think = (ram_total_mb > 0 and ram_total_mb < 8192)

    return HardwareProfile(
        name=_canonical_name(_carve_out_mb, is_apu, bool(gpus)),
        vram_budget_pct=vram_budget_pct,
        vram_os_reserve_mb=vram_os_reserve_mb,
        vram_effective_mb=vram_effective_mb,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        context_length=base_ctx,
        ram_target=0.75,
        kv_cache_quant=kv_quant,
        cpu_thread_ratio=cpu_thread_ratio,
        target_tps=target_tps,
        target_tps_source=target_tps_source,
        flash_attn=True,
        n_parallel=n_parallel,
        force_no_think=_force_no_think,
        speculative_decoding=False,
        context_vs_layers_priority=("layers" if is_apu else "balanced"),
        use_mlock=False,
    )


def detect_profile_from_system(
    system_info: Any,
    hardware_config: Optional[Dict[str, Any]] = None,
    model_bytes: int = 0,
    model_meta: Optional[Dict[str, Any]] = None,
    measured_decode_tps: float = 0.0,
) -> HardwareProfile:
    """Preferred entry point when a full SystemInfo is available.

    Applies config overrides the same way `detect_profile` does so the
    override contract is identical. `measured_decode_tps` is forwarded
    to compute_profile; the resulting target_tps_source makes the
    origin visible in the startup log.
    """
    hw_cfg = hardware_config or {}
    base = compute_profile(
        system_info,
        model_bytes=model_bytes,
        model_meta=model_meta,
        hardware_config=hw_cfg,
        measured_decode_tps=measured_decode_tps,
    )
    profile = _apply_global_overrides(base, hw_cfg)
    profile = _apply_profile_specific_overrides(profile, base.name, hw_cfg)
    log.info(
        "Computed profile: name=%s target_tps=%.2f [%s] "
        "n_batch=%d/%d ctx=%d kv=%s n_parallel=%d spec=%s "
        "priority=%s mlock=%s "
        "[ram_ch=%d bw_meas=%.1f bw_theo=%.1f gtt=%d eff_vram=%d]",
        profile.name, profile.target_tps, profile.target_tps_source,
        profile.n_batch, profile.n_ubatch,
        profile.context_length, profile.kv_cache_quant, profile.n_parallel,
        profile.speculative_decoding, profile.context_vs_layers_priority,
        profile.use_mlock,
        int(getattr(system_info, "ram_channels", 0) or 0),
        float(getattr(system_info, "ram_bandwidth_gbps_measured", 0.0) or 0.0),
        float(getattr(system_info, "ram_bandwidth_gbps_theoretical", 0.0) or 0.0),
        int(getattr(system_info, "vram_gtt_total_mb", 0) or 0),
        profile.vram_budget_mb(int(getattr(system_info, "vram_gtt_total_mb", 0) or 0) or 2048)
    )
    return profile


# -----------------------------------------------------------------------------
# Profile Detection — public API
# -----------------------------------------------------------------------------
def detect_profile(
        gpus: List[Any],
        ram_total_mb: int = 0,
        force_refresh: bool = False,
        hardware_config: Optional[Dict[str, Any]] = None,
        system_info: Optional[Any] = None,
        measured_decode_tps: float = 0.0,
) -> HardwareProfile:
    """Return the best HardwareProfile for the given GPU list.

    Args:
        gpus: List of GPUInfo objects.
        ram_total_mb: Total system RAM in MB (kept for API compat).
        force_refresh: If True, re‑detect the base profile (ignore cache).
        hardware_config: The 'hardware' section of config.yaml.
        system_info: Full SystemInfo, if available. When provided, the
            preferred `detect_profile_from_system` path is used, which
            derives the profile from measured facts rather than tiers.
        measured_decode_tps: A directly-measured decode rate from
            tools/bench_decode.py. When > 0 it becomes target_tps verbatim
            and the roofline estimator is skipped. This is the number
            that should be advertised (§7.3).
    """
    global _detected_base_profile_cache
    hw_cfg = hardware_config or {}

    # Preferred path: full SystemInfo → measured-profile computation.
    if system_info is not None and not force_refresh:
        try:
            return detect_profile_from_system(
                system_info, hardware_config,
                measured_decode_tps=measured_decode_tps,
            )
        except Exception as exc:
            log.warning(
                "detect_profile_from_system failed (%s); falling back to tier lookup",
                exc,
            )

    with _cache_lock:
        if force_refresh or _detected_base_profile_cache is None:
            supported = [g for g in gpus if getattr(g, "supported", False)]
            if not supported:
                log.info("HardwareProfile: no supported GPU → cpu_only")
                base = _BASE_PROFILES["cpu_only"]
            else:
                total_vram = sum(g.vram_total_mb for g in supported)
                is_apu = any(_is_apu(g, total_vram) for g in supported)

                if is_apu:
                    key = "apu_4gb" if total_vram > 3072 else "apu_2gb"
                elif total_vram < 5120:
                    key = "gpu_4gb"
                elif total_vram < 7168:
                    key = "gpu_6gb"
                elif total_vram < 10240:
                    key = "gpu_8gb"
                elif total_vram < 14336:
                    key = "gpu_12gb"
                elif total_vram < 20480:
                    key = "gpu_16gb"
                elif total_vram < 32768:
                    key = "gpu_24gb"
                elif total_vram < 49152:
                    key = "gpu_32gb"
                elif total_vram < 65536:
                    key = "gpu_48gb"
                elif total_vram < 98304:
                    key = "gpu_80gb"
                else:
                    key = "gpu_192gb+"

                base = _BASE_PROFILES[key]
                log.info(
                    "HardwareProfile: vram=%dMB apu=%s gpu_count=%d → %s "
                    "(tier default target_tps=%.0f n_batch=%d ctx=%d "
                    "kv=%s n_parallel=%d spec=%s priority=%s mlock=%s)",
                    total_vram,
                    is_apu,
                    len(supported),
                    key,
                    base.target_tps,
                    base.n_batch,
                    base.context_length,
                    base.kv_cache_quant,
                    base.n_parallel,
                    base.speculative_decoding,
                    base.context_vs_layers_priority,
                    base.use_mlock,
                )
            _detected_base_profile_cache = base
        else:
            base = _detected_base_profile_cache

    # If we have a measurement, replace the tier-default target_tps.
    # Applied OUTSIDE the cache so the same cached base can serve
    # different measured values on successive calls (unusual but
    # correct — the cache holds the tier default only).
    base = _apply_measured_tps(base, measured_decode_tps, "bench_measured")

    profile = _apply_global_overrides(base, hw_cfg)
    profile = _apply_profile_specific_overrides(profile, base.name, hw_cfg)
    return profile


def get_profile(
    name: str,
    hardware_config: Optional[Dict[str, Any]] = None,
    measured_decode_tps: float = 0.0,
) -> HardwareProfile:
    """Return a profile by name, with optional overrides from hardware_config.

    `measured_decode_tps` > 0 replaces the tier default target_tps; the
    resulting `target_tps_source` reports `bench_measured` so the origin
    is not hidden behind a tier name. User config overrides still apply
    on top and, if they set `target_tps`, become the reported source.
    """
    base = _BASE_PROFILES[name]
    if measured_decode_tps > 0:
        base = _apply_measured_tps(base, measured_decode_tps, "bench_measured")
    hw_cfg = hardware_config or {}
    profile = _apply_global_overrides(base, hw_cfg)
    return _apply_profile_specific_overrides(profile, name, hw_cfg)


def list_profile_keys() -> List[str]:
    return list(_BASE_PROFILES.keys())


def clear_detection_cache() -> None:
    """Clear the cached base hardware profile, forcing re‑detection next call."""
    global _detected_base_profile_cache
    with _cache_lock:
        _detected_base_profile_cache = None
        log.info("Detection cache cleared")
