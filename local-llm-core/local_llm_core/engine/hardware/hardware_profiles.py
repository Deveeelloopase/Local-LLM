from __future__ import annotations

# engine/hardware/hardware_profiles.py — Production‑Ready Hardware Profiles
# ==========================================================================
# v1.0.0 – Pure library: configuration overrides injected, no file I/O.
#
# Detects system hardware and returns an optimal HardwareProfile for inference.
# Overrides are applied from a provided hardware configuration dict.
# Profile detection is cached (base profile only) to avoid repeated computation,
# while overrides are re-applied each call to honour config changes.
# ==========================================================================

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


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
    context_vs_layers_priority: str = "balanced"   # "context", "layers", or "balanced"

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

    # -------------------------------------------------------------------------
    # Computed helpers
    # -------------------------------------------------------------------------
    def vram_budget_mb(self, vram_total_mb: int) -> int:
        total = self.vram_effective_mb if self.vram_effective_mb > 0 else vram_total_mb
        raw = int(total * self.vram_budget_pct)
        return max(0, raw - self.vram_os_reserve_mb)

    def kv_budget_mb(self, vram_total_mb: int, model_vram_mb: int) -> int:
        return max(0, self.vram_budget_mb(vram_total_mb) - model_vram_mb)

    # Compute effective context length from model and available memory
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

        Args:
            model_n_ctx_train: The model's native training context length.
            vram_total_mb: Total VRAM available (MB).
            model_vram_mb: VRAM already consumed by the model weights (MB).
            kv_cache_bytes_per_token: Estimated bytes per token for the KV cache.
                                      This is model‑specific (depends on n_layers,
                                      n_heads, head_dim, quant). Default 1.0 as
                                      a conservative fallback.

        Returns:
            Effective context length, never less than the static `context_length`
            defined in this profile.
        """
        kv_budget_mb = self.kv_budget_mb(vram_total_mb, model_vram_mb)
        # Convert budget to bytes
        headroom_bytes = kv_budget_mb * 1024 * 1024
        tokens_from_mem = int(headroom_bytes // kv_cache_bytes_per_token) if kv_cache_bytes_per_token > 0 else 0
        effective = min(model_n_ctx_train, tokens_from_mem)
        # Use the static value as a floor/fallback
        return max(self.context_length, effective)

    def n_threads(self) -> int:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
        except (ImportError, Exception):
            physical = None
        if physical is None:
            import os
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
        }
        kwargs.update(overrides)
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
        force_no_think=True,
        speculative_decoding=True,
        context_vs_layers_priority="balanced",
        use_mlock=True,   # APU tier: keep model weights resident in RAM
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
        force_no_think=True,
        speculative_decoding=True,
        context_vs_layers_priority="balanced",
        use_mlock=True,   # APU tier: keep model weights resident in RAM
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
        cpu_thread_ratio=1.0,
        target_tps=25.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=True,
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
        cpu_thread_ratio=1.0,
        target_tps=40.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=True,
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
        cpu_thread_ratio=1.0,
        target_tps=60.0,
        flash_attn=True,
        n_parallel=2,
        force_no_think=False,
        speculative_decoding=True,
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
        cpu_thread_ratio=1.0,
        target_tps=80.0,
        flash_attn=True,
        n_parallel=3,
        force_no_think=False,
        speculative_decoding=True,
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
        cpu_thread_ratio=1.0,
        target_tps=100.0,
        flash_attn=True,
        n_parallel=4,
        force_no_think=False,
        speculative_decoding=True,
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
        cpu_thread_ratio=1.0,
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
        cpu_thread_ratio=1.0,
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
        cpu_thread_ratio=1.0,
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
        cpu_thread_ratio=1.0,
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
        cpu_thread_ratio=1.0,
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
        cpu_thread_ratio=0.75,
        target_tps=3.0,
        flash_attn=False,
        n_parallel=1,
        force_no_think=True,
        speculative_decoding=True,
        context_vs_layers_priority="layers",  # On CPU, maximise layers (context is always CPU anyway)
    ),
}

# Legacy alias
_BASE_PROFILES["gpu_24gb+"] = _BASE_PROFILES["gpu_24gb"]


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


def compute_profile(
    system_info: Any,
    model_bytes: int = 0,
    model_meta: Optional[Dict[str, Any]] = None,
    hardware_config: Optional[Dict[str, Any]] = None,
) -> HardwareProfile:
    """Derive the profile from *measured* hardware rather than a tier table.

    Every field is either computed from a measured fact on `system_info`,
    or — when that fact is unavailable (0/empty) — falls back to the value
    the corresponding tier would have produced. On a machine where the new
    hardware_detector ran successfully, every numeric field below traces
    back to a measurement.

    The profile `name` is chosen so existing `config.yaml` overrides under
    `hardware_profiles.overrides.<name>` continue to apply.
    """
    import psutil

    gpus = [g for g in (system_info.gpus or []) if getattr(g, "supported", True)]
    vram_total_mb = sum(int(getattr(g, "vram_total_mb", 0) or 0) for g in gpus)

    is_apu = any(
        (getattr(g.memory_type, "value", str(g.memory_type)) or "").upper()
        in ("SHARED", "SHARED SYSTEM RAM", "UNIFIED MEMORY")
        for g in gpus
    )

    ram_total_mb = int(getattr(system_info, "ram_total_mb", 0) or 0)
    ram_free_mb  = int(getattr(system_info, "ram_free_mb", 0) or 0)

    # Bandwidth: prefer measured (captures single-vs-dual-channel), else
    # theoretical (from dmidecode speed × channels), else a conservative
    # default. All three feed the same downstream roofline math.
    #
    # Sanity: a measurement below 50% of theoretical is almost always
    # contention/throttling noise (compositor, thermal throttle, competing
    # processes during the ~200ms triad), not a real bandwidth ceiling.
    # Reject it and fall back to theory (which is what dmidecode told us).
    # If both are missing, use a conservative default rather than 0.
    bw_measured    = float(getattr(system_info, "ram_bandwidth_gbps_measured", 0.0) or 0.0)
    bw_theoretical = float(getattr(system_info, "ram_bandwidth_gbps_theoretical", 0.0) or 0.0)
    if bw_theoretical > 0 and (bw_measured <= 0 or bw_measured < bw_theoretical * 0.5):
        effective_bw_gbps = bw_theoretical
        _bw_source = "theory (rejected measured %.1f < 50%% of %.1f)" % (
            bw_measured, bw_theoretical,
        )
    elif bw_measured > 0:
        effective_bw_gbps = bw_measured
        _bw_source = "measurement"
    elif bw_theoretical > 0:
        effective_bw_gbps = bw_theoretical
        _bw_source = "theory (no measurement)"
    else:
        effective_bw_gbps = 25.0
        _bw_source = "conservative fallback"
    log.info("Bandwidth selection: %.1f GB/s (%s)", effective_bw_gbps, _bw_source)

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

    # ── target_tps from the roofline ──
    # bytes_per_token ≈ model_bytes for Q4_K_M (weights are the dominating
    # term for decode-bound workloads). If model_bytes is 0 the caller has
    # no model yet — use a tier-appropriate fallback.
    if model_bytes > 0:
        target_tps = 0.9 * effective_bw_gbps / (model_bytes / 1e9)
        target_tps = max(1.0, min(target_tps, 1000.0))
    else:
        target_tps = 8.0 if has_accel else 3.0

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
                "gtt=%d MB effective=%d MB reserve=%d MB — weights may "
                "be placed in system RAM addressable by the iGPU",
                _carve_out_mb, _gtt_mb, _effective_uma_mb, vram_os_reserve_mb,
            )
        else:
            # DEFAULT on APUs: plan against the hardware-enforced VRAM
            # aperture only. Anything beyond the carve-out is GTT, which
            # shares the CPU's memory bus. On single-channel APUs this
            # turns every decode step into a bus thrash and can hang the
            # iGPU's fence timer. Do not adopt GTT offload without a
            # measured win on this exact (GPU, kernel, model) triple —
            # see hardware_profiles.uma_gtt_offload: true to opt in.
            vram_budget_pct = 0.95
            vram_os_reserve_mb = 200
            vram_effective_mb = 0
            log.info(
                "UMA APU budget (carve-out ONLY): carve_out=%d MB "
                "gtt=%d MB (not used for planning) reserve=%d MB — "
                "offload limited to physical VRAM aperture",
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
    # Base decisions on the effective budget (GTT-inclusive for APUs,
    # unchanged for discrete GPUs).
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

    # ── Thread ratio ──
    cpu_thread_ratio = 1.0 if is_apu else 0.85

    return HardwareProfile(
        # Use carve-out for name so existing overrides under
        # hardware_profiles.overrides.apu_2gb (or apu_4gb) still apply.
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
        flash_attn=True,
        n_parallel=n_parallel,
        force_no_think=(ram_total_mb > 0 and ram_total_mb < 8192),
        speculative_decoding=False,   # opt-in only
        context_vs_layers_priority=("layers" if is_apu else "balanced"),
        use_mlock=False,
    )


def detect_profile_from_system(
    system_info: Any,
    hardware_config: Optional[Dict[str, Any]] = None,
    model_bytes: int = 0,
    model_meta: Optional[Dict[str, Any]] = None,
) -> HardwareProfile:
    """Preferred entry point when a full SystemInfo is available.

    Applies config overrides the same way `detect_profile` does so the
    override contract is identical.
    """
    hw_cfg = hardware_config or {}
    base = compute_profile(system_info, model_bytes=model_bytes, model_meta=model_meta, hardware_config=hw_cfg)
    profile = _apply_global_overrides(base, hw_cfg)
    profile = _apply_profile_specific_overrides(profile, base.name, hw_cfg)
    log.info(
        "Computed profile: name=%s target_tps=%.2f n_batch=%d/%d ctx=%d "
        "kv=%s n_parallel=%d spec=%s priority=%s mlock=%s "
        "[ram_ch=%d bw_meas=%.1f bw_theo=%.1f gtt=%d eff_vram=%d]",
        profile.name, profile.target_tps, profile.n_batch, profile.n_ubatch,
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
        system_info: Optional[Any] = None,  # NEW
) -> HardwareProfile:
    """
    Return the best HardwareProfile for the given GPU list.

    Args:
        gpus: List of GPUInfo objects.
        ram_total_mb: Total system RAM in MB (kept for API compat).
        force_refresh: If True, re‑detect the base profile (ignore cache).
        hardware_config: Dictionary representing the 'hardware' section
                         of the application config. Used for overrides.
                         If None, no overrides are applied.

    Returns:
        HardwareProfile with optimal settings and applied overrides.
    """
    global _detected_base_profile_cache
    hw_cfg = hardware_config or {}

    # Preferred path: if the caller handed us a full SystemInfo (which
    # carries measured bandwidth, RAM channels, and llama-cpp backend
    # list), derive the profile from measurements rather than tiers.
    if system_info is not None and not force_refresh:
        try:
            return detect_profile_from_system(system_info, hardware_config)
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
                    "(target_tps=%.0f n_batch=%d ctx=%d kv=%s n_parallel=%d spec=%s priority=%s mlock=%s)",
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

    # Apply overrides from provided config
    profile = _apply_global_overrides(base, hw_cfg)
    profile = _apply_profile_specific_overrides(profile, base.name, hw_cfg)
    return profile


def get_profile(name: str, hardware_config: Optional[Dict[str, Any]] = None) -> HardwareProfile:
    """Return a profile by name, with optional overrides from hardware_config."""
    base = _BASE_PROFILES[name]
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
        log.debug("Detection cache cleared")
