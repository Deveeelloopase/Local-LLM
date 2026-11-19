from __future__ import annotations

"""
hal/gpu_analyzer.py — Production‑Ready GPU Analyzer
=======================================================================
Deep GPU capability assessment: compute throughput profiles, PCIe
bandwidth estimation, thermal monitoring, and memory hierarchy
analysis — without requiring PyTorch or CUDA.

All parameters are provided via the `config` dict (or a
`GpuAnalyzerConfig` model). Sensible defaults are used when
config is omitted.

Config-driven: every numeric default comes from `self._cfg`.

version = 1.0.0
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class ComputeProfile:
    gpu_index: int
    peak_tflops_fp32: float
    peak_tflops_fp16: float
    peak_tops_int8: float
    memory_bw_gbps: float
    cache_l2_mb: float
    warp_size: int
    max_threads_sm: int

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (doctor hook)."""
        return {
            "gpu_index": self.gpu_index,
            "peak_tflops_fp32": self.peak_tflops_fp32,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "peak_tops_int8": self.peak_tops_int8,
            "memory_bw_gbps": self.memory_bw_gbps,
            "cache_l2_mb": self.cache_l2_mb,
            "warp_size": self.warp_size,
            "max_threads_sm": self.max_threads_sm,
        }


@dataclass
class ThermalStatus:
    """Thermal snapshot for one GPU.

    `temperature_c` is `Optional[float]`: `None` means "no sensor present
    or reading unavailable" — distinct from `0.0` (a real 0 °C reading,
    only meaningful for very cold climates). Every downstream threshold
    check must treat `None` as "unknown", not as "cold".
    """
    temperature_c: Optional[float]
    fan_speed_pct: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    throttling: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (doctor hook)."""
        return {
            "temperature_c": self.temperature_c,
            "fan_speed_pct": self.fan_speed_pct,
            "power_draw_w": self.power_draw_w,
            "power_limit_w": self.power_limit_w,
            "throttling": self.throttling,
        }


# -----------------------------------------------------------------------------
# Subprocess Helper
# -----------------------------------------------------------------------------
def _run(cmd: List[str], timeout: int = 5) -> Tuple[int, str]:
    """Run a subprocess with timeout; never raises. Returns (rc, stdout)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return r.returncode, r.stdout.strip()
    except Exception:
        return -1, ""


# =============================================================================
# GPU Analyzer
# =============================================================================
class GPUAnalyzer:
    """
    Analyses GPU capabilities using known specs and sysfs probing.

    All parameters are injected via the `config` argument.
    """

    # Base static tables (hardware specs – not configurable)
    _FP32: Dict[str, float] = {
        "H100": 67.0, "A100": 19.5,
        "RTX 4090": 82.6, "RTX 4080": 48.7, "RTX 4070 TI": 40.1,
        "RTX 4070": 29.1, "RTX 4060": 15.1,
        "RTX 3090": 35.6, "RTX 3080": 29.8, "RTX 3070": 20.3,
        "RTX 3060": 12.7, "RTX 2080 TI": 13.4,
        "RX 7900 XTX": 61.4, "RX 7900 XT": 51.5,
        "RX 6900 XT": 23.0, "RX 6800 XT": 20.7,
    }
    _MEM_BW: Dict[str, float] = {
        "H100": 3350.0, "A100": 2039.0,
        "RTX 4090": 1008.0, "RTX 4080": 717.0,
        "RTX 3090": 936.2, "RTX 3080": 760.3,
        "RTX 3070": 448.0, "RTX 3060": 360.0,
        "RX 7900 XTX": 960.0, "RX 7900 XT": 800.0,
        "RX 6900 XT": 512.0,
    }
    _L2: Dict[str, float] = {
        "RTX 4090": 72.0, "RTX 4080": 64.0, "RTX 4070 TI": 48.0,
        "RTX 3090": 6.0, "RTX 3080": 5.0,
        "A100": 40.0, "H100": 50.0,
    }

    def __init__(
        self,
        gpu_infos: List,
        *,
        config: Optional[Dict] = None,
        system_info: Optional[Any] = None,
    ) -> None:
        """
        Args:
            gpu_infos: List of GPUInfo objects (from HardwareDetector).
            config: Optional config dict (or GpuAnalyzerConfig) to override defaults.
            system_info: Optional pre-detected SystemInfo. When provided and it
                carries a positive `ram_bandwidth_gbps_measured`, that number
                is used directly — no fresh STREAM probe. This is how
                api_server will hand over the already-measured
                bandwidth so `/v1/gpu/analyze` and `/v1/system` agree.
        """
        self.gpu_infos = gpu_infos
        self._system_info = system_info

        if config is None:
            config = {}
        try:
            from local_llm_core.hal.config_loader import GpuAnalyzerConfig
            if isinstance(config, dict):
                cfg_model = GpuAnalyzerConfig(**config)
                self._cfg = cfg_model.model_dump()
            else:
                self._cfg = config.model_dump()
        except Exception as exc:
            log.warning("GPU analyzer config validation failed: %s; using defaults", exc)
            self._cfg = {}

        self._cmd_timeout = int(self._cfg.get("cmd_timeout_s", 8))
        self._thermal_throttle_temp = float(self._cfg.get("thermal_throttle_temp_c", 85.0))
        self._default_ram_bw_gbps = float(self._cfg.get("default_ram_bw_gbps", 50.0))
        self._default_fp32_tflops = float(self._cfg.get("default_fp32_tflops", 10.0))
        self._default_mem_bw_gbps = float(self._cfg.get("default_mem_bw_gbps", 200.0))
        self._default_l2_cache_mb = float(self._cfg.get("default_l2_cache_mb", 4.0))
        self._model_overrides = self._cfg.get("model_overrides", {})

        # cache the STREAM probe result. Without this,
        # `_estimate_ram_bw()` runs the 128 MB × 3 × 5-iteration STREAM
        # probe on every call — a different number each time (scheduler
        # jitter, thermal drift, competing processes) — which violates
        # ("one number per physical fact") AND makes
        # `/v1/gpu/analyze` measurably expensive on every poll.
        # Populated on first call; returned verbatim thereafter.
        self._ram_bw_cache: Optional[Tuple[float, Dict[str, Any]]] = None

        self._fp32_table = dict(self._FP32)
        self._mem_bw_table = dict(self._MEM_BW)
        self._l2_table = dict(self._L2)
        for model_name, overrides in self._model_overrides.items():
            key = model_name.upper()
            if "fp32_tflops" in overrides:
                self._fp32_table[key] = float(overrides["fp32_tflops"])
            if "mem_bw_gbps" in overrides:
                self._mem_bw_table[key] = float(overrides["mem_bw_gbps"])
            if "l2_cache_mb" in overrides:
                self._l2_table[key] = float(overrides["l2_cache_mb"])

    # -------------------------------------------------------------------------
    # Compute Profiles
    # -------------------------------------------------------------------------
    def compute_profiles(self) -> List[ComputeProfile]:
        profiles = []
        for gpu in self.gpu_infos:
            name = gpu.name
            fp32 = self._lookup(name, self._fp32_table, self._default_fp32_tflops)
            bw = self._lookup(name, self._mem_bw_table, self._default_mem_bw_gbps)
            l2 = self._lookup(name, self._l2_table, self._default_l2_cache_mb)

            cc = getattr(gpu, "compute_capability", None) or ""
            major = int(cc.split(".")[0]) if cc and cc[0].isdigit() else 7
            fp16_mult = 2.0 if major >= 8 else 1.0
            int8_mult = 4.0 if major >= 8 else 2.0

            # `getattr(vendor, "value", "")` returns the int
            # `auto()` value — never the string "amd". Every GPU previously
            # got warp=32 (including AMD, whose real warp size is 64).
            # Check `.name` instead (GPUVendor.AMD.name == "AMD").
            vendor = getattr(gpu, "vendor", None)
            vendor_name = (getattr(vendor, "name", "") or "").lower()
            warp = 64 if vendor_name == "amd" else 32

            profiles.append(ComputeProfile(
                gpu_index=gpu.index if hasattr(gpu, "index") else 0,
                peak_tflops_fp32=fp32,
                peak_tflops_fp16=fp32 * fp16_mult * 2,
                peak_tops_int8=fp32 * int8_mult * 4,
                memory_bw_gbps=bw,
                cache_l2_mb=l2,
                warp_size=warp,
                max_threads_sm=1536 if major >= 8 else 1024,
            ))
        return profiles

    # -------------------------------------------------------------------------
    # Thermal Monitoring
    # -------------------------------------------------------------------------
    def thermal_status(self, gpu_index: int = 0) -> ThermalStatus:
        if not self.gpu_infos:
            # None, not 0.0 — no GPU is not "0 °C".
            return ThermalStatus(None, None, None, None, False)

        gpu = next(
            (
                g for g in self.gpu_infos
                if (g.index if hasattr(g, "index") else 0) == gpu_index
            ),
            self.gpu_infos[0],
        )
        vendor = getattr(getattr(gpu, "vendor", None), "name", "").lower()

        if vendor == "nvidia":
            return self._nvidia_thermal(gpu_index)
        if vendor == "amd":
            return self._amd_thermal(gpu_index)
        return ThermalStatus(None, None, None, None, False)

    def _nvidia_thermal(self, idx: int) -> ThermalStatus:
        rc, out = _run([
            "nvidia-smi", f"--id={idx}",
            "--query-gpu=temperature.gpu,fan.speed,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ], timeout=self._cmd_timeout)
        if rc != 0 or not out:
            return ThermalStatus(None, None, None, None, False)

        parts = [x.strip() for x in out.split(",")]

        def _f(v: str) -> Optional[float]:
            # N/A on new drivers used to silently become 0.0
            # via `_f(parts[0]) or 0.0`. Return None instead — the caller
            # (threshold check below) treats None as "unknown sensor" and
            # does not falsely flag either hot or cold.
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        temp = _f(parts[0]) if len(parts) > 0 else None
        fan = _f(parts[1]) if len(parts) > 1 else None
        draw = _f(parts[2]) if len(parts) > 2 else None
        limit = _f(parts[3]) if len(parts) > 3 else None

        throttle = False
        if temp is not None and temp >= self._thermal_throttle_temp:
            throttle = True
        if draw is not None and limit is not None and limit > 0:
            if draw >= limit * 0.97:
                throttle = True

        return ThermalStatus(temp, fan, draw, limit, throttle)

    def _amd_thermal(self, idx: int) -> ThermalStatus:
        hwmon = Path(f"/sys/class/drm/card{idx}/device/hwmon")
        if hwmon.exists():
            for hw in hwmon.iterdir():
                tf = hw / "temp1_input"
                if tf.exists():
                    try:
                        temp = int(tf.read_text().strip()) / 1000.0
                    except Exception:
                        # fix: None, not 0.0 — a failed read is not 0 °C.
                        return ThermalStatus(None, None, None, None, False)
                    return ThermalStatus(
                        temp, None, None, None, temp >= self._thermal_throttle_temp,
                    )
        # No hwmon → no sensor. None, not 0.0
        return ThermalStatus(None, None, None, None, False)

    # -------------------------------------------------------------------------
    # RAM bandwidth (reconciled with hardware_detector STREAM probe)
    # -------------------------------------------------------------------------
    def _estimate_ram_bw(self) -> Tuple[float, Dict[str, Any]]:
        """Return (gbps, probe_metadata) — cached across calls.

        Source priority (one number per physical fact):
          1. `system_info.ram_bandwidth_gbps_measured > 0` → use it.
          2. Run `hardware_detector._measure_ram_bandwidth_gbps()` — the
             STREAM best-of-5 triad.
          3. `cfg.default_ram_bw_gbps` → config-provided fallback.
          4. `self._default_ram_bw_gbps` → hardcoded fallback.

        the result is cached after the first successful
        resolution. Without caching, every call re-runs the STREAM probe
        (128 MB × 3 arrays × 5 iterations ≈ 40-100ms on this APU) and
        returns a slightly different number each time due to scheduler
        jitter. That violated (one number per physical fact) and
        made `/v1/gpu/analyze` measurably expensive per request.
        """
        if self._ram_bw_cache is not None:
            return self._ram_bw_cache

        result: Tuple[float, Dict[str, Any]]

        # 1. Prefer SystemInfo snapshot
        if self._system_info is not None:
            bw = float(
                getattr(self._system_info, "ram_bandwidth_gbps_measured", 0.0) or 0.0
            )
            if bw > 0:
                probe = dict(
                    getattr(self._system_info, "ram_bandwidth_probe", {}) or {}
                )
                probe.setdefault("source", "system_info")
                result = (bw, probe)
                self._ram_bw_cache = result
                return result

        # 2. Run the same STREAM probe hardware_detector uses
        try:
            from local_llm_core.hal.hardware_detector import (
                _measure_ram_bandwidth_gbps,
            )
            gbps, meta = _measure_ram_bandwidth_gbps()
            if gbps > 0:
                probe = dict(meta)
                probe.setdefault("source", "stream_probe")
                result = (gbps, probe)
                self._ram_bw_cache = result
                return result
        except Exception as exc:
            log.debug("STREAM probe unavailable: %s", exc)

        # 3. Config default
        cfg_default = self._cfg.get("default_ram_bw_gbps")
        if cfg_default is not None:
            result = (float(cfg_default), {
                "source": "config_default",
                "method": "config",
            })
            self._ram_bw_cache = result
            return result

        # 4. Hardcoded default
        result = (float(self._default_ram_bw_gbps), {
            "source": "config_default",
            "method": "hardcoded_default",
        })
        self._ram_bw_cache = result
        return result

    # -------------------------------------------------------------------------
    # Memory Hierarchy
    # -------------------------------------------------------------------------
    def memory_hierarchy(self) -> Dict:
        profiles = self.compute_profiles()
        tiers = []
        for prof in profiles:
            gpu = (
                self.gpu_infos[prof.gpu_index]
                if prof.gpu_index < len(self.gpu_infos)
                else self.gpu_infos[0]
            )
            tiers.append({
                # Representative values — the analyzer does not probe
                # per-SM register / shared / L1 sizes; these are
                # documentation-grade constants, not measurements.
                "gpu_index": prof.gpu_index,
                "registers_kb": 256,
                "shared_kb": 48,
                "l1_kb": 128,
                "l2_mb": prof.cache_l2_mb,
                "vram_mb": gpu.vram_total_mb,
                "vram_bw_gbps": prof.memory_bw_gbps,
                "pcie_bw_gbps": self._pcie_bandwidth_theoretical(prof.gpu_index),
            })
        vm = psutil.virtual_memory()
        ram_bw, ram_bw_probe = self._estimate_ram_bw()
        return {
            "tiers": tiers,
            "host_ram_mb": vm.total >> 20,
            "host_ram_bw_gbps": ram_bw,
            "host_ram_bw_probe": ram_bw_probe,
        }

    # -------------------------------------------------------------------------
    # PCIe Bandwidth
    # -------------------------------------------------------------------------
    def _pcie_bandwidth_theoretical(self, gpu_index: int) -> float:
        gpu = next(
            (
                g for g in self.gpu_infos
                if (g.index if hasattr(g, "index") else 0) == gpu_index
            ),
            None,
        )
        if gpu is None:
            return self._cfg.get("default_pcie_bw_gbps", 16.0)

        gen = getattr(gpu, "pcie_gen", None) or self._cfg.get("default_pcie_gen", 4)
        lanes = (
            getattr(gpu, "pcie_lanes", None)
            or getattr(gpu, "pcie_width", None)
            or self._cfg.get("default_pcie_lanes", 16)
        )

        bw_per_lane = {3: 1.0, 4: 2.0, 5: 4.0}.get(gen, 1.0)
        return bw_per_lane * lanes

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _lookup(name: str, table: Dict, default: float) -> float:
        n = name.upper()
        for key, val in table.items():
            if key.upper() in n:
                return val
        return default

    # -------------------------------------------------------------------------
    # Full analysis (this is what /v1/gpu/analyze calls)
    # -------------------------------------------------------------------------
    def full_analysis(self) -> Dict[str, Any]:
        """One-shot combined report — the natural payload for /v1/gpu/analyze.

        Combines:
          • compute_profiles  — per-GPU TFLOPS / bandwidth / warp / SM limits
          • thermal_status    — per-GPU temp / fan / power / throttle
          • memory_hierarchy  — per-GPU VRAM + host RAM bandwidth (with probe)
          • ram_bandwidth_gbps / probe — top-level for convenience

        JSON-safe: every dataclass is converted via `.to_dict()`, every
        enum stringified, every value is a JSON primitive.

        Idempotent — pure reads from `self.gpu_infos` and `self._cfg`.
        Never raises; on any individual sub-failure the corresponding
        section is populated with empty list / `None`.
        """
        try:
            profiles = self.compute_profiles()
        except Exception as exc:
            log.warning("compute_profiles failed: %s", exc)
            profiles = []

        try:
            thermals = [
                self.thermal_status(p.gpu_index).to_dict() for p in profiles
            ]
        except Exception as exc:
            log.warning("thermal_status failed: %s", exc)
            thermals = []

        try:
            mem = self.memory_hierarchy()
        except Exception as exc:
            log.warning("memory_hierarchy failed: %s", exc)
            mem = {}

        ram_bw = (
            mem.get("host_ram_bw_gbps", 0.0)
            if isinstance(mem, dict) else 0.0
        )
        ram_bw_probe = (
            mem.get("host_ram_bw_probe", {})
            if isinstance(mem, dict) else {}
        )

        return {
            "gpu_count": len(self.gpu_infos),
            "compute_profiles": [p.to_dict() for p in profiles],
            "thermal_status": thermals,
            "memory_hierarchy": mem,
            "ram_bandwidth_gbps": ram_bw,
            "ram_bandwidth_probe": ram_bw_probe,
        }
