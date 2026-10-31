from __future__ import annotations

"""
hal/gpu_analyzer.py — Production‑Ready GPU Analyzer
===================================================
v1.0.0 – Config injected via constructor; no file I/O.

Deep GPU capability assessment: compute throughput profiles,
PCIe bandwidth estimation, thermal monitoring, and memory
hierarchy analysis — without requiring PyTorch or CUDA.

All parameters are provided via the `config` dict (or a
`GpuAnalyzerConfig` model). Sensible defaults are used when
config is omitted.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class ThermalStatus:
    temperature_c: float
    fan_speed_pct: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    throttling: bool


# -----------------------------------------------------------------------------
# Subprocess Helper
# -----------------------------------------------------------------------------
def _run(cmd: List[str], timeout: int = 5) -> tuple[int, str]:
    """Run a subprocess with timeout; never raises. Returns (rc, stdout)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
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
        "RTX 3090": 6.0,  "RTX 3080": 5.0,
        "A100": 40.0, "H100": 50.0,
    }

    def __init__(
        self,
        gpu_infos: List,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            gpu_infos: List of GPUInfo objects (from HardwareDetector).
            config: Optional config dict (or GpuAnalyzerConfig) to override defaults.
        """
        self.gpu_infos = gpu_infos

        # ---- normalise config ------------------------------------------------
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
            self._cfg = {}  # will rely on hardcoded defaults below

        # Extract parameters with precedence: config > hardcoded defaults
        self._cmd_timeout = int(self._cfg.get("cmd_timeout_s", 8))
        self._thermal_throttle_temp = float(self._cfg.get("thermal_throttle_temp_c", 85.0))
        self._default_ram_bw_gbps = float(self._cfg.get("default_ram_bw_gbps", 50.0))
        self._default_fp32_tflops = float(self._cfg.get("default_fp32_tflops", 10.0))
        self._default_mem_bw_gbps = float(self._cfg.get("default_mem_bw_gbps", 200.0))
        self._default_l2_cache_mb = float(self._cfg.get("default_l2_cache_mb", 4.0))
        self._model_overrides = self._cfg.get("model_overrides", {})

        # Apply per‑model overrides to the static lookup tables (shallow copy)
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

            vendor = getattr(gpu, "vendor", None)
            warp = 64 if vendor and getattr(vendor, "value", "") == "amd" else 32

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
            return ThermalStatus(0.0, None, None, None, False)

        gpu = next((g for g in self.gpu_infos
                    if (g.index if hasattr(g, "index") else 0) == gpu_index),
                   self.gpu_infos[0])
        vendor = getattr(getattr(gpu, "vendor", None), "value", "")

        if vendor == "nvidia":
            return self._nvidia_thermal(gpu_index)
        if vendor == "amd":
            return self._amd_thermal(gpu_index)
        return ThermalStatus(0.0, None, None, None, False)

    def _nvidia_thermal(self, idx: int) -> ThermalStatus:
        rc, out = _run([
            "nvidia-smi", f"--id={idx}",
            "--query-gpu=temperature.gpu,fan.speed,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ], timeout=self._cmd_timeout)
        if rc != 0 or not out:
            return ThermalStatus(0.0, None, None, None, False)

        parts = [x.strip() for x in out.split(",")]
        def _f(v): return float(v) if v.replace(".", "").isdigit() else None

        temp = _f(parts[0]) or 0.0
        fan = _f(parts[1]) if len(parts) > 1 else None
        draw = _f(parts[2]) if len(parts) > 2 else None
        limit = _f(parts[3]) if len(parts) > 3 else None

        throttle = bool(draw and limit and draw >= limit * 0.97) or temp >= self._thermal_throttle_temp
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
                        return ThermalStatus(0.0, None, None, None, False)
                    return ThermalStatus(temp, None, None, None, temp >= self._thermal_throttle_temp)
        return ThermalStatus(0.0, None, None, None, False)

    # -------------------------------------------------------------------------
    # Memory Hierarchy
    # -------------------------------------------------------------------------
    def memory_hierarchy(self) -> Dict:
        profiles = self.compute_profiles()
        tiers = []
        for prof in profiles:
            gpu = self.gpu_infos[prof.gpu_index] if prof.gpu_index < len(self.gpu_infos) else self.gpu_infos[0]
            tiers.append({
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
        return {
            "tiers": tiers,
            "host_ram_mb": vm.total >> 20,
            "host_ram_bw_gbps": self._estimate_ram_bw(),
        }

    # -------------------------------------------------------------------------
    # PCIe Bandwidth
    # -------------------------------------------------------------------------
    def _pcie_bandwidth_theoretical(self, gpu_index: int) -> float:
        gpu = next((g for g in self.gpu_infos
                    if (g.index if hasattr(g, "index") else 0) == gpu_index), None)
        if gpu is None:
            return self._cfg.get("default_pcie_bw_gbps", 16.0)

        gen = getattr(gpu, "pcie_gen", None) or self._cfg.get("default_pcie_gen", 4)
        lanes = (getattr(gpu, "pcie_lanes", None) or
                 getattr(gpu, "pcie_width", None) or
                 self._cfg.get("default_pcie_lanes", 16))

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

    def _estimate_ram_bw(self) -> float:
        """Estimate system RAM bandwidth via a quick numpy benchmark."""
        try:
            import numpy as np
            size = 128 * 1024 * 1024
            arr = np.empty(size // 8, dtype=np.float64)
            t0 = time.perf_counter()
            arr[:] = 1.0
            elapsed = max(time.perf_counter() - t0, 1e-9)
            return round((size / 1024 ** 3) / elapsed, 1)
        except Exception:
            return self._cfg.get("default_ram_bw_gbps", self._default_ram_bw_gbps)
