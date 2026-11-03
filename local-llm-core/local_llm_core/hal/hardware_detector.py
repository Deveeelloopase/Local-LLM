from __future__ import annotations

# hal/hardware_detector.py — Production‑Ready Hardware Detection
# ==============================================================
# v1.0.0 – Config injected via constructor; no filesystem reads.
#
# Detects system hardware (CPU, RAM, GPUs) with robust timeouts and
# graceful fallback to CPU‑only mode. Also reports the presence of
# custom kernels modules (cpu_optimizer, Vulkan, AMD compat layer)
# so that the kernels manager can make informed decisions.
#
# Key Features:
# - NVIDIA (nvidia‑smi), AMD (rocm‑smi, sysfs), Intel (sysfs) GPU detection.
# - Apple Silicon (MPS) and CPU fallback.
# - Configurable command timeouts and Vulkan ICD paths.
# - Global detection timeout with ThreadPoolExecutor.
# - Preferred backend ordering from config.
# - CPU model, cores, RAM info via psutil and /proc/cpuinfo.
# - Custom kernels module probing (cpu_optimizer, AMD compat, Vulkan).
# - Outputs SystemInfo with GPUInfo list and kernels module status.
# ==============================================================

import time
import json
import logging
import os
import platform
import re
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import psutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Default constants (used when no config is injected)
# -----------------------------------------------------------------------------
_DEFAULT_CMD_TIMEOUT = 5
_DEFAULT_DETECTION_TIMEOUT = 30.0
_DEFAULT_VULKAN_ICD_PATHS = [
    "/usr/share/vulkan/icd.d",
    "/etc/vulkan/icd.d",
    "/usr/lib/x86_64-linux-gnu/GL/vulkan/icd.d",
]
_DEFAULT_BACKEND_ORDER = [
    "cuda", "rocm", "vulkan", "opencl", "mps", "numba", "numpy", "cpu"
]
_DEFAULT_PREFERRED_BACKEND = "auto"

_AMD_APU_IDS: frozenset[str] = frozenset({
    "0x15d8", "0x15dd", "0x15d9",   # Picasso / Raven Ridge
    "0x1636", "0x164c",              # Renoir
    "0x1681", "0x15bf",              # Mendocino
    "0x164d", "0x1900",              # Phoenix / Hawk Point
    "0x13c0",                        # Barcelo
    "0x1506",                        # Rembrandt
    "0x1901", "0x1902",              # Strix Point
})

# =============================================================================
# Enums & Data Models
# =============================================================================
class GPUVendor(Enum):
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    APPLE = auto()
    QUALCOMM = auto()
    UNKNOWN = auto()


class MemoryType(Enum):
    GDDR5 = "GDDR5"
    GDDR6 = "GDDR6"
    GDDR6X = "GDDR6X"
    HBM2 = "HBM2"
    HBM2E = "HBM2E"
    HBM3 = "HBM3"
    LPDDR4 = "LPDDR4"
    LPDDR5 = "LPDDR5"
    SHARED = "Shared System RAM"
    UNIFIED = "Unified Memory"
    UNKNOWN = "UNKNOWN"


@dataclass
class GPUInfo:
    vendor: GPUVendor
    name: str
    vram_total_mb: int
    vram_free_mb: int
    memory_type: MemoryType
    driver_version: str
    compute_capability: Optional[str]
    cuda_cores: Optional[int]
    backend: str
    pcie_gen: Optional[int]
    pcie_lanes: Optional[int]
    bus_id: Optional[str]
    temperature_c: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    tdp_w: Optional[int]
    supported: bool = True
    fallback_reason: Optional[str] = None
    extra: dict = field(default_factory=dict)
    index: int = 0


@dataclass
class SystemInfo:
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_mb: int
    ram_free_mb: int
    os_name: str
    kernel_version: str
    arch: str
    gpus: List[GPUInfo] = field(default_factory=list)
    primary_gpu: Optional[GPUInfo] = None
    has_cuda: bool = False
    has_rocm: bool = False
    has_opencl: bool = False
    has_vulkan: bool = False
    has_numba: bool = False
    has_numpy: bool = True
    available_backends: List[str] = field(default_factory=list)
    recommended_backend: str = "cpu"
    kernel_modules: Dict[str, bool] = field(default_factory=dict)
    # ── Measured / derived facts (0/empty = not available on this platform) ──
    # Populated by the new helpers below. All are read-only facts about the
    # machine; nothing here implies a choice. The profile builder consumes
    # them to compute optimal settings — this is what makes the codebase
    # adaptive rather than tier-based.
    ram_channels: int = 0
    ram_bandwidth_gbps_theoretical: float = 0.0
    ram_bandwidth_gbps_measured: float = 0.0
    # Which backends the *installed* llama-cpp-python was compiled with.
    # Empty means llama-cpp-python is not importable or is CPU-only.
    llama_cpp_backends: List[str] = field(default_factory=list)
    llama_cpp_gpu_offload: bool = False
    # Unified-memory APUs expose their real usable VRAM as carve-out + GTT.
    # GTT is system RAM the iGPU can address through the Graphics Translation
    # Table. On Vega/Raven it is typically 60% of total system RAM. Without
    # this, the FLP planner treats a 13.8 GB UMA APU as a 2 GB dGPU and
    # leaves most of the model on the CPU path.
    vram_gtt_total_mb: int = 0

    @property
    def cpu_cores_logical(self) -> int:
        return self.cpu_threads

    @property
    def cpu_cores_physical(self) -> int:
        return self.cpu_cores

    @property
    def vram_total_mb(self) -> int:
        return self.primary_gpu.vram_total_mb if self.primary_gpu else 0

    @property
    def vram_free_mb(self) -> int:
        return self.primary_gpu.vram_free_mb if self.primary_gpu else 0


# =============================================================================
# Shared Utilities (with configurable timeouts)
# =============================================================================
def _run_with_kill(cmd: List[str], timeout: int = 3) -> tuple[int, str, str]:
    """Run command with guaranteed timeout (even on Windows)."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout.strip(), stderr.strip()
        except subprocess.TimeoutExpired:
            if platform.system() == "Windows":
                proc.kill()
            else:
                import signal
                proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=1)
            return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def _read_sysfs(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _safe_int(v: str) -> Optional[int]:
    try:
        return int(float(v.strip()))
    except (ValueError, AttributeError):
        return None


def _safe_float(v: str) -> Optional[float]:
    try:
        return float(v.strip())
    except (ValueError, AttributeError):
        return None


def _get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


# =============================================================================
# Per‑Vendor Detectors
# =============================================================================
class _BaseDetector(ABC):
    @abstractmethod
    def detect(self) -> List[GPUInfo]:
        pass


class _NvidiaDetector(_BaseDetector):
    def __init__(self, cmd_timeout: int = _DEFAULT_CMD_TIMEOUT):
        self._cmd_timeout = cmd_timeout

    def detect(self) -> List[GPUInfo]:
        rc, out, _ = _run_with_kill([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version,"
            "compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit",
            "--format=csv,noheader,nounits"
        ], timeout=self._cmd_timeout)
        if rc != 0 or not out:
            return []
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                log.warning("nvidia-smi format mismatch: %d fields", len(parts))
                continue
            idx, name, mem_total, mem_free, driver, compute, pcie_gen, pcie_lanes, pwr_limit = parts[:9]
            gpu = GPUInfo(
                vendor=GPUVendor.NVIDIA,
                name=name,
                vram_total_mb=_safe_int(mem_total) or 0,
                vram_free_mb=_safe_int(mem_free) or 0,
                memory_type=_infer_nvidia_mem_type(name),
                driver_version=driver,
                compute_capability=compute if compute not in ("", "[N/A]", "N/A") else None,
                cuda_cores=_estimate_cuda_cores(name, compute),
                backend="cuda",
                pcie_gen=_safe_int(pcie_gen),
                pcie_lanes=_safe_int(pcie_lanes),
                bus_id=None,
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=_safe_float(pwr_limit),
                tdp_w=None,
                index=_safe_int(idx) or 0,
            )
            gpus.append(gpu)
        return gpus


class _AMDDetector(_BaseDetector):
    def __init__(self, cmd_timeout: int = _DEFAULT_CMD_TIMEOUT):
        self._cmd_timeout = cmd_timeout

    # ── Find rocm-smi binary ──────────────────────────────────────────────────
    # Systemd units run with a minimal PATH that typically excludes
    # /opt/rocm/bin.  Try explicit absolute paths before falling back to PATH.
    # Results are cached for the lifetime of the detector instance.
    _smi_path: Optional[str] = None

    def _find_rocm_smi(self) -> str:
        if self._smi_path is not None:
            return self._smi_path
        _candidates = list(filter(None, [
            os.path.join(os.environ.get("ROCM_PATH", ""), "bin", "rocm-smi"),
            "/opt/rocm/bin/rocm-smi",
            "/usr/bin/rocm-smi",
        ]))
        for p in _candidates:
            if Path(p).is_file() and os.access(p, os.X_OK):
                self._smi_path = p
                return p
        self._smi_path = "rocm-smi"   # rely on PATH as last resort
        return self._smi_path

    def detect(self) -> List[GPUInfo]:
        smi = self._find_rocm_smi()
        rc, out, _ = _run_with_kill([smi, "--showmeminfo", "vram", "--json"],
                                     timeout=self._cmd_timeout)
        if rc == 0 and out:
            gpus = self._parse_rocm(out, smi_path=smi)
            if gpus:
                return gpus
        return self._detect_sysfs()

    def _parse_rocm(self, json_out: str, smi_path: str = "rocm-smi") -> List[GPUInfo]:
        try:
            data = json.loads(json_out)
        except json.JSONDecodeError:
            return []
        rc2, drv_out, _ = _run_with_kill([smi_path, "--showdriverversion", "--json"], timeout=3)
        drv_data = {}
        if rc2 == 0 and drv_out:
            try:
                drv_data = json.loads(drv_out)
            except json.JSONDecodeError:
                pass
        gpus = []
        for card_key, info in data.items():
            if not card_key.startswith("card"):
                continue
            vram_total = int(info.get("VRAM Total Memory (B)", 0)) >> 20
            vram_used = int(info.get("VRAM Total Used Memory (B)", 0)) >> 20
            driver = drv_data.get(card_key, {}).get("Driver version", "unknown")
            idx_match = re.search(r"\d+", card_key)
            idx = int(idx_match.group()) if idx_match else 0
            name = _amd_name_from_sysfs(idx) or f"AMD GPU {card_key}"
            gpus.append(GPUInfo(
                vendor=GPUVendor.AMD,
                name=name,
                vram_total_mb=vram_total,
                vram_free_mb=max(0, vram_total - vram_used),
                memory_type=_infer_amd_mem_type(name),
                driver_version=driver,
                compute_capability=None,
                cuda_cores=None,
                backend="rocm",
                pcie_gen=None,
                pcie_lanes=None,
                bus_id=None,
                temperature_c=_amd_temp_rocm(card_key),
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                index=idx,
            ))
        return gpus

    def _detect_sysfs(self) -> List[GPUInfo]:
        drm = Path("/sys/class/drm")
        seen_pci: set = set()

        # ── System-level ROCm indicators (run BEFORE PyTorch, no runtime needed) ──
        # These checks work even before the HIP runtime is initialised by the
        # inference engine, so hardware detection reports rocm correctly on the
        # first pass rather than incorrectly falling back to vulkan.
        #
        # Reliable pre-torch ROCm signals:
        #   /dev/kfd   — Kernel Fusion Driver; present on EVERY ROCm system.
        #                If it exists, ROCm works (or very likely works).
        #   HSA_OVERRIDE_GFX_VERSION  — ROCm-specific env var.  Nobody sets this
        #                without deliberately enabling ROCm for their GPU.
        #                For gfx902 (Picasso/Raven) the service sets "9.0.0".
        #   /opt/rocm or $ROCM_PATH   — Standard ROCm installation directory.
        #   rocm-smi reachable        — ROCm management tool.
        #
        # Policy: /dev/kfd alone OR HSA_OVERRIDE alone → assume ROCm.
        #         /opt/rocm alone (no kfd, no override)  → needs rocm-smi to confirm.
        _has_kfd      = Path("/dev/kfd").exists()
        _has_hsa_ovrd = bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""))
        _rocm_home    = os.environ.get("ROCM_PATH", "")
        _has_rocm_dir = (
            Path("/opt/rocm").is_dir()
            or (bool(_rocm_home) and Path(_rocm_home).is_dir())
        )
        _rocm_smi_candidates_pre = list(dict.fromkeys(filter(None, [
            os.path.join(_rocm_home, "bin", "rocm-smi") if _rocm_home else "",
            "/opt/rocm/bin/rocm-smi",
            "/usr/bin/rocm-smi",
            "/usr/local/bin/rocm-smi",
        ])))
        _has_rocm_smi = any(
            Path(p).is_file() and os.access(p, os.X_OK)
            for p in _rocm_smi_candidates_pre if p
        )

        # ── PyTorch HIP check (secondary – may fail before engine init) ──────
        # Must happen before the drm early-return so that environments without
        # /sys/class/drm (containers, some APU configs) still get backend=rocm
        # when PyTorch HIP is available.  The synthetic fallback below uses this.
        has_rocm = False
        try:
            import torch as _t  # noqa: PLC0415
            has_rocm = _t.cuda.is_available() and hasattr(_t.version, "hip")
            if has_rocm:
                log.debug(
                    "ROCm confirmed via PyTorch HIP (torch.version.hip=%r)",
                    getattr(_t.version, "hip", None),
                )
        except Exception:
            pass

        # ── System-level check (overrides PyTorch result when indicators found) ─
        # /dev/kfd or HSA_OVERRIDE_GFX_VERSION → confident ROCm (no install check needed)
        # /opt/rocm but no kfd/override         → require rocm-smi as additional confirmation
        if not has_rocm:
            if _has_kfd or _has_hsa_ovrd:
                has_rocm = True
                log.debug(
                    "ROCm confirmed via system indicators: kfd=%s  hsa_override=%s",
                    _has_kfd, _has_hsa_ovrd,
                )
            elif _has_rocm_dir and _has_rocm_smi:
                has_rocm = True
                log.debug(
                    "ROCm confirmed via /opt/rocm installation + rocm-smi binary"
                )

        if not drm.exists():
            if has_rocm:
                log.debug(
                    "/sys/class/drm not found but PyTorch HIP confirmed — "
                    "returning synthetic GPU entry"
                )
                return [GPUInfo(
                    vendor=GPUVendor.AMD,
                    name="AMD GPU (HIP-confirmed)",
                    vram_total_mb=0, vram_free_mb=0,
                    memory_type=MemoryType.SHARED,
                    driver_version="unknown", compute_capability=None,
                    cuda_cores=None, backend="rocm",
                    pcie_gen=None, pcie_lanes=None, bus_id="hip-synthetic",
                    temperature_c=None, power_draw_w=None,
                    power_limit_w=None, tdp_w=None, index=0,
                    supported=True, fallback_reason=None,
                )]
            return []
        sys_ram_mb = psutil.virtual_memory().total >> 20
        _, drv_ver, _ = _run_with_kill(["modinfo", "amdgpu", "-F", "version"], timeout=3)
        drv_ver = drv_ver.strip() or "unknown"
        # has_rocm already set above (system-level indicators + PyTorch HIP).
        # If still not confirmed, try rocm-smi --version as a last-resort probe.
        if not has_rocm:
            _rocm_smi_candidates = list(dict.fromkeys(filter(None, [
                os.path.join(os.environ.get("ROCM_PATH", ""), "bin", "rocm-smi"),
                "/opt/rocm/bin/rocm-smi",
                "/usr/bin/rocm-smi",
                "rocm-smi",
            ])))
            has_rocm = any(
                _run_with_kill([p, "--version"], timeout=2)[0] == 0
                for p in _rocm_smi_candidates
                if p.startswith("/") or "/" not in p
            )
            if has_rocm:
                log.debug("ROCm confirmed via rocm-smi binary probe (fallback)")

        has_vulkan = _probe_vulkan_quick()
        backend = "rocm" if has_rocm else ("vulkan" if has_vulkan else "opencl")
        gpus = []
        for card in sorted(drm.iterdir()):
            if not card.name.startswith("card"):
                continue
            driver_link = card / "device" / "driver"
            if not driver_link.exists():
                continue
            try:
                drv = os.path.basename(os.readlink(str(driver_link)))
            except OSError:
                continue
            if drv != "amdgpu":
                continue
            try:
                bus_id = os.path.basename(os.readlink(str(card / "device")))
            except OSError:
                bus_id = card.name
            if bus_id in seen_pci:
                continue
            seen_pci.add(bus_id)
            vram_total_mb = int(_read_sysfs(str(card / "device/mem_info_vram_total")) or 0) >> 20
            vram_used_mb = int(_read_sysfs(str(card / "device/mem_info_vram_used")) or 0) >> 20
            vendor_id = (_read_sysfs(str(card / "device/vendor")) or "").strip()
            device_id = (_read_sysfs(str(card / "device/device")) or "").strip()
            is_apu = device_id.lower() in _AMD_APU_IDS
            if is_apu:
                apu_cap = max(min(vram_total_mb, sys_ram_mb // 4), 512)
                vram_total_mb = apu_cap
                vram_used_mb = min(vram_used_mb, vram_total_mb)
                mem_type = MemoryType.SHARED
            else:
                mem_type = MemoryType.GDDR6
            gpu_name = _lspci_gpu_name(vendor_id, device_id) or \
                f"AMD {'Integrated' if is_apu else 'Discrete'} GPU [{vendor_id}:{device_id}]"
            temp_c = None
            hwmon_base = card / "device" / "hwmon"
            if hwmon_base.exists():
                for hw in hwmon_base.iterdir():
                    tf = hw / "temp1_input"
                    if tf.exists():
                        try:
                            temp_c = int(tf.read_text().strip()) / 1000.0
                        except Exception:
                            pass
                        break
            gen = _parse_pcie_gen(_read_sysfs(str(card / "device/current_link_speed")))
            lanes = _safe_int(_read_sysfs(str(card / "device/current_link_width")) or "")
            idx = int(card.name[4:]) if card.name.startswith("card") else 0
            gpus.append(GPUInfo(
                vendor=GPUVendor.AMD,
                name=gpu_name,
                vram_total_mb=vram_total_mb,
                vram_free_mb=max(0, vram_total_mb - vram_used_mb),
                memory_type=mem_type,
                driver_version=drv_ver,
                compute_capability=None,
                cuda_cores=None,
                backend=backend,
                pcie_gen=gen,
                pcie_lanes=lanes,
                bus_id=bus_id,
                temperature_c=temp_c,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                index=idx,
                supported=True,
                fallback_reason=None if has_rocm or has_vulkan else "ROCm/Vulkan not detected, using OpenCL",
            ))
        # ── Synthetic fallback when HIP is confirmed but no /dev/dri/card* ──
        # This handles: APU with unusual sysfs layout, containers, or any env
        # where torch.version.hip confirms HIP but drm entries aren't exposed.
        # Without this, has_rocm=True in _detect_sysfs() is never propagated
        # to SystemInfo.has_rocm (which is built from GPU list, not the flag).
        if not gpus and has_rocm:
            log.debug(
                "ROCm/HIP confirmed by PyTorch but no /dev/dri/card* found — "
                "creating synthetic GPU entry so recommended=rocm is reported"
            )
            gpus.append(GPUInfo(
                vendor=GPUVendor.AMD,
                name="AMD GPU (HIP-confirmed)",
                vram_total_mb=0,
                vram_free_mb=0,
                memory_type=MemoryType.SHARED,
                driver_version=drv_ver,
                compute_capability=None,
                cuda_cores=None,
                backend="rocm",
                pcie_gen=None,
                pcie_lanes=None,
                bus_id="hip-synthetic",
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                index=0,
                supported=True,
                fallback_reason=None,
            ))
        return gpus


class _IntelDetector(_BaseDetector):
    def __init__(self, cmd_timeout: int = _DEFAULT_CMD_TIMEOUT):
        self._cmd_timeout = cmd_timeout

    def detect(self) -> List[GPUInfo]:
        drm = Path("/sys/class/drm")
        if not drm.exists():
            return []
        seen = set()
        gpus = []
        for card in sorted(drm.iterdir()):
            if not card.name.startswith("card"):
                continue
            driver_link = card / "device" / "driver"
            if not driver_link.exists():
                continue
            try:
                drv = os.path.basename(os.readlink(str(driver_link)))
            except OSError:
                continue
            if drv not in ("i915", "xe"):
                continue
            try:
                bus_id = os.path.basename(os.readlink(str(card / "device")))
            except OSError:
                bus_id = card.name
            if bus_id in seen:
                continue
            seen.add(bus_id)
            lmem = _read_sysfs(str(card / "device/lmem_size_bytes"))
            mem_mb = int(lmem) >> 20 if lmem else 0
            _, drv_ver, _ = _run_with_kill(["modinfo", drv, "-F", "version"], timeout=3)
            vendor_id = (_read_sysfs(str(card / "device/vendor")) or "").strip()
            device_id = (_read_sysfs(str(card / "device/device")) or "").strip()
            gpu_name = _lspci_gpu_name(vendor_id, device_id) or f"Intel GPU [{vendor_id}:{device_id}]"
            idx = int(card.name[4:]) if card.name.startswith("card") else 0
            gpus.append(GPUInfo(
                vendor=GPUVendor.INTEL,
                name=gpu_name,
                vram_total_mb=mem_mb,
                vram_free_mb=mem_mb,
                memory_type=MemoryType.LPDDR5 if mem_mb == 0 else MemoryType.GDDR6,
                driver_version=drv_ver.strip() or "unknown",
                compute_capability=None,
                cuda_cores=None,
                backend="opencl",
                pcie_gen=None,
                pcie_lanes=None,
                bus_id=bus_id,
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                index=idx,
            ))
        return gpus


# =============================================================================
# Helper Functions (Memory Type, Cores, etc.)
# =============================================================================
def _estimate_cuda_cores(name: str, compute_cap: Optional[str]) -> Optional[int]:
    table = {
        "4090": 16384, "4080": 9728, "4070 Ti": 7680, "4070": 5888,
        "4060 Ti": 4352, "4060": 3072,
        "3090 Ti": 10752, "3090": 10496, "3080 Ti": 10240,
        "3080": 8704, "3070 Ti": 6144, "3070": 5888,
        "3060 Ti": 4864, "3060": 3584, "3050": 2560,
        "A100": 6912, "A6000": 10752, "A5000": 8192,
        "A4000": 6144, "A2000": 3328,
        "H100": 16896, "H200": 16896,
        "L40S": 18176, "L40": 18176,
        "2080 Ti": 4352, "2080": 2944, "2070": 2304, "2060": 1920,
        "1080 Ti": 3584, "1080": 2560, "1070": 1920,
        "1060": 1280, "1050 Ti": 768, "1050": 640,
    }
    for suffix, cores in table.items():
        if suffix in name:
            return cores
    return None


def _infer_nvidia_mem_type(name: str) -> MemoryType:
    n = name.upper()
    if any(x in n for x in ("H100", "H200", "A100", "A30")):
        return MemoryType.HBM2E
    if any(x in n for x in ("4090", "4080", "4070 TI")):
        return MemoryType.GDDR6X
    if re.search(r"[234]\d{3}", n):
        return MemoryType.GDDR6
    return MemoryType.GDDR5


def _infer_amd_mem_type(name: str) -> MemoryType:
    n = name.upper()
    if any(x in n for x in ("MI300", "MI325", "MI250", "MI210", "MI200", "MI100", "MI60", "MI50")):
        return MemoryType.HBM2E
    if any(x in n for x in ("RX 7900", "RX 7800", "RX 7700", "RX 7600", "RX 7500",
                            "RX 6900", "RX 6800", "RX 6700", "RX 6600", "RX 6500", "RX 6400",
                            "RX 5700", "RX 5600", "RX 5500", "RX 5300")):
        return MemoryType.GDDR6
    if "GDDR6X" in n:
        return MemoryType.GDDR6X
    if "GDDR6" in n:
        return MemoryType.GDDR6
    if any(x in n for x in ("RX 580", "RX 570", "RX 480", "RX 470", "RX 460")):
        return MemoryType.GDDR5
    if any(x in n for x in ("VEGA 64", "VEGA 56", "VEGA 20")):
        return MemoryType.HBM2
    if any(x in n for x in ("RADEON GRAPHICS", "VEGA", "PICASSO", "RENOIR", "CEZANNE",
                            "REMBRANDT", "PHOENIX", "HAWK POINT", "INTEGRATED", "APU",
                            "780M", "760M", "740M", "680M", "660M", "NAVI3", "GFX1103")):
        return MemoryType.SHARED
    return MemoryType.GDDR6


def _amd_name_from_sysfs(idx: int) -> str:
    device_id = _read_sysfs(f"/sys/class/drm/card{idx}/device/device") or "unknown"
    return f"AMD GPU [device {device_id}]"


def _amd_temp_rocm(card_key: str) -> Optional[float]:
    rc, out, _ = _run_with_kill(["rocm-smi", "--showtemp", "--json"], timeout=3)
    if rc != 0 or not out:
        return None
    try:
        data = json.loads(out)
        raw = data.get(card_key, {}).get("Temperature (Sensor edge) (C)")
        return float(raw) if raw is not None else None
    except Exception:
        return None


def _lspci_gpu_name(vendor_id: str, device_id: str) -> Optional[str]:
    if not vendor_id or not device_id:
        return None
    rc, out, _ = _run_with_kill(["lspci", "-d", f"{vendor_id}:{device_id}"], timeout=3)
    if rc != 0:
        return None
    for line in out.splitlines():
        if any(kw in line for kw in ("VGA", "Display", "3D", "GPU")):
            parts = line.split(":", 2)
            return parts[-1].strip() if len(parts) >= 2 else None
    return None


def _parse_pcie_gen(speed_str: Optional[str]) -> Optional[int]:
    if not speed_str:
        return None
    try:
        gtps = float(speed_str.split()[0])
        if gtps >= 32:
            return 5
        if gtps >= 16:
            return 4
        if gtps >= 8:
            return 3
        if gtps >= 5:
            return 2
        return 1
    except (ValueError, IndexError):
        return None


def _probe_vulkan_quick(icd_paths: List[str] = _DEFAULT_VULKAN_ICD_PATHS) -> bool:
    """Quick Vulkan detection using configurable ICD paths."""
    for p in icd_paths:
        if Path(p).exists() and any(Path(p).iterdir()):
            log.debug("Vulkan ICD directory found: %s", p)
            return True
    rc, out, _ = _run_with_kill(["vulkaninfo", "--summary"], timeout=1)
    if rc == 0 and out:
        return True
    rc, _, _ = _run_with_kill(["vulkaninfo"], timeout=1)
    return rc == 0

def _detect_ram_channels() -> int:
    """Return the number of *populated* DIMM slots (0 = unknown).

    Single-channel is the #1 performance limiter on CPU/hybrid machines.
    We try two independent sources so a container without dmidecode still
    gets an answer via EDAC when the kernel exposes it.
    """
    # ── 1. EDAC (preferred: no root required, kernel always exposes it) ──
    try:
        mc_dir = Path("/sys/devices/system/edac/mc")
        if mc_dir.exists():
            populated = 0
            for mc in mc_dir.glob("mc*"):
                for dimm in mc.glob("dimm*"):
                    try:
                        # dimm_size contains e.g. "8192 MB" — nonzero means populated
                        size_raw = (dimm / "size").read_text().strip()
                        if size_raw and size_raw != "0 MB":
                            populated += 1
                    except (OSError, ValueError):
                        pass
            if populated > 0:
                return populated
    except Exception:
        pass

    # ── 2. dmidecode (needs root; falls through silently if not) ──
    try:
        rc, out, _ = _run_with_kill(["dmidecode", "-t", "memory"], timeout=3)
        if rc == 0:
            populated = 0
            for line in out.splitlines():
                s = line.strip()
                if not s.startswith("Size:"):
                    continue
                if "No Module Installed" in s or "Unknown" in s:
                    continue
                # Any non-empty Size value that isn't the sentinels above
                # is a populated slot. Handles MB/GB/MiB/GiB/TB uniformly.
                _, _, val = s.partition(":")
                if val.strip():
                    populated += 1
            if populated > 0:
                return populated
    except Exception:
        pass

    return 0


def _theoretical_ram_bandwidth_gbps() -> float:
    """Estimate theoretical peak from DIMM speed × populated channels.

    Returns 0.0 when channels cannot be determined — the caller falls back
    to a measured or conservative value.
    """
    try:
        rc, out, _ = _run_with_kill(
            ["dmidecode", "-t", "memory"], timeout=3,
        )
        if rc != 0:
            return 0.0
        speeds = []
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Speed:"):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        speeds.append(int(parts[1]))  # MT/s
                    except ValueError:
                        pass
        if not speeds:
            return 0.0
        speed_mts = max(speeds)
        channels = _detect_ram_channels() or 1
        # DDR: bytes per second = MT/s × 8 bytes × channels.
        # e.g. 2400 MT/s × 8 B × 2 ch = 38.4 GB/s
        return (speed_mts * 8 * channels) / 1000.0
    except Exception:
        return 0.0


def _measure_ram_bandwidth_gbps() -> float:
    """Run a short triad copy; return effective GB/s (0.0 if numpy missing).

    The test is a small write-heavy triad on `128 × min(1, free/10%)` MB,
    which fits in L2/L3-adjacent working set on most machines. It is a
    *bandwidth* measurement, not a latency benchmark, so it also captures
    single-channel-vs-dual-channel and NUMA effects that dmidecode can
    miss (e.g. on ARM SoCs or VMs).
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0
    try:
        avail_mb = psutil.virtual_memory().available >> 20
        size_mb = min(128, max(16, int(avail_mb * 0.10)))
        n = (size_mb * 1024 * 1024) // 8
        a = np.ones(n, dtype=np.float64)
        b = np.full(n, 2.0, dtype=np.float64)
        c = np.empty(n, dtype=np.float64)
        # Warm
        c[:] = a + b * 2.0
        iters = 3
        t0 = time.perf_counter()
        for _ in range(iters):
            c[:] = a + b * 2.0
        elapsed = time.perf_counter() - t0
        # Triad moves 3 arrays in, 1 array out per iteration ≈ 3×n×8 bytes
        bytes_moved = 3 * n * 8 * iters
        return bytes_moved / max(elapsed, 1e-9) / 1e9
    except Exception:
        return 0.0


def _probe_llama_cpp_backends() -> tuple[List[str], bool]:
    """Return (list of compiled backend names, gpu_offload_supported).

    The backend is compiled in at `pip install` time; it cannot be swapped
    at runtime. This probe tells the profile builder *which* acceleration
    paths the current install actually exposes so it can plan offload
    correctly on any machine (NVIDIA/CUDA, AMD/ROCm, AMD-or-Intel/Vulkan,
    Apple/Metal, Intel/SYCL, OpenCL, CPU).
    """
    names: List[str] = []
    gpu_offload = False
    try:
        from llama_cpp import llama_cpp as _lc  # type: ignore
    except ImportError:
        return names, gpu_offload

    # GPU offload flag (present in llama-cpp-python >= 0.2.5)
    try:
        if hasattr(_lc, "llama_supports_gpu_offload"):
            gpu_offload = bool(_lc.llama_supports_gpu_offload())
    except Exception:
        pass

    # Backend symbol presence (each backend registers a distinct init fn)
    # Backend symbol presence (each backend registers a distinct init fn).
    # Try the extension's own .so first; if it has no matching symbols
    # (some distributions split ggml backends into a separate libggml.so
    # that the extension links against), fall back to the already-loaded
    # process handle. Some builds strip the *_reg symbols entirely, so
    # also accept the gpu_offload flag as a positive signal on its own.
    try:
        import ctypes
        try:
            lib = ctypes.CDLL(_lc.__file__)
        except OSError:
            # Fall back to the already-loaded handle (the module itself).
            lib = ctypes.CDLL(None)
        for sym, name in (
            ("ggml_backend_cuda_reg",   "cuda"),
            ("ggml_backend_vk_reg",     "vulkan"),
            ("ggml_backend_metal_reg",  "metal"),
            ("ggml_backend_sycl_reg",   "sycl"),
            ("ggml_backend_opencl_reg", "opencl"),
            ("ggml_backend_rpc_reg",    "rpc"),
        ):
            try:
                getattr(lib, sym)
                names.append(name)
            except AttributeError:
                pass
    except Exception:
        pass

    # If no explicit backend symbols were found but GPU offload is on,
    # trust the flag alone — some builds strip the reg symbols. We use
    # the generic name "gpu" so downstream logic only needs to check
    # `any(b != "cpu" for b in backends)` rather than enumerate names.
    if not names and gpu_offload:
        names.append("gpu")

    # CPU is always available if the library loaded
    if "cpu" not in names:
        names.append("cpu")
    return names, gpu_offload

# =============================================================================
# Custom Kernel Module Detection
# =============================================================================
def _detect_custom_kernel_modules() -> Dict[str, bool]:
    """Return dict of custom kernel module availability."""
    result = {
        "cpu_optimizer": False,
        "amd_compat": False,
        "vulkan": False,
        "cuda": False,
    }
    if Path("/sys/kernel/llm_cpu_optimizer").exists():
        result["cpu_optimizer"] = True
    if Path("/sys/module/amd_kali_compat").exists():
        result["amd_compat"] = True
    result["vulkan"] = _probe_vulkan_quick()
    if _run_with_kill(["nvidia-smi"], timeout=1)[0] == 0:
        result["cuda"] = True
    return result

def _detect_gtt_total_mb() -> int:
    """Return the total GTT size for the primary AMD GPU, in MiB (0 = unknown).

    GTT is the addressable pool an APU's iGPU can use through system RAM.
    Linux exposes it via sysfs on every amdgpu device:
        /sys/class/drm/card*/device/mem_info_gtt_total
    Falls back to 60% of total system RAM when sysfs is absent, which is
    the kernel's default GTT ceiling (`amdgpu.gttsize` default behaviour).
    """
    try:
        from pathlib import Path as _P
        for card in sorted(_P("/sys/class/drm").glob("card[0-9]*")):
            gtt = card / "device" / "mem_info_gtt_total"
            if gtt.exists():
                return int(gtt.read_text().strip()) >> 20
    except Exception:
        pass
    # Fallback: 60% of total RAM (the kernel default for amdgpu)
    try:
        return int((psutil.virtual_memory().total >> 20) * 0.60)
    except Exception:
        return 0

# =============================================================================
# Main HardwareDetector Class (config injected)
# =============================================================================
class HardwareDetector:
    """
    Detects system hardware using configurable timeouts and paths.

    All settings are taken from the `config` dict (or `HardwareConfig`
    model).  If omitted, production‑tested defaults are used.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        # Normalise config
        if config is None:
            config = {}
        _raw_config = dict(config) if isinstance(config, dict) else {}
        try:
            from local_llm_core.hal.config_loader import HardwareConfig
            if isinstance(config, dict):
                cfg = HardwareConfig(**config).model_dump()
            else:
                cfg = config.model_dump()
        except Exception as exc:
            log.warning("Hardware config validation failed: %s; using defaults", exc)
            cfg = {}

        # Extract parameters with precedence: config > hardcoded defaults
        # _raw_config preserves keys Pydantic strips (e.g. force_backend)
        self._cfg      = {**cfg, **{k: v for k, v in _raw_config.items() if k not in cfg}}
        self._cmd_timeout = int(cfg.get("cmd_timeout_s", _DEFAULT_CMD_TIMEOUT))
        self._detection_timeout = float(cfg.get("detection_timeout_s", _DEFAULT_DETECTION_TIMEOUT))
        self._vulkan_icd_paths = cfg.get("vulkan_icd_paths", _DEFAULT_VULKAN_ICD_PATHS)
        self._backend_order = cfg.get("backend_order", _DEFAULT_BACKEND_ORDER)
        self._preferred_backend = cfg.get("preferred_backend", _DEFAULT_PREFERRED_BACKEND)

        self._info: Optional[SystemInfo] = None
        self._detectors = [
            _NvidiaDetector(self._cmd_timeout),
            _AMDDetector(self._cmd_timeout),
            _IntelDetector(self._cmd_timeout),
        ]

    def _detect_gpus(self) -> List[GPUInfo]:
        all_gpus = []
        for det in self._detectors:
            try:
                found = det.detect()
                if found:
                    log.debug("%s: found %d GPU(s)", type(det).__name__, len(found))
                    all_gpus.extend(found)
            except subprocess.TimeoutExpired as e:
                log.warning("%s timed out", type(det).__name__)
            except PermissionError as e:
                log.warning("%s needs elevated privileges: %s", type(det).__name__, e)
            except Exception as exc:
                log.debug("%s raised exception: %s", type(det).__name__, exc)
        return all_gpus

    def detect(self, force_refresh: bool = False, timeout_seconds: Optional[float] = None) -> SystemInfo:
        if timeout_seconds is None:
            timeout_seconds = self._detection_timeout
        if self._info and not force_refresh:
            return self._info

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._detect_sync)
            try:
                info = future.result(timeout=timeout_seconds)
                detected_gpus = len([g for g in info.gpus if g.supported])
            except FutureTimeoutError:
                log.error("Hardware detection timed out after %.1f seconds", timeout_seconds)
                info = self._build_cpu_only_info("Hardware detection timed out")
                return info
            except Exception as exc:
                log.error("Hardware detection failed: %s", exc, exc_info=True)
                info = self._build_cpu_only_info(str(exc))
                return info

        self._info = info
        log.info("Detection done: %d GPU(s), primary=%s, recommended=%s",
                 detected_gpus, info.primary_gpu.name if info.primary_gpu else "None", info.recommended_backend)
        return info

    def _detect_sync(self) -> SystemInfo:
        log.info("Starting hardware detection ...")
        vm = psutil.virtual_memory()
        info = SystemInfo(
            cpu_model=_get_cpu_model(),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            ram_total_mb=vm.total >> 20,
            ram_free_mb=vm.available >> 20,
            os_name=platform.system(),
            kernel_version=platform.release(),
            arch=platform.machine(),
        )
        real_gpus = self._detect_gpus()
        all_gpus = list(real_gpus)
        if not all_gpus:
            log.warning("No GPU detected — CPU inference will be used.")
            all_gpus.append(GPUInfo(
                vendor=GPUVendor.UNKNOWN,
                name="CPU (No GPU)",
                vram_total_mb=0,
                vram_free_mb=0,
                memory_type=MemoryType.UNKNOWN,
                driver_version="N/A",
                compute_capability=None,
                cuda_cores=None,
                backend="cpu",
                pcie_gen=None,
                pcie_lanes=None,
                bus_id=None,
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                supported=False,
                fallback_reason="No GPU detected",
                index=0,
            ))
        info.gpus = all_gpus
        info.primary_gpu = max(all_gpus, key=lambda g: g.vram_total_mb)
        info.has_cuda = any(g.backend == "cuda" for g in all_gpus)
        info.has_rocm = any(g.backend == "rocm" for g in all_gpus)
        info.has_opencl = any(g.backend == "opencl" for g in all_gpus)
        info.has_vulkan = _probe_vulkan_quick(self._vulkan_icd_paths)
        info.has_numba = False
        try:
            import numpy
            info.has_numpy = True
        except ImportError:
            info.has_numpy = False

        active = {
            "cuda": info.has_cuda,
            "rocm": info.has_rocm,
            "vulkan": info.has_vulkan,
            "opencl": info.has_opencl,
            "mps": False,
            "numba": info.has_numba,
            "numpy": info.has_numpy,
            "cpu": True,
        }
        info.available_backends = [b for b in self._backend_order if active.get(b, False)]
        info.recommended_backend = self.best_backend(info)
        info.kernel_modules = _detect_custom_kernel_modules()
        # ── Adaptive-performance measurements ─────────────────────────
        # Every field below is either measured or absent. Absent is
        # strictly better than a hardcoded guess: the profile builder
        # falls back to tier defaults when the measurement is 0/empty,
        # so an ARM server, a VM without dmidecode, or a container still
        # gets a working profile — just not a *derived* one.
        info.ram_channels                  = _detect_ram_channels()
        info.ram_bandwidth_gbps_theoretical = _theoretical_ram_bandwidth_gbps()
        info.ram_bandwidth_gbps_measured   = _measure_ram_bandwidth_gbps()
        info.llama_cpp_backends, info.llama_cpp_gpu_offload = _probe_llama_cpp_backends()
        info.vram_gtt_total_mb = _detect_gtt_total_mb()
        log.info(
            "Adaptive facts: ram_channels=%d  bw_theoretical=%.1f GB/s  gtt_total=%d "
            "bw_measured=%.1f GB/s  llama_cpp_backends=%s  gpu_offload=%s",
            info.ram_channels,
            info.ram_bandwidth_gbps_theoretical,
            info.vram_gtt_total_mb,
            info.ram_bandwidth_gbps_measured,
            info.llama_cpp_backends,
            info.llama_cpp_gpu_offload,
        )
        return info

    def _build_cpu_only_info(self, reason: str) -> SystemInfo:
        vm = psutil.virtual_memory()
        _backends, _gpu_offload = _probe_llama_cpp_backends()
        return SystemInfo(
            cpu_model=_get_cpu_model(),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            ram_total_mb=vm.total >> 20,
            ram_free_mb=vm.available >> 20,
            os_name=platform.system(),
            kernel_version=platform.release(),
            arch=platform.machine(),
            gpus=[GPUInfo(
                vendor=GPUVendor.UNKNOWN,
                name="CPU (No GPU)",
                vram_total_mb=0,
                vram_free_mb=0,
                memory_type=MemoryType.UNKNOWN,
                driver_version="N/A",
                compute_capability=None,
                cuda_cores=None,
                backend="cpu",
                pcie_gen=None,
                pcie_lanes=None,
                bus_id=None,
                temperature_c=None,
                power_draw_w=None,
                power_limit_w=None,
                tdp_w=None,
                supported=False,
                fallback_reason=reason,
                index=0,
            )],
            has_cuda=False,
            has_rocm=False,
            has_opencl=False,
            has_vulkan=False,
            has_numba=False,
            has_numpy=True,
            available_backends=["cpu"],
            recommended_backend="cpu",
            kernel_modules=_detect_custom_kernel_modules(),

            # ── Adaptive-performance measurements ─────────────────────────
            # Every field below is either measured or absent. Absent is
            # strictly better than a hardcoded guess: the profile builder
            # falls back to tier defaults when the measurement is 0/empty,
            # so an ARM server, a VM without dmidecode, or a container still
            # gets a working profile — just not a *derived* one.
            ram_channels=_detect_ram_channels(),
            ram_bandwidth_gbps_theoretical=_theoretical_ram_bandwidth_gbps(),
            ram_bandwidth_gbps_measured=_measure_ram_bandwidth_gbps(),
            llama_cpp_backends=_backends,
            llama_cpp_gpu_offload=_gpu_offload,
            vram_gtt_total_mb = _detect_gtt_total_mb(),
        )

    def detect_all(self, force_refresh: bool = False) -> SystemInfo:
        return self.detect(force_refresh=force_refresh)

    def best_backend(self, info: Optional[SystemInfo] = None) -> str:
        if info is None:
            info = self.detect()

        # ── Config-level force_backend override ───────────────────────────────
        # When auto-detection cannot confirm ROCm (e.g. gfx902 APU without
        # HSA_OVERRIDE_GFX_VERSION=9.0.0 set in the systemd service unit),
        # set  hardware.force_backend: "rocm"  in config.yaml to bypass
        # detection entirely.  "hip" is normalised to "rocm" for consistency.
        _forced = self._cfg.get("force_backend")
        if _forced and str(_forced).lower() not in ("", "auto", "null", "none"):
            _fb = str(_forced).lower().strip()
            if _fb == "hip":
                _fb = "rocm"
            log.info("⚡ Backend forced to '%s' by config (hardware.force_backend)", _fb)
            return _fb

        # AMD compat kernel module takes priority — forces Vulkan when ROCm
        # is not in available_backends (e.g. gfx902 without HSA_OVERRIDE)
        if info.kernel_modules.get("amd_compat", False):
            if "vulkan" in info.available_backends and "rocm" not in info.available_backends:
                log.info("AMD compat module detected, preferring Vulkan backend")
                return "vulkan"

        # Normalise hip → rocm.  HIP (Heterogeneous Interface for Portability)
        # IS the ROCm runtime — they are the same backend named differently.
        # Config allows preferred_backend: "hip" for clarity; internally we
        # use "rocm" as the canonical name in available_backends.
        preferred = self._preferred_backend
        if preferred == "hip":
            preferred = "rocm"

        if preferred != "auto" and preferred in info.available_backends:
            return preferred

        # Warn when the configured preferred backend is absent (not yet available)
        if self._preferred_backend not in ("auto", preferred) or (
            preferred != "auto" and preferred not in info.available_backends
        ):
            log.warning(
                "Preferred backend '%s' not in available_backends %s — "
                "falling back to best available.%s",
                self._preferred_backend,
                info.available_backends,
                (
                    " For AMD gfx902 ROCm/HIP, set "
                    "HSA_OVERRIDE_GFX_VERSION=9.0.0 in the service environment."
                ) if self._preferred_backend in ("hip", "rocm") else "",
            )

        for b in self._backend_order:
            if b in info.available_backends:
                return b
        return "cpu"

    def print_report(self) -> None:
        info = self.detect()
        print(f"\n\n{'='*60}")
        print("  LOCAL LLM CLOUD — Hardware Report")
        print(f"\n{'='*60}")
        print(f"\n  OS          : {info.os_name} {info.kernel_version} ({info.arch})")
        print(f"\n  CPU         : {info.cpu_model}")
        print(f"\n  Cores/Threads: {info.cpu_cores} / {info.cpu_threads}")
        print(f"\n  RAM         : {info.ram_total_mb:,} MB total / {info.ram_free_mb:,} MB free")
        print(f"\n  Backend     : {info.recommended_backend.upper()}")
        if info.ram_channels:
            print(f"\n  RAM channels: {info.ram_channels}"
                  f"  ({info.ram_bandwidth_gbps_theoretical:.1f} GB/s theoretical)")
        if info.ram_bandwidth_gbps_measured:
            print(f"\n  RAM BW (meas): {info.ram_bandwidth_gbps_measured:.1f} GB/s")
        if info.llama_cpp_backends:
            print(f"\n  llama-cpp-python backends: {', '.join(info.llama_cpp_backends)}"
                  f"  (gpu_offload={info.llama_cpp_gpu_offload})")
        if len(info.available_backends) > 1:
            print(f"\n  All backends: {', '.join(info.available_backends)}")
        print()
        for i, gpu in enumerate(info.gpus):
            if not gpu.supported and gpu.vram_total_mb == 0:
                continue
            print(f"\n  GPU [{i}] {gpu.name}")
            print(f"\n    Vendor   : {gpu.vendor.name}")
            print(f"\n    VRAM     : {gpu.vram_total_mb:,} MB ({gpu.memory_type.value})")
            print(f"\n    Free     : {gpu.vram_free_mb:,} MB")
            print(f"\n    Driver   : {gpu.driver_version}")
            if gpu.compute_capability:
                print(f"\n    Compute  : {gpu.compute_capability}")
            if gpu.cuda_cores:
                print(f"\n    CUDA Cores: {gpu.cuda_cores:,}")
            if gpu.temperature_c is not None:
                print(f"\n    Temp     : {gpu.temperature_c}°C")
            if gpu.pcie_gen:
                print(f"\n    PCIe     : Gen{gpu.pcie_gen} x{gpu.pcie_lanes or '?'}")
            if gpu.fallback_reason:
                print(f"\n    ⚠ {gpu.fallback_reason}")
        print(f"\n  Custom Kernel Modules:")
        for name, present in info.kernel_modules.items():
            print(f"\n    {name}: {'present' if present else 'not detected'}")
        print(f"\n{'='*60}\n")


def _select_quantization(model_params_b: float, vram_gb: float) -> str:
    tiers = [
        ("Q8_0", 8.0 / 8),
        ("Q6_K", 6.0 / 8),
        ("Q5_K_M", 5.5 / 8),
        ("Q4_K_M", 4.5 / 8),
        ("Q4_K_S", 4.0 / 8),
        ("Q3_K_M", 3.35 / 8),
        ("Q3_K_S", 3.0 / 8),
        ("Q2_K", 2.5 / 8),
    ]
    overhead_gb = 0.5
    headroom = 0.95
    for quant, bpw in tiers:
        required_gb = model_params_b * 1e9 * bpw / (1024 ** 3) + overhead_gb
        if required_gb <= vram_gb * headroom:
            return quant
    return "Q2_K"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    HardwareDetector().print_report()


if __name__ == "__main__":
    main()
    