"""Machine facts for the auto-configuration planner.

:class:`MachineFacts` is a small, immutable, JSON-friendly snapshot of the
hardware the server is running on.  It is built from the detector's
``SystemInfo`` (so nothing is probed twice) and adds the few derived numbers a
planner needs:

* which inference backend the *installed* ``llama-cpp-python`` can really use
  (a GPU in the machine is irrelevant if the wheel is CPU-only),
* whether the accelerator is a unified-memory device (iGPU/APU) whose
  "VRAM" is carved out of system RAM,
* the memory the accelerator can address for weights and KV cache
  (:attr:`MachineFacts.device_budget_mb`),
* the locked-memory limit that decides whether ``mlock`` can succeed,
* a stable hardware fingerprint used to key benchmark results.

Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

__version__ = "1.0.0"

log = logging.getLogger(__name__)

# Backends that offload to an accelerator, in order of preference when the
# installed llama-cpp-python was built with more than one.
_GPU_BACKEND_ORDER: Tuple[str, ...] = ("cuda", "hip", "rocm", "metal", "sycl", "vulkan")

# Fraction of the addressable accelerator memory the planner may fill.
_DEVICE_FILL_UMA = 0.90
_DEVICE_FILL_DISCRETE = 0.92
_DEVICE_RESERVE_DISCRETE_MB = 128


@dataclass(frozen=True)
class MachineFacts:
    """Immutable description of the host used by the planner."""

    cpu_model: str = "unknown"
    physical_cores: int = 1
    logical_threads: int = 1
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    ram_channels: int = 0
    ram_bandwidth_gbps: float = 0.0
    gpu_name: str = ""
    gpu_vendor: str = ""
    gpu_count: int = 0
    backend: str = "cpu"
    uma: bool = False
    vram_carveout_mb: int = 0
    gtt_mb: int = 0
    device_budget_mb: int = 0
    memlock_mb: int = 0
    llama_cpp_version: str = ""
    notes: Tuple[str, ...] = field(default_factory=tuple)

    # ── derived ─────────────────────────────────────────────────────────
    @property
    def gpu_offload(self) -> bool:
        """True when layers can actually be placed on an accelerator."""
        return self.backend != "cpu" and self.device_budget_mb > 0

    @property
    def single_channel(self) -> bool:
        return self.ram_channels == 1

    def fingerprint(self) -> str:
        """Stable id for benchmark caching.

        Includes everything that changes decode speed (CPU, memory size and
        channel count, accelerator, backend, ``llama-cpp-python`` build) and
        excludes volatile values such as free memory.  Memory size is rounded
        to the nearest GiB so firmware-reserved differences between boots do
        not invalidate results.
        """
        parts = (
            self.cpu_model.strip().lower(),
            str(self.physical_cores),
            str(round(self.ram_total_mb / 1024.0)),
            str(self.ram_channels),
            self.gpu_name.strip().lower(),
            self.backend,
            self.llama_cpp_version,
        )
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["gpu_offload"] = self.gpu_offload
        d["fingerprint"] = self.fingerprint()
        return d

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def from_system_info(
        cls,
        si: Any,
        *,
        llama_cpp_version: Optional[str] = None,
        memlock_mb: Optional[int] = None,
    ) -> "MachineFacts":
        """Build facts from a detector ``SystemInfo`` (duck-typed)."""
        notes = []
        gpu = getattr(si, "primary_gpu", None)
        gpus = list(getattr(si, "gpus", None) or ([gpu] if gpu else []))
        if gpu is not None and not _is_real_gpu(gpu):
            gpu, gpus = None, []                 # the detector's "CPU (No GPU)" placeholder

        backend = _usable_backend(
            list(getattr(si, "llama_cpp_backends", None) or []),
            bool(getattr(si, "llama_cpp_gpu_offload", False)),
            str(getattr(getattr(gpu, "vendor", None), "name", "") or "") if gpu is not None else "",
        )

        if gpu is not None and backend == "cpu":
            notes.append(
                "an accelerator is present but the installed llama-cpp-python "
                "has no GPU backend; planning for CPU inference"
            )

        mem_type = getattr(getattr(gpu, "memory_type", None), "name", "") if gpu else ""
        vendor = getattr(getattr(gpu, "vendor", None), "name", "") if gpu else ""
        uma = bool(gpu) and (mem_type in ("SHARED", "UNIFIED") or vendor == "APPLE")

        carve = int(getattr(gpu, "vram_total_mb", 0) or 0) if gpu else 0
        gtt = int(getattr(si, "vram_gtt_total_mb", 0) or 0)
        ram_total = int(getattr(si, "ram_total_mb", 0) or 0)

        budget = 0
        same = [gpu] if gpu is not None else []
        if backend != "cpu" and gpu is not None:
            if uma:
                addressable = max(carve, gtt)
                if ram_total:
                    addressable = min(addressable, int(ram_total * 0.75)) or addressable
                budget = int(addressable * _DEVICE_FILL_UMA)
            else:
                same = [g for g in gpus if getattr(getattr(g, "vendor", None), "name", "") == vendor] or same
                free = sum(int(getattr(g, "vram_free_mb", 0) or 0) or int(getattr(g, "vram_total_mb", 0) or 0)
                           for g in same) or carve
                budget = max(0, int(free * _DEVICE_FILL_DISCRETE) - _DEVICE_RESERVE_DISCRETE_MB)

        _bw_meas = float(getattr(si, "ram_bandwidth_gbps_measured", 0.0) or 0.0)
        _bw_theo = float(getattr(si, "ram_bandwidth_gbps_theoretical", 0.0) or 0.0)
        if _bw_theo > 0:
            bw = min(_bw_theo, max(_bw_meas, 0.65 * _bw_theo))
        else:
            bw = _bw_meas

        return cls(
            cpu_model=str(getattr(si, "cpu_model", "") or "unknown"),
            physical_cores=max(1, int(getattr(si, "cpu_cores", 0) or 0)),
            logical_threads=max(1, int(getattr(si, "cpu_threads", 0) or 0)),
            ram_total_mb=ram_total,
            ram_available_mb=int(getattr(si, "ram_free_mb", 0) or 0),
            ram_channels=int(getattr(si, "ram_channels", 0) or 0),
            ram_bandwidth_gbps=round(bw, 2),
            gpu_name=str(getattr(gpu, "name", "") or "") if gpu else "",
            gpu_vendor=vendor,
            gpu_count=len(same),
            backend=backend,
            uma=uma,
            vram_carveout_mb=carve,
            gtt_mb=gtt,
            device_budget_mb=budget,
            memlock_mb=_memlock_mb() if memlock_mb is None else int(memlock_mb),
            llama_cpp_version=_llama_cpp_version() if llama_cpp_version is None else llama_cpp_version,
            notes=tuple(notes),
        )

    @classmethod
    def probe(cls, system_info: Any = None, hardware_config: Optional[Dict[str, Any]] = None) -> "MachineFacts":
        """Detect the machine.  Pass ``system_info`` to reuse a prior probe."""
        si = system_info
        if si is None:
            from local_llm_core.hal.hardware_detector import HardwareDetector

            si = HardwareDetector().detect_all()
        return cls.from_system_info(si)


# ── helpers ─────────────────────────────────────────────────────────────
def _is_real_gpu(gpu: Any) -> bool:
    """False for the detector's placeholder entry used on machines without a GPU."""
    name = str(getattr(gpu, "name", "") or "")
    vendor = getattr(getattr(gpu, "vendor", None), "name", "")
    return not name.startswith("CPU (No GPU)") and not (vendor == "UNKNOWN" and not name)


_VENDOR_BACKEND = {"NVIDIA": "cuda", "AMD": "hip", "APPLE": "metal", "INTEL": "sycl"}


def _usable_backend(compiled: list, gpu_offload: bool, vendor: str = "") -> str:
    """Pick the accelerator backend of the installed llama-cpp-python.

    The probe reports the generic token "gpu" when the build can offload but
    exposes no named backend symbol (HIP/ROCm builds, stripped builds). That
    is a positive signal, so resolve it from the GPU vendor instead of
    discarding it (discarding it forced every model to CPU).
    """
    if not gpu_offload:
        return "cpu"
    names = [str(b).lower() for b in compiled]
    for pref in _GPU_BACKEND_ORDER:
        if pref in names:
            return "hip" if pref == "rocm" else pref
    if "gpu" in names:
        return _VENDOR_BACKEND.get(str(vendor).upper(), "gpu")
    return "cpu"


def _memlock_mb() -> int:
    """Soft ``RLIMIT_MEMLOCK`` in MiB; ``-1`` means unlimited, ``0`` unknown."""
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft == resource.RLIM_INFINITY:
            return -1
        return int(soft) >> 20
    except Exception:
        return 0


def _llama_cpp_version() -> str:
    try:
        from importlib.metadata import version

        return version("llama-cpp-python")
    except Exception:
        return ""
