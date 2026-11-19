"""
memory_ledger.py — single source of truth for "how many workers may exist".

All inputs are measured or config-supplied. No file reads. No hardcoded
model assumptions. Works on any OS by relying on psutil only.

Revision notes (vs. the original):
  * Weights are only "shared once" when workers run on the CPU (mmap'd file
    pages).  When layers are offloaded to a GPU every worker process owns its
    own device copy, so the device budget is divided by (weights + KV) per
    worker.  The original counted weights once and never looked at VRAM, which
    on a 24 GB card would let it start N workers that each need a full copy.
  * On a UMA APU the device buffers are carved out of system RAM (GTT), so the
    per-worker copy is charged to RAM too and the GTT *capacity* is no longer
    subtracted from RAM on top of the weights (that double-counted them).

version = 1.0.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import psutil

log = logging.getLogger(__name__)

_UNLIMITED = 1_000_000


@dataclass(frozen=True)
class LedgerBudget:
    ram_total_mb: int
    ram_available_mb: int
    os_reserve_mb: int
    igpu_carveout_mb: int          # RAM permanently hidden from the OS (0 if none / not applicable)
    weights_shared_mb: int         # model file size
    kv_per_worker_mb: int          # NOT shared
    runtime_per_worker_mb: int     # python + allocator + scores buffer
    measured_bw_gbps: float        # effective bandwidth (see hardware_profiles._select_effective_bandwidth)
    cpu_physical_cores: int
    target_min_tps_per_worker: float
    configured_max_workers: int    # 0 = unlimited (respect computed)
    cpu_cores_per_worker: int = 2  # cores required per worker process
    max_n_seq_max: int = 8         # practical decode-batching ceiling
    # ── device-aware fields (all optional; defaults reproduce CPU-only behaviour)
    weights_on_device: bool = False   # layers offloaded to a GPU / iGPU
    uma: bool = False                 # device memory is carved out of system RAM
    device_budget_mb: int = 0         # usable memory PER device (VRAM, or GTT on UMA); 0 = unknown
    n_devices: int = 1
    device_margin_mb: int = 256       # compute buffers / driver overhead per worker

    # ── ceilings ────────────────────────────────────────────────────────
    def memory_ceiling(self) -> int:
        per_worker = self.kv_per_worker_mb + self.runtime_per_worker_mb
        shared = self.weights_shared_mb
        if self.weights_on_device and self.uma:
            # Device copies live in system RAM: one per worker, nothing shared.
            per_worker += self.weights_shared_mb
            shared = 0
        if per_worker <= 0:
            return 1
        # min(ram_total - shared, ram_available): ram_available is a boot-time
        # snapshot, so other consumers (PostgreSQL, another server, page
        # cache residue) tighten the ceiling.
        available_for_workers = (
            min(self.ram_total_mb - shared, self.ram_available_mb)
            - self.os_reserve_mb
            - self.igpu_carveout_mb
        )
        if available_for_workers <= 0:
            return 1
        return max(1, available_for_workers // per_worker)

    def device_ceiling(self) -> int:
        """Workers that fit in device memory (each owns a full weights copy)."""
        if not self.weights_on_device or self.device_budget_mb <= 0:
            return _UNLIMITED
        per_worker_dev = (
            self.weights_shared_mb + self.kv_per_worker_mb + self.device_margin_mb
        )
        if per_worker_dev <= 0:
            return _UNLIMITED
        per_device = max(1, self.device_budget_mb // per_worker_dev)
        return max(1, per_device * max(1, self.n_devices))

    def bandwidth_ceiling(self) -> int:
        if self.measured_bw_gbps <= 0 or self.target_min_tps_per_worker <= 0:
            return 1
        model_gb = max(0.1, self.weights_shared_mb / 1024.0)
        per_worker_bw = model_gb * self.target_min_tps_per_worker
        if per_worker_bw <= 0:
            return 1
        return max(1, int(self.measured_bw_gbps // per_worker_bw))

    def cpu_ceiling(self) -> int:
        if self.cpu_physical_cores <= 0:
            return 1
        per_worker = max(1, int(self.cpu_cores_per_worker))
        return max(1, self.cpu_physical_cores // per_worker)

    def max_workers(self) -> int:
        ceilings = [
            self.memory_ceiling(),
            self.device_ceiling(),
            self.bandwidth_ceiling(),
            self.cpu_ceiling(),
        ]
        if self.configured_max_workers > 0:
            ceilings.append(self.configured_max_workers)
        return max(1, min(ceilings))


class MemoryLedger:
    """
    Constructed once in the parent, before any worker spawns. All numbers
    are observable facts; nothing is estimated from a tier table.
    """

    def __init__(
        self,
        *,
        weights_shared_mb: int,
        kv_per_worker_mb: int,
        runtime_per_worker_mb: int,
        measured_bw_gbps: float,
        igpu_carveout_mb: int = 0,
        os_reserve_mb: int = 1024,
        target_min_tps_per_worker: float = 1.0,
        configured_max_workers: int = 0,
        cpu_cores_per_worker: int = 2,
        max_n_seq_max: int = 8,
        weights_on_device: bool = False,
        uma: bool = False,
        device_budget_mb: int = 0,
        n_devices: int = 1,
        device_margin_mb: int = 256,
    ) -> None:
        vm = psutil.virtual_memory()
        self._budget = LedgerBudget(
            ram_total_mb=int(vm.total >> 20),
            ram_available_mb=int(vm.available >> 20),
            os_reserve_mb=int(os_reserve_mb),
            igpu_carveout_mb=int(igpu_carveout_mb),
            weights_shared_mb=int(weights_shared_mb),
            kv_per_worker_mb=int(kv_per_worker_mb),
            runtime_per_worker_mb=int(runtime_per_worker_mb),
            measured_bw_gbps=float(measured_bw_gbps),
            cpu_physical_cores=int(psutil.cpu_count(logical=False) or 1),
            target_min_tps_per_worker=float(target_min_tps_per_worker),
            configured_max_workers=int(configured_max_workers),
            cpu_cores_per_worker=max(1, int(cpu_cores_per_worker)),
            max_n_seq_max=max(1, int(max_n_seq_max)),
            weights_on_device=bool(weights_on_device),
            uma=bool(uma),
            device_budget_mb=int(device_budget_mb),
            n_devices=max(1, int(n_devices)),
            device_margin_mb=int(device_margin_mb),
        )
        b = self._budget
        _dev = b.device_ceiling()
        log.info(
            "MemoryLedger: ram=%dMB avail=%dMB weights=%dMB "
            "kv/worker=%dMB runtime/worker=%dMB bw=%.1fGB/s "
            "target_min_tps=%.1f cores/worker=%d n_seq_hint=%d "
            "device(on=%s uma=%s budget=%dMB x%d) "
            "→ max_workers=%d (mem=%d dev=%s bw=%d cpu=%d)",
            b.ram_total_mb, b.ram_available_mb, b.weights_shared_mb,
            b.kv_per_worker_mb, b.runtime_per_worker_mb, b.measured_bw_gbps,
            b.target_min_tps_per_worker, b.cpu_cores_per_worker, b.max_n_seq_max,
            b.weights_on_device, b.uma, b.device_budget_mb, b.n_devices,
            b.max_workers(), b.memory_ceiling(),
            "n/a" if _dev >= _UNLIMITED else _dev,
            b.bandwidth_ceiling(), b.cpu_ceiling(),
        )

    @property
    def budget(self) -> LedgerBudget:
        return self._budget

    def max_workers(self) -> int:
        return self._budget.max_workers()

    def can_add_worker(self) -> bool:
        """Runtime check — available RAM, not the boot-time snapshot."""
        vm = psutil.virtual_memory()
        free_mb = int(vm.available >> 20)
        per_worker = (
            self._budget.kv_per_worker_mb + self._budget.runtime_per_worker_mb
        )
        if self._budget.weights_on_device and self._budget.uma:
            per_worker += self._budget.weights_shared_mb
        return free_mb > per_worker + self._budget.os_reserve_mb

    def refuse_pct(self) -> float:
        """Return current RAM usage as percent of total, for the admission guard."""
        vm = psutil.virtual_memory()
        return float(vm.percent)

    def n_seq_max_ceiling(self) -> int:
        """Upper bound for BatchedInferenceEngine.n_seq_max on this hardware."""
        return max(1, int(self._budget.max_n_seq_max))
