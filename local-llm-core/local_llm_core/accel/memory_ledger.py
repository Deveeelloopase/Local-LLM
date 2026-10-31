"""
memory_ledger.py — single source of truth for "how many workers may exist".

All inputs are measured or config-supplied. No file reads. No hardcoded
model assumptions. Works on any OS by relying on psutil only.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import psutil

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LedgerBudget:
    ram_total_mb: int
    ram_available_mb: int
    os_reserve_mb: int
    igpu_carveout_mb: int          # 0 on discrete-GPU or CPU-only machines
    weights_shared_mb: int         # mmap-backed, counted once
    kv_per_worker_mb: int          # NOT shared
    runtime_per_worker_mb: int     # python + allocator + scores buffer
    measured_bw_gbps: float        # sustained, not theoretical
    cpu_physical_cores: int
    target_min_tps_per_worker: float
    configured_max_workers: int    # 0 = unlimited (respect computed)

    def memory_ceiling(self) -> int:
        per_worker = self.kv_per_worker_mb + self.runtime_per_worker_mb
        if per_worker <= 0:
            return 1
        available_for_workers = (
            self.ram_total_mb
            - self.os_reserve_mb
            - self.igpu_carveout_mb
            - self.weights_shared_mb
        )
        if available_for_workers <= 0:
            return 1
        return max(1, available_for_workers // per_worker)

    def bandwidth_ceiling(self) -> int:
        if self.measured_bw_gbps <= 0 or self.target_min_tps_per_worker <= 0:
            return 1
        model_gb = max(0.1, self.weights_shared_mb / 1024.0)
        per_worker_bw = model_gb * self.target_min_tps_per_worker
        if per_worker_bw <= 0:
            return 1
        return max(1, int(self.measured_bw_gbps // per_worker_bw))

    def cpu_ceiling(self) -> int:
        # Two workers per physical core is the maximum before SMT starts
        # to hurt decode throughput more than concurrency helps.
        if self.cpu_physical_cores <= 0:
            return 1
        return max(1, self.cpu_physical_cores // 2)

    def max_workers(self) -> int:
        ceilings = [
            self.memory_ceiling(),
            self.bandwidth_ceiling(),
            self.cpu_ceiling(),
        ]
        if self.configured_max_workers > 0:
            ceilings.append(self.configured_max_workers)
        return max(1, min(ceilings))


class MemoryLedger:
    """
    Constructed once in the parent, before any worker spawns. All numbers are
    observable facts; nothing is estimated from a tier table.
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
        )
        log.info(
            "MemoryLedger: ram=%dMB weights=%dMB kv/worker=%dMB runtime/worker=%dMB "
            "bw=%.1fGB/s target_min_tps=%.1f → max_workers=%d "
            "(mem=%d bw=%d cpu=%d)",
            self._budget.ram_total_mb,
            self._budget.weights_shared_mb,
            self._budget.kv_per_worker_mb,
            self._budget.runtime_per_worker_mb,
            self._budget.measured_bw_gbps,
            self._budget.target_min_tps_per_worker,
            self._budget.max_workers(),
            self._budget.memory_ceiling(),
            self._budget.bandwidth_ceiling(),
            self._budget.cpu_ceiling(),
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
        return free_mb > per_worker + self._budget.os_reserve_mb

    def refuse_pct(self) -> float:
        """Return available RAM as percent of total, for the admission guard."""
        vm = psutil.virtual_memory()
        return float(vm.percent)
