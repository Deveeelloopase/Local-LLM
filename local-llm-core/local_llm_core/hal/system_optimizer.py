from __future__ import annotations

"""
hal/system_optimizer.py — Production-Ready OS-Level Tuner v1.0.0

Config injected via constructor; no file I/O at import time.

every sysctl / sysfs write now records the original
value to a persistent baseline file (~/.cache/local-llm/system_baseline.json
by default). On clean shutdown the originals are restored. If a prior run
crashed (SIGKILL, power loss), the stale baseline is restored on the next
startup before fresh tuning is applied — so the host is never left
permanently tuned by a crashed process.

All parameters are passed in via the `config` dictionary (the
`system` section of the application config). Sensible defaults
are used when `config` is omitted.
"""

import atexit
import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_sysctl(key: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, timeout=5, text=True, check=True,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _read_file_current(path: str) -> Optional[str]:
    """Read a sysfs file's *current* value.

    Sysfs often writes 'opt1 [opt2] opt3' to mark the active value in
    brackets. Writing back the bracketed form is rejected by some
    kernels, so strip the marker.
    """
    try:
        content = Path(path).read_text().strip()
        m = re.search(r"\[([^\]]+)\]", content)
        if m:
            return m.group(1)
        return content or None
    except Exception:
        return None


def _write_sysctl(key: str, value: str) -> bool:
    try:
        subprocess.run(
            ["sysctl", "-w", f"{key}={value}"],
            check=True, capture_output=True, timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _write_file(path: str, value: str) -> bool:
    try:
        Path(path).write_text(value)
        return True
    except OSError as e:
        log.debug("Cannot write %s: %s", path, e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Baseline store — L-05
# ─────────────────────────────────────────────────────────────────────────────
class _BaselineStore:
    """Persist the originals so a crash can be undone."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> bool:
        try:
            if not self._path.exists():
                self._loaded = True
                return False
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {str(k): str(v) for k, v in raw.items()}
            self._loaded = True
            return True
        except Exception as exc:
            log.warning("Baseline load failed (%s): %s", self._path, exc)
            self._loaded = True
            return False

    def record(self, kind: str, key: str, original: Optional[str]) -> None:
        if original is None or original == "":
            return
        with self._lock:
            composite = f"{kind}:{key}"
            # First write wins — the earliest original is the true baseline.
            if composite not in self._data:
                self._data[composite] = original

    def save(self) -> None:
        with self._lock:
            data = dict(self._data)
        if not data:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:
            log.warning("Baseline save failed (%s): %s", self._path, exc)

    def entries(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# SystemOptimizer
# ─────────────────────────────────────────────────────────────────────────────
class SystemOptimizer:
    """Apply low-level OS tuning for maximum LLM throughput.

    All configuration is read from the `config` argument (dict or
    `SystemConfig` model). If omitted, sensible defaults are used.

    Every sysctl/sysfs write records the original to a persistent
    baseline; `shutdown()` restores it. A baseline left over from a
    crashed run is restored on the *next* startup before fresh tuning.
    """

    def __init__(
        self,
        dry_run: bool = False,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dry_run = dry_run
        self._is_root = os.geteuid() == 0

        # Normalise config
        if config is None:
            config = {}
        try:
            from local_llm_core.hal.config_loader import SystemConfig
            if isinstance(config, dict):
                cfg_model = SystemConfig(**config)
                self._cfg = cfg_model.model_dump()
            else:
                self._cfg = config.model_dump()
        except Exception as exc:
            log.warning("System config validation failed: %s; using defaults", exc)
            self._cfg = config if isinstance(config, dict) else {}

        # L-05 baseline. Path configurable via system.baseline_path.
        baseline_path_str = self._cfg.get("baseline_path") or str(
            Path.home() / ".cache" / "local-llm" / "system_baseline.json"
        )
        self._baseline = _BaselineStore(Path(baseline_path_str))

        # Crash recovery: if a stale baseline exists, restore it before
        # applying fresh tuning. This is the case where a previous run
        # was killed by SIGKILL or power loss and never called shutdown().
        if not self.dry_run and self._baseline.load():
            entries = self._baseline.entries()
            if entries:
                log.warning(
                    "Found stale baseline (%d entries) from a prior "
                    "crashed run at %s — restoring before tuning",
                    len(entries), baseline_path_str,
                )
                self._restore_baseline()
                self._baseline.clear()

        # Belt and suspenders: atexit runs on clean Python exit (uvicorn
        # shutdown, SIGTERM handled by uvicorn, etc.). SIGKILL skips
        # atexit but the stale-baseline branch above catches it next start.
        atexit.register(self.shutdown)

    # ── baseline write/restore ──────────────────────────────────────────
    def _apply_sysctl(self, key: str, value: str) -> bool:
        if self.dry_run:
            log.info("[DRY-RUN] sysctl %s=%s", key, value)
            return True
        if not self._is_root:
            log.debug("Skipping sysctl %s (not root)", key)
            return False
        original = _read_sysctl(key)
        if original and original != value:
            self._baseline.record("sysctl", key, original)
            self._baseline.save()
        log.info("Applying: sysctl %s=%s (was %r)", key, value, original)
        return _write_sysctl(key, value)

    def _apply_file(self, path: str, value: str) -> bool:
        if self.dry_run:
            log.info("[DRY-RUN] write %s=%s", path, value)
            return True
        if not self._is_root:
            log.debug("Skipping write %s (not root)", path)
            return False
        original = _read_file_current(path)
        if original is not None and original != value:
            self._baseline.record("file", path, original)
            self._baseline.save()
        log.info("Applying: write %s=%s (was %r)", path, value, original)
        return _write_file(path, value)

    def _restore_baseline(self) -> None:
        for composite, original in self._baseline.entries().items():
            kind, _, key = composite.partition(":")
            if not key:
                continue
            try:
                if kind == "sysctl":
                    ok = _write_sysctl(key, original)
                elif kind == "file":
                    ok = _write_file(key, original)
                else:
                    continue
                if ok:
                    log.info("Restored %s %s=%r", kind, key, original)
                else:
                    log.warning("Failed to restore %s %s", kind, key)
            except Exception as exc:
                log.warning("Restore %s %s failed: %s", kind, key, exc)

    def shutdown(self) -> None:
        """Restore every applied sysctl/sysfs value. Safe to call repeatedly."""
        if self.dry_run:
            log.info("[DRY-RUN] would restore %d baseline entries", len(self._baseline.entries()))
            return
        entries = self._baseline.entries()
        if not entries:
            return
        log.info("Restoring %d baseline entries on shutdown", len(entries))
        self._restore_baseline()
        self._baseline.clear()

    # ── the rest is unchanged in spirit; every write now goes through
    #    _apply_sysctl / _apply_file so the baseline is recorded ────────
    def _auto_huge_page_count(self) -> int:
        hp_cfg = self._cfg.get("huge_pages", {})
        pct = float(hp_cfg.get("auto_pct", 12.0))
        min_pages = int(hp_cfg.get("min_pages", 64))
        max_pages = int(hp_cfg.get("max_pages", 512))
        try:
            total_mb = psutil.virtual_memory().total >> 20
        except Exception:
            total_mb = 8192
        count = int(total_mb * (pct / 100.0))
        return max(min_pages, min(count, max_pages))

    def enable_huge_pages(self, count: int = 0) -> bool:
        # Gate: llama.cpp does not MAP_HUGETLB; a pinned pool steals RAM
        # from the mmap working set. Keep the knob but refuse to allocate
        # unless explicitly forced.
        if not self._cfg.get("force_huge_pages", False):
            log.info("Huge pages: skipped (llama.cpp does not use MAP_HUGETLB)")
            return False

        hp_cfg = self._cfg.get("huge_pages", {})
        warn_pct = float(hp_cfg.get("warn_threshold_pct", 15.0))
        info_pct = float(hp_cfg.get("info_threshold_pct", 10.0))
        if count == 0:
            count = self._auto_huge_page_count()
        else:
            try:
                total_mb = psutil.virtual_memory().total >> 20
                reserved_mb = count * 2
                pct = reserved_mb / max(total_mb, 1) * 100
                if pct > warn_pct:
                    log.warning(
                        "Huge pages: %d pages = %d MB = %.1f%% of %d MB RAM — "
                        "values > %.0f%% may OOM. Reduce count or use count=0.",
                        count, reserved_mb, pct, total_mb, warn_pct,
                    )
                elif pct > info_pct:
                    log.info("Huge pages: %d pages = %d MB = %.1f%% of RAM",
                             count, reserved_mb, pct)
            except Exception:
                pass
        return self._apply_file(
            "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages",
            str(count),
        )

    def set_transparent_huge_pages(self, mode: Optional[str] = None) -> bool:
        if not self._cfg.get("force_thp", False):
            log.info("THP: skipped (default is fine; tuning is PROVE-gated)")
            return False

        if mode is None:
            mode = self._cfg.get("thp_mode", "madvise")
        return self._apply_file(
            "/sys/kernel/mm/transparent_hugepage/enabled", mode,
        )

    def set_cpu_governor(self, governor: Optional[str] = None) -> bool:
        if governor is None:
            governor = self._cfg.get("cpu_governor", "performance")
        ok = True
        for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
            gov = cpu / "cpufreq/scaling_governor"
            if gov.exists():
                ok &= self._apply_file(str(gov), governor)
        return ok

    def set_io_scheduler(
        self, device: Optional[str] = None, scheduler: Optional[str] = None,
    ) -> bool:
        if scheduler is None:
            scheduler = self._cfg.get("io_scheduler", "mq-deadline")
        targets = [device] if device else [str(p) for p in Path("/sys/block").iterdir()]
        ok = True
        for dev in targets:
            scheduler_path = f"{dev}/queue/scheduler"
            if Path(scheduler_path).exists():
                ok &= self._apply_file(scheduler_path, scheduler)
        return ok

    def lock_memory(self) -> bool:
        import resource
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
            log.info("RLIMIT_MEMLOCK raised to %d", hard)
            return True
        except (ValueError, resource.error) as e:
            log.warning("Could not raise RLIMIT_MEMLOCK: %s", e)
            return False

    def tune_vm(self) -> bool:
        vm_cfg = self._cfg.get("vm", {})
        defaults = {
            "swappiness": "10",
            "dirty_ratio": "40",
            "dirty_background_ratio": "5",
            "vfs_cache_pressure": "50",
            "overcommit_memory": "1",
        }
        settings = {k: str(vm_cfg.get(k, defaults[k])) for k in defaults}
        ok = True
        for k, v in settings.items():
            ok &= self._apply_sysctl(f"vm.{k}", v)
        return ok

    def apply_all(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        if self._cfg.get("enable_huge_pages", True):
            results["huge_pages"] = self.enable_huge_pages()
        else:
            results["huge_pages"] = False

        if self._cfg.get("enable_thp", True):
            results["thp_madvise"] = self.set_transparent_huge_pages()

        if self._cfg.get("enable_cpu_governor", True):
            results["cpu_perf"] = self.set_cpu_governor()

        if self._cfg.get("enable_io_scheduler", True):
            results["io_sched"] = self.set_io_scheduler()

        if self._cfg.get("enable_memlock", True):
            results["memlock"] = self.lock_memory()

        if self._cfg.get("enable_vm_tune", True):
            results["vm_tune"] = self.tune_vm()

        # CPU isolation via kernel module removed — replaced by
        # systemd `CPUWeight=` / `AllowedCPUs=` . The .ko was never rebuilt successfully; the write
        # path is deleted rather than left as dead code.
        results["cpu_isolation"] = False

        return results

