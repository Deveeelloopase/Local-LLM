"""Resolve model profiles: discover, plan, calibrate, persist, apply.

:func:`resolve` is the single entry point used by the server at start-up and by
the command-line tools.  It

1. collects the profiles declared in ``models.profiles`` and, when
   ``models.auto.discover`` is on, adds every unconfigured ``*.gguf`` found in
   ``models.allowed_dir``;
2. probes the machine and plans every profile (:mod:`.planner`);
3. calibrates the models that will be resident and have no current benchmark
   (:mod:`.calibrate`), then plans again with the measurements;
4. writes the result to ``model_profiles.json`` (:mod:`.store`).

:func:`apply_to_config` then merges the plan into the in-memory configuration so
the rest of the server sees complete profiles.

Configuration (all optional)::

    models:
      auto:
        enabled: true            # false = use profiles exactly as written
        discover: true           # add unconfigured *.gguf files from allowed_dir
        max_autoload: 3          # models resident together
        kv_cache_quant: auto     # auto | f16 | q8_0 | q4_0
        ram_reserve_mb: 0        # 0 = derived from the machine
        context_cap: 32768       # used when model_loader.max_context_tokens is 0
        calibration:
          mode: startup          # startup | off
          scope: autoload        # autoload | all | none
          budget_s: 900
          variant_timeout_s: 420

Version: 1.0.0
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .calibrate import CalibrationResult, Calibrator, Progress, Runner
from .machine import MachineFacts
from .planner import AUTO_FIELDS, Plan, Planner, PlannerSettings, discover_gguf_files
from .store import BenchStore, bench_path, plan_path, write_plan

__version__ = "1.0.0"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationSettings:
    """``models.auto.calibration`` options."""

    mode: str = "startup"            # startup | off
    scope: str = "autoload"          # autoload | all | none
    budget_s: float = 900.0
    variant_timeout_s: float = 420.0
    tokens: int = 32
    repeats: int = 2

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CalibrationSettings":
        cal = (((cfg or {}).get("models") or {}).get("auto") or {}).get("calibration") or {}

        def num(key: str, default: float) -> float:
            try:
                return float(cal.get(key, default))
            except (TypeError, ValueError):
                return default

        mode = str(cal.get("mode", "startup")).lower()
        scope = str(cal.get("scope", "autoload")).lower()
        return cls(
            mode=mode if mode in ("startup", "off") else "startup",
            scope=scope if scope in ("autoload", "all", "none") else "autoload",
            budget_s=max(30.0, num("budget_s", 900.0)),
            variant_timeout_s=max(30.0, num("variant_timeout_s", 420.0)),
            tokens=int(max(8, num("tokens", 32))),
            repeats=int(max(1, num("repeats", 2))),
        )


@dataclass
class Resolution:
    """Everything :func:`resolve` produced."""

    plan: Plan
    machine: MachineFacts
    settings: PlannerSettings
    calibration: List[CalibrationResult] = field(default_factory=list)
    plan_file: Optional[Path] = None
    elapsed_s: float = 0.0
    notes: List[str] = field(default_factory=list)
    profiles: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine.to_dict(),
            "budgets": {"ram_mb": self.plan.ram_budget_mb, "device_mb": self.plan.device_budget_mb},
            "autoload_order": list(self.plan.autoload_order),
            "deferred": list(self.plan.deferred),
            "calibration": [c.to_dict() for c in self.calibration],
            "warnings": list(self.plan.warnings) + list(self.notes),
            "models": {k: v.to_dict() for k, v in self.plan.models.items()},
        }


# ── inputs ──────────────────────────────────────────────────────────────
def collect_profiles(cfg: Mapping[str, Any], settings: PlannerSettings) -> List[Dict[str, Any]]:
    """Declared profiles plus (optionally) discovered ones."""
    models = (cfg or {}).get("models") or {}
    declared = [dict(p) for p in (models.get("profiles") or []) if isinstance(p, Mapping)]
    if not settings.discover:
        return declared
    root = models.get("allowed_dir")
    if not root:
        return declared
    known = [str(p.get("path")) for p in declared if p.get("path")]
    for path in discover_gguf_files(root, known):
        declared.append({"path": str(path), "autoload": False, "discovered": True})
    return declared


def _profile_hint(cfg: Mapping[str, Any], system_info: Any) -> Dict[str, Any]:
    """Batch sizes and flash-attention flag the loader will use."""
    if system_info is None:
        return {}
    try:
        from local_llm_core.engine.hardware.hardware_profiles import detect_profile_from_system

        prof = detect_profile_from_system(system_info, (cfg or {}).get("hardware_profiles") or {})
        return {"n_batch": prof.n_batch, "n_ubatch": prof.n_ubatch, "flash_attn": prof.flash_attn}
    except Exception as exc:                       # the hint is optional
        log.info("autoconfig: loader profile hint unavailable (%s)", exc)
        return {}


# ── main entry point ────────────────────────────────────────────────────
def resolve(
    cfg: Mapping[str, Any],
    *,
    system_info: Any = None,
    machine: Optional[MachineFacts] = None,
    calibrate: str = "auto",
    progress: Optional[Progress] = None,
    runner: Optional[Runner] = None,
    write: bool = True,
    state_directory: Optional[Path] = None,
    force_calibration: bool = False,
    only: Optional[List[str]] = None,
    kv_types: Optional[List[str]] = None,
    budget_s: Optional[float] = None,
) -> Resolution:
    """Plan (and, per configuration, calibrate) every model profile.

    ``calibrate`` is ``"auto"`` (follow ``models.auto.calibration``), ``"off"``
    or ``"on"`` (calibrate regardless of the configured mode).  ``only``
    limits calibration to the given model ids, ``kv_types`` overrides the
    KV-cache types measured and ``budget_s`` the time budget.
    """
    t0 = time.monotonic()
    settings = PlannerSettings.from_config(cfg)
    cal_cfg = CalibrationSettings.from_config(cfg)
    notes: List[str] = []

    mach = machine or MachineFacts.probe(system_info, (cfg or {}).get("hardware"))
    conc_mode = str(((cfg or {}).get("concurrency") or {}).get("mode", "")).lower()
    if conc_mode == "process_pool" and settings.max_autoload != 1:
        settings = replace(settings, max_autoload=1)
        notes.append(
            "concurrency.mode is process_pool: one primary model is served, so only the "
            "highest-priority autoload model is kept resident (use single_process for several)"
        )

    profiles = collect_profiles(cfg, settings)
    store = BenchStore(bench_path(state_directory) if state_directory else None)

    if not settings.enabled:
        notes.append("models.auto.enabled is false: profiles are used as written")

    planner = Planner(mach, settings, store.records())
    plan = planner.plan(profiles)

    results: List[CalibrationResult] = []
    do_cal = settings.enabled and (calibrate == "on" or (calibrate == "auto" and cal_cfg.mode == "startup"))
    if do_cal and (cal_cfg.scope != "none" or only):
        cal = Calibrator(
            mach, store, settings, runner=runner,
            profile_hint=_profile_hint(cfg, system_info),
            variant_timeout_s=cal_cfg.variant_timeout_s,
            tokens=cal_cfg.tokens, repeats=cal_cfg.repeats, kv_types=kv_types,
        )
        results = cal.calibrate_plan(
            plan, scope=cal_cfg.scope, budget_s=budget_s or cal_cfg.budget_s,
            force=force_calibration, progress=progress, only=only,
        )
        if any(r.status in ("measured", "failed") for r in results):
            plan = Planner(mach, settings, store.records()).plan(profiles)

    res = Resolution(plan=plan, machine=mach, settings=settings, calibration=results,
                     elapsed_s=time.monotonic() - t0, notes=notes, profiles=profiles)
    if write:
        target = plan_path(state_directory) if state_directory else None
        res.plan_file = write_plan(res.to_dict(), target)
    return res


# ── applying ────────────────────────────────────────────────────────────
def merged_profiles(res: Resolution) -> List[Dict[str, Any]]:
    """The input profiles with every planned field filled in."""
    out: List[Dict[str, Any]] = []
    src = res.profiles
    seen_ids = list(res.plan.models.keys())
    for idx, raw in enumerate(src):
        mp = res.plan.models.get(seen_ids[idx]) if idx < len(seen_ids) else None
        if mp is None:
            out.append(dict(raw))
            continue
        prof = dict(raw)
        for key in AUTO_FIELDS:
            if key in mp.fields:
                prof[key] = mp.fields[key]
        prof["kv_cache_quant"] = mp.kv_cache_quant
        if mp.target_tps > 0:
            prof["target_tps"] = round(mp.target_tps, 3)
            prof["target_tps_source"] = mp.target_tps_source
        prof["autoload_requested"] = mp.autoload_requested
        if mp.autoload_requested and not mp.autoload_effective:
            prof["autoload"] = False
        prof["plan_status"] = mp.status
        out.append(prof)
    return out


def apply_to_config(cfg: Dict[str, Any], res: Resolution) -> List[Dict[str, Any]]:
    """Replace ``cfg['models']['profiles']`` in place with the resolved profiles.

    The list object is updated in place, so every reference taken earlier in
    start-up sees the same data.  Returns the merged list.
    """
    models = cfg.setdefault("models", {})
    current = models.get("profiles")
    merged = merged_profiles(res)
    if isinstance(current, list):
        current[:] = merged
    else:
        models["profiles"] = merged
    return merged


def summary_lines(res: Resolution) -> List[str]:
    """Human-readable plan summary for logs and the command line."""
    m = res.machine
    lines = [
        f"machine: {m.cpu_model} · {m.physical_cores}c/{m.logical_threads}t · "
        f"RAM {m.ram_total_mb} MB ({m.ram_channels or '?'}-ch"
        + (f", {m.ram_bandwidth_gbps:.1f} GB/s" if m.ram_bandwidth_gbps else "") + ") · "
        f"{m.backend}" + (f" ({m.gpu_name}{', unified' if m.uma else ''})" if m.gpu_name else ""),
        f"budgets: RAM {res.plan.ram_budget_mb} MB · device {res.plan.device_budget_mb} MB · "
        f"resident: {', '.join(res.plan.autoload_order) or 'none'}",
    ]
    for mp in res.plan.models.values():
        f = mp.fields
        tps = f"{mp.target_tps:.2f} tok/s ({mp.target_tps_source})" if mp.target_tps else "n/a"
        flag = "AUTOLOAD" if mp.autoload_effective else ("deferred" if mp.autoload_requested else "on demand")
        lines.append(
            f"  {mp.id:<24} {mp.status:<9} {flag:<9} ngl={f.get('num_gpu_layers')!s:<3} "
            f"ctx={f.get('context_length')!s:<6} kv={mp.kv_cache_quant:<5} "
            f"ram={f.get('ram_required_mb')} MB dev={mp.device_need_mb} MB speed={tps}"
        )
    for c in res.calibration:
        lines.append(f"  calibration {c.model_id}: {c.status} {c.message}".rstrip())
    lines.extend(f"  warning: {w}" for w in res.plan.warnings + res.notes)
    return lines
