"""Automatic calibration: measure what the server will actually run.

The planner derives settings from specifications.  Calibration checks them
against reality, one model at a time, with the *same* settings the loader will
use (layer count, KV-cache type, threads, batch sizes), and records the result
per model in the :class:`~.store.BenchStore`.

What a calibration does for one model
-------------------------------------
1. **Verify the layer count.**  The planned configuration is loaded in an
   isolated process.  If it cannot be loaded (typically device memory), the
   layer count is stepped down by a quarter until a load succeeds; the highest
   count that worked is recorded as ``verified_gpu_layers`` and caps future
   plans.
2. **Pick the KV-cache type.**  ``f16`` and ``q8_0`` are measured at the
   verified layer count.  Some back-ends (for example Vulkan on Vega) decode
   much more slowly with a quantised cache, so the choice is measured, never
   assumed.  The fastest wins; within ``TIE_MARGIN`` of the fastest the higher
   quality type is kept.
3. **Record** decode and prefill throughput, the winning settings and every
   variant tried.

Calibration is skipped when a current record exists (same model fingerprint,
same hardware fingerprint, same ``llama-cpp-python`` version).  It works within
a wall-clock budget: models that do not fit the budget are measured on a later
start, and each finished model is persisted immediately.

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .machine import MachineFacts
from .planner import ModelPlan, Plan, PlannerSettings
from .store import BenchStore

__version__ = "1.0.0"

log = logging.getLogger(__name__)

#: Variants within this fraction of the fastest are considered equal in speed.
TIE_MARGIN = 0.06
KV_QUALITY = {"f16": 3, "q8_0": 2, "q4_0": 1}
#: Fraction by which the layer count is reduced after a failed load.
_STEP_DOWN = 0.75

Runner = Callable[[Dict[str, Any], float], Dict[str, Any]]
Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class Variant:
    """One configuration to measure."""

    ngl: int
    kv: str
    threads: int
    flash: bool


@dataclass
class CalibrationResult:
    model_id: str
    status: str                     # measured | cached | failed | skipped | over_budget
    record: Optional[Dict[str, Any]] = None
    elapsed_s: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "status": self.status,
                "elapsed_s": round(self.elapsed_s, 1), "message": self.message,
                "decode_tps": (self.record or {}).get("decode_tps"),
                "kv_cache_quant": (self.record or {}).get("kv_cache_quant")}


# ── runner ──────────────────────────────────────────────────────────────
def subprocess_runner(spec: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Run one measurement in a fresh interpreter and parse its result line."""
    cmd = [sys.executable, "-m", "local_llm_core.engine.autoconfig.bench_worker",
           "--spec", json.dumps(spec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {int(timeout_s)} s"}
    except OSError as exc:
        return {"ok": False, "error": f"could not start worker: {exc}"}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            try:
                return json.loads(line[7:])
            except ValueError:
                break
    tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    return {"ok": False, "error": f"worker exited with {proc.returncode}: {tail[0][:200]}"}


# ── calibrator ──────────────────────────────────────────────────────────
class Calibrator:
    """Measures models and records the result."""

    def __init__(
        self,
        machine: MachineFacts,
        store: BenchStore,
        settings: Optional[PlannerSettings] = None,
        *,
        runner: Optional[Runner] = None,
        profile_hint: Optional[Mapping[str, Any]] = None,
        variant_timeout_s: float = 420.0,
        tokens: int = 32,
        repeats: int = 2,
        ctx_cap: int = 4096,
        kv_types: Optional[Sequence[str]] = None,
    ) -> None:
        self.machine = machine
        self.store = store
        self.settings = settings or PlannerSettings()
        self.runner = runner or subprocess_runner
        hint = dict(profile_hint or {})
        self.n_batch = int(hint.get("n_batch", 512))
        self.n_ubatch = int(hint.get("n_ubatch", 512))
        self.flash = bool(hint.get("flash_attn", False))
        self.variant_timeout_s = float(variant_timeout_s)
        self.tokens = int(tokens)
        self.repeats = int(repeats)
        self.ctx_cap = int(ctx_cap)
        self._kv_types = [k for k in (kv_types or ()) if k in KV_QUALITY]

    # ---- decisions ----------------------------------------------------
    def needs_calibration(self, mp: ModelPlan, force: bool = False) -> bool:
        fc = mp.facts
        if fc is None or mp.status not in ("ok",):
            return False
        if force:
            return True
        hw, ll = self.machine.fingerprint(), self.machine.llama_cpp_version
        if self.store.is_current(fc.file_name, fc.fingerprint, hw, ll):
            return False
        if self.store.recently_failed(fc.file_name, fc.fingerprint, hw, ll):
            return False
        return True

    def kv_choices(self) -> List[str]:
        """KV-cache types worth measuring."""
        if self._kv_types:
            return list(self._kv_types)
        if self.settings.kv_cache_quant != "auto":
            return [self.settings.kv_cache_quant]
        return ["f16", "q8_0"]

    def estimate_s(self, mp: ModelPlan) -> float:
        """Rough wall-clock estimate for calibrating one model."""
        gb = (mp.facts.file_size_bytes / 1e9) if mp.facts else 1.0
        tps = max(mp.roofline_tps or 1.0, 0.5)
        per_variant = 6.0 + gb * 4.0 + (self.tokens * (self.repeats + 2)) / tps
        return per_variant * len(self.kv_choices())

    # ---- one model ----------------------------------------------------
    def _spec(self, mp: ModelPlan, v: Variant) -> Dict[str, Any]:
        fc = mp.facts
        assert fc is not None
        ctx = min(int(mp.fields.get("context_length") or self.ctx_cap), self.ctx_cap)
        return {
            "model_path": mp.fields["path"], "ctx": max(512, ctx), "ngl": v.ngl,
            "n_layer": fc.n_layer, "threads": v.threads, "kv": v.kv, "flash": v.flash,
            "n_batch": self.n_batch, "n_ubatch": self.n_ubatch,
            "tokens": self.tokens, "repeats": self.repeats,
        }

    def calibrate_model(
        self, mp: ModelPlan, progress: Optional[Progress] = None
    ) -> CalibrationResult:
        """Measure ``mp`` and persist the outcome."""
        fc = mp.facts
        t0 = time.monotonic()
        if fc is None:
            return CalibrationResult(mp.id, "skipped", message="no model facts")
        say = progress or (lambda _m, _t: None)
        threads = int(mp.fields.get("n_threads") or self.machine.physical_cores)
        planned = int(mp.fields.get("num_gpu_layers") or 0)
        kvs = self.kv_choices()
        base_kv = "f16" if "f16" in kvs else kvs[0]
        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        # 1. verify the layer count with the first KV type
        ngl, verified = planned, None
        attempts = 0
        while verified is None:
            v = Variant(ngl, base_kv, threads, self.flash)
            say(f"{mp.id}: measuring {base_kv} KV, {ngl}/{fc.n_layer} layers", self.variant_timeout_s)
            out = self.runner(self._spec(mp, v), self.variant_timeout_s)
            if out.get("ok"):
                verified = ngl
                results.append(self._row(v, out))
                break
            errors.append(f"ngl={ngl}: {out.get('error', 'failed')}")
            attempts += 1
            nxt = int(ngl * _STEP_DOWN)
            if ngl <= 0 or attempts > 6:
                break
            ngl = nxt if nxt != ngl else ngl - 1
        if verified is None:
            rec = self._failure_record(mp, "; ".join(errors)[:400])
            self.store.put(fc.file_name, rec)
            return CalibrationResult(mp.id, "failed", rec, time.monotonic() - t0, errors[-1])

        # 2a. measure the other KV types at the verified layer count
        for kv in kvs:
            if kv == base_kv:
                continue
            v = Variant(verified, kv, threads, self.flash)
            say(f"{mp.id}: measuring {kv} KV, {verified}/{fc.n_layer} layers", self.variant_timeout_s)
            out = self.runner(self._spec(mp, v), self.variant_timeout_s)
            if out.get("ok"):
                results.append(self._row(v, out))
            else:
                errors.append(f"kv={kv}: {out.get('error', 'failed')}")

        # 2b. A-B-A: the first run paid cold-cache / shader-compile cost -> measure the base again
        if len(kvs) > 1 and results:
            v = Variant(verified, base_kv, threads, self.flash)
            say(f"{mp.id}: re-measuring {base_kv} KV (warm)", self.variant_timeout_s)
            out = self.runner(self._spec(mp, v), self.variant_timeout_s)
            if out.get("ok"):
                again = self._row(v, out)
                if again.get("decode_tps", 0.0) > results[0].get("decode_tps", 0.0):
                    results[0] = again

        best = self.pick_best(results)
        rec = {
            "model_path": mp.fields["path"], "model_id": mp.id,
            "quantization": fc.quantization, "fingerprint": fc.fingerprint,
            "hardware": self.machine.fingerprint(), "llama_cpp": self.machine.llama_cpp_version,
            "decode_tps": best["decode_tps"], "prefill_tps": best["prefill_tps"],
            "kv_cache_quant": best["kv"], "gpu_layers": best["ngl"],
            "verified_gpu_layers": verified, "threads": best["threads"],
            "flash_attn": best["flash"], "ctx": self._spec(mp, Variant(0, "f16", 1, False))["ctx"],
            "backend": best.get("backend") or self.machine.backend,
            "variants": results, "errors": errors, "calibrated_by": "autoconfig",
        }
        self.store.put(fc.file_name, rec)
        log.info("calibrated %s: %.2f tok/s decode, %.1f tok/s prefill, KV=%s, layers=%d/%d",
                 mp.id, rec["decode_tps"], rec["prefill_tps"], rec["kv_cache_quant"],
                 verified, fc.n_layer)
        return CalibrationResult(mp.id, "measured", rec, time.monotonic() - t0)

    @staticmethod
    def _row(v: Variant, out: Mapping[str, Any]) -> Dict[str, Any]:
        return {"ngl": v.ngl, "kv": v.kv, "threads": v.threads, "flash": v.flash,
                "decode_tps": float(out.get("decode_tps", 0.0)),
                "prefill_tps": float(out.get("prefill_tps", 0.0)),
                "backend": out.get("backend", ""), "offloaded": out.get("offloaded", ""),
                "load_s": out.get("load_s", 0.0)}

    @staticmethod
    def pick_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Fastest decode; within ``TIE_MARGIN`` of it, the better KV quality."""
        top = max(r["decode_tps"] for r in rows)
        near = [r for r in rows if r["decode_tps"] >= (1.0 - TIE_MARGIN) * top]
        return max(near, key=lambda r: (KV_QUALITY.get(r["kv"], 0), r["decode_tps"]))

    def _failure_record(self, mp: ModelPlan, error: str) -> Dict[str, Any]:
        fc = mp.facts
        assert fc is not None
        return {"model_path": mp.fields["path"], "model_id": mp.id, "fingerprint": fc.fingerprint,
                "hardware": self.machine.fingerprint(), "llama_cpp": self.machine.llama_cpp_version,
                "failed": True, "error": error, "decode_tps": 0.0, "calibrated_by": "autoconfig"}

    # ---- many models --------------------------------------------------
    def calibrate_plan(
        self,
        plan: Plan,
        *,
        scope: str = "autoload",
        budget_s: float = 900.0,
        force: bool = False,
        progress: Optional[Progress] = None,
        only: Optional[Sequence[str]] = None,
    ) -> List[CalibrationResult]:
        """Calibrate the models of ``plan`` that need it, within ``budget_s``.

        ``scope`` is ``"autoload"`` (the resident set), ``"all"`` or ``"none"``.
        ``only`` restricts the run to the given model ids (overrides ``scope``).
        """
        if only:
            wanted = set(only)
            order = [m for m in plan.models.values() if m.id in wanted]
            missing = wanted - {m.id for m in order}
            if missing:
                log.warning("calibration: unknown model id(s): %s", ", ".join(sorted(missing)))
        elif scope == "none":
            return []
        elif scope == "all":
            order = sorted(plan.models.values(), key=lambda m: -int(m.fields.get("priority", 0)))
        else:
            order = [plan.models[i] for i in plan.autoload_order]
        out: List[CalibrationResult] = []
        started = time.monotonic()
        done = 0
        for mp in order:
            if mp.facts is None or mp.status != "ok":
                continue
            if not self.needs_calibration(mp, force):
                out.append(CalibrationResult(mp.id, "cached", self.store.get(mp.facts.file_name)))
                continue
            spent = time.monotonic() - started
            if done and spent + self.estimate_s(mp) > budget_s:
                out.append(CalibrationResult(
                    mp.id, "over_budget",
                    message=f"time budget of {int(budget_s)} s used; will be measured on a later start"))
                continue
            out.append(self.calibrate_model(mp, progress))
            done += 1
        return out
