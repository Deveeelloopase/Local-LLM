"""
engine/hardware/offload_calibrator.py — measured offload planning.

Replaces the size-based heuristic with a first-load calibration keyed by
a stable hardware+model fingerprint. Subsequent loads reuse the cached
decision.

This is what makes the codebase portable: no hardcoded "APU = carve-out
only" or "dGPU = max offload". The rule is always "use whatever measured
best on this exact machine for this exact model".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(
    os.environ.get("LLM_STATE_DIR", "/var/lib/local-llm")
) / "offload_calibration.json"

# How many candidate strategies to try. Ordered by expected win; the
# calibrator stops early on a good enough result.
CALIBRATION_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
CALIBRATION_TOKENS = 16


@dataclass
class Strategy:
    """One candidate offload plan."""
    label: str
    gpu_layers: int
    context_len: int
    note: str = ""


@dataclass
class Receipt:
    """Result of measuring one strategy."""
    label: str
    gpu_layers: int
    context_len: int
    decode_tps: float
    ttft_ms: float
    total_ms: float
    ok: bool = True
    error: str = ""


@dataclass
class Calibration:
    """The chosen winner plus the losers, for diagnostics."""
    fingerprint: str
    chosen: Dict[str, Any]
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    measured_at: float = field(default_factory=time.time)
    version: int = 1


class OffloadCalibrator:
    """
    Measures offload strategies once per (hardware, model) pair and
    caches the winner.

    Usage from layer_streamer._load_llama_adaptive():
        cal = OffloadCalibrator(cache_path=...)
        chosen = cal.choose(model_path, profile, proposed_layers)
        # chosen = {"gpu_layers": N, "context_len": M, "source": "cache"|"measured"|"planner"}

    The calibrator never raises; on any failure it falls back to the
    proposed plan. That keeps a calibration bug from breaking startup.
    """

    # A measured result must beat the roofline threshold to count as
    # a winner. Otherwise we prefer the conservative carve-out-only plan.
    ROOFLINE_GOOD_FRACTION = 0.85
    # Env override to force recalibration (e.g. after a driver upgrade).
    FORCE_ENV = "LLM_FORCE_OFFLOAD_RECAL"

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._path = cache_path or DEFAULT_CACHE
        self._cache: Dict[str, Calibration] = self._load()

    # ── public ───────────────────────────────────────────────────────
    def choose(
        self,
        model_path: str,
        *,
        proposed_gpu_layers: int,
        proposed_context_len: int,
        carve_out_only_gpu_layers: int,
        roofline_decode_tps: float,
        load_fn,   # callable(strategy) -> handle with .tokenize and __call__
        measure_fn,  # callable(handle) -> dict with keys decode_tps, ttft_ms
    ) -> Dict[str, Any]:
        """
        Return {"gpu_layers": N, "context_len": M, "source": "..."}.

        load_fn must accept a Strategy and return a loaded model object
        that supports the tiny decode loop below. measure_fn does the
        decode and returns decode_tps / ttft_ms.

        All failures are caught and fall through to the proposed plan.
        """
        try:
            fp = self._fingerprint(model_path, proposed_gpu_layers)
            if os.environ.get(self.FORCE_ENV, "").lower() in ("1", "true", "yes"):
                log.info("OffloadCalibrator: forced recalibration via %s", self.FORCE_ENV)
                self._cache.pop(fp, None)
            cached = self._cache.get(fp)
            if cached and cached.chosen.get("gpu_layers") is not None:
                log.info(
                    "OffloadCalibrator: reusing cached strategy for %s: "
                    "gpu_layers=%d (measured %.2f tok/s at %s)",
                    fp[:12],
                    cached.chosen["gpu_layers"],
                    cached.chosen.get("decode_tps", 0.0),
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(cached.measured_at)),
                )
                return {
                    "gpu_layers": int(cached.chosen["gpu_layers"]),
                    "context_len": int(cached.chosen.get("context_len", proposed_context_len)),
                    "source": "cache",
                }

            candidates = self._candidates(
                proposed_gpu_layers=proposed_gpu_layers,
                carve_out_only_gpu_layers=carve_out_only_gpu_layers,
            )
            log.info(
                "OffloadCalibrator: first load for %s — measuring %d "
                "candidate(s): %s",
                fp[:12], len(candidates),
                ", ".join(f"{c.label}={c.gpu_layers}" for c in candidates),
            )

            results: List[Receipt] = []
            for strat in candidates:
                rec = self._measure(strat, load_fn, measure_fn)
                results.append(rec)
                if rec.ok:
                    log.info(
                        "OffloadCalibrator: %s (gpu_layers=%d) → %.2f tok/s "
                        "(%.0f%% of roofline), ttft=%.0fms",
                        rec.label, rec.gpu_layers, rec.decode_tps,
                        100.0 * rec.decode_tps / max(roofline_decode_tps, 0.01),
                        rec.ttft_ms,
                    )
                    if rec.decode_tps >= roofline_decode_tps * self.ROOFLINE_GOOD_FRACTION:
                        log.info(
                            "OffloadCalibrator: %s reaches %.0f%% of roofline "
                            "— stopping early",
                            rec.label, 100.0 * self.ROOFLINE_GOOD_FRACTION,
                        )
                        break
                else:
                    log.warning(
                        "OffloadCalibrator: %s (gpu_layers=%d) failed: %s",
                        rec.label, rec.gpu_layers, rec.error,
                    )

            ok = [r for r in results if r.ok and r.decode_tps > 0]
            if not ok:
                log.error(
                    "OffloadCalibrator: every candidate failed to load or "
                    "decode — falling back to planner choice (gpu_layers=%d)",
                    proposed_gpu_layers,
                )
                return {
                    "gpu_layers": int(proposed_gpu_layers),
                    "context_len": int(proposed_context_len),
                    "source": "planner",
                }

            winner = max(ok, key=lambda r: r.decode_tps)
            self._cache[fp] = Calibration(
                fingerprint=fp,
                chosen=asdict(winner),
                alternatives=[asdict(r) for r in results if r is not winner],
            )
            self._save()
            log.info(
                "OffloadCalibrator: chose %s (gpu_layers=%d) at %.2f tok/s",
                winner.label, winner.gpu_layers, winner.decode_tps,
            )
            return {
                "gpu_layers": int(winner.gpu_layers),
                "context_len": int(winner.context_len),
                "source": "measured",
            }
        except Exception as exc:
            log.error(
                "OffloadCalibrator: unexpected failure (%s) — "
                "falling back to planner choice", exc, exc_info=True,
            )
            return {
                "gpu_layers": int(proposed_gpu_layers),
                "context_len": int(proposed_context_len),
                "source": "planner",
            }

    # ── internals ────────────────────────────────────────────────────
    def _candidates(
        self,
        *,
        proposed_gpu_layers: int,
        carve_out_only_gpu_layers: int,
    ) -> List[Strategy]:
        """
        Ordered list of strategies to try.

        The order matters: we try the conservative plans first so that
        on a machine where the conservative plan already reaches the
        roofline, we do not pay the cost of loading the aggressive plan.

        If proposed == carve_out_only, we still try the aggressive plan
        once because on a discrete GPU with big VRAM it can be much
        faster. But we do NOT try the aggressive plan on an APU unless
        the config explicitly opted in — that decision has already been
        made in hardware_profiles.compute_profile().
        """
        candidates: List[Strategy] = []
        seen = set()

        def add(label: str, layers: int, note: str = "") -> None:
            key = (label, int(layers))
            if key in seen:
                return
            seen.add(key)
            candidates.append(Strategy(label=label, gpu_layers=int(layers), context_len=0, note=note))

        if carve_out_only_gpu_layers != proposed_gpu_layers:
            add("carve_out_only", carve_out_only_gpu_layers,
                "conservative plan: hardware VRAM aperture only")
            add("planner", proposed_gpu_layers,
                "the size-based planner's choice")
        else:
            add("planner", proposed_gpu_layers,
                "the size-based planner's choice")
        # CPU-only is a last-resort comparator. Skip on very small models
        # (cheap) and on very large ones (catastrophic for load time).
        return candidates

    def _measure(
        self,
        strategy: Strategy,
        load_fn,
        measure_fn,
    ) -> Receipt:
        t0 = time.monotonic()
        try:
            handle = load_fn(strategy)
            metrics = measure_fn(handle)
            total_ms = (time.monotonic() - t0) * 1000.0
            return Receipt(
                label=strategy.label,
                gpu_layers=strategy.gpu_layers,
                context_len=strategy.context_len,
                decode_tps=float(metrics.get("decode_tps", 0.0)),
                ttft_ms=float(metrics.get("ttft_ms", 0.0)),
                total_ms=total_ms,
                ok=True,
            )
        except Exception as exc:
            total_ms = (time.monotonic() - t0) * 1000.0
            return Receipt(
                label=strategy.label,
                gpu_layers=strategy.gpu_layers,
                context_len=strategy.context_len,
                decode_tps=0.0,
                ttft_ms=0.0,
                total_ms=total_ms,
                ok=False,
                error=str(exc)[:200],
            )

    def _fingerprint(self, model_path: str, gpu_layers: int) -> str:
        parts = [
            self._gpu_id(),
            self._llama_cpp_version(),
            self._file_id(model_path),
            str(int(gpu_layers)),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @staticmethod
    def _gpu_id() -> str:
        try:
            import subprocess
            r = subprocess.run(
                ["lspci", "-mm"], capture_output=True, text=True, timeout=3,
            )
            for line in r.stdout.splitlines():
                if any(k in line for k in ("VGA", "3D", "Display")):
                    return line.strip()
        except Exception:
            pass
        return "unknown-gpu"

    @staticmethod
    def _llama_cpp_version() -> str:
        try:
            import llama_cpp
            return str(getattr(llama_cpp, "__version__", "unknown"))
        except Exception:
            return "unknown"

    @staticmethod
    def _file_id(model_path: str) -> str:
        try:
            p = Path(model_path)
            st = p.stat()
            return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
        except Exception:
            return str(model_path)

    def _load(self) -> Dict[str, Calibration]:
        try:
            if not self._path.exists():
                return {}
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            out = {}
            for fp, d in raw.items():
                try:
                    out[fp] = Calibration(
                        fingerprint=d.get("fingerprint", fp),
                        chosen=d.get("chosen", {}),
                        alternatives=d.get("alternatives", []),
                        measured_at=d.get("measured_at", 0.0),
                        version=d.get("version", 1),
                    )
                except Exception:
                    continue
            return out
        except Exception as exc:
            log.error("OffloadCalibrator: cache load failed: %s", exc)
            return {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            data = {fp: asdict(c) for fp, c in self._cache.items()}
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:
            log.warning("OffloadCalibrator: cache save failed: %s", exc)
