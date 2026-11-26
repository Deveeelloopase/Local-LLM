"""Calibrate models: measure decode speed with the settings the server will use.

The server calibrates automatically at start-up (``models.auto.calibration``),
so this tool is only needed to force a fresh measurement, to measure models
that are not marked ``autoload``, or to look at more KV-cache types.  Results
go to the same per-model store the server reads, so the next start uses them.

What is measured, per model
---------------------------
* the planned configuration (layer count, threads, batch sizes, context) is
  loaded in an isolated process; a configuration that does not load is stepped
  down in layer count until one does,
* decode speed (median of repeated runs after a warm-up) and prefill speed,
* every KV-cache type requested (default ``f16`` and ``q8_0``; ``--grid full``
  adds ``q4_0``) — the fastest wins, a near tie keeps the higher quality type.

Usage::

    llm-calibrate                       measure models that have no current record
    llm-calibrate --force               re-measure the resident models
    llm-calibrate --all                 every model, resident or not
    llm-calibrate --model qwen3-8b      one model (repeat the flag for several)
    llm-calibrate --grid full           also measure q4_0
    llm-calibrate --json-out r.json     write the full result as JSON

The service should be stopped first: a running model competes for memory and
bandwidth and lowers every number.  ``--even-if-running`` overrides the check.

Exit codes: 0 all models measured or already current, 1 failure, 2 at least
one model could not be measured.

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from ._common import detect_system_info, find_config, load_config, server_is_listening

__version__ = "1.0.0"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-calibrate",
        description="Measure decode speed per model and record it for the server.",
    )
    p.add_argument("--config", dest="config_path", default=None)
    p.add_argument("--model", action="append", default=None, metavar="ID",
                   help="model id to measure (repeatable)")
    p.add_argument("--all", action="store_true", help="measure every model, not only the resident set")
    p.add_argument("--force", action="store_true", help="re-measure even when a current record exists")
    p.add_argument("--grid", choices=["quick", "full"], default="quick",
                   help="quick: f16 and q8_0; full: also q4_0")
    p.add_argument("--budget-s", type=float, default=None, help="time budget in seconds")
    p.add_argument("--auto", action="store_true",
                   help="accepted for compatibility; calibration is always automatic")
    p.add_argument("--json-out", default=None, metavar="FILE")
    p.add_argument("--even-if-running", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args, _unknown = _parser().parse_known_args(argv)
    if find_config(args.config_path) is None:
        sys.stderr.write("config.yaml not found — pass --config FILE\n")
        return 1
    cfg = load_config(args.config_path)

    if not args.even_if_running and server_is_listening(cfg):
        sys.stderr.write(
            "The server appears to be running; its model would distort the measurements.\n"
            "Stop it first (for example `sudo systemctl stop local-llm`), or pass --even-if-running.\n"
        )
        return 1

    from local_llm_core.engine.autoconfig import resolve, summary_lines

    if args.all:
        (((cfg.setdefault("models", {})).setdefault("auto", {})).setdefault("calibration", {}))["scope"] = "all"

    def progress(msg: str, _timeout: float) -> None:
        sys.stderr.write(f"  measuring {msg}\n")
        sys.stderr.flush()

    try:
        res = resolve(
            cfg, system_info=detect_system_info(), calibrate="on", progress=progress,
            force_calibration=args.force, only=args.model,
            kv_types=["f16", "q8_0", "q4_0"] if args.grid == "full" else None,
            budget_s=args.budget_s,
        )
    except Exception as exc:
        sys.stderr.write(f"calibration failed: {type(exc).__name__}: {exc}\n")
        return 1

    print("\n".join(summary_lines(res)))
    print()
    failed = 0
    for c in res.calibration:
        rec = c.record or {}
        if c.status == "measured":
            print(f"  {c.model_id:<24} measured  {rec.get('decode_tps', 0):.2f} tok/s decode, "
                  f"{rec.get('prefill_tps', 0):.1f} tok/s prefill, KV={rec.get('kv_cache_quant')}, "
                  f"layers={rec.get('verified_gpu_layers')}  ({c.elapsed_s:.0f} s)")
            for v in rec.get("variants", []):
                print(f"      kv={v['kv']:<5} ngl={v['ngl']:<3} decode={v['decode_tps']:.2f} "
                      f"prefill={v['prefill_tps']:.1f}")
        elif c.status == "cached":
            print(f"  {c.model_id:<24} current   {rec.get('decode_tps', 0):.2f} tok/s "
                  f"(use --force to re-measure)")
        else:
            failed += c.status == "failed"
            print(f"  {c.model_id:<24} {c.status:<9} {c.message}")
    if not res.calibration:
        print("  nothing to measure (no resident models; use --all or --model ID)")
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(res.to_dict(), fh, indent=2, default=str)
            print(f"\nresult written to {args.json_out}")
        except OSError as exc:
            sys.stderr.write(f"could not write {args.json_out}: {exc}\n")
            return 1
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
