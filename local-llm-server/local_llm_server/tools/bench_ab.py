"""A/B comparison of decode settings for one model.

Use it to answer a question the automatic calibration does not ask, for
example "is 3 threads faster than 4?" or "does flash attention help with this
KV-cache type?".  Every combination of the values given is loaded in its own
process and measured exactly like a calibration run; the table is sorted by
decode speed.  ``--apply`` records the winner so the server uses it.

Usage::

    llm-bench-ab --model qwen3-8b --kv f16,q8_0
    llm-bench-ab --model qwen3-8b --threads 4,3,2 --kv f16
    llm-bench-ab --model qwen3-8b --ngl 36,24,0 --apply
    llm-bench-ab --model /models/Foo-Q4_K_M.gguf --flash 0,1

Values default to the model's computed profile.  The service should be
stopped first; ``--even-if-running`` overrides the check.

Exit codes: 0 ok, 1 failure or nothing measured.

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ._common import detect_system_info, find_config, load_config, server_is_listening

__version__ = "1.0.0"


def _ints(text: Optional[str]) -> Optional[List[int]]:
    return [int(x) for x in text.split(",") if x.strip()] if text else None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-bench-ab", description="Compare decode settings for one model.")
    p.add_argument("--config", dest="config_path", default=None)
    p.add_argument("--model", required=True, help="model id from config, or a path to a .gguf file")
    p.add_argument("--kv", default=None, help="KV-cache types, e.g. f16,q8_0")
    p.add_argument("--ngl", default=None, help="GPU layer counts, e.g. 36,24,0")
    p.add_argument("--threads", default=None, help="thread counts, e.g. 4,3,2")
    p.add_argument("--flash", default=None, help="flash attention 0,1")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=420.0)
    p.add_argument("--apply", action="store_true", help="record the fastest variant for the server")
    p.add_argument("--even-if-running", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def run_ab(
    spec_base: Dict[str, Any],
    grid: Dict[str, Sequence[Any]],
    runner: Callable[[Dict[str, Any], float], Dict[str, Any]],
    timeout_s: float,
    on_row: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Measure every combination of ``grid``; returns rows sorted by decode speed."""
    keys = list(grid)
    rows: List[Dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        settings = dict(zip(keys, combo))
        out = runner({**spec_base, **settings}, timeout_s)
        row = {**settings, "ok": bool(out.get("ok")),
               "decode_tps": float(out.get("decode_tps", 0.0)),
               "prefill_tps": float(out.get("prefill_tps", 0.0)),
               "error": out.get("error", ""), "backend": out.get("backend", "")}
        rows.append(row)
        if on_row:
            on_row(row)
    return sorted(rows, key=lambda r: (not r["ok"], -r["decode_tps"]))


def main(argv: Optional[List[str]] = None) -> int:
    args, _unknown = _parser().parse_known_args(argv)
    cfg = load_config(args.config_path) if find_config(args.config_path) else {}

    if not args.even_if_running and cfg and server_is_listening(cfg):
        sys.stderr.write("The server appears to be running; stop it first or pass --even-if-running.\n")
        return 1

    from local_llm_core.engine.autoconfig import (
        BenchStore, MachineFacts, PlannerSettings, plan_profiles, resolve,
    )
    from local_llm_core.engine.autoconfig.calibrate import subprocess_runner

    sysinfo = detect_system_info()
    machine = MachineFacts.probe(sysinfo)
    settings = PlannerSettings.from_config(cfg)
    store = BenchStore()

    path = Path(args.model)
    if path.is_file():
        profiles: List[Dict[str, Any]] = [{"path": str(path), "autoload": True}]
        wanted = None
    else:
        profiles = [dict(p) for p in ((cfg.get("models") or {}).get("profiles") or [])]
        wanted = args.model
    res = resolve({**cfg, "models": {**(cfg.get("models") or {}), "profiles": profiles,
                                      "auto": {**((cfg.get("models") or {}).get("auto") or {}), "discover": False}}},
                  machine=machine, calibrate="off", write=False)
    mp = res.plan.models.get(wanted) if wanted else next(iter(res.plan.models.values()), None)
    if mp is None or mp.facts is None:
        sys.stderr.write(f"model '{args.model}' not found or unreadable\n")
        return 1

    f = mp.fields
    kvs = (args.kv.split(",") if args.kv else [mp.kv_cache_quant])
    ngls = _ints(args.ngl) or [int(f["num_gpu_layers"])]
    thr = _ints(args.threads) or [int(f["n_threads"])]
    flash = [bool(x) for x in (_ints(args.flash) or [1])]
    base = {"model_path": f["path"], "ctx": args.ctx, "n_layer": mp.facts.n_layer,
            "n_batch": 512, "n_ubatch": 512, "tokens": args.tokens, "repeats": args.repeats}
    grid = {"kv": kvs, "ngl": ngls, "threads": thr, "flash": flash}

    print(f"{mp.id}: {len(kvs) * len(ngls) * len(thr) * len(flash)} variant(s), "
          f"{args.tokens} tokens x {args.repeats} runs each")

    def show(r: Dict[str, Any]) -> None:
        status = f"{r['decode_tps']:.2f} tok/s decode, {r['prefill_tps']:.1f} prefill" if r["ok"] else f"FAILED {r['error']}"
        print(f"  kv={r['kv']:<5} ngl={r['ngl']:<3} threads={r['threads']:<2} flash={int(r['flash'])}  {status}", flush=True)

    rows = run_ab(base, grid, subprocess_runner, args.timeout_s, show)
    good = [r for r in rows if r["ok"]]
    if not good:
        sys.stderr.write("no variant could be measured\n")
        return 1
    best = good[0]
    print(f"\nfastest: kv={best['kv']} ngl={best['ngl']} threads={best['threads']} "
          f"flash={int(best['flash'])} -> {best['decode_tps']:.2f} tok/s")
    if args.apply:
        rec = {
            "model_path": f["path"], "model_id": mp.id, "quantization": mp.facts.quantization,
            "fingerprint": mp.facts.fingerprint, "hardware": machine.fingerprint(),
            "llama_cpp": machine.llama_cpp_version, "decode_tps": best["decode_tps"],
            "prefill_tps": best["prefill_tps"], "kv_cache_quant": best["kv"],
            "gpu_layers": best["ngl"], "verified_gpu_layers": best["ngl"],
            "threads": best["threads"], "flash_attn": best["flash"], "ctx": args.ctx,
            "backend": best["backend"] or machine.backend, "variants": good,
            "calibrated_by": "bench_ab",
        }
        if store.put(mp.facts.file_name, rec):
            print(f"recorded for the server in {store.path}")
        else:
            sys.stderr.write("could not write the benchmark store\n")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
