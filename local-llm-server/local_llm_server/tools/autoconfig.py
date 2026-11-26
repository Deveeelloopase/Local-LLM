"""Compute, inspect and (optionally) calibrate model profiles.

``llm-autoconfig`` shows what the server will do at start-up: for every model
in ``models.profiles`` (plus any ``*.gguf`` in ``models.allowed_dir``) it reads
the GGUF header and the hardware, derives each profile field, decides which
``autoload`` models fit together and writes the result to
``model_profiles.json``.  Nothing has to be run by hand for the server itself:
it performs the same steps on every start.

Commands::

    llm-autoconfig                  plan and write model_profiles.json
    llm-autoconfig plan --yaml      also print the resolved profiles as YAML
    llm-autoconfig show             print the plan file written by the server
    llm-autoconfig explain ID       where each field of a model came from
    llm-autoconfig check            exit 1 when an autoload model cannot be resident

Options::

    --config FILE     config.yaml (default: auto-discover)
    --calibrate       also measure models that have no current benchmark
    --json            machine-readable output
    --no-write        do not write model_profiles.json

Exit codes: 0 ok, 1 problem found (``check``) or failure, 2 usage.

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from ._common import detect_system_info, find_config, load_config

__version__ = "1.0.0"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-autoconfig",
        description="Derive every model profile field from the GGUF files and the hardware.",
    )
    p.add_argument("command", nargs="?", default="plan",
                   choices=["plan", "show", "explain", "check"])
    p.add_argument("model_id", nargs="?", help="model id (explain)")
    p.add_argument("--config", dest="config_path", default=None)
    p.add_argument("--calibrate", action="store_true",
                   help="measure models that have no current benchmark")
    p.add_argument("--yaml", action="store_true", help="print resolved profiles as YAML")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def _profile_yaml(res) -> str:
    from local_llm_core.engine.autoconfig import AUTO_FIELDS, merged_profiles

    keys = list(AUTO_FIELDS) + ["kv_cache_quant", "target_tps"]
    lines: List[str] = ["models:", "  profiles:"]
    for prof in merged_profiles(res):
        first = True
        for k in keys:
            if k not in prof:
                continue
            v = prof[k]
            text = json.dumps(v) if not isinstance(v, str) or any(c in v for c in ':#"') else v
            lines.append(f"    {'- ' if first else '  '}{k}: {text}")
            first = False
        lines.append("")
    return "\n".join(lines)


def _explain(res, model_id: str) -> int:
    mp = res.plan.models.get(model_id)
    if mp is None:
        sys.stderr.write(f"unknown model id '{model_id}'. Known: {', '.join(res.plan.models)}\n")
        return 1
    print(f"{mp.id}  [{mp.status}]")
    for k, v in mp.fields.items():
        src = mp.sources.get(k)
        if src:
            print(f"  {k:<18} {v!s:<48} <- {src}")
    kv_src = "bench" if mp.target_tps_source == "bench" else "planner"
    print(f"  {'kv_cache_quant':<18} {mp.kv_cache_quant!s:<48} <- {kv_src}")
    print(f"  {'target_tps':<18} {mp.target_tps:<48.2f} <- {mp.target_tps_source or 'n/a'}")
    print(f"  memory: device {mp.device_need_mb} MB, RAM {mp.ram_need_mb} MB, "
          f"KV {mp.kv_mb_per_1k_tokens:.0f} MB per 1k tokens")
    if mp.facts is not None:
        f = mp.facts
        print(f"  gguf: arch={f.architecture} layers={f.n_layer} kv_layers={f.kv_layers} "
              f"kv_heads={f.n_kv_heads} head_dim={f.head_dim_k} trained_ctx={f.n_ctx_train}")
    for w in mp.warnings:
        print(f"  warning: {w}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args, _unknown = _parser().parse_known_args(argv)

    if args.command == "show":
        from local_llm_core.engine.autoconfig import plan_path, read_plan

        data = read_plan()
        if not data:
            sys.stderr.write(f"no plan file at {plan_path()} — start the server or run `llm-autoconfig`\n")
            return 1
        if args.json:
            print(json.dumps(data, indent=2))
            return 0
        print(f"plan file: {plan_path()}  (version {data.get('version')})")
        print(f"resident: {', '.join(data.get('autoload_order') or []) or 'none'}")
        for mid, m in (data.get("models") or {}).items():
            print(f"  {mid:<24} ngl={m.get('num_gpu_layers')!s:<3} ctx={m.get('context_length')!s:<6} "
                  f"kv={m.get('kv_cache_quant')} ram={m.get('ram_required_mb')} MB "
                  f"speed={m.get('target_tps')} ({m.get('target_tps_source') or 'n/a'})")
        return 0

    if find_config(args.config_path) is None:
        sys.stderr.write("config.yaml not found — pass --config FILE\n")
        return 1
    cfg = load_config(args.config_path)

    from local_llm_core.engine.autoconfig import resolve, summary_lines

    def progress(msg: str, _t: float) -> None:
        sys.stderr.write(f"  calibrating: {msg}\n")

    try:
        res = resolve(
            cfg, system_info=detect_system_info(),
            calibrate="on" if args.calibrate else "off",
            progress=progress, write=not args.no_write,
        )
    except Exception as exc:
        sys.stderr.write(f"autoconfig failed: {type(exc).__name__}: {exc}\n")
        return 1

    if args.command == "explain":
        if not args.model_id:
            sys.stderr.write("usage: llm-autoconfig explain MODEL_ID\n")
            return 2
        return _explain(res, args.model_id)

    problems: List[str] = []
    for mp in res.plan.models.values():
        if mp.autoload_requested and not mp.autoload_effective:
            problems.append(f"{mp.id}: {mp.deferred_reason or mp.status}")

    if args.json:
        out: Dict[str, Any] = res.to_dict()
        out["problems"] = problems
        print(json.dumps(out, indent=2, default=str))
    else:
        print("\n".join(summary_lines(res)))
        if res.plan_file:
            print(f"\nwritten: {res.plan_file}")
        if args.yaml:
            print("\n" + _profile_yaml(res))
    if args.command == "check" and problems:
        sys.stderr.write("\nautoload problems:\n" + "\n".join(f"  - {p}" for p in problems) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
