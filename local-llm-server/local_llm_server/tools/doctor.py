"""tools/doctor.py — environment and config diagnostics.

Runs independent checks against the config file, Python environment,
model files, and (optionally) the running server. Every check returns
an honest status — PASS / WARN / FAIL / SKIP. SKIP is used when a
dependency is missing, never disguised as a fabricated PASS.

Exit code:
  0 — all checks PASS or SKIP (WARN does not fail)
  1 — one or more checks FAIL

Per contract, doctor MUST work even when:
  * fastapi / uvicorn are not installed
  * the config file is missing or malformed
  * llama-cpp-python is not importable
  * the server is not running

Model checks understand automatic configuration: a profile may be as small as
``id`` / ``path`` / ``autoload``; the checks resolve the same plan the server
computes at start-up (without running any benchmark) and report on that.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


@dataclass
class Check:
    name: str
    status: str              # PASS | WARN | FAIL | SKIP
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _find_config(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in (
        Path.cwd() / "config.yaml",
        Path.home() / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ):
        if cand.is_file():
            return cand
    return None


# ── Checks ──────────────────────────────────────────────────────────────


def check_python_version() -> Check:
    v = sys.version_info
    if v >= (3, 10):
        return Check("python_version", "PASS", f"{v.major}.{v.minor}.{v.micro}")
    return Check(
        "python_version", "FAIL",
        f"{v.major}.{v.minor}.{v.micro} — requires >= 3.10",
    )


def check_config_exists(config_path: Optional[str]) -> Tuple[Check, Optional[Path]]:
    p = _find_config(config_path)
    if p is None:
        return Check(
            "config_exists", "FAIL",
            "no config.yaml found (searched: --config, cwd, ~, /etc/local-llm/)",
        ), None
    return Check("config_exists", "PASS", str(p)), p


def check_config_parses(p: Path) -> Tuple[Check, Optional[Dict[str, Any]]]:
    if _yaml is None:
        return Check(
            "config_parses", "SKIP",
            "PyYAML not installed — cannot parse config",
        ), None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return Check(
                "config_parses", "FAIL",
                f"top-level YAML is not a mapping (got {type(data).__name__})",
            ), None
        return Check("config_parses", "PASS", f"{len(data)} top-level sections"), data
    except Exception as exc:
        return Check("config_parses", "FAIL", str(exc)), None


_REQUIRED_SECTIONS = (
    "server", "api", "hardware", "hardware_profiles", "memory",
    "inference", "models", "model_loader", "conversation", "logging",
    "security",
)


def check_config_sections(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("config_sections", "SKIP", "config not loaded")
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        return Check(
            "config_sections", "WARN",
            f"missing section(s): {', '.join(missing)}",
        )
    return Check(
        "config_sections", "PASS",
        f"all {len(_REQUIRED_SECTIONS)} required present",
    )


def check_secrets_contracted(cfg: Optional[Dict[str, Any]]) -> Check:
    """api.key, api.master_key, api.password,
    conversation.password must be ${...} or empty."""
    if cfg is None:
        return Check("secrets_contracted", "SKIP", "config not loaded")
    api = (cfg.get("api") or {})
    conv = (cfg.get("conversation") or {})
    problems: List[str] = []
    for label, value in (
        ("api.key", api.get("key")),
        ("api.master_key", api.get("master_key")),
        ("api.password", api.get("password")),
        ("conversation.password", conv.get("password")),
    ):
        v = str(value or "")
        if v == "" or (v.startswith("${") and v.endswith("}")):
            continue
        problems.append(f"{label} has literal value")
    if problems:
        return Check("secrets_contracted", "FAIL", "; ".join(problems))
    return Check("secrets_contracted", "PASS", "no plaintext secrets in config")


def check_env_vars(cfg: Optional[Dict[str, Any]]) -> Check:
    """Verify ${...} references in config resolve from the environment."""
    if cfg is None:
        return Check("env_vars_resolvable", "SKIP", "config not loaded")
    api = (cfg.get("api") or {})
    conv = (cfg.get("conversation") or {})
    missing: List[str] = []
    for value in (
        api.get("key"), api.get("master_key"), api.get("password"),
        conv.get("password"),
    ):
        v = str(value or "")
        if not (v.startswith("${") and v.endswith("}")):
            continue
        var = v[2:-1].split(":", 1)[0]
        if var and not os.environ.get(var):
            missing.append(var)
    if missing:
        return Check(
            "env_vars_resolvable", "FAIL",
            f"env var(s) not set: {', '.join(sorted(set(missing)))}",
        )
    return Check("env_vars_resolvable", "PASS", "all ${...} references resolvable")


_PKGS_REQUIRED = (
    ("fastapi", "fastapi>=0.109.1"),
    ("uvicorn", "uvicorn[standard]>=0.27.0"),
    ("pydantic", "pydantic>=2.5.0"),
    ("yaml", "pyyaml>=6.0"),
    ("psutil", "psutil>=5.9.0"),
    ("aiosqlite", "aiosqlite>=0.19.0"),
)
_PKGS_OPTIONAL = (
    ("llama_cpp", "llama-cpp-python"),
    ("lz4", "lz4"),
    ("torch", "torch"),
    ("asyncpg", "asyncpg (only for postgresql storage)"),
)


def check_packages_required() -> Check:
    missing: List[str] = []
    for mod, pip_name in _PKGS_REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        return Check(
            "packages_required", "FAIL",
            f"missing: {', '.join(missing)} — run `pip install -r requirements.txt`",
        )
    return Check("packages_required", "PASS", f"all {len(_PKGS_REQUIRED)} present")


def check_packages_optional() -> Check:
    missing: List[str] = []
    for mod, pip_name in _PKGS_OPTIONAL:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        return Check(
            "packages_optional", "WARN",
            f"missing: {', '.join(missing)} (reduced functionality)",
        )
    return Check("packages_optional", "PASS", f"all {len(_PKGS_OPTIONAL)} present")

_PLAN_CACHE: Dict[int, Any] = {}

def _resolve_plan(cfg: Optional[Dict[str, Any]]):
    """Plan for ``cfg`` without benchmarking or writing files; ``None`` when unavailable."""
    if cfg is None:
        return None
    key = id(cfg)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    res = None
    try:
        from local_llm_core.engine.autoconfig import resolve

        info = None
        try:
            from local_llm_core.hal.hardware_detector import HardwareDetector

            info = HardwareDetector().detect_all()
        except Exception:
            info = None
        res = resolve(cfg, system_info=info, calibrate="off", write=False)
    except Exception:
        res = None
    _PLAN_CACHE[key] = res
    return res

def check_autoconfig(cfg: Optional[Dict[str, Any]]) -> Check:
    """The plan the server will compute: resident set, deferred models, problems."""
    if cfg is None:
        return Check("autoconfig", "SKIP", "config not loaded")
    res = _resolve_plan(cfg)
    if res is None:
        return Check("autoconfig", "SKIP", "automatic configuration unavailable")
    plan = res.plan
    bad = [f"{m.id}: {m.status}" for m in plan.models.values()
           if m.autoload_requested and m.status in ("missing", "unreadable")]
    if bad:
        return Check("autoconfig", "FAIL", "; ".join(bad))
    deferred = [f"{i} ({plan.models[i].deferred_reason})" for i in plan.deferred]
    resident = ", ".join(plan.autoload_order) or "none"
    if deferred:
        return Check("autoconfig", "WARN",
                     f"resident: {resident}; not resident: {'; '.join(deferred)}")
    return Check("autoconfig", "PASS",
                 f"resident: {resident} (RAM budget {plan.ram_budget_mb} MB, "
                 f"device {plan.device_budget_mb} MB)")

def check_benchmarks(cfg: Optional[Dict[str, Any]]) -> Check:
    """Resident models have a benchmark for this machine and library build."""
    res = _resolve_plan(cfg)
    if res is None:
        return Check("benchmarks", "SKIP", "automatic configuration unavailable")
    from local_llm_core.engine.autoconfig import BenchStore

    store, hw = BenchStore(), res.machine.fingerprint()
    missing = []
    for mid in res.plan.autoload_order:
        fc = res.plan.models[mid].facts
        if fc is None:
            continue
        if not store.is_current(fc.file_name, fc.fingerprint, hw, res.machine.llama_cpp_version):
            missing.append(mid)
    if not res.plan.autoload_order:
        return Check("benchmarks", "SKIP", "no resident models")
    if missing:
        return Check("benchmarks", "WARN",
                     f"no current benchmark for {', '.join(missing)} — measured automatically "
                     f"at the next start (or run llm-calibrate)")
    return Check("benchmarks", "PASS", "current for every resident model")

def check_performance_state() -> Check:
    """CPU governor and GPU power state; single-channel memory is reported too."""
    notes: List[str] = []
    warn = False
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        if gov.is_file():
            g = gov.read_text().strip()
            notes.append(f"governor={g}")
            if g not in ("performance",):
                warn = True
                notes.append("(decode speed is lower with this governor: cpupower frequency-set -g performance)")
    except OSError:
        pass
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")) if Path("/sys/class/drm").is_dir() else []:
        dpm = card / "device" / "power_dpm_force_performance_level"
        try:
            if dpm.is_file():
                level = dpm.read_text().strip()
                notes.append(f"gpu_dpm={level}")
                if level not in ("high", "manual"):
                    warn = True
                    notes.append("(GPU may clock down: echo high > " + str(dpm) + ")")
                break
        except OSError:
            continue
    if not notes:
        return Check("performance_state", "SKIP", "no cpufreq or DRM power controls found")
    return Check("performance_state", "WARN" if warn else "PASS", " ".join(notes))

def check_models_exist(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("models_exist", "SKIP", "config not loaded")

    profiles = (cfg.get("models") or {}).get("profiles") or []
    if not profiles:
        return Check("models_exist", "WARN", "no models.profiles configured")
    checked = 0
    missing: List[str] = []
    for p in profiles:
        if not p.get("autoload"):
            continue
        mid = p.get("id") or Path(str(p.get("path", ""))).stem or "?"
        path_str = p.get("path", "")
        if not path_str:
            missing.append(f"{mid}: path not set")
            continue
        checked += 1
        if not Path(path_str).exists():
            missing.append(f"{mid}: {path_str} not found")
    if missing:
        return Check(
            "models_exist", "FAIL",
            f"{len(missing)} autoload missing: " + "; ".join(missing),
        )
    if checked == 0:
        return Check(
            "models_exist", "SKIP",
            "no autoload models to check",
        )
    return Check("models_exist", "PASS", f"{checked} autoload present")


def check_allowed_dir(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("allowed_dir", "SKIP", "config not loaded")
    allowed = (cfg.get("models") or {}).get("allowed_dir")
    if not allowed:
        return Check(
            "allowed_dir", "WARN",
            "models.allowed_dir not set (model loading restricted)",
        )
    p = Path(allowed)
    if not p.is_dir():
        return Check("allowed_dir", "FAIL", f"{allowed} not a directory")
    if not os.access(allowed, os.R_OK):
        return Check("allowed_dir", "FAIL", f"{allowed} not readable")
    return Check("allowed_dir", "PASS", str(allowed))


def check_disk_space(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("disk_space", "SKIP", "config not loaded")
    mem = (cfg.get("memory") or {})
    swap_path = mem.get("ssd_swap_path") or (mem.get("swap") or {}).get("ssd_swap_path")
    max_gb = mem.get("ssd_swap_max_gb") or (mem.get("swap") or {}).get("ssd_swap_max_gb")
    if not swap_path:
        return Check("disk_space", "SKIP", "no ssd_swap_path configured")
    p = Path(swap_path)
    check_dir = p if p.exists() else p.parent
    try:
        usage = shutil.disk_usage(str(check_dir))
        free_gb = usage.free / (1024 ** 3)
    except Exception as exc:
        return Check("disk_space", "WARN", f"stat {check_dir}: {exc}")
    if max_gb:
        need = float(max_gb)
        if free_gb < need:
            return Check(
                "disk_space", "FAIL",
                f"{check_dir}: {free_gb:.1f}GB free, swap needs {need:.1f}GB",
            )
        return Check(
            "disk_space", "PASS",
            f"{check_dir}: {free_gb:.1f}GB free (swap needs {need:.1f}GB)",
        )
    return Check("disk_space", "PASS", f"{check_dir}: {free_gb:.1f}GB free")


def check_memory(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("memory", "SKIP", "config not loaded")
    try:
        import psutil
    except ImportError:
        return Check("memory", "SKIP", "psutil not installed")
    vm = psutil.virtual_memory()
    ram_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    need_mb = 0
    res = _resolve_plan(cfg)
    if res is not None and res.plan.autoload_order:
        # derived requirement of the whole resident set
        need_mb = sum(res.plan.models[i].ram_need_mb for i in res.plan.autoload_order)
    else:
        for p in (cfg.get("models") or {}).get("profiles") or []:
            if p.get("autoload") and p.get("ram_required_mb"):
                need_mb = max(need_mb, int(p["ram_required_mb"]))
    if need_mb > 0:
        need_gb = need_mb / 1024
        if avail_gb < need_gb:
            return Check(
                "memory", "FAIL",
                f"{avail_gb:.1f}GB available, autoload needs ~{need_gb:.1f}GB",
            )
        return Check(
            "memory", "PASS",
            f"{ram_gb:.1f}GB total, {avail_gb:.1f}GB avail "
            f"(needs ~{need_gb:.1f}GB)",
        )
    return Check("memory", "PASS", f"{ram_gb:.1f}GB total, {avail_gb:.1f}GB avail")


def check_gpu() -> Check:
    try:
        from local_llm_core.hal.hardware_detector import HardwareDetector
    except ImportError:
        return Check(
            "gpu", "SKIP",
            "local_llm_core not importable (pip install -e local-llm-core)",
        )
    try:
        info = HardwareDetector().detect_all()
        if info.gpus:
            names = ", ".join(
                g.name for g in info.gpus if getattr(g, "supported", True)
            )
            return Check(
                "gpu", "PASS", f"{info.recommended_backend}: {names}",
            )
        return Check("gpu", "WARN", "no GPUs detected — will run CPU-only")
    except Exception as exc:
        return Check("gpu", "FAIL", f"detection failed: {exc}")


def check_llama_cpp() -> Check:
    try:
        import llama_cpp
    except ImportError:
        return Check(
            "llama_cpp", "FAIL",
            "llama-cpp-python not installed (inference impossible)",
        )
    return Check(
        "llama_cpp", "PASS",
        f"v{getattr(llama_cpp, '__version__', 'unknown')}",
    )


def check_server_reachable(
    url: Optional[str], api_key: str, timeout: float,
) -> Check:
    if not url:
        return Check("server", "SKIP", "no --url provided")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        req = urllib.request.Request(f"{url}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
            status = body.get("status") or body.get("health_status") or "?"
            if status in ("ok", "healthy"):
                return Check("server", "PASS", f"{url} → {status}")
            if status in ("degraded", "loading"):
                return Check("server", "WARN", f"{url} → {status}")
            return Check("server", "FAIL", f"{url} → unexpected {status!r}")
    except urllib.error.HTTPError as e:
        return Check("server", "FAIL", f"{url}/health → HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Check("server", "FAIL", f"{url}/health → {e}")
    except Exception as e:
        return Check(
            "server", "FAIL",
            f"{url}/health → {type(e).__name__}: {e}",
        )


def check_db(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("database", "SKIP", "config not loaded")
    api = (cfg.get("api") or {})
    conv = (cfg.get("conversation") or {})
    storage = str(api.get("storage") or conv.get("storage") or "").lower()
    if storage in ("", "sqlite", "memory"):
        return Check("database", "PASS", f"storage={storage or 'none'}")
    if storage in ("postgresql", "postgres"):
        try:
            importlib.import_module("asyncpg")
        except ImportError:
            return Check(
                "database", "FAIL",
                "storage=postgresql but asyncpg not installed",
            )
        return Check("database", "PASS", f"storage={storage} (asyncpg OK)")
    if storage == "mysql":
        try:
            importlib.import_module("aiomysql")
        except ImportError:
            return Check(
                "database", "FAIL",
                "storage=mysql but aiomysql not installed",
            )
        return Check("database", "PASS", f"storage={storage} (aiomysql OK)")
    return Check("database", "WARN", f"unknown storage backend: {storage}")


def check_port_bindable(cfg: Optional[Dict[str, Any]]) -> Check:
    if cfg is None:
        return Check("port", "SKIP", "config not loaded")
    server = (cfg.get("server") or {})
    host = str(server.get("host", "127.0.0.1"))
    port_raw = server.get("port", "8080")
    try:
        port = int(port_raw)
    except (ValueError, TypeError):
        return Check("port", "FAIL", f"server.port is not int: {port_raw!r}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return Check("port", "PASS", f"{host}:{port} bindable")
    except OSError as exc:
        return Check(
            "port", "WARN",
            f"{host}:{port} in use ({exc.strerror or exc}) — expected if "
            f"a server is running; stop it first to verify a fresh bind",
        )


# ── Driver ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="llm-doctor",
        description="Diagnose environment, config, and server issues.",
    )
    p.add_argument("--config", dest="config_path", default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--master-key", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="Skip HTTP probes (no network calls).")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--debug", action="store_true")
    args, unknown = p.parse_known_args(argv)
    if args.debug and unknown:
        sys.stderr.write(f"doctor: ignoring unrecognised args: {unknown}\n")

    checks: List[Check] = [check_python_version()]
    chk_cfg, cfg_p = check_config_exists(args.config_path)
    checks.append(chk_cfg)
    cfg_data: Optional[Dict[str, Any]] = None
    if cfg_p is not None:
        chk_parse, cfg_data = check_config_parses(cfg_p)
        checks.append(chk_parse)
    checks.append(check_config_sections(cfg_data))
    checks.append(check_secrets_contracted(cfg_data))
    checks.append(check_env_vars(cfg_data))
    checks.append(check_packages_required())
    checks.append(check_packages_optional())
    checks.append(check_llama_cpp())
    checks.append(check_gpu())
    checks.append(check_allowed_dir(cfg_data))
    checks.append(check_models_exist(cfg_data))
    checks.append(check_autoconfig(cfg_data))
    checks.append(check_benchmarks(cfg_data))
    checks.append(check_performance_state())
    checks.append(check_db(cfg_data))
    checks.append(check_disk_space(cfg_data))
    checks.append(check_memory(cfg_data))
    checks.append(check_port_bindable(cfg_data))

    if args.quick:
        checks.append(Check("server", "SKIP", "--quick: HTTP probes skipped"))
    else:
        cfg_or_empty = cfg_data or {}
        server_cfg = cfg_or_empty.get("server") or {}
        api_cfg = cfg_or_empty.get("api") or {}
        url = (
            args.url
            or os.environ.get("LLM_SERVER_URL")
            or (
                f"http://{server_cfg.get('host', '127.0.0.1')}:"
                f"{server_cfg.get('port', '8080')}"
                if server_cfg else None
            )
        )
        api_key = (
            args.api_key
            or os.environ.get("LLM_API_KEY")
            or str(api_cfg.get("key", ""))
        )
        if api_key.startswith("${"):
            api_key = ""
        checks.append(check_server_reachable(
            url.rstrip("/") if url else None, api_key, args.timeout,
        ))

    failed = [c for c in checks if c.status == "FAIL"]
    warned = [c for c in checks if c.status == "WARN"]
    skipped = [c for c in checks if c.status == "SKIP"]
    passed = [c for c in checks if c.status == "PASS"]

    if args.json:
        print(json.dumps({
            "checks": [c.to_dict() for c in checks],
            "summary": {
                "passed": len(passed), "warned": len(warned),
                "failed": len(failed), "skipped": len(skipped),
                "total": len(checks),
            },
        }, indent=2))
    else:
        print()
        print("  Local LLM Cloud — doctor report")
        print("  " + "=" * 70)
        for c in checks:
            marker = {"PASS": "OK  ", "WARN": "WARN", "FAIL": "FAIL",
                      "SKIP": "SKIP"}[c.status]
            print(f"  [{marker}] {c.name:<26s}  {c.detail}")
        print("  " + "-" * 70)
        print(
            f"  {len(passed)} passed, {len(warned)} warned, "
            f"{len(failed)} failed, {len(skipped)} skipped "
            f"(of {len(checks)})"
        )
        print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
