"""Model profile planner.

Given the profiles declared in ``config.yaml`` (which may be nearly empty),
the facts read from each GGUF header (:mod:`.gguf_inspect`) and the facts about
the machine (:mod:`.machine`), the planner derives every field of a model
profile and decides which of the models marked ``autoload`` can be resident
together.

Rules in brief
--------------
* A field written in ``config.yaml`` always wins.  A field that is omitted,
  ``null``, empty, ``0`` (numeric sizes), ``-1`` (``num_gpu_layers``) or
  ``"auto"`` is derived.  ``num_gpu_layers: 0`` is an explicit "CPU only".
* Layers come first: the planner offloads as many layers as the accelerator
  can hold *at the minimum context*, then spends any remaining memory on a
  longer context (never above the trained window, the configured ceiling, or
  what RAM and the accelerator can hold).
* The RAM budget is the smaller of "total minus a reserve for the OS and
  runtime" and ``memory.ram_utilization_target`` of total RAM, so the plan
  never promises more than the server will grant.
* Memory is accounted for both budgets.  Weights and KV cache placed on a
  unified-memory accelerator also live in system RAM, so they count against
  the RAM budget as well as the device budget.
* Models marked ``autoload`` are admitted in priority order while they fit.
  The context of each admitted model is sized from the headroom that is left,
  split evenly.  A model that does not fit is *deferred* (kept loadable on
  demand) and the reason is recorded.

The planner is pure: it performs no I/O except reading GGUF headers through
:func:`inspect_gguf`.

Version: 1.0.0
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .gguf_inspect import GGUFError, ModelFacts, inspect_gguf
from .machine import MachineFacts

__version__ = "1.0.0"

log = logging.getLogger(__name__)

MIB = float(1 << 20)

#: Every profile field the planner is responsible for.
AUTO_FIELDS: Tuple[str, ...] = (
    "id", "path", "format", "description", "autoload", "model_type",
    "quantization", "context_length", "num_gpu_layers", "n_threads",
    "use_mlock", "rope_scaling", "gpu_split", "priority", "ram_required_mb",
    "suppress_thinking", "tags",
)

_KV_FALLBACK_ORDER: Tuple[str, ...] = ("f16", "q8_0", "q4_0")
_CTX_ALIGN = 256
_ROOFLINE_EFFICIENCY = 0.85

# Approximate bits per weight, used only to size the token-embedding and
# output matrices when splitting weights between device and host.
_BPW_BY_LABEL: Dict[str, float] = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56,
    "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_0": 5.5, "Q5_1": 6.0,
    "Q4_K_M": 4.85, "Q4_K_S": 4.6, "Q4_0": 4.5, "Q4_1": 5.0,
    "Q3_K_M": 3.9, "Q3_K_L": 4.3, "Q3_K_S": 3.5, "Q2_K": 2.9,
}

_BITNET_ARCHS = frozenset({"bitnet", "bitnet-b1.58", "bitnet-25"})
_QUANT_SUFFIX_RE = re.compile(
    r"[-_.](?:i?q\d+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|i2_s|tl[12])$", re.IGNORECASE
)
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


# ═══════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PlannerSettings:
    """Tunables read from ``models.auto`` and ``model_loader``."""

    enabled: bool = True
    discover: bool = True
    ram_reserve_mb: int = 0          # 0 = derive from the machine
    context_floor: int = 4096        # model_loader.min_context_tokens
    context_ceiling: int = 0         # model_loader.max_context_tokens (0 = none)
    context_cap: int = 32768         # used when the ceiling is 0
    kv_cache_quant: str = "auto"     # auto | f16 | q8_0 | q4_0
    max_autoload: int = 3
    threads: int = 0                 # 0 = physical cores
    device_budget_mb: int = 0        # 0 = from the machine facts
    os_reserve_mb: int = 1024        # concurrency.os_reserve_mb
    runtime_per_worker_mb: int = 300 # concurrency.runtime_per_worker_mb
    ram_fraction: float = 0.75       # memory.ram_utilization_target
    device: str = "auto"             # models.auto.device: auto | cpu | gpu | hybrid
    gpu_layers: int = 0              # models.auto.gpu_layers: cap for hybrid (0 = auto-fit)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PlannerSettings":
        models = (cfg or {}).get("models") or {}
        auto = models.get("auto") or {}
        loader = (cfg or {}).get("model_loader") or {}
        conc = (cfg or {}).get("concurrency") or {}

        def num(src: Mapping[str, Any], key: str, default: int) -> int:
            try:
                return int(src.get(key, default))
            except (TypeError, ValueError):
                return default

        mem = (cfg or {}).get("memory") or {}
        try:
            frac = float(mem.get("ram_utilization_target", mem.get("ram_target", 0.75)))
        except (TypeError, ValueError):
            frac = 0.75
        frac = min(1.0, max(0.1, frac))

        kv = str(auto.get("kv_cache_quant", "auto") or "auto").lower()
        device = str(auto.get("device", "auto") or "auto").lower()
        if device not in ("auto", "cpu", "gpu", "hybrid"):
            device = "auto"
        try:
            gpu_layers = max(0, int(auto.get("gpu_layers", 0) or 0))
        except (TypeError, ValueError):
            gpu_layers = 0
        if kv not in ("auto", "f16", "q8_0", "q4_0"):
            kv = "auto"
        return cls(
            enabled=bool(auto.get("enabled", True)),
            discover=bool(auto.get("discover", True)),
            ram_reserve_mb=max(0, num(auto, "ram_reserve_mb", 0)),
            context_floor=max(256, num(loader, "min_context_tokens", 4096)),
            context_ceiling=max(0, num(loader, "max_context_tokens", 0)),
            context_cap=max(1024, num(auto, "context_cap", 32768)),
            kv_cache_quant=kv,
            max_autoload=max(1, num(auto, "max_autoload", 3)),
            threads=max(0, num(auto, "threads", 0)),
            device_budget_mb=max(0, num(auto, "device_budget_mb", 0)),
            os_reserve_mb=max(0, num(conc, "os_reserve_mb", 1024)),
            runtime_per_worker_mb=max(0, num(conc, "runtime_per_worker_mb", 300)),
            ram_fraction=frac,
            device=device,
            gpu_layers=gpu_layers,
        )


# ═══════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class ModelPlan:
    """Resolved profile for one model plus the reasoning behind it."""

    id: str
    fields: Dict[str, Any]
    sources: Dict[str, str]
    status: str = "ok"                    # ok | missing | unreadable | directory | insufficient_memory
    autoload_requested: bool = False
    autoload_effective: bool = False
    kv_cache_quant: str = "f16"
    target_tps: float = 0.0
    target_tps_source: str = ""
    roofline_tps: float = 0.0
    device_need_mb: int = 0
    ram_need_mb: int = 0
    kv_mb_per_1k_tokens: float = 0.0
    facts: Optional[ModelFacts] = field(default=None, repr=False)
    warnings: List[str] = field(default_factory=list)
    deferred_reason: str = ""
    discovered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.fields)
        d.update({
            "kv_cache_quant": self.kv_cache_quant,
            "target_tps": round(self.target_tps, 3),
            "target_tps_source": self.target_tps_source,
            "roofline_tps": round(self.roofline_tps, 2),
            "status": self.status,
            "autoload_requested": self.autoload_requested,
            "autoload_effective": self.autoload_effective,
            "device_need_mb": self.device_need_mb,
            "ram_need_mb": self.ram_need_mb,
            "kv_mb_per_1k_tokens": round(self.kv_mb_per_1k_tokens, 1),
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
            "deferred_reason": self.deferred_reason,
            "discovered": self.discovered,
        })
        if self.facts is not None:
            d["facts"] = self.facts.to_dict()
        return d


@dataclass
class Plan:
    """Complete plan for every configured and discovered model."""

    models: Dict[str, ModelPlan]
    autoload_order: List[str]
    deferred: List[str]
    ram_budget_mb: int
    device_budget_mb: int
    warnings: List[str] = field(default_factory=list)

    def by_id(self, model_id: str) -> Optional[ModelPlan]:
        return self.models.get(model_id)


# ═══════════════════════════════════════════════════════════════════════
# Field semantics
# ═══════════════════════════════════════════════════════════════════════
def is_auto(key: str, value: Any) -> bool:
    """True when ``value`` means "derive this field"."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "auto"):
        return True
    if key in ("context_length", "n_threads", "ram_required_mb"):
        return _as_int(value) == 0
    if key == "num_gpu_layers":
        return _as_int(value, 0) < 0
    if key == "tags":
        return isinstance(value, (list, tuple)) and len(value) == 0
    return False


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════
# Naming
# ═══════════════════════════════════════════════════════════════════════
def derive_model_id(file_name: str) -> str:
    """``Qwen3-8B-Q4_K_M.gguf`` -> ``qwen3-8b``."""
    stem = Path(file_name).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    stem = _SHARD_RE.sub("", stem)
    for _ in range(3):
        new = _QUANT_SUFFIX_RE.sub("", stem)
        if new == stem:
            break
        stem = new
    stem = re.sub(r"[\s_]+", "-", stem.strip()).strip("-.").lower()
    stem = re.sub(r"-{2,}", "-", stem)
    return stem or "model"


def unique_id(candidate: str, taken: Iterable[str]) -> str:
    used = set(taken)
    if candidate not in used:
        return candidate
    n = 2
    while f"{candidate}-{n}" in used:
        n += 1
    return f"{candidate}-{n}"


def discover_gguf_files(root: "str | Path", known_paths: Iterable[str] = (), max_depth: int = 2) -> List[Path]:
    """GGUF files below ``root`` that are not already configured.

    Skips multimodal projector files, tiny files and every shard of a split
    model except the first.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    known = set()
    for p in known_paths:
        try:
            known.add(str(Path(p).resolve()))
        except OSError:
            known.add(str(p))
    found: List[Path] = []
    root_depth = len(base.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).resolve().parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if not fn.lower().endswith(".gguf") or "mmproj" in fn.lower():
                continue
            m = _SHARD_RE.search(fn[:-5])
            if m and m.group(1) != "00001":
                continue
            full = Path(dirpath) / fn
            try:
                if full.stat().st_size < (1 << 20):
                    continue
                if str(full.resolve()) in known:
                    continue
            except OSError:
                continue
            found.append(full)
    return found


# ═══════════════════════════════════════════════════════════════════════
# Memory model
# ═══════════════════════════════════════════════════════════════════════
def _embed_mb(f: ModelFacts) -> float:
    bpw = _BPW_BY_LABEL.get(f.quantization, 4.85)
    return max(0.0, f.vocab_size * f.n_embd * bpw / 8.0 / MIB)


def _split_weights(f: ModelFacts) -> Tuple[float, float, float]:
    """(embedding_mb, output_mb, body_mb) for a model file."""
    total = float(f.file_size_mb)
    emb = min(_embed_mb(f), total * 0.20)
    out = min(emb, total * 0.14)
    body = max(total - emb - out, total * 0.5)
    return emb, out, body


def gpu_weights_mb(f: ModelFacts, ngl: int) -> float:
    """Weights that sit on the accelerator for ``ngl`` offloaded layers.

    The loader requests ``n_layer + 1`` when every layer is offloaded, which
    also places the output matrix on the device.
    """
    if ngl <= 0 or f.n_layer <= 0:
        return 0.0
    _emb, out, body = _split_weights(f)
    layers = min(ngl, f.n_layer)
    w = body * layers / f.n_layer
    if ngl >= f.n_layer:
        w += out
    return w


def compute_buffer_mb(f: ModelFacts, on_device: bool) -> float:
    """Scratch buffers for graph evaluation."""
    w = float(f.file_size_mb)
    return max(256.0, 0.045 * w) if on_device else max(128.0, 0.02 * w)


def _kv_mb(f: ModelFacts, kv: str, ctx: int) -> float:
    return f.kv_bytes_per_token(kv) * ctx / MIB


def _need(
    f: ModelFacts, m: MachineFacts, kv: str, ngl: int, ctx: int
) -> Tuple[float, float]:
    """(device_mb, host_ram_mb) needed to run ``f`` with the given settings."""
    on_dev = ngl > 0 and m.gpu_offload
    frac = min(1.0, ngl / f.n_layer) if (on_dev and f.n_layer) else 0.0
    kv_total = _kv_mb(f, kv, ctx)
    gw = gpu_weights_mb(f, ngl) if on_dev else 0.0
    dev = (gw + kv_total * frac + compute_buffer_mb(f, True)) if on_dev else 0.0
    weights = float(f.file_size_mb)
    if m.uma and on_dev:
        host = weights + kv_total + compute_buffer_mb(f, True)
    else:
        host = (weights - gw) + kv_total * (1.0 - frac) + compute_buffer_mb(f, on_dev)
    return dev, host


def _align(n: float, step: int = _CTX_ALIGN) -> int:
    return int(n) // step * step


# ═══════════════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class _Work:
    """Mutable per-model state while planning."""

    src: Dict[str, Any]
    plan: ModelPlan
    facts: Optional[ModelFacts]
    kv: str = "f16"
    ngl: int = 0
    ctx_floor: int = 0
    ctx_max: int = 0
    explicit_ctx: int = 0
    explicit_ngl: Optional[int] = None
    ngl_cap: Optional[int] = None
    is_draft: bool = False
    cpu_only: bool = False


class Planner:
    """Derives profiles and the autoload plan for one machine."""

    def __init__(
        self,
        machine: MachineFacts,
        settings: Optional[PlannerSettings] = None,
        bench: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.m = machine
        self.s = settings or PlannerSettings()
        self.bench = dict(bench or {})

    # ── budgets ─────────────────────────────────────────────────────────
    @property
    def ram_budget_mb(self) -> int:
        total = self.m.ram_total_mb
        if total <= 0:
            return 0
        reserve = self.s.ram_reserve_mb or max(
            self.s.os_reserve_mb + self.s.runtime_per_worker_mb, int(total * 0.12)
        )
        # Never above the server-wide cap (memory.ram_utilization_target), the
        # same budget the autoload gate and the RAM optimiser work against.
        return max(0, min(total - reserve, int(total * self.s.ram_fraction)))

    @property
    def device_budget_mb(self) -> int:
        if not self.m.gpu_offload:
            return 0
        return self.s.device_budget_mb or self.m.device_budget_mb

    # ── public ──────────────────────────────────────────────────────────
    def plan(self, profiles: Sequence[Mapping[str, Any]]) -> Plan:
        works: List[_Work] = []
        taken: List[str] = []
        warnings: List[str] = []
        for order, raw in enumerate(profiles):
            w = self._prepare(dict(raw), taken, order)
            taken.append(w.plan.id)
            works.append(w)

        ram_left = float(self.ram_budget_mb)
        dev_left = float(self.device_budget_mb)
        admitted: List[_Work] = []
        deferred: List[str] = []

        # priority: higher first; drafts after primaries; configuration order breaks ties
        def sort_key(iw: Tuple[int, _Work]) -> Tuple[int, int, int]:
            i, w = iw
            return (1 if w.is_draft else 0, -int(w.plan.fields["priority"]), i)

        candidates = [w for _, w in sorted(enumerate(works), key=sort_key)
                      if w.plan.autoload_requested and w.facts is not None
                      and w.plan.status in ("ok", "insufficient_memory")]

        for w in candidates:
            primaries = sum(1 for a in admitted if not a.is_draft)
            if not w.is_draft and primaries >= self.s.max_autoload:
                self._defer(w, f"max_autoload={self.s.max_autoload} reached")
                deferred.append(w.plan.id)
                continue
            ok, reason = self._try_admit(w, dev_left, ram_left)
            if not ok:
                self._defer(w, reason)
                deferred.append(w.plan.id)
                continue
            dev, host = _need(w.facts, self.m, w.kv, w.ngl, w.ctx_floor)
            dev_left -= dev
            ram_left -= host
            admitted.append(w)

        # grow context from the remaining headroom, split evenly
        if admitted:
            share_dev = max(0.0, dev_left) / len(admitted)
            share_ram = max(0.0, ram_left) / len(admitted)
        for w in admitted:
            self._grow_context(w, share_dev, share_ram)
            w.plan.autoload_effective = True

        # models that are not part of the resident set are planned on their own
        resident = {id(w) for w in admitted}
        for w in works:
            if id(w) in resident or w.facts is None or w.plan.status == "missing":
                continue
            self._plan_solo(w)

        order_ids = [w.plan.id for w in admitted]
        plans: Dict[str, ModelPlan] = {}
        for w in works:
            self._finalise(w)
            plans[w.plan.id] = w.plan
        for p in plans.values():
            warnings.extend(f"{p.id}: {x}" for x in p.warnings)
        return Plan(
            models=plans,
            autoload_order=order_ids,
            deferred=deferred,
            ram_budget_mb=self.ram_budget_mb,
            device_budget_mb=self.device_budget_mb,
            warnings=warnings,
        )

    # ── stage 1: facts + identity fields ────────────────────────────────
    def _prepare(self, src: Dict[str, Any], taken: List[str], order: int) -> _Work:
        path = str(src.get("path") or "")
        sources: Dict[str, str] = {}
        fields: Dict[str, Any] = {}
        warnings: List[str] = []
        facts: Optional[ModelFacts] = None
        status = "ok"

        p = Path(path) if path else None
        if p is None or not p.exists():
            status = "missing"
            warnings.append(f"model file not found: {path or '(no path)'}")
        elif p.is_dir():
            status = "directory"
        else:
            try:
                facts = inspect_gguf(p)
            except (GGUFError, OSError) as exc:
                status = "unreadable"
                warnings.append(f"cannot read GGUF header: {exc}")

        # identity -------------------------------------------------------
        mid = src.get("id")
        if is_auto("id", mid):
            base = facts.file_name if facts else (p.name if p else "model")
            mid = unique_id(derive_model_id(base), taken)
            sources["id"] = "derived"
        else:
            sources["id"] = "config"
        fields["id"] = str(mid)
        fields["path"] = path
        sources["path"] = "config" if src.get("path") else "derived"

        fmt = src.get("format")
        if is_auto("format", fmt):
            fmt = "gguf" if (facts or (p and p.suffix.lower() == ".gguf")) else ("bitnet" if status == "directory" else "gguf")
            sources["format"] = "detected"
        else:
            sources["format"] = "config"
        fields["format"] = fmt

        mtype = src.get("model_type")
        if is_auto("model_type", mtype):
            mtype = "bitnet" if (facts and _is_bitnet(facts)) else ("bitnet" if fmt == "bitnet" else "gguf")
            sources["model_type"] = "detected"
        else:
            sources["model_type"] = "config"
        fields["model_type"] = mtype

        quant = src.get("quantization")
        if is_auto("quantization", quant):
            quant = (facts.quantization if facts and facts.quantization else "auto")
            sources["quantization"] = "detected"
        else:
            sources["quantization"] = "config"
        fields["quantization"] = quant

        fields["autoload"] = bool(src.get("autoload", False))
        sources["autoload"] = "config"

        # description ----------------------------------------------------
        desc = src.get("description")
        if is_auto("description", desc):
            desc = _describe(facts, fields["id"], status)
            sources["description"] = "derived"
        else:
            sources["description"] = "config"
        fields["description"] = desc

        # tags / priority / thinking -------------------------------------
        tags = src.get("tags")
        explicit_tags = list(tags) if isinstance(tags, (list, tuple)) and tags else []
        if explicit_tags:
            fields["tags"] = explicit_tags
            sources["tags"] = "config"
        else:
            fields["tags"] = _derive_tags(facts)
            sources["tags"] = "derived"
        is_draft = "draft" in fields["tags"]

        if is_auto("priority", src.get("priority")):
            fields["priority"] = _derive_priority(facts, is_draft)
            sources["priority"] = "derived"
        else:
            fields["priority"] = _as_int(src.get("priority"), 10)
            sources["priority"] = "config"

        if src.get("suppress_thinking") is None:
            fields["suppress_thinking"] = bool(facts.thinking_capable) if facts else False
            sources["suppress_thinking"] = "derived"
        else:
            fields["suppress_thinking"] = bool(src["suppress_thinking"])
            sources["suppress_thinking"] = "config"

        # rope: the loader forwards this value as the rope base only when it
        # differs from 1.0, so the neutral value keeps the model's own scaling.
        if src.get("rope_scaling") is None:
            fields["rope_scaling"] = 1.0
            sources["rope_scaling"] = "derived"
        else:
            fields["rope_scaling"] = float(src["rope_scaling"])
            sources["rope_scaling"] = "config"

        # carry through non-planned keys untouched
        for k, v in src.items():
            fields.setdefault(k, v)

        plan = ModelPlan(
            id=fields["id"], fields=fields, sources=sources, status=status,
            autoload_requested=bool(fields["autoload"]), facts=facts, warnings=warnings,
            discovered=bool(src.get("discovered", False)),
        )
        w = _Work(src=src, plan=plan, facts=facts, is_draft=is_draft)
        w.explicit_ctx = 0 if is_auto("context_length", src.get("context_length")) else _as_int(src["context_length"])
        ngl_raw = src.get("num_gpu_layers")
        w.explicit_ngl = None if is_auto("num_gpu_layers", ngl_raw) else _as_int(ngl_raw)
        w.cpu_only = (mtype == "bitnet") or str(src.get("device", "")).lower() == "cpu"

        if facts is not None:
            self._prepare_memory(w)
        else:
            self._prepare_without_facts(w)
        return w

    def _prepare_without_facts(self, w: _Work) -> None:
        """Directories and unreadable files: keep explicit values, derive the rest simply."""
        f, src, srcs = w.plan.fields, w.src, w.plan.sources
        size_mb = 0
        p = Path(f["path"]) if f["path"] else None
        try:
            if p and p.is_dir():
                size_mb = int(sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) >> 20)
            elif p and p.is_file():
                size_mb = int(p.stat().st_size >> 20)
        except OSError:
            size_mb = 0
        self._fill_simple(w, size_mb)

    def _fill_simple(self, w: _Work, size_mb: int) -> None:
        f, src, srcs = w.plan.fields, w.src, w.plan.sources
        if is_auto("context_length", src.get("context_length")):
            f["context_length"] = self.s.context_floor
            srcs["context_length"] = "derived"
        else:
            f["context_length"] = _as_int(src["context_length"]); srcs["context_length"] = "config"
        if w.explicit_ngl is None:
            f["num_gpu_layers"] = 0; srcs["num_gpu_layers"] = "derived"
        else:
            f["num_gpu_layers"] = w.explicit_ngl; srcs["num_gpu_layers"] = "config"
        self._fill_threads_mlock_split(w, fully_offloaded=False)
        if is_auto("ram_required_mb", src.get("ram_required_mb")):
            f["ram_required_mb"] = int(size_mb * 1.15) if size_mb else 0
            srcs["ram_required_mb"] = "derived"
        else:
            f["ram_required_mb"] = _as_int(src["ram_required_mb"]); srcs["ram_required_mb"] = "config"
        w.plan.ram_need_mb = int(f["ram_required_mb"])

    # ── stage 2: per-model memory-dependent fields ─────────────────────
    def _prepare_memory(self, w: _Work) -> None:
        fc = w.facts
        assert fc is not None
        s = self.s
        # KV type: configured, else bench-selected, else f16
        want = s.kv_cache_quant
        rec = self._bench_for(fc)
        if want == "auto":
            want = str((rec or {}).get("kv_cache_quant") or "f16").lower()
            if want not in _KV_FALLBACK_ORDER:
                want = "f16"
        w.kv = want

        ceiling = s.context_ceiling or s.context_cap
        trained = fc.n_ctx_train or ceiling
        w.ctx_max = max(256, _align(min(trained, ceiling)))
        w.ctx_floor = max(256, min(_align(s.context_floor), w.ctx_max))
        if w.explicit_ctx:
            w.ctx_floor = w.ctx_max = w.explicit_ctx

        # Operator device policy (models.auto.device). A per-profile
        # num_gpu_layers / device always wins over this global policy.
        if w.explicit_ngl is None:
            _dev = self.s.device
            if _dev in ("gpu", "hybrid") and not self.m.gpu_offload:
                w.plan.warnings.append(
                    "models.auto.device=%s requested but no usable GPU backend "
                    "was detected; running on CPU" % _dev)
            if _dev == "cpu":
                w.cpu_only = True
            elif _dev == "gpu" and fc.n_layer:
                w.explicit_ngl = fc.n_layer
            elif _dev == "hybrid" and self.s.gpu_layers > 0:
                w.explicit_ngl = (min(self.s.gpu_layers, fc.n_layer)
                                  if fc.n_layer else self.s.gpu_layers)
        offload_ok = self.m.gpu_offload and not w.cpu_only
        if rec is not None and rec.get("verified_gpu_layers") is not None:
            w.ngl_cap = max(0, int(rec["verified_gpu_layers"]))
        if w.explicit_ngl is not None:
            w.ngl = w.explicit_ngl if offload_ok or w.explicit_ngl == 0 else 0
        else:
            w.ngl = fc.n_layer if offload_ok else 0
            if w.ngl_cap is not None:
                w.ngl = min(w.ngl, w.ngl_cap)

    def _try_admit(self, w: _Work, dev_left: float, ram_left: float) -> Tuple[bool, str]:
        fc = w.facts
        assert fc is not None
        explicit = w.explicit_ngl
        # largest layer count that fits the remaining device memory
        for kv in self._kv_candidates(w):
            ngl = self._max_layers(w, kv, dev_left) if explicit is None else w.ngl
            dev, host = _need(fc, self.m, kv, ngl, w.ctx_floor)
            if host <= ram_left and (dev <= dev_left or ngl == 0 or explicit is not None):
                w.kv, w.ngl = kv, ngl
                preferred = self._kv_candidates(w)[0]
                if kv != preferred and self.s.kv_cache_quant == "auto":
                    w.plan.warnings.append(
                        f"KV cache set to {kv}: {self._pretty(preferred)} does not fit"
                    )
                return True, ""
        dev, host = _need(fc, self.m, w.kv, w.ngl, w.ctx_floor)
        return False, (
            f"needs ~{int(host)} MB RAM at the minimum context but only "
            f"{int(max(ram_left, 0))} MB remain in the budget"
        )

    @staticmethod
    def _pretty(kv: str) -> str:
        return kv

    def _kv_candidates(self, w: _Work) -> List[str]:
        if self.s.kv_cache_quant != "auto":
            return [self.s.kv_cache_quant]
        first = w.kv if w.kv in _KV_FALLBACK_ORDER else "f16"
        rest = [k for k in _KV_FALLBACK_ORDER if k != first]
        return [first] + rest

    def _max_layers(self, w: _Work, kv: str, dev_left: float) -> int:
        fc = w.facts
        assert fc is not None
        if not self.m.gpu_offload or w.cpu_only:
            return 0
        top = fc.n_layer if w.ngl_cap is None else min(fc.n_layer, w.ngl_cap)
        for ngl in range(top, 0, -1):
            dev, _host = _need(fc, self.m, kv, ngl, w.ctx_floor)
            if dev <= dev_left:
                return ngl
        return 0

    def _grow_context(self, w: _Work, share_dev: float, share_ram: float) -> None:
        fc = w.facts
        assert fc is not None
        if w.explicit_ctx:
            w.ctx_floor = w.ctx_max = w.explicit_ctx
            return
        bpt = fc.kv_bytes_per_token(w.kv)
        if bpt <= 0:
            w.ctx_floor = w.ctx_max
            return
        on_dev = w.ngl > 0 and self.m.gpu_offload
        frac = min(1.0, w.ngl / fc.n_layer) if (on_dev and fc.n_layer) else 0.0
        per_tok_dev = bpt * frac / MIB
        per_tok_ram = (bpt if (self.m.uma and on_dev) else bpt * (1.0 - frac)) / MIB
        extra = float("inf")
        if per_tok_dev > 0:
            extra = min(extra, share_dev / per_tok_dev)
        if per_tok_ram > 0:
            extra = min(extra, share_ram / per_tok_ram)
        ctx = w.ctx_floor + (0 if extra == float("inf") else int(extra))
        ctx = min(ctx, w.ctx_max) if extra != float("inf") else w.ctx_max
        w.ctx_floor = max(w.ctx_floor, _align(ctx))

    def _plan_solo(self, w: _Work) -> None:
        """Plan a model against the full budgets (loaded on its own)."""
        fc = w.facts
        assert fc is not None
        ram = float(self.ram_budget_mb)
        dev = float(self.device_budget_mb)
        for kv in self._kv_candidates(w):
            ngl = self._max_layers(w, kv, dev) if w.explicit_ngl is None else w.ngl
            d, h = _need(fc, self.m, kv, ngl, w.ctx_floor)
            if h <= ram:
                w.kv, w.ngl = kv, ngl
                break
        else:
            w.plan.status = "insufficient_memory" if w.plan.status == "ok" else w.plan.status
            w.plan.warnings.append(
                f"does not fit this machine even on its own "
                f"(needs ~{int(h)} MB, budget {int(ram)} MB)"
            )
        d, h = _need(fc, self.m, w.kv, w.ngl, w.ctx_floor)
        self._grow_context(w, max(0.0, dev - d), max(0.0, ram - h))

    def _defer(self, w: _Work, reason: str) -> None:
        w.plan.autoload_effective = False
        w.plan.deferred_reason = reason
        w.plan.warnings.append(f"autoload deferred: {reason}")

    # ── stage 3: write the final field values ───────────────────────────
    def _finalise(self, w: _Work) -> None:
        p, f, src, srcs = w.plan, w.plan.fields, w.src, w.plan.sources
        fc = w.facts
        if fc is None:
            p.kv_cache_quant = self.s.kv_cache_quant if self.s.kv_cache_quant != "auto" else "f16"
            return

        # context
        if w.explicit_ctx:
            f["context_length"] = w.explicit_ctx; srcs["context_length"] = "config"
        else:
            f["context_length"] = int(w.ctx_floor); srcs["context_length"] = "derived"

        # layers
        if w.explicit_ngl is not None:
            f["num_gpu_layers"] = w.explicit_ngl; srcs["num_gpu_layers"] = "config"
        else:
            f["num_gpu_layers"] = int(w.ngl); srcs["num_gpu_layers"] = "derived"
        ngl = int(f["num_gpu_layers"])

        full = ngl >= fc.n_layer > 0 and self.m.gpu_offload
        self._fill_threads_mlock_split(w, fully_offloaded=full)

        dev, host = _need(fc, self.m, w.kv, ngl, int(f["context_length"]))
        p.device_need_mb = int(round(dev))
        if is_auto("ram_required_mb", src.get("ram_required_mb")):
            f["ram_required_mb"] = int(round(host)); srcs["ram_required_mb"] = "derived"
        else:
            f["ram_required_mb"] = _as_int(src["ram_required_mb"]); srcs["ram_required_mb"] = "config"
        p.ram_need_mb = int(round(host))
        p.kv_cache_quant = w.kv
        p.kv_mb_per_1k_tokens = fc.kv_bytes_per_token(w.kv) * 1000 / MIB

        # speed: measured when a record exists, else the bandwidth roofline
        gb = max(fc.file_size_bytes, 1) / 1e9
        if self.m.ram_bandwidth_gbps > 0:
            p.roofline_tps = self.m.ram_bandwidth_gbps * _ROOFLINE_EFFICIENCY / gb
        rec = self._bench_for(fc)
        if rec and float(rec.get("decode_tps", 0) or 0) > 0:
            p.target_tps = float(rec["decode_tps"]); p.target_tps_source = "bench"
        elif p.roofline_tps > 0:
            p.target_tps = p.roofline_tps; p.target_tps_source = "roofline"

        # advisories
        if fc.n_layer and 0 < ngl < fc.n_layer and self.m.gpu_offload:
            p.warnings.append(
                f"only {ngl} of {fc.n_layer} layers fit the accelerator; "
                "decode will be limited by system-memory bandwidth"
            )
        if self.m.single_channel and ngl < fc.n_layer:
            p.warnings.append("single-channel RAM: CPU-resident layers decode slowly")
        if fc.n_ctx_train and w.explicit_ctx > fc.n_ctx_train > 0:
            p.warnings.append(
                f"context_length {w.explicit_ctx} exceeds the trained window {fc.n_ctx_train}"
            )

    def _fill_threads_mlock_split(self, w: _Work, fully_offloaded: bool) -> None:
        f, src, srcs = w.plan.fields, w.src, w.plan.sources
        if is_auto("n_threads", src.get("n_threads")):
            n = self.s.threads or self.m.physical_cores
            f["n_threads"] = int(max(1, min(n, self.m.logical_threads or n)))
            srcs["n_threads"] = "derived"
        else:
            f["n_threads"] = _as_int(src["n_threads"]); srcs["n_threads"] = "config"

        if src.get("use_mlock") is None:
            f["use_mlock"] = self._mlock_ok(w, fully_offloaded)
            srcs["use_mlock"] = "derived"
        else:
            f["use_mlock"] = bool(src["use_mlock"]); srcs["use_mlock"] = "config"

        if is_auto("gpu_split", src.get("gpu_split")) or src.get("gpu_split") == []:
            f["gpu_split"] = None
            srcs["gpu_split"] = "derived"
        else:
            f["gpu_split"] = list(src["gpu_split"]); srcs["gpu_split"] = "config"

    def _mlock_ok(self, w: _Work, fully_offloaded: bool) -> bool:
        """Lock weights only when they stay on the host and the limit allows it."""
        if fully_offloaded or (w.plan.fields.get("num_gpu_layers") or 0) > 0 and self.m.uma:
            return False
        size = w.facts.file_size_mb if w.facts else 0
        if size <= 0:
            return False
        lim = self.m.memlock_mb
        if lim == 0:                      # unknown
            return False
        if lim != -1 and lim < size:
            return False
        return size <= self.ram_budget_mb * 0.6

    def _bench_for(self, fc: ModelFacts) -> Optional[Mapping[str, Any]]:
        rec = self.bench.get(fc.fingerprint) or self.bench.get(fc.file_name)
        if not rec or rec.get("failed"):
            return None
        if rec.get("fingerprint") and rec["fingerprint"] != fc.fingerprint:
            return None
        hw = rec.get("hardware")
        if hw and hw != self.m.fingerprint():
            return None
        ll = rec.get("llama_cpp")
        if ll and self.m.llama_cpp_version and ll != self.m.llama_cpp_version:
            return None
        return rec


# ═══════════════════════════════════════════════════════════════════════
# Derivation helpers
# ═══════════════════════════════════════════════════════════════════════
def _is_bitnet(f: ModelFacts) -> bool:
    return f.architecture.lower() in _BITNET_ARCHS or f.quantization == "I2_S"


def _describe(f: Optional[ModelFacts], model_id: str, status: str) -> str:
    if f is None:
        return f"{model_id} ({status})"
    head = f.name or f.basename or model_id
    bits = [f"{f.quantization}" if f.quantization else "", f"{f.n_layer} layers" if f.n_layer else ""]
    if f.n_ctx_train:
        bits.append(f"{f.n_ctx_train // 1024}K context" if f.n_ctx_train >= 1024 else f"{f.n_ctx_train} context")
    caps = []
    if f.is_embedding:
        caps.append("embeddings")
    else:
        if f.has_chat_template:
            caps.append("chat")
        if f.code_oriented:
            caps.append("code")
        if f.thinking_capable:
            caps.append("reasoning")
        if f.tools_capable:
            caps.append("tool calling")
    detail = ", ".join(b for b in bits if b)
    tail = f" — {', '.join(caps)}" if caps else ""
    return f"{head} ({detail}){tail}" if detail else f"{head}{tail}"


def _derive_tags(f: Optional[ModelFacts]) -> List[str]:
    if f is None:
        return []
    tags: List[str] = []
    if f.is_embedding:
        return ["embedding"]
    if f.has_chat_template:
        tags.append("chat")
    if f.code_oriented:
        tags.append("code")
    if f.thinking_capable:
        tags.append("reasoning")
    if f.tools_capable:
        tags.append("tools")
    if f.is_moe:
        tags.append("moe")
    if f.is_hybrid:
        tags.append("hybrid")
    if f.params_b >= 3.0:
        tags.append("primary")
    elif f.params_b > 0:
        tags.extend(["small", "fast"])
    return tags


def _derive_priority(f: Optional[ModelFacts], is_draft: bool) -> int:
    if is_draft:
        return 8
    if f is None:
        return 10
    if f.is_embedding:
        return 3
    if 0 < f.params_b <= 2.0:
        return 5
    return 10


def plan_profiles(
    profiles: Sequence[Mapping[str, Any]],
    machine: MachineFacts,
    settings: Optional[PlannerSettings] = None,
    bench: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Plan:
    """Convenience wrapper around :class:`Planner`."""
    return Planner(machine, settings, bench).plan(profiles)
