"""GGUF header inspection.

Reads only the metadata section of a ``.gguf`` file (never the tensors) and
returns a :class:`ModelFacts` record describing everything the planner needs
to size a model: layer count, KV-cache geometry, trained context window,
quantisation, rope settings, chat-template capabilities and a stable content
fingerprint.

Design notes
------------
* Keys are resolved through the model's own architecture prefix
  (``<arch>.block_count``) so auxiliary sections such as vision towers never
  shadow the language-model values.
* ``<arch>.attention.head_count_kv`` may be a scalar or a per-layer array.
  Hybrid architectures (attention + state-space layers) store ``0`` for layers
  without attention, so the number of layers that actually own a KV cache
  (:attr:`ModelFacts.kv_layers`) can be far smaller than ``n_layer``.
* Parsing is bounds-checked: a truncated or corrupt file raises
  :class:`GGUFError` instead of returning guessed values.
* Results are cached per ``(path, size, mtime)``.

Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import logging
import mmap
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__version__ = "1.0.0"

log = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"

# Largest single metadata string that is retained in memory (chat templates
# are the largest legitimate string; everything bigger is skipped).
_MAX_KEPT_STRING = 1 << 20
# Per-layer arrays longer than this are treated as malformed.
_MAX_LAYER_ARRAY = 4096
# Bytes of the file hashed for the content fingerprint.
_FINGERPRINT_BYTES = 2 << 20

_SCALAR_FMT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_T_STRING = 8
_T_ARRAY = 9

# ``general.file_type`` values (llama.cpp ``llama_ftype``) -> label.
_FILE_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}

# Approximate bits per weight, used only to estimate the parameter count when
# the file carries no ``general.size_label``.
_BITS_PER_WEIGHT: Dict[str, float] = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56,
    "Q5_K_M": 5.69, "Q5_K_S": 5.54, "Q5_1": 6.0, "Q5_0": 5.5,
    "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q4_1": 5.0, "Q4_0": 4.5,
    "Q3_K_L": 4.27, "Q3_K_M": 3.91, "Q3_K_S": 3.5, "Q2_K": 2.96,
    "Q2_K_S": 2.8, "IQ4_NL": 4.5, "IQ4_XS": 4.25, "IQ3_M": 3.66,
    "IQ3_S": 3.44, "IQ3_XS": 3.3, "IQ3_XXS": 3.06, "IQ2_M": 2.7,
    "IQ2_S": 2.5, "IQ2_XS": 2.31, "IQ2_XXS": 2.06, "IQ1_M": 1.75,
    "IQ1_S": 1.56, "I2_S": 2.0, "TQ1_0": 1.69, "TQ2_0": 2.06,
}

# Bytes per element of each KV-cache type: block bytes / block elements.
_KV_ELEM_BYTES: Dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0, "q5_0": 22.0 / 32.0, "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}

_QUANT_FILENAME_RE = re.compile(
    r"(?i)(?:^|[._-])(q[2-8]_(?:k_[sml]|k|0|1)|iq[1-4]_[a-z0-9]+|f16|f32|bf16|i2_s|tq[12]_0)"
    r"(?=[._-]|$)"
)
_SIZE_LABEL_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*([KMBT])")
_THINKING_RE = re.compile(
    r"<think>|enable_thinking|reasoning_effort|thinking_budget|<\|thinking\|>"
)
_TOOLS_RE = re.compile(r"\btools?\b|<tool_call>|tool_calls|<\|tool")
_CODE_NAME_RE = re.compile(r"(?i)(coder|code|codestral|starcoder|deepseek-coder)")


class GGUFError(ValueError):
    """The file is not a readable GGUF container."""


@dataclass(frozen=True)
class ModelFacts:
    """Everything the planner needs to know about one model file."""

    path: str
    file_name: str
    file_size_bytes: int
    fingerprint: str
    gguf_version: int
    architecture: str
    name: str = ""
    basename: str = ""
    finetune: str = ""
    size_label: str = ""
    file_type: int = -1
    quantization: str = ""
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_kv_heads: int = 0
    kv_layers: int = 0
    n_ff: int = 0
    head_dim_k: int = 0
    head_dim_v: int = 0
    n_ctx_train: int = 0
    vocab_size: int = 0
    rope_type: str = ""
    rope_factor: float = 0.0
    rope_freq_base: float = 0.0
    rope_orig_ctx: int = 0
    expert_count: int = 0
    expert_used: int = 0
    kv_lora_rank: int = 0
    rope_dim: int = 0
    has_ssm: bool = False
    pooling_type: Optional[int] = None
    tokenizer_model: str = ""
    chat_template: str = field(default="", repr=False, compare=False)
    tags: Tuple[str, ...] = ()
    params_b: float = 0.0

    # ── derived capabilities ────────────────────────────────────────────
    @property
    def has_chat_template(self) -> bool:
        return bool(self.chat_template)

    @property
    def thinking_capable(self) -> bool:
        return bool(_THINKING_RE.search(self.chat_template))

    @property
    def tools_capable(self) -> bool:
        return bool(_TOOLS_RE.search(self.chat_template))

    @property
    def code_oriented(self) -> bool:
        probe = " ".join((self.name, self.basename, self.finetune, self.file_name))
        return bool(_CODE_NAME_RE.search(probe)) or any(
            t.lower() in ("code", "coding") for t in self.tags
        )

    @property
    def is_embedding(self) -> bool:
        return self.pooling_type not in (None, 0) and not self.has_chat_template

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1

    @property
    def is_hybrid(self) -> bool:
        return self.has_ssm or (0 < self.kv_layers < self.n_layer)

    @property
    def file_size_mb(self) -> int:
        return int(self.file_size_bytes >> 20)

    @property
    def kv_elems_per_token_layer(self) -> int:
        """K+V elements stored per token in one attention layer."""
        if self.kv_lora_rank > 0:
            return self.kv_lora_rank + max(self.rope_dim, 0)
        return self.n_kv_heads * (self.head_dim_k + (self.head_dim_v or self.head_dim_k))

    def kv_bytes_per_token(self, kv_quant: str = "f16") -> float:
        """KV-cache bytes for one token across all layers that own a cache."""
        return float(self.kv_layers) * self.kv_elems_per_token_layer * kv_bytes_per_element(kv_quant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture, "name": self.name,
            "size_label": self.size_label, "quantization": self.quantization,
            "file_size_mb": self.file_size_mb, "fingerprint": self.fingerprint,
            "n_layer": self.n_layer, "kv_layers": self.kv_layers,
            "n_embd": self.n_embd, "n_head": self.n_head,
            "n_kv_heads": self.n_kv_heads, "head_dim_k": self.head_dim_k,
            "head_dim_v": self.head_dim_v, "n_ctx_train": self.n_ctx_train,
            "vocab_size": self.vocab_size, "rope_type": self.rope_type,
            "rope_factor": self.rope_factor, "rope_freq_base": self.rope_freq_base,
            "expert_count": self.expert_count, "hybrid": self.is_hybrid,
            "params_b": round(self.params_b, 2),
            "chat_template": self.has_chat_template,
            "thinking_capable": self.thinking_capable,
            "tools_capable": self.tools_capable,
        }


def kv_bytes_per_element(kv_quant: str) -> float:
    """Bytes per stored element for a KV-cache type (llama.cpp block layout)."""
    return _KV_ELEM_BYTES.get(str(kv_quant).lower().strip(), 2.0)


# ── low-level reader ────────────────────────────────────────────────────────
class _Cursor:
    __slots__ = ("mm", "pos", "end")

    def __init__(self, mm: "mmap.mmap", end: int) -> None:
        self.mm, self.pos, self.end = mm, 0, end

    def need(self, n: int) -> None:
        if n < 0 or self.pos + n > self.end:
            raise GGUFError("file is truncated inside the metadata section")

    def unpack(self, fmt: str, size: int) -> Any:
        self.need(size)
        v = struct.unpack_from(fmt, self.mm, self.pos)[0]
        self.pos += size
        return v

    def u32(self) -> int:
        return self.unpack("<I", 4)

    def u64(self) -> int:
        return self.unpack("<Q", 8)

    def string(self, keep: bool = True) -> Optional[str]:
        n = self.u64()
        self.need(n)
        start = self.pos
        self.pos += n
        if not keep or n > _MAX_KEPT_STRING:
            return None
        return bytes(self.mm[start:start + n]).decode("utf-8", "replace")

    def skip(self, n: int) -> None:
        self.need(n)
        self.pos += n


def _read_value(c: _Cursor, vtype: int, keep_array: bool) -> Any:
    if vtype == _T_STRING:
        return c.string()
    if vtype == _T_ARRAY:
        etype = c.u32()
        count = c.u64()
        if etype in _SCALAR_FMT:
            fmt, size = _SCALAR_FMT[etype]
            if keep_array and count <= _MAX_LAYER_ARRAY:
                c.need(count * size)
                vals = list(struct.unpack_from(f"<{count}{fmt[1]}", c.mm, c.pos))
                c.pos += count * size
                return vals
            c.skip(count * size)
            return _ArrayStub(count)
        if etype == _T_STRING:
            keep = keep_array and count <= 256
            vals = []
            for _ in range(count):
                s = c.string(keep=keep)
                if keep:
                    vals.append(s)
            return vals if keep else _ArrayStub(count)
        raise GGUFError(f"unsupported array element type {etype}")
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        return c.unpack(fmt, size)
    raise GGUFError(f"unknown metadata value type {vtype}")


class _ArrayStub:
    """Placeholder for an array whose contents were skipped; keeps its length."""

    __slots__ = ("count",)

    def __init__(self, count: int) -> None:
        self.count = int(count)


_KEEP_ARRAY_SUFFIXES = (
    ".attention.head_count_kv", ".attention.head_count",
    ".feed_forward_length", "general.tags",
)


def _parse_metadata(path: Path) -> Tuple[int, Dict[str, Any]]:
    try:
        size = path.stat().st_size
        if size < 24:
            raise GGUFError("file is too small to be a GGUF container")
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    except OSError as exc:
        raise GGUFError(f"cannot open {path}: {exc}") from exc
    try:
        if bytes(mm[:4]) != GGUF_MAGIC:
            raise GGUFError("missing GGUF magic")
        c = _Cursor(mm, size)
        c.pos = 4
        version = c.u32()
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        c.u64()                      # tensor count
        n_kv = c.u64()
        if n_kv > 1_000_000:
            raise GGUFError("implausible metadata count")
        kv: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = c.string()
            vtype = c.u32()
            keep = any(key.endswith(s) for s in _KEEP_ARRAY_SUFFIXES)
            kv[key] = _read_value(c, vtype, keep)
        return version, kv
    finally:
        mm.close()


def _fingerprint(path: Path, size: int) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(struct.pack("<Q", size))
    with open(path, "rb") as fh:
        h.update(fh.read(min(size, _FINGERPRINT_BYTES)))
    return f"{size}-{h.hexdigest()}"


def _as_int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _as_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _quant_label(file_type: int, arch: str, file_name: str) -> str:
    if file_type == 40 and arch.lower().startswith("bitnet"):
        return "I2_S"
    label = _FILE_TYPE_NAMES.get(file_type, "")
    if label:
        return label
    m = _QUANT_FILENAME_RE.search(file_name)
    return m.group(1).upper() if m else ""


def _parse_size_label(label: str) -> float:
    m = _SIZE_LABEL_RE.search(label or "")
    if not m:
        return 0.0
    mult = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}[m.group(2).upper()]
    return float(m.group(1)) * mult


def _build_facts(path: Path, version: int, kv: Dict[str, Any], size: int) -> ModelFacts:
    arch = _as_str(kv.get("general.architecture")) or "unknown"

    def a(suffix: str) -> Any:
        return kv.get(f"{arch}.{suffix}")

    n_layer = _as_int(a("block_count"))
    n_head_raw = a("attention.head_count")
    n_kv_raw = a("attention.head_count_kv")
    n_embd = _as_int(a("embedding_length"))

    if isinstance(n_head_raw, list):
        n_head = max((int(x) for x in n_head_raw), default=0)
    else:
        n_head = _as_int(n_head_raw)

    if isinstance(n_kv_raw, list) and n_kv_raw:
        n_kv_heads = max(int(x) for x in n_kv_raw)
        kv_layers = sum(1 for x in n_kv_raw if int(x) > 0)
    else:
        n_kv_heads = _as_int(n_kv_raw) or n_head
        kv_layers = n_layer

    interval = _as_int(a("full_attention_interval"))
    if interval > 1 and n_layer:
        kv_layers = max(1, -(-n_layer // interval))

    head_dim_k = _as_int(a("attention.key_length")) or (n_embd // n_head if n_head else 0)
    head_dim_v = _as_int(a("attention.value_length")) or head_dim_k

    ff_raw = a("feed_forward_length")
    n_ff = max((int(x) for x in ff_raw), default=0) if isinstance(ff_raw, list) else _as_int(ff_raw)

    file_type = _as_int(kv.get("general.file_type")) if "general.file_type" in kv else -1
    file_name = path.name
    quant = _quant_label(file_type, arch, file_name)

    tokens = kv.get("tokenizer.ggml.tokens")
    vocab = tokens.count if isinstance(tokens, _ArrayStub) else (len(tokens) if isinstance(tokens, list) else 0)
    vocab = vocab or _as_int(a("vocab_size"))

    tags_raw = kv.get("general.tags")
    tags = tuple(t for t in tags_raw if isinstance(t, str)) if isinstance(tags_raw, list) else ()

    size_label = _as_str(kv.get("general.size_label"))
    params_b = _parse_size_label(size_label)
    if not params_b and quant in _BITS_PER_WEIGHT:
        params_b = size * 8.0 / _BITS_PER_WEIGHT[quant] / 1e9

    pooling = a("pooling_type")
    rope_factor = a("rope.scaling.factor")

    return ModelFacts(
        path=str(path), file_name=file_name, file_size_bytes=size,
        fingerprint=_fingerprint(path, size), gguf_version=version,
        architecture=arch,
        name=_as_str(kv.get("general.name")),
        basename=_as_str(kv.get("general.basename")),
        finetune=_as_str(kv.get("general.finetune")),
        size_label=size_label, file_type=file_type, quantization=quant,
        n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_heads=n_kv_heads,
        kv_layers=kv_layers, n_ff=n_ff, head_dim_k=head_dim_k, head_dim_v=head_dim_v,
        n_ctx_train=_as_int(a("context_length")), vocab_size=vocab,
        rope_type=_as_str(a("rope.scaling.type")),
        rope_factor=float(rope_factor) if isinstance(rope_factor, (int, float)) else 0.0,
        rope_freq_base=float(a("rope.freq_base") or 0.0),
        rope_orig_ctx=_as_int(a("rope.scaling.original_context_length")),
        expert_count=_as_int(a("expert_count")), expert_used=_as_int(a("expert_used_count")),
        kv_lora_rank=_as_int(a("attention.kv_lora_rank")),
        rope_dim=_as_int(a("rope.dimension_count")),
        has_ssm=any(k.startswith(f"{arch}.ssm.") for k in kv),
        pooling_type=_as_int(pooling) if pooling is not None else None,
        tokenizer_model=_as_str(kv.get("tokenizer.ggml.model")),
        chat_template=_as_str(kv.get("tokenizer.chat_template")),
        tags=tags, params_b=params_b,
    )


_CACHE: Dict[Tuple[str, int, int], ModelFacts] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 64


def inspect_gguf(path: "str | Path") -> ModelFacts:
    """Return :class:`ModelFacts` for ``path`` (cached per size and mtime).

    Raises :class:`GGUFError` when the file is not a valid GGUF container or
    lacks the hyper-parameters a language model needs.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError as exc:
        raise GGUFError(f"cannot stat {p}: {exc}") from exc
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    version, kv = _parse_metadata(p)
    facts = _build_facts(p, version, kv, st.st_size)
    if facts.n_layer <= 0 and facts.pooling_type is None:
        raise GGUFError(f"{p.name}: no '{facts.architecture}.block_count' in metadata")
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = facts
    return facts


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def legacy_meta(path: "str | Path") -> Dict[str, Any]:
    """Dictionary view used by long-standing call sites (``read_gguf_meta``).

    Keeps the historical keys and defaults and adds ``kv_layers`` (layers that
    own a KV cache) and ``arch`` details so callers can size hybrid models.
    """
    f = inspect_gguf(path)
    n_head = f.n_head or 32
    return {
        "n_layer": f.n_layer or 32,
        "n_head": n_head,
        "n_embd": f.n_embd or 4096,
        "n_kv_heads": f.n_kv_heads or n_head,
        "kv_layers": f.kv_layers or f.n_layer or 32,
        "vocab_size": f.vocab_size or 32000,
        "arch": f.architecture or "llama",
        "context_length": min(f.n_ctx_train, 32768) if f.n_ctx_train else 4096,
        "n_ff": f.n_ff or 11008,
        "n_ctx_train": f.n_ctx_train or (min(f.n_ctx_train, 32768) if f.n_ctx_train else 4096),
        "head_dim": f.head_dim_k or ((f.n_embd or 4096) // n_head),
        "head_dim_v": f.head_dim_v or f.head_dim_k or ((f.n_embd or 4096) // n_head),
    }
