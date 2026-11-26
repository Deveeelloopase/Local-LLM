"""
capabilities.py — probe the installed llama-cpp-python without assuming any version.

Continuous batching (v1.0.0+):
    Six fields gate the batched engine. All six must be True for
    BatchedInferenceEngine to run. The critical correction vs. the first
    version of this probe: n_seq_max and kv_unified in llama-cpp-python
    0.3.x are NOT explicit parameters of ``LlamaContext.__init__`` (which
    takes ``model``, ``params``, ``verbose``). They are fields on the
    ``llama_context_params`` C struct, applied via ``Llama.__init__``'s
    ``**kwargs`` passthrough loop:

        for key, value in kwargs.items():
            if key in dir(_params):
                setattr(_params, key, value)

    So the *static* probe verifies the C struct exposes the fields —
    that is the honest test of whether ``Llama(n_seq_max=N)`` will take
    effect. ``verify_context_n_seq_max()`` confirms at runtime, because
    kwargs passthrough is a *silent* mechanism (no error if the field is
    missing, just a no-op).

"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaCppCaps:
    version: str
    has_llama_supports_gpu_offload: bool
    has_set_cache: bool
    cache_class_name: Optional[str]
    cache_is_longest_prefix: bool
    init_kwargs: frozenset[str]
    supports_n_threads_batch: bool
    supports_type_k: bool
    supports_type_v: bool
    supports_flash_attn: bool
    supports_use_mlock: bool
    supports_draft_model: bool
    supports_last_n_tokens_size: bool
    logits_all_default: bool
    save_state_copies_scores: bool
    has_public_batch_generate: bool

    # ── Continuous batching probes (v1.0.0+) ────────────────────────────
    supports_n_seq_max: bool = False         # C params struct exposes n_seq_max
    supports_kv_unified: bool = False        # C params struct exposes kv_unified
    supports_llama_batch: bool = False       # LlamaBatch class importable
    supports_add_sequence: bool = False      # LlamaBatch.add_sequence exists
    has_llama_ctx_decode: bool = False       # LlamaContext.decode exists
    has_llama_memory_seq_rm: bool = False    # KV seq eviction available

    batching_probe_detail: str = ""
    llama_batch_location: str = ""


@lru_cache(maxsize=1)
def probe_llama_cpp() -> LlamaCppCaps:
    """Probe the installed llama-cpp-python. Cached — call once at startup."""
    try:
        import llama_cpp
        from llama_cpp import Llama
    except Exception as exc:
        log.error("llama-cpp-python not importable: %s", exc)
        return _conservative_caps("unknown", "unavailable")

    version = getattr(llama_cpp, "__version__", "unknown")

    # ── 1. Llama.__init__ kwargs ────────────────────────────────────────
    try:
        sig = inspect.signature(Llama.__init__)
        init_kwargs = frozenset(sig.parameters.keys())
    except (ValueError, TypeError):
        sig = None
        init_kwargs = frozenset()

    def _has_explicit(name: str) -> bool:
        """True only if name is an explicit named parameter.

        Deliberately does NOT treat **kwargs as accepting a name. Doing so
        produced a false positive in the first version of this probe:
        Llama.__init__ has **kwargs, so `n_seq_max` looked accepted when
        it was silently swallowed. Explicit-only is the honest check.
        """
        return name in init_kwargs

    # ── 2. GPU offload symbol ──────────────────────────────────────────
    try:
        from llama_cpp import llama_cpp as _lc
        has_gpu_offload = bool(getattr(_lc, "llama_supports_gpu_offload", None)) \
            and bool(_lc.llama_supports_gpu_offload())
    except Exception:
        has_gpu_offload = False

    # ── 3. Cache class ─────────────────────────────────────────────────
    cache_class_name = None
    cache_is_longest_prefix = False
    has_set_cache = hasattr(Llama, "set_cache")
    if has_set_cache:
        for name in ("LlamaRAMCache", "LlamaTrieCache"):
            try:
                import llama_cpp.llama_cache as _cache_mod
                if hasattr(_cache_mod, name):
                    cache_class_name = name
                    cache_is_longest_prefix = True
                    break
            except ImportError:
                continue

    # ── 4. save_state scores-copy detection ────────────────────────────
    save_state_copies_scores = False
    try:
        src = inspect.getsource(Llama.save_state)
        save_state_copies_scores = (
            "_scores" in src and ("copy" in src or "deepcopy" in src)
        )
    except (OSError, TypeError):
        save_state_copies_scores = True

    # ── 5. Public batch-generation API ─────────────────────────────────
    has_public_batch_generate = False
    try:
        gen_sig = inspect.signature(Llama.generate)
        params = set(gen_sig.parameters)
        has_public_batch_generate = (
            "batch" in params or "prompts" in params or "sequences" in params
        )
    except (ValueError, TypeError):
        pass

    # ── 6. logits_all default ──────────────────────────────────────────
    logits_all_default = False
    try:
        p = sig.parameters.get("logits_all") if sig else None
        if p is not None and p.default is not inspect.Parameter.empty:
            logits_all_default = bool(p.default)
    except Exception:
        pass

    # ── 7. Continuous batching probes ──────────────────────────────────
    diag: list[str] = []

    # 7a. n_seq_max / kv_unified — the actual mechanism on 0.3.x is
    #     **kwargs → llama_context_params C struct field.
    c_params_n_seq_max = False
    c_params_kv_unified = False
    c_params_default_n_seq_max = 0
    c_params_default_kv_unified: Optional[bool] = None
    try:
        from llama_cpp import llama_cpp as _lc3
        _p = _lc3.llama_context_default_params()
        try:
            c_params_default_n_seq_max = int(_p.n_seq_max)
            c_params_n_seq_max = True
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            c_params_default_kv_unified = bool(_p.kv_unified)
            c_params_kv_unified = True
        except (AttributeError, TypeError, ValueError):
            pass
    except Exception as exc:
        diag.append(f"C params probe failed: {exc}")

    ctx_explicit = False
    try:
        from llama_cpp._internals import LlamaContext as _Ctx
        _csig = inspect.signature(_Ctx.__init__)
        ctx_explicit = "n_seq_max" in _csig.parameters
    except Exception:
        pass

    supports_n_seq_max = c_params_n_seq_max or _has_explicit("n_seq_max") or ctx_explicit
    supports_kv_unified = c_params_kv_unified or _has_explicit("kv_unified") or ctx_explicit

    if not supports_n_seq_max:
        diag.append(
            "n_seq_max not on C params struct and not on Llama/LlamaContext signature "
            "— batched engine cannot run on this build"
        )
    if not supports_kv_unified:
        diag.append(
            "kv_unified not on C params struct and not on Llama/LlamaContext signature"
        )

    # 7b. LlamaBatch — search the actual module paths used by 0.3.x.
    #     0.3.36 defines it in llama_cpp._internals and re-exports via
    #     llama_cpp.llama. Also probe add_sequence while we have the class.
    supports_llama_batch = False
    supports_add_sequence = False
    llama_batch_location = ""
    for _mod_name, _attr in (
        ("llama_cpp._internals",   "LlamaBatch"),
        ("llama_cpp.llama",        "LlamaBatch"),
        ("llama_cpp.llama_batch",  "LlamaBatch"),
        ("llama_cpp",              "LlamaBatch"),
        ("llama_cpp.llama_cpp",    "LlamaBatch"),
    ):
        try:
            import importlib
            _m = importlib.import_module(_mod_name)
            _cls = getattr(_m, _attr, None)
            if _cls is not None:
                supports_llama_batch = True
                llama_batch_location = f"{_mod_name}.{_attr}"
                supports_add_sequence = hasattr(_cls, "add_sequence")
                break
        except Exception:
            continue
    if not supports_llama_batch:
        diag.append("LlamaBatch not importable from any known module")
    elif not supports_add_sequence:
        diag.append("LlamaBatch.add_sequence missing")

    # 7c. LlamaContext.decode — the batched forward-pass primitive.
    has_llama_ctx_decode = False
    for _mod_name, _attr in (
        ("llama_cpp._internals", "LlamaContext"),
        ("llama_cpp",            "LlamaContext"),
        ("llama_cpp.llama",      "LlamaContext"),
    ):
        try:
            import importlib
            _m = importlib.import_module(_mod_name)
            _cls = getattr(_m, _attr, None)
            if _cls is not None and hasattr(_cls, "decode"):
                has_llama_ctx_decode = True
                break
        except Exception:
            continue
    if not has_llama_ctx_decode:
        diag.append("LlamaContext.decode missing")

    # 7d. KV seq eviction
    has_llama_memory_seq_rm = False
    try:
        from llama_cpp import llama_cpp as _lc2
        if hasattr(_lc2, "llama_memory_seq_rm") or hasattr(_lc2, "llama_kv_cache_seq_rm"):
            has_llama_memory_seq_rm = True
    except Exception:
        pass
    if not has_llama_memory_seq_rm:
        try:
            from llama_cpp import LlamaContext
            if hasattr(LlamaContext, "kv_cache_seq_rm"):
                has_llama_memory_seq_rm = True
        except Exception:
            pass
    if not has_llama_memory_seq_rm:
        try:
            from llama_cpp._internals import LlamaContext as _Ctx2
            if hasattr(_Ctx2, "kv_cache_seq_rm"):
                has_llama_memory_seq_rm = True
        except Exception:
            pass
    if not has_llama_memory_seq_rm:
        diag.append(
            "No KV seq_rm binding (neither llama_memory_seq_rm nor "
            "LlamaContext.kv_cache_seq_rm available)"
        )

    caps = LlamaCppCaps(
        version=version,
        has_llama_supports_gpu_offload=has_gpu_offload,
        has_set_cache=has_set_cache,
        cache_class_name=cache_class_name,
        cache_is_longest_prefix=cache_is_longest_prefix,
        init_kwargs=init_kwargs,
        supports_n_threads_batch=_has_explicit("n_threads_batch"),
        supports_type_k=_has_explicit("type_k"),
        supports_type_v=_has_explicit("type_v"),
        supports_flash_attn=_has_explicit("flash_attn"),
        supports_use_mlock=_has_explicit("use_mlock"),
        supports_draft_model=_has_explicit("draft_model"),
        supports_last_n_tokens_size=_has_explicit("last_n_tokens_size"),
        logits_all_default=logits_all_default,
        save_state_copies_scores=save_state_copies_scores,
        has_public_batch_generate=has_public_batch_generate,
        supports_n_seq_max=supports_n_seq_max,
        supports_kv_unified=supports_kv_unified,
        supports_llama_batch=supports_llama_batch,
        supports_add_sequence=supports_add_sequence,
        has_llama_ctx_decode=has_llama_ctx_decode,
        has_llama_memory_seq_rm=has_llama_memory_seq_rm,
        batching_probe_detail="; ".join(diag),
        llama_batch_location=llama_batch_location,
    )
    log.info(
        "llama-cpp-python caps: version=%s gpu_offload=%s cache=%s longest_prefix=%s "
        "scores_bloat=%s batch_api=%s | continuous_batching: n_seq_max=%s "
        "kv_unified=%s llama_batch=%s(%s) add_sequence=%s ctx_decode=%s seq_rm=%s "
        "params_default(n_seq_max=%d kv_unified=%s)%s",
        caps.version, caps.has_llama_supports_gpu_offload,
        caps.cache_class_name, caps.cache_is_longest_prefix,
        caps.save_state_copies_scores, caps.has_public_batch_generate,
        caps.supports_n_seq_max, caps.supports_kv_unified,
        caps.supports_llama_batch, caps.llama_batch_location or "?",
        caps.supports_add_sequence, caps.has_llama_ctx_decode,
        caps.has_llama_memory_seq_rm,
        c_params_default_n_seq_max,
        c_params_default_kv_unified,
        f" [diag: {caps.batching_probe_detail}]" if caps.batching_probe_detail else "",
    )
    return caps


def _conservative_caps(version: str, reason: str) -> LlamaCppCaps:
    log.warning("Using conservative llama-cpp-python caps (%s): %s", reason, version)
    return LlamaCppCaps(
        version=version,
        has_llama_supports_gpu_offload=False,
        has_set_cache=False,
        cache_class_name=None,
        cache_is_longest_prefix=False,
        init_kwargs=frozenset(),
        supports_n_threads_batch=False,
        supports_type_k=False,
        supports_type_v=False,
        supports_flash_attn=False,
        supports_use_mlock=False,
        supports_draft_model=False,
        supports_last_n_tokens_size=False,
        logits_all_default=False,
        save_state_copies_scores=True,
        has_public_batch_generate=False,
        supports_n_seq_max=False,
        supports_kv_unified=False,
        supports_llama_batch=False,
        supports_add_sequence=False,
        has_llama_ctx_decode=False,
        has_llama_memory_seq_rm=False,
        batching_probe_detail=reason,
        llama_batch_location="",
    )


def continuous_batching_supported(caps: Optional[LlamaCppCaps] = None) -> bool:
    """Return True iff the installed llama-cpp-python can host BatchedInferenceEngine.

    All six continuous-batching flags must be True. Verified statically —
    the runtime path is confirmed by ``verify_batching_runtime()`` on the
    actual loaded ``Llama`` instance.
    """
    if caps is None:
        caps = probe_llama_cpp()
    return bool(
        caps.supports_n_seq_max
        and caps.supports_kv_unified
        and caps.supports_llama_batch
        and caps.supports_add_sequence
        and caps.has_llama_ctx_decode
        and caps.has_llama_memory_seq_rm
    )


def verify_context_n_seq_max(
    llm: Any, expected_n_seq_max: int = 0,
) -> Tuple[bool, str]:
    """Confirm the loaded context actually has the requested n_seq_max.

    Because Llama(**kwargs) is a *silent* mechanism (unknown kwargs are
    swallowed — no error if the field doesn't exist), we must check at
    runtime. Reads from:
      1. ``llm.context_params.n_seq_max`` (the C struct — always present)
      2. ``llm._desired_n_seq_max`` (set by BatchedInferenceEngine after
         context rebuild)

    Note: LlamaContext on 0.3.36 does NOT expose ``n_seq_max`` as a
    Python attribute — the authoritative value is on the C struct.

    If ``expected_n_seq_max=0`` only checks the value is retrievable.

    Returns ``(ok, detail)``. ``ok`` is False on any mismatch or read failure.
    """
    ctx = getattr(llm, "_ctx", None) or getattr(llm, "ctx", None)
    if ctx is None:
        return False, "no context on llm (checked ._ctx and .ctx)"

    actual = None
    params = getattr(llm, "context_params", None)
    if params is not None:
        try:
            actual = int(params.n_seq_max)
        except (AttributeError, TypeError, ValueError):
            pass
    if actual is None:
        try:
            actual = int(getattr(llm, "_desired_n_seq_max"))
        except (AttributeError, TypeError, ValueError):
            pass

    if actual is None:
        return False, (
            "cannot read n_seq_max from context "
            "(no attribute on _ctx and no context_params.n_seq_max)"
        )

    if expected_n_seq_max > 0 and actual != expected_n_seq_max:
        return False, (
            f"n_seq_max mismatch: requested {expected_n_seq_max}, "
            f"context has {actual} — Llama(**kwargs) did not apply it"
        )

    return True, f"n_seq_max={actual}"


def verify_batching_runtime(llm: Any) -> Tuple[bool, str]:
    """Runtime verification against a loaded Llama object.

    Checks the low-level APIs the batched engine actually calls:
      * llm._model (for n_vocab, tokenization)
      * llm._ctx.decode(batch) — decode path
      * llm._ctx.kv_cache_seq_rm (or .clear) — KV eviction
      * LlamaBatch importable, has add_sequence, constructable
      * llm._model.n_vocab present for sampling

    Returns ``(ok, detail)``. Does NOT run a forward pass — just verifies
    the surface exists.
    """
    problems: list[str] = []

    model = getattr(llm, "_model", None)
    ctx = getattr(llm, "_ctx", None)
    if model is None:
        problems.append("llm._model missing")
    if ctx is None:
        problems.append("llm._ctx missing")
    if problems:
        return False, "; ".join(problems)

    if not (hasattr(ctx, "eval") or hasattr(ctx, "decode")):
        problems.append("llm._ctx has neither eval() nor decode()")

    if not (hasattr(ctx, "kv_cache_seq_rm") or hasattr(ctx, "clear")):
        problems.append("llm._ctx has no kv_cache_seq_rm / clear")

    batch_cls = None
    for mod_name, attr in (
        ("llama_cpp._internals", "LlamaBatch"),
        ("llama_cpp.llama",      "LlamaBatch"),
        ("llama_cpp",            "LlamaBatch"),
    ):
        try:
            import importlib
            _m = importlib.import_module(mod_name)
            _cls = getattr(_m, attr, None)
            if _cls is not None:
                batch_cls = _cls
                break
        except Exception:
            continue
    if batch_cls is None:
        problems.append("LlamaBatch not importable")
    else:
        if not hasattr(batch_cls, "add_sequence"):
            problems.append("LlamaBatch.add_sequence missing")
        try:
            batch_cls(n_tokens=4, embd=0, n_seq_max=2)
        except TypeError:
            try:
                batch_cls(4, 0, 2)
            except Exception as exc:
                problems.append(f"LlamaBatch construction failed: {exc}")
        except Exception as exc:
            problems.append(f"LlamaBatch construction failed: {exc}")

    if not hasattr(model, "n_vocab"):
        problems.append("llm._model has no n_vocab (sampling will fail)")

    if problems:
        return False, "; ".join(problems)
    return True, "ok"
