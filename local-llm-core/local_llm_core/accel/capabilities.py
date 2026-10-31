"""
capabilities.py — probe the installed llama-cpp-python without assuming any version.

Every version-sensitive behaviour is *measured* here, never read from a version
string. On 0.3.23 and 0.3.36 every field below has been confirmed by the
signature the operator posted; on future 0.4.x the probe degrades to the
conservative default rather than crashing.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaCppCaps:
    version: str
    has_llama_supports_gpu_offload: bool
    has_set_cache: bool
    cache_class_name: Optional[str]
    cache_is_longest_prefix: bool         # True for LlamaRAMCache / LlamaTrieCache
    init_kwargs: frozenset[str]           # set of accepted Llama(**kwargs) names
    supports_n_threads_batch: bool
    supports_type_k: bool
    supports_type_v: bool
    supports_flash_attn: bool
    supports_use_mlock: bool
    supports_draft_model: bool
    supports_last_n_tokens_size: bool
    logits_all_default: bool
    save_state_copies_scores: bool        # True when logits_all=True poisons saved states
    has_public_batch_generate: bool       # True only if a public multi-request API exists


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

    # 1. Kwarg whitelist from the live signature
    try:
        sig = inspect.signature(Llama.__init__)
        init_kwargs = frozenset(sig.parameters.keys())
    except (ValueError, TypeError):
        init_kwargs = frozenset()

    def _has(name: str) -> bool:
        return name in init_kwargs or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in (inspect.signature(Llama.__init__).parameters.values()
                      if init_kwargs else [])
        )

    # 2. GPU offload symbol
    try:
        from llama_cpp import llama_cpp as _lc
        has_gpu_offload = bool(getattr(_lc, "llama_supports_gpu_offload", None)) \
            and bool(_lc.llama_supports_gpu_offload())
    except Exception:
        has_gpu_offload = False

    # 3. Cache class
    cache_class_name = None
    cache_is_longest_prefix = False
    has_set_cache = hasattr(Llama, "set_cache")
    if has_set_cache:
        for name in ("LlamaRAMCache", "LlamaTrieCache"):
            try:
                import llama_cpp.llama_cache as _cache_mod
                if hasattr(_cache_mod, name):
                    cache_class_name = name
                    # Both implement BaseLlamaCache.__getitem__ with
                    # _find_longest_prefix_key on 0.3.x
                    cache_is_longest_prefix = True
                    break
            except ImportError:
                continue

    # 4. save_state scores-copy detection (the D4 bug).
    #    On 0.3.x, save_state does `copy.deepcopy(self._scores)` if logits_all=True.
    #    We detect this by source inspection so the check survives patch bumps.
    save_state_copies_scores = False
    try:
        src = inspect.getsource(Llama.save_state)
        save_state_copies_scores = (
            "_scores" in src and ("copy" in src or "deepcopy" in src)
        )
    except (OSError, TypeError):
        # Source not available (Cython/pyc) — assume worst case
        save_state_copies_scores = True

    # 5. Public batch-generation API. llama-cpp-python has never exposed one;
    #    if a future version does, this becomes True and MicroBatchEngine activates.
    has_public_batch_generate = False
    try:
        gen_sig = inspect.signature(Llama.generate)
        params = set(gen_sig.parameters)
        # A real batch API would take a list of prompts or a LlamaBatch.
        has_public_batch_generate = (
            "batch" in params or "prompts" in params or "sequences" in params
        )
    except (ValueError, TypeError):
        pass

    # 6. logits_all default from the live signature
    logits_all_default = False
    try:
        p = sig.parameters.get("logits_all") if init_kwargs else None
        if p is not None and p.default is not inspect.Parameter.empty:
            logits_all_default = bool(p.default)
    except Exception:
        pass

    caps = LlamaCppCaps(
        version=version,
        has_llama_supports_gpu_offload=has_gpu_offload,
        has_set_cache=has_set_cache,
        cache_class_name=cache_class_name,
        cache_is_longest_prefix=cache_is_longest_prefix,
        init_kwargs=init_kwargs,
        supports_n_threads_batch=_has("n_threads_batch"),
        supports_type_k=_has("type_k"),
        supports_type_v=_has("type_v"),
        supports_flash_attn=_has("flash_attn"),
        supports_use_mlock=_has("use_mlock"),
        supports_draft_model=_has("draft_model"),
        supports_last_n_tokens_size=_has("last_n_tokens_size"),
        logits_all_default=logits_all_default,
        save_state_copies_scores=save_state_copies_scores,
        has_public_batch_generate=has_public_batch_generate,
    )
    log.info(
        "llama-cpp-python caps: version=%s gpu_offload=%s cache=%s longest_prefix=%s "
        "scores_bloat=%s batch_api=%s",
        caps.version, caps.has_llama_supports_gpu_offload,
        caps.cache_class_name, caps.cache_is_longest_prefix,
        caps.save_state_copies_scores, caps.has_public_batch_generate,
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
    )
