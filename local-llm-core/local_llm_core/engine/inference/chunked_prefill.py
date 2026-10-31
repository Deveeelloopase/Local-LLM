"""chunked_prefill.py — interruptible, observable prompt evaluation with
content-addressed prefix reuse.

Design notes:
  * Prefix reuse consults `LlamaRAMCache` / `LlamaTrieCache` (both are
    longest-prefix matchers in llama-cpp-python >= 0.3.x), falling back
    to the live `llm._input_ids` match when the cache is empty or the
    installed version has no cache attached.
  * Every decision point is logged so operators can distinguish
    "cache empty" (save_state failing) from "cache populated but no
    shared prefix" (clients sending token-divergent prompts). That
    distinction was previously invisible and cost hours of debugging.

Verified against llama-cpp-python v0.3.23 `llama.py`:
  * `Llama.eval()` honours `n_tokens` and truncates KV beyond it.
  * `Llama.generate()` prefix-matches `self._input_ids` against the
    prompt tokens, so after we evaluate tokens[:-1] here, it evaluates
    exactly 1 more prompt token before the first generated token.
  * `BaseLlamaCache.__getitem__` calls `_find_longest_prefix_key`,
    which returns the longest shared token prefix — it is NOT an
    exact-match dict lookup.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

log = logging.getLogger(__name__)


def tokenize_like_llama_cpp(llm, prompt: str) -> List[int]:
    """Mirror llama-cpp-python 0.3.x tokenisation for a plain-str prompt."""
    toks = list(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
    try:
        add_bos = bool(llm._model.add_bos_token())
    except Exception:
        add_bos = False
    if add_bos:
        bos = llm.token_bos()
        if bos is not None and bos != -1:
            toks = [bos] + toks
    return toks

def _shared_prefix_len(state, toks: List[int]) -> int:
    """Length of the true shared token prefix between a cached state and `toks`.

    `cache[key]` returns the state of the stored key with the longest shared
    prefix, but `state.n_tokens` is the *stored sequence's* length, which can
    exceed the shared prefix. Trusting `state.n_tokens` after `load_state()`
    puts tokens from a different prompt into positions >= shared_prefix in the
    KV cache. This helper computes the real shared length so the caller can
    truncate before generation continues.
    """
    try:
        stored = getattr(state, "input_ids", None)
        if stored is None:
            # Older LlamaState shape; clamp to prompt length so we cannot
            # load more than was asked.
            n = int(getattr(state, "n_tokens", 0) or 0)
            return min(n, len(toks))
        n = int(getattr(state, "n_tokens", len(stored)) or len(stored))
        stored_slice = list(stored[:n])
        m = 0
        limit = min(len(stored_slice), len(toks))
        for i in range(limit):
            if stored_slice[i] != toks[i]:
                break
            m = i + 1
        return m
    except Exception as exc:
        log.debug("_shared_prefix_len: %s", exc)
        return 0

def chunked_prefill(
    llm,
    prompt: str,
    cancel: threading.Event,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    *,
    chunk: int = 128,
    min_tokens: int = 256,
    prefix_block: int = 256,
) -> dict:
    """Evaluate the prompt in small chunks BEFORE calling llm(...).

    Returns ``{"total", "reused", "evaluated", "cancelled", "cache_hit",
    "cache_entries", "cache_source"}``. The two extra fields exist purely
    for observability and can be consumed by the engine's existing
    `Prefill ...` log line without breaking the return contract.
    """
    toks = tokenize_like_llama_cpp(llm, prompt)
    total = len(toks)

    # ── Locate the cache once ────────────────────────────────────────────
    cache = getattr(llm, "_llama_cache", None) or getattr(llm, "cache", None)
    cache_present = cache is not None and hasattr(cache, "__getitem__")
    cache_entries = 0
    if cache_present:
        try:
            cache_entries = len(getattr(cache, "cache_state", {}) or {})
        except Exception:
            cache_entries = -1  # unknown; do not let a stat bug break prefill

    log.debug(
        "chunked_prefill entry: total=%d min_tokens=%d chunk=%d "
        "cache_present=%s cache_entries=%d",
        total, int(min_tokens), int(chunk), cache_present, cache_entries,
    )

    if total < int(min_tokens):
        log.debug(
            "chunked_prefill skip: prompt %d tok < min_tokens %d — "
            "leaving evaluation to the caller's generate() path",
            total, int(min_tokens),
        )
        return {
            "total": total, "reused": 0, "evaluated": 0,
            "cancelled": False, "cache_hit": False,
            "cache_entries": cache_entries, "cache_source": "skip",
        }

    # ── Try the content-addressed cache first ────────────────────────────
    cache_hit = False
    cache_source = "miss"
    reused = 0
    if cache_present and hasattr(cache, "__getitem__"):
        try:
            state = cache[tuple(toks)]
            if state is not None:
                # Compute the shared prefix BEFORE loading. state.n_tokens is
                # the stored sequence length, which can be longer than what we
                # share with this prompt. Loading it and trusting n_tokens
                # would leave prompt-A tokens at positions [shared, stored_n)
                # in the KV cache and generate from the wrong position.
                shared = _shared_prefix_len(state, toks)
                # Leave at least one prompt token for generate() to evaluate
                # (matches the miss path's `target = total - 1` contract).
                shared = min(shared, max(0, total - 1))
                if shared <= 0:
                    log.debug(
                        "chunked_prefill cache returned state but shared "
                        "prefix is 0 (stored=%d, prompt=%d) — treating "
                        "as miss",
                        int(getattr(state, "n_tokens", 0) or 0), total,
                    )
                else:
                    llm.load_state(state)
                    try:
                        llm.n_tokens = shared
                    except Exception:
                        pass
                    # Truncate the KV cache to `shared`. Without this, the
                    # attention kernel sees stale prompt-A keys at positions
                    # [shared, stored_n) during generation.
                    try:
                        llm._ctx.kv_cache_seq_rm(-1, shared, -1)
                    except (AttributeError, RuntimeError, TypeError) as exc:
                        log.debug(
                            "chunked_prefill: kv_cache_seq_rm unavailable "
                            "(%s) — relying on eval() n_tokens to truncate", exc,
                        )
                    reused = shared
                    cache_hit = True
                    cache_source = type(cache).__name__
                    log.info(
                        "chunked_prefill cache HIT via %s: reused=%d/%d "
                        "tokens (stored seq had %d)",
                        type(cache).__name__, reused, total,
                        int(getattr(state, "n_tokens", shared) or shared),
                    )
        except KeyError:
            log.warning(
                "chunked_prefill cache MISS: %d entries, no shared prefix "
                "with this %d-token prompt",
                cache_entries, total,
            )
        except Exception as exc:
            log.warning(
                "chunked_prefill cache lookup failed (%s): %s — "
                "falling back to live _input_ids match",
                type(cache).__name__, exc,
            )

    # ── Fall back to the live-KV prefix match ────────────────────────────
    if not cache_hit:
        try:
            cached_ids = llm._input_ids
            for a, b in zip(cached_ids, toks[:-1]):
                if a != b:
                    break
                reused += 1
        except Exception as exc:
            log.debug("chunked_prefill live prefix match failed: %s", exc)
            reused = 0
        llm.n_tokens = reused
        if reused > 1:
            cache_source = "input_ids"
            log.debug(
                "chunked_prefill live prefix match: reused=%d tokens",
                reused,
            )

    target = total - 1                # leave the last token for generate()
    pos = reused
    evaluated = 0
    step = max(8, int(chunk))
    last_ckpt = reused

    # `_cache_usable` turns off the save path after the FIRST save_state
    # failure. Without this, every block boundary retries a call that
    # will always fail — measurable latency for zero benefit.
    _cache_usable = cache_present

    if on_progress:
        on_progress(pos, total, reused)

    while pos < target:
        if cancel.is_set():
            log.debug(
                "chunked_prefill cancelled at pos=%d/%d (evaluated=%d)",
                pos, target, evaluated,
            )
            return {
                "total": total, "reused": reused,
                "evaluated": evaluated, "cancelled": True,
                "cache_hit": cache_hit,
                "cache_entries": cache_entries, "cache_source": cache_source,
            }
        end = min(pos + step, target)
        llm.eval(toks[pos:end])
        evaluated += end - pos
        pos = end
        if on_progress:
            on_progress(pos, total, reused)
        # Capture state at block boundaries so longer shared prefixes
        # benefit from caching too, not only the final position.
        if _cache_usable and pos - last_ckpt >= prefix_block:
            try:
                _state = llm.save_state()
                if _state is not None and getattr(_state, "llama_state_size", 0) > 0:
                    cache[tuple(toks[:pos])] = _state
                    last_ckpt = pos
                    log.debug(
                        "chunked_prefill saved block state at pos=%d "
                        "(size=%d bytes)",
                        pos, getattr(_state, "llama_state_size", 0),
                    )
                else:
                    log.warning(
                        "chunked_prefill save_state returned empty at pos=%d "
                        "— disabling cache for the rest of this request",
                        pos,
                    )
                    _cache_usable = False
            except Exception as exc:
                log.warning(
                    "chunked_prefill save_state failed at pos=%d (%s) — "
                    "disabling cache for the rest of this request",
                    pos, exc,
                )
                _cache_usable = False


    log.debug(
        "chunked_prefill done: total=%d reused=%d evaluated=%d "
        "cache_hit=%s source=%s",
        total, reused, evaluated, cache_hit, cache_source,
    )
    return {
        "total": total, "reused": reused,
        "evaluated": evaluated, "cancelled": False,
        "cache_hit": cache_hit,
        "cache_entries": cache_entries, "cache_source": cache_source,
    }
