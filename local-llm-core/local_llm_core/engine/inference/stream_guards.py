"""stream_guards.py - pure, dependency-free helpers shared by engine and server.

Contents
  * resolve_max_tokens / UNLIMITED_TOKENS : one definition of "no limit"
  * kv_bytes_per_token / kv_budget_mb / plan_context_length : memory-aware n_ctx
  * ThinkStripper / strip_think_text : stateful <think> tag removal (orphans too)
  * RepetitionGuard : the only non-EOS stop for unlimited generation
  * slide_window : rolling context window so generation never hits n_ctx

version = 1.0.0
"""
from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Tuple
from collections import Counter

log = logging.getLogger(__name__)

UNLIMITED_TOKENS = 2 ** 31 - 1


def resolve_max_tokens(requested) -> int:
    """Resolve the caller's max_tokens to an honest integer or UNLIMITED.

    Rules (matching the docstring contract that has existed since v1):
      * Python `int` and integer-like (`operator.index` accepts
        numpy.int64 etc.) → honoured if > 0, else UNLIMITED.
      * `bool` → UNLIMITED (bool is an int subclass; True/False are never
        legitimate token counts).
      * `float`, `str`, `None`, anything else → UNLIMITED.

    The previous implementation used `int(requested)` and got tripped by
    `int(1.5) == 1` — silently truncating a float to one token. That
    broke resume (`resume_generation` uses the return value as
    `extra_tokens`) with no error surfaced.
    """
    if isinstance(requested, bool):
        return UNLIMITED_TOKENS
    try:
        import operator
        v = operator.index(requested)
    except (TypeError, ValueError, OverflowError):
        return UNLIMITED_TOKENS
    return v if v > 0 else UNLIMITED_TOKENS


# ---------------------------------------------------------------- context size
# Bytes per element for llama.cpp KV cache quant types.
# Sources: llama.cpp ggml/src/ggml-common.h block struct sizes.
#
# Small-block formats (32 elements per block):
#   Q4_0: 18 B/block → 0.5625 B/elem     Q4_1: 20 B/block → 0.625 B/elem
#   Q5_0: 22 B/block → 0.6875 B/elem     Q5_1: 24 B/block → 0.75 B/elem
#   Q8_0: 34 B/block → 1.0625 B/elem     Q8_1: 36 B/block → 1.125 B/elem
#   IQ4_NL: 18 B/block → 0.5625 B/elem
# K-quant formats (256 elements per block):
#   Q2_K: 84 B/block → 0.328125 B/elem   Q3_K: 110 B/block → 0.4296875 B/elem
#   Q4_K: 144 B/block → 0.5625 B/elem    Q5_K: 176 B/block → 0.6875 B/elem
#   Q6_K: 210 B/block → 0.8203125 B/elem Q8_K: 292 B/block → 1.140625 B/elem
_KV_BYTES = {
    "f32":    4.0,
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   34 / 32,
    "q8_1":   36 / 32,
    "q5_1":   24 / 32,
    "q5_0":   22 / 32,
    "q4_1":   20 / 32,
    "q4_0":   18 / 32,
    "iq4_nl": 18 / 32,
    "q2_k":   84 / 256,
    "q3_k":   110 / 256,
    "q4_k":   144 / 256,
    "q5_k":   176 / 256,
    "q6_k":   210 / 256,
    "q8_k":   292 / 256,
}
# Unknown quant fallback = f16 (honest minimum for planning; never 0).
# Logged once per unknown quant name, not per call.
_UNKNOWN_KV_QUANTS: set = set()


def _kv_bytes_per_elem(quant: str) -> float:
    """Resolve a KV quant name to bytes-per-element; warn on unknown."""
    q = (quant or "").lower().strip()
    v = _KV_BYTES.get(q)
    if v is not None:
        return v
    if q not in _UNKNOWN_KV_QUANTS:
        _UNKNOWN_KV_QUANTS.add(q)
        log.warning(
            "_kv_bytes_per_elem: unknown KV quant %r — falling back to "
            "f16 (2.0 B/elem). Known: %s. Context planner may under-size "
            "n_ctx if the actual quant is denser than f16.",
            q, sorted(_KV_BYTES.keys()),
        )
    return 2.0


def kv_bytes_per_token(n_layer: int, n_kv_heads: int, head_dim_k: int,
                       head_dim_v: Optional[int] = None,
                       kv_quant_k: str = "q8_0",
                       kv_quant_v: Optional[str] = None) -> float:
    """Real KV-cache bytes per token (matches llama.cpp block layout)."""
    head_dim_v = head_dim_v or head_dim_k
    bk = _kv_bytes_per_elem(kv_quant_k)
    bv = _kv_bytes_per_elem(kv_quant_v or kv_quant_k)
    return float(n_layer) * n_kv_heads * (head_dim_k * bk + head_dim_v * bv)


def kv_budget_mb(ram_available_mb: int, share: float = 0.35,
                 hard_cap_mb: int = 0) -> int:
    """Return the KV-cache MB budget for the given available RAM."""
    b = int(max(0, ram_available_mb) * max(0.0, min(share, 0.9)))
    return min(b, hard_cap_mb) if hard_cap_mb > 0 else b


def plan_context_length(*, n_ctx_train: int, kv_per_token_bytes: float,
                        kv_budget_mb: int, floor: int = 2048,
                        ceiling: int = 0, align: int = 256) -> int:
    """Largest n_ctx that fits the KV budget, never above the trained window.

    If the caller's `floor` exceeds what the KV memory budget can support,
    returns `floor` (the caller's policy decision wins) AND logs a
    WARNING — the KV cache may thrash or fail to allocate at this size.
    Callers are expected to lower `min_context_tokens` or free memory when
    this warning appears.
    """
    cap = int(n_ctx_train) if n_ctx_train and n_ctx_train > 0 else int(floor)
    if ceiling and ceiling > 0:
        cap = min(cap, int(ceiling))
    by_mem = int((kv_budget_mb * 1048576) // max(kv_per_token_bytes, 1.0))
    floor_capped = min(int(floor), cap)
    n = (min(cap, by_mem) // align) * align
    if floor_capped > by_mem:
        log.warning(
            "plan_context_length: floor=%d > memory-capable=%d "
            "(kv_budget=%.1f MB, kv_per_token=%.0f B). Returning floor; "
            "KV cache may thrash or fail to allocate. Consider lowering "
            "model_loader.min_context_tokens or freeing memory.",
            floor_capped, by_mem, kv_budget_mb, kv_per_token_bytes,
        )
    return max(floor_capped, n)


# ------------------------------------------------------------------ think tags
_OPEN = ("<think>", "<thinking>")
_CLOSE = ("</think>", "</thinking>")
_ALL = _OPEN + _CLOSE
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think(?:ing)?>")


def strip_think_text(text: str) -> str:
    """Remove think blocks AND orphan tags from stored/historic text."""
    if not text:
        return text
    out = _THINK_BLOCK_RE.sub("", text)
    out = _THINK_TAG_RE.sub("", out)
    return out.lstrip("\n") if out != text else out


def _earliest(buf: str, tags) -> Optional[Tuple[int, str]]:
    best = None
    for t in tags:
        i = buf.find(t)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, t)
    return best


def _partial_tail(buf: str, tags) -> int:
    """Length of the longest suffix of buf that is a proper prefix of a tag."""
    best = 0
    for t in tags:
        for n in range(min(len(buf), len(t) - 1), 0, -1):
            if buf.endswith(t[:n]):
                best = max(best, n)
                break
    return best


class ThinkStripper:
    """Stateful stream filter. feed() returns text that is safe to show.

    strip=True  : drops complete think blocks and orphan closers such as the
                  stray '</think>' small Qwen3 models emit after an empty
                  think prefix. Tags split across chunks are handled.
    strip=False : pass-through (user enabled thinking; clients render it).

        Buffer bounds: `_think` is STRICTLY capped at
    `max_think_chars` (64 KiB default) — its append is clamped to the
    remaining budget so it can never exceed `_max`, even when the caller
    feeds a single chunk larger than `_max`. `_buf` is naturally bounded
    by `chunk_size + max_tag_len` in both branches (every path leaves
    only a small suffix ≤ len(_CLOSE) - 1 ≈ 9 chars); `_cap_buf()` is
    defensive, guarding against future changes to `_partial_tail`.
    """

    def __init__(self, strip: bool = True, max_think_chars: int = 65536):
        self.strip = strip
        self._buf = ""
        self._think = ""
        self._in_think = False
        self._lstrip = False
        self._emitted = False
        self._max = max_think_chars

    def _emit(self, s: str) -> str:
        if not s:
            return ""
        if self._lstrip:
            s = s.lstrip()
            if s:
                self._lstrip = False
        if s.strip():
            self._emitted = True
        return s

    def _cap_buf(self) -> None:
        """Trim `_buf` if it exceeds `_max`. Keeps the newest bytes.

        Called after `_buf` grows inside an unclosed think block. Trimming
        from the FRONT (oldest) is safe — that content would eventually be
        inside the think block and never emitted even if the block closed.
        """
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def feed(self, chunk: str) -> str:
        if not self.strip or not chunk:
            return chunk or ""
        self._buf += chunk
        out: List[str] = []
        while True:
            if self._in_think:
                m = _earliest(self._buf, _CLOSE)
                if m is None:
                    keep = _partial_tail(self._buf, _CLOSE)
                    cut = len(self._buf) - keep
                    # strictly cap `_think` at `_max`. The
                    # previous `if len(self._think) < self._max: self._think
                    # += self._buf[:cut]` allowed a single append to push
                    # `_think` up to `_max + cut` — e.g. feeding a 200-char
                    # chunk with `max_think_chars=100` produced a 200-char
                    # `_think`. Now `cut` is clamped to the remaining budget
                    # so `_think` never exceeds `_max`. Bytes beyond the
                    # budget are dropped; same policy as `_cap_buf` — the
                    # oldest content is the least useful for the
                    # unclosed-block fallback.
                    remaining = self._max - len(self._think)
                    if remaining > 0:
                        self._think += self._buf[:min(cut, remaining)]
                    self._buf = self._buf[cut:]
                    self._cap_buf()
                    break
                i, tag = m
                self._buf = self._buf[i + len(tag):]
                self._think = ""
                self._in_think = False
                self._lstrip = not self._emitted
                continue
            m = _earliest(self._buf, _ALL)
            if m is None:
                keep = _partial_tail(self._buf, _ALL)
                cut = len(self._buf) - keep
                out.append(self._emit(self._buf[:cut]))
                self._buf = self._buf[cut:]
                self._cap_buf()
                break
            i, tag = m
            out.append(self._emit(self._buf[:i]))
            self._buf = self._buf[i + len(tag):]
            if tag in _OPEN:
                self._in_think = True
            else:                                   # orphan closer
                self._lstrip = not self._emitted
        return "".join(out)

    def flush(self) -> str:
        if not self.strip:
            return ""
        rest, self._buf = self._buf, ""
        if self._in_think:
            # Never turn an unclosed think block into an empty answer.
            held = self._think + rest
            self._think, self._in_think = "", False
            return self._emit(held) if not self._emitted else ""
        return self._emit(rest)


# ------------------------------------------------------------ repetition guard
class RepetitionGuard:
    """Detects degenerate loops so unlimited generation always terminates.

    Fires when the tail of the output is one block of 4..2000 chars repeated
    until it covers >= min_cover chars (and >= 3 repetitions).

    Used as the fallback when `inference.loop_guard_enabled: false` in
    config. `LoopGuardV2` is preferred for its additional detectors
    (line / heading / shingle); this class is the minimal safety net.
    """

    def __init__(self, min_cover: int = 800, window: int = 8000,
                 check_every: int = 48):
        self._tail = ""
        self._min_cover = min_cover
        self._window = window
        self._every = check_every
        self._since = 0

    def feed(self, text: str) -> bool:
        if not text:
            return False
        self._tail = (self._tail + text)[-self._window:]
        self._since += len(text)
        if self._since < self._every:
            return False
        self._since = 0
        return self._looping()

    def _looping(self) -> bool:
        t = self._tail
        n = len(t)
        for p in range(4, 2001):
            reps = max(3, -(-self._min_cover // p))
            need = p * reps
            if need > n:
                break
            block = t[n - p:]
            if t[n - need:] == block * reps:
                return True
        return False


# --------------------------------------------------------------- window slide
def slide_window(head: str, generated: str, *, n_ctx: int, seg_max: int,
                 safety: int, count_tokens: Callable[[str], int],
                 target_fill: float = 0.60, min_tail: int = 256) -> str:
    """Return head + tail(generated) so the next segment always fits n_ctx.

    head      : the original templated prompt (kept verbatim).
    generated : everything the model wrote so far (only its tail is kept).
    The tail starts on a paragraph/sentence boundary and is sized so the
    context is ~target_fill full afterwards, amortising the re-prefill.
    Returns head+generated unchanged if it already fits.

    When text IS dropped (start > 0), a short [INTERNAL_CONTINUATION] marker
    is prepended to the retained tail so the model knows it is mid-answer and
    does not restart its own numbering / headings / code fences. The marker
    is prompt-only — it is fed to the model and does not appear in the
    emitted completion (see generate_segmented in inference_engine.py).
    """
    full = head + generated
    if count_tokens(full) + seg_max + safety <= n_ctx:
        return full
    head_tok = count_tokens(head)
    tail_budget = max(min_tail, int(n_ctx * target_fill) - head_tok)
    lo, hi = 0, len(generated)            # smallest start with tail <= budget
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(generated[mid:]) <= tail_budget:
            hi = mid
        else:
            lo = mid + 1
    start = lo
    window = generated[start:start + 600]
    for sep in ("\n\n", "\n", ". ", "? ", "! ", " "):
        j = window.find(sep)
        if j >= 0:
            start += j + len(sep)
            break
    tail = generated[start:]
    if start > 0:
        marker = (
            "\n\n[INTERNAL_CONTINUATION — you are mid-answer; do NOT restart, "
            "do NOT repeat previous items; preserve numbering, code fences, "
            "tables]\n\n"
        )
        tail = marker + tail
    return head + tail


# -------------------------------------------------------- loop guard v2
_ROLE_TAG_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
_NORM_RE = re.compile(r"[^a-z0-9 ]+")


class LoopGuardV2:
    """Structured runaway-detection.

    Fires on the FIRST of:
      (a) exact periodic tail (legacy RepetitionGuard behaviour),
      (b) the same long line (>= ``line_min_chars``) seen >= ``line_repeats`` times,
      (c) the same markdown heading seen >= ``heading_repeats`` times,
      (d) 8-word-shingle duplication ratio >= ``shingle_ratio`` over
          ``shingle_window`` chars (catches paraphrased loops),
      (e) leaked chat-template control tokens (self-dialogue).

    Ignores table rows, rules and code fences so legitimate repetitive
    output is not cut. All thresholds are constructor args; the engine
    reads them from config, so nothing is hard-coded.

    `reason` after a `feed()` that returns True is one of:
        "control_token" | "periodic" | "heading" | "line" | "shingle"
    Callers (inference_engine._continuous_tokens, generate_segmented) log
    this value so an operator can see WHY generation ended.
    """
    def __init__(
        self, *,
        min_cover: int = 800,
        window: int = 8000,
        check_every: int = 48,
        line_min_chars: int = 40,
        line_repeats: int = 3,
        heading_min_chars: int = 12,
        heading_repeats: int = 3,
        shingle_n: int = 8,
        shingle_window: int = 6000,
        shingle_ratio: float = 0.45,
        shingle_min_words: int = 600,
    ) -> None:
        self._tail = ""
        self._min_cover = int(min_cover)
        self._window = int(window)
        self._every = int(check_every)
        self._since = 0
        self._line_min = int(line_min_chars)
        self._line_rep = int(line_repeats)
        self._head_min = int(heading_min_chars)
        self._head_rep = int(heading_repeats)
        self._sh_n = int(shingle_n)
        self._sh_win = int(shingle_window)
        self._sh_ratio = float(shingle_ratio)
        self._sh_min_words = int(shingle_min_words)
        self._lines: Counter = Counter()
        self._heads: Counter = Counter()
        self._pending = ""
        self.reason = ""

    def feed(self, text: str) -> bool:
        if not text:
            return False
        if _ROLE_TAG_RE.search(self._pending[-16:] + text):
            self.reason = "control_token"
            return True
        self._tail = (self._tail + text)[-self._window:]
        self._since += len(text)
        self._pending += text
        hit = False
        if "\n" in self._pending:
            *done, self._pending = self._pending.split("\n")
            for ln in done:
                if self._count_line(ln):
                    hit = True
        if hit:
            return True
        if self._since < self._every:
            return False
        self._since = 0
        if self._periodic():
            self.reason = "periodic"
            return True
        if self._shingles():
            self.reason = "shingle"
            return True
        return False

    def _count_line(self, ln: str) -> bool:
        raw = ln.strip()
        if not raw:
            return False
        if raw.startswith("#"):
            key = _NORM_RE.sub("", raw.lower()).strip()
            if len(key) >= self._head_min:
                self._heads[key] += 1
                if self._heads[key] >= self._head_rep:
                    self.reason = "heading"
                    return True
            return False
        if raw.startswith(("|", "---", "```")):
            return False
        key = _NORM_RE.sub("", raw.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if len(key) >= self._line_min:
            self._lines[key] += 1
            if self._lines[key] >= self._line_rep:
                self.reason = "line"
                return True
        return False

    def _periodic(self) -> bool:
        t = self._tail
        n = len(t)
        for p in range(4, 2001):
            reps = max(3, -(-self._min_cover // p))
            need = p * reps
            if need > n:
                break
            if t[n - need:] == t[n - p:] * reps:
                return True
        return False

    def _shingles(self) -> bool:
        words = self._tail[-self._sh_win:].split()
        if len(words) < self._sh_min_words:
            return False
        n = self._sh_n
        seen, dup, total = set(), 0, 0
        for i in range(len(words) - n + 1):
            sh = tuple(words[i:i + n])
            total += 1
            if sh in seen:
                dup += 1
            else:
                seen.add(sh)
        return total > 0 and (dup / total) >= self._sh_ratio


# ------------------------------------------ solicitation (self-answer) guard
_BUILTIN_SOLICIT_FAMILIES = (
    # Conservative, whole-line matches against the tail of the last
    # completed line.  Each family is tried independently; a hit in any
    # arms the guard.  Extendable via config (solicitation_families).
    r"(?i)^\s*(?:[-*>]\s*)?(?:would you like|do you want|would you prefer)\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?which (?:option|one|of these|approach|format)\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?shall i\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?should i (?:proceed|continue|go ahead)\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?let me know (?:if|which|what|how)\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?please let me know\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?(?:if you (?:have|need|want|would like))\b[^\n]{0,200}$",
    r"(?i)^\s*(?:[-*>]\s*)?(?:anything else|feel free to ask|any other questions)\b[^\n]{0,200}$",
)

# Signals that the model is answering its own solicitation.
_SELF_ANSWER_HINT_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:"
    r"option\s*(?:\d+|[A-Z])"
    r"|here(?:'s| is)"
    r"|#{1,6}\s+\S"
    r"|(?:\d{1,2})[\.\)]\s+\S"
    r"|first(?:ly)?[,:\s]"
    r")"
)


class SolicitationGuard:
    """Second-layer guard: ends generation when the model asks the *user* a
    closing question and then keeps writing by answering itself.

    Design (5 independent detectors, all thresholds configurable):

      1. Closing-question detector — matches the *tail* of the last
         completed line against a family of curated regexes.
      2. Self-answer detector — after arming, fires if the next
         ``grace_tokens`` tokens contain any of: ``Option N``, ``Option X``,
         ``Here is``/``Here's``, a markdown heading, or a numbered-list
         start.
      3. Menu-prompt detector — a solicitation line followed by two or
         more list items inside the grace window fires even without a
         self-answer hint.
      4. EOS-tolerant grace window — the guard waits ``grace_tokens``
         *without* the tail changing shape; natural EOS inside the window
         means the check never fires (no truncation of legitimate endings).
      5. Once-per-stream, min-arm-tokens — never fires twice, never fires
         within the first ``min_arm_tokens`` tokens.

    `families=[]` and `families=None` both use the built-in families.
    To disable the guard entirely, set `solicitation_guard_enabled: false`
    in config (checked by `inference_engine._continuous_tokens`).
    """
    def __init__(
        self, *,
        grace_tokens: int = 64,
        min_arm_tokens: int = 256,
        families: Optional[List[str]] = None,
    ) -> None:
        src = list(families) if families else list(_BUILTIN_SOLICIT_FAMILIES)
        self._families = [re.compile(p) for p in src]
        self._grace = int(grace_tokens)
        self._min_arm = int(min_arm_tokens)
        self._buf = ""            # sliding text window (per line for matching)
        self._pending = ""        # current incomplete line
        self._n = 0               # tokens seen by this guard
        self._armed_at: Optional[int] = None
        self._list_items_after_arm = 0
        self._fired = False
        self.reason = "solicitation"

    def feed(self, text: str) -> bool:
        if not text or self._fired:
            return False
        self._n += 1
        if self._n < self._min_arm:
            return False

        # Accumulate for line-boundary detection
        self._pending += text
        if "\n" in self._pending:
            *done, self._pending = self._pending.split("\n")
            for ln in done:
                self._observe_line(ln)

        # Not yet armed?  Only the line detector can arm.
        if self._armed_at is None:
            return False

        # Armed: check the self-answer / menu-prompt signals.
        elapsed = self._n - self._armed_at
        if elapsed > self._grace * 4:
            # Grace fully expired with no signal → disarm (model finished
            # answering the question naturally without answering itself).
            self._armed_at = None
            self._list_items_after_arm = 0
            return False

        tail = self._pending + text
        if _SELF_ANSWER_HINT_RE.search(tail):
            self._fired = True
            return True

        # Menu-prompt: two or more list items within the grace window.
        if re.search(r"(?m)^\s*(?:[-*+]|\d+[\.\)])\s+\S", tail):
            self._list_items_after_arm += 1
            if self._list_items_after_arm >= 2:
                self._fired = True
                return True

        return False

    def _observe_line(self, ln: str) -> None:
        # Only arm on a *complete* line; use a bounded tail so we don't
        # false-positive on long paragraphs that merely mention the phrase.
        s = ln.strip()
        if not s or len(s) > 400:
            return
        for pat in self._families:
            if pat.match(s):
                if self._armed_at is None:
                    self._armed_at = self._n
                    self._list_items_after_arm = 0
                return
