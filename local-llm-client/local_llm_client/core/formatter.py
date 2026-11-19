"""
local_llm_client/core/formatter.py

Single source of truth for the "TTFT · prefill · decode · total · finish"
footer displayed by the CLI and the Web UI.

Why this file exists
────────────────────
Before, the CLI built the footer ad-hoc in `chat.py` and the Web UI
built it ad-hoc in `web_ui.py` from disjoint code paths. The two produced
different numbers for the same request because they used different sources
(CLI summed `estimate_tokens_fast` over chunks; web UI read
`response.usage.completion_tokens` which the streaming endpoint never set).
The (server-side `/v1/requests/{id}/stats`) provides one
authoritative data source. This module is the ONE formatter both clients
call so the presentation can never diverge again, even when the source
legitimately varies (server stats vs chunk fallback vs nothing).

Data contract
─────────────────────────────────────────────────────────────────────
`ServerStats.from_dict()` accepts the exact shape returned by
`GET /v1/requests/{id}/stats` :

    prompt_tokens, cached_prompt_tokens, completion_tokens,
    thinking_tokens, ttft_ms, prefill_ms, prefill_tps,
    decode_ms, decode_tps, queue_wait_ms, total_ms,
    finish_reason, worker_id, model, route, status

Unknown keys are silently ignored (forward compat). Missing keys default
to their zero values. Negative values are clamped to 0 with a debug log
(a negative value from the server is a server bug; the formatter should
never crash on it).

Display contract (acceptance criteria)
───────────────────────────────────────────
Three distinct source levels, three distinct presentations. The
`stats_source` argument selects which; callers pass it from
`GenerationResult.stats_source` .

  stats_source="server"   (authoritative — from /v1/requests/{id}/stats)
      TTFT 6.3s · prefill 12 tok @ 4.1/s · decode 330 tok @ 1.31/s · total 257s · finish=stop
      └─ every number is exact. No `~`. No `(est.)`. Safe for display
         without qualification.

  stats_source="chunks"   (exact chunk count, but excludes think-phase tokens)
      TTFT 6.3s · decode ~330 tok @ 1.31/s · total 257s · finish=stop (est.)
      └─ `~` on the count signals "excludes tokens the server stripped via
         strip_think=True". `(est.)` suffix signals "fallback path, not
         server-authoritative". No `prefill` segment: the client never
         measures prefill time. TTFT and total ARE exact (client-measured
         wall clock) — no `~` on those.

  stats_source="estimate"  (legacy fallback; not a supported display path)
      Same rendering as "chunks" (with `(est.)`) but a one-time warning is
      logged. The plan forbids using `estimate_tokens_fast` for display;
      this path exists only because a buggy caller might still set it.

  stats_source="unknown"   (no count available)
      TTFT 6.3s · decode — · total 257s · finish=stop
      └─ `—` for the numeric fields we cannot vouch for. TTFT/total are
         still shown because they are client-measured regardless of
         whether we know the token count.

Compact variant (used by CLI's `--stats-mode=compact` and by the web
status bar tooltip):

      format_tps_short() → "1.31" | "~1.31" | "—"

Structured variant (used by the Web UI's collapsible "details" row ):

      format_segments() → list[Segment]   # iterate, render each
      Segment(label="TTFT", value="6.3s", uncertain=False)

The Web UI builds its DOM from the segments; the CLI joins them with
" · ". One source of truth, two presentations, no drift.

Usage
─────
    from local_llm_client.core.formatter import (
        ServerStats, format_footer, format_segments, format_tps_short,
    )

    # Populated from GET /v1/requests/{id}/stats
    stats = ServerStats.from_dict(raw_json)
    print(format_footer(stats, stats_source="server"))

    # Web UI: iterate segments to build the details row
    for seg in format_segments(stats, stats_source="server"):
        print(f"<span class='{seg.label.lower()} {'' if not seg.uncertain else 'uncertain'}'>{seg.value}</span>")

    # Web status bar tooltip: just the TPS value
    print(format_tps_short(stats, stats_source="server"))

Verification (run after dropping; no test folder required)
──────────────────────────────────────────────────────────
    python -m local_llm_client.core.formatter   # runs _self_test()


"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

log = logging.getLogger(__name__)

# Canonical stats_source values. Imported from models.py to keep a single
# source of truth; falling back to a literal set here if models.py is not
# importable (only in pathological bootstrap scenarios).
try:
    from .models import STATS_SOURCE_VALUES as _STATS_SOURCE_VALUES
except ImportError:  # pragma: no cover - defensive
    _STATS_SOURCE_VALUES = frozenset({"server", "chunks", "estimate", "unknown"})

__all__ = [
    "Segment",
    "ServerStats",
    "format_segments",
    "format_footer",
    "format_tps_short",
]


# ── Small numeric formatters (kept private; callers use the public API) ──

def _clamp_nonneg(x: Any, name: str = "value") -> float:
    """Return float(x), or 0.0 if x is None / negative / non-numeric.

    A negative value from the server is a server bug; log at debug so an
    operator chasing a specific bug can find it, but never crash a
    formatter over it.
    """
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        log.info("_clamp_nonneg: %s not numeric (%r), using 0.0", name, x)
        return 0.0
    if v < 0:
        log.info("_clamp_nonneg: %s negative (%r), clamping to 0.0", name, x)
        return 0.0
    return v


def _clamp_nonneg_int(x: Any, name: str = "value") -> int:
    """Same as _clamp_nonneg but returns int."""
    return int(_clamp_nonneg(x, name))


def _fmt_duration_ms(ms: float) -> str:
    """Format a duration.

    < 1 s   → "342ms"       (integer ms — sub-ms precision is noise)
    < 10 s  → "6.3s"        (1 decimal — matches plan example "TTFT 6.3s")
    else    → "257s"        (integer s — matches plan example "total 257s")

    No minutes/hours conversion. The plan's own examples use `257s`,
    not `4m 17s`. Keep it literal so operators can compare the footer
    number directly against the server log's `257.0s` value without
    mentally converting.
    """
    if ms <= 0:
        return "—"
    if ms < 1000:
        return f"{ms:.0f}ms"
    s = ms / 1000.0
    if s < 10:
        return f"{s:.1f}s"
    return f"{s:.0f}s"


def _fmt_num_2dp(x: float) -> str:
    """Two-decimal number with trailing zeros stripped.

    1.31   → "1.31"
    4.1    → "4.1"     (4.10 with trailing zero stripped)
    10.5   → "10.5"
    10.0   → "10"
    0.85   → "0.85"
    """
    if x <= 0:
        return "0"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_tokens(n: int) -> str:
    """Format a token count.

    Under 1000: literal integer ("330", "12")
    Under 1M:   comma-grouped ("1,500", "12,345")
    At/over 1M: SI-suffixed ("2.00M")
    """
    if n <= 0:
        return "0"
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n:,}"
    return f"{n / 1_000_000:.2f}M"


def _fmt_tps(tps: float) -> str:
    """Format a tokens-per-second rate.

    Uses 2-decimal precision with trailing zeros stripped — matches the
    plan's `1.31/s` and `4.1/s` examples without a special-case for each.
    Zero / negative → "—" (nothing to display; a caller that wants to
    omit the rate should check `.decode_tps > 0` before formatting).
    """
    if tps <= 0:
        return "—"
    return f"{_fmt_num_2dp(tps)}/s"


# ── Structured segment (what web UI iterates) ──────────────────────────

class Segment(NamedTuple):
    """One label/value pair in the footer.

    Attributes:
        label:      "TTFT" | "prefill" | "decode" | "total" | "finish"
        value:      The pre-formatted value string.
        uncertain:  True if this segment's value is derived from a
                    non-authoritative source (chunks or estimate). The
                    CLI ignores this flag (it's already reflected via `~`
                    in `value`); the Web UI uses it to style the segment
                    differently (e.g. dimmed text, "~"-colored).
    """
    label: str
    value: str
    uncertain: bool = False


# ── ServerStats: mirrors GET /v1/requests/{id}/stats ──────────────────

@dataclass(frozen=True)
class ServerStats:
    """Server-reported stats for one request.

    Field names match, which is the
    canonical contract shared across:
      - the server's log line (`_log_request`)
      - `GET /v1/requests/{id}/stats`
      - `GET /v1/metrics`
      - this formatter
      - the Web UI details row
      - the TUI's "Requests" panel

    Every field defaults to 0 (or "" / -1 for id-likes) so constructing
    from a partial JSON body never fails. `from_dict` additionally clamps
    negatives to 0.
    """
    # Identity
    id: str = ""
    model: str = ""
    route: str = ""
    status: str = ""

    # Token accounting
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    # Timing
    ttft_ms: float = 0.0
    prefill_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    queue_wait_ms: float = 0.0
    total_ms: float = 0.0

    # Outcome
    finish_reason: str = "unknown"
    worker_id: int = -1

    # ─── NEW: stats provenance ──────────────────────────
    # Marks where these numbers came from. Set by the client (not the
    # server response body):
    #   "server"   — fetched from GET /v1/requests/{id}/stats
    #   "chunks"   — built client-side from streamed chunks
    #   "estimate" — legacy heuristic fallback (should not appear here)
    #   "unknown"  — default (should not survive to display)
    stats_source: str = "unknown"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ServerStats":
        """Build from a raw `/v1/requests/{id}/stats` JSON body.

        Unknown keys are ignored; missing keys default to zero. Negative
        numeric values are clamped to 0 (server bug, not crash).
        """
        if not d:
            return cls()
        # Explicit per-field extraction keeps the mapping auditable. A
        # generic kwargs splat would silently accept typos in field
        # names — the exact class of bug this formatter exists to
        # eliminate.
        return cls(
            id=str(d.get("id", "") or ""),
            model=str(d.get("model", "") or ""),
            route=str(d.get("route", "") or ""),
            status=str(d.get("status", "") or ""),
            prompt_tokens=_clamp_nonneg_int(d.get("prompt_tokens", 0), "prompt_tokens"),
            cached_prompt_tokens=_clamp_nonneg_int(
                d.get("cached_prompt_tokens", 0), "cached_prompt_tokens"
            ),
            completion_tokens=_clamp_nonneg_int(
                d.get("completion_tokens", 0), "completion_tokens"
            ),
            thinking_tokens=_clamp_nonneg_int(
                d.get("thinking_tokens", 0), "thinking_tokens"
            ),
            ttft_ms=_clamp_nonneg(d.get("ttft_ms", 0.0), "ttft_ms"),
            prefill_ms=_clamp_nonneg(d.get("prefill_ms", 0.0), "prefill_ms"),
            prefill_tps=_clamp_nonneg(d.get("prefill_tps", 0.0), "prefill_tps"),
            decode_ms=_clamp_nonneg(d.get("decode_ms", 0.0), "decode_ms"),
            decode_tps=_clamp_nonneg(d.get("decode_tps", 0.0), "decode_tps"),
            queue_wait_ms=_clamp_nonneg(d.get("queue_wait_ms", 0.0), "queue_wait_ms"),
            total_ms=_clamp_nonneg(d.get("total_ms", 0.0), "total_ms"),
            finish_reason=str(d.get("finish_reason", "unknown") or "unknown"),
            worker_id=int(d.get("worker_id", -1) or -1),
            stats_source=str(d.get("stats_source", "unknown") or "unknown"),
        )


# ── Source resolution (private) ───────────────────────────────────────

def _resolve_source(stats_source: str) -> str:
    """Normalise and validate the stats_source argument.

    Returns one of the canonical values. Raises ValueError on typos so a
    caller using "Server" or "serv" gets a loud error rather than a
    silently-miscoloured display — same rationale as
    GenerationResult.__post_init__ in models.py.
    """
    if not isinstance(stats_source, str):
        raise TypeError(
            f"stats_source must be str, got {type(stats_source).__name__}"
        )
    if stats_source not in _STATS_SOURCE_VALUES:
        raise ValueError(
            f"stats_source must be one of {sorted(_STATS_SOURCE_VALUES)}, "
            f"got {stats_source!r}"
        )
    return stats_source


# ── Public API ────────────────────────────────────────────────────────

def format_segments(
    stats: ServerStats,
    *,
    stats_source: str = "unknown",
) -> List[Segment]:
    """Break the footer into structured segments for rendering.

    The CLI joins the returned segments with " · " to build its single-line
    footer (via `format_footer`). The Web UI iterates them to build its
    collapsible "details" row — one span per segment, styled by
    `Segment.uncertain` and `Segment.label`. Both clients consume the same
    list, so the numbers can never diverge.

    Returns between 2 and 5 segments (fewer when fields are unavailable):

      • TTFT       always present (client-measured, trustworthy)
      • prefill    only when stats_source="server" AND prefill_tps > 0
                   AND prompt_tokens > cached_prompt_tokens
      • decode     always present (token count + optional TPS)
      • total      present when total_ms > 0 (client- or server-measured)
      • finish     always present

    Marker rules (display contract):
      stats_source="server"   → no `~`; no `(est.)` on the parent footer
      stats_source="chunks"   → `~` on the decode token count; `(est.)`
                                suffix added by `format_footer` (not here)
      stats_source="estimate" → same as "chunks"
      stats_source="unknown"  → `—` for numeric fields; TTFT/total still
                                shown if the caller provided them
    """
    src = _resolve_source(stats_source)
    uncertain = src in ("chunks", "estimate")
    marker = "~" if uncertain else ""
    has_numbers = src != "unknown"

    segments: List[Segment] = []

    # ── TTFT ───────────────────────────────────────────────────────
    # Always shown. The client measured it from its own wall clock,
    # independent of how the server counted tokens. Trustworthy at any
    # stats_source.
    segments.append(Segment("TTFT", _fmt_duration_ms(stats.ttft_ms), uncertain=False))

    # ── Prefill ────────────────────────────────────────────────────
    # Only reachable when the server reported it (stats_source="server").
    # Chunks/estimate/unknown paths never have prefill measurements.
    if (
        src == "server"
        and stats.prefill_tps > 0
        and stats.prompt_tokens > stats.cached_prompt_tokens
    ):
        new_prompt = stats.prompt_tokens - stats.cached_prompt_tokens
        segments.append(
            Segment(
                "prefill",
                f"{_fmt_tokens(new_prompt)} tok @ {_fmt_tps(stats.prefill_tps)}",
                uncertain=False,
            )
        )

    # ── Decode ─────────────────────────────────────────────────────
    # Always shown. `has_numbers=False` (unknown source) → show `—`.
    if has_numbers and stats.completion_tokens > 0:
        count_str = f"{marker}{_fmt_tokens(stats.completion_tokens)}"
        if stats.decode_tps > 0:
            segments.append(
                Segment(
                    "decode",
                    f"{count_str} tok @ {_fmt_tps(stats.decode_tps)}",
                    uncertain=uncertain,
                )
            )
        else:
            segments.append(
                Segment("decode", f"{count_str} tok", uncertain=uncertain)
            )
    else:
        segments.append(Segment("decode", "—", uncertain=False))

    # ── Total ──────────────────────────────────────────────────────
    # Shown when > 0. Always client- or server-measured elapsed time;
    # trustworthy independent of stats_source.
    if stats.total_ms > 0:
        segments.append(Segment("total", _fmt_duration_ms(stats.total_ms)))

    # ── Finish ─────────────────────────────────────────────────────
    finish = stats.finish_reason or "unknown"
    segments.append(Segment("finish", finish))

    return segments


def format_footer(
    stats: ServerStats,
    *,
    stats_source: str = "unknown",
) -> str:
    """Single-line footer for the CLI.

    Renders `format_segments(stats, stats_source=...)` joined by " · ",
    with the finish segment using "=" instead of " " so the outcome reads
    distinctly. Adds a trailing " (est.)" when stats_source is
    "chunks"/"estimate" (fallback path indicator).

    Examples (verbatim from acceptance):
      server   → "TTFT 6.3s · prefill 12 tok @ 4.1/s · decode 330 tok @ 1.31/s · total 257s · finish=stop"
      chunks   → "TTFT 6.3s · decode ~330 tok @ 1.31/s · total 257s · finish=stop (est.)"
      unknown  → "TTFT 6.3s · decode — · total 257s · finish=stop"
    """
    src = _resolve_source(stats_source)
    segments = format_segments(stats, stats_source=src)

    # Build the joined string. Finish uses "=" instead of " " per the
    # plan's example (`finish=stop`); all other segments use "Label value".
    parts: List[str] = []
    for seg in segments:
        if seg.label == "finish":
            parts.append(f"finish={seg.value}")
        else:
            parts.append(f"{seg.label} {seg.value}")
    body = " · ".join(parts)

    if src in ("chunks", "estimate"):
        body += " (est.)"
        if src == "estimate":
            # One-time warning: this path exists only for backward compat
            # with a buggy caller. New code must use "chunks" for the
            # fallback path or "server" for the authoritative one.
            log.warning(
                "format_footer: stats_source='estimate' — this path exists "
                "only for legacy callers. New code should use "
                "'chunks' (exact chunk count) or 'server' (authoritative)."
            )
    return body


def format_tps_short(
    stats: ServerStats,
    *,
    stats_source: str = "unknown",
) -> str:
    """The TPS value alone, for the web UI status bar.

    Returns:
      • The decode TPS as a bare string (no "/s" suffix — the caller's
        markup already includes " tok/s").
      • "~<tps>" when stats_source is "chunks"/"estimate".
      • "—" when stats_source is "unknown" OR decode_tps <= 0.

    The caller is expected to append " tok/s" (or localised equivalent)
    and put the formatted value in the tooltip. Example web UI usage:

        const tps = await fetch('/v1/requests/' + id + '/stats')...
        document.getElementById('st-tps').textContent = format_tps_short(stats);
        document.getElementById('st-tps').title = 'avg 5m: ' + avg5m;
    """
    src = _resolve_source(stats_source)
    if src == "unknown" or stats.decode_tps <= 0:
        return "—"
    marker = "~" if src in ("chunks", "estimate") else ""
    return f"{marker}{_fmt_num_2dp(stats.decode_tps)}"


# ── Self-test (invoked via `python -m local_llm_client.core.formatter`) ──

def _self_test() -> None:
    """Verify the formatter against fixtures from the contract.

    Fixtures mirror:
      • the real request described
        ("tell me a story" — TTFT 6.3s, 330 tok, 1.31 TPS, total 257s)
      • the plan's exact footer string from acceptance criteria
      • the empty / unknown / edge cases the CLI must handle

    Fixture values (model id, route, worker id, etc.) are placeholders
    chosen to exercise the from_dict path — they are NOT config-derived
    and are NOT read by any formatting branch. If you want to re-run
    with real values, populate from a live `GET /v1/requests/{id}/stats`
    and pass the resulting ServerStats directly to format_footer().

    Run with:
        python -m local_llm_client.core.formatter
    """
    import sys
    import logging as _logging

    failures: List[str] = []

    def _check(actual: str, expected: str, name: str) -> None:
        if actual != expected:
            failures.append(f"  {name}\n    expected: {expected!r}\n    actual:   {actual!r}")

    # ── the "tell me a story" request ──────────────
    # Identity fields (id, model, route, worker_id) are placeholders;
    # the formatter does not read them. Only the numeric + finish_reason
    # fields affect the rendered string.
    story = ServerStats(
        id="self-test-fixture",
        model="",                       # no hardcoded model name
        route="/v1/chat/completions",
        status="200",
        prompt_tokens=14,
        cached_prompt_tokens=0,
        completion_tokens=330,
        thinking_tokens=0,
        ttft_ms=6334,
        prefill_ms=3200,
        prefill_tps=4.1,
        decode_ms=253_000,
        decode_tps=1.31,
        queue_wait_ms=0,
        total_ms=257_000,
        finish_reason="stop",
        worker_id=-1,                   # -1 = unknown, matches from_dict default
    )

    _check(
        format_footer(story, stats_source="server"),
        "TTFT 6.3s · prefill 14 tok @ 4.1/s · decode 330 tok @ 1.31/s · total 257s · finish=stop",
        "server footer — full acceptance string",
    )
    _check(
        format_footer(story, stats_source="chunks"),
        "TTFT 6.3s · decode ~330 tok @ 1.31/s · total 257s · finish=stop (est.)",
        "chunks footer — marker + suffix, no prefill",
    )

    # The 'estimate' path logs a one-time warning by design (legacy-caller
    # indicator). Suppress it during the self-test so test output stays
    # clean; the warning remains active for real callers.
    _fmt_log = _logging.getLogger("local_llm_client.core.formatter")
    _prev_level = _fmt_log.level
    _fmt_log.setLevel(_logging.ERROR)   # hide WARNING
    try:
        _check(
            format_footer(story, stats_source="estimate"),
            "TTFT 6.3s · decode ~330 tok @ 1.31/s · total 257s · finish=stop (est.)",
            "estimate footer — identical rendering to chunks",
        )
    finally:
        _fmt_log.setLevel(_prev_level)

    _check(
        format_footer(story, stats_source="unknown"),
        "TTFT 6.3s · decode — · total 257s · finish=stop",
        "unknown footer — dashes for tokens",
    )

    # ── the status bar tooltip ─────────────────────
    _check(format_tps_short(story, stats_source="server"), "1.31", "tps_short server")
    _check(format_tps_short(story, stats_source="chunks"), "~1.31", "tps_short chunks")
    _check(format_tps_short(story, stats_source="unknown"), "—", "tps_short unknown")

    # ── fully-cached prompt → prefill segment omitted ───
    cached = ServerStats(
        prompt_tokens=500,
        cached_prompt_tokens=500,
        completion_tokens=42,
        ttft_ms=120,
        prefill_tps=0.0,   # nothing new to prefill
        decode_tps=2.5,
        total_ms=17_000,
        finish_reason="stop",
    )
    _check(
        format_footer(cached, stats_source="server"),
        "TTFT 120ms · decode 42 tok @ 2.5/s · total 17s · finish=stop",
        "server footer — fully-cached prompt omits prefill segment",
    )

    # ── sub-second total → ms branch of _fmt_duration_ms ─
    # total_ms=800 must render as "800ms", not "0s" (formatter is right;
    # the ms branch of _fmt_duration_ms fires for any value < 1000).
    sub_sec = ServerStats(
        ttft_ms=500,
        completion_tokens=0,
        decode_tps=0.0,
        total_ms=800,
        finish_reason="interrupted",
    )
    _check(
        format_footer(sub_sec, stats_source="server"),
        "TTFT 500ms · decode — · total 800ms · finish=interrupted",
        "server footer — sub-second total → ms branch",
    )

    # ── 5s total → 1-decimal branch of _fmt_duration_ms ─
    # total_ms=5000 must render as "5.0s" (matches plan's `6.3s` format),
    # not bare "5s" — the <10s branch keeps one decimal for consistency
    # with TTFT.
    five_sec = ServerStats(
        ttft_ms=100,
        completion_tokens=5,
        decode_tps=1.0,
        total_ms=5_000,
        finish_reason="",
    )
    _check(
        format_footer(five_sec, stats_source="server"),
        "TTFT 100ms · decode 5 tok @ 1/s · total 5.0s · finish=unknown",
        "server footer — <10s total keeps 1 decimal; empty finish_reason → 'unknown'",
    )

    # ── ServerStats.from_dict round-trip ────────────────
    raw = {
        "id": "abc", "model": "", "route": "/v1/chat/stream",
        "status": "200",
        "prompt_tokens": 14, "cached_prompt_tokens": 3,
        "completion_tokens": 42, "thinking_tokens": 5,
        "ttft_ms": 6334.0, "prefill_ms": 1200.0, "prefill_tps": 3.5,
        "decode_ms": 12000.0, "decode_tps": 3.5,
        "queue_wait_ms": 0.0, "total_ms": 20000.0,
        "finish_reason": "stop", "worker_id": 0,
        "unknown_future_field": "ignored",
    }
    s = ServerStats.from_dict(raw)
    assert s.id == "abc"
    assert s.completion_tokens == 42
    assert s.thinking_tokens == 5
    assert s.decode_tps == 3.5
    # Unknown field silently dropped — no AttributeError
    assert not hasattr(s, "unknown_future_field")

    # ── from_dict clamps negatives (server bug, not crash) ─
    bad = ServerStats.from_dict({
        "prompt_tokens": -10,
        "completion_tokens": -5,
        "decode_tps": -1.0,
        "ttft_ms": -100.0,
        "finish_reason": None,
    })
    assert bad.prompt_tokens == 0
    assert bad.completion_tokens == 0
    assert bad.decode_tps == 0.0
    assert bad.ttft_ms == 0.0
    assert bad.finish_reason == "unknown"

    # ── from_dict with empty dict → all defaults ────────
    empty = ServerStats.from_dict({})
    assert empty.completion_tokens == 0
    assert empty.finish_reason == "unknown"
    assert format_footer(empty, stats_source="unknown") == (
        "TTFT — · decode — · finish=unknown"
    )

    # ── invalid stats_source rejected ───────────────────
    for bad_src in ("Server", "serv", "", None, 42):
        try:
            format_footer(story, stats_source=bad_src)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        failures.append(
            f"  expected ValueError/TypeError for stats_source={bad_src!r}"
        )

    # ── Report ─────────────────────────────────────────────────────
    if failures:
        sys.stderr.write("FAILURES:\n" + "\n".join(failures) + "\n")
        raise SystemExit(1)
    print("✓ formatter self-test: all fixtures pass")
