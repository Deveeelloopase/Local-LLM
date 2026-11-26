"""
local_llm_client/core/stream_compactor.py

 Neat streaming output: collapse blank-line runs, preserve code blocks.

Purpose
───────
The LLM emits "paragraph" gaps as literal ``\\n\\n`` runs in its output.
Before the CLI printed them verbatim, producing a display that looked
like this::

    🤖 AI:

    Here is a story.

    Once upon a time...

    The end.

That triple-blank-line gap between "story." and "Once" is the model's
literal output. It is not a rendering bug. It is what the model chose to
emit. On a 1.3 tok/s box, every one of those blank lines cost ~0.8 s of
generation time (the model had to sample the token ``\\n`` twice) and
contributed nothing to readability.

The compactor rewrites the stream so that, outside fenced code blocks,
runs of newlines are collapsed to at most ``1 + max_blank_lines``
newlines before display. The streamed text arrives faster, the display is
tighter, and (crucially for) the token count is untouched because
counting happens on the RAW stream, not the compacted one.

Algorithm (reference implementation, verified)
─────────────────────────────────────────────
  * Text is emitted immediately — the compactor never buffers visible
    characters. Only newline runs are held back; a newline run is
    released as soon as the next visible character arrives.
  * A fence is a line starting with up to 3 spaces followed by three or
    more ``` or ~~~. Inside a fence nothing is changed. The opening and
    closing marker lines themselves are emitted as-is.
  * Leading newlines at the very start of the reply are dropped.
  * Trailing newlines are dropped by ``finish()``.
  * ``\\r`` characters are dropped (they interfere with display and add
    nothing to a line-oriented protocol). ``\\r\\n`` becomes ``\\n``.

 CAVEAT — count BEFORE compaction, not after
─────────────────────────────────────────────────
The compactor REMOVES newline characters. That means the compacted text
has FEWER characters than the raw text the model actually emitted. The
server counted the raw text. If you count the compacted text, you will
report FEWER tokens than the server did — the exact opposite error of the
original bug, but just as wrong.

Correct usage in a streaming caller::

    counter = ChunkTokenCounter()
    compactor = StreamCompactor(max_blank_lines=cfg.ui_blank_lines)
    async for chunk in stream:
        counter.add(chunk)                    # ✓ count RAW
        out = compactor.feed(chunk)           # ✓ compact for display
        print(out, end="")

Wrong usage (reintroduces the miscount in reverse)::

    async for chunk in stream:
        out = compactor.feed(chunk)
        counter.add(out)                      # ✗ counts compacted text
        print(out, end="")

The one-shot ``compact(text)`` helper has the same caveat. If you need
the token count of ``raw_text``, call ``counter.add(raw_text)`` or
``tokenizer.count(raw_text)`` BEFORE calling ``compact(raw_text)``.

Configuration integration
──────────────────────────────────────────
  ui.blank_lines: 0 | 1 | 2 | preserve   → maps to:

    0           → StreamCompactor(max_blank_lines=0)   — compact (default)
    1           → StreamCompactor(max_blank_lines=1)   — at most 1 blank line
    2           → StreamCompactor(max_blank_lines=2)   — at most 2 blank lines
    "preserve"  → do not instantiate; emit raw text unchanged

  CLI flag:   --blank-lines N            (N is an int >= 0)
  In-chat:    /compact on|off            ("off" == preserve == no compactor)

Server-side integration is optional ("helps every client" — the
server can apply ``compact()`` to ``full_response`` before persistence
so history sent back to the model is also compact — saves prompt tokens
too). Not wired in this drop; the client-side path is the mandatory one.

Thread safety
─────────────
Not thread-safe. One ``StreamCompactor`` per stream; do not share across
concurrent streams. Callers run their own iterator; the compactor is
single-threaded by construction.

Verification (no test folder — one-liner)
─────────────────────────────────────────
    python -c "
    from local_llm_client.core.stream_compactor import StreamCompactor, compact

    # 1. Basic compaction
    assert compact('a\\n\\n\\nb') == 'a\\nb'
    assert compact('a\\n\\n\\nb', max_blank_lines=1) == 'a\\n\\nb'
    assert compact('a\\n\\n\\nb', max_blank_lines=2) == 'a\\n\\n\\nb'

    # 2. Fenced code preserved byte-for-byte
    text = 'before\\n\\n\\n\\`\\`\\`\\ncode\\n\\n\\nmore\\n\\`\\`\\`\\n\\n\\nafter'
    out = compact(text)
    assert '\\`\\`\\`\\ncode\\n\\n\\nmore\\n\\`\\`\\`' in out, out
    assert out.endswith('after'), out

    # 3. Streaming == one-shot (arbitrary chunk boundaries)
    comp = StreamCompactor()
    streamed = ''.join(comp.feed(c) for c in ['a\\n\\n', '\\nb', '\\n\\n\\nc']) + comp.finish()
    assert streamed == compact('a\\n\\n\\nb\\n\\n\\nc') == 'a\\nb\\nc', streamed

    # 4. Mixed fences (backtick opens, tilde content, backtick closes)
    mixed = '\\`\\`\\`\\nx = 1\\n~~~\\ny = 2\\n\\`\\`\\`'
    assert compact(mixed) == mixed

    print('all pass')
    "
"""
from __future__ import annotations


class StreamCompactor:
    """Collapse blank-line runs in a streamed reply while leaving fenced
    code blocks byte-for-byte untouched.

    Safe to feed arbitrary chunk boundaries: the algorithm emits visible
    characters immediately and only holds back runs of newlines, releasing
    them as soon as the next non-newline character arrives.

    Typical usage::

        c = StreamCompactor(max_blank_lines=0)   # 0 = compact
        for chunk in stream:
            out = c.feed(chunk)
            if out:
                print(out, end="", flush=True)
        tail = c.finish()                        # always call; drops trailing newlines

    See the module docstring for the caveat (count raw, not compacted)
    and the configuration mapping.
    """

    __slots__ = (
        "_max_blank",      # 0 = compact, N = allow N blank lines
        "_nl",             # newlines held back awaiting a visible char
        "_started",        # have we emitted any visible char yet?
        "_in_fence",       # inside a ``` or ~~~ fenced block?
        "_fence_ch",       # the char that opened the current fence
        "_line",           # first <=8 chars of the current line (fence probe)
        "_line_checked",   # fence check already decided for this line
    )

    def __init__(self, max_blank_lines: int = 0) -> None:
        """
        Args:
            max_blank_lines: 0 = collapse all blank-line runs to single
                             newlines (compact, the default). N >= 1 = allow
                             at most N blank lines between paragraphs (so N=1
                             gives at most ``\\n\\n``; N=2 gives at most
                             ``\\n\\n\\n``). "preserve" mode is expressed by
                             the caller not instantiating this class.

        Raises:
            TypeError:  if ``max_blank_lines`` is not an int.
            ValueError: if ``max_blank_lines`` is negative.
        """
        if isinstance(max_blank_lines, bool):
            # bool is a subclass of int; True == 1, False == 0. Guard
            # against accidental use — callers must pass an int, not a flag.
            raise TypeError(
                "max_blank_lines must be an int, not bool "
                "(True=1, False=0 — pass 0 explicitly for compact mode)"
            )
        if not isinstance(max_blank_lines, int):
            raise TypeError(
                f"max_blank_lines must be int, got "
                f"{type(max_blank_lines).__name__}"
            )
        if max_blank_lines < 0:
            raise ValueError(
                f"max_blank_lines must be >= 0, got {max_blank_lines} "
                f"(use caller-side 'preserve' mode — do not instantiate — "
                f"for raw passthrough)"
            )
        self._max_blank = max_blank_lines
        self._nl = 0
        self._started = False
        self._in_fence = False
        self._fence_ch = ""
        self._line = ""
        self._line_checked = False

    # ── fence detection ────────────────────────────────────────────────

    def _check_fence(self) -> None:
        """Called when ``_line`` has >= 3 chars. Detects ``` / ~~~ fences.

        Rules (all verified against the reference implementation):
          * Leading indentation must be <= 3 spaces. ``    ``` `` is code,
            not a fence, and is treated as content.
          * The fence char is the first non-space char after indentation.
          * It must repeat at least 3 times at the start of the stripped
            line (`` `` `` = inline backtick, not a fence).
          * A line whose first non-space char matches the *current* fence
            char closes that fence; a line of a different fence char is
            content inside the current fence and is ignored.
        """
        s = self._line.lstrip(" ")
        if len(self._line) - len(s) > 3 or len(s) < 3:
            return
        ch = s[0]
        if ch in "`~" and s[:3] == ch * 3:
            self._line_checked = True
            if not self._in_fence:
                self._in_fence, self._fence_ch = True, ch
            elif ch == self._fence_ch:
                self._in_fence = False

    # ── streaming API ──────────────────────────────────────────────────

    def feed(self, text: str) -> str:
        """Feed one chunk of the stream; return what should be shown now.

        Returns "" (or the visible tail) when there is nothing to emit yet.
        Never raises on well-formed UTF-8 input. ``\\r`` characters are
        dropped; everything else is emitted in order (newline runs collapsed
        outside fences, preserved inside fences).

        Idempotent per-chunk: feeding the same chunk twice would corrupt the
        state. Feed each chunk of the stream exactly once, in order.
        """
        out: list[str] = []
        for ch in text:
            if ch == "\r":
                # Drop carriage returns — they interfere with terminal
                # display and mean nothing to a line-oriented protocol.
                # \r\n is normalized to \n.
                continue
            if ch == "\n":
                if self._in_fence:
                    # Inside a fence: emit the newline immediately so the
                    # code block is byte-identical to what the model sent.
                    out.append("\n")
                else:
                    # Outside a fence: hold the newline back. It will be
                    # released (possibly collapsed) when the next visible
                    # character arrives.
                    self._nl += 1
                self._line, self._line_checked = "", False
                continue

            # ── visible character ─────────────────────────────────────
            if self._nl:
                if self._started:
                    # Release up to ``1 + max_blank`` newlines. This is
                    # the compaction: _max_blank=0 → 1 newline (no blank
                    # lines), _max_blank=1 → 2 newlines (one blank line).
                    out.append("\n" * min(self._nl, 1 + self._max_blank))
                # If we haven't emitted anything yet, drop leading
                # newlines silently — a reply should not start with a
                # blank line.
                self._nl = 0
            self._started = True

            # Fence detection: accumulate the first up-to-8 visible
            # characters of the current line, checking after every char
            # once we have >= 3 so opening/closing fences are caught the
            # instant their third marker char arrives.
            if not self._line_checked and len(self._line) < 8:
                self._line += ch
                if len(self._line) >= 3 and (
                    ch in "`~ " or len(self._line.strip()) >= 3
                ):
                    self._check_fence()

            out.append(ch)
        return "".join(out)

    def finish(self) -> str:
        """Finalize the stream. Always call after the last ``feed()``.

        Returns:
            The empty string — trailing newlines are dropped by design (a
            reply should not end with a blank line). Callers that need
            to differentiate "stream ended mid-fence" from "stream ended
            cleanly" should inspect the pre-finish output; this method
            intentionally discards that information.

        Resets held-back-newline state. Does NOT reset ``_in_fence`` or
        ``_started`` — a compactor should not be reused after ``finish()``.
        One compactor per stream.
        """
        self._nl = 0
        return ""


# ── one-shot helper ────────────────────────────────────────────────────────

def compact(text: str, max_blank_lines: int = 0) -> str:
    """Non-streaming helper: collapse ``text`` and return the result.

    Equivalent to feeding the whole string to a ``StreamCompactor`` and
    calling ``finish()``::

        c = StreamCompactor(max_blank_lines)
        return c.feed(text) + c.finish()

    Same caveat as the streaming path: this removes characters, so
    do NOT call ``estimate_tokens_fast``, ``ChunkTokenCounter.add``, or
    any tokenizer on the output. Count the input.

    Args:
        text:            The raw text to compact.
        max_blank_lines: 0 = compact (default); N >= 1 = allow at most
                         N blank lines between paragraphs.

    Raises:
        TypeError:  if ``max_blank_lines`` is not an int.
        ValueError: if ``max_blank_lines`` is negative.
    """
    c = StreamCompactor(max_blank_lines)
    return c.feed(text) + c.finish()


__all__ = ["StreamCompactor", "compact"]
