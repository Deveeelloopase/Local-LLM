"""
local_llm_client/core/render.py — streaming terminal Markdown renderer.

Purpose
───────
Turn whatever a model emits (blockquote-wrapped replies such as Granite's
"> Dear Team,", doubled blank lines, "**bold**" noise, pipe tables, fenced
code) into a clean, modern, colourised terminal display WITHOUT Rich Live,
panels or cursor tricks. Output is plain ANSI text written line by line, so
terminal scrollback, selection and copy/paste keep working.

Design rules
────────────
* Streaming: prose is emitted as soon as it arrives (character level). Only
  structures that need a whole line (code, tables, fence headers, rules)
  wait for the newline.
* Chunk-boundary independent: feeding a text in any chunking produces the
  same output as feeding it at once (verified by ``_selfcheck``).
* Copy friendly: code lines carry NO gutter or prefix, only colour. The
  header/footer rules are separate lines, so selecting the code copies clean
  code. ``/copy`` (see chat.py) puts a block on the clipboard directly.
* Safe: ESC and every other C0/C1 control character, plus bidi override
  characters, are removed from model text before display, so a model (or an
  injected prompt) cannot drive the terminal (OSC 52 clipboard writes, title
  changes, cursor movement, hidden text).
* Robust: not a TTY -> output is the model text unchanged (only blank-line
  policy and control stripping apply), so pipes and files stay raw Markdown.
  ``NO_COLOR`` keeps the structure but drops colour. Terminals that cannot
  encode box-drawing characters get an ASCII glyph set.
* History stays RAW. Callers must store the model's raw text in the
  conversation history (not the rendered text) so the server's KV prefix
  cache sees the exact tokens the model produced.

Public API
──────────
    sanitize_terminal(text)      strip controls, tabs -> 4 spaces, drop CR
    strip_controls(text)         strip controls only (keeps \\t and \\n)
    StreamRenderer(max_blank_lines=0, *, width=None, color=None,
                   structured=None, stream=None)
        .feed(chunk) -> str      text to write now
        .finish()    -> str      flush; call once at end of stream
        .code_blocks             list[(lang, code)] seen so far
    render_text(text, **kw)      one-shot helper
    extract_code_blocks(text)    [(lang, code)] from raw Markdown
    style_footer(footer)         colourise "TTFT … · decode … · total …"
    write_out(console, s)        write+flush through a Rich console's file
    copy_to_clipboard(text)      OSC52 / pbcopy / clip / wl-copy / xclip
    enable_windows_vt()          switch on ANSI on Windows consoles

Verification (no test folder):
    python -m local_llm_client.core.render

version = 1.0.0
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

__version__ = "1.0.0"

__all__ = [
    "sanitize_terminal", "strip_controls", "StreamRenderer", "render_text",
    "extract_code_blocks", "style_footer", "write_out", "copy_to_clipboard",
    "enable_windows_vt",
]

# ──────────────────────────────────────────────────────────────────────────
# Sanitising
# ──────────────────────────────────────────────────────────────────────────
# C0 (except \t \n), DEL, C1, bidi embedding/override/isolate marks.
_CTRL_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f\x80-\x9f‪-‮⁦-⁩]"
)


def strip_controls(text: str) -> str:
    """Remove terminal-control and bidi-override characters (keeps \\t, \\n)."""
    if not text:
        return text
    return _CTRL_RE.sub("", text)


def sanitize_terminal(text: str) -> str:
    """strip_controls + tabs -> 4 spaces. CR is removed by strip_controls."""
    if not text:
        return text
    return _CTRL_RE.sub("", text).replace("\t", "    ")


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _vis_len(s: str) -> int:
    return sum(_char_width(c) for c in _ANSI_RE.sub("", s))


# ──────────────────────────────────────────────────────────────────────────
# Terminal capability helpers
# ──────────────────────────────────────────────────────────────────────────
def enable_windows_vt() -> bool:
    """Enable ANSI escape processing on Windows 10+ consoles. No-op elsewhere."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ok = False
        for std in (-11, -12):  # STDOUT, STDERR
            h = k32.GetStdHandle(std)
            mode = ctypes.c_uint32()
            if h and k32.GetConsoleMode(h, ctypes.byref(mode)):
                if k32.SetConsoleMode(h, mode.value | 0x0004):
                    ok = True
        return ok
    except Exception:
        return False


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _unicode_ok(stream) -> bool:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "•◦▪▎─│┼┌└…☐☑".encode(enc)
        return True
    except Exception:
        return False


def _term_width(default: int = 100) -> int:
    try:
        w = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        w = default
    return max(40, min(int(w), 200))


# ──────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────
class _Theme:
    """256-colour palette that reads on both dark and light terminals."""
    ACCENT = 81
    H1, H2, H3, H4 = 117, 111, 177, 250
    QUOTE = 244
    RULE = 240
    CODE_FG, CODE_BG = 215, 237
    LINK = 111
    DIM = 245
    KW, STR, NUM, COM, FUNC, DEC = 176, 114, 80, 244, 75, 215
    FOOT = {"ttft": 117, "prefill": 177, "decode": 114, "total": 215, "finish": 245}


_UNI = {
    "bullets": ("•", "◦", "▪"), "gutter": "▎ ", "hr": "─", "tl": "┌", "bl": "└",
    "h": "─", "v": "│", "x": "┼", "chk": "☑", "unchk": "☐", "ell": "…",
}
_ASC = {
    "bullets": ("*", "-", "+"), "gutter": "| ", "hr": "-", "tl": "+", "bl": "+",
    "h": "-", "v": "|", "x": "+", "chk": "[x]", "unchk": "[ ]", "ell": "...",
}


def _fg(n: int) -> str:
    return f"38;5;{n}"


# ──────────────────────────────────────────────────────────────────────────
# Inline (bold / italic / code / strike / links) — incremental
# ──────────────────────────────────────────────────────────────────────────
_PUNCT_ESC = set("\\`*_{}[]()#+-.!|~>")


class _Inline:
    """Incremental inline-Markdown styler for ONE line.

    ``feed(ch)`` returns the text to print for that character (possibly empty
    while a delimiter run is undecided); ``flush()`` ends the line and returns
    the tail plus a style reset. All style state is reset per line, so a stray
    ``**`` can never bleed colour into following lines.
    """

    def __init__(self, r: "StreamRenderer", base: Tuple[str, ...] = ()) -> None:
        self.r = r
        self.base = base
        self.attrs: set = set()
        self.prev = ""          # previous visible char
        self.run = ""           # pending delimiter run (* _ ~ `)
        self.ticks = 0          # >0 while inside an inline code span
        self.esc = False        # backslash pending
        self.link = ""          # captured "[text](url" text, "" when idle
        self.vis = 0            # visible char count (heading underline)
        self.sp = 0             # held-back spaces

    # -- style plumbing ---------------------------------------------------
    def _codes(self) -> str:
        if not self.r.color:
            return ""
        c: List[str] = list(self.base)
        if "b" in self.attrs:
            c.append("1")
        if "i" in self.attrs:
            c.append("3")
        if "s" in self.attrs:
            c.append("9")
        if "c" in self.attrs:
            c += [_fg(_Theme.CODE_FG), f"48;5;{_Theme.CODE_BG}"]
        return "\x1b[0;" + ";".join(c) + "m" if c else "\x1b[0m"

    def start(self) -> str:
        return self._codes() if (self.base and self.r.color) else ""

    def _change(self, add=(), drop=()) -> str:
        before = set(self.attrs)
        self.attrs |= set(add)
        self.attrs -= set(drop)
        return self._codes() if (self.attrs != before and self.r.color) else ""

    # -- emit -------------------------------------------------------------
    def _lit(self, s: str) -> str:
        if s:
            self.prev = s[-1]
            self.vis += len(s)
        return s

    def _resolve_run(self, nxt: str) -> str:
        run, self.run = self.run, ""
        c, n = run[0], len(run)
        prev, nxt_c = self.prev, nxt
        nxt_space = (nxt_c == "" or nxt_c.isspace())
        prev_space = (prev == "" or prev.isspace())
        prev_alnum = prev.isalnum()
        nxt_alnum = nxt_c.isalnum()
        if c == "`":
            self.ticks = n
            return self._change(add="c")
        if c == "~" and n != 2:
            return self._lit(run)
        if n > 3:
            return self._lit(run)
        can_open = (not nxt_space) and (n >= 2 or not prev_alnum or c == "~")
        if c == "_":
            can_open = (not nxt_space) and not prev_alnum
        can_close = (not prev_space)
        if c == "_":
            can_close = (not prev_space) and not nxt_alnum
        want = {1: {"i"}, 2: {"s"} if c == "~" else {"b"}, 3: {"b", "i"}}[n]
        if want <= self.attrs and can_close:
            return self._change(drop=want)
        if not (want & self.attrs) and can_open:
            return self._change(add=want)
        return self._lit(run)

    def _flush_link(self, ok_text: str = "") -> str:
        raw, self.link = self.link, ""
        if ok_text:
            return ok_text
        return self._lit(raw)

    def feed(self, ch: str) -> str:
        # Spaces are held back so trailing spaces (Markdown hard-break "  ")
        # never reach the terminal; they are released by the next character.
        if ch == " " and not (self.link or self.esc or self.ticks or self.run):
            self.sp += 1
            self.prev = " "
            return ""
        pre = ""
        if self.sp:
            pre = " " * self.sp
            self.vis += self.sp
            self.sp = 0
        out: List[str] = []
        self._feed1(ch, out)
        return pre + "".join(out)

    def _feed1(self, ch: str, out: List[str]) -> None:
        # link capture ----------------------------------------------------
        if self.link:
            self.link += ch
            L = self.link
            if len(L) > 400:
                out.append(self._flush_link())
                return
            if "](" in L:
                if ch == ")":
                    m = re.fullmatch(r"\[([^\]]*)\]\(([^)\s]*)\)", L)
                    if m and m.group(1):
                        text, url = m.group(1), m.group(2)
                        out.append(self._render_link(text, url))
                        self.link = ""
                    else:
                        out.append(self._flush_link())
                return
            if "]" in L:
                i = L.index("]")
                if i < len(L) - 1 and L[i + 1] != "(":
                    out.append(self._flush_link())
            return
        # backslash escape --------------------------------------------------
        if self.esc:
            self.esc = False
            if ch in _PUNCT_ESC:
                out.append(self._lit(ch))
                return
            out.append(self._lit("\\"))
        # inside inline code ---------------------------------------------
        if self.ticks:
            if ch == "`":
                self.run += "`"
                return
            if self.run:
                n = len(self.run)
                self.run = ""
                if n == self.ticks:
                    self.ticks = 0
                    out.append(self._change(drop="c"))
                    self._feed1(ch, out)
                    return
                out.append(self._lit("`" * n))
            out.append(self._lit(ch))
            return
        # delimiter runs ----------------------------------------------------
        if self.run:
            if ch == self.run[0]:
                self.run += ch
                return
            out.append(self._resolve_run(ch))
            if self.ticks:                      # a code span just opened
                self._feed1(ch, out)
                return
        if ch in "*_~`":
            self.run = ch
            return
        if ch == "\\":
            self.esc = True
            return
        if ch == "[":
            self.link = "["
            return
        if ch == " ":
            self.sp += 1
            self.prev = " "
            return
        out.append(self._lit(ch))

    def _render_link(self, text: str, url: str) -> str:
        self.vis += len(text) + len(url) + 3
        self.prev = ")"
        if not self.r.color:
            return f"{text} ({url})" if url and url != text else text
        base = self._codes()
        s = f"\x1b[0;4;{_fg(_Theme.LINK)}m{text}\x1b[0m"
        if url and url != text:
            s += f"\x1b[0;2;{_fg(_Theme.DIM)}m ({url})\x1b[0m"
        return s + (base if base else "")

    def flush(self) -> str:
        out: List[str] = []
        self.sp = 0
        if self.esc:
            out.append(self._lit("\\"))
            self.esc = False
        if self.link:
            out.append(self._flush_link())
        if self.run:
            if self.ticks:
                run, self.run = self.run, ""
                if len(run) != self.ticks:
                    out.append(self._lit(run))
                else:
                    out.append(self._change(drop="c"))
            else:
                out.append(self._resolve_run(""))
                if self.ticks:                 # unterminated code span opener
                    pass
        self.attrs.clear()
        self.ticks = 0
        if self.r.color and (self.base or True):
            out.append("\x1b[0m")
        return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Syntax highlighting (single line, tiny state for block comments/strings)
# ──────────────────────────────────────────────────────────────────────────
def _kw(s: str) -> frozenset:
    return frozenset(s.split())


_PY = _kw("False None True and as assert async await break class continue def del "
          "elif else except finally for from global if import in is lambda nonlocal "
          "not or pass raise return try while with yield match case self cls")
_JS = _kw("function const let var return if else for while do switch case break "
          "continue try catch finally throw new this class extends super import export "
          "from as default async await yield typeof instanceof in of null undefined "
          "true false void delete static get set interface type enum implements "
          "public private protected readonly namespace declare")
_C = _kw("int char float double void long short unsigned signed const volatile static "
         "extern inline struct union enum typedef sizeof if else for while do switch "
         "case default break continue return goto class namespace using template "
         "typename this new delete public private protected virtual override final "
         "auto bool true false nullptr NULL include define")
_JAVA = _kw("class interface enum extends implements import package public private "
            "protected static final abstract synchronized volatile transient native "
            "if else for while do switch case default break continue return throw "
            "throws try catch finally new this super instanceof boolean byte char "
            "short int long float double void true false null var record")
_GO = _kw("package import func var const type struct interface map chan go select "
          "defer if else for range switch case default break continue fallthrough "
          "return goto nil true false iota make new append len cap")
_RS = _kw("fn pub struct enum impl trait use mod extern static mut let match if else "
          "for while loop return break continue async await unsafe type where self "
          "Self super true false None Some Ok Err crate move ref dyn as in")
_SH = _kw("if then else elif fi for while do done case esac function return export "
          "local echo exit in set source cd ls rm cp mv mkdir cat grep sed awk sudo "
          "apt pip npm git curl wget docker systemctl chmod chown kill")
_SQL = _kw("select from where insert into values update set delete create table alter "
           "drop join left right inner outer on as and or not null is in like group by "
           "order having limit distinct union all exists case when then end primary key "
           "foreign references default index view with")
_RB = _kw("def class module require include extend attr_accessor attr_reader "
          "attr_writer if elsif else unless case when for while until begin rescue "
          "ensure raise return yield super self nil true false and or not in do end")
_PHP = _kw("function class interface trait use namespace extends implements public "
           "private protected static final abstract if else elseif for foreach while "
           "do switch case default break continue return throw try catch finally new "
           "this parent self echo print die exit isset unset empty array true false null")

_LANGS: Dict[str, dict] = {}


def _reg(names: str, kw, com=None, block=None, tq=False, ci=False) -> None:
    spec = {"kw": kw, "com": com, "block": block, "tq": tq, "ci": ci}
    for n in names.split():
        _LANGS[n] = spec


_reg("python py python3", _PY, "#", tq=True)
_reg("javascript js jsx typescript ts tsx node json5 mjs cjs", _JS, "//", ("/*", "*/"))
_reg("c h cpp c++ cc hpp cxx cs csharp objc arduino ino", _C, "//", ("/*", "*/"))
_reg("java kotlin kt scala swift dart", _JAVA, "//", ("/*", "*/"))
_reg("go golang", _GO, "//", ("/*", "*/"))
_reg("rust rs", _RS, "//", ("/*", "*/"))
_reg("bash sh shell zsh fish console terminal shell-session", _SH, "#")
_reg("sql mysql postgres postgresql sqlite psql", _SQL, "--", ("/*", "*/"), ci=True)
_reg("ruby rb", _RB, "#")
_reg("php", _PHP, "//", ("/*", "*/"))
_reg("yaml yml toml ini conf cfg env dockerfile makefile make", frozenset(), "#")
_reg("json jsonc", frozenset({"true", "false", "null"}), None)
_reg("css scss less", frozenset(), None, ("/*", "*/"))
_reg("html xml svg vue", frozenset(), None, ("<!--", "-->"))
_reg("lua", frozenset({"function", "end", "if", "then", "else", "elseif", "for", "while",
                       "do", "return", "local", "nil", "true", "false", "and", "or",
                       "not", "in", "break"}), "--", ("--[[", "]]"))

_STR_RE = r"""(?:[rbfuRBFU]{0,2}"(?:[^"\\]|\\.)*"|[rbfuRBFU]{0,2}'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"""
_NUM_RE = r"\b(?:0[xX][0-9a-fA-F_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)\b"


def _paint(r: "StreamRenderer", tok: str, s: str) -> str:
    if not r.color or not s:
        return s
    col = {"k": _Theme.KW, "s": _Theme.STR, "n": _Theme.NUM, "c": _Theme.COM,
           "f": _Theme.FUNC, "d": _Theme.DEC, "t": _Theme.KW, "a": _Theme.NUM}[tok]
    extra = "3;" if tok == "c" else ("1;" if tok == "k" else "")
    return f"\x1b[{extra}{_fg(col)}m{s}\x1b[0m"


class _Highlighter:
    """Per-code-block, line-at-a-time highlighter with minimal cross-line state."""

    def __init__(self, r: "StreamRenderer", lang: str) -> None:
        self.r = r
        self.lang = lang
        self.spec = _LANGS.get(lang)
        self.open_tq: Optional[str] = None      # '"""' or "'''"
        self.in_block = False
        self._rx = None
        if self.spec:
            kws = "|".join(sorted(map(re.escape, self.spec["kw"]), key=len, reverse=True))
            parts = []
            if self.spec["com"]:
                parts.append(r"(?P<c>" + re.escape(self.spec["com"]) + r".*)")
            if self.spec["block"]:
                parts.append(r"(?P<c2>" + re.escape(self.spec["block"][0]) + r".*?(?:"
                             + re.escape(self.spec["block"][1]) + r"|$))")
            parts.append(r"(?P<s>" + _STR_RE + r")")
            if lang in ("html", "xml", "svg", "vue"):
                parts.append(r"(?P<t></?[A-Za-z][\w:.-]*|/?>)")
            if lang == "python":
                parts.append(r"(?P<d>^\s*@[\w.]+)")
            parts.append(r"(?P<n>" + _NUM_RE + r")")
            if kws:
                parts.append(r"(?P<k>\b(?:" + kws + r")\b)")
            parts.append(r"(?P<f>\b[A-Za-z_]\w*(?=\())")
            flags = re.IGNORECASE if self.spec["ci"] else 0
            self._rx = re.compile("|".join(parts), flags)

    def line(self, text: str) -> str:
        r = self.r
        if not r.color:
            return text
        if self.lang in ("diff", "patch"):
            if text.startswith("+") and not text.startswith("+++"):
                return f"\x1b[{_fg(_Theme.STR)}m{text}\x1b[0m"
            if text.startswith("-") and not text.startswith("---"):
                return f"\x1b[{_fg(203)}m{text}\x1b[0m"
            if text.startswith("@@"):
                return f"\x1b[{_fg(_Theme.FUNC)}m{text}\x1b[0m"
            return text
        if self._rx is None:
            return text
        out: List[str] = []
        pos = 0
        s = text
        if self.in_block:
            end = self.spec["block"][1]
            j = s.find(end)
            if j < 0:
                return _paint(r, "c", s)
            out.append(_paint(r, "c", s[: j + len(end)]))
            pos = j + len(end)
            self.in_block = False
        elif self.open_tq:
            j = s.find(self.open_tq)
            if j < 0:
                return _paint(r, "s", s)
            out.append(_paint(r, "s", s[: j + 3]))
            pos = j + 3
            self.open_tq = None
        while pos < len(s):
            if self.spec["tq"]:
                m3 = re.compile(r"""[rbfuRBFU]{0,2}("{3}|'{3})""").match(s, pos)
                if m3:
                    q = m3.group(1)
                    j = s.find(q, m3.end())
                    if j < 0:
                        out.append(_paint(r, "s", s[pos:]))
                        self.open_tq = q
                        pos = len(s)
                        break
                    out.append(_paint(r, "s", s[pos: j + 3]))
                    pos = j + 3
                    continue
            m = self._rx.search(s, pos)
            if not m:
                out.append(s[pos:])
                break
            if self.spec["tq"]:
                # make sure a triple-quote start earlier than m is handled first
                m3 = re.compile(r"""[rbfuRBFU]{0,2}("{3}|'{3})""").search(s, pos)
                if m3 and m3.start() <= m.start():
                    out.append(s[pos: m3.start()])
                    pos = m3.start()
                    continue
            out.append(s[pos: m.start()])
            kind = m.lastgroup or ""
            tok = m.group(0)
            if kind == "c2":
                if not tok.endswith(self.spec["block"][1]) or len(tok) < len(
                        self.spec["block"][0]) + len(self.spec["block"][1]):
                    if not tok.rstrip().endswith(self.spec["block"][1]):
                        self.in_block = True
                kind = "c"
            if self.lang in ("json", "jsonc") and kind == "s":
                rest = s[m.end():].lstrip()
                kind = "f" if rest.startswith(":") else "s"
            out.append(_paint(r, kind if kind in "kscnfdta" else "n", tok))
            pos = m.end()
        return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Block renderer
# ──────────────────────────────────────────────────────────────────────────
_FENCE_OPEN_RE = re.compile(r"^(?P<ind> {0,3})(?P<fence>`{3,}|~{3,})\s*(?P<info>[^`]*)$")
_TABLE_SEP_RE = re.compile(r"^\s*:?-{1,}:?\s*$")
_TASKS = ("[ ] ", "[x] ", "[X] ")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


class StreamRenderer:
    """Incremental Markdown -> ANSI renderer. See module docstring."""

    def __init__(self, max_blank_lines: Optional[int] = 0, *,
                 width: Optional[int] = None, color: Optional[bool] = None,
                 structured: Optional[bool] = None, stream=None) -> None:
        if isinstance(max_blank_lines, bool):
            raise TypeError("max_blank_lines must be int or None, not bool")
        if max_blank_lines is not None:
            if not isinstance(max_blank_lines, int):
                raise TypeError("max_blank_lines must be int or None")
            if max_blank_lines < 0:
                raise ValueError("max_blank_lines must be >= 0")
        out = stream if stream is not None else sys.stdout
        tty = _stream_is_tty(out)
        no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        self.structured = tty if structured is None else bool(structured)
        self.color = (tty and not no_color) if color is None else bool(color)
        if self.color:
            self.structured = True
        self.glyph = _UNI if _unicode_ok(out) else _ASC
        self.width = int(width) if width else _term_width()
        self.max_blank = max_blank_lines
        self.code_blocks: List[Tuple[str, str]] = []
        # block state
        self._line = ""
        self._mode = "bol"               # bol | text | buf
        self._buf_kind = ""              # code | fence | table | hr
        self._nl = 0
        self._started = False
        self._inline: Optional[_Inline] = None
        self._h_level = 0
        self._tail_hold = ""
        self._in_fence = False
        self._fence_ch = ""
        self._fence_len = 0
        self._fence_lang = ""
        self._code_lines: List[str] = []
        self._hl: Optional[_Highlighter] = None
        self._tbl: List[str] = []
        self._unwrap: Optional[bool] = None   # decided on first non-blank line

    # -- helpers ------------------------------------------------------------
    def _sgr(self, *codes: str) -> str:
        return ("\x1b[" + ";".join(codes) + "m") if (self.color and codes) else ""

    def _rst(self) -> str:
        return "\x1b[0m" if self.color else ""

    def _release(self, raw: bool = False, at_least: int = 0) -> str:
        if not self._started:
            self._nl = 0
            return ""
        if raw or self.max_blank is None:
            n = self._nl
        else:
            n = min(self._nl, 1 + self.max_blank)
        n = max(n, at_least)
        self._nl = 0
        return "\n" * n

    def _emit_line(self, text: str, raw: bool = False, at_least: int = 0) -> str:
        s = self._release(raw=raw, at_least=at_least) + text
        self._started = True
        self._nl = 1
        return s

    # -- public ---------------------------------------------------------------
    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if not self.structured:
            return self._feed_plain(chunk)
        out: List[str] = []
        for ch in sanitize_terminal(chunk):
            self._char(ch, out)
        return "".join(out)

    def finish(self) -> str:
        if not self.structured:
            self._nl = 0
            return ""
        out: List[str] = []
        if self._line or self._mode in ("text", "buf"):
            self._end_line(out, final=True)
        if self._tbl:
            out.append(self._flush_table())
        if self._in_fence:
            out.append(self._close_fence(unclosed=True))
        if self._inline is not None:
            out.append(self._inline.flush())
            self._inline = None
        self._nl = 0
        return "".join(out)

    # -- plain mode (not a TTY): blank-line policy + control stripping ----
    def _feed_plain(self, chunk: str) -> str:
        out: List[str] = []
        for ch in strip_controls(chunk):
            if ch == "\n":
                if self._in_fence:
                    out.append("\n")
                else:
                    self._nl += 1
                self._pl = ""
                continue
            if self._nl:
                if self._started:
                    if self.max_blank is None:
                        out.append("\n" * self._nl)
                    else:
                        out.append("\n" * min(self._nl, 1 + self.max_blank))
                self._nl = 0
            self._started = True
            out.append(ch)
            pl = getattr(self, "_pl", "")
            if len(pl) < 8:
                pl += ch
                self._pl = pl
                s = pl.lstrip(" ")
                if len(pl) - len(s) <= 3 and len(s) >= 3 and s[0] in "`~" and s[:3] == s[0] * 3:
                    self._in_fence = not self._in_fence
                    self._pl = "\x00" * 8     # decided for this line
        return "".join(out)

    # -- character state machine ---------------------------------------------
    def _char(self, ch: str, out: List[str]) -> None:
        if ch == "\n":
            self._end_line(out, final=False)
            return
        if self._mode == "text":
            out.append(self._inline.feed(ch))  # type: ignore[union-attr]
            return
        self._line += ch
        if self._mode == "bol":
            if self._in_fence:
                self._mode, self._buf_kind = "buf", "code"
                return
            self._try_classify(out, final=False)

    # -- classification of a line start ------------------------------------
    def _try_classify(self, out: List[str], final: bool) -> bool:
        s = self._line
        n = len(s)
        i = 0
        while i < n and s[i] == " ":
            i += 1
        indent = i
        if i == n:
            if final:
                self._blank_line(out)
                return True
            return False
        depth = 0
        j = i
        while j < n and s[j] in "> ":
            if s[j] == ">":
                depth += 1
            j += 1
        if depth == 0:
            j = i
        rest = s[j:]
        if depth and rest == "":
            if final:
                self._emit_text_line(out, indent, depth, "", [])
                return True
            return False
        c = rest[0]

        def text_line(content: str = rest, base: Tuple[str, ...] = (), prefix: str = "") -> bool:
            self._start_text(out, indent, depth, prefix, base, content)
            return True

        if c == "#":
            k = 0
            while k < len(rest) and rest[k] == "#":
                k += 1
            if k > 6:
                return text_line()
            if k == len(rest):
                return text_line() if final else False
            if rest[k] == " ":
                return self._start_heading(out, indent, depth, k, rest[k + 1:])
            return text_line()
        if c in "-*+":
            if len(rest) == 1:
                return text_line() if final else False
            if rest[1] == " ":
                tail = rest[2:]
                if not final and any(t.startswith(tail) and t != tail for t in _TASKS):
                    return False
                for t in _TASKS:
                    if tail.startswith(t):
                        return self._start_bullet(out, indent, depth, tail[4:], task=t)
                return self._start_bullet(out, indent, depth, tail)
            cand = rest.replace(" ", "")
            if c in "-*" and set(cand) == {c} and len(cand) >= 2 and depth == 0:
                if final:
                    if len(cand) >= 3:
                        out.append(self._hr(indent))
                        return True
                    return text_line()
                return False
            return text_line()
        if c.isdigit():
            k = 0
            while k < len(rest) and rest[k].isdigit():
                k += 1
            if k == len(rest):
                return text_line() if final else False
            if rest[k] in ".)":
                if k + 1 == len(rest):
                    return text_line() if final else False
                if rest[k + 1] == " " and k <= 9:
                    return self._start_ordered(out, indent, depth, rest[:k], rest[k + 2:])
            return text_line()
        if c == "|" and depth == 0:
            self._mode, self._buf_kind = "buf", "table"
            return True
        if c in "`~" and depth == 0:
            if len(rest) < 3 and rest == c * len(rest) and not final:
                return False
            if rest[:3] == c * 3 and indent <= 3:
                self._mode, self._buf_kind = "buf", "fence"
                return True
        return text_line()

    # -- block starters --------------------------------------------------------
    def _eff_depth(self, depth: int) -> int:
        """Quote depth to display. A reply whose FIRST non-blank line is a
        quote (Granite style) is unwrapped: one quote level is dropped for the
        whole reply. Otherwise quotes are shown with a gutter bar."""
        if self._unwrap is None:
            if depth:
                self._unwrap = not self._started
            elif self._started:
                self._unwrap = False
        d = depth - 1 if (self._unwrap and depth) else depth
        return max(d, 0)

    def _quote_prefix(self, depth: int) -> str:
        d = self._eff_depth(depth)
        if d <= 0:
            return ""
        return self._sgr(_fg(_Theme.QUOTE)) + self.glyph["gutter"] * d + self._rst()

    def _mark_first_line(self, depth: int) -> None:
        self._eff_depth(depth)

    def _begin(self, out: List[str], at_least: int = 0) -> None:
        out.append(self._release(at_least=at_least))
        self._started = True

    def _start_text(self, out, indent, depth, prefix, base, content) -> None:
        self._mark_first_line(depth)
        self._begin(out)
        out.append(self._quote_prefix(depth) if depth else "")
        if indent and not depth:
            out.append(" " * min(indent, 8))
        out.append(prefix)
        self._h_level = 0
        self._inline = _Inline(self, base)
        out.append(self._inline.start())
        self._mode = "text"
        for ch in content:
            out.append(self._inline.feed(ch))
        self._line = ""

    def _emit_text_line(self, out, indent, depth, content, base) -> None:
        """A bare '>' line: a bar-only line, or a plain blank line when the
        reply is unwrapped."""
        if self._eff_depth(depth) <= 0:
            self._blank_line(out)
            return
        self._begin(out)
        out.append(self._quote_prefix(depth))
        out.append(self._rst())
        self._nl = 1
        self._mode = "bol"
        self._line = ""

    def _start_heading(self, out, indent, depth, level, content) -> bool:
        col = {1: _Theme.H1, 2: _Theme.H2, 3: _Theme.H3}.get(level, _Theme.H4)
        self._mark_first_line(depth)
        self._begin(out, at_least=2)
        out.append(self._quote_prefix(depth))
        self._inline = _Inline(self, ("1", _fg(col)))
        out.append(self._inline.start())
        self._h_level = level
        self._mode = "text"
        for ch in content:
            out.append(self._inline.feed(ch))
        self._line = ""
        return True

    def _start_bullet(self, out, indent, depth, content, task: str = "") -> bool:
        lvl = min(indent // 2, 6)
        if task:
            g = self.glyph["unchk"] if task == "[ ] " else self.glyph["chk"]
        else:
            g = self.glyph["bullets"][lvl % 3]
        pre = " " * (2 + 2 * lvl) + self._sgr(_fg(_Theme.ACCENT)) + g + self._rst() + " "
        self._start_text(out, 0, depth, pre, (), content)
        return True

    def _start_ordered(self, out, indent, depth, num, content) -> bool:
        lvl = min(indent // 2, 6)
        pre = (" " * (2 + 2 * lvl) + self._sgr(_fg(_Theme.ACCENT), "1") + num + "."
               + self._rst() + " ")
        self._start_text(out, 0, depth, pre, (), content)
        return True

    def _hr(self, indent: int) -> str:
        w = max(10, min(self.width - 2, 72))
        self._line = ""
        self._mode = "bol"
        return self._emit_line(self._sgr(_fg(_Theme.RULE)) + self.glyph["hr"] * w + self._rst())

    def _blank_line(self, out: List[str]) -> None:
        if self._tbl:
            out.append(self._flush_table())
        self._nl += 1
        self._line = ""
        self._mode = "bol"

    # -- end of line ---------------------------------------------------------
    def _end_line(self, out: List[str], final: bool) -> None:
        mode = self._mode
        if mode == "text":
            tail = self._inline.flush()  # type: ignore[union-attr]
            out.append(tail)
            if self._h_level == 1:
                w = min(max(self._inline.vis, 3), self.width - 2)  # type: ignore[union-attr]
                out.append("\n" + self._sgr(_fg(_Theme.RULE)) + self.glyph["hr"] * w + self._rst())
            self._inline = None
            self._h_level = 0
            self._nl = 1
            self._mode = "bol"
            self._line = ""
            return
        if mode == "buf":
            line = self._line
            self._line = ""
            self._mode = "bol"
            kind = self._buf_kind
            self._buf_kind = ""
            if kind == "code":
                out.append(self._code_line(line))
            elif kind == "fence":
                out.append(self._fence_line(line))
            elif kind == "table":
                self._tbl.append(line)
            return
        # mode == "bol": undecided prefix line
        if self._in_fence:
            out.append(self._code_line(self._line))
            self._line = ""
            return
        if self._tbl:
            # a non-table line ends the table (this line is classified below)
            pass
        if self._line == "" or self._line.strip(" ") == "":
            self._line = ""
            self._blank_line(out)
            return
        if self._tbl and not self._line.lstrip(" ").startswith("|"):
            out.append(self._flush_table())
        decided = self._try_classify(out, final=True)
        if decided and self._mode == "text":
            self._end_line(out, final=True)
        elif decided and self._mode == "buf":
            self._end_line(out, final=True)
        self._line = "" if self._mode == "bol" else self._line

    # -- fences & code ---------------------------------------------------------
    def _fence_line(self, line: str) -> str:
        if self._tbl:
            pre = self._flush_table()
        else:
            pre = ""
        m = _FENCE_OPEN_RE.match(line)
        if not m:
            # not a real fence (e.g. ``` inside text): render as text
            return pre + self._plain_text_line(line)
        info = m.group("info").strip()
        lang = (info.split() or [""])[0].strip("{}.").lower()
        lang = lang.split("{")[0].split(":")[0].split(",")[0]
        self._in_fence = True
        self._fence_ch = m.group("fence")[0]
        self._fence_len = len(m.group("fence"))
        self._fence_lang = lang
        self._code_lines = []
        self._hl = _Highlighter(self, lang)
        w = max(16, min(self.width - 1, 80))
        name = lang or "text"
        head = (self._sgr(_fg(_Theme.RULE)) + self.glyph["tl"] + self.glyph["h"] + " "
                + self._rst() + self._sgr("1", _fg(_Theme.H1)) + name + self._rst()
                + " " + self._sgr(_fg(_Theme.RULE))
                + self.glyph["h"] * max(2, w - len(name) - 4) + self._rst())
        return pre + self._emit_line(head, at_least=1 if self._started else 0)

    def _code_line(self, line: str) -> str:
        m = re.match(r"^ {0,3}(" + re.escape(self._fence_ch) + "{" + str(self._fence_len) + r",})\s*$", line)
        if m:
            return self._close_fence()
        self._code_lines.append(line)
        text = self._hl.line(line) if self._hl else line  # type: ignore[union-attr]
        return self._emit_line(text, raw=True)

    def _close_fence(self, unclosed: bool = False) -> str:
        code = "\n".join(self._code_lines)
        self.code_blocks.append((self._fence_lang, code))
        n = len(self._code_lines)
        label = f"{n} line{'s' if n != 1 else ''}"
        if unclosed:
            label += " (unterminated)"
        w = max(16, min(self.width - 1, 80))
        foot = (self._sgr(_fg(_Theme.RULE)) + self.glyph["bl"] + self.glyph["h"] * max(2, w - len(label) - 3)
                + " " + self._sgr(_fg(_Theme.DIM)) + label + self._rst())
        self._in_fence = False
        self._hl = None
        self._code_lines = []
        self._mode = "bol"
        return self._emit_line(foot, raw=True)

    def _plain_text_line(self, line: str) -> str:
        il = _Inline(self)
        body = "".join(il.feed(c) for c in line) + il.flush()
        return self._emit_line(body)

    # -- tables ---------------------------------------------------------------
    @staticmethod
    def _split_row(line: str) -> List[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|") and not s.endswith("\\|"):
            s = s[:-1]
        cells, cur, i = [], [], 0
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
                cur.append("|")
                i += 2
                continue
            if s[i] == "|":
                cells.append("".join(cur).strip())
                cur = []
            else:
                cur.append(s[i])
            i += 1
        cells.append("".join(cur).strip())
        return [_BR_RE.sub(" ", c) for c in cells]

    def _inline_once(self, text: str, base: Tuple[str, ...] = ()) -> str:
        il = _Inline(self, base)
        s = il.start() + "".join(il.feed(c) for c in text) + il.flush()
        return s

    def _flush_table(self) -> str:
        lines, self._tbl = self._tbl, []
        rows = [self._split_row(l) for l in lines if l.strip()]
        if not rows:
            return ""
        has_sep = len(rows) >= 2 and all(_TABLE_SEP_RE.match(c) for c in rows[1] if True) and len(rows[1]) >= 1
        if len(rows) < 2 or not has_sep:
            return "".join(self._plain_text_line(l.strip()) for l in lines if l.strip())
        aligns = []
        for c in rows[1]:
            c = c.strip()
            aligns.append("c" if c.startswith(":") and c.endswith(":") else ("r" if c.endswith(":") else "l"))
        head, body = rows[0], rows[2:]
        ncol = max(len(head), *(len(r) for r in body)) if body else len(head)
        norm = lambda r: (r + [""] * ncol)[:ncol]
        head, body = norm(head), [norm(r) for r in body]
        aligns = (aligns + ["l"] * ncol)[:ncol]
        cells_h = [self._inline_once(c, ("1", _fg(_Theme.H1))) for c in head]
        cells_b = [[self._inline_once(c) for c in r] for r in body]
        widths = [max([_vis_len(cells_h[i])] + [_vis_len(r[i]) for r in cells_b]) for i in range(ncol)]
        total = sum(widths) + 3 * (ncol - 1) + 2
        if total > self.width:
            # too wide for the terminal: stacked "header: value" records
            out = []
            for r in cells_b:
                for i in range(ncol):
                    out.append(self._emit_line("  " + cells_h[i] + self._rst() + self._sgr(_fg(_Theme.DIM)) + ": "
                                               + self._rst() + r[i]))
                out.append(self._emit_line(self._sgr(_fg(_Theme.RULE)) + "  " + self.glyph["hr"] * 8 + self._rst()))
            if out:
                self._nl = 1
            return "".join(out)

        def pad(s: str, w: int, a: str) -> str:
            gap = w - _vis_len(s)
            if a == "r":
                return " " * gap + s
            if a == "c":
                return " " * (gap // 2) + s + " " * (gap - gap // 2)
            return s + " " * gap

        sep = self._sgr(_fg(_Theme.RULE)) + " " + self.glyph["v"] + " " + self._rst()
        out = [self._emit_line(" " + sep.join(pad(cells_h[i], widths[i], aligns[i]) + self._rst()
                                              for i in range(ncol)), at_least=0)]
        rule = (self._sgr(_fg(_Theme.RULE)) + self.glyph["h"] + (self.glyph["h"] + self.glyph["x"] + self.glyph["h"]).join(
            self.glyph["h"] * widths[i] for i in range(ncol)) + self.glyph["h"] + self._rst())
        out.append(self._emit_line(rule))
        for r in cells_b:
            out.append(self._emit_line(" " + sep.join(pad(r[i], widths[i], aligns[i]) + self._rst()
                                                      for i in range(ncol))))
        return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Helpers used by chat.py
# ──────────────────────────────────────────────────────────────────────────
def render_text(text: str, max_blank_lines: Optional[int] = 0, **kw) -> str:
    """One-shot render of a complete reply."""
    r = StreamRenderer(max_blank_lines, **kw)
    return r.feed(text) + r.finish()


_FENCE_RE = re.compile(
    r"^[ ]{0,3}(?P<f>`{3,}|~{3,})[ \t]*(?P<info>[^\n`]*)\n(?P<body>.*?)(?:^[ ]{0,3}(?P=f)[`~]*[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Return [(lang, code)] for every fenced block in raw Markdown."""
    out: List[Tuple[str, str]] = []
    for m in _FENCE_RE.finditer(strip_controls(text or "").replace("\r", "")):
        info = m.group("info").strip().split()
        lang = (info[0] if info else "").strip("{}.").lower()
        body = m.group("body")
        if body.endswith("\n"):
            body = body[:-1]
        out.append((lang, body))
    return out


def style_footer(footer: str, color: Optional[bool] = None, stream=None) -> str:
    """Colourise the ``TTFT … · decode … · total … · finish=…`` footer."""
    if color is None:
        s = stream if stream is not None else sys.stdout
        color = _stream_is_tty(s) and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    footer = strip_controls(footer)
    if not color:
        return footer
    parts = footer.split(" · ")
    out = []
    for p in parts:
        key = ("ttft" if p.startswith("TTFT") else "prefill" if p.startswith("prefill")
               else "decode" if p.startswith("decode") else "total" if p.startswith("total")
               else "finish" if p.startswith("finish") else "")
        if key:
            label, _, rest = p.partition(" ")
            out.append(f"\x1b[2;{_fg(_Theme.DIM)}m{label}\x1b[0m \x1b[{_fg(_Theme.FOOT[key])}m{rest}\x1b[0m")
        else:
            out.append(f"\x1b[2m{p}\x1b[0m")
    return f"\x1b[2;{_fg(_Theme.RULE)}m·\x1b[0m ".join(out)


def write_out(console, s: str) -> None:
    """Write already-rendered text through a Rich console's output file."""
    if not s:
        return
    f = getattr(console, "file", None) or sys.stdout
    try:
        f.write(s)
        f.flush()
    except UnicodeEncodeError:
        enc = getattr(f, "encoding", None) or "ascii"
        f.write(s.encode(enc, "replace").decode(enc))
        f.flush()


def copy_to_clipboard(text: str, stream=None) -> str:
    """Copy ``text`` to the system clipboard. Returns the method used, or ''."""
    data = text.encode("utf-8")
    tries: List[List[str]] = []
    if sys.platform == "darwin":
        tries.append(["pbcopy"])
    elif os.name == "nt":
        tries.append(["powershell", "-NoProfile", "-Command",
                      "[Console]::InputEncoding=[Text.Encoding]::UTF8;"
                      "Set-Clipboard -Value ([Console]::In.ReadToEnd())"])
    else:
        if os.environ.get("WAYLAND_DISPLAY"):
            tries.append(["wl-copy"])
        if os.environ.get("DISPLAY"):
            tries += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in tries:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=data, check=True, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return cmd[0]
            except Exception:
                continue
    out = stream if stream is not None else sys.stdout
    if _stream_is_tty(out) and len(data) <= 100_000:
        osc = "\x1b]52;c;" + base64.b64encode(data).decode("ascii") + "\x07"
        try:
            out.write(osc)
            out.flush()
            return "osc52"
        except Exception:
            pass
    return ""


# ──────────────────────────────────────────────────────────────────────────
# Self check:  python -m local_llm_client.core.render
# ──────────────────────────────────────────────────────────────────────────
def _selfcheck() -> None:  # pragma: no cover - exercised by `python -m`
    import random

    def strip(s: str) -> str:
        return _ANSI_RE.sub("", s)

    sample = (
        "> Dear Team,\n>\n> We **extend** the sprint by `one week`.\n>\n> - item one\n"
        "> - item two\n\n\n\n# Title\nText with snake_case_name and 2 * 3 * 4 and *it* **b**.\n"
        "## Sub\n1. first\n2) second\n- [ ] todo\n- [x] done\n  - nested\n---\n"
        "| a | b |\n|---|--:|\n| 1 | **2** |\n| 3 | 4 |\n\n"
        "```python\ndef f(x):\n    \"\"\"doc\n    more\"\"\"\n    return x  # c\n```\n"
        "after [link](http://e.x) \x1b]52;c;ZXZpbA==\x07end"
    )
    one = render_text(sample, 0, width=100, color=True, structured=True)
    assert "\x1b]" not in one and "\x07" not in one, "escape injection survived"
    assert "evil" not in strip(one)
    plain = strip(one)
    assert plain.startswith("Dear Team,\nWe extend"), repr(plain[:40])
    assert ">" not in plain.split("# ")[0], plain
    two = strip(render_text("Intro line\n> quoted a\n>\n> quoted b\n", 0, width=100, color=True, structured=True))
    assert "▎ quoted a" in two and "▎ quoted b" in two and "\n▎" in two, repr(two)
    assert "snake_case_name" in plain and "2 * 3 * 4" in plain
    assert "one week" in plain and "**" not in plain
    assert "☐ todo" in plain and "☑ done" in plain
    assert "def f(x):" in plain and "return x  # c" in plain
    assert "link (http://e.x)" in plain
    assert "│" in plain and "1 │ 2" in plain.replace("  ", " ") or "│" in plain
    for seed in range(60):
        rnd = random.Random(seed)
        r = StreamRenderer(0, width=100, color=True, structured=True)
        got, i = [], 0
        while i < len(sample):
            k = rnd.randint(1, 9)
            got.append(r.feed(sample[i:i + k]))
            i += k
        got.append(r.finish())
        assert "".join(got) == one, f"chunking changed output (seed {seed})"
    r = StreamRenderer(0, color=False, structured=False)
    raw = r.feed("a\n\n\n\nb\x1b[31m\n") + r.finish()
    assert raw == "a\nb[31m", repr(raw)
    blocks = extract_code_blocks(sample)
    assert blocks and blocks[0][0] == "python" and "def f(x):" in blocks[0][1]
    print("render.py selfcheck: OK")


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
