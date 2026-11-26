"""
serving/chat_template.py — model-native prompt rendering and thinking control.

Why this exists
---------------
Every GGUF carries its own Jinja chat template (``tokenizer.chat_template``).
That template is the ONLY format the model was trained on. Hand-written
formatters work for a handful of families and silently produce garbage for
the rest (the model sees an unknown "### User:" layout, never emits its
end-of-turn token, and invents extra turns).

This module renders the model's own template inside a sandbox, detects how
the template controls reasoning ("dialect"), and returns a finished prompt
string. Any failure returns None so the caller falls back to the legacy
formatter — it can never make a request fail.

Version 1.0.0 — no new dependencies (jinja2 ships with llama-cpp-python).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger("chat_template")

__all__ = ["TemplateRenderer", "register", "get_cached", "detect_dialect"]

# ── thinking dialects ───────────────────────────────────────────────────
#   none      template has no reasoning switch (model cannot be told to think)
#   qwen3     kwarg ``enable_thinking``; off = template pre-fills an empty block
#   granite   kwarg ``thinking``
#   gptoss    kwarg ``reasoning_effort`` (low/medium/high); no hard "off"
#   forced    generation prompt ends inside an open <think>; off = close it
_RE_ENABLE = re.compile(r"\benable_thinking\b")
_RE_THINKING_VAR = re.compile(r"\{%-?\s*if\s+thinking\b|\bthinking\s+is\s+(?:defined|true)", re.I)
_RE_EFFORT = re.compile(r"\breasoning_effort\b")
_RE_ANY_THINK = re.compile(r"<think>|<thinking>|<\|thinking\|>|reasoning")


def detect_dialect(template: str) -> str:
    """Classify how a chat template switches reasoning on or off."""
    t = template or ""
    if _RE_ENABLE.search(t):
        return "qwen3"
    if _RE_EFFORT.search(t):
        return "gptoss"
    if _RE_THINKING_VAR.search(t):
        return "granite"
    if _RE_ANY_THINK.search(t):
        return "forced"            # decided after a dry render, see _probe_forced
    return "none"


# ── jinja environment ───────────────────────────────────────────────────
def _make_env():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str):
        raise ValueError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    env.filters["tojson"] = tojson
    return env


def _fold_for_strict_templates(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if system:
        if rest and rest[0].get("role") == "user":
            rest[0]["content"] = system + "\n\n" + (rest[0].get("content") or "")
        else:
            rest.insert(0, {"role": "user", "content": system})
    merged: List[Dict[str, Any]] = []
    for m in rest:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = (merged[-1].get("content") or "") + "\n\n" + (m.get("content") or "")
        else:
            merged.append(m)
    return merged


@dataclass
class TemplateRenderer:
    template: str
    bos_token: str = ""
    eos_token: str = ""
    dialect: str = "none"
    _compiled: Any = None

    def __post_init__(self) -> None:
        self._compiled = _make_env().from_string(self.template)
        self.dialect = detect_dialect(self.template)
        if self.dialect == "forced" and not self._probe_forced():
            self.dialect = "none"

    # -- internals -------------------------------------------------------
    def _render_raw(self, messages: List[Dict[str, Any]], **kw: Any) -> str:
        return self._compiled.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kw,
        )

    def _probe_forced(self) -> bool:
        """True when the generation prompt itself ends inside an open <think>."""
        try:
            out = self._render_raw([{"role": "user", "content": "x"}])
        except Exception:
            return False
        return out.rstrip().endswith(("<think>", "<thinking>"))

    # -- public ----------------------------------------------------------
    def render(self, messages: List[Dict[str, Any]], no_think: bool) -> str:
        kw: Dict[str, Any] = {}
        if self.dialect == "qwen3":
            kw["enable_thinking"] = not no_think
        elif self.dialect == "granite":
            kw["thinking"] = not no_think
        elif self.dialect == "gptoss":
            kw["reasoning_effort"] = "low" if no_think else "medium"
        try:
            out = self._render_raw(messages, **kw)
        except ValueError:
            # Some templates (Gemma, Mistral) reject a system role or require
            # strict user/assistant alternation. Fold the system text into the
            # first user turn, merge same-role neighbours, and retry once.
            out = self._render_raw(_fold_for_strict_templates(messages), **kw)
        if self.dialect == "forced" and no_think:
            stripped = out.rstrip()
            for tag, close in (("<think>", "</think>"), ("<thinking>", "</thinking>")):
                if stripped.endswith(tag):
                    out = stripped + "\n\n" + close + "\n\n"
                    break
        return out

    @property
    def can_think(self) -> bool:
        return self.dialect != "none"


# ── per-model cache ─────────────────────────────────────────────────────
_CACHE: Dict[str, Optional[TemplateRenderer]] = {}
_LOCK = threading.Lock()


def _special_text(llm: Any, token_id: int) -> str:
    try:
        if token_id is None or int(token_id) < 0:
            return ""
        return llm.detokenize([int(token_id)], special=True).decode("utf-8", "replace")
    except Exception:
        return ""


def _from_llama(llm: Any) -> Optional[TemplateRenderer]:
    meta = getattr(llm, "metadata", None) or {}
    tmpl = meta.get("tokenizer.chat_template") or ""
    if not isinstance(tmpl, str) or not tmpl.strip():
        return None
    bos = _special_text(llm, getattr(llm, "token_bos", lambda: -1)())
    eos = _special_text(llm, getattr(llm, "token_eos", lambda: -1)())
    return TemplateRenderer(template=tmpl, bos_token=bos, eos_token=eos)


def register(model_id: str, tokenize_callable: Any) -> None:
    """Build and cache the renderer for ``model_id`` (idempotent, never raises).

    ``tokenize_callable`` is the bound ``Llama.tokenize`` the server already
    caches; its ``__self__`` is the Llama instance that carries GGUF metadata.
    """
    if not model_id or model_id in _CACHE or tokenize_callable is None:
        return
    with _LOCK:
        if model_id in _CACHE:
            return
        try:
            llm = getattr(tokenize_callable, "__self__", None)
            renderer = _from_llama(llm) if llm is not None else None
            _CACHE[model_id] = renderer
            if renderer is None:
                log.info("chat_template: %s has no embedded template — legacy formatter stays active", model_id)
            else:
                log.info("chat_template: %s uses its embedded template (dialect=%s, think-switch=%s)",
                         model_id, renderer.dialect, renderer.can_think)
        except Exception as exc:                       # never break a request
            _CACHE[model_id] = None
            log.warning("chat_template: could not build template for %s: %s", model_id, exc)


def get_cached(model_id: str) -> Optional[TemplateRenderer]:
    return _CACHE.get(model_id)
