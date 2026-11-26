"""
Crash-loop guard + native llama.cpp log capture.

WHY THIS EXISTS
---------------
llama.cpp reports some allocation failures with ``ggml_abort()`` -> ``abort()``.
That is a SIGABRT raised from C code: no Python ``except`` can catch it, the
process dies, systemd restarts it, and the same code path aborts again
(observed: 7 identical crashes ~85 s apart in crash.log).

The guard turns an uncatchable abort into a *remembered* one:

    arm(fp)      -> write a sentinel file BEFORE the risky C call
    disarm()     -> remove it AFTER the call returned
    previous_start_crashed(requested)
                 -> on the next start: if the sentinel is still there, the last
                    process died inside the C call.  The sentinel is promoted to
                    a durable ``.crashed`` record and the caller must NOT repeat
                    a request that is at least as large as the one that crashed.

The refusal is monotonic: a request that is strictly smaller than the crashed
one (lower n_seq_max, or smaller n_ctx) is allowed again.  A different model or
a different kv_unified setting is allowed.  ``LLM_IGNORE_CTX_CRASH=1`` clears
the record (use after upgrading llama.cpp / changing hardware).

This module has no heavy imports and never raises into the caller.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("local_llm.ctx_guard")

_SENTINEL_NAME = "ctx_rebuild.sentinel"
_CRASHED_NAME = "ctx_rebuild.crashed"


# ── paths ────────────────────────────────────────────────────────────────
def state_dir() -> Path:
    """Same resolution order as inference_engine._default_state_dir()."""
    try:
        from local_llm_core.engine.inference.inference_engine import (
            _default_state_dir,
        )
        return Path(_default_state_dir())
    except Exception:
        pass
    explicit = os.environ.get("LLM_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    var_lib = Path("/var/lib/local-llm")
    try:
        var_lib.mkdir(parents=True, exist_ok=True)
        if os.access(var_lib, os.W_OK | os.R_OK):
            return var_lib
    except OSError:
        pass
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "local-llm"


def sentinel_path() -> Path:
    return state_dir() / _SENTINEL_NAME


def crashed_path() -> Path:
    return state_dir() / _CRASHED_NAME


# ── fingerprint ──────────────────────────────────────────────────────────
def fingerprint(model_path: Any, n_ctx: int, n_seq_max: int,
                kv_unified: bool) -> Dict[str, Any]:
    """Stable description of one context-build request."""
    return {
        "model": str(model_path or ""),
        "n_ctx": int(n_ctx),
        "n_seq_max": int(n_seq_max),
        "kv_unified": bool(kv_unified),
    }


# ── arm / disarm ─────────────────────────────────────────────────────────
def arm(fp: Dict[str, Any]) -> bool:
    """Durably record 'about to enter the risky C call'. Returns success."""
    try:
        p = sentinel_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        payload = json.dumps({**fp, "ts": time.time(), "pid": os.getpid()})
        with open(tmp, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return True
    except Exception as exc:  # never break the caller
        log.warning("ctx_guard.arm failed: %s", exc)
        return False


def disarm() -> None:
    try:
        sentinel_path().unlink(missing_ok=True)
    except Exception as exc:
        log.warning("ctx_guard.disarm failed: %s", exc)


# ── next-start decision ──────────────────────────────────────────────────
def _covers(rec: Dict[str, Any], req: Dict[str, Any]) -> bool:
    """True when ``req`` is at least as demanding as the crashed ``rec``.

    Missing / corrupt fields are treated conservatively (refuse)."""
    if not rec or "model" not in rec:
        return True          # empty / truncated record: assume the worst
    try:
        if str(rec.get("model", "")) != str(req.get("model", "")):
            return False
        if bool(rec.get("kv_unified", True)) != bool(req.get("kv_unified", True)):
            return False
        if int(req.get("n_seq_max", 1)) < int(rec.get("n_seq_max", 1)):
            return False
        if int(req.get("n_ctx", 0)) < int(rec.get("n_ctx", 0)):
            return False
        return True
    except Exception:
        return True


def previous_start_crashed(requested: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the crash record when ``requested`` must be refused, else None.

    Side effects: promotes a live sentinel to the durable ``.crashed`` file."""
    sp, cp = sentinel_path(), crashed_path()
    if os.environ.get("LLM_IGNORE_CTX_CRASH", "").strip() == "1":
        for p in (sp, cp):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return None
    rec: Optional[Dict[str, Any]] = None
    try:
        if sp.exists():
            try:
                rec = json.loads(sp.read_text() or "{}")
            except Exception:
                rec = {}
            if not isinstance(rec, dict):
                rec = {}
            rec["crashed_detected_at"] = time.time()
            rec["detected_by_pid"] = os.getpid()
            cp.write_text(json.dumps(rec))
            sp.unlink(missing_ok=True)
        elif cp.exists():
            try:
                rec = json.loads(cp.read_text() or "{}")
            except Exception:
                rec = {}
            if not isinstance(rec, dict):
                rec = {}
    except Exception as exc:
        log.warning("ctx_guard.previous_start_crashed: %s", exc)
        return None
    if rec is None:
        return None
    return rec if _covers(rec, requested) else None


# ── native llama.cpp log capture ─────────────────────────────────────────
_native_cb_ref: Any = None          # keep-alive: a GC'd callback segfaults
_native_fh: Any = None


def install_native_log(max_bytes: int = 2_000_000) -> Optional[Path]:
    """Mirror llama.cpp / ggml log lines (INFO, WARN, ERROR, CONT) into
    ``<state_dir>/llama_native.log``, flushed per line, so the lines that
    precede an ``abort()`` survive the crash.

    Note: ``ggml_abort`` itself writes its assert text straight to stderr (not
    through this callback) -> that line is in ``journalctl -u local-llm-cloud``.
    Idempotent.  Returns the log path or None."""
    global _native_cb_ref, _native_fh
    if _native_cb_ref is not None:
        return None
    try:
        import llama_cpp
        path = state_dir() / "llama_native.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.exists() and path.stat().st_size > max_bytes:
                path.replace(path.with_suffix(".log.1"))
        except OSError:
            pass
        _native_fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        fh = _native_fh

        @llama_cpp.llama_log_callback
        def _cb(level, text, user_data):  # pragma: no cover - C callback
            try:
                # ggml_log_level: 0 NONE, 1 INFO, 2 WARN, 3 ERROR, 4 DEBUG, 5 CONT
                if level in (1, 2, 3, 5):
                    fh.write(text.decode("utf-8", "replace"))
            except Exception:
                pass

        llama_cpp.llama_log_set(_cb, ctypes.c_void_p(0))
        _native_cb_ref = _cb
        return path
    except Exception as exc:
        log.warning("install_native_log failed: %s", exc)
        return None
    