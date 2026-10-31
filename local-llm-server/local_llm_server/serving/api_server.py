"""
api_server.py  v1.0.0 — Production Server
=================================================================
OpenAI-compatible LLM inference API. GGUF + BitNet. FastAPI + uvicorn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
import hashlib
import hmac
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

import psutil
import yaml
from fastapi import (
    BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request,
    Response, WebSocket, WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from local_llm_server.serving.log_handler import (
    TUILogHandler,
    create_formatter,
)
from local_llm_server.serving.middleware import (
    IPRateLimiter,
    ProductionMiddleware,
    create_ip_rate_limiter,
    install_request_id_filter,
    validate_trusted_proxies,
)
from local_llm_server.serving.sd_notify import sd_notify
from local_llm_server.serving.session_signing import (
    _session_secret,
    is_secure_request,
    sign_session,
    verify_session,
)
from local_llm_server.memory.conversation_store import create_conversation_store
from local_llm_server.auth.key_store import KeyStore, create_key_store, KeyAlreadyExistsError

from local_llm_core.engine.hardware.hardware_profiles import HardwareProfile
from local_llm_core.hal.suspend_inhibitor import SuspendInhibitor
from local_llm_core.engine.inference.stream_guards import (
    resolve_max_tokens, strip_think_text,
)

log = logging.getLogger(__name__)
VERSION = "1.0.0"

def _default_state_dir() -> Path:
    """State directory for pid, crash log, swap, kv-cache, and similar.

    Priority:
      1. $LLM_STATE_DIR (explicit override)
      2. /var/lib/local-llm (systemd StateDirectory=local-llm)
      3. $XDG_STATE_HOME/local-llm, defaulting to ~/.local/state/local-llm

    Creates the dir with mode 0700 if it doesn't exist. (L-06)
    """
    explicit = os.environ.get("LLM_STATE_DIR", "").strip()
    if explicit:
        p = Path(explicit)
    else:
        var_lib = Path("/var/lib/local-llm")
        chosen: Optional[Path] = None
        try:
            var_lib.mkdir(parents=True, exist_ok=True)
            if os.access(var_lib, os.W_OK | os.R_OK):
                chosen = var_lib
        except (OSError, PermissionError):
            chosen = None
        if chosen is None:
            xdg = Path(
                os.environ.get("XDG_STATE_HOME")
                or (Path.home() / ".local" / "state")
            )
            chosen = xdg / "local-llm"
        p = chosen
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p

def _resolve_ram_target(cfg: dict) -> float:
    """Return the canonical RAM utilisation budget as a fraction of total RAM.

    three different RAM budgets used to coexist:

      * ``memory.ram_utilization_target: 0.75`` — RAMOptimizer's cap
      * ``memory.ram_target: 0.85``             — autoload's usable_mb
      * ``HardwareProfile.ram_target``          — model_loader's "free RAM"

    The first two disagreed, so autoloading believed it had 10 percentage
    points more RAM than the runtime would actually grant — models could
    pass the autoload guard and then starve the KV cache. This helper
    makes ``memory.ram_utilization_target`` the single source of truth
    and reads it from every call site.

    Precedence:
      1. ``memory.ram_utilization_target`` (the canonical key)
      2. ``memory.ram_target`` — legacy, deprecated, warns on use
      3. ``0.75`` — historical default

    The per-profile ``HardwareProfile.ram_target`` is deliberately NOT
    consulted: that value describes what a given hardware tier typically
    has after OS overhead, not how much RAM this specific deployment is
    allowed to allocate. The two roles are different.
    """
    mem = (cfg or {}).get("memory", {}) or {}
    if "ram_utilization_target" in mem:
        try:
            return float(mem["ram_utilization_target"])
        except (TypeError, ValueError):
            log.warning(
                "memory.ram_utilization_target=%r is not numeric — "
                "falling back to legacy ram_target/default",
                mem["ram_utilization_target"],
            )
    legacy = mem.get("ram_target")
    if legacy is not None:
        log.warning(
            "memory.ram_target=%r is DEPRECATED — use "
            "memory.ram_utilization_target instead. Honouring for now; "
            "remove memory.ram_target from config.yaml to silence.",
            legacy,
        )
        try:
            return float(legacy)
        except (TypeError, ValueError):
            pass
    return 0.75

# ── Checkpoint retry state ──────────────────────────────────────────────────
_CHECKPOINT_FAILURES: Dict[str, int] = {}
_CHECKPOINT_DISABLED: Set[str] = set()
_CHECKPOINT_LOCK = asyncio.Lock()
MAX_CHECKPOINT_FAILURES = 3


# ── Centralised error response helper ────────────────────────────────────────
def _json_error(
    status_code: int,
    message: str,
    *,
    request: Optional["Request"] = None,
    detail: Any = None,
    headers: Optional[Dict[str, str]] = None,
    log_exc: Optional[BaseException] = None,
) -> JSONResponse:
    if log_exc is not None:
        log.error("HTTP %d — %s", status_code, message, exc_info=log_exc)

    body: Dict[str, Any] = {
        "error": {
            "status":  status_code,
            "message": message,
        }
    }
    if detail is not None and status_code < 500:
        body["error"]["detail"] = detail
    if request is not None:
        req_id = getattr(request.state, "request_id", None) or \
                 request.headers.get("X-Request-ID")
        if req_id:
            body["request_id"] = req_id

    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=headers,
    )


def _check_thermal_throttle(req: Request) -> None:
    thermal = req.app.state.ctx.get("thermal_state")
    if thermal and thermal.throttling:
        log.warning(
            "Request rejected due to thermal throttling (%.1f°C) — Retry-After 5s",
            thermal.temperature_c
        )
        raise HTTPException(
            503,
            "GPU thermal throttling active – please wait",
            headers={"Retry-After": "5"}
        )

def _apply_key_quotas(req: Request, max_tokens: int, timeout_s: float) -> Tuple[int, float]:
    """Clamp request-level max_tokens / timeout_s against the caller's key quotas.

    The key record is stashed by verify_api_key on
    ``request.state.key_quotas``. Both fields are capped at zero meaning
    "no per-key cap" — the server-wide inference.max_total_tokens and
    inference.default_timeout_s still apply on top.

    Returns the (possibly clamped) pair. A client that requested a
    larger value than their key allows will silently see the smaller
    value take effect — the model stops at the cap. A client that
    requested 0 (unlimited) gets the cap if one is configured.
    """
    quotas = getattr(req.state, "key_quotas", None) or {}
    max_out = int(quotas.get("max_output_tokens") or 0)
    max_secs = float(quotas.get("max_request_seconds") or 0.0)
    if max_out > 0 and (max_tokens <= 0 or max_tokens > max_out):
        max_tokens = max_out
    if max_secs > 0 and (timeout_s <= 0 or timeout_s > max_secs):
        timeout_s = max_secs
    return max_tokens, timeout_s

def _check_server_capacity(req: Request, *, max_gate_waiters: int = 32) -> None:
    """Return HTTP 503 with Retry-After if the server is at capacity.

    Called at the top of every chat/completion handler BEFORE any
    expensive work (model existence check, prompt build, engine dispatch)
    so an overloaded server fails fast instead of queueing forever.

    Three signals checked, in order of severity:

      1. Circuit breaker OPEN — the engine will refuse any new request;
          return 503 with Retry-After: 60 so clients don't hammer.
      2. In-flight count at queue max — the engine's own guard would
          return finish_reason="overloaded" at 200; converting that to a
          proper 503 here saves a round-trip.
      3. Gate waiters above threshold — even if the queue has room, N
          waiters already means a long delay; reject early with a hint.

    All introspection is defensive: if any check raises (engine in a
    transient init state, attribute missing), the check is skipped rather
    than failing the request.
    """
    engine = req.app.state.ctx.get("engine")
    if engine is None:
        return

    # bump the capacity-rejection counter so /metrics and the
    # alert rules (prometheus-alerts.yml) can see the reason. We track
    # per-reason counters in a process-local dict on ctx, exported by the
    # /metrics endpoint below.
    _counters = req.app.state.ctx.setdefault("_capacity_rejects", {
        "circuit_open": 0,
        "inflight": 0,
        "gate_waiters": 0,
    })

    def _reject(reason: str, status: int, detail: str, retry: str) -> None:
        _counters[reason] = _counters.get(reason, 0) + 1
        raise HTTPException(status, detail, headers={"Retry-After": retry})

    # 1. Circuit breaker
    try:
        breaker = getattr(engine, "_breaker", None)
        if breaker is not None:
            state = breaker.state_name() if hasattr(breaker, "state_name") else "closed"
            if state == "open":
                _reject(
                    "circuit_open", 503,
                    "Circuit breaker open — inference temporarily unavailable",
                    "60",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    # 2. In-flight
    try:
        inflight = int(getattr(engine, "_inflight", 0))
        max_inflight = int(engine._queue._max)
        if inflight >= max_inflight:
            _reject(
                "inflight", 503,
                f"Server at capacity ({inflight}/{max_inflight} requests in flight)",
                "30",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # 3. Gate waiters
    try:
        gate_waiters = int(getattr(engine, "_gate_waiters", 0))
        if gate_waiters >= max_gate_waiters:
            _reject(
                "gate_waiters", 503,
                f"Inference queue saturated ({gate_waiters} waiting)",
                "60",
            )
    except HTTPException:
        raise
    except Exception:
        pass

# ── Heartbeat control frames (prefill/slow-decode keep-alive) ────────────
# Frame wire format: \x1f STATUS:{json} \x1e
# Those two bytes are illegal in any UTF-8 model output, so old clients
# that don't understand them render nothing; new clients strip them via
# the FrameFilter class before feeding the visible text through md().
import json as _json_hb
from typing import Callable as _HbCallable, Tuple as _HbTuple

HB_OPEN, HB_CLOSE = "\x1f", "\x1e"


def make_status_frame(phase: str, **fields) -> bytes:
    fields["phase"] = phase
    return (
        f"{HB_OPEN}STATUS:"
        f"{_json_hb.dumps(fields, separators=(',', ':'))}"
        f"{HB_CLOSE}"
    ).encode()


async def with_heartbeat(
    source: AsyncIterator,
    *,
    interval_s: float,
    status_fn: _HbCallable[[], dict],
) -> AsyncIterator[_HbTuple[str, object]]:
    """Yield ("data", item) for every source item, and ("hb", bytes) every
    interval_s seconds while the source is silent.  Uses asyncio.wait (not
    wait_for) so the pending __anext__ is never cancelled."""
    it = source.__aiter__()
    nxt: Optional[asyncio.Future] = None
    try:
        while True:
            if nxt is None:
                nxt = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({nxt}, timeout=interval_s)
            if not done:
                st = status_fn() or {}
                yield "hb", make_status_frame(st.pop("phase", "working"), **st)
                continue
            try:
                item = nxt.result()
            except StopAsyncIteration:
                return
            finally:
                nxt = None
            yield "data", item
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()
            try:
                await nxt
            except BaseException:
                pass
        aclose = getattr(it, "aclose", None)
        if aclose:
            try:
                await aclose()
            except BaseException:
                pass

async def with_sse_heartbeat(
    source: AsyncIterator[bytes],
    *,
    interval_s: float,
    status_fn: _HbCallable[[], dict],
) -> AsyncIterator[bytes]:
    """Wrap an SSE byte-stream and inject SSE comment heartbeats during silence.

    Comment lines (starting with ':' per the SSE spec) are ignored by every
    compliant data parser, so old clients see no new content — but bytes
    arriving on the wire reset any client-side idle timer. The comment body
    carries the same status JSON as make_status_frame() so a future client
    can opt in to render progress for these routes without a wire change.

    Unlike with_heartbeat (which yields (kind, bytes) tuples for plain-text
    routes), this yields raw bytes and is designed to wrap a generator that
    ALREADY produces SSE frames — heartbeats are interleaved between them."""
    it = source.__aiter__()
    nxt: Optional[asyncio.Future] = None
    try:
        while True:
            if nxt is None:
                nxt = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({nxt}, timeout=interval_s)
            if not done:
                st = status_fn() or {}
                phase = st.pop("phase", "working")
                body = _json_hb.dumps({"phase": phase, **st}, separators=(",", ":"))
                # RFC-compliant SSE comment: any line beginning with ':' is
                # ignored by data parsers; raw bytes still reset idle timers.
                yield f": hb {body}\n\n".encode()
                continue
            try:
                item = nxt.result()
            except StopAsyncIteration:
                return
            finally:
                nxt = None
            yield item
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()
            try:
                await nxt
            except BaseException:
                pass
        aclose = getattr(it, "aclose", None)
        if aclose:
            try:
                await aclose()
            except BaseException:
                pass

def _hb_status_fn(engine, ir, t0):
    """Build the callable with_heartbeat() polls for the frame payload.

    `ir` may be:
      * an InferenceRequest (uses .root_id or .request_id),
      * any object with a .request_id attribute (e.g. GenerationCheckpoint),
      * a plain str key already known to the engine's progress map.
    """
    if isinstance(ir, str):
        root_id = ir
    else:
        root_id = getattr(ir, "root_id", "") or getattr(ir, "request_id", "")

    def _fn():
        p: dict = {}
        try:
            if root_id:
                p = engine.get_progress(root_id) or {}
        except Exception:
            pass
        return {
            "phase":  p.get("phase", "queued" if not p else "working"),
            "done":   int(p.get("done", 0)),
            "total":  int(p.get("total", 0)),
            "reused": int(p.get("reused", 0)),
            "elapsed": round(time.monotonic() - t0, 1),
        }
    return _fn

# ── Monitored stream wrapper for mid‑generation health checks ──────────────
async def _monitored_stream(
    source: AsyncIterator[str],
    req: Request,
    ir: Any,
    profile: Optional[HardwareProfile],
    *,
    check_interval_tokens: int = 5,
    stall_timeout: float = 10.0,
) -> AsyncIterator[str]:
    from local_llm_core.engine.inference.inference_engine import StreamStallError
    last_token_time = time.monotonic()
    token_count = 0
    try:
        async for token in source:
            token_count += 1
            now = time.monotonic()
            if token_count % check_interval_tokens == 0:
                thermal = req.app.state.ctx.get("thermal_state")
                if thermal and thermal.throttling:
                    raise StreamStallError(
                        phase="thermal",
                        elapsed_s=0.0,
                        budget_s=stall_timeout,
                        partial_tokens=token_count,
                    )
                elapsed = now - last_token_time
                if elapsed > stall_timeout:
                    raise StreamStallError(
                        phase="monitored",
                        elapsed_s=elapsed,
                        budget_s=stall_timeout,
                        partial_tokens=token_count,
                    )
            last_token_time = now
            yield token
    except asyncio.CancelledError:
        raise
    finally:
        if hasattr(source, "aclose"):
            try:
                await source.aclose()
            except (GeneratorExit, asyncio.CancelledError):
                pass
            except Exception as e:
                # Standing rule: no log.debug — warnings only.
                log.warning("Error closing monitored source: %s", e)


def _wrap_stream(
    source_factory,
    req: Request,
    ir: Any,
    profile: Optional[HardwareProfile],
    max_tokens: int,
) -> AsyncIterator[str]:
    # Prefer measured TPS over target — target is an aspiration, measured
    # is what the hardware actually delivers. read from
    # token_source (pool in pool mode) not just engine (empty in pool mode).
    measured_tps = 0.0
    try:
        _ts = (
            req.app.state.ctx.get("token_source")
            or req.app.state.ctx.get("engine")
        )
        if _ts is not None:
            s = _ts.stats()
            measured_tps = float(s.get("avg_tps", 0.0) or 0.0)
    except Exception:
        pass
    if measured_tps > 0.1:
        effective_tps = measured_tps
    else:
        effective_tps = (profile.target_tps if profile else 10.0)

    inter_token_latency = 1.0 / max(effective_tps, 0.1)
    stall_timeout = max(15.0, inter_token_latency * 8.0)
    check_interval_tokens = max(4, int(effective_tps * 0.5))

    source = source_factory()
    return _monitored_stream(
        source, req, ir, profile,
        check_interval_tokens=check_interval_tokens,
        stall_timeout=stall_timeout,
    )


class _SafeStream:
    def __init__(
        self,
        source: AsyncIterator[str],
        request_id: str,
        error_marker: str = "[Error: Model generated no output — please retry]",
    ):
        self._source = source
        self._request_id = request_id
        self._error_marker = error_marker
        self._emitted = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if hasattr(self._source, "aclose"):
                await self._source.aclose()
        except (GeneratorExit, asyncio.CancelledError):
            pass
        except Exception as e:
            log.warning("[%s] Error closing stream source: %s",
                        self._request_id[:8], e)
        return False

    def __aiter__(self):
        return self._safe_iterator()

    async def _safe_iterator(self) -> AsyncIterator[str]:
        try:
            from local_llm_core.engine.inference.inference_engine import (
                StreamStallError, _make_stall_marker,
            )
        except Exception:
            StreamStallError = None  # type: ignore[assignment]
            _make_stall_marker = None  # type: ignore[assignment]

        try:
            async for chunk in validate_stream_response(
                self._source,
                request_id=self._request_id,
                min_chunks=1,
            ):
                yield chunk
        except RuntimeError as e:
            log.error("[%s] Stream empty: %s", self._request_id[:8], e)
            if not self._emitted:
                self._emitted = True
                yield self._error_marker
        except Exception as e:
            # If the wrapper (_monitored_stream) raised a stall, surface
            # the canonical marker so the UI shows a Continue button
            # rather than an abrupt close.
            if (StreamStallError is not None
                    and isinstance(e, StreamStallError)
                    and _make_stall_marker is not None):
                log.warning(
                    "[%s] Stream stall surfaced: %s",
                    self._request_id[:8], e,
                )
                if not self._emitted:
                    self._emitted = True
                    yield _make_stall_marker(e.phase, e.partial_tokens)
                return
            raise


try:
    from local_llm_core.compatibility.transformers_compat import StateFileReader as _SFR
    _SFR.apply_spectre_pythonpath()
    del _SFR
except Exception:
    pass

_api_key_header    = APIKeyHeader(name="X-API-Key",    auto_error=False)
_master_key_header = APIKeyHeader(name="X-Master-Key", auto_error=False)


# =============================================================================
# Environment variable expansion
# =============================================================================
def _expand_env_vars(value: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return value
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(2)
            return os.environ.get(var, default if default is not None else "")
        return re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', _repl, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(i, _depth + 1) for i in value]
    return value


# =============================================================================
# Request size middleware
# =============================================================================
class RequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024) -> None:
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > self.max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds {self.max_size // 1_000_000} MB"},
            )
        return await call_next(request)


# =============================================================================
# Timeout decorator
# =============================================================================
def with_timeout(seconds: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(503, detail=f"Timeout after {seconds}s")
        return wrapper
    return decorator


# =============================================================================
# Pydantic models
# =============================================================================
class ChatMessage(BaseModel):
    """
    One message in a chat conversation.

    Field semantics follow the OpenAI chat-completions spec so any
    OpenAI-compatible client (AMB-APP's LLMCloudClient, the Web UI, a
    third-party SDK) can send the full message shape without getting a
    422.

    The server itself is JSON-in-text only — it does NOT dispatch tools.
    `tool_calls` / `tool_call_id` / `name` are accepted (and folded into
    the prompt as text; see `_build_prompt`) purely so clients that
    prefix history with tool turns round-trip cleanly.
    """
    role:          str
    content:       str = ""
    tool_calls:    Optional[List[Dict[str, Any]]] = None
    tool_call_id:  Optional[str] = None
    name:          Optional[str] = None

    # Allow any extra fields (provider-specific extensions, `function_call`,
    # `refusal`, …). The prompt builder only looks at the fields it knows
    # about, so unknown keys are harmless.
    model_config = {"extra": "allow"}


class ChatRequest(BaseModel):
    model:                   str
    messages:                List[ChatMessage]
    max_tokens:              int   = Field(0,  ge=0, le=2147483647)
    temperature:             float = Field(0.7,   ge=0.0, le=2.0)
    top_p:                   float = Field(0.95,  ge=0.0, le=1.0)
    top_k:                   int   = Field(40,    ge=1)
    stream:                  bool  = False
    stop:                    Optional[List[str]] = None
    no_think:                Optional[bool] = None
    stream_format:           str   = Field("json", pattern="^(json|text)$")
    thinking:                bool  = False
    thinking_budget_tokens:  int   = Field(0, ge=0)
    timeout_s:               float = Field(0, ge=0)
    history_mode:            str   = Field("auto", pattern="^(auto|append|replace)$")
    tools:                   Optional[List[Dict[str, Any]]] = None  # accepted, ignored
    # ── Constrained-decoding hints (optional) ─────────────────────────────
    # `grammar` is a raw GBNF grammar string (llama.cpp native format).
    # `response_format` follows the OpenAI spec:
    #   {"type": "json_object"}                       → valid JSON, any shape
    #   {"type": "json_schema", "json_schema": {...}} → schema-constrained
    #   {"type": "text"}                              → unconstrained
    # Both are best-effort: an unsupported backend or a malformed grammar
    # logs a warning and generates unconstrained rather than failing.
    grammar:         Optional[str] = Field(None, max_length=65536)
    response_format: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_grammar(self) -> "ChatRequest":
        rf = self.response_format
        if rf is not None:
            rtype = (rf.get("type") or "").lower()
            if rtype not in ("json_object", "json_schema", "text"):
                raise ValueError(
                    "response_format.type must be one of: "
                    "json_object, json_schema, text"
                )
            if rtype == "json_schema":
                js = rf.get("json_schema")
                if not isinstance(js, dict) or not isinstance(js.get("schema"), dict):
                    raise ValueError(
                        "response_format.json_schema must be an object "
                        "containing a 'schema' object"
                    )
        return self

class SmartChatRequest(ChatRequest):
    """Auto-Router: the server picks the model, so `model` is optional."""
    model: Optional[str] = None


class CompletionRequest(BaseModel):
    model:       str
    prompt:      str
    # 0 = unlimited logical generation; positive values are exact ceilings.
    max_tokens:  int   = Field(0, ge=0, le=2147483647)
    temperature: float = Field(0.7,  ge=0.0, le=2.0)
    stop:        Optional[List[str]] = None
    stream:      bool = False


class LoadModelRequest(BaseModel):
    model_id:         str
    model_path:       str
    quantization:     str           = "auto"
    context_length:   int           = Field(0,  ge=0, le=131072)
    num_gpu_layers:   int           = Field(-1, ge=-1)
    n_threads:        int           = Field(0,  ge=0)
    description:      str           = ""
    model_type:       str           = Field("auto",  description="'gguf', 'bitnet', or 'auto'")
    ram_required_mb:  Optional[int] = Field(None, ge=0)


class CreateKeyRequest(BaseModel):
    key_id:                str
    name:                  Optional[str]   = None
    rate_limit_rps:        float           = Field(10.0,  ge=0.1, le=1000.0)
    rate_limit_capacity:   float           = Field(20.0,  ge=1.0, le=2000.0)
    description:           str             = ""
    expires_in_seconds:    Optional[float] = None
    active:                bool            = True
    # ── Per-key quotas ──────────────────────────────────────────
    # 0 = no per-key cap for that dimension. Server-wide caps in the
    # [inference] section still apply on top.
    max_output_tokens:     int   = Field(0, ge=0, le=2147483647,
        description="Cap on completion tokens per request. 0 = unlimited.")
    max_request_seconds:   float = Field(0.0, ge=0.0, le=86400.0,
        description="Wall-clock cap per request, in seconds. 0 = unlimited.")
    max_requests_per_day:  int   = Field(0, ge=0, le=10000000,
        description="Rolling daily request cap. 0 = unlimited.")

    @model_validator(mode="after")
    def _set_desc(self) -> "CreateKeyRequest":
        if self.name and not self.description:
            self.description = self.name
        return self


class UpdateKeyRequest(BaseModel):
    rate_limit_rps:       Optional[float] = None
    rate_limit_capacity:  Optional[float] = None
    description:          Optional[str]   = None
    active:               Optional[bool]  = None
    # ── Per-key quotas ──────────────────────────────────────────
    # None = leave unchanged. 0 = clear the cap (unlimited).
    max_output_tokens:    Optional[int]   = Field(None, ge=0, le=2147483647)
    max_request_seconds:  Optional[float] = Field(None, ge=0.0, le=86400.0)
    max_requests_per_day: Optional[int]   = Field(None, ge=0, le=10000000)


class ResumeRequest(BaseModel):
    """
    Request body for POST /v1/chat/resume.

    The client (Web UI) sends `conv_id`, `max_tokens`, `stream`,
    `temperature`, `model`, `no_think`, and `history_mode`. All of the
    additional fields below are accepted so Pydantic does not silently
    drop them; the ones that do not affect resume semantics
    (`no_think`, `history_mode`, `timeout_s`) are documented here but
    intentionally ignored by the handler:

      * `no_think` — the resume prompt is already templated; thinking
        state is baked into `original_prompt` and cannot be flipped
        per-resume without re-templating the whole conversation.
      * `history_mode` — resume does not merge history; the
        accumulated text in the checkpoint IS the history.
      * `timeout_s` — the engine's own stall detection
        (AsyncTokenBuffer first-token + inter-token budgets) is
        authoritative; an extra client-supplied total deadline would
        just race with it.
      * `model` — accepted for client-compat but ignored: the model
        is taken from the checkpoint, because the partial answer was
        generated by that model.

    `top_p` / `top_k` ARE forwarded to the engine so the user's
    sampler settings actually take effect on resume (previously the
    engine hard-coded 0.95/40).
    """
    conv_id:          str
    checkpoint_id:    Optional[str] = None
    # 0 = unlimited logical generation (the default). Positive = exact cap.
    max_tokens: int = Field(0, ge=0, le=2147483647)
    temperature:      Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p:            Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k:            Optional[int]   = Field(None, ge=1)
    stream:           bool  = True
    # Client-compat fields (accepted, see docstring above for whether used):
    model:            Optional[str] = None
    no_think:         Optional[bool] = None
    history_mode:     str = Field("auto", pattern="^(auto|append|replace)$")
    timeout_s:        float = Field(0, ge=0)


# =============================================================================
# Prompt formatters
# =============================================================================
@dataclass
class PromptFormatter:
    name:              str
    system_prefix:     str
    user_prefix:       str
    assistant_prefix:  str
    suffix:            str
    supports_thinking: bool = True


FORMATTERS: Dict[str, PromptFormatter] = {
    "qwen": PromptFormatter(
        "qwen",
        "<|im_start|>system\n",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>\n",
        supports_thinking=True,
    ),
    "llama3": PromptFormatter(
        "llama3",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
        supports_thinking=False,
    ),
    "phi3": PromptFormatter(
        "phi3",
        "<|system|>\n", "<|user|>\n", "<|assistant|>\n", "<|end|>\n",
        supports_thinking=False,
    ),
    "bitnet": PromptFormatter(
        "bitnet",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
        supports_thinking=False,
    ),
    "default": PromptFormatter(
        "default",
        "### System:\n", "### User:\n", "### Assistant:\n", "\n",
        supports_thinking=False,
    ),
}


def _auto_detect_format(model_id: str) -> Optional[str]:
    if not model_id:
        return None
    m = model_id.lower()
    if "qwen"   in m: return "qwen"
    if "llama"  in m or "mistral" in m: return "llama3"
    if "bitnet" in m or "b1.58"   in m: return "bitnet"
    return None


class PromptTooLargeError(Exception):
    """Raised when even the trimmed prompt exceeds the model's context."""


async def _count_prompt_tokens(engine, model_id: str, prompt: str) -> int:
    """Real prompt token count using the loaded model's tokenizer.

    Falls back to a conservative char/2.5 estimate if the tokenizer is
    unavailable — over-counting causes safe trimming, under-counting causes
    silent context overflow.
    """
    try:
        with engine._models_lock:
            model = engine._unified_models.get(model_id)
        if model is None and hasattr(engine.loader, "get"):
            model = engine.loader.get(model_id)
        llm = getattr(model, "llm", model)
        tokenizer = getattr(llm, "tokenize", None)
        if callable(tokenizer):
            ids = tokenizer(prompt.encode("utf-8", "surrogatepass"))
            if isinstance(ids, (list, tuple)):
                return len(ids)
    except Exception as exc:
        log.warning("Real tokenizer unavailable for %s: %s", model_id, exc)
    return max(1, int(len(prompt) / 2.5))


async def _fit_prompt_to_context(
    engine, model_id: str,
    messages: List[dict],
    requested_max_tokens: int,
    inf_cfg: dict,
    profile,
    *,
    safety: int = 128,
    no_think_override: Optional[bool] = None,
) -> str:
    """Build a prompt that fits within the model's real context.

    Trimming policy: keep system prompt + newest turn. Drop oldest non-system
    turns until the prompt fits. Tail-keeping is correct for chat UX.
    """
    # Read the *loaded* model's real context window.
    n_ctx = 0
    try:
        n_ctx = int(engine._model_n_ctx(model_id))
    except Exception:
        pass
    n_ctx = n_ctx or int(getattr(profile, "context_length", 2048) or 2048)

    # Unlimited (<=0) reserves half the window for the reply; the engine
    # will slide the window if the reply grows past that. Positive values
    # are honoured as an exact per-call ceiling.
    if requested_max_tokens and requested_max_tokens > 0:
        response_budget = min(requested_max_tokens, n_ctx // 2)
    else:
        response_budget = n_ctx // 2

    budget = max(64, n_ctx - response_budget - safety)

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    def _build(candidate):
        return _build_prompt(
            system_msgs + candidate,
            profile,
            suppress_thinking=inf_cfg.get("suppress_thinking", False),
            no_think_override=no_think_override,
            model_id=model_id,
        )

    prompt = _build(non_system)
    tokens = await _count_prompt_tokens(engine, model_id, prompt)
    if tokens <= budget:
        return prompt

    kept = non_system[-2:] if len(non_system) >= 2 else non_system[:]
    dropped = 0
    while True:
        prompt = _build(kept)
        tokens = await _count_prompt_tokens(engine, model_id, prompt)
        if tokens <= budget or len(kept) <= 1:
            break
        kept = kept[1:]
        dropped += 1

    if dropped:
        log.warning(
            "Prompt for '%s' exceeded context (%d > %d tokens) — "
            "dropped %d oldest messages.",
            model_id, tokens, budget, dropped,
        )
    if tokens > budget:
        raise PromptTooLargeError(
            f"Prompt still exceeds context after trimming "
            f"({tokens} > {budget} tokens). Shorten the message."
        )
    return prompt


def _build_prompt(
    messages,
    hw_profile=None,
    suppress_thinking: bool = False,
    no_think_override: Optional[bool] = None,
    model_id: str = "",
) -> str:
    fmt_name = (
        getattr(hw_profile, "prompt_format", None)
        if hw_profile is not None else None
    ) or _auto_detect_format(model_id) or "default"
    fmt = FORMATTERS.get(fmt_name, FORMATTERS["default"])

    force_no_think = getattr(hw_profile, "force_no_think", False) if hw_profile else False
    no_think = suppress_thinking or force_no_think
    if no_think_override is not None:
        no_think = bool(no_think_override) or force_no_think

    cleaned = []
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role", "user"))
            content = str(m.get("content", "") or "")
            _tc = m.get("tool_calls")
        else:
            role = str(getattr(m, "role", "user"))
            content = str(getattr(m, "content", "") or "")
            _tc = getattr(m, "tool_calls", None)
        if "/no_think" in content:
            no_think = True
            content = content.replace("/no_think", "").strip()
        # Fold any tool_calls into the assistant's text representation.
        # The model sees the call as plain text — which is exactly what
        # it already does when it emits JSON-in-text actions of its own,
        # so the training distribution matches.
        if role == "assistant" and _tc:
            try:
                _tc_text = json.dumps(_tc, ensure_ascii=False, separators=(",", ":"))
                content = (content + "\n" if content else "") + _tc_text
            except Exception:
                pass
        if role == "assistant":
            content = _strip_stream_marker(strip_think_text(content))
        cleaned.append((role, content))

    parts = []
    for role, content in cleaned:
        if role == "system":
            parts.append(f"{fmt.system_prefix}{content}{fmt.suffix}")
        elif role == "user":
            parts.append(f"{fmt.user_prefix}{content}{fmt.suffix}")
        elif role == "assistant":
            parts.append(f"{fmt.assistant_prefix}{content}{fmt.suffix}")
        elif role == "tool":
            # Server has no native tool dispatch — tool output is context,
            # not a turn. Tag it as a user message so the model can reason
            # about it without confusing it with the client's own words.
            parts.append(f"{fmt.user_prefix}[tool result]\n{content}{fmt.suffix}")
        # Unknown roles are silently dropped.

    if fmt.supports_thinking and no_think:
        parts.append(f"{fmt.assistant_prefix}<think>\n\n</think>\n\n")
    else:
        parts.append(fmt.assistant_prefix)

    return "".join(parts)


# =============================================================================
# History merge (idempotent, history_mode aware)
# =============================================================================
def _msg_fingerprint(role: str, content: str) -> bytes:
    """Stable 16-byte fingerprint of a chat message.

    Collision risk with sha256[:16] is ~2^-64 over any realistic
    conversation length; the truncation matters because we compare
    fingerprints millions of times under production load.
    """
    h = hashlib.sha256()
    h.update(role.encode("utf-8", "surrogatepass"))
    h.update(b"\x00")
    h.update(content.encode("utf-8", "surrogatepass"))
    return h.digest()[:16]


def _to_message_dict(m) -> Dict[str, str]:
    if isinstance(m, dict):
        return {"role": str(m.get("role", "user")),
                "content": str(m.get("content", ""))}
    return {"role": str(getattr(m, "role", "user")),
            "content": str(getattr(m, "content", ""))}


def _longest_suffix_prefix_overlap(
    stored: List[Dict[str, str]],
    incoming: List[Dict[str, str]],
) -> int:
    """Length k of the longest suffix of `stored` that equals a prefix
    of `incoming` (message-by-message).

    Chat semantics guarantee messages are strictly appended, never
    edited or reordered — so this is the only possible overlap shape.
    O(min(|stored|, |incoming|)) with precomputed fingerprints.
    """
    n, m = len(stored), len(incoming)
    kmax = min(n, m)
    if kmax == 0:
        return 0
    sh = [_msg_fingerprint(x["role"], x["content"]) for x in stored[-kmax:]]
    ih = [_msg_fingerprint(x["role"], x["content"]) for x in incoming[:kmax]]
    k = 0
    for i in range(kmax):
        if sh[i] != ih[i]:
            return k
        k += 1
    return k


def _merge_history(
    stored: List[dict],
    incoming: List[ChatMessage],
    *,
    history_mode: str = "auto",
) -> List[dict]:
    """Idempotent history merge.

    Modes:
      "replace": incoming is authoritative.
      "append" : append incoming to stored.
      "auto"   : <=1 non-system incoming → append; else replace.
    """
    stored_msgs = [_to_message_dict(m) for m in (stored or [])]
    incoming_msgs = [_to_message_dict(m) for m in incoming]

    incoming_system = next((m for m in incoming_msgs if m["role"] == "system"), None)
    stored_system = next((m for m in stored_msgs if m["role"] == "system"), None)
    incoming_body = [m for m in incoming_msgs if m["role"] != "system"]
    stored_body = [m for m in stored_msgs if m["role"] != "system"]

    effective_mode = history_mode
    if effective_mode == "auto":
        effective_mode = "append" if len(incoming_body) <= 1 else "replace"

    if effective_mode == "append":
        merged = stored_body + incoming_body
    else:
        overlap = _longest_suffix_prefix_overlap(stored_body, incoming_body)
        if overlap == 0 and stored_body and incoming_body:
            log.warning(
                "History merge: stored=%d incoming=%d overlap=0 — "
                "preferring incoming (client authoritative)",
                len(stored_body), len(incoming_body),
            )
        merged = stored_body[:len(stored_body) - overlap] + incoming_body

    system = incoming_system or stored_system
    return ([system] if system else []) + merged


# =============================================================================
# Stream markers and error tokens
# =============================================================================
_ERROR_PATTERN = re.compile(
    r"^\[(request timed out|error|stream cancelled|cancelled|\w+ error|Error:[^\]]*)\]\s*$",
    re.IGNORECASE,
)

_STREAM_MARKER_RE = re.compile(
    r"\[⚡(TRUNCATED|INTERRUPTED|STALLED):[^\]]*?tokens=(\d+)[^\]]*\]"
)

_TOOLS_WARNED: bool = False

def _maybe_warn_about_tools(body: Any) -> None:
    """Log once per process that a client sent `tools` on a server that
    has no native function-calling. Rate-limit this to one warning so an
    aggressive client can't flood the log."""
    global _TOOLS_WARNED
    if _TOOLS_WARNED:
        return
    tools = getattr(body, "tools", None) or getattr(body, "functions", None)
    if tools:
        _TOOLS_WARNED = True
        log.warning(
            "Client sent `tools` (%d entries); this server has no native "
            "function-calling. The field is ignored — use the JSON-in-text "
            "action protocol instead.",
            len(tools),
        )

def _extract_stream_marker(text: str) -> Optional[tuple]:
    """Return (kind, tokens) if a canonical marker is present, else None."""
    m = _STREAM_MARKER_RE.search(text)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def _strip_stream_marker(text: str) -> str:
    return _STREAM_MARKER_RE.sub("", text).rstrip()


def _is_error_token(text: str) -> bool:
    return bool(_ERROR_PATTERN.match(text.strip()))


# =============================================================================
# Admin rate limiting & audit
# =============================================================================
class AdminRateLimiter:
    def __init__(self, max_requests_per_hour: int = 1000) -> None:
        self.rps      = max_requests_per_hour / 3600.0
        self.capacity = float(max_requests_per_hour)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock    = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now               = time.monotonic()
            tokens, last      = self._buckets.get(key, (self.capacity, now))
            tokens            = min(self.capacity, tokens + (now - last) * self.rps)
            self._buckets[key]= (tokens, now)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            return False


admin_rate_limiter: Optional[AdminRateLimiter] = None

_ADMIN_AUDIT_READONLY = frozenset({"list_keys", "key_stats"})


def _log_admin_action(
    master_key: str,
    action: str,
    target: str = "",
    success: bool = True,
    extra: Optional[dict] = None,
) -> None:
    if action in _ADMIN_AUDIT_READONLY:
        log.info("ADMIN: key=%s... action=%s target=%s", master_key[:8], action, target)
        return
    log.info(
        "ADMIN: key=%s... action=%s target=%s success=%s extra=%s",
        master_key[:8], action, target, success, extra,
    )


# =============================================================================
# Session / key verification
# =============================================================================
async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(_api_key_header),
    master_key: Optional[str] = Depends(_master_key_header),
) -> Tuple[bool, Optional[str], Optional[KeyStore]]:
    PUBLIC_PATHS = {
        "/health", "/ready",
        "/v1/models", "/v1/models/current",
        "/v1/runtime/limits",
    }
    if request.url.path in PUBLIC_PATHS:
        return True, "public", None

    if master_key:
        try:
            _cfg = getattr(request.app.state, "config", None) or {}
            _master_cfg = _cfg.get("api", {}).get("master_key")
            if _master_cfg and master_key and hmac.compare_digest(master_key, _master_cfg):
                return True, "master", None
        except Exception:
            pass

    try:
        store: Optional[KeyStore] = request.app.state.ctx.get("key_store")
    except Exception:
        store = None

    # ── Static-key fallback (fail-closed by default) ────────────
    # The static-key path is only reachable when key_store_enabled=false.
    # In a security-critical deployment that fallback is a liability: the
    # key is either committed to config (S1 finding) or, when empty, the
    # whole auth tier silently disappears (S3 finding — "fail-open"). Both
    # are removed here.
    #
    # security.allow_static_key_fallback: false (new default)
    #   When the key store is disabled AND this is false, requests are
    #   REFUSED with 503 rather than served with no auth or a static key.
    #   Operators who genuinely need static-key mode (single-tenant dev)
    #   set it to true AND set api.key explicitly.
    if store is None:
        try:
            cfg = request.app.state.config or {}
        except Exception:
            cfg = {}
        api_cfg = (cfg or {}).get("api", {}) or {}
        sec_cfg = (cfg or {}).get("security", {}) or {}
        static_key = (api_cfg.get("key") or "").strip()
        allow_static = bool(sec_cfg.get("allow_static_key_fallback", False))
        allow_noauth = bool(sec_cfg.get("allow_no_auth", False))

        if static_key:
            if not allow_static:
                log.error(
                    "Static-key auth attempted but "
                    "security.allow_static_key_fallback=false — refusing. "
                    "Either enable the key store or set "
                    "security.allow_static_key_fallback: true (single-tenant only)."
                )
                return False, None, None
            if not api_key or not hmac.compare_digest(api_key, static_key):
                return False, None, None
            # Stash empty quotas — static-key mode has no per-key caps.
            request.state.key_quotas = None
            return True, "static", None

        # No store, no static key: this is the "no auth configured at all"
        # state. Previously returned True (fail-open). Now it fails closed
        # unless the operator explicitly opts in via allow_no_auth=true.
        if allow_noauth:
            log.warning(
                "Serving WITHOUT authentication (security.allow_no_auth=true) — "
                "do not expose this service to any untrusted network"
            )
            request.state.key_quotas = None
            return True, "no_auth", None
        log.error(
            "Auth refused: key store disabled, no static key, "
            "allow_no_auth=false. Set api.key_store_enabled: true, or "
            "security.allow_no_auth: true (NOT for production)."
        )
        return False, None, None

    # ── Key-store path ────────────────────────────────────────────────────
    if not api_key:
        return False, None, store

    try:
        ok, key_id, _ = await store.validate_key(
            api_key, client_ip=request.client.host if request.client else "unknown"
        )
    except Exception as exc:
        log.warning("Key validation error: %s", exc)
        return False, None, store

    if ok and key_id:
        # Stash the per-key quotas on request.state so downstream handlers
        # can clamp max_tokens / timeout_s. This is a separate
        # query — cheaper than returning the whole key record from
        # validate_key, and keeps the validate hot path unchanged.
        try:
            quotas = await store.get_key_quotas(key_id)  # type: ignore[attr-defined]
            request.state.key_quotas = quotas
        except Exception as exc:
            # Never fail auth on quota introspection. Falls back to the
            # server-wide caps (inference.max_total_tokens, default_timeout_s).
            log.debug("Quota fetch failed for key '%s': %s", key_id, exc)
            request.state.key_quotas = None

    return ok, key_id, store


async def verify_master_key(
    request: Request,
    master_key: Optional[str] = Depends(_master_key_header),
) -> str:
    cfg = getattr(request.app.state, "config", None) or {}
    master_cfg = cfg.get("api", {}).get("master_key")
    if not master_cfg:
        raise HTTPException(403, "Master key not configured")
    if not master_key or not hmac.compare_digest(master_key, master_cfg):
        raise HTTPException(403, "Invalid master key")
    return master_key


_SSE_HEADERS: Dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control":     "no-cache, no-store",
    "Connection":        "keep-alive",
}

# Headers used on HTML responses so the browser never serves a stale
# single-page app after the server has been updated. Without these the
# client can silently run a version that is days out of date.
_HTML_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma":        "no-cache",
    "Expires":       "0",
}


# =============================================================================
# Checkpoint callback helper
# =============================================================================

async def get_session_id(
    request: Request,
    response: Response,
    key_info: tuple = Depends(verify_api_key),
) -> str:
    """Return the caller's session ID, bound to their API key.

    the returned string is a raw session_id suitable for
    use as a ``conv_store`` key. The *transport* value (cookie or header)
    is a signed token that binds the session to the caller's key_id. Three
    sources are checked, in order, and only signatures that verify against
    the current key are accepted:

      1. ``X-Session-ID`` header, if it contains a valid signed token.
         Unsigned headers are ignored — a bare session_id sent by a client
         could belong to another key, so it cannot be trusted.

      2. ``llm_session`` cookie, if it contains a valid signed token.

      3. None of the above → generate a fresh session_id, sign it for the
         current key, and emit it in a ``Set-Cookie`` so the next request
         carries it back.

    Operators who need legacy unsigned-header behaviour (single-tenant
    static-key deployments only — where every caller is equally trusted)
    can set ``security.require_signed_sessions: false``. Default is true.

    Dependency ordering note: ``Depends(verify_api_key)`` is resolved by
    FastAPI *before* this coroutine runs, and the same result is cached
    and reused when the caller (e.g. ``chat_completions``) also declares
    ``Depends(verify_api_key)`` — auth still runs exactly once per request.
    """
    is_valid, key_id, _ = key_info
    if not is_valid:
        # verify_api_key raises HTTPException(401/403) on invalid keys, so
        # this branch is defensive: if it ever triggers, something is wrong
        # with the dependency graph and we must not silently proceed with
        # an unauthenticated session.
        raise HTTPException(401, "Invalid or missing API key")

    key_id_str = key_id or "anonymous"

    # ── Signing secret ────────────────────────────────────────────────────
    # Derived from api.master_key (startup_validation guarantees its
    # presence — the server does not start without it). Derivation keeps
    # the master key cryptographically separated from session tokens.
    cfg = getattr(request.app.state, "config", None) or {}
    master_key = (cfg.get("api", {}) or {}).get("master_key", "")
    if not master_key:
        # Defense in depth — startup_validation should have caught this.
        log.error("get_session_id: api.master_key is empty at request time")
        raise HTTPException(500, "Server misconfigured: session signing unavailable")
    secret = _session_secret(master_key)

    # Legacy mode: operator opted out of signature checking. Only safe for
    # single-tenant static-key deployments (see config.yaml comment).
    require_signed = True
    try:
        require_signed = bool(
            (cfg.get("security", {}) or {}).get(
                "require_signed_sessions", True
            )
        )
    except Exception:
        require_signed = True

    # ── 1. Signed X-Session-ID header ─────────────────────────────────────
    header_val = (request.headers.get("X-Session-ID") or "").strip()
    if header_val:
        if require_signed:
            sid = verify_session(header_val, key_id_str, secret)
            if sid:
                return sid
            # Unsigned or wrong-key header — fall through to cookie. Logged
            # at INFO once per request so a legacy client that keeps
            # sending a bare session_id is visible to operators.
            log.info(
                "Ignoring unsigned/mismatched X-Session-ID header — "
                "clients must send the signed value from the Set-Cookie "
                "response. Falling back to signed cookie."
            )
        else:
            # Legacy mode — accept raw value. Documented in config.yaml as
            # a security trade-off.
            return header_val

    # ── 2. Signed llm_session cookie ──────────────────────────────────────
    cookie_val = (request.cookies.get("llm_session") or "").strip()
    if cookie_val:
        if require_signed:
            sid = verify_session(cookie_val, key_id_str, secret)
            if sid:
                return sid
            log.debug(
                "Discarding llm_session cookie: not signed for this key "
                "(expected when a client switches API keys)"
            )
        else:
            return cookie_val

    # ── 3. Generate + sign a new session ──────────────────────────────────
    sid = str(uuid.uuid4())
    signed = sign_session(sid, key_id_str, secret)
    # `secure` is auto-detected: HTTPS, or loopback + X-Forwarded-Proto:
    # https. Previously this was hard-coded `secure=True`, meaning the
    # cookie was never sent back over plain HTTP (dev, or a proxy that
    # forgot to set X-Forwarded-Proto) — every request started a fresh
    # session. is_secure_request handles both cases correctly.
    response.set_cookie(
        key="llm_session",
        value=signed,
        max_age=60 * 60 * 24 * 30,   # 30 days
        httponly=True,
        samesite="lax",
        secure=is_secure_request(request),
    )
    return sid

async def _save_checkpoint_callback(
    conv_store,
    conv_id: str,
    checkpoint: Any,
    original_prompt: Optional[str] = None,
    merged_history: Optional[List[dict]] = None,
) -> None:
    request_id = checkpoint.request_id

    async with _CHECKPOINT_LOCK:
        if request_id in _CHECKPOINT_DISABLED:
            return

    checkpoint_data = {
        "model_id": checkpoint.model_id,
        "segment_id": checkpoint.segment_number,
        "accumulated_text": checkpoint.accumulated_text,
        "tokens_generated": checkpoint.tokens_generated,
        "status": checkpoint.status,
    }
    if original_prompt:
        checkpoint_data["original_prompt"] = original_prompt
    if merged_history:
        checkpoint_data["merged_history"] = merged_history

    try:
        await conv_store.save_checkpoint(conv_id, checkpoint_data)
        async with _CHECKPOINT_LOCK:
            _CHECKPOINT_FAILURES.pop(request_id, None)
            _CHECKPOINT_DISABLED.discard(request_id)
    except Exception as e:
        async with _CHECKPOINT_LOCK:
            count = _CHECKPOINT_FAILURES.get(request_id, 0) + 1
            _CHECKPOINT_FAILURES[request_id] = count
            if count >= MAX_CHECKPOINT_FAILURES:
                _CHECKPOINT_DISABLED.add(request_id)
                log.error(
                    "Checkpoint save failed %d times for request %s, conv %s – "
                    "disabling further checkpoints for this request",
                    count, request_id[:8], conv_id[:8]
                )
            else:
                log.warning(
                    "Checkpoint save failed (attempt %d/%d) for request %s, conv %s: %s",
                    count, MAX_CHECKPOINT_FAILURES, request_id[:8], conv_id[:8], e
                )


# =============================================================================
# Segmented streaming helper
# =============================================================================
async def _stream_segmented(
    engine,
    ir: Any,
    body: ChatRequest,
    conv_store,
    conv_id: str,
    req: Request,
    key_id: str,
    store: Optional[KeyStore],
    start_time: float,
    profile: Optional[HardwareProfile],
    stream_fmt: str,
    merged: List[dict],
    first_token_s: Optional[float] = None,  # kept for API compat; unused
) -> AsyncIterator[bytes]:
    """Segmented streaming with server-side merged history."""
    async def checkpoint_cb(checkpoint):
        await _save_checkpoint_callback(
            conv_store, conv_id, checkpoint,
            original_prompt=getattr(ir, "prompt", None),
            merged_history=list(merged) if merged else None,
        )

    full_text = ""
    token_count = 0
    client_aborted = False
    _first_ts: Optional[float] = None

    async def _do_persist() -> None:
        marker = _extract_stream_marker(full_text)
        clean_text = _strip_stream_marker(full_text) if marker else full_text

        if conv_store and clean_text and not _is_error_token(clean_text):
            should_checkpoint = bool(marker) or client_aborted
            if should_checkpoint:
                if marker:
                    kind, tokens_gen = marker
                else:
                    kind = "INTERRUPTED"
                    tokens_gen = max(1, token_count, int(len(clean_text) / 3.5))
                try:
                    ckpt = {
                        "model_id": body.model,
                        "segment_id": 0,
                        "accumulated_text": clean_text,
                        "tokens_generated": tokens_gen,
                        "status": "interrupted",
                    }
                    _orig = getattr(ir, "prompt", None)
                    if _orig:
                        ckpt["original_prompt"] = _orig
                    if merged:
                        ckpt["merged_history"] = list(merged)
                    await conv_store.save_checkpoint(conv_id, ckpt)
                    log.info(
                        "[%s] Checkpointed segmented %s stream "
                        "(conv=%s, tokens=%d, original_prompt=%s, "
                        "merged_history=%s)",
                        ir.request_id[:8], kind.lower(), conv_id[:8],
                        tokens_gen, "yes" if _orig else "no",
                        "yes" if merged else "no",
                    )
                except Exception as _cpe:
                    log.warning(
                        "[%s] Segmented checkpoint save failed: %s",
                        ir.request_id[:8], _cpe,
                    )
            else:
                history_with_reply = (
                    list(merged) + [{"role": "assistant", "content": clean_text}]
                )
                try:
                    await conv_store.create(conv_id, history_with_reply)
                except Exception as ce:
                    log.warning(
                        "[%s] Segmented conversation save failed: %s",
                        ir.request_id[:8], ce,
                    )

        elapsed = time.monotonic() - start_time
        _ttft_ms = ((_first_ts - start_time) * 1000.0) if _first_ts else None
        _decode_s = elapsed - ((_first_ts - start_time) if _first_ts else 0.0)
        _decode_tps = (
            (token_count - 1) / _decode_s
            if _decode_s > 0.05 and token_count > 1 else None
        )
        try:
            await _log_request(
                req, key_id, store, "/v1/chat/completions",
                499 if client_aborted else 200,
                elapsed * 1000, token_count, request_id=ir.request_id,
                ttft_ms=_ttft_ms, decode_tps=_decode_tps,
            )
        except Exception as le:
            log.warning(
                "[%s] Segmented request log failed: %s",
                ir.request_id[:8], le,
            )

        with suppress(Exception):
            asyncio.create_task(cleanup_after_generation(req.app.state.ctx))

    # Resolve token_source BEFORE the try so `_toks` is always bound in
    # `finally` — a GeneratorExit fired at the very first line of the
    # try-block must not cause a NameError during cleanup. Same for
    # `_async_iter`, so `aclose()` in `finally` never dereferences an
    # unbound name.
    _toks = req.app.state.ctx.get("token_source") or engine
    _gen = getattr(_toks, "generate_segmented", None)
    if _gen is None:
        # Pool: emulate segmented via plain stream
        _async_iter = _toks.generate_stream(ir)
    else:
        _async_iter = _gen(ir, checkpoint_cb)
    try:
        async for token in _async_iter:
            # Same disconnect check as _chat_sse. Without this, the
            # GeneratorExit thrown by aclose() on client disconnect
            # bypasses both except asyncio.CancelledError and
            # except Exception, so client_aborted stays False and
            # _do_persist saves a truncated reply as COMPLETE.
            if await req.is_disconnected():
                log.info(
                    "[%s] Client disconnected during segmented stream "
                    "after %d tokens — treating as interruption",
                    ir.request_id[:8], token_count,
                )
                client_aborted = True
                break
            if token.startswith("[Error:") or token.startswith("[Response interrupted"):
                if stream_fmt == "json":
                    payload = json.dumps({
                        "id": ir.request_id,
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [{"delta": {"content": token},
                                     "finish_reason": "interrupted", "index": 0}],
                    })
                    yield f"data: {payload}\n\n".encode()
                else:
                    yield token.encode()
                break
            if stream_fmt == "json":
                payload = json.dumps({
                    "id": ir.request_id,
                    "object": "chat.completion.chunk",
                    "model": body.model,
                    "choices": [{"delta": {"content": token},
                                 "finish_reason": None, "index": 0}],
                })
                yield f"data: {payload}\n\n".encode()
            else:
                yield token.encode()
            if _first_ts is None:
                _first_ts = time.monotonic()
            full_text += token
            token_count += 1
    except (asyncio.CancelledError, GeneratorExit):
        client_aborted = True
        log.info(
            "[%s] Segmented stream cancelled/disconnected after %d tokens — "
            "propagating",
            ir.request_id[:8], token_count,
        )
        raise
    except Exception as _seg_exc:
        from local_llm_core.engine.inference.inference_engine import (
            StreamStallError as _SSE, _make_stall_marker as _mk,
        )
        if isinstance(_seg_exc, _SSE):
            log.warning(
                "[%s] Segmented stream stalled during %s after %d tokens — "
                "emitting ⚡STALLED marker",
                ir.request_id[:8], _seg_exc.phase, token_count,
            )
            _m = _mk(_seg_exc.phase, _seg_exc.partial_tokens or token_count)
            if stream_fmt == "json":
                payload = json.dumps({
                    "id": ir.request_id,
                    "object": "chat.completion.chunk",
                    "model": body.model,
                    "choices": [{"delta": {"content": _m},
                                 "finish_reason": "stalled", "index": 0}],
                })
                yield f"data: {payload}\n\n".encode()
            else:
                yield _m.encode()
            full_text += _m
        else:
            raise
    finally:
        # Close the inner stream BEFORE running persistence. Without
        # this, `async for` alone does not call aclose() when the loop
        # exits via `break` (client disconnect, error marker) or an
        # exception — so the inner pool/engine stream keeps producing
        # until its own cancellation path fires, and any tokens it
        # generates after our break are lost to `full_text` /
        # checkpoint callbacks. `aclose()` here:
        #   * stops the inner producer immediately,
        #   * lets any pending pool `cancelled` frame drain through,
        #   * is a no-op when the stream already finished cleanly.
        with suppress(Exception):
            await _async_iter.aclose()
        with suppress(Exception):
            _toks.clear_progress(ir.root_id or ir.request_id)
        asyncio.create_task(_do_persist())

    yield b"data: [DONE]\n\n"


# =============================================================================
# Streaming primitives
# =============================================================================
async def _openai_sse_stream(
    engine,
    ir,
    model_id: str,
    mode: str = "chat",
    optimizer=None,
    req: Optional[Request] = None,
    profile: Optional[HardwareProfile] = None,
) -> AsyncIterator[bytes]:
    cmpl_id = (
        f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if mode == "chat"
        else f"cmpl-{uuid.uuid4().hex[:12]}"
    )
    created = int(time.time())
    _toks = req.app.state.ctx.get("token_source") or engine
    source = optimizer.generate_stream(ir) if optimizer else _toks.generate_stream(ir)

    max_tokens = getattr(ir, "max_tokens", 0)

    if req is not None:
        source = _wrap_stream(
            lambda: source,
            req,
            ir,
            profile,
            max_tokens,
        )

    t_start = time.monotonic()
    # t_last measures time since the *last* token, not since stream start.
    # The old code killed the stream when TOTAL elapsed exceeded
    # ir.timeout_s, which on a 4 tps box at 720 s was ~2,900 tokens —
    # an artificial ceiling disguised as a timeout. Now we only trip when
    # the stream has produced nothing for ir.timeout_s (a genuine stall).
    t_last = t_start
    chunk_count = 0

    try:
        async for tok in source:
            if ir.timeout_s > 0 and time.monotonic() - t_last > ir.timeout_s:
                yield (f"data: {json.dumps({'error': 'stream timeout'})}\n\n").encode()
                return
            t_last = time.monotonic()
            chunk_count += 1
            if mode == "chat":
                chunk = {
                    "id": cmpl_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant", "content": tok},
                                 "finish_reason": None}],
                }
            else:
                chunk = {
                    "id": cmpl_id, "object": "text_completion",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0, "text": tok, "finish_reason": None}],
                }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

    except asyncio.CancelledError:
        yield f"data: {json.dumps({'error': 'stream cancelled'})}\n\n".encode()
    except Exception as exc:
        log.error("SSE stream error [%s]: %s",
                  getattr(ir, "request_id", "")[:8], exc)
        yield f"data: {json.dumps({'error': str(exc)[:200]})}\n\n".encode()
    finally:
        if hasattr(source, "aclose"):
            try:
                await source.aclose()
            except (GeneratorExit, asyncio.CancelledError):
                pass
            except Exception as e:
                log.warning("[%s] Error closing SSE source: %s",
                            getattr(ir, "request_id", "")[:8], e)

    # Terminal frames emitted outside the try/finally so aclose() of this
    # generator (via _SafeStream or an outer wrapper) can complete cleanly.
    if chunk_count == 0:
        yield f"data: {json.dumps({'error': 'Model generated no output — please retry'})}\n\n".encode()
        yield b"data: [DONE]\n\n"
        #`return` here prevents (a) a second `data: [DONE]` from
        # the code below, and (b) a spurious `finish_reason: "stop"` frame
        # emitted after an error — some clients render that as an empty
        # assistant turn, and some strict SSE parsers treat the duplicate
        # DONE as a protocol error.
        return

    if mode == "chat":
        final = {"id": cmpl_id, "object": "chat.completion.chunk",
                 "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    else:
        final = {"id": cmpl_id, "object": "text_completion",
                 "created": created, "model": model_id,
                 "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _stream_plain_text(
    engine,
    ir,
    optimizer=None,
    req: Optional[Request] = None,
    profile: Optional[HardwareProfile] = None,
) -> AsyncIterator[bytes]:
    _toks = (req.app.state.ctx.get("token_source") or engine) if req is not None else engine
    source = optimizer.generate_stream(ir) if optimizer else _toks.generate_stream(ir)
    max_tokens = getattr(ir, "max_tokens", 0)
    if req is not None:
        source = _wrap_stream(
            lambda: source,
            req,
            ir,
            profile,
            max_tokens,
        )
    try:
        async for tok in source:
            yield tok.encode() if isinstance(tok, str) else tok
    finally:
        if hasattr(source, "aclose"):
            try:
                await source.aclose()
            except (GeneratorExit, asyncio.CancelledError):
                pass
            except Exception as e:
                log.warning("Error closing plain-text source: %s", e)


def _resp(result, model_id: str) -> dict:
    text          = getattr(result, "text", "") or ""
    tokens_gen    = getattr(result, "tokens_gen", 0)
    prompt_tokens = getattr(result, "prompt_tokens", max(1, len(text.split()) // 4))
    finish_reason = getattr(result, "finish_reason", "stop") or "stop"
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": tokens_gen,
            "total_tokens":      prompt_tokens + tokens_gen,
        },
    }


def validate_stream_chunk(chunk: str, request_id: str = "") -> bool:
    return isinstance(chunk, str) and len(chunk.strip()) > 0


def ensure_response_not_empty(full_text: str, model_id: str = "", request_id: str = "") -> str:
    if not full_text or not full_text.strip():
        log.error("[%s] Model %s generated empty response", request_id[:8], model_id)
        return "[Error: Model generated no output — please retry]"
    return full_text


async def validate_stream_response(
    stream_gen: AsyncIterator[str],
    request_id: str = "",
    min_chunks: int = 1,
) -> AsyncIterator[str]:
    chunk_count = 0
    total_chars = 0
    async for chunk in stream_gen:
        if not chunk:
            continue
        chunk_count += 1
        total_chars += len(chunk)
        yield chunk
    if chunk_count == 0 or total_chars == 0:
        log.error("[%s] Empty stream — backend produced zero tokens.", request_id[:8])
        raise RuntimeError(
            f"[{request_id[:8]}] Backend produced no tokens. "
            "The model may be stuck or overloaded — please retry. "
            "Check server logs for timeout or circuit-breaker messages."
        )


async def cleanup_after_generation(ctx: dict) -> None:
    try:
        await asyncio.sleep(0.5)
        vram = ctx.get("vram")
        if vram and hasattr(vram, "defragment_gpu_memory"):
            with suppress(Exception):
                freed = vram.defragment_gpu_memory()
                if freed > 0:
                    log.info("VRAM defrag: freed %d MB", freed >> 20)
        import gc
        gc.collect()
    except Exception as exc:
        log.warning("cleanup_after_generation: %s", exc)

async def _log_request(
    req: Request,
    key_id: Optional[str],
    store: Optional[KeyStore],
    path: str,
    status: int,
    latency_ms: float,
    tokens: int = 0,
    request_id: Optional[str] = None,
    ttft_ms: Optional[float] = None,
    decode_tps: Optional[float] = None,
) -> None:

    client_ip = req.client.host if req.client else "unknown"
    if store:
        with suppress(Exception):
            await store.log_request(
                key_id, path, req.method, status, latency_ms, tokens,
                client_ip, request_id,
            )
        # Accumulate per-key token totals.
        # validate_key() runs at request START with tokens=0 — the real
        # count is unknown until the stream finishes. Without this call,
        # api_keys.total_tokens stays 0 forever and /admin/keys/stats/{id}
        # always reports total_tokens: 0 regardless of actual usage.
        if key_id and tokens:
            with suppress(Exception):
                rec = getattr(store, "record_usage", None)
                if rec is not None:
                    await rec(key_id, int(tokens), client_ip)
    log.info(
        "%s %s → %d  %.1fms  tok=%d  ttft=%s  dec_tps=%s  key=%s  id=%s",
        req.method, path, status, latency_ms,
        int(tokens or 0),
        f"{ttft_ms:.0f}ms" if ttft_ms is not None else "n/a",
        f"{decode_tps:.2f}" if decode_tps is not None else "n/a",
        key_id or "anon",
        (request_id or "")[:8],
        extra={
            "request_id": request_id or "",
            "key_id":     key_id or "anon",
            "path":       path,
            "method":     req.method,
            "status":     status,
            "latency_ms": float(latency_ms or 0.0),
            "tokens":     int(tokens or 0),
            "client_ip":  client_ip,
            "ttft_ms":    ttft_ms,
            "decode_tps": decode_tps,
        },
    )

# =============================================================================
# Engine helpers
# =============================================================================
def _engine_get_drive_wired(engine) -> bool:
    return bool(getattr(engine, "drive_wired", False) or getattr(engine, "_drive_wired", False))


def _engine_get_ctx_cache_stats(engine) -> Optional[dict]:
    c = getattr(engine, "context_cache", None) or getattr(engine, "_ctx_cache", None)
    return c.stats() if c and hasattr(c, "stats") else None


def _engine_get_throughput_stats(engine) -> Optional[dict]:
    c = getattr(engine, "throughput_ctrl", None) or getattr(engine, "_tput_ctrl", None)
    return c.stats() if c and hasattr(c, "stats") else None


def _engine_get_ngram_hit_rate(engine) -> float:
    n = getattr(engine, "ngram_cache", None) or getattr(engine, "_ngram", None)
    return round(float(n.hit_rate), 3) if n and hasattr(n, "hit_rate") else 0.0


def _engine_get_concurrency(engine) -> int:
    m = getattr(engine, "token_mgr", None) or getattr(engine, "_token_mgr", None)
    return m.capacity if m and hasattr(m, "capacity") else getattr(engine, "max_batch", 1)


def _engine_clear_ctx_cache(engine) -> bool:
    c = getattr(engine, "context_cache", None) or getattr(engine, "_ctx_cache", None)
    if c and hasattr(c, "clear"):
        c.clear(); return True
    return False


def _validate_model_path_strict(model_path: str, allowed_dir: str) -> str:
    resolved = Path(model_path).resolve(strict=False)
    if not resolved.exists():
        raise HTTPException(404, f"Model path not found: {model_path}")
    allowed = Path(allowed_dir).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError:
        raise HTTPException(403, f"Path outside allowed_dir: {model_path}")
    if not os.access(resolved, os.R_OK):
        raise HTTPException(403, f"Model path not readable: {model_path}")
    return str(resolved)


def _detect_model_type(model_path: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return "gguf"
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                s = json.dumps(data).lower()
                if "bitnet" in s or "1bit" in s or "bitlinear" in s:
                    return "bitnet"
            except Exception:
                pass
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return "bitnet"
    return "unknown"


# =============================================================================
# Runtime limits — hardware + model aware, exposed to clients
# =============================================================================
def _derive_limits(effective_tps: float, ctx_headroom: int, min_response: int):
    """Return (idle_s, chunk, response_tokens) for the given throughput."""
    if effective_tps < 3.0:
        idle_s, chunk_s, resp_s = 900, 240, 480
    elif effective_tps < 10.0:
        idle_s, chunk_s, resp_s = 600, 240, 480
    elif effective_tps < 30.0:
        idle_s, chunk_s, resp_s = 300, 180, 360
    elif effective_tps < 100.0:
        idle_s, chunk_s, resp_s = 180, 120, 300
    else:
        idle_s, chunk_s, resp_s = 60, 60, 240

    chunk    = max(min_response, min(int(effective_tps * chunk_s), ctx_headroom))
    response = max(min_response, min(int(effective_tps * resp_s),  ctx_headroom))
    return idle_s, chunk, response

def _resolve_long_request_threshold(
    engine, model_id: str, inf_cfg: dict,
) -> int:
    """Return the effective 'go segmented' threshold for this request.

    -1 (auto)   → min(n_ctx // 4, segment_size × max_segments)
     0          → 0 (always segmented for any positive max_tokens)
     N > 0      → N (segmented only when the caller asks for > N)
    """
    cfg = int(inf_cfg.get("long_request_threshold_tokens", -1))
    if cfg >= 0:
        return cfg
    try:
        n_ctx = int(engine._model_n_ctx(model_id)) or 0
    except Exception:
        n_ctx = 0
    seg = int(inf_cfg.get("segment_size", 512) or 512)
    mx  = int(inf_cfg.get("max_segments", 0) or 0)
    by_ctx = max(256, n_ctx // 4) if n_ctx else 1024
    by_cap = seg * mx if mx > 0 else by_ctx
    return max(256, min(by_ctx, by_cap))

def _compute_runtime_limits(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Single source of truth for every dynamic limit in the system."""
    engine = ctx.get("token_source") or ctx.get("engine")
    profile = ctx.get("hw_profile")
    sys_info = ctx.get("sys_info")

    # Prefer the *loaded* model's real context window over the static
    # profile field. This endpoint is the single source of truth the Web
    # UI reads to size its prompts, and it must reflect reality.
    n_ctx = 0
    if engine is not None:
        try:
            _loaded_ids = engine.get_loaded_model_ids()
            if _loaded_ids:
                n_ctx = int(engine._model_n_ctx(_loaded_ids[0]))
        except Exception:
            pass
    n_ctx = n_ctx or int(getattr(profile, "context_length", 2048) or 2048)

    # Throughput: prefer measured, fall back to target. Below 0.5 tps the
    # measurement is almost certainly noise (cold start, single warmup
    # token), so use the profile's target instead.
    measured_tps = 0.0
    target_tps   = float(getattr(profile, "target_tps", 8.0) or 8.0)
    if engine is not None:
        try:
            measured_tps = float(engine.stats().get("avg_tps", 0.0) or 0.0)
        except Exception:
            pass
    effective_tps = measured_tps if measured_tps >= 0.5 else target_tps
    # Measured prefill throughput — 0.0 when the engine has no samples yet.
    # Documented for clients; not a substitute for their own fallback.
    prefill_tps_meas = 0.0
    if engine is not None and hasattr(engine, "prefill_tps"):
        with suppress(Exception):
            prefill_tps_meas = float(engine.prefill_tps())

    # Measured decode-only throughput — 0.0 when the engine has no samples.
    # This excludes prefill time, unlike `effective_tps`. AMB-APP's
    # CapabilityProfile uses this for latency math (F-06).
    decode_tps_meas = 0.0
    if engine is not None and hasattr(engine, "decode_tps"):
        with suppress(Exception):
            decode_tps_meas = float(engine.decode_tps())

    # Average new prompt tokens per call (cache-miss tokens only).
    # The client's F-06 estimator uses this to size the prefill budget.
    avg_new_prompt = 200
    if engine is not None and hasattr(engine, "avg_new_prompt_tokens"):
        with suppress(Exception):
            avg_new_prompt = int(engine.avg_new_prompt_tokens())

    # Configurable floors from the engine's own config.
    safety       = 256
    min_response = 64
    segment_size = 512
    max_segments = 200
    if engine is not None:
        try:
            safety       = int(engine._cfg.get("context_safety_tokens", 256))
        except Exception:
            pass
        try:
            min_response = int(engine._cfg.get("min_response_tokens", 64))
        except Exception:
            pass
        segment_size = int(getattr(engine, "_segment_size", 512) or 512)
        max_segments = int(getattr(engine, "_max_segments", 200) or 200)

    # Memory snapshot
    vram_total_mb = 0
    vram_free_mb  = 0
    if sys_info and getattr(sys_info, "gpus", None):
        for g in sys_info.gpus:
            if getattr(g, "supported", True):
                vram_total_mb += int(getattr(g, "vram_total_mb", 0) or 0)
                vram_free_mb  += int(getattr(g, "vram_free_mb",  0) or 0)
    try:
        vm           = psutil.virtual_memory()
        ram_total_mb = int(vm.total >> 20)
        ram_free_mb  = int(vm.available >> 20)
    except Exception:
        ram_total_mb = 0
        ram_free_mb  = 0

    # Context headroom: reserve 25 % of the context for the prompt, then
    # subtract the safety margin. This is a conservative estimate — the
    # server still runs _fit_prompt_to_context per request.
    ctx_headroom = max(min_response, n_ctx - safety - (n_ctx // 4))

    idle_s, resume_chunk, rec_response = _derive_limits(
        effective_tps, ctx_headroom, min_response
    )

    # Tier label (for logging + client-side display)
    if effective_tps < 3.0:      tier = "minimal"
    elif effective_tps < 10.0:   tier = "small"
    elif effective_tps < 30.0:   tier = "medium"
    elif effective_tps < 100.0:  tier = "large"
    else:                        tier = "very_large"

    # Model metadata. In pool mode, `engine` is the pool which has no
    # `_unified_models`; fall back to the pool's load_spec.model_path.
    model_id      = ""
    model_size_mb = 0
    if engine is not None:
        try:
            loaded = engine.get_loaded_model_ids()
            if loaded:
                model_id = loaded[0]
                # Try engine-style first
                _mp = ""
                m = getattr(engine, "_unified_models", {}).get(model_id) \
                    if hasattr(engine, "_unified_models") else None
                if m is not None:
                    cfg = getattr(m, "config", None)
                    _mp = getattr(cfg, "model_path", "") or ""
                # Fall back to pool-style: load_spec.model_path
                if not _mp and hasattr(engine, "_spec"):
                    _mp = (engine._spec or {}).get("model_path", "") or ""
                if _mp and Path(_mp).exists():
                    model_size_mb = int(Path(_mp).stat().st_size >> 20)
        except Exception:
            pass

    return {
        "version":                     VERSION,
        "tier":                        tier,
        "hardware_profile":            getattr(profile, "name", "unknown"),
        "backend":                     getattr(sys_info, "recommended_backend", "cpu")
                                       if sys_info else "cpu",
        "model_id":                    model_id,
        "model_size_mb":               model_size_mb,
        "context_length":              n_ctx,
        "context_safety_tokens":       safety,
        "min_response_tokens":         min_response,
        "segment_size":                segment_size,
        "max_segments":                max_segments,

        # ── Generation policy ──────────────────────────────────────────
        # The server is now unbounded by default. The engine slides its
        # own context window when a reply approaches n_ctx, so the client
        # never has to ask the user for "Continue" except on a genuine
        # interruption (client cancel, real stall, disconnect).
        "generation_mode":             "unbounded_logical",
        "auto_continuation":           True,
        "unlimited_output":            True,

        # ── Legacy recommendation fields ───────────────────────────────
        # Kept for backward compat with older client builds. New clients
        # MUST NOT use these as output caps — they are informational.
        # A warning is logged once per process if a client ever sets
        # max_tokens equal to recommended_response_tokens.
        "recommended_response_tokens": rec_response,
        "recommended_resume_chunk":    resume_chunk,
        "recommended_idle_timeout_s":  idle_s,
        "recommended_n_parallel":      int(getattr(profile, "n_parallel", 1) or 1),

        # ── Telemetry ──────────────────────────────────────────────────
        "measured_tps":                round(measured_tps, 2),
        "target_tps":                  round(target_tps, 2),
        "effective_tps":               round(effective_tps, 2),
        "prefill_tps":                 round(prefill_tps_meas, 2),
        "decode_tps":                  round(decode_tps_meas, 2),
        "avg_new_prompt_tokens":       int(avg_new_prompt),
        "vram_total_mb":               vram_total_mb,
        "vram_free_mb":                vram_free_mb,
        "ram_total_mb":                ram_total_mb,
        "ram_free_mb":                 ram_free_mb,
    }


# =============================================================================
# Startup validation
# =============================================================================
async def startup_validation(cfg: dict, config_path: str) -> None:
    log.info("=== Running startup validation ===")
    errors: List[str]   = []
    warnings: List[str] = []

    if not Path(config_path).exists():
        errors.append(f"Config not found: {config_path}")

    models_cfg  = cfg.get("models", {})
    allowed_dir = models_cfg.get("allowed_dir")
    if allowed_dir:
        p = Path(allowed_dir)
        if not p.exists():
            warnings.append(f"models.allowed_dir missing: {allowed_dir}")
        elif not os.access(allowed_dir, os.R_OK):
            errors.append(f"models.allowed_dir not readable: {allowed_dir}")
    else:
        warnings.append("models.allowed_dir not set — model loading restricted")

    profiles    = models_cfg.get("profiles", [])
    autoload_all = models_cfg.get("autoload_all", False)
    for p in profiles:
        if p.get("autoload") or autoload_all:
            path = p.get("path", "")
            if not path:
                errors.append(f"Profile '{p.get('id', '?')}' missing path")
            elif not Path(path).exists():
                warnings.append(f"Profile '{p.get('id', '?')}' path not found: {path}")

    api_cfg = cfg.get("api", {})
    _ks_enabled = bool(api_cfg.get("key_store_enabled", False))
    _static_key = str(api_cfg.get("key") or "").strip()
    _master_key = str(api_cfg.get("master_key") or "").strip()

    # Hard failures: a secret that expanded from ${VAR} to empty means the
    # operator forgot to export it. Do NOT let the server start half-
    # configured — the API endpoints would silently break in confusing ways.
    if not _ks_enabled and not _static_key:
        errors.append(
            "api.key is empty and key_store_enabled=false — set "
            "LLM_API_KEY env var or configure a key store."
        )
    if _ks_enabled and _static_key and _static_key.startswith("${"):
        # _expand_env_vars would have already substituted it, so a remaining
        # "${" means the var was never exported.
        errors.append("api.key ${LLM_API_KEY} did not expand — export the env var.")

    if not _master_key or _master_key.startswith("${"):
        errors.append(
            "api.master_key is empty — set LLM_MASTER_KEY env var before "
            "starting the server (admin endpoints will be permanently "
            "unavailable otherwise)."
        )

    for _db_section in ("api", "conversation"):
        _sect = cfg.get(_db_section, {}) or {}
        _pwd = str(_sect.get("password") or "").strip()
        _storage = str(_sect.get("storage") or "").strip().lower()
        if _storage in ("postgresql", "postgres") and (not _pwd or _pwd.startswith("${")):
            errors.append(
                f"{_db_section}.password is empty — set LLM_DB_PASSWORD env "
                f"var (Postgres auth will fail otherwise)."
            )

    if not _static_key and not _ks_enabled:
        warnings.append(
            "api.key not set and key_store_enabled=false — server running WITHOUT authentication!"
        )

    for db_key, db_cfg_key in [
        ("Key store", "key_store_db"),
        ("Conversation DB", "sqlite_path"),
    ]:
        path = api_cfg.get(db_cfg_key) or cfg.get("conversation", {}).get(db_cfg_key)
        if path:
            d = Path(path).parent
            try:
                d.mkdir(parents=True, exist_ok=True)
                if not os.access(d, os.W_OK):
                    errors.append(f"{db_key} dir not writable: {d}")
            except Exception as exc:
                errors.append(f"{db_key} dir check failed: {exc}")

    swap_dir = cfg.get("memory", {}).get("ssd_swap_path", _default_state_dir() / "llm_swap")
    try:
        Path(swap_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        warnings.append(f"Swap dir check: {exc}")

    log_file = cfg.get("logging", {}).get("file", _default_state_dir() / "llm_server.log")
    if log_file:
        log_dir = Path(log_file).parent
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(log_dir, os.W_OK):
                errors.append(f"Log dir not writable: {log_dir}")
        except Exception as exc:
            errors.append(f"Log dir check: {exc}")

    try:
        import llama_cpp  # noqa
        log.info("✓ llama-cpp-python available")
    except ImportError:
        errors.append("llama-cpp-python not installed (required for GGUF inference)")

    try:
        import aiosqlite  # noqa
        log.info("✓ aiosqlite available")
    except ImportError:
        errors.append("aiosqlite not installed (required for key store / conversations)")

    for w in warnings:
        log.warning("⚠ %s", w)

    if errors:
        log.error("=== STARTUP VALIDATION FAILED ===")
        for e in errors:
            log.error("❌ %s", e)
        raise RuntimeError("Startup validation failed — see above")

    try:
        from local_llm_core.compatibility.transformers_compat import StateFileReader
        StateFileReader.apply_spectre_pythonpath()
        log.info("✓ SPECTRE PYTHONPATH applied")
    except Exception as exc:
        log.warning("SPECTRE in validation: %s", exc)

    try:
        from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader  # noqa: F401
        _profiles = cfg.get("models", {}).get("profiles", [])
        _al_all   = cfg.get("models", {}).get("autoload_all", False)
        _active   = next(
            (p for p in _profiles if p.get("autoload") or _al_all), None
        )
        if _active:
            _pid   = _active.get("id", "?")
            _ppath = _active.get("path", "")
            _is_bn = False
            _backend = "gguf"
            try:
                from local_llm_core.engine.adaptive_awareness.model_unified_env import (
                    classify_gguf, GGUFModelType,
                )
                if _ppath.endswith(".gguf"):
                    _hdr = classify_gguf(_ppath)
                    _is_bn = (
                        _hdr.model_type == GGUFModelType.BITNET
                        or _active.get("quantization") in ("1bit", "i2_s", "tl1", "tl2")
                        or any(x in _ppath.lower() for x in ("i2_s", "tl1", "tl2", "bitnet"))
                    )
                    if _is_bn:
                        try:
                            from local_llm_core.engine.bitnet.bitnet_cpp_backend import (
                                _find_binary as _bn_find_binary,
                                _LLAMA_SERVER_NAMES, _LLAMA_CLI_NAMES,
                            )
                            if _bn_find_binary(_LLAMA_SERVER_NAMES) is not None:
                                _backend = "bitnet_cpp_server"
                            elif _bn_find_binary(_LLAMA_CLI_NAMES) is not None:
                                _backend = "bitnet_cpp_cli"
                            else:
                                _backend = "bitnet (NO BINARY BUILT — will fail to load)"
                        except Exception:
                            _backend = "bitnet (binary check unavailable)"
                else:
                    _backend = "transformers"
            except Exception as _cls_exc:
                log.warning("Startup banner: classify_gguf check failed: %s", _cls_exc)
                _backend = "gguf" if _ppath.endswith(".gguf") else "transformers"
            label = "BitNet" if _is_bn else "Model"
            log.info("✓ %s: loader ready  (profile='%s'  backend=%s)", label, _pid, _backend)
        else:
            log.info("✓ BitNet: loader ready")
    except Exception as exc:
        log.warning("BitNet check: %s", exc)

    log.info("=== Startup validation passed ===")

# =============================================================================
# _persist_after_stream — shared tail for every streaming path.
#
# Lives at MODULE level (not inside create_app) so that
# _stream_with_thinking_retry, which is also module-level, can call it.
# The function is stateless: it uses only module-level helpers
# (_extract_stream_marker, _strip_stream_marker, _is_error_token,
# _log_request, cleanup_after_generation) and accesses the per-request
# state via req.app.state.ctx. No closure over create_app is required.
# =============================================================================
async def _persist_after_stream(
        *, conv_store, session_id, body, ir, full, req, key_id, store, start,
        merged: List[dict],
        status: int = 200,
        path: str = "/v1/chat/stream",
        ttft_ms: Optional[float] = None,
        decode_tps: Optional[float] = None,
        tokens: Optional[int] = None,
) -> None:
    """Post-stream bookkeeping.

    ttft_ms / decode_tps — forwarded to _log_request so the access log
    captures time-to-first-token and decode-only tokens/sec for this
    request. Both are None when the stream produced no content tokens
    (e.g. client aborted before first token).

    tokens — real content-token count from the streaming closure. When
    None (e.g. callers that haven't been updated yet) the old estimate
    len(clean_text) // 3.5 is used. Passing real tokens is preferred:
    api_keys.total_tokens accumulation and rate-limit accounting both
    read this value, and the estimate undercounts whitespace-heavy
    outputs by ~15%.
    """
    marker = _extract_stream_marker(full)
    clean_text = _strip_stream_marker(full) if marker else full

    if conv_store and clean_text and not _is_error_token(clean_text):
        should_checkpoint = bool(marker) or (status == 499)
        if should_checkpoint:
            if marker:
                kind, tokens_gen = marker
            else:
                kind = "INTERRUPTED"
                tokens_gen = max(1, int(len(clean_text) / 3.5))
            try:
                ckpt = {
                    "model_id": body.model,
                    "segment_id": 0,
                    "accumulated_text": clean_text,
                    "tokens_generated": tokens_gen,
                    "status": "interrupted",
                }
                _orig = getattr(ir, "prompt", None)
                if _orig:
                    ckpt["original_prompt"] = _orig
                if merged:
                    ckpt["merged_history"] = list(merged)
                await conv_store.save_checkpoint(session_id, ckpt)
                log.info(
                    "Checkpointed %s stream %s (conv=%s, tokens=%d, "
                    "original_prompt=%s, merged_history=%s)",
                    kind.lower(), ir.request_id[:8], session_id[:8],
                    tokens_gen, "yes" if _orig else "no",
                    "yes" if merged else "no",
                )
            except Exception as _cpe:
                log.warning("Checkpoint save failed: %s", _cpe)
        else:
            history_with_reply = (
                list(merged) + [{"role": "assistant", "content": clean_text}]
            )
            try:
                await conv_store.create(session_id, history_with_reply)
            except Exception as _ce:
                log.warning("Conversation save failed: %s", _ce)

    try:
        await _log_request(
            req, key_id, store, path, status,
            (time.monotonic() - start) * 1000,
            tokens if tokens is not None else len(clean_text) // 3.5,
            request_id=ir.request_id,
            ttft_ms=ttft_ms,
            decode_tps=decode_tps,
        )
    except Exception as _le:
        log.warning("Request log failed: %s", _le)

    with suppress(Exception):
        asyncio.create_task(cleanup_after_generation(req.app.state.ctx))

# =============================================================================
# Streaming with thinking retry
# =============================================================================
async def _stream_with_thinking_retry(
    engine,
    ir,
    optimizer,
    body: ChatRequest,
    request_id: str,
    stream_fmt: str,
    conv_store,
    session_id: str,
    merged_messages: List[dict],
    req: Request,
    key_id: str,
    store: Optional[KeyStore],
    start_time: float,
    profile: Optional[HardwareProfile],
) -> AsyncIterator[bytes]:
    full_text      = ""
    retry_attempted = False
    chunks_yielded  = 0
    _first_ts: Optional[float] = None
    _token_count    = 0

    _toks = req.app.state.ctx.get("token_source") or engine
    source = optimizer.generate_stream(ir) if optimizer else _toks.generate_stream(ir)
    max_tokens = getattr(ir, "max_tokens", 0)
    source = _wrap_stream(
        lambda: source,
        req,
        ir,
        profile,
        max_tokens,
    )

    try:
        async for chunk in validate_stream_response(source, request_id=ir.request_id):
            if await req.is_disconnected():
                log.info(
                    "[%s] Client disconnected during thinking-mode stream "
                    "after %d tokens",
                    request_id[:8], _token_count,
                )
                # Persist the partial reply as an interrupted checkpoint
                # so /resume works — this path is not covered by the
                # segmented handler.
                await _persist_after_stream(
                    conv_store=conv_store,
                    session_id=session_id,
                    body=body, ir=ir, full=full_text,
                    req=req, key_id=key_id, store=store,
                    start=start_time,
                    merged=merged_messages,
                    status=499,
                    path="/v1/chat/completions",
                    ttft_ms=((_first_ts - start_time) * 1000.0) if _first_ts else None,
                    decode_tps=None,
                    tokens=_token_count,
                )
                return
            if not validate_stream_chunk(chunk, ir.request_id):
                continue
            if stream_fmt == "json":
                payload = json.dumps({
                    "id":      ir.request_id,
                    "object":  "chat.completion.chunk",
                    "model":   body.model,
                    "choices": [{"delta": {"content": chunk}, "finish_reason": None, "index": 0}],
                })
                yield f"data: {payload}\n\n".encode()
            else:
                yield chunk.encode()
            if _first_ts is None:
                _first_ts = time.monotonic()
            _token_count += 1
            full_text     += chunk
            chunks_yielded += 1
    except (asyncio.CancelledError, GeneratorExit):
        # Persist a checkpoint before propagating so /resume can pick up.
        with suppress(Exception):
            await _persist_after_stream(
                conv_store=conv_store,
                session_id=session_id,
                body=body, ir=ir, full=full_text,
                req=req, key_id=key_id, store=store,
                start=start_time,
                merged=merged_messages,
                status=499,
                path="/v1/chat/completions",
                ttft_ms=((_first_ts - start_time) * 1000.0) if _first_ts else None,
                decode_tps=None,
                tokens=_token_count,
            )
        raise
    except RuntimeError as _empty_exc:
        log.error("[%s] First stream attempt empty: %s", request_id[:8], _empty_exc)

    if chunks_yielded == 0 and body.thinking and not retry_attempted:
        retry_attempted = True
        log.warning("[%s] Empty thinking-mode response — retrying without thinking", request_id[:8])

        inf_cfg = req.app.state.ctx.get("cfg", {}).get("inference", {})
        prompt_no_think = _build_prompt(
            merged_messages, profile,
            suppress_thinking=inf_cfg.get("suppress_thinking", False),
            no_think_override=True,
            model_id=body.model,
        )
        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir_retry = InferenceRequest(
            request_id=f"{ir.request_id}_retry",
            model_id=ir.model_id,
            prompt=prompt_no_think,
            max_tokens=ir.max_tokens,
            temperature=ir.temperature,
            top_p=ir.top_p,
            top_k=ir.top_k,
            stop=ir.stop,
            stream=True,
            timeout_s=ir.timeout_s,
        )
        retry_source = optimizer.generate_stream(ir_retry) if optimizer else _toks.generate_stream(ir_retry)
        retry_source = _wrap_stream(
            lambda: retry_source,
            req,
            ir_retry,
            profile,
            ir_retry.max_tokens,
        )
        retry_text = ""
        try:
            async for chunk in validate_stream_response(retry_source, request_id=ir_retry.request_id):
                if await req.is_disconnected():
                    log.info(
                        "[%s] Client disconnected during thinking-mode stream "
                        "after %d tokens",
                        request_id[:8], _token_count,
                    )
                    # Persist the partial reply as an interrupted checkpoint
                    # so /resume works — this path is not covered by the
                    # segmented handler.
                    await _persist_after_stream(
                        conv_store=conv_store,
                        session_id=session_id,
                        body=body, ir=ir_retry, full=retry_text,
                        req=req, key_id=key_id, store=store,
                        start=start_time,
                        merged=merged_messages,
                        status=499,
                        path="/v1/chat/completions",
                        ttft_ms=((_first_ts - start_time) * 1000.0) if _first_ts else None,
                        decode_tps=None,
                        tokens=_token_count,
                    )
                    return
                if not validate_stream_chunk(chunk, ir_retry.request_id):
                    continue
                if stream_fmt == "json":
                    payload = json.dumps({
                        "id": ir_retry.request_id,
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [{"delta": {"content": chunk}, "finish_reason": None, "index": 0}],
                    })
                    yield f"data: {payload}\n\n".encode()
                else:
                    yield chunk.encode()
                if _first_ts is None:
                    _first_ts = time.monotonic()
                _token_count += 1
                retry_text += chunk
                chunks_yielded += 1
        except (asyncio.CancelledError, GeneratorExit):
            # Persist a checkpoint before propagating so /resume can pick up.
            with suppress(Exception):
                await _persist_after_stream(
                    conv_store=conv_store,
                    session_id=session_id,
                    body=body, ir=ir_retry, full=retry_text,
                    req=req, key_id=key_id, store=store,
                    start=start_time,
                    merged=merged_messages,
                    status=499,
                    path="/v1/chat/completions",
                    ttft_ms=((_first_ts - start_time) * 1000.0) if _first_ts else None,
                    decode_tps=None,
                    tokens=_token_count,
                )
            raise
        except RuntimeError as _empty_exc:
            log.error("[%s] First stream attempt empty: %s", request_id[:8], _empty_exc)

        full_text = retry_text

    full_text = ensure_response_not_empty(full_text, body.model, request_id)
    if conv_store and full_text and not _is_error_token(full_text):
        merged_messages.append({"role": "assistant", "content": full_text})
        with suppress(Exception):
            await conv_store.create(session_id, merged_messages)

    elapsed = time.monotonic() - start_time
    _ttft_ms = ((_first_ts - start_time) * 1000.0) if _first_ts else None
    _decode_s = elapsed - ((_first_ts - start_time) if _first_ts else 0.0)
    _decode_tps = (
        (_token_count - 1) / _decode_s
        if _decode_s > 0.05 and _token_count > 1 else None
    )
    await _log_request(
        req, key_id, store, "/v1/chat/completions",
        200, elapsed * 1000, len(full_text) // 3.5, request_id=request_id,
        ttft_ms=_ttft_ms, decode_tps=_decode_tps,
    )
    with suppress(Exception):
        asyncio.create_task(cleanup_after_generation(req.app.state.ctx))

    yield b"data: [DONE]\n\n"


# =============================================================================
# Web UI fallback
# =============================================================================
try:
    from local_llm_client.web.web_ui import HTML as _WEB_DEMO_HTML
except ImportError:
    try:
        from local_llm_server.serving.web_ui import HTML as _WEB_DEMO_HTML
    except ImportError:
        _WEB_DEMO_HTML = (
            "<html><body>"
            "<h1>Local LLM Cloud v{version}</h1>"
            "<p>Web UI package not installed.</p>"
            "<p>API docs: <a href='/docs'>/docs</a></p>"
            "<p>Health: <a href='/health'>/health</a></p>"
            "</body></html>"
        )


# =============================================================================
# BitNet loader helper
# =============================================================================
async def _load_bitnet_model(
    engine,
    loader,
    profile: dict,
    ctx: dict,
    timeout: float = 600.0,
) -> Any:
    from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
    from local_llm_core.engine.utils.config_loader import (
        BitNetSystemConfig,
        BitNetProfileConfig,
    )

    model_id = profile.get("id", "bitnet-model")
    path     = profile.get("path", "")

    cfg          = ctx.get("cfg", {})
    bitnet_yaml  = cfg.get("bitnet", {})

    try:
        bitnet_sys_cfg = BitNetSystemConfig(**bitnet_yaml)
    except Exception as e:
        log.warning(
            "BitNetSystemConfig from config.yaml failed (%s) — using safe CPU defaults.  "
            "Check your config.yaml [bitnet] section.",
            e,
        )
        bitnet_sys_cfg = BitNetSystemConfig(
            bitnet_n_gpu_layers=0,
            bitnet_force_backend=None,
        )

    try:
        bitnet_profile = BitNetProfileConfig(
            id                   = model_id,
            path                 = path,
            quantization         = profile.get("quantization", "1bit"),
            autoload             = profile.get("autoload", False),
            device               = profile.get("device", "cpu"),
            num_gpu_layers       = profile.get("num_gpu_layers", 0),
            use_flash_attention_2= profile.get("use_flash_attention_2", False),
            context_length       = profile.get("context_length", 4096),
            priority             = profile.get("priority", 5),
            tags                 = profile.get("tags", []),
            model_type           = profile.get("model_type", "bitnet"),
            prompt_format        = profile.get("prompt_format"),
            tokenizer_path       = profile.get("tokenizer_path"),
        )
    except Exception as e:
        log.warning(
            "BitNetProfileConfig build failed for '%s' (%s) — using minimal safe profile.",
            model_id, e,
        )
        bitnet_profile = BitNetProfileConfig(id=model_id, path=path)

    try:
        loader_sf = BitNetLoader.from_state_file()
        loader_sf.config = bitnet_sys_cfg
        wrapper = await asyncio.wait_for(
            loader_sf.load("state_model"), timeout=timeout
        )
        log.info(
            "BitNet '%s' loaded via state file (bitnet.cpp backend). "
            "port=%d threads=%d",
            model_id,
            bitnet_sys_cfg.bitnet_cpp_port,
            bitnet_sys_cfg.bitnet_n_threads,
        )
        return wrapper
    except Exception as exc:
        log.info(
            "BitNet state-file path unavailable for '%s' (%s) — "
            "falling through to direct-path load.",
            model_id, exc,
        )

    if path:
        bl = BitNetLoader(
            config          = bitnet_sys_cfg,
            model_profiles  = [bitnet_profile],
        )
        wrapper = await asyncio.wait_for(
            bl.load(model_id, force_reload=True), timeout=timeout
        )
        backend = getattr(wrapper, "force_backend", "unknown")
        log.info(
            "BitNet '%s' loaded from path '%s' (backend: %s).",
            model_id, path, backend,
        )
        return wrapper

    raise RuntimeError(
        f"Cannot load BitNet model '{model_id}': "
        f"state file unavailable and no path in profile. "
        f"Set models.profiles[{model_id!r}].path in config.yaml or run local-llm-setup."
    )


# =============================================================================
# App factory
# =============================================================================
def create_app(config_path: str = "config.yaml") -> FastAPI:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env_vars(cfg)

    # ── Compute draft model IDs for filtering ──────────────────────────────
    _profiles = cfg.get("models", {}).get("profiles", [])
    draft_model_ids = {p.get("id") for p in _profiles if "draft" in p.get("tags", [])}
    log.info("Draft models (will be hidden from user-facing listings): %s", draft_model_ids)

    _log_cfg = cfg.get("logging", {})
    _log_level_s = _log_cfg.get("level", "info").upper()
    _log_level = getattr(logging, _log_level_s, logging.INFO)

    root_log = logging.getLogger()
    for h in root_log.handlers[:]:
        root_log.removeHandler(h)
        h.close()

    root_log.setLevel(_log_level)

    _tui = TUILogHandler.install(config_path=config_path)
    _tui.setLevel(logging.DEBUG)
    root_log.addHandler(_tui)

    # Attach the RequestIDFilter to the root logger so any log record that
    # passes through a handler using a `%(request_id)s` formatter resolves
    # to the current request's ID (populated by ProductionMiddleware via
    # request_id_var) instead of raising KeyError. the filter was
    # defined in middleware.py but never installed. Both current handlers
    # hardcode their formats so this is defensive — but config.yaml's
    # logging.format field DOES contain `%(request_id)s`, so if that ever
    # gets wired to a live handler, it would raise on every line without
    # this call.
    install_request_id_filter()

    _log_file = _log_cfg.get("file", _default_state_dir() / "llm_server.log")
    _log_format = (_log_cfg.get("format") or "text").lower()
    if _log_file:
        try:
            fh = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
            fh.setLevel(_log_level)
            # honor logging.format. "json" gives structured,
            # machine-parseable log lines suitable for log shippers and
            # simple jq pipelines; "text" retains the human-readable
            # format used since v1.
            fh.setFormatter(create_formatter(_log_format))
            root_log.addHandler(fh)
            log.info("Log format: %s → %s", _log_format, _log_file)
        except Exception as exc:
            log.warning("Log file open failed: %s", exc)

    log.info("⚡ Local LLM Cloud v%s starting", VERSION)

    _api_cfg = cfg.get("api", {})
    _required_key = _api_cfg.get("key", "")
    if not _required_key and not _api_cfg.get("key_store_enabled", False):
        log.warning("⚠ api.key not set — server running WITHOUT authentication")

    _allowed_dir = cfg.get("models", {}).get("allowed_dir")
    bitnet_enabled = cfg.get("bitnet", {}).get("enabled", False)
    log.info("BitNet enabled: %s", bitnet_enabled)

    global admin_rate_limiter
    admin_rate_limiter = AdminRateLimiter(
        max_requests_per_hour=_api_cfg.get("admin_rate_limit", {}).get(
            "max_requests_per_hour", 1000
        )
    )

    ctx: Dict[str, Any] = {}
    ctx["draft_model_ids"] = draft_model_ids
    _loading_status: Dict[str, dict]              = {}
    _loading_futures: Dict[str, asyncio.Future]   = {}
    _loading_lock   = asyncio.Lock()
    _bg_tasks: Set[asyncio.Task]                  = set()
    _model_status: Dict[str, str]                 = {}
    _model_status_lock = asyncio.Lock()
    _external_models: Dict[str, Dict[str, Any]]   = {}
    _external_models_lock = asyncio.Lock()

    def _add_bg_task(coro) -> None:
        t = asyncio.create_task(coro)
        _bg_tasks.add(t)
        def _bg_done(task: asyncio.Task) -> None:
            _bg_tasks.discard(task)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                log.error(
                    "Unhandled exception in background task %r: %s",
                    task.get_name(), exc,
                    exc_info=exc,
                )
        t.add_done_callback(_bg_done)

    def validate_model_path(model_path: str) -> str:
        if not _allowed_dir:
            raise HTTPException(403, "models.allowed_dir not configured")
        return _validate_model_path_strict(model_path, _allowed_dir)

    def _register_bitnet_in_engine(engine, model_id: str, wrapper) -> None:
        if hasattr(engine, "register_model"):
            engine.register_model(model_id, wrapper)
        elif hasattr(engine, "_unified_models"):
            engine._unified_models[model_id] = wrapper
        elif hasattr(engine, "_models"):
            engine._models[model_id] = wrapper

    async def register_external_model(model_id: str, info: Dict[str, Any]) -> None:
        async with _external_models_lock:
            _external_models[model_id] = info

    async def unregister_external_model(model_id: str) -> bool:
        async with _external_models_lock:
            if model_id in _external_models:
                del _external_models[model_id]
                return True
        return False

    def _make_bitnet_info(profile: dict, wrapper) -> Dict[str, Any]:
        mem = {}
        with suppress(Exception):
            mem = wrapper.get_memory_usage() if hasattr(wrapper, "get_memory_usage") else {}
        return {
            "id":           profile.get("id", "unknown"),
            "object":       "model",
            "owned_by":     "local-llm-cloud",
            "model_type":   "bitnet",
            "path":         profile.get("path", ""),
            "quantization": profile.get("quantization", "1bit"),
            "vram_mb":      mem.get("vram_mb", 0),
            "strategy":     "bitnet",
            "status":       "ready",
        }

    async def _register_bitnet_in_loader(loader, model_id: str, profile: dict, wrapper) -> None:
        info = _make_bitnet_info(profile, wrapper)
        await register_external_model(model_id, info)

    async def _is_model_loaded(loader, engine, model_id: str) -> bool:
        if loader.get(model_id):
            return True
        for attr in ("_unified_models", "_models"):
            if model_id in getattr(engine, attr, {}):
                return True
        async with _external_models_lock:
            if model_id in _external_models:
                return True
        return False

    async def _get_all_models(loader, include_draft: bool = False) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            gguf_models = await asyncio.wait_for(
                loop.run_in_executor(None, loader.list_models), timeout=5.0
            )
        except Exception:
            gguf_models = []

        async with _external_models_lock:
            external = list(_external_models.values())

        all_models = gguf_models + external
        seen       = {m["id"] for m in all_models}

        def _norm_path(p: str) -> str:
            if not p:
                return ""
            try:
                return str(Path(p).expanduser().resolve())
            except Exception:
                return str(p)

        loaded_paths = {
            _norm_path(m.get("path", "")) for m in all_models if m.get("path")
        }

        try:
            from local_llm_core.engine.adaptive_awareness.model_unified_env import BitNetStateManager
            state = BitNetStateManager.load()
            _state_path = _norm_path(state.gguf_path) or _norm_path(state.model_path)
            _already_loaded = bool(_state_path) and _state_path in loaded_paths
            if (
                state.available
                and state.model_name
                and state.model_name not in seen
                and not _already_loaded
            ):
                all_models.append({
                    "id":         state.model_name,
                    "object":     "model",
                    "owned_by":   "local-llm-cloud",
                    "model_type": "bitnet",
                    "gguf_path":  state.gguf_path,
                    "path":       state.gguf_path,
                    "quant_type": state.quant_type,
                    "strategy":   "bitnet_cpp" if state.llama_server_binary else "bitnet_transformers",
                    "status":     "ready",
                })
                seen.add(state.model_name)
        except Exception:
            pass

        async with _model_status_lock:
            for m in all_models:
                mid = m["id"]
                m["status"] = _model_status.get(mid, "ready" if mid in seen else "loading")

        if not include_draft and ctx.get("draft_model_ids"):
            all_models = [m for m in all_models if m.get("id") not in ctx["draft_model_ids"]]

        return all_models

    def _profile_to_mc(profile: dict, hw_profile: Optional[HardwareProfile] = None):
        from local_llm_core.engine.gguf.model_loader import ModelConfig
        use_mlock = profile.get("use_mlock")
        if use_mlock is None and hw_profile is not None:
            use_mlock = getattr(hw_profile, "use_mlock", False)
        elif use_mlock is None:
            use_mlock = False
        return ModelConfig(
            model_path     = profile.get("path", ""),
            model_id       = profile.get("id", "unknown"),
            quantization   = profile.get("quantization", "auto"),
            context_length = profile.get("context_length", 0),
            num_gpu_layers = profile.get("num_gpu_layers", -1),
            n_threads      = profile.get("n_threads", 0),
            use_mmap       = profile.get("use_mmap", True),
            use_mlock      = use_mlock,
            rope_scaling   = profile.get("rope_scaling", 1.0),
            gpu_split      = profile.get("gpu_split", None),
        )

    # ── Lifespan ──────────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("╔══ Local LLM Cloud v%s starting (pid=%d) ══╗", VERSION, os.getpid())
        key_store = None

        try:
            from local_llm_core.compatibility.transformers_compat import StateFileReader
            StateFileReader.apply_spectre_pythonpath()
        except Exception:
            pass

        try:
            await startup_validation(cfg, config_path)

            # Hardware detection
            sys_info = None
            try:
                from local_llm_core.hal.hardware_detector import HardwareDetector
                det = HardwareDetector()
                sys_info = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, det.detect_all),
                    timeout=30.0,
                )
                log.info("GPU detection: %s (%d GPU(s))", sys_info.recommended_backend, len(sys_info.gpus))
            except Exception as exc:
                log.warning("⚠ GPU detection: %s — using CPU-only", exc)

            if sys_info is None:
                from local_llm_core.hal.hardware_detector import SystemInfo
                sys_info = SystemInfo(
                    cpu_model="unknown",
                    cpu_cores=psutil.cpu_count(logical=False) or 1,
                    cpu_threads=psutil.cpu_count(logical=True) or 1,
                    ram_total_mb=psutil.virtual_memory().total >> 20,
                    ram_free_mb=psutil.virtual_memory().available >> 20,
                    os_name=os.name, kernel_version="unknown", arch="unknown",
                    gpus=[], primary_gpu=None,
                    available_backends=["cpu"], recommended_backend="cpu",
                    has_cuda=False, has_rocm=False, has_opencl=False, has_vulkan=False,
                    vram_total_mb=0,
                )

            ctx["sys_info"] = sys_info
            ctx["cached_system_info"] = sys_info

            try:
                from local_llm_core.hal.system_optimizer import SystemOptimizer
                sys_cfg = cfg.get("system", {})
                sysopt = SystemOptimizer(
                    dry_run=not sys_cfg.get("allow_kernel_tuning", False),
                    config=sys_cfg,
                )
                sysopt.apply_all()
                ctx["sysopt"] = sysopt
            except Exception as exc:
                log.warning("SystemOptimizer: %s", exc)

            # Memory managers
            from local_llm_core.memory.vram_manager import VRAMManager
            from local_llm_core.memory.ram_optimizer import RAMOptimizer
            from local_llm_core.memory.swap_handler import SwapHandler
            from local_llm_core.memory.hybrid_manager import HybridMemoryManager

            mem_cfg     = cfg.get("memory", {})
            vram_target = float(mem_cfg.get("vram_utilization_target", 0.95))
            # Canonical RAM budget — same helper autoload uses below, so
            # every component in this process works against the same number
            ram_target  = _resolve_ram_target(cfg)
            log.info(
                "Memory budget: vram_target=%.2f ram_target=%.2f "
                "(source=memory.ram_utilization_target)",
                vram_target, ram_target,
            )
            total_vram  = max(
                sum(g.vram_total_mb for g in sys_info.gpus
                    if getattr(g, "supported", True)) * 1024 * 1024, 1
            )

            vram_mgr = VRAMManager(
                total_vram_bytes=total_vram,
                config={"memory": {"vram_target_utilization": vram_target}},
            )
            ram_mgr  = RAMOptimizer(
                max_ram_mb=int(sys_info.ram_total_mb * ram_target),
                compression=mem_cfg.get("compression_enabled", True),
            )
            swap_mgr = SwapHandler(config={
                "memory": {
                    "ssd_swap_path": mem_cfg.get("ssd_swap_path", _default_state_dir() / "llm_swap"),
                    "ssd_swap_max_gb": mem_cfg.get("ssd_swap_max_gb", 32),
                }
            })
            hybrid   = HybridMemoryManager(
                vram_bytes=total_vram,
                ram_bytes=sys_info.ram_total_mb * 1024 * 1024,
                config={"memory": {
                    "hybrid_ssd_path":   mem_cfg.get("ssd_swap_path", _default_state_dir() / "llm_swap"),
                    "hybrid_ssd_max_gb": mem_cfg.get("ssd_swap_max_gb", 32),
                }},
            )
            ctx.update(vram=vram_mgr, ram=ram_mgr, swap=swap_mgr, hybrid=hybrid)

            # Inference engine
            from local_llm_core.engine.gguf.model_loader import ModelLoader
            from local_llm_core.engine.inference.inference_engine import InferenceEngine, RateLimiter

            loader = ModelLoader(
                vram_mgr,
                ram_mgr,
                swap_mgr,
                sys_info.gpus,
                cfg,
                hybrid=hybrid,
                system_info=sys_info,
            )
            _hw_profile  = getattr(loader, "_profile", None) or getattr(loader, "profile", None)
            _default_tps = getattr(_hw_profile, "target_tps", 8.0) if _hw_profile else 8.0
            inf_cfg      = cfg.get("inference", {})

            _profiles    = cfg.get("models", {}).get("profiles", [])
            _al_all      = cfg.get("models", {}).get("autoload_all", False)
            _bitnet_profile_configured = any(
                (p.get("autoload") or _al_all) and
                p.get("model_type", "").lower() in ("bitnet", "1bit")
                for p in _profiles
            )

            from local_llm_server.monitoring.metrics_collector import ThermalState
            dummy_thermal = ThermalState(throttling=False, temperature_c=0.0, timestamp=0.0)
            ctx["thermal_state"] = dummy_thermal

            # kernels.enabled lives at the TOP LEVEL of config.yaml, not
            # inside [inference]. Read it here and pass explicitly — the
            # engine's `config=` argument is the [inference] section only.
            _kernels_cfg = cfg.get("kernels", {}) or {}
            _kernels_enabled = bool(_kernels_cfg.get("enabled", False))
            log.info("Kernels subsystem: enabled=%s", _kernels_enabled)
            engine = InferenceEngine(
                loader,
                max_batch_size             = inf_cfg.get("batch_size", 4),
                target_tps                 = inf_cfg.get("aimd_target_tps", _default_tps),
                config                     = inf_cfg,
                bitnet_profile_configured  = _bitnet_profile_configured,
                thermal_state              = ctx["thermal_state"],
                bitnet_enabled             = bitnet_enabled,
                kernels_enabled            = _kernels_enabled,
            )
            _rps = float(_api_cfg.get("api_key_rate_limit_rps", 10.0))
            InferenceEngine.rate_limiter = RateLimiter(default_rps=_rps)
            engine.start_background_tasks()

            # ── Concurrency pool (shared-weight process pool) ──────────────
            # Initialises only if concurrency.mode == "process_pool".
            # Everything else keeps using the single-process engine.
            _conc_cfg = cfg.get("concurrency", {}) or {}
            _conc_mode = str(_conc_cfg.get("mode", "process_pool")).lower()

            if _conc_mode == "process_pool":
                try:
                    from local_llm_core.accel.memory_ledger import MemoryLedger
                    from local_llm_core.accel.capabilities import probe_llama_cpp
                    from local_llm_server.serving.pool import (
                        SharedWeightProcessPool, CAASWeights,
                    )

                    _caps = probe_llama_cpp()

                    # Extract the primary autoload model's spec to build the pool
                    _primary_profile = None
                    for _p in cfg.get("models", {}).get("profiles", []):
                        if _p.get("autoload") and "draft" not in _p.get("tags", []):
                            _primary_profile = _p
                            break
                    if _primary_profile is None:
                        log.warning(
                            "concurrency.mode=process_pool but no autoload primary "
                            "profile is configured — pool stays idle"
                        )
                    else:
                        # Memory figures for the ledger
                        from local_llm_core.engine.pipelines.layer_streamer import (
                            read_gguf_meta,
                        )
                        from local_llm_core.engine.inference.stream_guards import (
                            kv_bytes_per_token,
                        )
                        from pathlib import Path as _P
                        _mp = _P(_primary_profile.get("path", ""))
                        _weights_mb = (
                            (_mp.stat().st_size >> 20) if _mp.exists() else 0
                        )
                        _meta = read_gguf_meta(_mp) if _mp.exists() else {}
                        _ctx_target = int(
                            _primary_profile.get("context_length")
                            or _hw_profile.context_length if _hw_profile else 8192
                        )
                        _kv_quant = (
                            _hw_profile.kv_cache_quant if _hw_profile else "q8_0"
                        )
                        _kv_bpt = int(kv_bytes_per_token(
                            int(_meta.get("n_layer") or 32),
                            int(_meta.get("n_kv_heads") or 8),
                            int(_meta.get("head_dim") or 128),
                            int(_meta.get("head_dim_v") or 0) or None,
                            kv_quant_k=_kv_quant,
                        ))
                        _kv_per_worker_mb = int(_kv_bpt * _ctx_target / 1_048_576)

                        # Measured BW, not theoretical
                        _bw = float(
                            getattr(sys_info, "ram_bandwidth_gbps_measured", 0.0) or 0.0
                        ) or 12.5

                        # iGPU carve-out (0 on discrete or CPU)
                        _igpu = 0
                        try:
                            if any(
                                (getattr(g.memory_type, "value", str(g.memory_type)) or "").upper()
                                in ("SHARED SYSTEM RAM", "UNIFIED MEMORY")
                                for g in sys_info.gpus if getattr(g, "supported", True)
                            ):
                                _igpu = int(
                                    getattr(sys_info, "vram_gtt_total_mb", 0) or 0
                                ) or 2048
                        except Exception:
                            _igpu = 0

                        _ledger = MemoryLedger(
                            weights_shared_mb=_weights_mb,
                            kv_per_worker_mb=_kv_per_worker_mb,
                            runtime_per_worker_mb=int(
                                _conc_cfg.get("runtime_per_worker_mb", 300)
                            ),
                            measured_bw_gbps=_bw,
                            igpu_carveout_mb=_igpu,
                            os_reserve_mb=int(_conc_cfg.get("os_reserve_mb", 1024)),
                            target_min_tps_per_worker=float(
                                _conc_cfg.get("target_min_tps_per_worker", 1.0)
                            ),
                            configured_max_workers=int(
                                _conc_cfg.get("max_workers", 0) or 0
                            ),
                        )

                        _load_spec = {
                            "model_id": _primary_profile["id"],
                            "model_path": _primary_profile["path"],
                            "load_kwargs": {
                                "quantization": _primary_profile.get("quantization", "auto"),
                                "context_length": _ctx_target,
                                "num_gpu_layers": _primary_profile.get("num_gpu_layers", -1),
                                "n_threads": _primary_profile.get("n_threads", 0),
                                "use_mmap": _primary_profile.get("use_mmap", True),
                                "use_mlock": _primary_profile.get("use_mlock", False),
                            },
                            "engine_config": cfg.get("inference", {}) or {},
                            "hardware_config": cfg.get("hardware", {}) or {},
                            "memory_config": cfg.get("memory", {}) or {},
                            "runtime": {"caps_version": _caps.version},
                        }

                        _pool = SharedWeightProcessPool(
                            load_spec=_load_spec,
                            router_weights=CAASWeights(
                                prefix=float(_conc_cfg.get("affinity_prefix_weight", 1.0)),
                                thermal=float(_conc_cfg.get("affinity_thermal_weight", 0.5)),
                                staleness=float(_conc_cfg.get("affinity_staleness_weight", 1.0)),
                                throttle_start_c=float(_conc_cfg.get("throttle_start_c", 85.0)),
                            ),
                            initial_workers=int(_conc_cfg.get("initial_workers", 1)),
                            max_workers=_ledger.max_workers(),
                            ram_refuse_pct=float(_conc_cfg.get("ram_refuse_pct", 90.0)),
                            global_queue_max=int(_conc_cfg.get("global_queue_max", 256)),
                            heartbeat_s=float(_conc_cfg.get("heartbeat_s", 0.5)),
                        )
                        await _pool.start()
                        ctx["pool"] = _pool
                        ctx["token_source"] = _pool
                        log.info(
                            "Pool ready: max_workers=%d (mode=%s)",
                            _ledger.max_workers(), _conc_mode,
                        )
                except Exception as _pool_exc:
                    log.error(
                        "Pool init failed (%s) — falling back to single-process engine",
                        _pool_exc, exc_info=True,
                    )

            # Fallback wiring: routes call ctx["token_source"] if present,
            # else ctx["engine"]. This is the ONLY change needed in routes.
            if "token_source" not in ctx:
                ctx["token_source"] = engine

            # Post‑kernel backend reconciliation
            try:
                import torch as _t
                _hip_active = _t.cuda.is_available() and hasattr(_t.version, "hip")
                if _hip_active and sys_info.recommended_backend not in ("rocm", "cuda"):
                    _old_be = sys_info.recommended_backend
                    sys_info.recommended_backend = "rocm"
                    sys_info.has_rocm = True
                    ctx["sys_info"]           = sys_info
                    ctx["cached_system_info"] = sys_info
                    log.info(
                        "✓ GPU: rocm (%d)  ← HIP runtime confirmed post-engine-init "
                        "(pre-init detection reported '%s')",
                        len(sys_info.gpus), _old_be,
                    )
                else:
                    log.info(
                        "✓ GPU: %s (%d)%s",
                        sys_info.recommended_backend,
                        len(sys_info.gpus),
                        "  [HIP/ROCm active]" if _hip_active else "",
                    )
            except Exception:
                log.info("✓ GPU: %s (%d)", sys_info.recommended_backend, len(sys_info.gpus))

            # Throughput optimizer
            _optimizer = None
            _draft_id  = inf_cfg.get("draft_model_id", "")
            _draft_profile = None
            for _p in cfg.get("models", {}).get("profiles", []):
                if _p.get("id") == _draft_id:
                    _draft_profile = _p
                    break
            _draft_autoloads = bool(
                _draft_profile and (
                    _draft_profile.get("autoload")
                    or cfg.get("models", {}).get("autoload_all", False)
                )
            )
            _draft_on_disk = bool(
                _draft_profile and Path(_draft_profile.get("path", "")).exists()
            )
            if (inf_cfg.get("throughput_optimizer", False)
                    and _draft_id
                    and _draft_autoloads
                    and _draft_on_disk):
                if ctx.get("pool") is not None:
                    log.info(
                        "ThroughputOptimizer skipped: process pool is the "
                        "concurrency mechanism. Disable "
                        "inference.throughput_optimizer to silence this line."
                    )
                else:
                    try:
                        from local_llm_core.engine.inference.throughput_optimizer import ThroughputOptimizer
                        _optimizer = ThroughputOptimizer(
                            engine=engine, draft_model_id=_draft_id,
                            target_model_id=inf_cfg.get("target_model_id", ""),
                            max_batch=inf_cfg.get("batch_size", 8),
                            spec_k=inf_cfg.get("spec_k", 8),
                            kv_cache_dir=mem_cfg.get("ssd_swap_path", _default_state_dir() / "llm_swap") + "/kv_cache",
                            kv_cache_max_gb=inf_cfg.get("kv_cache_max_gb", 10.0),
                        )
                        ctx["optimizer"] = _optimizer
                        log.info("ThroughputOptimizer: draft=%s spec_k=%d", _draft_id, inf_cfg.get("spec_k", 8))
                    except Exception as exc:
                        log.warning("ThroughputOptimizer unavailable: %s", exc)
            elif inf_cfg.get("throughput_optimizer", False) and _draft_id:
                log.info(
                    "ThroughputOptimizer disabled: draft=%s autoload=%s on_disk=%s — "
                    "enable models.profiles[%r].autoload and download the draft model to use it",
                    _draft_id, _draft_autoloads, _draft_on_disk, _draft_id,
                )

            # Telemetry
            from local_llm_server.monitoring.model_telemetry import ModelTelemetry, model_idle_reaper
            telemetry = ModelTelemetry()
            _tel_collector = None
            try:
                from local_llm_server.monitoring.telemetry import create_telemetry_collector
                _tel_collector = create_telemetry_collector(cfg)
                await _tel_collector.start()
            except Exception:
                pass

            ctx.update(
                loader=loader, engine=engine,
                hw_profile=_hw_profile, telemetry=telemetry,
                tel_collector=_tel_collector,
            )
            try:
                from local_llm_core.hal.gpu_analyzer import GPUAnalyzer
                ctx["analyzer"] = GPUAnalyzer(sys_info.gpus)
            except Exception:
                pass

            # Metrics
            from local_llm_server.monitoring.metrics_collector import MetricsCollector
            _mon_cfg = cfg.get("monitoring", {})
            metrics  = MetricsCollector(
                sample_interval     = _mon_cfg.get("metrics_interval_s", 5.0),
                prometheus_enabled  = _mon_cfg.get("prometheus_enabled", False),
                prometheus_port     = _mon_cfg.get("prometheus_port", 9090),
            )
            ctx["metrics"] = metrics
            ctx["thermal_state"] = metrics.thermal_state
            if hasattr(engine, "_scheduler") and engine._scheduler is not None:
                engine._scheduler._thermal_state = metrics.thermal_state
            if hasattr(metrics, "attach_engine"):
                metrics.attach_engine(engine.stats)

            # Memory pressure
            from local_llm_core.engine.pipelines.drive_pipeline import MemoryPressureMonitor
            pressure = MemoryPressureMonitor(vram_mgr, ram_mgr, swap_mgr, vram_target, ram_target)
            ctx["pressure"] = pressure
            ctx["cfg"]      = cfg
            ctx["config_path"] = config_path

            async def _pressure_loop():
                while True:
                    await asyncio.sleep(10)
                    try:
                        await pressure.check_and_enforce_async()
                    except asyncio.CancelledError:
                        break
                    except Exception as exc:
                        log.warning("pressure_loop: %s", exc)
            _add_bg_task(_pressure_loop())

            async def _smart_defrag():
                _iv = _mon_cfg.get("defrag_check_interval_s", 30.0)
                _qt = _mon_cfg.get("defrag_queue_threshold", 2)
                while True:
                    await asyncio.sleep(_iv)
                    try:
                        if engine.stats().get("queue", {}).get("pending", 0) < _qt:
                            await pressure.check_and_enforce_async()
                    except asyncio.CancelledError:
                        break
                    except Exception as exc:
                        log.warning("smart_defrag: %s", exc)
            _add_bg_task(_smart_defrag())

            await TUILogHandler.start_reaper_task()
            _add_bg_task(model_idle_reaper(
                loader, telemetry,
                idle_s          = _mon_cfg.get("idle_model_timeout_s", 1800.0),
                check_interval  = _mon_cfg.get("idle_reaper_interval_s", 60.0),
            ))

            # Autoload profiles (with reserved-first allocation)
            models_cfg   = cfg.get("models", {})
            autoload_all = models_cfg.get("autoload_all", False)
            profiles     = models_cfg.get("profiles", [])
            candidates   = [p for p in profiles if p.get("autoload") or autoload_all]

            reserved = [p for p in candidates if "draft" not in p.get("tags", [])]
            residual = [p for p in candidates if "draft" in p.get("tags", [])]
            reserved.sort(key=lambda p: p.get("priority", 100), reverse=True)
            residual.sort(key=lambda p: p.get("priority", 100), reverse=True)
            candidates = reserved + residual
            log.info("Autoload order: reserved (primary) first, then residual (draft)")

            def _bitnet_prereqs(profile: dict):
                model_path = profile.get("path", "")
                tok_path   = profile.get("tokenizer_path", "")
                is_gguf    = model_path.lower().endswith(".gguf")

                try:
                    from local_llm_core.engine.bitnet.bitnet_cpp_backend import (
                        BitNetCppBackend,
                    )
                    _bin = getattr(BitNetCppBackend, "_find_binary", lambda: None)()
                    if _bin:
                        return True, "bitnet_cpp", f"binary={_bin}"
                except Exception:
                    pass

                hf_dir = tok_path if (tok_path and Path(tok_path).is_dir()
                                      and (Path(tok_path) / "config.json").exists()) else None
                if not hf_dir and not is_gguf and Path(model_path).is_dir():
                    hf_dir = model_path
                if hf_dir:
                    try:
                        from local_llm_core.engine.bitnet.bitnet_engine import (
                            _BITLINEAR_AVAILABLE,
                        )
                        if _BITLINEAR_AVAILABLE:
                            return True, "transformers", f"hf_dir={hf_dir}"
                    except Exception:
                        pass

                if is_gguf:
                    return (
                        False, None,
                        "bitnet.cpp binary not built (run `local-llm-setup`) "
                        "and BitLinear not in transformers "
                        "(run `local-llm-setup --minimal-build` for pre-built binary)"
                    )
                return (
                    False, None,
                    "BitLinear not available — run: "
                    "pip install 'git+https://github.com/huggingface/transformers.git"
                    "@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4'"
                )

            async def _autoload():
                _al_timeout = models_cfg.get("autoload_timeout_s", 120.0)
                _warmup = models_cfg.get("warmup_enabled", True)
                total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
                # Same canonical budget as RAMOptimizer above.
                # Previously read memory.ram_target (0.85) while RAMOptimizer
                # read memory.ram_utilization_target (0.75), so autoloading
                # models could pass the RAM gate but then be unable to get
                # KV-cache memory from RAMOptimizer at runtime — silent
                # thrash under load.
                usable_mb = int(total_ram_mb * _resolve_ram_target(cfg))
                allocated_mb = 0

                # pool mode owns the model. The parent loader stays
                # empty on purpose — autoloading here would duplicate ~5 GB
                # of weights into the parent for zero benefit (workers
                # already have their own copy, shared via mmap page cache).
                if ctx.get("pool") is not None:
                    log.info(
                        "Parent autoload skipped (pool mode active): "
                        "pool workers own the model. Loader stays empty; "
                        "/v1/models proxies pool stats."
                    )
                    return

                for p in candidates:
                    model_id  = p.get("id", "unknown")
                    path      = p.get("path", "")
                    if not Path(path).exists():
                        log.warning("Autoload skip '%s': %s not found", model_id, path)
                        continue
                    req_mb = int(p.get("ram_required_mb") or (
                        Path(path).stat().st_size // (1024 * 1024)
                        if _detect_model_type(path) == "gguf"
                        else 3000
                    ))
                    if allocated_mb + req_mb > usable_mb:
                        log.info("Autoload skip '%s': need %d MB, %d MB free",
                                 model_id, req_mb, usable_mb - allocated_mb)
                        continue

                    if Path(path).is_file() and Path(path).suffix.lower() == ".gguf":
                        model_type = "gguf"
                    else:
                        model_type = p.get("model_type", _detect_model_type(path))

                    if model_type == "bitnet":
                        _ok, _tier, _reason = _bitnet_prereqs(p)
                        if not _ok:
                            async with _model_status_lock:
                                _model_status[model_id] = "pending_setup"
                            log.info(
                                "⏸  Autoload '%s': prerequisites not met — %s",
                                model_id, _reason,
                            )
                            continue
                        log.warning(
                            "Autoload '%s': prerequisites OK (tier=%s %s)",
                            model_id, _tier, _reason,
                        )

                    async with _model_status_lock:
                        _model_status[model_id] = "loading"
                    try:
                        if model_type == "bitnet":
                            wrapper = await asyncio.wait_for(
                                _load_bitnet_model(engine, loader, p, ctx), timeout=_al_timeout
                            )
                            _register_bitnet_in_engine(engine, model_id, wrapper)
                            await _register_bitnet_in_loader(loader, model_id, p, wrapper)
                        else:
                            mc = _profile_to_mc(p, hw_profile=_hw_profile)
                            await asyncio.wait_for(loader.load(mc), timeout=_al_timeout)

                        async with _model_status_lock:
                            _model_status[model_id] = "ready"
                        allocated_mb += req_mb
                        log.info("✓ Autoloaded: %s (%d MB)", model_id, req_mb)

                        if _warmup:
                            try:
                                from local_llm_server.monitoring.model_telemetry import warmup_model
                                _wup_engine = ctx.get("engine") or loader
                                await asyncio.wait_for(
                                    warmup_model(_wup_engine, model_id), timeout=30.0
                                )
                                log.info("✓ Warmed up: %s", model_id)
                            except Exception as exc:
                                log.warning("Warmup '%s': %s", model_id, exc)
                    except ImportError as exc:
                        async with _model_status_lock:
                            _model_status[model_id] = "setup_required"
                        _exc_first_line = str(exc).strip().splitlines()[0].strip() or str(exc)
                        log.warning(
                            "⚠  Autoload '%s': setup required — %s  "
                            "(run `local-llm-setup` or `local-llm-setup --minimal-build`)",
                            model_id, _exc_first_line,
                        )
                    except asyncio.TimeoutError:
                        async with _model_status_lock:
                            _model_status[model_id] = "timeout"
                        log.warning(
                            "⚠  Autoload '%s': timed out after %.0fs "
                            "(increase autoload_timeout_s in config.yaml if the model is large)",
                            model_id, _al_timeout,
                        )
                    except Exception as exc:
                        async with _model_status_lock:
                            _model_status[model_id] = "error"
                        log.error("✗  Autoload '%s': %s", model_id, exc)

            _add_bg_task(_autoload())

            # Conversation store
            conv_store = await create_conversation_store(cfg)
            ctx["conversation_store"] = conv_store
            if conv_store:
                async def _prune_loop():
                    while True:
                        await asyncio.sleep(3600)
                        try:
                            n = await conv_store.prune_old()
                            if n:
                                log.info("Pruned %d conversations", n)
                        except Exception as exc:
                            log.warning("prune: %s", exc)
                _add_bg_task(_prune_loop())

            # Key store
            if _api_cfg.get("key_store_enabled", False):
                key_store = await create_key_store(cfg)
                if not key_store:
                    raise RuntimeError("KeyStore enabled but failed to init")
                ctx["key_store"] = key_store
                log.info("✓ KeyStore enabled")

                _bs_cfg = _api_cfg.get("bootstrap_key", {})
                if _bs_cfg.get("enabled", False):
                    try:
                        _cfg_dir   = Path(config_path).resolve().parent
                        _sentinel  = _cfg_dir / ".bootstrap_key_done"
                        _keyfile   = _cfg_dir / "bootstrap_key.txt"
                        existing   = await key_store.list_keys()
                        if not existing and not _sentinel.exists():
                            _bs_id  = str(_bs_cfg.get("key_id", "default"))
                            _bs_rps = float(_bs_cfg.get("rate_limit_rps", 100.0))
                            _bs_cap = float(_bs_cfg.get("rate_limit_capacity", 1000.0))
                            plain_key, _ = await key_store.create_key(
                                key_id=_bs_id,
                                rate_limit_rps=_bs_rps,
                                rate_limit_capacity=_bs_cap,
                            )
                            _fd = os.open(
                                str(_keyfile), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                            )
                            with os.fdopen(_fd, "w") as fh:
                                fh.write(
                                    f"key_id={_bs_id}\n"
                                    f"api_key={plain_key}\n"
                                    f"created={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                )
                            _sentinel.write_text(
                                f"Bootstrap key '{_bs_id}' created on "
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')}. "
                                f"Delete this file to allow bootstrap to run again.\n"
                            )
                            log.warning(
                                "Bootstrap API key created (one-time). "
                                "Secret written to %s (mode 0600) — NOT logged here. "
                                "Read it, export LLM_API_KEY=<value>, then consider "
                                "deleting the file. Future keys: use api.master_key "
                                "with POST /admin/keys/create.",
                                _keyfile,
                            )
                        elif existing and not _sentinel.exists():
                            _sentinel.write_text(
                                "Bootstrap skipped: key store already had keys "
                                f"at startup on {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                            )
                    except FileExistsError:
                        log.warning("Bootstrap key file already exists — skipping")
                    except Exception as _bs_exc:
                        log.warning("KeyStore bootstrap failed: %s", _bs_exc)
            else:
                ctx["key_store"] = None
                log.info("KeyStore disabled (static API key mode)")

            # ── Suspend inhibitor ────────────────────────────────────────
            _inhibitor = None
            if cfg.get("power", {}).get("prevent_suspend", False):
                _inhibitor = SuspendInhibitor(who="local-llm-server")
                _inhibitor.start()
            ctx["_suspend_inhibitor"] = _inhibitor

            ctx["startup_error"] = None
            app.state._start_time = time.time()
            log.info("╚══ Ready  GPUs=%d  CPU=%dC/%dT  RAM=%dMB  profile=%s  auth=%s ══╝",
                     len(sys_info.gpus),
                     sys_info.cpu_cores,
                     sys_info.cpu_threads,
                     sys_info.ram_total_mb,
                     getattr(_hw_profile, "name", "unknown"),
                     "keystore" if key_store else ("static" if _required_key else "off"))

            # ── Runtime limits banner (deferred until autoload completes) ─
            # Now: the banner is emitted from a helper that waits for
            # _autoload() to finish (success or failure), then computes
            # the real limits. Falls back to a short-delay emission if
            # autoload is a no-op so the banner still shows up.
            async def _runtime_limits_banner_when_ready():
                try:
                    # Give autoload up to 15 minutes to finish (matches
                    # autoload_timeout_s=900s). A model that large on this
                    # hardware genuinely needs that long.
                    _loaded = False
                    for _ in range(900):  # 900 × 1s
                        try:
                            if ctx.get("engine") and ctx["engine"].get_loaded_model_ids():
                                _loaded = True
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(1)
                    if not _loaded:
                        log.info(
                            "Runtime limits banner: no model loaded within 15 min "
                            "— emitting banner with profile defaults"
                        )
                    _rl = _compute_runtime_limits(ctx)
                    log.info(
                        "Runtime limits: tier=%s decode_tps=%.2f prefill_tps=%.2f "
                        "ctx=%d resume_chunk=%d response=%d idle=%ss",
                        _rl["tier"], _rl["effective_tps"], _rl["prefill_tps"],
                        _rl["context_length"],
                        _rl["recommended_resume_chunk"],
                        _rl["recommended_response_tokens"],
                        _rl["recommended_idle_timeout_s"],
                    )
                except Exception as _rl_exc:
                    log.warning("Runtime limits banner failed: %s", _rl_exc)
            _add_bg_task(_runtime_limits_banner_when_ready())

            log.info("══" * 50)

            # ── systemd readiness + watchdog ─────────────────────────────
            # READY=1 tells systemd (with Type=notify) that the daemon has
            # finished initialising and can accept traffic. Must fire AFTER
            # the socket is bound and the last setup step has completed so
            # downstream After=local-llm-cloud.service units don't connect
            # before we can answer.
            #
            # NOTE: this fires at the END OF LIFESPAN, not when the model
            # finishes loading. Autoload runs as a background task AFTER
            # the yield (see _autoload() above). The /ready endpoint (not
            # the systemd signal) is the correct readiness check for load
            # balancers that need "model loaded". systemd's READY means
            # "port bound and accepting" — which is exactly what
            # After=local-llm-cloud.service consumers need.
            #
            # To make READY mean "model loaded", move the sd_notify("READY=1")
            # call into the autoload task, immediately after
            # `log.info("✓ Autoloaded: %s (%d MB)", ...)`. That trade-off is:
            #   + systemctl start returns only when the model can serve
            #   - systemctl start blocks ~10s even for an already-warm cache
            # Default here is the fast path; adjust if your deployment needs
            # strict semantics.
            sd_notify("READY=1")

            async def _watchdog_beat():
                # 15s < WatchdogSec/2. With WatchdogSec=60 systemd allows
                # 30s of silence before considering the unit hung, so a
                # single missed tick during GC or a long prefill cannot
                # trigger. MUST be driven from the event loop, not a
                # thread: a thread-driven heartbeat would keep systemd
                # convinced the process is healthy even while the loop
                # (which is what handles requests) is frozen — exactly
                # the class of bug that caused the P0 hang.
                while True:
                    try:
                        sd_notify("WATCHDOG=1")
                    except Exception:
                        pass
                    await asyncio.sleep(15)
            _add_bg_task(_watchdog_beat())

            # ── event-loop lag tracker ─────────────────────────
            # Measures scheduling delay in the asyncio loop. If the loop is
            # blocked (deadlock, sync I/O on the event thread, long GIL
            # hold), this value rises; healthy loops stay under 10ms. The
            # /metrics route exposes it as llm_event_loop_lag_seconds and
            # prometheus-alerts.yml fires an alert at 2s.
            ctx["_event_loop_lag_s"] = 0.0
            ctx.setdefault("_capacity_rejects", {
                "circuit_open": 0,
                "inflight": 0,
                "gate_waiters": 0,
            })

            async def _event_loop_lag_tracker():
                interval = 0.1
                while True:
                    t0 = time.monotonic()
                    await asyncio.sleep(interval)
                    actual = time.monotonic() - t0
                    lag = max(0.0, actual - interval)
                    prev = ctx.get("_event_loop_lag_s", 0.0)
                    # EWMA smoother — single spike doesn't trigger an alert,
                    # sustained lag does.
                    ctx["_event_loop_lag_s"] = 0.9 * prev + 0.1 * lag
            _add_bg_task(_event_loop_lag_tracker())

            yield

        except Exception as fatal_exc:
            log.critical("Fatal startup error: %s", fatal_exc, exc_info=True)
            ctx["startup_error"] = str(fatal_exc)
            ctx.setdefault("cfg", cfg)

            # ── Fail-fast vs degraded-mode ────────────────────────────────
            # Previously this block always yielded, keeping a half-broken
            # server alive. Consequences in production:
            #   * /health returns 200 with health_status="startup_failed"
            #     — a naive liveness probe never notices,
            #   * every real endpoint 503s,
            #   * systemd reports active(running),
            #   * the operator only finds out via a client 503 + log grep.
            #
            # Default is now fail-fast: raise → uvicorn logs "Application
            # startup failed", exits non-zero, systemd restarts. Set
            # server.fail_fast_on_startup_error: false in config.yaml to
            # restore the previous degraded-mode behaviour (useful on a
            # dev box only).
            _fail_fast = True
            try:
                _fail_fast = bool(
                    (cfg.get("server", {}) or {}).get(
                        "fail_fast_on_startup_error", True
                    )
                )
            except Exception:
                _fail_fast = True
            if _fail_fast:
                raise
            yield

        finally:
            log.info("Shutting down…")
            _sysopt = ctx.get("sysopt")
            if _sysopt is not None:
                with suppress(Exception):
                    _sysopt.shutdown()
            with suppress(Exception):
                await TUILogHandler.stop_reaper_task()
            _pool = ctx.get("pool")
            if _pool is not None:
                with suppress(Exception):
                    await asyncio.wait_for(_pool.shutdown(), timeout=60.0)
            for t in list(_bg_tasks):
                t.cancel()
            await asyncio.gather(*_bg_tasks, return_exceptions=True)
            _bg_tasks.clear()

            _inh = ctx.get("_suspend_inhibitor")
            if _inh is not None:
                _inh.stop()

            eng = ctx.get("engine")
            if eng:
                with suppress(Exception):
                    await asyncio.wait_for(eng.shutdown(), timeout=30.0)

            for k in ("metrics", "vram", "hybrid", "ram", "swap"):
                c = ctx.get(k)
                if c and hasattr(c, "shutdown"):
                    with suppress(Exception): c.shutdown()

            tel = ctx.get("tel_collector")
            if tel:
                with suppress(Exception): await tel.stop()

            ldr = ctx.get("loader")
            if ldr and hasattr(ldr, "unload_all"):
                with suppress(Exception): await ldr.unload_all()

            if key_store:
                with suppress(Exception):
                    await key_store.stop_background_tasks()
                    await key_store.close()

            log.info("Server stopped.")

    # ── FastAPI app ──────────────────────────────────────────────────────
    app = FastAPI(
        title       = "Local LLM Cloud",
        version     = VERSION,
        description = "Production LLM inference — OpenAI-compatible. GGUF + BitNet.",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )
    app.state.ctx    = ctx
    app.state.config = cfg
    app.state.server_start_time = time.time()

    # ── Exception handlers ────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def _val_exc(req: Request, exc: RequestValidationError) -> JSONResponse:
        return _json_error(
            422,
            "Request validation failed",
            request=req,
            detail=exc.errors(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(req: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _json_error(
            exc.status_code,
            str(exc.detail) if exc.detail else "HTTP error",
            request=req,
            detail=exc.detail if exc.status_code < 500 else None,
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(Exception)
    async def _generic_exc(req: Request, exc: Exception) -> JSONResponse:
        return _json_error(
            500,
            "Internal server error — please retry or contact the administrator",
            request=req,
            log_exc=exc,
        )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = cfg.get("server", {}).get("cors_origins", ["*"]),
        allow_methods     = ["*"],
        allow_headers     = ["*"],
        allow_credentials = True,
        expose_headers    = ["X-Request-ID", "X-Accel-Buffering", "X-Stream-Format",
                             "X-Prompt-Tokens", "X-Model-Selected"],
    )
    app.add_middleware(
        RequestSizeMiddleware,
        max_size=cfg.get("server", {}).get("max_body_size_mb", 100) * 1_000_000,
    )

    ip_rl = create_ip_rate_limiter(cfg)
    _trusted = validate_trusted_proxies(
        cfg.get("server", {}).get("trusted_proxies", [])
    ) or None
    app.add_middleware(ProductionMiddleware, rate_limiter=ip_rl, trusted_proxies=_trusted)

    public_limiter = IPRateLimiter(capacity=10.0, rate=0.1667)
    ready_limiter  = IPRateLimiter(capacity=5.0,  rate=0.0833)
    health_timeout = cfg.get("circuit_breaker", {}).get("health_timeout_s", 5.0)

    # ──────────────────────────────────────────────────────────────────────
    # System / health routes
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/")
    @app.get("/demo")
    async def web_demo(req: Request):
        allowed, _ = await public_limiter.allow(req.client.host if req.client else "0.0.0.0")
        if not allowed:
            raise HTTPException(429, "Too many requests")
        html = _WEB_DEMO_HTML.format(version=VERSION) if "{version}" in _WEB_DEMO_HTML else _WEB_DEMO_HTML
        # SECURITY: do NOT embed the static API key in the served
        # HTML. Previously:
        #     if _required_key:
        #         html = html.replace("__API_KEY_PLACEHOLDER__", _required_key)
        # — that made the key trivially obtainable by anyone who could
        # reach GET / or /demo (both unauthenticated). Clients must supply
        # the key themselves (X-API-Key header, or the UI's own prompt
        # stored in sessionStorage).
        # `_HTML_HEADERS` still carries no-store so the browser never
        # serves a stale SPA after a server-side update.
        return HTMLResponse(content=html, headers=_HTML_HEADERS)

    @app.get("/health")
    @with_timeout(health_timeout)
    async def health_check(req: Request):
        # /health now returns 503 when the health_status is not
        # "healthy". Previously it returned 200 with a body saying
        # "unhealthy" — an liveness probe (checking status_code only)
        # would pass while the DB was down. Liveness probes should use
        # /livez (always 200 if the process is up); readiness probes
        # should use /readyz. /health remains for backwards-compat and
        # full diagnostics.
        allowed, _ = await public_limiter.allow(req.client.host if req.client else "0.0.0.0")
        if not allowed:
            raise HTTPException(429, "Too many requests")
        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            return JSONResponse(status_code=503, content={
                "status":        "error",
                "health_status": "startup_failed",
                "startup_error": startup_error,
                "version":       VERSION,
                "timestamp":     time.time(),
            })
        status = "healthy"
        checks: Dict[str, str] = {
            "api_server":       "ok",
            "model_loader":     "unknown",
            "inference_engine": "unknown",
            "memory":           "unknown",
            "database":         "unknown",
        }
        loader = req.app.state.ctx.get("loader")
        if loader:
            try:
                bk = getattr(loader, "loader_breaker_stats", lambda: {"state": "closed"})()
                checks["model_loader"] = "degraded" if bk.get("state") == "open" else "ok"
            except Exception:
                checks["model_loader"] = "error"; status = "degraded"
        else:
            checks["model_loader"] = "missing"; status = "degraded"

        engine = req.app.state.ctx.get("engine")
        if engine:
            try:
                s = engine.stats()
                checks["inference_engine"] = (
                    "degraded"
                    if s.get("circuit_breaker", {}).get("state") == "open"
                    else "ok"
                )
            except Exception:
                checks["inference_engine"] = "error"; status = "degraded"
        else:
            checks["inference_engine"] = "missing"; status = "degraded"

        vram = req.app.state.ctx.get("vram")
        if vram:
            try:
                checks["memory"] = "critical" if vram.stats.utilization_pct > 95 else "ok"
            except Exception:
                checks["memory"] = "error"
        else:
            checks["memory"] = "missing"

        db_ok    = True
        db_error = None
        key_store = req.app.state.ctx.get("key_store")
        if key_store:
            try:
                await key_store.list_keys(active_only=True)
                checks["database"] = "ok"
            except Exception as exc:
                checks["database"] = "error"
                db_error = str(exc)
                db_ok    = False
                status   = "unhealthy"
        else:
            checks["database"] = "disabled"

        bitnet_info: Dict[str, Any] = {}
        try:
            from local_llm_core.engine.adaptive_awareness.model_unified_env import (
                auto_detect_backend, auto_detect_backend_info,
            )
            checks["bitnet_backend"] = auto_detect_backend()
            bitnet_info              = auto_detect_backend_info()
        except Exception as exc:
            checks["bitnet_backend"] = f"error: {exc}"

        body = {
            "status":         "ok" if status == "healthy" else "degraded",
            "health_status":  status,
            "version":        VERSION,
            "timestamp":      time.time(),
            "checks":         checks,
            "database_error": db_error,
            "bitnet":         bitnet_info,
        }
        # 503 when NOT healthy → strict liveness probes fail fast.
        http_status = 200 if status == "healthy" else 503
        return JSONResponse(status_code=http_status, content=body)

    @app.get("/livez")
    async def livez(req: Request):
        """Liveness: the process is running and the event loop responds.

        Always returns 200 if we get here at all. A deadlock or crash will
        fail at the TCP connection layer (or trigger the systemd watchdog),
        not here — so this probe is safe to use in `livenessProbe` /
        `Restart=on-failure` scenarios where any restart on transient
        degradation would be counter-productive.
        """
        return {"status": "alive", "uptime_s": round(
            time.time() - getattr(req.app.state, "server_start_time", time.time()), 1,
        )}

    @app.get("/readyz")
    @with_timeout(health_timeout)
    async def readyz(req: Request):
        """Readiness: a model can serve traffic right now.

        Pool mode: workers own the model; the parent loader is intentionally
        empty. Check the pool instead.
        """
        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Startup failed: {startup_error}")

        _pool = req.app.state.ctx.get("pool")
        if _pool is not None:
            try:
                _st = _pool.stats()
                _workers = _st.get("workers") or []
                _live = [w for w in _workers if w.get("alive")]
                if not _live:
                    raise HTTPException(503, "No live pool workers")
                if _pool._supervisor.all_degraded():
                    raise HTTPException(503, "All pool workers degraded")
                return {
                    "status": "ready",
                    "mode": "process_pool",
                    "workers": len(_live),
                    "max_workers": _st.get("max_workers", len(_live)),
                    "queue_depth": _st.get("queue_depth", 0),
                }
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(503, f"Pool not ready: {exc}")

        loader = req.app.state.ctx.get("loader")
        if not loader:
            raise HTTPException(503, "Loader not initialized")
        all_m  = await _get_all_models(loader, include_draft=False)
        ready  = [m for m in all_m if m.get("status") == "ready"]
        if not ready:
            raise HTTPException(503, "No models loaded and ready")
        bk = getattr(loader, "loader_breaker_stats", lambda: {"state": "closed"})()
        if bk.get("state") == "open":
            raise HTTPException(503, "Loader circuit breaker open")
        return {"status": "ready", "models": len(ready), "mode": "single_process"}

    @app.get("/metrics")
    async def prometheus_metrics(req: Request):
        """Prometheus exposition .

        Serves `text/plain; version=0.0.4` format directly. No auth —
        Prometheus is expected to scrape from 127.0.0.1 (or via a
        trusted-proxy chain). If you need auth, put this behind the
        reverse proxy and use a bearer token, or check
        `request.client.host` here.

        Kept outside the `PrometheusMiddleware` (which
        `prometheus_enabled` would otherwise spin up): this route is
        intentionally dependency-free and works even when the metrics
        collector is broken.
        """
        ctx = req.app.state.ctx
        engine = ctx.get("token_source") or ctx.get("engine")
        lag = float(ctx.get("_event_loop_lag_s", 0.0) or 0.0)
        rejects = ctx.get("_capacity_rejects") or {}
        started = getattr(req.app.state, "server_start_time", time.time())

        lines: List[str] = []

        def emit(name: str, typ: str, help_text: str, value) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {typ}")
            lines.append(f"{name} {value}")

        # Uptime + loop lag
        emit("llm_uptime_seconds", "gauge",
             "Seconds since server start",
             f"{time.time() - started:.3f}")
        emit("llm_event_loop_lag_seconds", "gauge",
             "EWMA of event-loop scheduling delay (seconds; healthy < 0.01)",
             f"{lag:.4f}")

        # System
        try:
            vm = psutil.virtual_memory()
            emit("llm_ram_used_bytes",    "gauge", "RAM used",    str(vm.used))
            emit("llm_ram_total_bytes",   "gauge", "RAM total",   str(vm.total))
            emit("llm_ram_percent",       "gauge", "RAM percent", f"{vm.percent:.1f}")
            emit("llm_cpu_percent",       "gauge", "CPU percent",
                 f"{psutil.cpu_percent(interval=0):.1f}")
        except Exception:
            pass

        # Engine
        if engine is not None:
            try:
                s = engine.stats()
                emit("llm_requests_total",   "counter", "Total inference requests",
                     str(s.get("total_requests", 0)))
                emit("llm_tokens_total",     "counter", "Total tokens generated",
                     str(s.get("total_tokens", 0)))
                emit("llm_errors_total",     "counter", "Total errors",
                     str(s.get("total_errors", 0)))
                emit("llm_avg_ttft_ms",      "gauge",   "Average TTFT (ms)",
                     str(s.get("avg_ttft_ms", 0)))
                emit("llm_avg_tps",          "gauge",   "Average tokens/sec",
                     str(s.get("avg_tps", 0)))
                emit("llm_decode_tps",       "gauge",   "Decode-only tokens/sec",
                     str(s.get("decode_tps", 0)))
                emit("llm_gate_waiters",     "gauge",   "Requests waiting at gate",
                     str(s.get("gate_waiters", 0)))
                emit("llm_inflight",         "gauge",   "Requests in flight",
                     str(s.get("inflight", 0)))
                emit("llm_slots_in_use",     "gauge",   "Slots in use",
                     str(s.get("slots_in_use", 0)))
                emit("llm_slots_capacity",   "gauge",   "Slot capacity",
                     str(s.get("slot_capacity", 0)))
                cb_state = s.get("circuit_breaker", {}).get("state", "closed")
                cb_val = 1 if cb_state == "open" else (0.5 if cb_state == "half-open" else 0)
                emit("llm_circuit_breaker_state", "gauge",
                     "0=closed 0.5=half-open 1=open", str(cb_val))
            except Exception as exc:
                log.warning("metrics: engine.stats failed: %s", exc)

        # Emit per-reason with a label
        lines.append("# HELP llm_capacity_rejected_total Pre-flight rejections by reason")
        lines.append("# TYPE llm_capacity_rejected_total counter")
        for reason, count in rejects.items():
            lines.append(f'llm_capacity_rejected_total{{reason="{reason}"}} {int(count)}')

        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/ready")
    @with_timeout(health_timeout)
    async def ready(req: Request):
        allowed, _ = await ready_limiter.allow(
            req.client.host if req.client else "0.0.0.0"
        )
        if not allowed:
            raise HTTPException(429, "Too many requests")
        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Startup failed: {startup_error}")

        _pool = req.app.state.ctx.get("pool")
        if _pool is not None:
            try:
                _st = _pool.stats()
                _workers = _st.get("workers") or []
                _live = [w for w in _workers if w.get("alive")]
                if not _live:
                    raise HTTPException(503, "No live pool workers")
                if _pool._supervisor.all_degraded():
                    raise HTTPException(503, "All pool workers degraded")
                return {
                    "status": "ready",
                    "mode": "process_pool",
                    "workers": len(_live),
                    "max_workers": _st.get("max_workers", len(_live)),
                    "queue_depth": _st.get("queue_depth", 0),
                }
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(503, f"Pool not ready: {exc}")

        loader = req.app.state.ctx.get("loader")
        if not loader:
            raise HTTPException(503, "Loader not initialized")
        all_m  = await _get_all_models(loader, include_draft=False)
        ready  = [m for m in all_m if m.get("status") == "ready"]
        if not ready:
            raise HTTPException(503, "No models loaded and ready")
        bk = getattr(loader, "loader_breaker_stats", lambda: {"state": "closed"})()
        if bk.get("state") == "open":
            raise HTTPException(503, "Loader circuit breaker open")
        return {"status": "ready", "models": len(ready), "mode": "single_process"}

    @app.get("/v1/me")
    async def get_key_info(key_info: tuple = Depends(verify_api_key)):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        if not store or not key_id:
            return {"key_id": "default", "rate_limit_rps": 200.0, "rate_limit_capacity": 2000.0,
                    "total_requests": 0, "total_tokens": 0}
        stats = await store.get_key_stats(key_id)
        return stats or {"key_id": key_id, "rate_limit_rps": 200.0, "total_requests": 0}

    @app.get("/v1/system")
    @with_timeout(2.0)
    async def system_info(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        info = req.app.state.ctx.get("cached_system_info") or req.app.state.ctx.get("sys_info")
        if not info:
            return {"error": "system info unavailable"}
        vm = psutil.virtual_memory()
        gpus = []
        for g in (info.gpus or []):
            if getattr(g, "supported", True):
                gpus.append({
                    "name":       g.name,
                    "vendor":     g.vendor.name if hasattr(g.vendor, "name") else str(g.vendor),
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb":  getattr(g, "vram_free_mb", 0),
                    "backend":    g.backend,
                    "driver":     g.driver_version,
                    "temperature_c": getattr(g, "temperature_c", None),
                    "supported":  True,
                })
        return {
            "cpu_model":           info.cpu_model,
            "cpu_cores":           info.cpu_cores,
            "cpu_threads":         info.cpu_threads,
            "ram_total_mb":        info.ram_total_mb,
            "ram_free_mb":         int(vm.available / 1024 / 1024),
            "ram_pct":             round(vm.percent, 1),
            "os_name":             info.os_name,
            "kernel_version":      info.kernel_version,
            "arch":                info.arch,
            "recommended_backend": getattr(info, "recommended_backend", "unknown"),
            "available_backends":  getattr(info, "available_backends", []),
            "has_cuda":            getattr(info, "has_cuda", False),
            "has_rocm":            getattr(info, "has_rocm", False),
            "gpus":                gpus,
            "hw_profile":          getattr(req.app.state.ctx.get("hw_profile"), "name", "unknown"),
        }

    @app.get("/v1/hardware/backends")
    async def backends(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            from local_llm_core.engine.pipelines.layer_streamer import BackendProbe
            return {"best": BackendProbe.best(), "available": BackendProbe.all_available()}
        except Exception as exc:
            return {"best": "cpu", "available": ["cpu"], "error": str(exc)}

    @app.get("/v1/logs")
    async def get_logs(
        n:        int   = Query(100, ge=1, le=500),
        since:    float = Query(0.0),
        key_info: tuple = Depends(verify_api_key),
    ):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        return {"entries": TUILogHandler.recent(n=n, since=since)}

    @app.websocket("/v1/ws/logs")
    async def ws_logs(ws: WebSocket, key_info: tuple = Depends(verify_api_key)):
        # Auth via the SAME dependency as REST endpoints. Previously this
        # handler used its own `if _required_key and api_key != _required_key`
        # check, which had two bugs:
        #   1. Fail-OPEN when _required_key == "" (the state reached after
        #      the static key is moved to env vars / key store). Anyone
        #      who could reach the port got the full log stream.
        #   2. Divergent from HTTP — a client with a valid key-store key
        #      could not authenticate to the WS because the WS required
        #      the STATIC key from config.yaml, which may not exist.
        # Using verify_api_key fixes both: same store, same fail-closed
        # semantics, same PUBLIC_PATHS rules (WS path is not public).
        is_valid, _, _ = key_info
        if not is_valid:
            await ws.close(code=1008, reason="Unauthorized")
            return
        await ws.accept()
        q = TUILogHandler.add_ws_client()
        try:
            for e in TUILogHandler.recent(n=50):
                msg = json.dumps(e)
                if len(msg) > 65536:
                    msg = msg[:65536]
                await asyncio.wait_for(ws.send_text(msg), timeout=2.0)
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.1)
                    if len(msg) > 65536:
                        msg = msg[:65536]
                    await ws.send_text(msg)
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            TUILogHandler.remove_ws_client(q)

    @app.get("/v1/config/models")
    async def config_models(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        c = req.app.state.ctx.get("cfg", {})
        return {"autoload_all": c.get("models", {}).get("autoload_all", False),
                "profiles":     c.get("models", {}).get("profiles", [])}

    @app.get("/v1/models")
    async def list_models(
        req: Request,
        key_info: tuple = Depends(verify_api_key),
        show_internal: bool = Query(False, description="Include internal/draft models"),
    ):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        loader = req.app.state.ctx.get("loader")
        if not loader:
            return {"object": "list", "data": []}
        loaded = await _get_all_models(loader, include_draft=show_internal)
        loaded_ids = {m["id"] for m in loaded}
        loading_entries = []
        async with _model_status_lock:
            for mid, status in _model_status.items():
                if status in ("loading", "warming_up") and mid not in loaded_ids:
                    if not show_internal and mid in ctx.get("draft_model_ids", set()):
                        continue
                    loading_entries.append({
                        "id": mid,
                        "object": "model",
                        "status": status,
                        "ready": False,
                        "message": (
                            "Model is loading in the background. "
                            "Requests will be accepted once loading completes."
                        ),
                    })
        return {"object": "list", "data": loaded + loading_entries}

    @app.get("/v1/models/current")
    async def current_model(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        loader = req.app.state.ctx.get("loader")
        if not loader:
            raise HTTPException(503, "Server not ready")
        all_m = await _get_all_models(loader, include_draft=False)
        ready = [m for m in all_m if m.get("status") == "ready"]
        if not ready:
            raise HTTPException(503, "No model loaded and ready")
        return {"object": "list", "data": ready, "ready": True}

    @app.get("/v1/queue/status")
    async def queue_status(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        engine = req.app.state.ctx.get("engine")
        if not engine:
            return {"queue_depth": 0, "gate_waiters": 0, "server_busy": False, "estimated_wait_s": 0.0}
        s     = engine.stats()
        depth = s.get("gate_waiters", 0)
        avg   = s.get("avg_tps", 0.0)
        tr    = s.get("total_requests", 1)
        tk    = s.get("total_tokens",   0)
        atpr  = tk / tr if tr > 0 else 100
        est   = min(depth * atpr / avg, 600.0) if avg > 0.1 else min(depth * 30.0, 600.0)
        return {"queue_depth": depth, "gate_waiters": depth, "server_busy": depth > 0,
                "estimated_wait_s": round(est, 1)}

    # ──────────────────────────────────────────────────────────────────────
    # Runtime limits — public, key-less, hardware + model aware.
    #
    # This endpoint is the single source of truth for every dynamic limit
    # the client needs to honour. It is intentionally key-less so the Web
    # UI can fetch it during first paint, before the user has entered an
    # API key. Nothing here is sensitive: no prompts, no conversations,
    # no key material. Just CPU/GPU/RAM numbers and a tier recommendation.
    # ──────────────────────────────────────────────────────────────────────
    @app.get("/v1/runtime/limits")
    async def runtime_limits_ep(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            return _compute_runtime_limits(req.app.state.ctx)
        except Exception as exc:
            log.warning("runtime_limits computation failed: %s", exc)
            # Never 5xx on a read-only endpoint: return the safe defaults
            # so the client can still function, and let them refresh on
            # the next tick.
            return {
                "version":                     VERSION,
                "tier":                        "small",
                "hardware_profile":            "unknown",
                "backend":                     "cpu",
                "model_id":                    "",
                "model_size_mb":               0,
                "context_length":              2048,
                "context_safety_tokens":       256,
                "min_response_tokens":         64,
                "segment_size":                512,
                "max_segments":                200,
                "recommended_response_tokens": 1024,
                "recommended_resume_chunk":    192,
                "recommended_idle_timeout_s":  300,
                "recommended_n_parallel":      1,
                "measured_tps":                0.0,
                "target_tps":                  8.0,
                "effective_tps":               8.0,
                "vram_total_mb":               0,
                "vram_free_mb":                0,
                "ram_total_mb":                0,
                "ram_free_mb":                 0,
                "error":                       str(exc),
            }

    @app.post("/v1/models/load", status_code=202)
    async def load_model(
        body:             LoadModelRequest,
        req:              Request,
        background_tasks: BackgroundTasks,
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        validate_model_path(body.model_path)
        if not Path(body.model_path).exists():
            raise HTTPException(404, f"Model file not found: {body.model_path}")

        model_type = body.model_type
        if model_type == "auto":
            model_type = _detect_model_type(body.model_path)
        if Path(body.model_path).is_file() and Path(body.model_path).suffix.lower() == ".gguf":
            model_type = "gguf"
        if model_type == "unknown":
            raise HTTPException(400, f"Unknown model type: {body.model_path}")

        loader = req.app.state.ctx["loader"]
        engine = req.app.state.ctx["engine"]

        async with _loading_lock:
            if await _is_model_loaded(loader, engine, body.model_id):
                return {"status": "already_loaded", "model_id": body.model_id}
            if body.model_id in _loading_status:
                return {"status": "loading", "model_id": body.model_id}
            loop           = asyncio.get_running_loop()
            future         = loop.create_future()
            _loading_futures[body.model_id] = future
            _loading_status[body.model_id]  = {"status": "loading", "start_time": time.time()}
            async with _model_status_lock:
                _model_status[body.model_id] = "loading"

        from local_llm_core.engine.gguf.model_loader import ModelConfig
        mc = ModelConfig(
            model_path     = body.model_path, model_id=body.model_id,
            quantization   = body.quantization, context_length=body.context_length,
            num_gpu_layers = body.num_gpu_layers, n_threads=body.n_threads,
        )

        async def _load_bg():
            try:
                if model_type == "bitnet":
                    wrapper = await _load_bitnet_model(engine, loader, {
                        "id": body.model_id, "path": body.model_path,
                        "quantization": body.quantization,
                        "n_threads": body.n_threads,
                    }, req.app.state.ctx)
                    _register_bitnet_in_engine(engine, body.model_id, wrapper)
                    await _register_bitnet_in_loader(loader, body.model_id, {
                        "id": body.model_id, "path": body.model_path,
                        "quantization": body.quantization,
                    }, wrapper)
                    m = wrapper
                else:
                    hw_profile = req.app.state.ctx.get("hw_profile")
                    mc = _profile_to_mc({
                        "path": body.model_path,
                        "id": body.model_id,
                        "quantization": body.quantization,
                        "context_length": body.context_length,
                        "num_gpu_layers": body.num_gpu_layers,
                        "n_threads": body.n_threads,
                    }, hw_profile)
                    m = await loader.load(mc)
                async with _loading_lock:
                    _loading_status[body.model_id] = {"status": "loaded", "model": m}
                async with _model_status_lock:
                    _model_status[body.model_id] = "ready"
                if not future.done():
                    future.set_result(m)
            except Exception as exc:
                async with _loading_lock:
                    _loading_status[body.model_id] = {"status": "failed", "error": str(exc)}
                async with _model_status_lock:
                    _model_status[body.model_id] = "error"
                if not future.done():
                    future.set_exception(exc)
            finally:
                await asyncio.sleep(60)
                async with _loading_lock:
                    _loading_status.pop(body.model_id, None)
                    _loading_futures.pop(body.model_id, None)

        background_tasks.add_task(_load_bg)
        return {"status": "loading", "model_id": body.model_id, "model_type": model_type,
                "poll": f"/v1/models/loading/{body.model_id}"}

    @app.get("/v1/models/loading/{model_id}")
    async def get_load_status(model_id: str, req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        if req.app.state.ctx["loader"].get(model_id):
            return {"status": "loaded", "model_id": model_id}
        async with _loading_lock:
            s = _loading_status.get(model_id)
        if not s:
            raise HTTPException(404, f"No load record for '{model_id}'")
        if s["status"] == "loading":
            return {"status": "loading", "model_id": model_id,
                    "elapsed_seconds": round(time.time() - s["start_time"], 1)}
        if s["status"] == "failed":
            return {"status": "failed",  "model_id": model_id, "error": s.get("error", "?")}
        return {"status": "loaded", "model_id": model_id}

    @app.post("/v1/models/load-profile/{profile_id}", status_code=202)
    async def load_by_profile(
        profile_id:       str,
        req:              Request,
        background_tasks: BackgroundTasks,
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        c        = req.app.state.ctx.get("cfg", {})
        profiles = c.get("models", {}).get("profiles", [])
        p        = next((x for x in profiles if x.get("id") == profile_id), None)
        if not p:
            raise HTTPException(404, f"Profile '{profile_id}' not found")
        validate_model_path(p.get("path", ""))
        if not Path(p["path"]).exists():
            raise HTTPException(404, f"Path not found: {p['path']}")

        if Path(p["path"]).is_file() and Path(p["path"]).suffix.lower() == ".gguf":
            model_type = "gguf"
        else:
            model_type = p.get("model_type", _detect_model_type(p["path"]))
        model_id   = p.get("id", "unknown")
        loader     = req.app.state.ctx["loader"]
        engine     = req.app.state.ctx["engine"]

        if await _is_model_loaded(loader, engine, model_id):
            return {"status": "already_loaded", "model_id": model_id}

        async with _loading_lock:
            if model_id in _loading_status:
                return {"status": "loading", "model_id": model_id}
            loop   = asyncio.get_running_loop()
            future = loop.create_future()
            _loading_futures[model_id] = future
            _loading_status[model_id]  = {"status": "loading", "start_time": time.time()}
            async with _model_status_lock:
                _model_status[model_id] = "loading"

        async def _load_profile_bg():
            try:
                if model_type == "bitnet":
                    wrapper = await _load_bitnet_model(engine, loader, p, req.app.state.ctx)
                    _register_bitnet_in_engine(engine, model_id, wrapper)
                    await _register_bitnet_in_loader(loader, model_id, p, wrapper)
                    m = wrapper
                else:
                    hw_profile = req.app.state.ctx.get("hw_profile")
                    mc = _profile_to_mc(p, hw_profile)
                    m  = await loader.load(mc)
                async with _loading_lock:
                    _loading_status[model_id] = {"status": "loaded", "model": m}
                async with _model_status_lock:
                    _model_status[model_id] = "ready"
                if not future.done():
                    future.set_result(m)
            except Exception as exc:
                async with _loading_lock:
                    _loading_status[model_id] = {"status": "failed", "error": str(exc)}
                async with _model_status_lock:
                    _model_status[model_id] = "error"
                if not future.done():
                    future.set_exception(exc)
            finally:
                await asyncio.sleep(60)
                async with _loading_lock:
                    _loading_status.pop(model_id, None)
                    _loading_futures.pop(model_id, None)

        background_tasks.add_task(_load_profile_bg)
        return {"status": "loading", "model_id": model_id, "model_type": model_type,
                "poll": f"/v1/models/loading/{model_id}"}

    @app.delete("/v1/models/{model_id}")
    async def unload_model(model_id: str, req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        ok = await req.app.state.ctx["loader"].unload(model_id)
        if ok:
            async with _model_status_lock:
                _model_status.pop(model_id, None)
            return {"status": "unloaded", "model_id": model_id}
        engine = req.app.state.ctx["engine"]
        for attr in ("_unified_models", "_models"):
            getattr(engine, attr, {}).pop(model_id, None)
        removed = await unregister_external_model(model_id)
        if removed:
            async with _model_status_lock:
                _model_status.pop(model_id, None)
            return {"status": "unloaded", "model_id": model_id}
        raise HTTPException(404, f"'{model_id}' not loaded")

    @app.post("/v1/models/{model_id}/swap")
    async def hot_swap(
        model_id:         str,
        body:             LoadModelRequest,
        req:              Request,
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        validate_model_path(body.model_path)
        from local_llm_core.engine.gguf.model_loader import ModelConfig
        mc = ModelConfig(
            model_path=body.model_path, model_id=body.model_id,
            quantization=body.quantization, context_length=body.context_length,
            num_gpu_layers=body.num_gpu_layers,
        )
        m = await req.app.state.ctx["loader"].hot_swap(model_id, mc)
        return {"status": "swapped", "new_model_id": m.config.model_id}

    @app.get("/v1/model-status")
    async def model_status_dashboard(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        loader = req.app.state.ctx.get("loader")
        if not loader:
            return {"models": []}
        return {"models": await _get_all_models(loader, include_draft=False)}

    # ──────────────────────────────────────────────────────────────────────
    # Chat completions (SSE JSON or plain text, segmented for long requests)
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(
        body:             ChatRequest,
        req:              Request,
        response:         Response,
        background_tasks: BackgroundTasks,
        session_id:       str   = Depends(get_session_id),
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        _maybe_warn_about_tools(body)

        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Server startup failed: {startup_error}")

        loader = req.app.state.ctx["loader"]
        engine = req.app.state.ctx["engine"]

        if not await _is_model_loaded(loader, engine, body.model):
            await _log_request(req, key_id, store, "/v1/chat/completions", 503, 0)
            raise HTTPException(503, f"Model '{body.model}' not loaded",
                                headers={"Retry-After": "5"})
        async with _model_status_lock:
            status = _model_status.get(body.model, "ready")
        if status != "ready":
            raise HTTPException(503, f"Model '{body.model}' status: {status}")

        _check_thermal_throttle(req)
        _check_server_capacity(req)

        if body.thinking and body.thinking_budget_tokens == 0:
            body.thinking_budget_tokens = 2048
        # Only raise an *explicit* max_tokens. When the caller asked for
        # unlimited (max_tokens <= 0), leave it unlimited — the thinking
        # budget is an internal floor for the reasoning phase, not a cap
        # on the visible answer.
        if body.thinking and body.thinking_budget_tokens > 0 and body.max_tokens > 0:
            body.max_tokens = max(body.max_tokens, body.thinking_budget_tokens)

        conv_store = req.app.state.ctx.get("conversation_store")
        merged     = body.messages
        if conv_store:
            stored = await conv_store.get(session_id) or []
            merged = _merge_history(
                stored,
                body.messages,
                history_mode=getattr(body, "history_mode", "auto"),
            )

        inf_cfg     = req.app.state.ctx.get("cfg", {}).get("inference", {})
        profile     = req.app.state.ctx.get("hw_profile")
        _suppress   = inf_cfg.get("suppress_thinking", False)
        _no_think   = getattr(body, "no_think", None)
        if body.thinking:
            _no_think = False

        merged_for_prompt = merged
        try:
            prompt = await _fit_prompt_to_context(
                engine, body.model, merged_for_prompt,
                body.max_tokens or 0, inf_cfg, profile,
                no_think_override=_no_think,
            )
        except PromptTooLargeError as _ptle:
            raise HTTPException(413, detail=str(_ptle))

        prompt_tokens = await _count_prompt_tokens(engine, body.model, prompt)
        response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
        _timeout = (body.timeout_s if body.timeout_s > 0
                    else float(inf_cfg.get("default_timeout_s", 0.0)))
        if body.thinking and _timeout < 600:
            _timeout = 600.0

        # ── Compute routing + effective max_tokens BEFORE building the IR ──
        # -1 (auto) in config → per-request threshold from n_ctx &
        # segment_size × max_segments.  Never a cap; a routing decision.
        _lr_threshold = _resolve_long_request_threshold(engine, body.model, inf_cfg)

        # Client-facing default when caller sends <=0.  0 = truly unlimited
        # (termination is by EOS / stop / loop guard / stall detection,
        # none of which depend on hardware speed).
        _effective_max_tokens = body.max_tokens
        if _effective_max_tokens <= 0:
            _effective_max_tokens = int(inf_cfg.get("default_max_tokens", 0) or 0)

        # Apply per-key quotas — clamps against the caller's
        # max_output_tokens and max_request_seconds, both server-wide caps
        # still applied on top by the engine.
        _effective_max_tokens, _timeout = _apply_key_quotas(
            req, _effective_max_tokens, _timeout,
        )

        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir = InferenceRequest(
            request_id  = str(uuid.uuid4()),
            model_id    = body.model,
            prompt      = prompt,
            max_tokens  = _effective_max_tokens,   # ← use the resolved value
            temperature = body.temperature,
            top_p       = body.top_p,
            top_k       = body.top_k,
            stop        = (body.stop or []) + inf_cfg.get("default_stop", []),
            stream      = body.stream,
            timeout_s   = _timeout,
            grammar=body.grammar,
            response_format=body.response_format,
        )

        opt        = req.app.state.ctx.get("optimizer")
        start_time = time.monotonic()

        # Unbounded (effective <=0) → segmented (the only path with
        # automatic window sliding).  Bounded → segmented only when it
        # exceeds the auto/cfg threshold.
        if _effective_max_tokens <= 0:
            use_segmented = True
        else:
            use_segmented = _effective_max_tokens > _lr_threshold

        try:
            if body.stream and use_segmented:
                _seg_stream = _stream_segmented(
                    engine, ir, body, conv_store, session_id,
                    req, key_id, store, start_time, profile,
                    getattr(body, "stream_format", "json"),
                    merged,
                )
                # Heartbeats on the segmented path. Prefill on this hardware
                # can be silent for many minutes on a large prompt, so the
                # client needs a signal-of-life on the wire or its own idle
                # timer (or a total deadline) fires mid-generation. SSE
                # comment frames (": hb {...}") are ignored by every
                # spec-compliant parser, so old clients see no change.
                # Only wrap for stream_format=json; injecting ": hb ..."
                # into a raw text/plain body would corrupt it.
                _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
                _toks_src = req.app.state.ctx.get("token_source") or engine
                if _hb_iv > 0 and getattr(body, "stream_format", "json") == "json":
                    _seg_stream = with_sse_heartbeat(
                        _seg_stream,
                        interval_s=_hb_iv,
                        status_fn = _hb_status_fn(_toks_src, ir, start_time),
                    )
                return StreamingResponse(
                    _seg_stream,
                    media_type="text/event-stream" if getattr(body, "stream_format",
                                                              "json") == "json" else "text/plain",
                    headers=_SSE_HEADERS,
                )
            elif body.stream:
                stream_fmt = getattr(body, "stream_format", "json")

                if body.thinking:
                    _thinking_gen = _stream_with_thinking_retry(
                        engine, ir, opt, body, ir.request_id, stream_fmt,
                        conv_store, session_id, merged,
                        req, key_id, store, start_time,
                        profile,
                    )
                    if stream_fmt == "json":
                        _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
                        _toks_src = req.app.state.ctx.get("token_source") or engine
                        if _hb_iv > 0:
                            _thinking_gen = with_sse_heartbeat(
                                _thinking_gen,
                                interval_s=_hb_iv,
                                status_fn=_hb_status_fn(_toks_src, ir, start_time),
                            )
                    return StreamingResponse(
                        _thinking_gen,
                        media_type="text/event-stream" if stream_fmt == "json" else "text/plain",
                        headers=_SSE_HEADERS,
                    )

                if stream_fmt == "json":
                    async def _chat_sse():
                        full = ""
                        client_aborted = False
                        _first_ts: Optional[float] = None
                        _token_count = 0

                        async def _do_persist():
                            _ttft_ms = (
                                (_first_ts - start_time) * 1000.0
                            ) if _first_ts else None
                            _decode_s = (
                                time.monotonic() - start_time
                                - ((_first_ts - start_time) if _first_ts else 0.0)
                            )
                            _decode_tps = (
                                (_token_count - 1) / _decode_s
                                if _decode_s > 0.05 and _token_count > 1 else None
                            )
                            try:
                                await _persist_after_stream(
                                    conv_store=conv_store,
                                    session_id=session_id,
                                    body=body, ir=ir, full=full,
                                    req=req, key_id=key_id, store=store,
                                    start=start_time,
                                    merged=merged,
                                    status=499 if client_aborted else 200,
                                    path="/v1/chat/completions",
                                    ttft_ms=_ttft_ms,
                                    decode_tps=_decode_tps,
                                    tokens=_token_count,
                                )
                            except Exception as exc:
                                log.error(
                                    "Persistence failed for %s: %s",
                                    ir.request_id[:8], exc, exc_info=True,
                                )

                        try:
                            async for b in _openai_sse_stream(
                                engine, ir, body.model, "chat", opt,
                                req=req, profile=profile,
                            ):
                                if await req.is_disconnected():
                                    log.info(
                                        "Client disconnected during stream, "
                                        "cancelling request %s", ir.request_id[:8],
                                    )
                                    client_aborted = True
                                    break
                                try:
                                    if b.startswith(b"data: ") and b != b"data: [DONE]\n\n":
                                        obj   = json.loads(b[6:].strip())
                                        delta = obj.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            if _first_ts is None:
                                                _first_ts = time.monotonic()
                                            _token_count += 1
                                            full += content
                                except Exception:
                                    pass
                                yield b
                        except (asyncio.CancelledError, GeneratorExit):
                            client_aborted = True
                            raise
                        finally:
                            with suppress(Exception):
                                engine.clear_progress(ir.root_id or ir.request_id)
                            asyncio.create_task(_do_persist())

                    _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
                    _toks_src = req.app.state.ctx.get("token_source") or engine
                    _chat_sse_stream = _chat_sse()
                    if _hb_iv > 0:
                        _chat_sse_stream = with_sse_heartbeat(
                            _chat_sse_stream,
                            interval_s=_hb_iv,
                            status_fn=_hb_status_fn(_toks_src, ir, start_time),
                        )
                    return StreamingResponse(
                        _chat_sse_stream,
                        media_type = "text/event-stream",
                        headers    = _SSE_HEADERS,
                    )
                else:
                    async def _chat_plain():
                        full = ""
                        client_aborted = False
                        _first_ts: Optional[float] = None
                        _token_count = 0

                        async def _do_persist():
                            _ttft_ms = (
                                (_first_ts - start_time) * 1000.0
                            ) if _first_ts else None
                            _decode_s = (
                                time.monotonic() - start_time
                                - ((_first_ts - start_time) if _first_ts else 0.0)
                            )
                            _decode_tps = (
                                (_token_count - 1) / _decode_s
                                if _decode_s > 0.05 and _token_count > 1 else None
                            )
                            try:
                                await _persist_after_stream(
                                    conv_store=conv_store,
                                    session_id=session_id,
                                    body=body, ir=ir, full=full,
                                    req=req, key_id=key_id, store=store,
                                    start=start_time,
                                    merged=merged,
                                    status=499 if client_aborted else 200,
                                    path="/v1/chat/completions",
                                    ttft_ms=_ttft_ms,
                                    decode_tps=_decode_tps,
                                    tokens=_token_count,
                                )
                            except Exception as exc:
                                log.error(
                                    "Persistence failed for %s: %s",
                                    ir.request_id[:8], exc, exc_info=True,
                                )

                        source = opt.generate_stream(ir) if opt else engine.generate_stream(ir)
                        source = _wrap_stream(
                            lambda: source,
                            req,
                            ir,
                            profile,
                            ir.max_tokens,
                        )
                        try:
                            async with _SafeStream(source, ir.request_id) as safe_stream:
                                async for chunk in safe_stream:
                                    if await req.is_disconnected():
                                        log.info(
                                            "Client disconnected during stream, "
                                            "cancelling request %s", ir.request_id[:8],
                                        )
                                        client_aborted = True
                                        break
                                    if not validate_stream_chunk(chunk, ir.request_id):
                                        continue
                                    if _first_ts is None:
                                        _first_ts = time.monotonic()
                                    _token_count += 1
                                    yield chunk.encode()
                                    full += chunk
                        except (asyncio.CancelledError, GeneratorExit):
                            client_aborted = True
                            raise
                        finally:
                            with suppress(Exception):
                                engine.clear_progress(ir.root_id or ir.request_id)
                            asyncio.create_task(_do_persist())

                    return StreamingResponse(
                        _chat_plain(),
                        media_type="text/plain; charset=utf-8",
                        headers=_SSE_HEADERS,
                    )

            else:
                _toks = req.app.state.ctx.get("token_source") or engine
                result = await _toks.generate(ir)
                if result.finish_reason == "timeout":
                    raise HTTPException(504, detail=result.text)
                if result.finish_reason == "cancelled":
                    raise HTTPException(499, detail="request cancelled")
                req.app.state.ctx["metrics"].record_request(body.model)
                req.app.state.ctx["metrics"].record_latency(result.ttft_ms, result.tps)

                if not result.text or not result.text.strip():
                    result.text         = "[Error: Model generated no output]"
                    result.finish_reason = "error"

                if "telemetry" in req.app.state.ctx:
                    with suppress(Exception):
                        req.app.state.ctx["telemetry"].record(
                            body.model, result.tokens_gen, result.ttft_ms, result.tps,
                            error=None if result.finish_reason not in ("error", "oom")
                                  else result.text,
                        )

                if result.finish_reason == "circuit_open":
                    raise HTTPException(503, detail=result.text,
                                        headers={"Retry-After": "60"})
                if result.finish_reason == "overloaded":
                    # Engine hit its _inflight >= _queue._max check.
                    # Pre-check should have caught this before we got here,
                    # but keep it for races.
                    raise HTTPException(503, detail=result.text,
                                        headers={"Retry-After": "30"})
                if result.finish_reason == "rate_limited":
                    raise HTTPException(429, detail=result.text,
                                        headers={"Retry-After": "10"})

                if conv_store and result.text and not _is_error_token(result.text):
                    merged.append({"role": "assistant", "content": result.text})
                    with suppress(Exception):
                        await conv_store.create(session_id, merged)

                elapsed = time.monotonic() - start_time
                await _log_request(req, key_id, store, "/v1/chat/completions", 200,
                                   elapsed * 1000, result.tokens_gen,
                                   request_id=ir.request_id,
                                   ttft_ms=result.ttft_ms,
                                   decode_tps=result.tps)

                background_tasks.add_task(cleanup_after_generation, req.app.state.ctx)

                return _resp(result, body.model)

        except asyncio.TimeoutError:
            await _log_request(req, key_id, store, "/v1/chat/completions", 504,
                               (time.monotonic() - start_time) * 1000)
            raise HTTPException(504, f"Request timeout after {_timeout}s")
        except asyncio.CancelledError:
            await _log_request(req, key_id, store, "/v1/chat/completions", 499,
                               (time.monotonic() - start_time) * 1000)
            raise HTTPException(499, "Client closed request")
        except Exception as exc:
            log.error("chat_completions [%s]: %s", ir.request_id[:8], exc, exc_info=True)
            await _log_request(req, key_id, store, "/v1/chat/completions", 500,
                               (time.monotonic() - start_time) * 1000)
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Resume endpoint
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/v1/chat/resume")
    async def chat_resume(
        body: ResumeRequest,
        req: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        key_info: tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Server startup failed: {startup_error}")

        _check_thermal_throttle(req)

        conv_store = req.app.state.ctx.get("conversation_store")
        if not conv_store:
            raise HTTPException(503, "Conversation store not available")

        if body.checkpoint_id:
            checkpoint_data = await conv_store.get_checkpoint_by_id(body.conv_id, body.checkpoint_id)
        else:
            checkpoint_data = await conv_store.get_latest_checkpoint(body.conv_id)

        if not checkpoint_data:
            raise HTTPException(404, f"No checkpoint found for conversation '{body.conv_id}'")

        if checkpoint_data["status"] == "complete":
            raise HTTPException(400, "This conversation is already complete")

        model_id      = checkpoint_data["model_id"]
        accumulated   = checkpoint_data["accumulated_text"]
        tokens_so_far = checkpoint_data["tokens_generated"]
        segment_id    = checkpoint_data["segment_id"] + 1

        stored_checkpoint_id = checkpoint_data.get("checkpoint_id")

        original_prompt = checkpoint_data.get("original_prompt") or ""
        resume_prompt   = (original_prompt + accumulated) if original_prompt else accumulated
        merged_history: List[dict] = checkpoint_data.get("merged_history") or []

        temperature = body.temperature if body.temperature is not None else 0.7
        top_p       = body.top_p       if body.top_p       is not None else 0.95
        top_k       = body.top_k       if body.top_k       is not None else 40

        from local_llm_core.engine.inference.inference_engine import GenerationCheckpoint

        checkpoint_obj = GenerationCheckpoint(
            request_id=f"resume_{body.conv_id}_{int(time.time())}",
            model_id=model_id,
            accumulated_text=resume_prompt,
            tokens_generated=tokens_so_far,
            segment_number=segment_id,
            status="in_progress",
        )

        try:
            response.headers["X-Prompt-Tokens"] = str(max(1, int(len(resume_prompt) / 2.5)))
        except Exception:
            pass

        async def checkpoint_cb(new_checkpoint):
            try:
                if (original_prompt
                        and new_checkpoint.accumulated_text.startswith(original_prompt)):
                    new_checkpoint.accumulated_text = (
                        new_checkpoint.accumulated_text[len(original_prompt):]
                    )
            except Exception:
                pass
            await _save_checkpoint_callback(
                conv_store, body.conv_id, new_checkpoint,
                original_prompt=original_prompt or None,
                merged_history=merged_history or None,
            )

        engine = req.app.state.ctx["engine"]
        _toks  = req.app.state.ctx.get("token_source") or engine
        inf_cfg = req.app.state.ctx.get("cfg", {}).get("inference", {})
        start_time = time.monotonic()

        if body.stream:
            async def _resume_sse():
                full_text   = accumulated
                token_count = tokens_so_far
                state = {"interrupted": False}
                _first_ts: Optional[float] = None

                async def _post_resume_persist():
                    clean_text = full_text

                    if not _is_error_token(clean_text):
                        if state["interrupted"]:
                            try:
                                ckpt = {
                                    "model_id": model_id,
                                    "segment_id": checkpoint_data["segment_id"],
                                    "accumulated_text": clean_text,
                                    "tokens_generated": token_count,
                                    "status": "interrupted",
                                }
                                if original_prompt:
                                    ckpt["original_prompt"] = original_prompt
                                if merged_history:
                                    ckpt["merged_history"] = merged_history
                                await conv_store.save_checkpoint(body.conv_id, ckpt)
                                log.info(
                                    "Resume for %s interrupted — checkpoint "
                                    "re-saved (conv=%s, tokens=%d)",
                                    checkpoint_obj.request_id[:8],
                                    body.conv_id[:8], token_count,
                                )
                            except Exception as _cpe:
                                log.warning(
                                    "Interrupted-resume checkpoint save failed: %s",
                                    _cpe,
                                )
                        else:
                            try:
                                if stored_checkpoint_id:
                                    await conv_store.update_checkpoint_status(
                                        body.conv_id,
                                        stored_checkpoint_id,
                                        "complete",
                                    )
                                if merged_history:
                                    conv = list(merged_history)
                                else:
                                    conv = await conv_store.get(body.conv_id) or []
                                conv.append({"role": "assistant", "content": clean_text})
                                await conv_store.create(body.conv_id, conv)
                            except Exception as e:
                                log.error(
                                    "Failed to save complete conversation: %s", e,
                                )

                    elapsed = time.monotonic() - start_time
                    # New tokens produced by THIS resume request only.
                    # `token_count` starts at tokens_so_far (accumulated
                    # from prior segments); ttft/decode describe the
                    # current stream, not the whole conversation.
                    _new_tokens = token_count - tokens_so_far
                    _ttft_ms = ((_first_ts - start_time) * 1000.0) if _first_ts else None
                    _decode_s = (
                        elapsed - ((_first_ts - start_time) if _first_ts else 0.0)
                    )
                    _decode_tps = (
                        (_new_tokens - 1) / _decode_s
                        if _decode_s > 0.05 and _new_tokens > 1 else None
                    )
                    try:
                        await _log_request(
                            req, key_id, store, "/v1/chat/resume",
                            499 if state["interrupted"] else 200,
                            elapsed * 1000,
                            _new_tokens,
                            request_id=checkpoint_obj.request_id,
                            ttft_ms=_ttft_ms,
                            decode_tps=_decode_tps,
                        )
                    except Exception as le:
                        log.warning("Resume log failed: %s", le)
                    with suppress(Exception):
                        asyncio.create_task(
                            cleanup_after_generation(req.app.state.ctx),
                        )

                try:
                    async for token in _toks.resume_generation(
                            checkpoint_obj, checkpoint_cb,
                            extra_tokens=resolve_max_tokens(body.max_tokens),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                    ):
                        if await req.is_disconnected():
                            log.info(
                                "[%s] Client disconnected during resume after "
                                "%d new tokens",
                                checkpoint_obj.request_id[:8],
                                token_count - tokens_so_far,
                            )
                            state["interrupted"] = True
                            break

                        is_marker = _extract_stream_marker(token) is not None
                        is_legacy_interrupt = (
                            token.startswith("[Response interrupted")
                            or token.startswith("[Generation paused")
                        )
                        is_error = token.startswith("[Error:")

                        if is_marker or is_legacy_interrupt or is_error:
                            state["interrupted"] = True

                        payload = json.dumps({
                            "id": checkpoint_obj.request_id,
                            "object": "chat.completion.chunk",
                            "model": model_id,
                            "choices": [{
                                "delta": {"content": token},
                                "finish_reason": None,
                                "index": 0,
                            }],
                        })
                        yield f"data: {payload}\n\n".encode()

                        if is_error:
                            break

                        if is_marker or is_legacy_interrupt:
                            continue

                        if _first_ts is None:
                            _first_ts = time.monotonic()
                        full_text += token
                        token_count += 1
                except (asyncio.CancelledError, GeneratorExit):
                    client_aborted = True
                    raise
                except Exception as _stall_exc:
                    # Same as _stream_segmented: turn an engine-side
                    # StreamStallError into a canonical ⚡STALLED marker
                    # so the client's parseStreamMarker recognises it and
                    # the auto-resume loop fires, instead of an HTTP 500.
                    from local_llm_core.engine.inference.inference_engine import (
                        StreamStallError as _SSE, _make_stall_marker as _mk,
                    )
                    if isinstance(_stall_exc, _SSE):
                        log.warning(
                            "Resume %s stalled during %s after %d new tokens — "
                            "emitting ⚡STALLED marker",
                            checkpoint_obj.request_id[:8],
                            _stall_exc.phase, token_count - tokens_so_far,
                        )
                        state["interrupted"] = True
                        _m = _mk(_stall_exc.phase, _stall_exc.partial_tokens or 0)
                        payload = json.dumps({
                            "id": checkpoint_obj.request_id,
                            "object": "chat.completion.chunk",
                            "model": model_id,
                            "choices": [{"delta": {"content": _m},
                                         "finish_reason": "stalled", "index": 0}],
                        })
                        yield f"data: {payload}\n\n".encode()
                    else:
                        raise
                finally:
                    with suppress(Exception):
                        engine.clear_progress(
                            f"{checkpoint_obj.request_id}_resume"
                        )
                    asyncio.create_task(_post_resume_persist())

                yield b"data: [DONE]\n\n"

            _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
            _toks_src = req.app.state.ctx.get("token_source") or engine
            _resume_stream = _resume_sse()
            if _hb_iv > 0:
                # Engine progress is keyed on "<request>_resume" — this is the
                # exact key resume_generation's internal InferenceRequest uses.
                _resume_stream = with_sse_heartbeat(
                    _resume_stream,
                    interval_s=_hb_iv,
                    status_fn=_hb_status_fn(
                        _toks_src,
                        f"{checkpoint_obj.request_id}_resume",
                        start_time,
                    ),
                )
            return StreamingResponse(
                _resume_stream,
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        else:
            all_tokens: List[str] = []
            interrupted = False
            had_error = False
            error_detail = ""
            try:
                async for token in _toks.resume_generation(
                        checkpoint_obj, checkpoint_cb,
                        extra_tokens=resolve_max_tokens(body.max_tokens),
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                ):
                    is_marker = _extract_stream_marker(token) is not None
                    is_legacy_interrupt = (
                        token.startswith("[Response interrupted")
                        or token.startswith("[Generation paused")
                    )
                    is_error = token.startswith("[Error:")

                    if is_marker or is_legacy_interrupt:
                        interrupted = True
                        continue
                    if is_error:
                        interrupted = True
                        had_error = True
                        error_detail = token
                        continue

                    all_tokens.append(token)
            except asyncio.CancelledError:
                await _log_request(req, key_id, store, "/v1/chat/resume", 499,
                                   (time.monotonic() - start_time) * 1000)
                raise HTTPException(499, "Client closed request")

            full_text = accumulated + "".join(all_tokens)
            clean_text = _strip_stream_marker(full_text)
            elapsed = time.monotonic() - start_time

            if had_error:
                await _log_request(req, key_id, store, "/v1/chat/resume", 200,
                                   elapsed * 1000, 0)
                return {"error": "Generation failed",
                        "detail": error_detail or clean_text}

            if interrupted:
                try:
                    ckpt = {
                        "model_id": model_id,
                        "segment_id": checkpoint_data["segment_id"],
                        "accumulated_text": clean_text,
                        "tokens_generated": tokens_so_far + len(all_tokens),
                        "status": "interrupted",
                    }
                    if original_prompt:
                        ckpt["original_prompt"] = original_prompt
                    if merged_history:
                        ckpt["merged_history"] = merged_history
                    await conv_store.save_checkpoint(body.conv_id, ckpt)
                except Exception as e:
                    log.warning("Interrupted-resume checkpoint save failed: %s", e)
                await _log_request(req, key_id, store, "/v1/chat/resume", 200,
                                   elapsed * 1000, len(all_tokens),
                                   request_id=checkpoint_obj.request_id)
                return {
                    "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": clean_text},
                        "finish_reason": "interrupted",
                    }],
                    "usage": {
                        "prompt_tokens":     tokens_so_far,
                        "completion_tokens": len(all_tokens),
                        "total_tokens":      tokens_so_far + len(all_tokens),
                    },
                }

            if stored_checkpoint_id:
                with suppress(Exception):
                    await conv_store.update_checkpoint_status(
                        body.conv_id, stored_checkpoint_id, "complete"
                    )
            if merged_history:
                conv = list(merged_history)
            else:
                conv = await conv_store.get(body.conv_id) or []
            conv.append({"role": "assistant", "content": clean_text})
            with suppress(Exception):
                await conv_store.create(body.conv_id, conv)

            await _log_request(req, key_id, store, "/v1/chat/resume", 200,
                               elapsed * 1000, len(all_tokens),
                               request_id=checkpoint_obj.request_id)
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": clean_text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens":     tokens_so_far,
                    "completion_tokens": len(all_tokens),
                    "total_tokens":      tokens_so_far + len(all_tokens),
                },
            }

    # ──────────────────────────────────────────────────────────────────────
    # Completions (text)
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/v1/completions")
    async def completions(
        body:             CompletionRequest,
        req:              Request,
        background_tasks: BackgroundTasks,
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Server startup failed: {startup_error}")

        loader = req.app.state.ctx["loader"]
        engine = req.app.state.ctx["engine"]
        if not await _is_model_loaded(loader, engine, body.model):
            raise HTTPException(503, f"Model '{body.model}' not loaded",
                                headers={"Retry-After": "5"})
        async with _model_status_lock:
            if _model_status.get(body.model, "ready") != "ready":
                raise HTTPException(503, f"Model '{body.model}' not ready")

        _check_thermal_throttle(req)
        _check_server_capacity(req)

        inf_cfg    = req.app.state.ctx.get("cfg", {}).get("inference", {})
        _timeout   = float(inf_cfg.get("default_timeout_s", 0.0))
        request_id = str(uuid.uuid4())
        profile    = req.app.state.ctx.get("hw_profile")

        _effective_max_tokens = body.max_tokens
        if _effective_max_tokens <= 0:
            _effective_max_tokens = int(inf_cfg.get("default_max_tokens", 0) or 0)

        _effective_max_tokens, _timeout = _apply_key_quotas(
            req, _effective_max_tokens, _timeout,
        )

        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir = InferenceRequest(
            request_id=request_id,
            model_id=body.model,
            prompt=body.prompt,
            max_tokens=_effective_max_tokens,
            temperature=body.temperature,
            stop        = (body.stop or []) + inf_cfg.get("default_stop", []),
            stream      = body.stream,
            timeout_s   = _timeout,
            grammar=body.grammar,
            response_format=body.response_format,
        )

        start = time.monotonic()
        try:
            if body.stream:
                async def _comp_sse():
                    _first_token_ts: Optional[float] = None
                    _token_count = 0
                    try:
                        async for b in _openai_sse_stream(
                                engine, ir, body.model, "completion",
                                req=req, profile=profile,
                        ):
                            if _first_token_ts is None:
                                _first_token_ts = time.monotonic()
                            _token_count += 1
                            if await req.is_disconnected():
                                log.info("Client disconnected during stream, cancelling request %s",
                                         ir.request_id[:8])
                                break
                            yield b
                    except (asyncio.CancelledError, GeneratorExit):
                        # No `client_aborted` here: this closure has no
                        # persistent state to mark — the `finally` below
                        # logs the request either way, and `_log_request`
                        # is always reached. The re-raise is the whole
                        # contract.
                        raise
                    finally:
                        with suppress(Exception):
                            engine.clear_progress(ir.root_id or ir.request_id)
                        _elapsed = time.monotonic() - start
                        _decode_tps = (
                            (_token_count - 1) / max(_elapsed - (_first_token_ts - start), 0.001)
                            if _first_token_ts and _token_count > 1 else None
                        )
                        _ttft_ms = ((_first_token_ts - start) * 1000.0) if _first_token_ts else None
                        try:
                            await _log_request(
                                req, key_id, store, "/v1/completions", 200,
                                _elapsed * 1000, _token_count,
                                request_id=ir.request_id,
                                ttft_ms=_ttft_ms,
                                decode_tps=_decode_tps,
                            )
                        except Exception as _le:
                            log.warning("completions stream log failed: %s", _le)
                        with suppress(Exception):
                            asyncio.create_task(cleanup_after_generation(req.app.state.ctx))

                _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
                _toks_src = req.app.state.ctx.get("token_source") or engine
                _comp_stream = _comp_sse()
                if _hb_iv > 0:
                    _comp_stream = with_sse_heartbeat(
                        _comp_stream,
                        interval_s=_hb_iv,
                        status_fn=_hb_status_fn(_toks_src, ir, start),
                    )
                return StreamingResponse(
                    _comp_stream,
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            else:
                _toks = req.app.state.ctx.get("token_source") or engine
                result = await _toks.generate(ir)
                if result.finish_reason == "timeout":
                    raise HTTPException(504, detail=result.text)
                if result.finish_reason == "cancelled":
                    raise HTTPException(499, detail="request cancelled")
                if result.finish_reason == "circuit_open":
                    raise HTTPException(503, detail=result.text,
                                        headers={"Retry-After": "60"})
                if result.finish_reason == "overloaded":
                    raise HTTPException(503, detail=result.text,
                                        headers={"Retry-After": "30"})
                if result.finish_reason == "rate_limited":
                    raise HTTPException(429, detail=result.text,
                                        headers={"Retry-After": "10"})
                elapsed = time.monotonic() - start
                await _log_request(req, key_id, store, "/v1/completions",
                                   200, elapsed * 1000, result.tokens_gen, request_id=request_id)
                background_tasks.add_task(cleanup_after_generation, req.app.state.ctx)
                return {
                    "id":      f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object":  "text_completion",
                    "created": int(time.time()),
                    "model":   body.model,
                    "choices": [{"text": result.text, "index": 0,
                                 "finish_reason": result.finish_reason}],
                    "usage": {
                        "prompt_tokens":     result.prompt_tokens,
                        "completion_tokens": result.tokens_gen,
                        "total_tokens":      result.prompt_tokens + result.tokens_gen,
                    },
                }
        except asyncio.CancelledError:
            await _log_request(req, key_id, store, "/v1/completions", 499,
                               (time.monotonic() - start) * 1000)
            raise HTTPException(499, "Client closed request")
        except Exception as exc:
            await _log_request(req, key_id, store, "/v1/completions", 500,
                               (time.monotonic() - start) * 1000)
            raise

    @app.post("/v1/embeddings")
    async def embeddings(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        raise HTTPException(501, "Embeddings not supported")

    # ──────────────────────────────────────────────────────────────────────
    # Smart routing chat (Auto-Router)
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/v1/chat/smart")
    async def chat_smart_route(
        body:       SmartChatRequest,
        req:        Request,
        response:   Response,
        session_id: str   = Depends(get_session_id),
        key_info:   tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        loader = req.app.state.ctx["loader"]
        engine = req.app.state.ctx["engine"]
        all_m  = await _get_all_models(loader, include_draft=False)
        ready  = {m["id"]: m for m in all_m if m.get("status") == "ready"}
        if not ready:
            raise HTTPException(503, "No ready models available")

        _check_thermal_throttle(req)
        _check_server_capacity(req)

        FAST  = ("1b", "0.5b", "tiny", "mini", "small")
        fast  = [mid for mid in ready if any(p in mid.lower() for p in FAST)]
        main  = [mid for mid in ready if mid not in fast]

        all_text   = " ".join(
            str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in body.messages
        )
        words      = len(all_text.split())
        selected   = (
            fast[0] if words < 50 and fast
            else main[0] if main
            else next(iter(ready))
        )

        conv_store = req.app.state.ctx.get("conversation_store")
        merged     = body.messages
        if conv_store:
            stored = await conv_store.get(session_id) or []
            merged = _merge_history(
                stored, body.messages,
                history_mode=getattr(body, "history_mode", "auto"),
            )

        inf_cfg = req.app.state.ctx.get("cfg", {}).get("inference", {})
        profile = req.app.state.ctx.get("hw_profile")
        merged_for_prompt = merged

        _no_think = getattr(body, "no_think", None)
        if getattr(body, "thinking", False):
            _no_think = False

        try:
            prompt = await _fit_prompt_to_context(
                engine, selected, merged_for_prompt,
                body.max_tokens or 0, inf_cfg, profile,
                no_think_override=_no_think,
            )
        except PromptTooLargeError as _ptle:
            raise HTTPException(413, detail=str(_ptle))

        prompt_tokens = await _count_prompt_tokens(engine, selected, prompt)
        response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
        _timeout = (body.timeout_s if body.timeout_s > 0
                    else float(inf_cfg.get("default_timeout_s", 0.0)))

        _effective_max_tokens = body.max_tokens
        if _effective_max_tokens <= 0:
            _effective_max_tokens = int(inf_cfg.get("default_max_tokens", 0) or 0)

        _effective_max_tokens, _timeout = _apply_key_quotas(
            req, _effective_max_tokens, _timeout,
        )

        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir = InferenceRequest(
            request_id=str(uuid.uuid4()), model_id=selected, prompt=prompt,
            max_tokens=_effective_max_tokens, temperature=body.temperature,
            top_p=body.top_p, top_k=body.top_k,
            stop=(body.stop or []) + inf_cfg.get("default_stop", []),
            stream=True, timeout_s=_timeout,
            grammar=body.grammar,
            response_format=body.response_format,
        )
        opt       = req.app.state.ctx.get("optimizer")
        start     = time.monotonic()

        async def _smart_gen():
            full = ""
            client_aborted = False
            _first_ts: Optional[float] = None
            _token_count = 0

            async def _do_persist():
                _ttft_ms = ((_first_ts - start) * 1000.0) if _first_ts else None
                _decode_s = (
                    time.monotonic() - start
                    - ((_first_ts - start) if _first_ts else 0.0)
                )
                _decode_tps = (
                    (_token_count - 1) / _decode_s
                    if _decode_s > 0.05 and _token_count > 1 else None
                )
                try:
                    await _persist_after_stream(
                        conv_store=conv_store,
                        session_id=session_id,
                        body=body, ir=ir, full=full,
                        req=req, key_id=key_id, store=store,
                        start=start,
                        merged=merged,
                        status=499 if client_aborted else 200,
                        path="/v1/chat/smart",
                        ttft_ms=_ttft_ms,
                        decode_tps=_decode_tps,
                        tokens=_token_count,
                    )
                except Exception as exc:
                    log.error(
                        "Smart persist failed for %s: %s",
                        ir.request_id[:8], exc, exc_info=True,
                    )

            source = _stream_plain_text(engine, ir, optimizer=opt, req=req, profile=profile)
            _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
            _toks_src = req.app.state.ctx.get("token_source") or engine
            _merged = (with_heartbeat(
                source, interval_s=_hb_iv,
                status_fn=_hb_status_fn(_toks_src, ir, start),
            ) if _hb_iv > 0 else None)
            try:
                if _merged is None:
                    async for b in source:
                        if await req.is_disconnected():
                            client_aborted = True
                            break
                        if _first_ts is None:
                            _first_ts = time.monotonic()
                        _token_count += 1
                        full += b.decode() if isinstance(b, bytes) else b
                        yield b
                else:
                    async for kind, b in _merged:
                        if await req.is_disconnected():
                            log.info(
                                "Client disconnected during smart stream, "
                                "cancelling request %s", ir.request_id[:8],
                            )
                            client_aborted = True
                            break
                        if kind == "hb":
                            yield b          # control frame — NEVER added to `full`
                            continue
                        if _first_ts is None:
                            _first_ts = time.monotonic()
                        _token_count += 1
                        full += b.decode() if isinstance(b, bytes) else b
                        yield b
            except (asyncio.CancelledError, GeneratorExit):
                client_aborted = True
                raise
            finally:
                if _merged is not None:
                    with suppress(BaseException):
                        await _merged.aclose()
                with suppress(Exception):
                    engine.clear_progress(ir.root_id or ir.request_id)
                asyncio.create_task(_do_persist())

        return StreamingResponse(
            _smart_gen(),
            media_type = "text/plain; charset=utf-8",
            headers    = {**_SSE_HEADERS, "X-Model-Selected": selected,
                          "X-Prompt-Words": str(words)},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Plain text stream (used by the Web UI's explicit-model mode)
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/v1/chat/stream")
    async def chat_stream_text(
        body:       ChatRequest,
        req:        Request,
        response:   Response,
        session_id: str   = Depends(get_session_id),
        key_info:   tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        loader = req.app.state.ctx["loader"]
        engine = req.app.state.ctx["engine"]
        if not await _is_model_loaded(loader, engine, body.model):
            raise HTTPException(503, f"Model '{body.model}' not loaded",
                                headers={"Retry-After": "5"})

        _check_thermal_throttle(req)
        _check_server_capacity(req)

        conv_store = req.app.state.ctx.get("conversation_store")
        merged     = body.messages
        if conv_store:
            stored = await conv_store.get(session_id) or []
            merged = _merge_history(
                stored, body.messages,
                history_mode=getattr(body, "history_mode", "auto"),
            )

        inf_cfg = req.app.state.ctx.get("cfg", {}).get("inference", {})
        profile = req.app.state.ctx.get("hw_profile")
        if body.thinking and body.thinking_budget_tokens > 0 and body.max_tokens > 0:
            body.max_tokens = max(body.max_tokens, body.thinking_budget_tokens)

        _no_think = getattr(body, "no_think", None)
        if body.thinking:
            _no_think = False

        merged_for_prompt = merged
        try:
            prompt = await _fit_prompt_to_context(
                engine, body.model, merged_for_prompt,
                body.max_tokens or 0, inf_cfg, profile,
                no_think_override=_no_think,
            )
        except PromptTooLargeError as _ptle:
            raise HTTPException(413, detail=str(_ptle))

        prompt_tokens = await _count_prompt_tokens(engine, body.model, prompt)
        response.headers["X-Prompt-Tokens"] = str(prompt_tokens)
        _timeout = (body.timeout_s if body.timeout_s > 0
                    else float(inf_cfg.get("default_timeout_s", 0.0)))
        if body.thinking and _timeout < 600:
            _timeout = 600.0

        _effective_max_tokens = body.max_tokens
        if _effective_max_tokens <= 0:
            _effective_max_tokens = int(inf_cfg.get("default_max_tokens", 0) or 0)

        _effective_max_tokens, _timeout = _apply_key_quotas(
            req, _effective_max_tokens, _timeout,
        )

        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir = InferenceRequest(
            request_id=str(uuid.uuid4()), model_id=body.model, prompt=prompt,
            max_tokens=_effective_max_tokens, temperature=body.temperature,
            top_p=body.top_p, top_k=body.top_k,
            stop=(body.stop or []) + inf_cfg.get("default_stop", []),
            stream=True, timeout_s=_timeout,
            grammar=body.grammar,
            response_format=body.response_format,
        )
        opt   = req.app.state.ctx.get("optimizer")
        start_time = time.monotonic()

        async def _stream():
            full = ""
            client_aborted = False
            _first_ts: Optional[float] = None
            _token_count = 0

            async def _do_persist() -> None:
                _ttft_ms = ((_first_ts - start_time) * 1000.0) if _first_ts else None
                _decode_s = (
                    time.monotonic() - start_time
                    - ((_first_ts - start_time) if _first_ts else 0.0)
                )
                _decode_tps = (
                    (_token_count - 1) / _decode_s
                    if _decode_s > 0.05 and _token_count > 1 else None
                )
                try:
                    await _persist_after_stream(
                        conv_store=conv_store,
                        session_id=session_id,
                        body=body, ir=ir, full=full,
                        req=req, key_id=key_id, store=store, start=start_time,
                        merged=merged,
                        status=499 if client_aborted else 200,
                        path="/v1/chat/stream",
                        ttft_ms=_ttft_ms,
                        decode_tps=_decode_tps,
                        tokens=_token_count,
                    )
                except Exception as exc:
                    log.error(
                        "Persistence failed for %s: %s",
                        ir.request_id[:8], exc, exc_info=True,
                    )

            source = _stream_plain_text(
                engine, ir, optimizer=opt, req=req, profile=profile,
            )
            _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
            _toks_src = req.app.state.ctx.get("token_source") or engine
            _merged = None
            try:
                async with _SafeStream(source, ir.request_id) as safe_stream:
                    _merged = with_heartbeat(
                        safe_stream,
                        interval_s=(_hb_iv if _hb_iv > 0 else 3600.0),
                        status_fn=_hb_status_fn(_toks_src, ir, start_time),
                    )
                    async for kind, b in _merged:
                        if await req.is_disconnected():
                            log.info(
                                "Client disconnected during stream "
                                "(request %s) — persisting partial response",
                                ir.request_id[:8],
                            )
                            client_aborted = True
                            break
                        if kind == "hb":
                            yield b
                            continue
                        tok = b.decode() if isinstance(b, bytes) else b
                        if _first_ts is None:
                            _first_ts = time.monotonic()
                        _token_count += 1
                        full += tok
                        yield b.encode() if isinstance(b, str) else b
            except (asyncio.CancelledError, GeneratorExit):
                client_aborted = True
                raise
            finally:
                if _merged is not None:
                    with suppress(BaseException):
                        await _merged.aclose()
                with suppress(Exception):
                    engine.clear_progress(ir.root_id or ir.request_id)
                asyncio.create_task(_do_persist())

        return StreamingResponse(
            _stream(),
            media_type = "text/plain; charset=utf-8",
            headers    = _SSE_HEADERS,
        )

    # ──────────────────────────────────────────────────────────────────────
    # BitNet endpoints
    # ──────────────────────────────────────────────────────────────────────
    @app.get("/v1/bitnet/status")
    async def bitnet_status(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            from local_llm_core.engine.adaptive_awareness.model_unified_env import auto_detect_backend_info
            return auto_detect_backend_info()
        except Exception as exc:
            return {"error": str(exc), "best_backend": "unknown"}

    @app.get("/v1/bitnet/info")
    async def bitnet_info(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            from local_llm_core.engine.adaptive_awareness.model_unified_env import BitNetStateManager
            s = BitNetStateManager.load()
            return {
                "available":      s.available,
                "model_name":     s.model_name,
                "model_path":     s.model_path,
                "gguf_path":      s.gguf_path,
                "quant_type":     s.quant_type,
                "quant_embd":     s.quant_embd,
                "llama_server":   s.llama_server_binary,
                "llama_cli":      s.llama_cli_binary,
                "spectre_mode":   s.spectre_mode,
                "atlas_threads":  s.atlas_threads,
                "verify_passed":  s.verify_passed,
                "setup_version":  s.setup_version,
                "setup_timestamp": s.setup_timestamp,
            }
        except Exception as exc:
            raise HTTPException(500, f"BitNet info error: {exc}")

    @app.post("/v1/bitnet/generate")
    async def bitnet_generate(
        body:             CompletionRequest,
        req:              Request,
        background_tasks: BackgroundTasks,
        key_info:         tuple = Depends(verify_api_key),
    ):
        is_valid, key_id, store = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        startup_error = req.app.state.ctx.get("startup_error")
        if startup_error:
            raise HTTPException(503, f"Startup failed: {startup_error}")

        engine = req.app.state.ctx["engine"]
        _check_thermal_throttle(req)
        _check_server_capacity(req)

        try:
            from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
            from local_llm_core.engine.adaptive_awareness.model_unified_env import BitNetStateManager
            state    = BitNetStateManager.load()
            model_id = state.model_name or body.model or "bitnet-state"

            loader = req.app.state.ctx.get("loader", engine)
            if not await _is_model_loaded(loader, engine, model_id):
                log.info("Auto-loading BitNet for /v1/bitnet/generate: %s", model_id)
                bl      = BitNetLoader.from_state_file()
                wrapper = await bl.load("state_model")
                _register_bitnet_in_engine(engine, model_id, wrapper)
                await register_external_model(
                    model_id, _make_bitnet_info({"id": model_id, "path": state.model_path}, wrapper)
                )
                async with _model_status_lock:
                    _model_status[model_id] = "ready"

        except Exception as exc:
            raise HTTPException(503,
                f"BitNet not available: {exc}. Run local-llm-setup or --minimal-build.")

        inf_cfg    = req.app.state.ctx.get("cfg", {}).get("inference", {})
        _timeout   = float(inf_cfg.get("default_timeout_s", 0.0))
        request_id = str(uuid.uuid4())
        profile    = req.app.state.ctx.get("hw_profile")

        _effective_max_tokens = body.max_tokens
        if _effective_max_tokens <= 0:
            _effective_max_tokens = int(inf_cfg.get("default_max_tokens", 0) or 0)

        _effective_max_tokens, _timeout = _apply_key_quotas(
            req, _effective_max_tokens, _timeout,
        )

        from local_llm_core.engine.inference.inference_engine import InferenceRequest
        ir = InferenceRequest(
            request_id=request_id,
            model_id=body.model,
            prompt=body.prompt,
            max_tokens=_effective_max_tokens,
            temperature=body.temperature,
            stop        = (body.stop or []) + inf_cfg.get("default_stop", []),
            stream      = body.stream,
            timeout_s   = _timeout,
            grammar=body.grammar,
            response_format=body.response_format,
        )

        start = time.monotonic()
        if body.stream:
            async def _bitnet_sse():
                _first_token_ts: Optional[float] = None
                _token_count = 0
                try:
                    async for b in _openai_sse_stream(
                            engine, ir, model_id, "completion",
                            req=req, profile=profile,
                    ):
                        if _first_token_ts is None:
                            _first_token_ts = time.monotonic()
                        _token_count += 1
                        if await req.is_disconnected():
                            log.info("Client disconnected during stream, cancelling request %s", ir.request_id[:8])
                            break
                        yield b
                    _elapsed = time.monotonic() - start
                    _decode_tps = (
                        (_token_count - 1) / max(_elapsed - (_first_token_ts - start), 0.001)
                        if _first_token_ts and _token_count > 1 else None
                    )
                    _ttft_ms = ((_first_token_ts - start) * 1000.0) if _first_token_ts else None
                    await _log_request(req, key_id, store, "/v1/bitnet/generate",
                                       200, _elapsed * 1000, _token_count,
                                       request_id=request_id,
                                       ttft_ms=_ttft_ms,
                                       decode_tps=_decode_tps)
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                finally:
                    with suppress(Exception):
                        engine.clear_progress(ir.root_id or ir.request_id)

            _hb_iv = float(inf_cfg.get("heartbeat_interval_s", 5.0) or 0.0)
            _toks_src = req.app.state.ctx.get("token_source") or engine
            _bn_stream = _bitnet_sse()
            if _hb_iv > 0:
                _bn_stream = with_sse_heartbeat(
                    _bn_stream,
                    interval_s=_hb_iv,
                    status_fn = _hb_status_fn(_toks_src, ir, start),
                )
            return StreamingResponse(_bn_stream, media_type="text/event-stream",
                                     headers=_SSE_HEADERS)
        else:
            _toks = req.app.state.ctx.get("token_source") or engine
            result = await _toks.generate(ir)
            if result.finish_reason == "timeout":
                raise HTTPException(504, detail=result.text)
            if result.finish_reason == "cancelled":
                raise HTTPException(499, detail="request cancelled")
            if result.finish_reason == "circuit_open":
                raise HTTPException(503, detail=result.text,
                                    headers={"Retry-After": "60"})
            if result.finish_reason == "overloaded":
                raise HTTPException(503, detail=result.text,
                                    headers={"Retry-After": "30"})
            if result.finish_reason == "rate_limited":
                raise HTTPException(429, detail=result.text,
                                    headers={"Retry-After": "10"})
            elapsed = time.monotonic() - start
            await _log_request(req, key_id, store, "/v1/bitnet/generate",
                               200, elapsed * 1000, result.tokens_gen, request_id=request_id)
            background_tasks.add_task(cleanup_after_generation, req.app.state.ctx)
            return {
                "id":      f"cmpl-{uuid.uuid4().hex[:12]}",
                "object":  "text_completion",
                "created": int(time.time()),
                "model":   model_id,
                "choices": [{"text": result.text, "index": 0,
                             "finish_reason": result.finish_reason}],
                "usage": {
                    "prompt_tokens":     result.prompt_tokens,
                    "completion_tokens": result.tokens_gen,
                    "total_tokens":      result.prompt_tokens + result.tokens_gen,
                },
            }

    @app.get("/v1/bitnet/backend")
    async def bitnet_backend(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            from local_llm_core.engine.bitnet import get_status
            return get_status()
        except Exception as exc:
            return {"error": str(exc), "available": False}

    # ──────────────────────────────────────────────────────────────────────
    # Monitoring endpoints
    # ──────────────────────────────────────────────────────────────────────
    @app.get("/v1/metrics")
    async def metrics_ep(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        c = req.app.state.ctx
        if c.get("startup_error"):
            return {"error": c["startup_error"]}

        engine = c.get("engine")
        vram   = c.get("vram")
        info   = c.get("sys_info")
        vm     = psutil.virtual_memory()

        vram_used = 0
        _total_mb = 0
        if vram:
            try:
                vram_used = vram.stats.used_bytes
            except Exception:
                pass
        if info and info.gpus:
            _total_mb = sum(g.vram_total_mb for g in info.gpus if getattr(g, "supported", True))

        eng_stats     = engine.stats() if engine else {}
        metrics_snap  = c["metrics"].snapshot() if c.get("metrics") else {}
        req_rate      = metrics_snap.get("req_rate", 0.0)

        vram_block = None
        if vram:
            vram_block = {
                "total_bytes":       _total_mb * 1024 * 1024,
                "used_bytes":        vram_used,
                "free_bytes":        max(0, _total_mb * 1024 * 1024 - vram_used),
                "fragmented_bytes":  getattr(getattr(vram, "stats", None), "fragmented_bytes", 0),
                "pinned_bytes":      getattr(getattr(vram, "stats", None), "pinned_bytes", 0),
                "utilization_pct":   round(vram_used / max(_total_mb * 1024 * 1024, 1) * 100, 1),
                "cache_hits":        getattr(getattr(vram, "stats", None), "cache_hits", 0),
                "cache_misses":      getattr(getattr(vram, "stats", None), "cache_misses", 0),
                "evictions":         getattr(getattr(vram, "stats", None), "evictions", 0),
                "defrag_runs":       getattr(getattr(vram, "stats", None), "defrag_runs", 0),
            }

        return {
            "timestamp": time.time(),
            "vram":      vram_block,
            "ram":       c["ram"].stats()   if c.get("ram")   else {},
            "swap":      c["swap"].stats()  if c.get("swap")  else {},
            "hybrid":    c["hybrid"].utilization() if c.get("hybrid") else {},
            "engine":    eng_stats,
            "system": {
                "cpu_pct":          round(psutil.cpu_percent(interval=0), 1),
                "ram_pct":          round(vm.percent, 1),
                "ram_available_mb": int(vm.available / 1024 / 1024),
                "uptime_s":         time.time() - getattr(req.app.state, "_start_time", time.time()),
                "req_rate":         req_rate,
            },
        }

    @app.get("/v1/performance")
    async def performance(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")

        from local_llm_server.monitoring.performance_analyzer import evaluate_sla
        snap     = req.app.state.ctx["metrics"].snapshot() if "metrics" in req.app.state.ctx else {}
        info     = req.app.state.ctx.get("sys_info")
        vram_gb  = sum(g.vram_total_mb for g in (info.gpus or [])) / 1024 if info else 0.0

        try:
            sla = evaluate_sla(
                vram_gb  = vram_gb,
                ttft_ms  = float(snap.get("avg_latency_ms", 0) or 0),
                tps      = float(snap.get("avg_tps", 0) or 0),
            )
            if isinstance(sla, dict):
                return sla
            if hasattr(sla, "to_dict"):
                return sla.to_dict()
            from dataclasses import asdict
            d = asdict(sla)
            for k, v in d.items():
                if hasattr(v, "value"):
                    d[k] = v.value
            return d
        except Exception as exc:
            log.error("SLA evaluation error: %s", exc)
            return {
                "grade":           "degraded",
                "ttft_ms":         round(float(snap.get("avg_latency_ms", 0) or 0), 2),
                "tps":             round(float(snap.get("avg_tps", 0) or 0), 2),
                "vram_gb":         round(vram_gb, 2),
                "recommendations": ["Unable to compute SLA due to internal error"],
                "error":           str(exc),
            }

    @app.get("/v1/drive/stats")
    async def drive_stats(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        e        = req.app.state.ctx.get("engine")
        ldr      = req.app.state.ctx.get("loader")
        pressure = req.app.state.ctx.get("pressure")
        profile  = req.app.state.ctx.get("hw_profile")

        pressure_stats = None
        if pressure:
            try:
                result = pressure.check_and_enforce_async()
                if asyncio.iscoroutine(result):
                    pressure_stats = await result
                else:
                    pressure_stats = result
            except Exception as exc:
                pressure_stats = {"error": str(exc)}

        def _safe(fn, *a, **kw):
            try: return fn(*a, **kw)
            except Exception as exc: return {"error": str(exc)}

        return {
            "drive_wired":     _engine_get_drive_wired(e) if e else False,
            "context_cache":   _safe(_engine_get_ctx_cache_stats, e) if e else None,
            "throughput":      _safe(_engine_get_throughput_stats, e) if e else None,
            "ngram_cache":     {"hit_rate": _engine_get_ngram_hit_rate(e) if e else 0.0},
            "cpu_throttler":   _safe(ldr.cpu_throttler_stats) if ldr else {},
            "memory_pressure": pressure_stats,
            "hw_profile":      getattr(profile, "name", "unknown"),
            "concurrency":     _engine_get_concurrency(e) if e else 0,
        }

    @app.delete("/v1/context-cache/clear")
    async def clear_ctx_cache(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        e = req.app.state.ctx.get("engine")
        return {"status": "cleared" if (e and _engine_clear_ctx_cache(e)) else "no_cache"}

    @app.get("/v1/telemetry")
    async def telemetry_ep(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        t = req.app.state.ctx.get("telemetry")
        return {
            "models":          t.all_profiles()    if t else [],
            "recommendations": t.recommendations() if t else {},
        }

    @app.get("/v1/middleware/stats")
    async def middleware_stats(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        return {"rate_limiter": ip_rl.stats() if hasattr(ip_rl, "stats") else {}}

    @app.get("/v1/throughput/stats")
    async def throughput_stats(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        opt = req.app.state.ctx.get("optimizer")
        return opt.stats() if opt and hasattr(opt, "stats") else {"enabled": False}

    @app.get("/v1/cluster")
    async def cluster_info(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        return {"nodes": 1, "role": "standalone", "version": VERSION}

    @app.get("/v1/gpu/analyze")
    async def gpu_analyze(req: Request, key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        analyzer = req.app.state.ctx.get("analyzer")
        if not analyzer:
            return {"analysis": "unavailable"}
        try:
            return analyzer.full_analysis()
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/v1/backend")
    async def backend_info(key_info: tuple = Depends(verify_api_key)):
        is_valid, _, _ = key_info
        if not is_valid:
            raise HTTPException(403, "Invalid or missing API key")
        try:
            from local_llm_core.engine.bitnet import get_status
            return get_status()
        except Exception as exc:
            return {"error": str(exc)}

    # ──────────────────────────────────────────────────────────────────────
    # Admin endpoints
    # ──────────────────────────────────────────────────────────────────────
    @app.post("/admin/keys/create", dependencies=[Depends(verify_master_key)])
    async def admin_create_key(
        body:       CreateKeyRequest,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        try:
            plain_key, key_hash = await store.create_key(
                body.key_id, body.rate_limit_rps, body.rate_limit_capacity,
                body.description, body.expires_in_seconds, body.active,
                max_output_tokens=body.max_output_tokens,
                max_request_seconds=body.max_request_seconds,
                max_requests_per_day=body.max_requests_per_day,
            )
        except KeyAlreadyExistsError:
            raise HTTPException(
                409,
                f"Key '{body.key_id}' already exists. Use a different id, "
                f"or revoke it first: POST /admin/keys/revoke/{body.key_id}"
            )
        _log_admin_action(master_key, "create_key", target=body.key_id, success=True)
        return {"key_id": body.key_id, "plain_key": plain_key, "key_hash": key_hash}

    @app.post("/admin/keys/revoke/{key_id}", dependencies=[Depends(verify_master_key)])
    async def admin_revoke_key(
        key_id:     str,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        if not await store.revoke_key(key_id):
            raise HTTPException(404, f"Key '{key_id}' not found")
        _log_admin_action(master_key, "revoke_key", target=key_id, success=True)
        return {"status": "revoked", "key_id": key_id}

    @app.post("/admin/keys/activate/{key_id}", dependencies=[Depends(verify_master_key)])
    async def admin_activate_key(
        key_id:     str,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        if not await store.activate_key(key_id):
            raise HTTPException(404, f"Key '{key_id}' not found")
        _log_admin_action(master_key, "activate_key", target=key_id, success=True)
        return {"status": "activated", "key_id": key_id}

    @app.patch("/admin/keys/update/{key_id}", dependencies=[Depends(verify_master_key)])
    async def admin_update_key(
        key_id:     str,
        body:       UpdateKeyRequest,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        if body.rate_limit_rps is not None or body.rate_limit_capacity is not None:
            stats = await store.get_key_stats(key_id)
            if not stats:
                raise HTTPException(404, f"Key '{key_id}' not found")
            await store.update_key_limits(
                key_id,
                body.rate_limit_rps        if body.rate_limit_rps        is not None else stats["rate_limit_rps"],
                body.rate_limit_capacity   if body.rate_limit_capacity   is not None else stats["rate_limit_capacity"],
            )
        # Quotas None means "leave unchanged", per the model
        # field default; the update_key_quotas helper handles that.
        if (body.max_output_tokens is not None
                or body.max_request_seconds is not None
                or body.max_requests_per_day is not None):
            upd = getattr(store, "update_key_quotas", None)
            if upd is not None:
                await upd(
                    key_id,
                    max_output_tokens=body.max_output_tokens,
                    max_request_seconds=body.max_request_seconds,
                    max_requests_per_day=body.max_requests_per_day,
                )
        if body.active is not None:
            if body.active:
                await store.activate_key(key_id)
            else:
                await store.revoke_key(key_id)
        _log_admin_action(master_key, "update_key", target=key_id, success=True)
        return {"status": "updated", "key_id": key_id}

    @app.delete("/admin/keys/delete/{key_id}", dependencies=[Depends(verify_master_key)])
    async def admin_delete_key(
        key_id:     str,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        if not await store.delete_key(key_id):
            raise HTTPException(404, f"Key '{key_id}' not found")
        _log_admin_action(master_key, "delete_key", target=key_id, success=True)
        return {"status": "deleted", "key_id": key_id}

    @app.get("/admin/keys/list", dependencies=[Depends(verify_master_key)])
    async def admin_list_keys(
        req:         Request,
        active_only: bool = Query(True),
        quiet:       bool = Query(False, description="When True, suppresses INFO-level audit log."),
        master_key:  str  = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        keys = await store.list_keys(active_only=active_only)
        if quiet:
            if len(keys) > 1000:
                log.warning(
                    "ADMIN(quiet-poll): key=%s... action=list_keys count=%d "
                    "(unusually large — check for key-store bloat)",
                    master_key[:8], len(keys),
                )
        else:
            _log_admin_action(master_key, "list_keys", success=True)
        return {"keys": keys}

    @app.get("/admin/keys/stats/{key_id}", dependencies=[Depends(verify_master_key)])
    async def admin_key_stats(
        key_id:     str,
        req:        Request,
        master_key: str = Depends(_master_key_header),
    ):
        if not admin_rate_limiter.allow(master_key):
            raise HTTPException(429, "Too many admin requests")
        store = req.app.state.ctx.get("key_store")
        if not store:
            raise HTTPException(503, "Key store not enabled")
        stats = await store.get_key_stats(key_id)
        if not stats:
            raise HTTPException(404, f"Key '{key_id}' not found")
        _log_admin_action(master_key, "key_stats", target=key_id, success=True)
        return stats

    return app

# =============================================================================
# Graceful serving
# =============================================================================
def _serve_gracefully(app, *, drain_timeout: float, **uvicorn_kwargs) -> None:
    """Run uvicorn so that a stop/restart never cuts a generation in half.

    Why this exists: on this hardware prefill alone can take minutes, so there
    is almost always a stream in flight. Two things used to kill it abruptly:

      * SIGHUP (``serve --restart``) is not among uvicorn's handled signals,
        so the default action terminated the process on the spot - no drain,
        no lifespan shutdown, connection cut mid-chunk. It now maps to the same
        graceful path as SIGTERM.
      * Without ``timeout_graceful_shutdown`` the drain window was undefined;
        combined with systemd's ``TimeoutStopSec`` it ended in SIGKILL.

    On SIGTERM/SIGINT/SIGHUP uvicorn stops accepting new connections
    immediately, lets in-flight requests finish for up to ``drain_timeout``
    seconds (0 = wait indefinitely), and only then runs the lifespan shutdown
    that unloads the model. ``TimeoutStopSec`` in the unit file must be larger
    than ``drain_timeout``. Clients that connect while the server is draining
    or restarting wait for it and retry (see AsyncLLMCloudClient).
    """
    import signal
    import uvicorn

    kwargs = dict(uvicorn_kwargs)
    if drain_timeout and drain_timeout > 0:
        kwargs["timeout_graceful_shutdown"] = int(drain_timeout)
    try:
        config = uvicorn.Config(app, **kwargs)
    except TypeError:
        # uvicorn < 0.24 has no timeout_graceful_shutdown
        kwargs.pop("timeout_graceful_shutdown", None)
        config = uvicorn.Config(app, **kwargs)
    server = uvicorn.Server(config)

    def _on_sighup(signum, frame):
        log.warning(
            "SIGHUP received - draining in-flight requests "
            "(up to %ss), then shutting down gracefully", drain_timeout or "unlimited",
        )
        server.should_exit = True

    with suppress(Exception):  # no SIGHUP on Windows / not the main thread
        signal.signal(signal.SIGHUP, _on_sighup)
    server.run()

# =============================================================================
# CLI entry point
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m local_llm_core.api_server")
    p.add_argument("subcommand", nargs="?", default="serve")
    p.add_argument("--host", default=None,
                   help="Override bind host (default: server.host from config, else 127.0.0.1)")
    p.add_argument("--port", type=int, default=None,
                   help="Override bind port (default: server.port from config, else 8080)")
    p.add_argument("--config", default="config.yaml", dest="config_path")
    p.add_argument("--log-level", default="info", dest="log_level",
                   choices=["debug", "info", "warning", "error"], type=str.lower)
    p.add_argument("--log-file", default=_default_state_dir() / "llm_server.log", dest="log_file")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--drain-timeout", type=float, default=120.0, dest="drain_timeout",
                   help="Seconds to let in-flight generations finish on stop/restart "
                        "before cutting them (0 = wait indefinitely). Keep the systemd "
                        "unit's TimeoutStopSec larger than this.")
    p.add_argument("--reload", action="store_true")
    p.add_argument("--restart", action="store_true", help="Force restart (send SIGHUP)")
    p.add_argument("--stop", action="store_true", help="Send SIGTERM to running server")
    args = p.parse_args(argv)

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    for _noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    _missing = []
    for _pkg, _pip in [
        ("fastapi",   "fastapi"),
        ("uvicorn",   "uvicorn"),
        ("pydantic",  "pydantic"),
        ("yaml",      "pyyaml"),
        ("psutil",    "psutil"),
        ("aiosqlite", "aiosqlite"),
    ]:
        try:
            __import__(_pkg)
        except ImportError:
            _missing.append((_pkg, _pip))

    if _missing:
        sys.stderr.write("\n  ❌  Missing dependencies:\n")
        for pkg, pip_name in _missing:
            sys.stderr.write(f"    • {pkg}  →  pip install \"{pip_name}\"\n")
        sys.stderr.write("\n  pip install -r requirements.txt\n\n")
        sys.exit(1)

    cfg_path = Path(args.config_path)
    if not cfg_path.exists():
        sys.stderr.write(f"[ERROR] Config not found: {args.config_path}\n")
        sys.exit(1)
    if cfg_path.suffix.lower() not in (".yaml", ".yml"):
        sys.stderr.write(f"[ERROR] Config must be .yaml/.yml: {args.config_path}\n")
        sys.exit(1)

    if args.stop:
        pid_file = _default_state_dir() / "server.pid"
        if Path(pid_file).exists():
            pid = int(Path(pid_file).read_text().strip())
            os.kill(pid, 15)
            sys.stdout.write(f"Sent SIGTERM to pid {pid}\n")
        else:
            sys.stderr.write("No PID file found at /tmp/llm_server.pid\n")
        return

    if args.restart:
        pid_file = _default_state_dir() / "server.pid"
        if Path(pid_file).exists():
            pid = int(Path(pid_file).read_text().strip())
            import signal
            os.kill(pid, signal.SIGHUP)
            sys.stdout.write(f"Sent SIGHUP to pid {pid}\n")
        else:
            sys.stderr.write("No PID file found — starting fresh\n")

    sys.stdout.write(f"\n  ⚡  Local LLM Cloud v{VERSION} — {args.host}:{args.port}\n")
    sys.stdout.write(f"  Config : {cfg_path.resolve()}\n")
    sys.stdout.write(f"  Docs   : http://localhost:{args.port}/docs\n")
    sys.stdout.write(f"  Health : http://localhost:{args.port}/health\n\n")
    sys.stdout.flush()

    _pid_path = _default_state_dir() / "server.pid"
    with suppress(Exception):
        _pid_path.write_text(str(os.getpid()))

    _uvicorn_log_cfg = {
        "version": 1, "disable_existing_loggers": False,
        "handlers": {"null": {"class": "logging.NullHandler"}},
        "loggers": {
            "uvicorn":        {"handlers": ["null"], "level": "INFO",    "propagate": True},
            "uvicorn.error":  {"handlers": ["null"], "level": "INFO",    "propagate": True},
            "uvicorn.access": {"handlers": ["null"], "level": "WARNING", "propagate": True},
        },
    }

    cfg_str = str(cfg_path.resolve())
    import uvicorn
    try:
        if args.workers > 1:
            os.environ["LLM_CONFIG_PATH"] = cfg_str
            uvicorn.run(
                "local_llm_core.api_server:create_app",
                host=args.host, port=args.port,
                workers=args.workers,
                log_level=args.log_level,
                log_config=_uvicorn_log_cfg,
                access_log=False,
                factory=True,
            )
        elif args.reload:
            uvicorn.run("local_llm_server.serving.api_server:create_app", factory=True, reload=True,
                        reload_dirs=[str(Path(__file__).resolve().parent.parent)],
                        host=args.host, port=args.port, log_level=args.log_level,
                        log_config=_uvicorn_log_cfg, access_log=False)
        else:
            _app = create_app(cfg_str)
            _cfg_host = (_app.state.config or {}).get("server", {}).get("host", "127.0.0.1")
            _cfg_port = int((_app.state.config or {}).get("server", {}).get("port", 8080))
            _serve_gracefully(
                _app,
                drain_timeout=args.drain_timeout,
                host=args.host or _cfg_host,
                port=args.port or _cfg_port,
                log_level=args.log_level,
                log_config=_uvicorn_log_cfg,
                access_log=False,
            )
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n  ⚡  Local LLM Cloud — stopped.\n\n")
        sys.stdout.flush()
        with suppress(Exception):
            (_default_state_dir() / "server.pid").unlink()


if __name__ == "__main__":
    main()
