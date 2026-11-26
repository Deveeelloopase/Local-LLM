"""
local_llm_client/core/async_client.py

Production‑ready asynchronous HTTP client for Local LLM Cloud.

Features:
- Uses aiohttp (preferred) with connection pooling and timeouts
- Falls back to a truly asynchronous streaming fallback if aiohttp is missing
- Circuit breaker with configurable thresholds and automatic recovery
- Exponential backoff retry with jitter
- Word‑buffered streaming via ChunkBuffer (from _accel)
- Fast token counting via estimate_tokens_fast
- Full coverage of all server endpoints (model management, key admin, logs, etc.)
- Automatic master‑key header injection for admin routes
- Dynamic API key update via set_api_key()
- Model readiness checks using server‑side status field
- No hardcoded paths – all configuration via ClientConfig
- Checkpointed, resumable generation (Section 2) – resume() method
- Server‑restart tolerance: streams wait out a restarting server and either
  replay safely (before any token) or surface a StreamInterruptedError
  carrying the partial text (after tokens began flowing).

 REAL STREAM STATS
────────────────────────────────────────────────────────────────────────
This drop replaces that pattern with:

  1. `StreamResult` — an `AsyncIterator[str]` wrapper with a `stats`
     attribute that becomes populated with `ServerStats` (a real dataclass
     from `formatter.py`) once the stream exhausts.

  2. `X-Request-ID` capture — every streaming response carries the
     server's request id in a response header. We capture it before the
     body starts.

  3. `GET /v1/requests/{id}/stats` fetch — after the stream ends, if
     `ClientConfig.fetch_server_stats=True` (default) and we have an id,
     we call the server's authoritative per-request stats endpoint. The response populates `stats_source="server"` — every
     number is exact.

  4. Chunk fallback — if the fetch fails or is disabled (old server,
     network blip, `fetch_server_stats=False`), we fall back to counting
     streamed chunks client-side. For SSE (`stream_format="json"`) this is
     exact: one chunk == one token per server contract. For plain text
     (`stream_format="text"`) it's approximate (ChunkBuffer word-buffers)
     — marked `stats_source="chunks"` so the display layer can add `~`.

  5. Client-measured timing — TTFT and total time come from our own
     `time.monotonic()` wall clock, so they are always exact regardless
     of stats_source. Only `prefill_tps`, `completion_tokens`, and
     `decode_tps` depend on the source.

Display contract (enforced by `local_llm_client/core/formatter.py`):
  stats_source="server"   → clean output (authoritative)
  stats_source="chunks"   → `~` on token count, `(est.)` suffix
  stats_source="unknown"  → `—` for numeric fields

Callers should treat `StreamResult.stats` as `None` until iteration
completes normally. Accessing it mid-stream returns `None`.

version = 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Set

from .models import ClientConfig
from .exceptions import (
    AuthError,
    BadRequestError,
    CircuitOpenError,
    ConflictError,
    ConnectionError as LLMConnectionError,
    LLMCloudError,
    ModelNotLoadedError,
    NotFoundError,
    PayloadTooLargeError,
    QueueFullError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    StreamError,
    StreamInterruptedError,
    TimeoutError as LLMTimeoutError,
    UnsupportedMediaTypeError,
)
from ._accel import ChunkBuffer, estimate_tokens_fast
from .formatter import ServerStats

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# aiohttp availability
# ------------------------------------------------------------------------------
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    _HAS_AIOHTTP = False


# Plain-text streaming routes interleave "\x1f STATUS:{json} \x1e" heartbeat
# frames with the answer. They are bytes on the wire, not answer content.
_HB_FRAME_RE = re.compile(r"\x1f[^\x1e]*\x1e")

# fire-and-forget `loop.create_task` calls need a strong reference
# # or the CPython GC can drop the Task before its coroutine runs. The
# # `__del__` path below is the only such site in this module; the client's
# # normal request paths await their tasks and do not need this set.
_CLOSE_GUARD: "set" = set()

def _is_connection_drop(exc: BaseException) -> bool:
    """True when ``exc`` means the TCP connection went away (server restart,
    crash, kill, network reset) rather than the server answering with an error.

    The classic signature is aiohttp's ``ClientPayloadError: Response payload
    is not completed ... Not enough data to satisfy transfer length header``,
    raised when a chunked response is cut off before its terminating chunk."""
    if isinstance(exc, (OSError, asyncio.IncompleteReadError)):
        return True
    if _HAS_AIOHTTP:
        return isinstance(
            exc,
            (
                aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
            ),
        )
    return False


def _as_llm_error(exc: BaseException) -> BaseException:
    """Map raw transport errors onto the client's own exception hierarchy so
    callers (and the CLI) only ever have to handle ``LLMCloudError``."""
    if isinstance(exc, LLMCloudError):
        return exc
    if _is_connection_drop(exc) or (_HAS_AIOHTTP and isinstance(exc, aiohttp.ClientError)):
        return LLMConnectionError(f"{type(exc).__name__}: {exc}")
    return exc


# ------------------------------------------------------------------------------
# StreamResult —  streaming with real stats after exhaustion
# ------------------------------------------------------------------------------
class StreamResult:
    """`AsyncIterator[str]` wrapper that exposes `.stats` after exhaustion.

    All streaming methods on `AsyncLLMCloudClient` (`chat_stream`, `stream`,
    `resume(stream=True)`) return a `StreamResult` instead of a bare async
    generator. The iteration protocol is identical:

        async for chunk in client.chat_stream(...):   # yields str
            print(chunk, end="")
        # stream.stats is now populated

    `.stats` is `None` until the async iteration completes normally (or
    fails and the fallback path populated it). Accessing `.stats` mid-
    stream returns `None` — do not poll it during iteration.

    On stream completion (success, StreamInterruptedError, or partial
    cancellation), `.stats` is a `ServerStats` object with:

      • `stats_source="server"` — when `/v1/requests/{id}/stats` succeeded
        (authoritative; includes prefill_tps, thinking_tokens, etc.)
      • `stats_source="chunks"` — when only client-side chunk counting was
        available (exact for SSE, approximate for plain text)
      • `stats_source="unknown"` — when the stream produced no output and
        no server stats existed (e.g. cancelled before the first chunk)

    Attributes:
        stats: Optional[ServerStats] — populated after iteration completes.
    """
    __slots__ = ("_impl", "stats")

    def __init__(self) -> None:
        self._impl: Optional[AsyncIterator[str]] = None
        self.stats: Optional[ServerStats] = None

    def __aiter__(self) -> AsyncIterator[str]:
        if self._impl is None:
            # Defensive: caller constructed StreamResult directly instead of
            # receiving it from a client method. Only the client sets _impl.
            raise RuntimeError(
                "StreamResult is uninitialized — obtain it from "
                "client.chat_stream(), client.stream(), or "
                "await client.resume(stream=True) instead of constructing it."
            )
        return self._impl

    async def aclose(self) -> None:
        """Close the underlying async generator (if still open).

        Safe to call multiple times; idempotent. Callers that break out of
        `async for` early should call this so the server sees a disconnect
        and the connection is released promptly.
        """
        if self._impl is not None:
            close = getattr(self._impl, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass


# ------------------------------------------------------------------------------
# AsyncLLMCloudClient
# ------------------------------------------------------------------------------
class AsyncLLMCloudClient:
    """
    Async client for Local LLM Cloud. Compatible with asyncio and when using ai_agent.py.

    Usage:
        async with AsyncLLMCloudClient() as client:
            text = await client.generate("Hello!")
            stream = client.stream("Tell me a story")
            async for chunk in stream:
                print(chunk, end="", flush=True)
            print(stream.stats.decode_tps)   # real number
    """

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self.cfg = config or ClientConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._closed = False

        # Circuit breaker – uses ClientConfig fields if present
        self._cb_failures = 0
        self._cb_state = "closed"  # closed | open | half_open
        self._cb_opened_at = 0.0
        self._cb_lock = asyncio.Lock()
        self._half_open_probe_timeout = getattr(self.cfg, "half_open_probe_timeout_s", 10.0)
        self._half_open_probe_started = 0.0
        # Wall-clock throttle for `/livez` probes while the breaker is
        # open — see `_cb_allow`. 0.0 means "never probed".
        self._cb_last_probe_ts = 0.0
        # Recovery-window short-circuit: once a probe closes the breaker,
        # this is how we know to skip the remaining wait.
        self._cb_early_recovery_window_s = float(
            getattr(self.cfg, "circuit_breaker_early_probe_interval_s", 2.0)
        )

        # Master key (for admin endpoints)
        self._master_key: Optional[str] = getattr(self.cfg, "master_key", None)

        # Optional progress hook. Called with a short human-readable message
        # while a streaming request waits for a restarting server, so a UI can
        # show "server restarting" instead of a frozen spinner. Never required.
        self.on_stream_event: Optional[Callable[[str], None]] = None

        # Ensure missing config defaults
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        """Add missing configuration fields with safe defaults."""
        defaults = {
            "retry_delay_max_s": 30.0,
            "retry_backoff_factor": 2.0,
            "connect_timeout_s": 5.0,
            "pool_maxsize": 50,
            # stats-fetch controls (present in ClientConfig from
            # guard here so a caller using a stub ClientConfig gets
            # safe defaults rather than AttributeError).
            "fetch_server_stats": True,
            "stats_fetch_timeout_s": 5.0,
        }
        for attr, default in defaults.items():
            if not hasattr(self.cfg, attr):
                setattr(self.cfg, attr, default)

    # --------------------------------------------------------------------------
    # API key management
    # --------------------------------------------------------------------------
    def set_api_key(self, key: str) -> None:
        """Update the API key used for subsequent requests."""
        self.cfg.api_key = key

    def set_master_key(self, key: str) -> None:
        """Set the master key used for admin endpoints."""
        self._master_key = key

    @property
    def is_authenticated(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.cfg.api_key)

    # --------------------------------------------------------------------------
    # Session management
    # --------------------------------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a reusable aiohttp session, creating one if necessary."""
        if not _HAS_AIOHTTP:
            raise ImportError(
                "aiohttp is required for AsyncLLMCloudClient. "
                "Install it: pip install aiohttp"
            )

        async with self._lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self.cfg.pool_maxsize,
                    limit_per_host=self.cfg.pool_maxsize,
                )
                timeout = aiohttp.ClientTimeout(
                    connect=self.cfg.connect_timeout_s,
                    total=self.cfg.timeout,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
            self._closed = True

    async def __aenter__(self) -> "AsyncLLMCloudClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def __del__(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # keep a strong reference until the coroutine
                # finishes; the done-callback removes the entry so the
                # set does not grow with client churn.
                _t = loop.create_task(self.close())
                _CLOSE_GUARD.add(_t)
                _t.add_done_callback(_CLOSE_GUARD.discard)
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass

    @staticmethod
    def _is_connection_refused(exc: BaseException) -> bool:
        """True only for 'server is not accepting connections' — the
        exact signature of a systemd restart in flight.

        Deliberately narrower than `_is_connection_drop`:
          * `ClientConnectorError` for DNS / TLS / cert failures is
            NOT a restart — it counts as a real failure;
          * `ConnectionResetError` mid-stream is NOT a restart — it
            counts (something killed the connection; worth flagging);
          * `ServerDisconnectedError` on a keep-alive pool connection
            during a restart IS a restart — counts as refused.

        Used by `_cb_failure` to skip the threshold check when the
        signal is 'the port is not listening right now', which on this
        deployment is what a systemd restart looks like for 15-25 s.
        """
        import errno as _errno
        if isinstance(exc, ConnectionRefusedError):
            return True
        if _HAS_AIOHTTP:
            if isinstance(exc, aiohttp.ClientConnectorError):
                try:
                    os_err = getattr(exc, "os_error", None)
                    if os_err is not None and getattr(os_err, "errno", None) == _errno.ECONNREFUSED:
                        return True
                except Exception:
                    pass
            if isinstance(exc, aiohttp.ServerDisconnectedError):
                return True
        return False

    async def _probe_livez_locked(self) -> bool:
        """Probe `/livez`. Called while `_cb_lock` is held.

        2 s timeout so `_cb_allow` cannot be stalled for long. `/livez`
        is public — no auth needed — but sending headers costs nothing
        and keeps the code uniform with `_wait_until_ready`.
        """
        if not _HAS_AIOHTTP:
            return False
        try:
            session = await self._get_session()
            async with session.get(
                self.cfg.base_url + "/livez",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as r:
                return r.status == 200
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # Circuit breaker
    # --------------------------------------------------------------------------
    async def _cb_allow(self) -> bool:
        """Gate every request: closed → always allow; open → probe.

        Behaviour change:

          * The recovery window is still `circuit_breaker_recovery`
            seconds (default 60 s). We do not touch that constant —
            operators can still configure it.
          * While in the *open* state and *before* the window has
            elapsed, `/livez` is probed every
            `circuit_breaker_early_probe_interval_s` seconds (default
            2 s). A 200 closes the breaker immediately, letting the
            client resume the moment systemd's restart completes
            (~15-25 s on this box), instead of sitting out the rest of
            the 60 s window.
          * Once `_cb_state` transitions to closed via the probe, the
            next request runs normally.
          * The probe is throttled so a burst of client requests
            during an outage does not hammer `/livez` — the first
            caller in the 2 s window does the probe; the rest see the
            updated state.

        Deadlock note: `_probe_livez_locked` calls `_get_session`,
        which acquires `self._lock` (the session lock) — not
        `self._cb_lock` (the one we hold here). Different locks, no
        nesting risk. Nothing else in the client acquires `_cb_lock`
        while holding `_lock`, so the ordering is acyclic.
        """
        if getattr(self.cfg, "circuit_breaker_enabled", True) and self.cfg.max_retries <= 0:
            return True
        async with self._cb_lock:
            now = time.monotonic()
            if self._cb_state == "closed":
                return True
            if self._cb_state == "open":
                elapsed = now - self._cb_opened_at
                # Full recovery window elapsed → allow one half-open
                # probe (existing semantics, unchanged).
                if elapsed >= self.cfg.circuit_breaker_recovery:
                    self._cb_state = "half_open"
                    self._half_open_probe_started = now
                    return True
                # Otherwise, probe early: a systemd restart usually
                # finishes well inside the recovery window, and the
                # client should not sit idle when the server is already
                # back.
                if (now - self._cb_last_probe_ts) >= self._cb_early_recovery_window_s:
                    self._cb_last_probe_ts = now
                    _ok = await self._probe_livez_locked()
                    if _ok:
                        log.info(
                            "Circuit breaker: /livez answered after %.1fs "
                            "(recovery window is %.1fs) — closing early",
                            elapsed, self.cfg.circuit_breaker_recovery,
                        )
                        self._cb_state = "closed"
                        self._cb_failures = 0
                        return True
                return False
            if self._cb_state == "half_open":
                if now - self._half_open_probe_started < self._half_open_probe_timeout:
                    return True
                self._cb_state = "open"
                self._cb_opened_at = now
                return False
            return False

    async def _cb_success(self) -> None:
        async with self._cb_lock:
            self._cb_failures = 0
            self._cb_state = "closed"

    async def _cb_failure(self, exc: Optional[BaseException] = None) -> None:
        """Record a failure.

        Connection-refused (and equivalent signatures of 'the server's
        TCP port is not accepting right now') does NOT count toward the
        threshold. Rationale: on this deployment, systemd restarts the
        server on any crash or watchdog timeout, and a restart presents
        to the client as one of the following for ~15-25 s:
          * `ConnectionRefusedError`
          * `aiohttp.ServerDisconnectedError` (keep-alive pool)
          * `ClientConnectorError` wrapping `ECONNREFUSED`

        None of those indicate that the *service* is malfunctioning —
        they indicate that the *process* is cycling, which the client
        should ride out, not lock out. `_cb_allow` closes the breaker
        as soon as `/livez` answers, so if the restart is healthy the
        next request succeeds; if the restart is in a crash loop, `/livez`
        never answers and the breaker stays open, which is the correct
        state.

        All other failure modes (timeouts, 5xx, TLS handshake failures,
        real network drops) DO count — they indicate real trouble.
        """
        if exc is not None and self._is_connection_refused(exc):
            log.info(
                "Circuit breaker: connection-refused (likely restart) — "
                "not counting toward threshold (state=%s, failures=%d)",
                self._cb_state, self._cb_failures,
            )
            return
        async with self._cb_lock:
            self._cb_failures += 1
            threshold = getattr(self.cfg, "circuit_breaker_threshold", self.cfg.max_retries)
            if self._cb_failures >= threshold and self._cb_state != "open":
                self._cb_state = "open"
                self._cb_opened_at = time.monotonic()
                log.warning(
                    "AsyncLLMCloud circuit breaker OPEN after %d failures "
                    "(recovery=%.0fs, early-probe interval=%.1fs)",
                    self._cb_failures, self.cfg.circuit_breaker_recovery,
                    self._cb_early_recovery_window_s,
                )

    # --------------------------------------------------------------------------
    # Headers
    # --------------------------------------------------------------------------
    def _headers(self, extra: Optional[Dict] = None, admin: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.cfg.api_key:
            h["X-API-Key"] = self.cfg.api_key
        if admin and self._master_key:
            h["X-Master-Key"] = self._master_key
        if extra:
            h.update(extra)
        return h

    # --------------------------------------------------------------------------
    # Core request with retry and circuit breaker
    # --------------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        timeout: Optional[float] = None,
        admin: bool = False,
    ) -> Dict[str, Any]:
        """Send request, parse JSON, handle retries + circuit breaker."""
        if not await self._cb_allow():
            raise CircuitOpenError(
                f"Circuit breaker OPEN — retry after {self.cfg.circuit_breaker_recovery:.0f}s"
            )

        url = self.cfg.base_url + path
        t_out = timeout or self.cfg.timeout
        last_exc: Exception = RuntimeError("no attempt")

        # Fallback if aiohttp is missing
        if not _HAS_AIOHTTP:
            log.warning(
                "aiohttp not installed – using slow sync fallback. "
                "Install aiohttp for better performance: pip install aiohttp"
            )
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(
                    None, self._sync_request, method, url, body, t_out, admin
                )
                await self._cb_success()
                return result
            except Exception:
                await self._cb_failure()
                raise

        for attempt in range(self.cfg.max_retries + 1):
            try:
                session = await self._get_session()
                req_timeout = aiohttp.ClientTimeout(total=t_out)

                async with session.request(
                    method, url,
                    json=body,
                    headers=self._headers(admin=admin),
                    timeout=req_timeout,
                ) as resp:
                    status = resp.status
                    raw = await resp.read()
                    self._check_status(status, raw, path)
                    await self._cb_success()
                    data = json.loads(raw)
                    # Additional auth check
                    if isinstance(data, dict) and "detail" in data and isinstance(data["detail"], str):
                        if "Invalid or missing API key" in data["detail"]:
                            raise AuthError(data["detail"])
                    return data

            except (CircuitOpenError, AuthError, ModelNotLoadedError):
                raise

            # propagate 503 with structured error_code immediately.
            # _check_status populated error_code + retry_after; the caller
            # (chat.py) knows how to honour retry_after and distinguish
            # "model_starting" from "pool_degraded". Retrying here with
            # generic exponential backoff (1s, 2s, 4s = 7s of delay) would
            # only ADD latency on top of the caller's own retry loop —
            # and could push a single request past the caller's deadline
            # for no benefit.
            except ServiceUnavailableError:
                raise

            except aiohttp.ClientConnectorError as exc:
                last_exc = LLMConnectionError(f"Cannot connect to {url}: {exc}")
                await self._cb_failure(exc)

            except asyncio.TimeoutError:
                last_exc = LLMTimeoutError(f"Request to {url} timed out ({t_out}s)")
                await self._cb_failure()

            except Exception as exc:
                last_exc = exc
                await self._cb_failure(exc)

            if attempt < self.cfg.max_retries:
                delay = self.cfg.retry_delay * (self.cfg.retry_backoff ** attempt)
                delay = min(delay, self.cfg.retry_delay_max_s)
                jitter = random.uniform(-0.1, 0.1)
                delay = delay * (1 + jitter)
                log.warning(
                    "Async request failed (attempt %d/%d): %s — retrying in %.2fs",
                    attempt + 1, self.cfg.max_retries + 1, last_exc, delay
                )
                await asyncio.sleep(delay)

        raise last_exc

    def _check_status(self, status: int, body: bytes, path: str) -> None:
        """Translate an HTTP error response into a typed LLMCloudError.

        parses structured error bodies so every raised exception
        carries ``error_code`` (machine-readable) and ``retry_after``
        (seconds) — the two fields ``chat.py`` and the Web UI now branch
        on to distinguish "retry (model_starting)" from "give up
        (pool_degraded)" from "not retryable (404)".

        Body shapes accepted:
          * our own:       {"error": {"code": "...", "message": "...",
                                      "retry_after": 5.0}}
          * FastAPI 422:   {"detail": "..."} or {"detail": [...]}
          * generic:       {"message": "..."}
          * plain text:    the raw body becomes the message

        Backwards compatibility: 503 bodies with ``"not loaded"`` in the
        text (no structured code) still raise ModelNotLoadedError; 503
        bodies with ``"queue"`` still raise QueueFullError. Those are
        what the API used to emit before the pool became structured.
        """
        if status == 200:
            return

        raw_body = ""
        if body:
            try:
                raw_body = body.decode("utf-8", errors="replace")[:500]
            except Exception:
                raw_body = ""

        # ── Parse structured error payload ──────────────────────────────
        error_code: Optional[str] = None
        error_message: Optional[str] = None
        retry_after: float = 0.0
        if body:
            try:
                parsed = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    _c = err.get("code")
                    if isinstance(_c, str) and _c:
                        error_code = _c
                    _m = err.get("message")
                    if isinstance(_m, str) and _m:
                        error_message = _m
                    _r = err.get("retry_after")
                    if _r is not None:
                        try:
                            retry_after = float(_r)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                elif isinstance(parsed.get("detail"), str):
                    # FastAPI default error shape
                    error_message = parsed["detail"]
                elif isinstance(parsed.get("message"), str):
                    error_message = parsed["message"]

        msg = error_message or raw_body or f"HTTP {status}"

        # ── 401 / 403 — auth ────────────────────────────────────────────
        if status in (401, 403):
            raise AuthError(
                f"Authentication failed ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )

        # ── 503 legacy shapes (no structured code) ──────────────────────
        if status == 503 and error_code is None and (
            "No model" in msg or "not loaded" in msg.lower()
        ):
            raise ModelNotLoadedError(
                "No model loaded on server.",
                error_code=error_code,
                retry_after=retry_after,
            )
        if status in (429, 503) and error_code is None and "queue" in msg.lower():
            raise QueueFullError(
                f"Server queue full ({status}): {msg}",
                retry_after=retry_after,
                error_code=error_code,
            )

        # ── 429 — rate limit ────────────────────────────────────────────
        if status == 429:
            raise RateLimitError(
                f"Rate limited ({status}): {msg}",
                retry_after=retry_after,
                error_code=error_code,
            )

        # ── 503 — service unavailable (with or without structured code) ──
        # Subclass of ServerError, distinct class so callers can catch
        # "pool is starting / degraded" separately from generic 5xx.
        if status == 503:
            raise ServiceUnavailableError(
                f"Service unavailable: {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )

        # ── 404 — unknown resource (e.g. unknown model id) ──────────────
        # Non-retryable. Previously a 404 fell through to the generic
        # `except Exception` retry path — callers retried for 7s before
        # giving up on an error that would never succeed.
        if status == 404:
            raise NotFoundError(
                f"Not found ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )

        # ── Other client-side 4xx (non-retryable) ───────────────────────
        if status == 400:
            raise BadRequestError(
                f"Bad request ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )
        if status == 409:
            raise ConflictError(
                f"Conflict ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )
        if status == 413:
            raise PayloadTooLargeError(
                f"Request too large ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )
        if status == 415:
            raise UnsupportedMediaTypeError(
                f"Unsupported media type ({status}): {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )

        # ── Other 5xx — retryable per class rule ────────────────────────
        if status >= 500:
            raise ServerError(
                f"Server error {status} on {path}: {msg}",
                error_code=error_code,
                retry_after=retry_after,
            )

        # ── Any other 4xx we did not enumerate ──────────────────────────
        raise LLMCloudError(
            f"HTTP {status} on {path}: {msg}",
            error_code=error_code,
            retry_after=retry_after,
        )

    def _sync_request(
        self,
        method: str,
        url: str,
        body: Optional[Dict],
        timeout: float,
        admin: bool = False,
    ) -> Dict:
        """Synchronous http.client fallback used when aiohttp is unavailable."""
        import http.client as hc
        import urllib.parse as up
        parsed = up.urlsplit(url)
        host = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn_cls = hc.HTTPSConnection if parsed.scheme == "https" else hc.HTTPConnection
        conn = conn_cls(host, timeout=timeout)
        payload = json.dumps(body).encode() if body else None
        headers = self._headers(admin=admin)
        if payload:
            headers["Content-Length"] = str(len(payload))
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            self._check_status(resp.status, raw, path)
            return json.loads(raw) if raw else {}
        except LLMCloudError:
            # do NOT wrap typed cloud-level exceptions (AuthError,
            # ServerError, ServiceUnavailableError, NotFoundError, ...)
            # in a connection-error wrapper. Before this, every non-2xx
            # response through the sync fallback — including AuthError
            # — emerged as LLMConnectionError, so callers that keyed on
            # exception type saw the wrong diagnosis.
            raise
        except (ConnectionRefusedError, OSError) as exc:
            raise LLMConnectionError(f"Cannot connect to {host}{path}: {exc}") from exc
        except Exception as exc:
            raise LLMConnectionError(f"http.client fallback failed: {exc}") from exc
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Server-restart tolerance helpers
    # --------------------------------------------------------------------------
    def _emit_stream_event(self, message: str) -> None:
        hook = self.on_stream_event
        if hook is None:
            return
        try:
            hook(message)
        except Exception:  # a broken UI callback must never break a request
            log.warning("on_stream_event hook failed", exc_info=True)

    async def _wait_until_ready(self, max_wait_s: float, poll_s: float = 1.0) -> bool:
        """Poll until the server can take a request again.

        Returns True as soon as ``/ready`` answers 200 (model loaded). Servers
        without ``/ready`` (404) fall back to ``/health``. Connection errors
        are expected while the process is down and are simply retried; the
        pooled keep-alive connections from before the restart are dead, which
        shows up as one extra ServerDisconnectedError and is handled the same
        way."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, max_wait_s)
        probe_timeout = aiohttp.ClientTimeout(total=3.0)
        delay = max(0.2, poll_s)
        while True:
            try:
                session = await self._get_session()
                async with session.get(
                    self.cfg.base_url + "/ready",
                    headers=self._headers(), timeout=probe_timeout,
                ) as r:
                    if r.status == 200:
                        return True
                    if r.status == 404:
                        async with session.get(
                            self.cfg.base_url + "/health",
                            headers=self._headers(), timeout=probe_timeout,
                        ) as h:
                            if h.status == 200:
                                return True
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 5.0)

    # --------------------------------------------------------------------------
    #  — per-request stats (server-authoritative or client-derived)
    # --------------------------------------------------------------------------
    async def _fetch_request_stats(self, request_id: str) -> Optional[ServerStats]:
        """Fetch `GET /v1/requests/{id}/stats` with one 100 ms retry.

        Retry rationale: the streaming protocol allows the server to send
        `data: [DONE]` a fraction of a millisecond before its own
        `_do_persist` background task has finished scheduling and writing
        the in-memory stats map entry. That is a real race — the server
        now writes the map entry *before* any Postgres call inside
        `_log_request`, but the task spawn itself still takes one event-
        loop cycle. A single 100 ms retry absorbs that delay on every
        real deployment.

        A 404 on the *second* attempt means one of:
          * the server predates §3 (no such endpoint) — the debug line
            below notes this case, and the caller falls back to chunk-
            derived stats silently, which is the intended behaviour;
          * the id was evicted from the server's 2000-entry ring (old
            request);
          * a genuine bug.

        None of the above is actionable from the client, so this method
        logs at debug on final miss — not warning, not error.
        """
        if not getattr(self.cfg, "fetch_server_stats", True):
            return None
        if not request_id:
            return None
        try:
            session = await self._get_session()
        except Exception as exc:
            log.warning(
                "_fetch_request_stats(%s): session unavailable: %s",
                request_id[:8], exc,
            )
            return None

        url = f"{self.cfg.base_url}/v1/requests/{request_id}/stats"
        t = aiohttp.ClientTimeout(total=float(self.cfg.stats_fetch_timeout_s))

        async def _attempt() -> Optional[ServerStats]:
            try:
                async with session.get(url, headers=self._headers(), timeout=t) as resp:
                    if resp.status == 404:
                        return None
                    if resp.status != 200:
                        log.warning(
                            "_fetch_request_stats(%s): HTTP %d — falling "
                            "back to chunk count",
                            request_id[:8], resp.status,
                        )
                        return None
                    raw = await resp.json()
                    return ServerStats.from_dict({**raw, "stats_source": "server"})
            except asyncio.TimeoutError:
                log.warning(
                    "_fetch_request_stats(%s): timeout after %.1fs",
                    request_id[:8], self.cfg.stats_fetch_timeout_s,
                )
                return None
            except Exception as exc:
                log.warning(
                    "_fetch_request_stats(%s): %s", request_id[:8], exc,
                )
                return None

        result = await _attempt()
        if result is not None:
            return result

        # One 100 ms retry — absorbs the [DONE]-vs-map-write race described
        # in the docstring.
        await asyncio.sleep(0.1)
        result = await _attempt()
        if result is None:
            log.warning(
                "_fetch_request_stats(%s): not found after retry — "
                "falling back to chunk-derived stats (older server, or "
                "id evicted from the ring)", request_id[:8],
            )
        return result

    def _build_stats_from_chunks(self, state: Dict[str, Any]) -> ServerStats:
        """Client-side stats derived from stream tracking (fallback path).

        `stats_source="chunks"` — the plan's term for "client counted
        streamed chunks; exact for SSE (one chunk == one token per server
        contract), approximate for plain text (ChunkBuffer word-buffers)".

        Fields that depend on server-side detail are left at 0:
          * `prompt_tokens`, `cached_prompt_tokens`, `thinking_tokens` —
            unknown; the client does not see them
          * `prefill_ms`, `prefill_tps` — unknown; TTFT conflates prefill
            and first-token latency
          * `queue_wait_ms` — unknown; the client only sees total elapsed

        Fields we CAN compute exactly from wall clock:
          * `ttft_ms` — time from send to first yielded chunk
          * `decode_ms` — time from first yield to last yield
          * `decode_tps` — (chunk_count - 1) / decode_s, same convention
            the server uses (first token is part of TTFT, not decode)
          * `total_ms` — time from send to last yield
          * `completion_tokens` — exact chunk count (see caveat above)
        """
        now = time.monotonic()
        start_ts = float(state.get("start_ts") or now)
        first_ts = state.get("first_ts")
        last_ts = state.get("last_ts") or now
        chunk_count = int(state.get("chunk_count") or 0)

        ttft_ms = (max(0.0, first_ts - start_ts) * 1000.0) if first_ts else 0.0
        total_ms = (max(0.0, last_ts - start_ts) * 1000.0) if last_ts else 0.0
        decode_ms = max(0.0, total_ms - ttft_ms)
        decode_tps = (
            (chunk_count - 1) / (decode_ms / 1000.0)
            if decode_ms > 50.0 and chunk_count > 1 else 0.0
        )

        return ServerStats(
            id=str(state.get("request_id") or ""),
            completion_tokens=chunk_count,
            ttft_ms=round(ttft_ms, 2),
            decode_ms=round(decode_ms, 2),
            decode_tps=round(decode_tps, 3),
            total_ms=round(total_ms, 2),
            finish_reason=str(state.get("finish_reason") or "stop"),
            stats_source="chunks",
        )

    async def _finalize_stream_stats_async(self, state: Dict[str, Any]) -> ServerStats:
        """Orchestrate: try server stats, fall back to client chunk-derived.

        Called from `_stream_post` immediately before a successful `return`
        (or in the exceptional cases where a partial stream should still
        carry stats — e.g. StreamInterruptedError). Always returns a
        `ServerStats`, never raises.
        """
        request_id = state.get("request_id")
        if request_id:
            server_stats = await self._fetch_request_stats(request_id)
            if server_stats is not None:
                # Ensure the id is present even if the server omitted it.
                if not server_stats.id:
                    # frozen dataclass — use dataclasses.replace
                    import dataclasses
                    server_stats = dataclasses.replace(
                        server_stats,
                        id=request_id,
                        stats_source="server",
                    )
                return server_stats
        return self._build_stats_from_chunks(state)

    # --------------------------------------------------------------------------
    # Generic streaming POST (with restart-tolerant retry)
    # --------------------------------------------------------------------------
    async def _stream_post(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        admin: bool = False,
        _result: Optional[StreamResult] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send a POST request and stream the response as text.

        Handles both plain text and SSE (Server-Sent Events).

        when `_result` is provided (all internal callers set it),
        this method populates `_result.stats` with a `ServerStats` object
        before returning / raising. Sources tried in order:
          1. `GET /v1/requests/{id}/stats` (authoritative) — requires
             `X-Request-ID` in the response header and
             `ClientConfig.fetch_server_stats=True`.
          2. Client-side chunk counting (exact for SSE, approximate for
             plain text) — used when (1) is unavailable.
          3. `stats_source="unknown"` — when zero chunks were produced.
        """
        # State dict shared with `_finalize_stream_stats_async` and the
        # `finally` fallback. Initialized once so the `finally` block can
        # still compute chunk-derived stats if we exit via GeneratorExit.
        state: Dict[str, Any] = {
            "request_id": None,
            "first_ts": None,
            "last_ts": None,
            "chunk_count": 0,
            "finish_reason": "stop",
            "start_ts": time.monotonic(),
        }

        def _finalize_sync() -> None:
            """Populate `_result.stats` from chunk-derived data — sync, safe in finally."""
            if _result is not None and _result.stats is None:
                _result.stats = self._build_stats_from_chunks(state)

        if not _HAS_AIOHTTP:
            # Fallback using sync client (same as chat_stream)
            log.warning("aiohttp not installed – using sync fallback with async chunking")
            loop = asyncio.get_running_loop()
            from .client import LLMCloudClient
            sync_client = LLMCloudClient(self.cfg)

            q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
            done = asyncio.Event()

            def _produce():
                try:
                    raise NotImplementedError(
                        "Generic streaming without aiohttp is not supported. Please install aiohttp."
                    )
                except Exception as exc:
                    loop.call_soon_threadsafe(lambda e=exc: q.put_nowait(f"[error: {e}]"))
                finally:
                    loop.call_soon_threadsafe(done.set)

            await loop.run_in_executor(None, _produce)

            while not done.is_set() or not q.empty():
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield chunk
                except asyncio.TimeoutError:
                    continue
            sync_client.close()
            _finalize_sync()
            return

        # ── aiohttp path ────────────────────────────────────────────────────
        url = self.cfg.base_url + path
        # Streaming MUST NOT carry a `total` deadline: a long-but-active
        # prefill legitimately takes 15+ minutes on this hardware, and
        # the server now emits an SSE comment heartbeat every
        # `heartbeat_interval_s` (5 s) during silence. Only bound the
        # IDLE gap between reads via `sock_read`; leave `total` unset
        # so an active stream cannot be killed by wall-clock alone.
        t_out = aiohttp.ClientTimeout(
            connect=float(getattr(self.cfg, "connect_timeout_s", 5.0)),
            sock_read=float(timeout or self.cfg.timeout),
            total=None,
        )
        session = await self._get_session()
        buffer = ChunkBuffer(flush_interval_ms=self.cfg.stream_chunk_ms)

        # Retry policy
        # ------------
        # Two different failures need two different answers:
        #
        #  * "Nothing useful arrived yet" (connect refused, or the connection
        #    dropped while only heartbeats had been received): safe to replay
        #    the request. A restarting server needs ~40-60 s, so a 1-2-4 s
        #    backoff burns every retry before it is back. Instead we poll
        #    /ready until the server answers, then re-POST.
        #  * "Part of the answer was already delivered": replaying would
        #    print a second answer after the partial one. We stop and raise
        #    StreamInterruptedError carrying what was received.
        #
        # Heartbeats (": hb ..." SSE comments / \x1f STATUS frames) are bytes
        # on the wire but NOT answer content. Counting them as "data received"
        # (the old `received_any`) disabled the safe-to-replay path for the
        # entire multi-minute prefill, when a restart is most likely.

        max_reconnects = int(getattr(self.cfg, "stream_reconnects", 1))
        restart_wait_s = float(getattr(self.cfg, "server_restart_wait_s", 180.0))

        last_exc: Exception = RuntimeError("no attempt")
        attempt = 0
        reconnects = 0
        try:
            while True:
                got_content = False          # real answer text reached the caller
                delivered: List[str] = []    # that text, for StreamInterruptedError
                try:
                    async with session.post(
                        url,
                        json=body,
                        headers=self._headers({"Accept": "text/plain, text/event-stream"}),
                        timeout=t_out,
                    ) as resp:
                        status = resp.status
                        # capture the request id from the response
                        # header BEFORE reading the body. aiohttp headers
                        # are case-insensitive, so one lookup suffices.
                        _rid = resp.headers.get("X-Request-ID")
                        if _rid:
                            state["request_id"] = _rid
                        if status != 200:
                            raw = await resp.read()
                            self._check_status(status, raw, path)
                        await self._cb_success()

                        content_type = resp.headers.get("Content-Type", "")
                        is_sse = "text/event-stream" in content_type

                        done_seen = False
                        async for raw_chunk in resp.content.iter_chunked(4096):
                            if not raw_chunk:
                                continue
                            try:
                                text_chunk = raw_chunk.decode("utf-8")
                            except UnicodeDecodeError:
                                text_chunk = raw_chunk.decode("utf-8", errors="replace")

                            if is_sse:
                                # Parse SSE lines. Comment lines (": hb ...") and
                                # blank lines carry no content and are skipped.
                                for line in text_chunk.splitlines():
                                    if not line.strip():
                                        continue
                                    if line.startswith("data: "):
                                        payload = line[6:]
                                        if payload == "[DONE]":
                                            # Do NOT return from inside the
                                            # `async with` — that skips the
                                            # stats-fetch below. Flag and
                                            # break out so flush + fetch run.
                                            done_seen = True
                                            break
                                        try:
                                            obj = json.loads(payload)
                                            if "choices" in obj and obj["choices"]:
                                                choice = obj["choices"][0]
                                                # capture finish_reason
                                                fr = choice.get("finish_reason")
                                                if fr:
                                                    state["finish_reason"] = str(fr)
                                                delta = choice.get("delta", {})
                                                content = delta.get("content", "")
                                                if content:
                                                    got_content = True
                                                    delivered.append(content)
                                                    # Track stats BEFORE yield so
                                                    # a consumer that breaks
                                                    # immediately after receiving
                                                    # this chunk still counts it.
                                                    now = time.monotonic()
                                                    if state["first_ts"] is None:
                                                        state["first_ts"] = now
                                                    state["last_ts"] = now
                                                    state["chunk_count"] += 1
                                                    yield content
                                            elif "error" in obj:
                                                raise StreamError(obj["error"])
                                        except json.JSONDecodeError:
                                            # Not JSON, treat as plain text
                                            got_content = True
                                            delivered.append(payload)
                                            now = time.monotonic()
                                            if state["first_ts"] is None:
                                                state["first_ts"] = now
                                            state["last_ts"] = now
                                            state["chunk_count"] += 1
                                            yield payload
                                if done_seen:
                                    break  # out of raw_chunk loop
                            else:
                                # Plain text – strip heartbeat control frames
                                # BEFORE buffering. ChunkBuffer flushes on
                                # whitespace, so pushing the raw chunk would
                                # leak \x1f STATUS:… \x1e bytes into the user's
                                # answer whenever a flush fired.
                                visible = _HB_FRAME_RE.sub("", text_chunk)
                                if not visible:
                                    continue
                                if visible.strip("\x1f\x1e \r\n\t"):
                                    got_content = True
                                    delivered.append(visible)
                                chunk = buffer.push(visible)
                                if chunk:
                                    now = time.monotonic()
                                    if state["first_ts"] is None:
                                        state["first_ts"] = now
                                    state["last_ts"] = now
                                    state["chunk_count"] += 1
                                    yield chunk
                        # Flush remaining buffer
                        remaining = buffer.flush()
                        if remaining:
                            now = time.monotonic()
                            if state["first_ts"] is None:
                                state["first_ts"] = now
                            state["last_ts"] = now
                            state["chunk_count"] += 1
                            yield remaining

                        # A stream that ends with no [DONE] and not one visible
                        # token is a dropped connection (server aborted after the
                        # 200 header), NOT an empty answer. ConnectionResetError
                        # is an OSError -> _is_connection_drop() -> "nothing
                        # reached the caller, replaying is safe" retry path.
                        # (Do not use StreamError: it is fail-fast, no retry.)
                        if not got_content and not done_seen:
                            raise ConnectionResetError(
                                "server closed the stream before sending any tokens"
                            )

                        # ── fetch stats before return ─────────────
                        # We are still inside `async with session.post(...)`,
                        # which means the response is not yet released, but
                        # aiohttp has already released the connection back
                        # to the pool (the body iterator was exhausted).
                        # The stats fetch will reuse a pooled connection.
                        if _result is not None:
                            try:
                                _result.stats = await self._finalize_stream_stats_async(state)
                            except Exception as exc:
                                log.warning(
                                    "Stream stats finalization failed for %s: %s — "
                                    "falling back to chunk-derived stats",
                                    (state.get("request_id") or "?")[:8], exc,
                                )
                                _result.stats = self._build_stats_from_chunks(state)
                        return  # success

                except (CircuitOpenError, AuthError, ModelNotLoadedError, StreamError):
                    raise
                    # same fail-fast as _request — 503 with a structured
                    # error_code carries a retry_after the caller will honour; the
                    # exponential-backoff retry loop below cannot see it.
                except ServiceUnavailableError:
                    raise
                except asyncio.TimeoutError:
                    last_exc = LLMTimeoutError(f"Stream request to {path} timed out")
                    await self._cb_failure()
                    if got_content:
                        # Part of an answer is already on screen; retrying would
                        # concatenate a fresh answer onto it.
                        log.warning(
                            "Stream timed out after partial delivery — not "
                            "retrying (would duplicate the answer)"
                        )
                        state["finish_reason"] = "timeout"
                        # Populate stats from what we delivered before raising
                        if _result is not None:
                            try:
                                _result.stats = await self._finalize_stream_stats_async(state)
                            except Exception:
                                _result.stats = self._build_stats_from_chunks(state)
                        # `asyncio.TimeoutError` carries no extra
                        # information beyond our LLMTimeoutError — suppress
                        # the implicit chaining so the user-facing traceback
                        # is the domain error, not the internal timeout.
                        raise last_exc from None
                except Exception as exc:
                    last_exc = exc
                    await self._cb_failure(exc)
                    dropped = _is_connection_drop(exc)

                    if got_content:
                        # Hand over whatever the word-buffer was still holding.
                        tail = buffer.flush()
                        if tail:
                            delivered.append(tail)
                            yield tail
                        partial = "".join(delivered)
                        state["finish_reason"] = "interrupted"
                        # Populate stats from delivered chunks before raising
                        if _result is not None:
                            try:
                                _result.stats = await self._finalize_stream_stats_async(state)
                            except Exception:
                                _result.stats = self._build_stats_from_chunks(state)
                        if dropped:
                            raise StreamInterruptedError(
                                "The server connection was lost in the middle of the "
                                "answer (it most likely restarted).",
                                partial_text=partial,
                                tokens=state.get("chunk_count", 0),
                                details={"cause": f"{type(exc).__name__}: {exc}"},
                            ) from exc
                        err = _as_llm_error(exc)
                        if err is exc:
                            raise
                        raise err from exc

                    if dropped:
                        # Nothing reached the caller -> replaying is safe.
                        if reconnects >= max_reconnects:
                            # the phrase "server crashed while handling
                            # this request" is load-bearing — a surface for the
                            # CLI and the Web UI to display *specifically*, so
                            # the operator knows to look for a request-shaped
                            # crash rather than a generic network problem.
                            raise LLMConnectionError(
                                "The server crashed twice in a row while "
                                "handling this request (no tokens arrived "
                                "either time). This request is likely "
                                "triggering the crash. Check server logs: "
                                "journalctl -u local-llm-cloud --since "
                                "'5 minutes ago'"
                            ) from exc
                        reconnects += 1
                        msg = (
                            "Server connection lost before the answer started "
                            f"({type(exc).__name__}). Waiting for it to come back "
                            f"(up to {restart_wait_s:.0f}s)…"
                        )
                        log.warning("%s [reconnect %d/%d]", msg, reconnects, max_reconnects)
                        self._emit_stream_event(msg)
                        if await self._wait_until_ready(restart_wait_s):
                            self._emit_stream_event("Server is back — resending your message…")
                            continue
                        raise LLMConnectionError(
                            f"The server at {self.cfg.base_url} did not come back "
                            f"within {restart_wait_s:.0f}s."
                        ) from exc

                # Anything else that is retryable: bounded exponential backoff.
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.retry_delay * (self.cfg.retry_backoff ** attempt)
                    delay = min(delay, self.cfg.retry_delay_max_s)
                    jitter = random.uniform(-0.1, 0.1)
                    delay = delay * (1 + jitter)
                    log.warning(
                        "Stream request failed (attempt %d/%d): %s — retrying in %.2fs",
                        attempt + 1, self.cfg.max_retries + 1, last_exc, delay
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                err = _as_llm_error(last_exc)
                if err is last_exc:
                    raise last_exc
                raise err from last_exc
        finally:
            # guarantee `.stats` is populated on every exit path —
            # normal completion, exception, or GeneratorExit (client
            # cancelled the stream via `break` / `.aclose()`).
            #
            # On the normal-completion path, `_finalize_stream_stats_async`
            # above has already set `_result.stats`. This fallback only fires
            # when the stream was abandoned before that line, in which case
            # chunk-derived stats are the best we can offer (a fetch would
            # require `await`, which `finally` in an async generator CAN do
            # but would delay the consumer's exit).
            _finalize_sync()

    # --------------------------------------------------------------------------
    # Public API – health & meta
    # --------------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def is_healthy(self) -> bool:
        # accept both the current schema ("status" == "ok") and
        # the legacy field ("health_status" == "healthy") that older
        # servers sent. With a current server both fields carry the
        # same value ("ok" / "degraded" / "startup_failed"), so either
        # works, and the fallback covers mixed-version deployments.
        try:
            h = await self.health()
            s = h.get("status") or h.get("health_status")
            return s in ("ok", "healthy")
        except Exception:
            return False

    async def ready(self) -> bool:
        try:
            await self._request("GET", "/ready")
            return True
        except (ModelNotLoadedError, ServerError):
            return False
        except Exception:
            return False

    async def models(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/v1/models")
        return data.get("data", [])

    async def current_model(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/models/current")

    async def queue_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/queue/status")

    async def system_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/system")

    async def metrics(self) -> Dict[str, Any]:
        return await self._request("GET", "/v1/metrics")

    async def model_ready(self, model_id: str) -> Optional[bool]:
        """Return True if the specified model is loaded and ready."""
        models = await self.models()
        for m in models:
            if m.get("id") == model_id:
                return m.get("status") == "ready"
        return None

    # --------------------------------------------------------------------------
    # Model management
    # --------------------------------------------------------------------------
    async def load_model(
        self,
        model_id: str,
        model_path: str,
        num_gpu_layers: int = -1,
        context_length: int = 0,
        model_type: str = "auto",
        n_threads: int = 0,
        timeout: float = 900.0,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST", "/v1/models/load",
            body={
                "model_id": model_id,
                "model_path": model_path,
                "num_gpu_layers": num_gpu_layers,
                "context_length": context_length,
                "model_type": model_type,
                "n_threads": n_threads,
            },
            timeout=timeout,
        )

    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/v1/models/{model_id}")

    async def clear_cache(self) -> Dict[str, Any]:
        return await self._request("DELETE", "/v1/context-cache/clear")

    async def hot_swap(
        self,
        old_id: str,
        new_id: str,
        new_path: str,
        num_gpu_layers: int = -1,
        context_length: int = 0,
        quantization: str = "auto",
        timeout: float = 900.0,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/models/{old_id}/swap",
            body={
                "model_id": new_id,
                "model_path": new_path,
                "num_gpu_layers": num_gpu_layers,
                "context_length": context_length,
                "quantization": quantization,
            },
            timeout=timeout,
        )

    async def load_profile(self, profile_id: str, timeout: float = 900.0) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/models/load-profile/{profile_id}",
            timeout=timeout,
        )

    # --------------------------------------------------------------------------
    # Inference — non-streaming (unchanged; caller parses response.usage)
    # --------------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model_id or self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": top_p if top_p is not None else 0.95,
            "top_k": top_k if top_k is not None else 40,
            "stop": stop,
            "stream": False,
        }
        data = await self._request("POST", "/v1/chat/completions", body=body, timeout=timeout or self.cfg.timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return str(data)

    async def chat_completions(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        timeout: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion. Returns the raw JSON dict.

        callers building a `GenerationResult` should use
        `GenerationResult.from_chat_response(data)` — it sets
        `stats_source="server"` because the non-streaming JSON body has
        authoritative `usage.completion_tokens` and, on a server with,
        `usage.thinking_tokens` and `usage.cached_prompt_tokens`. Do NOT
        re-count with `estimate_tokens_fast` — that was the CLI bug.
        """
        body = {
            "model": model_id or self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stop": stop or [],
            "stream": stream,
        }

        if "no_think" in kwargs or "thinking" in kwargs:
            enable_thinking = kwargs.get("thinking")
            if enable_thinking is None and "no_think" in kwargs:
                enable_thinking = not kwargs["no_think"]
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(enable_thinking)
        if "thinking_budget_tokens" in kwargs:
            body["thinking_budget_tokens"] = kwargs["thinking_budget_tokens"]
        if "stream_format" in kwargs:
            body["stream_format"] = kwargs["stream_format"]
        if "timeout_s" in kwargs:
            body["timeout_s"] = kwargs["timeout_s"]

        return await self._request("POST", "/v1/chat/completions", body=body, timeout=timeout or self.cfg.timeout)

    # --------------------------------------------------------------------------
    # Inference — streaming (returns StreamResult with .stats)
    # --------------------------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        timeout: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> StreamResult:
        """Stream a chat completion, yielding text chunks as they arrive.

        returns a `StreamResult` instead of a bare async generator.
        The iteration protocol is identical (`async for chunk in ...` still
        yields `str`), but `.stats` is populated with a `ServerStats`
        object once the stream completes. Callers that need the token
        count / TPS should read `stream.stats` after the loop:

            stream = client.chat_stream(...)
            async for chunk in stream:
                print(chunk, end="")
            print(stream.stats.decode_tps)   # real number

        Stats sources (in order of authority):
          1. `GET /v1/requests/{id}/stats` — authoritative
          2. Client chunk count — exact for SSE, approximate for plain text
          3. `stats_source="unknown"` — when no output was produced
        """
        body = {
            "model": model_id or self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stop": stop or [],
            "stream": True,
        }

        if "no_think" in kwargs or "thinking" in kwargs:
            enable_thinking = kwargs.get("thinking")
            if enable_thinking is None and "no_think" in kwargs:
                enable_thinking = not kwargs["no_think"]
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(enable_thinking)
        if "stream_format" in kwargs:
            body["stream_format"] = kwargs["stream_format"]
        if "timeout_s" in kwargs:
            body["timeout_s"] = kwargs["timeout_s"]

        result = StreamResult()
        result._impl = self._stream_post(
            "/v1/chat/completions",
            body=body,
            timeout=timeout,
            _result=result,
        )
        return result

    def stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> StreamResult:
        """Stream a single-prompt completion. Returns `StreamResult`."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(
            messages=messages,
            model_id=model_id,
            temperature=temperature or self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            top_p=top_p or 0.95,
            top_k=top_k or 40,
            stop=stop,
            timeout=timeout,
        )

    # --------------------------------------------------------------------------
    # Resume (checkpointed generation) – Section 2 (streaming returns StreamResult)
    # --------------------------------------------------------------------------
    async def resume(
            self,
            conv_id: str,
            checkpoint_id: Optional[str] = None,
            # 0 = unlimited. The server's checkpoint carries the
            # accumulated text; resumption is bounded only by the engine's
            # own EOS / stop / loop guard / stall detection. A client-side
            # default of 4096 truncated every resume past that token count.
            max_tokens: int = 0,
            temperature: Optional[float] = None,
            stream: bool = True,
            timeout: Optional[float] = None,
    ) -> Any:
        """
        Resume an interrupted generation from a checkpoint.

        `max_tokens` = 0 (the default) means "unlimited": the server
        continues from the checkpoint's accumulated text until the
        engine terminates the stream on its own (EOS, stop sequence,
        loop guard, or stall detection). Any positive value caps the
        new tokens generated by THIS resume call.

        Returns:
          • When `stream=True`: a `StreamResult` — iterate for
            chunks, read `.stats` after exhaustion. The iteration protocol
            is identical to the behavior (`async for chunk in ...`).
          • When `stream=False`: the parsed JSON dict, as before.
        """
        body = {
            "conv_id": conv_id,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if checkpoint_id:
            body["checkpoint_id"] = checkpoint_id
        if temperature is not None:
            body["temperature"] = temperature

        if stream:
            result = StreamResult()
            result._impl = self._stream_post(
                "/v1/chat/resume",
                body=body,
                timeout=timeout or self.cfg.timeout,
                _result=result,
            )
            return result
        else:
            return await self._request(
                "POST",
                "/v1/chat/resume",
                body=body,
                timeout=timeout or self.cfg.timeout,
            )

    # --------------------------------------------------------------------------
    # Logs
    # --------------------------------------------------------------------------
    async def get_logs(self, n: int = 100, since: float = 0.0) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/logs?n={n}&since={since}")

    # --------------------------------------------------------------------------
    # API Key management (admin endpoints)
    # --------------------------------------------------------------------------
    async def create_key(
        self,
        key_id: str,
        rate_limit_rps: float = 10.0,
        rate_limit_capacity: float = 20.0,
        description: str = "",
        expires_in_seconds: Optional[float] = None,
        active: bool = True,
    ) -> Dict[str, Any]:
        body = {
            "key_id": key_id,
            "rate_limit_rps": rate_limit_rps,
            "rate_limit_capacity": rate_limit_capacity,
            "description": description,
            "active": active,
        }
        if expires_in_seconds is not None:
            body["expires_in_seconds"] = expires_in_seconds
        return await self._request("POST", "/admin/keys/create", body=body, admin=True)

    async def revoke_key(self, key_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/admin/keys/revoke/{key_id}", admin=True)

    async def activate_key(self, key_id: str) -> Dict[str, Any]:
        return await self._request("POST", f"/admin/keys/activate/{key_id}", admin=True)

    async def delete_key(self, key_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/admin/keys/delete/{key_id}", admin=True)

    async def list_keys(self, active_only: bool = True) -> List[Dict[str, Any]]:
        data = await self._request("GET", f"/admin/keys/list?active_only={str(active_only).lower()}", admin=True)
        return data.get("keys", [])

    async def key_stats(self, key_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/admin/keys/stats/{key_id}", admin=True)

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Word-count heuristic — for BUDGET sizing only, not display.

        For real token counting on the client, use `ChunkTokenCounter`
        (streamed) or `TokenizerAdapter` (arbitrary text) from `_accel.py`.
        For authoritative counts after a stream, read `StreamResult.stats`
        populated by `chat_stream` / `stream` / `resume(stream=True)`.
        """
        return estimate_tokens_fast(text)
