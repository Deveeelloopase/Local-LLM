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
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set

from .models import ClientConfig
from .exceptions import (
    AuthError,
    CircuitOpenError,
    ConnectionError as LLMConnectionError,
    LLMCloudError,
    ModelNotLoadedError,
    QueueFullError,
    ServerError,
    StreamError,
    StreamInterruptedError,
    TimeoutError as LLMTimeoutError,
)
from ._accel import ChunkBuffer, estimate_tokens_fast

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
# AsyncLLMCloudClient
# ------------------------------------------------------------------------------
class AsyncLLMCloudClient:
    """
    Async client for Local LLM Cloud. Compatible with asyncio and ai_agent.py.

    Usage:
        async with AsyncLLMCloudClient() as client:
            text = await client.generate("Hello!")
            async for chunk in client.stream("Tell me a story"):
                print(chunk, end="", flush=True)
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
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Circuit breaker
    # --------------------------------------------------------------------------
    async def _cb_allow(self) -> bool:
        if getattr(self.cfg, "circuit_breaker_enabled", True) and self.cfg.max_retries <= 0:
            return True
        async with self._cb_lock:
            now = time.monotonic()
            if self._cb_state == "closed":
                return True
            if self._cb_state == "open":
                if now - self._cb_opened_at >= self.cfg.circuit_breaker_recovery:
                    self._cb_state = "half_open"
                    self._half_open_probe_started = now
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

    async def _cb_failure(self) -> None:
        async with self._cb_lock:
            self._cb_failures += 1
            threshold = getattr(self.cfg, "circuit_breaker_threshold", self.cfg.max_retries)
            if self._cb_failures >= threshold and self._cb_state != "open":
                self._cb_state = "open"
                self._cb_opened_at = time.monotonic()
                log.warning("AsyncLLMCloud circuit breaker OPEN after %d failures", self._cb_failures)

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
            except Exception as exc:
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

            except aiohttp.ClientConnectorError as exc:
                last_exc = LLMConnectionError(f"Cannot connect to {url}: {exc}")
                await self._cb_failure()

            except asyncio.TimeoutError:
                last_exc = LLMTimeoutError(f"Request to {url} timed out ({t_out}s)")
                await self._cb_failure()

            except Exception as exc:
                last_exc = exc
                await self._cb_failure()

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
        if status == 200:
            return
        msg = body.decode(errors="replace")[:300] if body else ""
        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}): {msg}")
        if status == 503 and ("No model" in msg or "not loaded" in msg.lower()):
            raise ModelNotLoadedError("No model loaded on server.")
        if status in (429, 503) and "queue" in msg.lower():
            raise QueueFullError(f"Server queue full ({status}): {msg}")
        if status >= 500:
            raise ServerError(f"Server error {status} on {path}: {msg}")

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
            log.debug("on_stream_event hook failed", exc_info=True)

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
    # Generic streaming POST (with restart-tolerant retry)
    # --------------------------------------------------------------------------
    async def _stream_post(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        admin: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Send a POST request and stream the response as text.
        Handles both plain text and SSE (Server-Sent Events).
        """
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
        max_reconnects = int(getattr(self.cfg, "stream_reconnects", 2))
        restart_wait_s = float(getattr(self.cfg, "server_restart_wait_s", 180.0))

        last_exc: Exception = RuntimeError("no attempt")
        attempt = 0
        reconnects = 0
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
                    if status != 200:
                        raw = await resp.read()
                        self._check_status(status, raw, path)
                    await self._cb_success()

                    content_type = resp.headers.get("Content-Type", "")
                    is_sse = "text/event-stream" in content_type

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
                                        return
                                    try:
                                        obj = json.loads(payload)
                                        if "choices" in obj and obj["choices"]:
                                            delta = obj["choices"][0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                got_content = True
                                                delivered.append(content)
                                                yield content
                                        elif "error" in obj:
                                            raise StreamError(obj["error"])
                                    except json.JSONDecodeError:
                                        # Not JSON, treat as plain text
                                        got_content = True
                                        delivered.append(payload)
                                        yield payload
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
                                yield chunk
                    # Flush remaining buffer
                    remaining = buffer.flush()
                    if remaining:
                        yield remaining
                    return  # success

            except (CircuitOpenError, AuthError, ModelNotLoadedError, StreamError):
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
                    raise last_exc
            except Exception as exc:
                last_exc = exc
                await self._cb_failure()
                dropped = _is_connection_drop(exc)

                if got_content:
                    # Hand over whatever the word-buffer was still holding.
                    tail = buffer.flush()
                    if tail:
                        delivered.append(tail)
                        yield tail
                    partial = "".join(delivered)
                    if dropped:
                        raise StreamInterruptedError(
                            "The server connection was lost in the middle of the "
                            "answer (it most likely restarted).",
                            partial_text=partial,
                            tokens=estimate_tokens_fast(partial),
                            details={"cause": f"{type(exc).__name__}: {exc}"},
                        ) from exc
                    err = _as_llm_error(exc)
                    if err is exc:
                        raise
                    raise err from exc

                if dropped:
                    # Nothing reached the caller -> replaying is safe.
                    if reconnects >= max_reconnects:
                        raise LLMConnectionError(
                            f"Lost the connection to {self.cfg.base_url} "
                            f"{reconnects + 1} times in a row while waiting for "
                            "the first token."
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

    # --------------------------------------------------------------------------
    # Public API – health & meta
    # --------------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def is_healthy(self) -> bool:
        try:
            h = await self.health()
            return h.get("status") == "ok"
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
    # Inference
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

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        timeout: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion, yielding text chunks as they arrive.
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

        # Use the generic streaming POST
        async for chunk in self._stream_post(
            "/v1/chat/completions",
            body=body,
            timeout=timeout,
        ):
            yield chunk

    async def stream(
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
    ) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async for chunk in self.chat_stream(
            messages=messages,
            model_id=model_id,
            temperature=temperature or self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            top_p=top_p or 0.95,
            top_k=top_k or 40,
            stop=stop,
            timeout=timeout,
        ):
            yield chunk

    # --------------------------------------------------------------------------
    # Resume (checkpointed generation) – Section 2
    # --------------------------------------------------------------------------
    async def resume(
        self,
        conv_id: str,
        checkpoint_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        stream: bool = True,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Resume an interrupted generation from a checkpoint.
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
            return self._stream_post(
                "/v1/chat/resume",
                body=body,
                timeout=timeout or self.cfg.timeout,
            )
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
        return estimate_tokens_fast(text)
