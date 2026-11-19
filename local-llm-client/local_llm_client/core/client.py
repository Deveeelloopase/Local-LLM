"""
local_llm_client/core/client.py

Production‑ready synchronous HTTP client for Local LLM Cloud.

Features:
- Thread‑safe with connection pooling (urllib3) and fallback to http.client.
- Circuit breaker with configurable thresholds and exponential backoff jitter.
- Word‑buffered streaming via ChunkBuffer (from _accel).
- Fast token counting via estimate_tokens_fast (budget sizing only).
- Full coverage of all server endpoints (model management, key admin, logs, etc.).
- Dynamic API key update via set_api_key().
- Model readiness checks using the server‑side status field.
- Automatic master‑key header injection for admin routes.
- No hardcoded paths – all configuration via ClientConfig.
- Comprehensive retry classification and error propagation.
- Graceful fallback when optional dependencies (urllib3) are missing.
- Checkpointed, resumable generation (Section 2) – resume() and resume_stream().

— REAL STREAM STATS (sync parity with async_client.py)
───────────────────────────────────────────────────────────────────
Same design as `async_client.py` (see that module's docstring for the
full rationale). Streaming methods return a `StreamResult` wrapper whose
`.stats` attribute is populated with `ServerStats` after iteration:

  * `stats_source="server"`  — server-authoritative (from
                                GET /v1/requests/{id}/stats)
  * `stats_source="chunks"`  — client-side chunk count (exact for SSE,
                                approximate for plain text; formatter
                                adds `~` + `(est.)` marker)
  * `stats_source="unknown"` — no chunks arrived, no server stats

Sync-iteration contract:

    stream = client.chat_stream(...)   # ← returns StreamResult, not Generator
    for chunk in stream:               # ← yields str, unchanged
        print(chunk, end="")
    print(stream.stats.decode_tps)     # ← real number

The iteration protocol is 100% unchanged for existing callers. Only the
return type's identity differs (`StreamResult` instead of raw generator),
and `.stats` is new.

version = 1.0.0
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

from .models import ClientConfig, ModelInfo, ModelList
from .exceptions import (
    AuthError,
    CircuitOpenError,
    ConnectionError as LLMConnectionError,
    ModelNotLoadedError,
    QueueFullError,
    ServerError,
    StreamError,
    TimeoutError as LLMTimeoutError,
)
from ._accel import ChunkBuffer, estimate_tokens_fast
from .formatter import ServerStats

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# HTTP backend: prefer urllib3, fall back to http.client
# ------------------------------------------------------------------------------
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _HAS_URLLIB3 = True
except ImportError:
    _HAS_URLLIB3 = False


# ------------------------------------------------------------------------------
# StreamResult — streaming with real stats after exhaustion (sync)
# ------------------------------------------------------------------------------
class StreamResult:
    """Sync iterator wrapper that exposes `.stats` after exhaustion.

    Parallel to `async_client.StreamResult` but for synchronous iteration.
    All streaming methods on `LLMCloudClient` (`chat_stream`, `stream`,
    `resume_stream`) return a `StreamResult` instead of a bare generator.
    Iteration is identical:

        stream = client.chat_stream(...)   # ← StreamResult
        for chunk in stream:               # ← yields str
            print(chunk, end="")
        # stream.stats is now populated

    `.stats` is `None` until iteration completes. On completion (normal or
    early-close), `.stats` is a `ServerStats` with:
      * `stats_source="server"` — server-authoritative
      * `stats_source="chunks"` — client chunk count (formatter adds `~`)
      * `stats_source="unknown"` — no data available
    """
    __slots__ = ("_impl", "stats")

    def __init__(self) -> None:
        self._impl: Optional[Iterator[str]] = None
        self.stats: Optional[ServerStats] = None

    def __iter__(self) -> Iterator[str]:
        if self._impl is None:
            raise RuntimeError(
                "StreamResult is uninitialized — obtain it from "
                "client.chat_stream(), client.stream(), or "
                "client.resume_stream() instead of constructing it directly."
            )
        return self._impl

    def close(self) -> None:
        """Close the underlying generator if still open. Idempotent.

        Callers that break out of `for chunk in stream` early should call
        this so the connection is released promptly.
        """
        if self._impl is not None:
            close_fn = getattr(self._impl, "close", None)
            if close_fn is not None:
                try:
                    close_fn()
                except Exception:
                    pass


# ------------------------------------------------------------------------------
# Circuit breaker
# ------------------------------------------------------------------------------
class _CircuitBreaker:
    """
    Thread-safe circuit breaker with three states and configurable thresholds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int,
        recovery_s: float,
        half_open_probe_timeout: float = 10.0,
    ) -> None:
        self._threshold = threshold
        self._recovery_s = recovery_s
        self._half_open_probe_timeout = half_open_probe_timeout
        self._failures = 0
        self._state = self.CLOSED
        self._opened_at = 0.0
        self._half_open_probe_started = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN:
                if now - self._half_open_probe_started < self._half_open_probe_timeout:
                    return True
                # Probe expired without success → move back to OPEN
                self._state = self.OPEN
                self._opened_at = now
                return False
            if self._state == self.OPEN:
                if now - self._opened_at >= self._recovery_s:
                    self._state = self.HALF_OPEN
                    self._half_open_probe_started = now
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._state != self.OPEN:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                log.warning("LLMCloud circuit breaker OPEN after %d failures", self._failures)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


# ------------------------------------------------------------------------------
# LLMCloudClient
# ------------------------------------------------------------------------------
class LLMCloudClient:
    """
    Synchronous client for Local LLM Cloud.
    Thread‑safe: one instance can be shared across threads.

    Streaming methods (`chat_stream`, `stream`, `resume_stream`) return a
    `StreamResult`. The iteration protocol is identical to the
    pre- generator behavior — `.stats` is the only addition.
    """

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self.cfg = config or ClientConfig()
        self._validate_base_url()

        # Circuit breaker configuration
        cb_enabled = getattr(self.cfg, "circuit_breaker_enabled", True)
        cb_threshold = getattr(self.cfg, "circuit_breaker_threshold", self.cfg.max_retries)
        cb_recovery = getattr(self.cfg, "circuit_breaker_recovery", 60.0)
        cb_half_open_timeout = getattr(self.cfg, "half_open_probe_timeout_s", 10.0)
        self._cb = (
            _CircuitBreaker(
                threshold=cb_threshold,
                recovery_s=cb_recovery,
                half_open_probe_timeout=cb_half_open_timeout,
            )
            if cb_enabled and self.cfg.max_retries > 0
            else None
        )

        # Master key (for admin endpoints)
        self._master_key: Optional[str] = getattr(self.cfg, "master_key", None)

        self._pool = self._make_pool()

    def _validate_base_url(self) -> None:
        url = self.cfg.base_url
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {url}")
        if url.endswith("/"):
            self.cfg.base_url = url.rstrip("/")

    # --------------------------------------------------------------------------
    # HTTP pool (urllib3) or fallback
    # --------------------------------------------------------------------------
    def _make_pool(self):
        if not _HAS_URLLIB3:
            return None
        from urllib3 import HTTPConnectionPool, HTTPSConnectionPool

        url = self.cfg.base_url
        scheme = "https" if url.startswith("https") else "http"
        rest = url.split("://", 1)[1]
        host_port = rest.split("/")[0]
        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 443 if scheme == "https" else 80

        cls = HTTPSConnectionPool if scheme == "https" else HTTPConnectionPool
        return cls(
            host,
            port=port,
            maxsize=self.cfg.pool_size,
            retries=False,
            timeout=urllib3.Timeout(connect=self.cfg.connect_timeout_s, read=self.cfg.read_timeout_s),
        )

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
    # Headers
    # --------------------------------------------------------------------------
    def _headers(self, extra: Optional[Dict] = None, admin: bool = False) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.cfg.api_key:
            h["X-API-Key"] = self.cfg.api_key
        if admin and self._master_key:
            h["X-Master-Key"] = self._master_key
        if extra:
            h.update(extra)
        return h

    # --------------------------------------------------------------------------
    # Core request with retry + circuit breaker
    # --------------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
        admin: bool = False,
    ) -> Any:
        """
        Execute an HTTP request with retries, circuit breaker, and error classification.
        Returns parsed JSON (dict) or a raw response object for streaming.
        """
        if self._cb and not self._cb.allow():
            raise CircuitOpenError(
                f"Circuit breaker OPEN — server was unreachable. "
                f"Retry after {getattr(self.cfg, 'circuit_breaker_recovery', 60.0):.0f}s."
            )

        url = self.cfg.base_url + path
        payload = json.dumps(body).encode() if body else None
        headers = self._headers(admin=admin)
        t_out = timeout or self.cfg.timeout
        last_exc: Exception = RuntimeError("no attempt made")

        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._raw_request(method, url, payload, headers, t_out, stream)
                if self._cb:
                    self._cb.record_success()

                if stream:
                    return resp  # raw response for streaming

                status = self._get_status(resp)
                raw = self._read_body(resp)
                self._check_status(status, raw, path)
                data = json.loads(raw)
                # If the response indicates an API key issue, raise immediately
                if isinstance(data, dict) and "detail" in data and isinstance(data["detail"], str):
                    if "Invalid or missing API key" in data["detail"]:
                        raise AuthError(data["detail"])
                return data

            except (CircuitOpenError, AuthError, ModelNotLoadedError, ValueError) as non_retryable:
                raise

            except (LLMConnectionError, LLMTimeoutError, QueueFullError, ServerError) as retryable:
                last_exc = retryable
                if self._cb:
                    self._cb.record_failure()
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.retry_delay * (self.cfg.retry_backoff ** attempt)
                    delay = min(delay, self.cfg.retry_delay_max_s)
                    jitter = random.uniform(-0.1, 0.1)
                    delay = delay * (1 + jitter)
                    log.warning(
                        "Request failed (attempt %d/%d): %s — retrying in %.2fs",
                        attempt + 1, self.cfg.max_retries + 1, retryable, delay
                    )
                    time.sleep(delay)
                else:
                    log.error("Request failed after %d attempts: %s", attempt + 1, retryable)
                    raise

            except Exception as exc:
                log.error("Non‑retryable error in request: %s", exc)
                raise

        raise last_exc

    def _raw_request(self, method, url, payload, headers, timeout, stream):
        """Execute one raw HTTP request. Returns response object."""
        if _HAS_URLLIB3 and self._pool is not None:
            import urllib.parse as up
            parsed = up.urlsplit(url)
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            kwargs = {
                "method": method,
                "url": path,
                "body": payload,
                "headers": headers,
                "timeout": timeout,
                "preload_content": not stream,
            }
            try:
                return self._pool.urlopen(**kwargs)
            except urllib3.exceptions.MaxRetryError as exc:
                raise LLMConnectionError(f"Cannot connect to {url}: {exc.reason}") from exc
            except urllib3.exceptions.NameResolutionError as exc:
                raise LLMConnectionError(f"DNS resolution failed for {url}: {exc}") from exc
            except urllib3.exceptions.TimeoutError as exc:
                raise LLMTimeoutError(f"Request to {url} timed out ({timeout}s)") from exc

        # Fallback to http.client
        import http.client as hc
        import urllib.parse as up
        parsed = up.urlsplit(url)
        conn_cls = hc.HTTPSConnection if parsed.scheme == "https" else hc.HTTPConnection
        host = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn = conn_cls(host, timeout=timeout)
        try:
            conn.request(method, path, body=payload, headers=headers)
            return conn.getresponse()
        except Exception as exc:
            conn.close()
            raise LLMConnectionError(f"Cannot connect to {host}: {exc}") from exc

    @staticmethod
    def _get_status(resp) -> int:
        if hasattr(resp, "status"):
            return resp.status
        return 500  # fallback

    @staticmethod
    def _read_body(resp) -> bytes:
        if hasattr(resp, "data"):
            return resp.data
        return resp.read()

    def _check_status(self, status: int, body: bytes, path: str) -> None:
        if status == 200:
            return
        msg = body.decode(errors="replace")[:300] if body else ""
        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}): {msg}")
        if status == 503 and ("No model" in msg or "not loaded" in msg.lower()):
            raise ModelNotLoadedError("No model loaded on server. Load one first.")
        if status in (429, 503) and "queue" in msg.lower():
            raise QueueFullError(f"Server queue full ({status}): {msg}")
        if 500 <= status < 600:
            raise ServerError(f"Server error {status} on {path}: {msg}")
        if status == 404:
            raise LLMConnectionError(f"Endpoint not found: {path}")
        raise ValueError(f"HTTP {status} error on {path}: {msg}")

    # --------------------------------------------------------------------------
    # — per-request stats (server-authoritative or client-derived)
    # --------------------------------------------------------------------------
    def _fetch_request_stats(self, request_id: str) -> Optional[ServerStats]:
        """Fetch `GET /v1/requests/{id}/stats` — sync, short timeout, no retries.

        Returns None on any failure (404 from old server, network error,
        malformed JSON, timeout, feature disabled). Callers fall back to
        `_build_stats_from_chunks` in that case.
        """
        if not getattr(self.cfg, "fetch_server_stats", True):
            log.warning("_fetch_request_stats(%s): disabled by config",
                      (request_id or "?")[:8])
            return None
        if not request_id:
            return None
        url = f"{self.cfg.base_url}/v1/requests/{request_id}/stats"
        timeout = float(getattr(self.cfg, "stats_fetch_timeout_s", 5.0))
        try:
            resp = self._raw_request("GET", url, None, self._headers(), timeout, stream=False)
            status = self._get_status(resp)
            if status == 404:
                # Old server. Not an error — fall back.
                log.warning(
                    "_fetch_request_stats(%s): 404 (endpoint missing on old server) "
                    "— falling back to chunk count", request_id[:8],
                )
                try:
                    if hasattr(resp, "close"): resp.close()
                except Exception:
                    pass
                return None
            if status != 200:
                log.warning(
                    "_fetch_request_stats(%s): HTTP %d — falling back to chunk count",
                    request_id[:8], status,
                )
                try:
                    if hasattr(resp, "close"): resp.close()
                except Exception:
                    pass
                return None
            raw = self._read_body(resp)
            data = json.loads(raw)
            stats = ServerStats.from_dict({**data, "stats_source": "server"})
            return stats
        except Exception as exc:
            log.warning("_fetch_request_stats(%s): %s", (request_id or "?")[:8], exc)
            return None

    def _build_stats_from_chunks(self, state: Dict[str, Any]) -> ServerStats:
        """Client-side stats derived from stream tracking (fallback path).

        `stats_source="chunks"` — exact for SSE (one chunk == one token per
        server contract), approximate for plain text (ChunkBuffer word-buffers
        so chunk count != token count).

        Fields that depend on server-side detail are left at 0:
          * `prompt_tokens`, `cached_prompt_tokens`, `thinking_tokens` —
            unknown client-side
          * `prefill_ms`, `prefill_tps` — unknown (TTFT conflates prefill
            and first-token latency)
          * `queue_wait_ms` — unknown

        Client-measured (exact): `ttft_ms`, `decode_ms`, `decode_tps`,
        `total_ms`, `completion_tokens` (per the caveat above).
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

    def _finalize_stream_stats_sync(self, state: Dict[str, Any]) -> ServerStats:
        """Orchestrate: try server stats, fall back to client chunk-derived.

        Always returns a `ServerStats`, never raises.
        """
        request_id = state.get("request_id")
        if request_id:
            server_stats = self._fetch_request_stats(request_id)
            if server_stats is not None:
                # Ensure id is present even if server omitted it.
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
    # Public API – health & meta (unchanged)
    # --------------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def is_healthy(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except Exception:
            return False

    def ready(self) -> bool:
        try:
            self._request("GET", "/ready")
            return True
        except (ModelNotLoadedError, ServerError):
            return False
        except Exception:
            return False

    def models(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/v1/models")
        return data.get("data", [])

    def current_model(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/models/current")

    def queue_status(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/queue/status")

    def system_info(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/system")

    def metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/metrics")

    def model_ready(self, model_id: str) -> Optional[bool]:
        """Return True if the specified model is loaded and ready."""
        models = self.models()
        for m in models:
            if m.get("id") == model_id:
                return m.get("status") == "ready"
        return None

    # --------------------------------------------------------------------------
    # Model management (unchanged)
    # --------------------------------------------------------------------------
    def load_model(
        self,
        model_id: str,
        model_path: str,
        num_gpu_layers: int = -1,
        context_length: int = 0,
        model_type: str = "auto",
        n_threads: int = 0,
        timeout: float = 900.0,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/v1/models/load",
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

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/v1/models/{model_id}")

    def clear_cache(self) -> Dict[str, Any]:
        return self._request("DELETE", "/v1/context-cache/clear")

    def hot_swap(
        self,
        old_id: str,
        new_id: str,
        new_path: str,
        num_gpu_layers: int = -1,
        context_length: int = 0,
        quantization: str = "auto",
        timeout: float = 900.0,
    ) -> Dict[str, Any]:
        return self._request(
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

    def load_profile(self, profile_id: str, timeout: float = 900.0) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/models/load-profile/{profile_id}",
            timeout=timeout,
        )

    # --------------------------------------------------------------------------
    # Inference — non-streaming (unchanged)
    # --------------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        effective_model = model_id or self.cfg.model
        if effective_model is None:
            raise ValueError("No model specified. Provide model_id or set ClientConfig.model.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": top_p if top_p is not None else 0.95,
            "top_k": top_k if top_k is not None else 40,
            "stop": stop,
            "stream": False,
        }
        data = self._request("POST", "/v1/chat/completions", body=body, timeout=timeout or self.cfg.timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return str(data)

    def chat_completions(
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
        """Non-streaming chat completion. Returns raw JSON dict.

        callers building a `GenerationResult` should use
        `GenerationResult.from_chat_response(data)` (models.py) — it sets
        `stats_source="server"` because the non-streaming JSON has
        authoritative `usage.completion_tokens`. Do NOT re-count with
        `estimate_tokens_fast` — that was the CLI bug.
        """
        body = {
            "model": model_id or self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "stop": stop or [],
            "stream": stream,
        }
        if "no_think" in kwargs:
            body["no_think"] = kwargs["no_think"]
        if "thinking" in kwargs:
            body["thinking"] = kwargs["thinking"]
        if "thinking_budget_tokens" in kwargs:
            body["thinking_budget_tokens"] = kwargs["thinking_budget_tokens"]
        if "stream_format" in kwargs:
            body["stream_format"] = kwargs["stream_format"]
        if "timeout_s" in kwargs:
            body["timeout_s"] = kwargs["timeout_s"]

        return self._request("POST", "/v1/chat/completions", body=body, timeout=timeout or self.cfg.timeout)

    # --------------------------------------------------------------------------
    # Inference — streaming (returns StreamResult with .stats)
    # --------------------------------------------------------------------------
    def _stream_with_stats(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: Optional[float],
        *,
        sse: bool,
        _result: StreamResult,
    ) -> Generator[str, None, None]:
        """Wrap the raw stream parser with stats tracking.

        - Captures `X-Request-ID` from response headers BEFORE reading body
        - Tracks chunk count, TTFT, decode time via `state` dict
        - On normal completion: fetches `/v1/requests/{id}/stats`
          and populates `_result.stats` with authoritative data; falls back
          to `_build_stats_from_chunks` if fetch fails.
        - On early close: uses `_build_stats_from_chunks` only (no blocking
          HTTP in the early-exit path — the request may still be in flight
          server-side, and a fetch now would be wasteful).
        """
        state: Dict[str, Any] = {
            "request_id": None,
            "first_ts": None,
            "last_ts": None,
            "chunk_count": 0,
            "finish_reason": "stop",
            "start_ts": time.monotonic(),
        }
        resp = None
        completed_ok = False
        try:
            resp = self._request(
                "POST", path, body=body, timeout=timeout, stream=True,
            )

            # ── capture X-Request-ID before reading body ───────
            try:
                headers = getattr(resp, "headers", None)
                if headers is not None:
                    _rid = headers.get("X-Request-ID")
                    if _rid:
                        state["request_id"] = _rid
            except Exception:
                pass

            status = self._get_status(resp)
            if status != 200:
                body_bytes = b""
                try:
                    if hasattr(resp, "read"):
                        body_bytes = resp.read(512)
                except Exception:
                    pass
                self._check_status(status, body_bytes, path)

            # ── Choose parser: SSE (per-token) vs plain text (word-buffered)
            src = (
                self._read_sse_stream(resp, _state=state)
                if sse
                else self._read_stream(resp, _state=state)
            )
            for chunk in src:
                yield chunk
            completed_ok = True
        finally:
            if resp is not None:
                try:
                    if hasattr(resp, "close"):
                        resp.close()
                    if hasattr(resp, "connection") and resp.connection:
                        resp.connection.close()
                except Exception:
                    pass
            # ── populate _result.stats on every exit path ─────
            try:
                if completed_ok:
                    _result.stats = self._finalize_stream_stats_sync(state)
                else:
                    # Early close / error — no blocking HTTP in the hot path
                    _result.stats = self._build_stats_from_chunks(state)
            except Exception as exc:
                log.warning(
                    "Stream stats finalization failed: %s — using chunk-derived",
                    exc,
                )
                try:
                    _result.stats = self._build_stats_from_chunks(state)
                except Exception:
                    pass

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
        """Stream a chat completion. Returns `StreamResult`.

        Iteration is unchanged (`for chunk in stream: ...` yields `str`).
        `.stats` is populated after iteration completes — read it to get
        the real token count and TPS without relying on `estimate_tokens_fast`.
        """
        body = {
            "model": model_id or self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "stop": stop or [],
            "stream": True,
        }
        if "no_think" in kwargs:
            body["no_think"] = kwargs["no_think"]
        if "thinking" in kwargs:
            body["thinking"] = kwargs["thinking"]
        if "stream_format" in kwargs:
            body["stream_format"] = kwargs["stream_format"]
        if "timeout_s" in kwargs:
            body["timeout_s"] = kwargs["timeout_s"]

        # Note: the sync client historically used /v1/chat/stream (plain
        # text) while the async client uses /v1/chat/completions (SSE).
        # Preserve the plain-text endpoint for backward compat; the caller
        # gets word-buffered chunks — chunk count is approximate. Server
        # stats provide the exact count when available.
        result = StreamResult()
        result._impl = self._stream_with_stats(
            "/v1/chat/stream",
            body=body,
            timeout=timeout or self.cfg.timeout,
            sse=False,
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
        stop: Optional[List[str]] = None,
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
    # Stream parsers (accept optional `_state` for stats tracking)
    # --------------------------------------------------------------------------
    def _read_stream(self, resp, _state: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Parse a plain-text stream, yielding word-buffered chunks.

        When `_state` is provided, updates:
          * `first_ts`   — time of first yield
          * `last_ts`    — time of last yield
          * `chunk_count`— number of yields (word-flush count, approximate
                          for token counting — see method docstring in
                          `_build_stats_from_chunks`)
        """
        buffer = ChunkBuffer(flush_interval_ms=self.cfg.stream_chunk_ms)
        try:
            import codecs
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        except ImportError:
            decoder = None

        try:
            if _HAS_URLLIB3 and hasattr(resp, "stream"):
                chunk_iter = resp.stream(4096)
            else:
                chunk_iter = self._iter_http_client(resp)

            for raw in chunk_iter:
                if not raw:
                    continue
                txt = decoder.decode(raw) if decoder else raw.decode("utf-8", errors="replace")
                flushed = buffer.push(txt)
                if flushed:
                    if _state is not None:
                        now = time.monotonic()
                        if _state["first_ts"] is None:
                            _state["first_ts"] = now
                        _state["last_ts"] = now
                        _state["chunk_count"] += 1
                    yield flushed

            if decoder:
                final = decoder.decode(b"", final=True)
                if final:
                    flushed = buffer.push(final)
                    if flushed:
                        if _state is not None:
                            now = time.monotonic()
                            if _state["first_ts"] is None:
                                _state["first_ts"] = now
                            _state["last_ts"] = now
                            _state["chunk_count"] += 1
                        yield flushed

            remaining = buffer.flush()
            if remaining:
                if _state is not None:
                    now = time.monotonic()
                    if _state["first_ts"] is None:
                        _state["first_ts"] = now
                    _state["last_ts"] = now
                    _state["chunk_count"] += 1
                yield remaining
        except Exception as exc:
            raise StreamError(f"Stream read error: {exc}") from exc
        finally:
            try:
                if hasattr(resp, "close"):
                    resp.close()
                if hasattr(resp, "connection") and resp.connection:
                    resp.connection.close()
            except Exception:
                pass

    @staticmethod
    def _iter_http_client(resp) -> Iterator[bytes]:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            yield chunk

    # --------------------------------------------------------------------------
    # Resumable generation (Section 2) — streaming returns StreamResult
    # --------------------------------------------------------------------------
    def resume(
        self,
        conv_id: str,
        checkpoint_id: Optional[str] = None,
        max_tokens: int = 0,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Resume an interrupted generation (non‑streaming).

        Args:
            conv_id: Conversation ID.
            checkpoint_id: Optional specific checkpoint ID (uses latest if None).
            max_tokens: Additional tokens to generate.
            temperature: Override temperature (uses original if None).
            timeout: Request timeout.

        Returns:
            Dict with the full completion (same schema as /v1/chat/completions).
        """
        body = {
            "conv_id": conv_id,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if checkpoint_id:
            body["checkpoint_id"] = checkpoint_id
        if temperature is not None:
            body["temperature"] = temperature

        return self._request(
            "POST",
            "/v1/chat/resume",
            body=body,
            timeout=timeout or self.cfg.timeout,
        )

    def resume_stream(
        self,
        conv_id: str,
        checkpoint_id: Optional[str] = None,
        max_tokens: int = 0,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> StreamResult:
        """
        Resume an interrupted generation (streaming).

        Returns a `StreamResult`. Iteration yields `str` chunks
        identically to the pre-generator behavior; `.stats` is
        populated after exhaustion with authoritative server numbers or a chunk-derived fallback.

        The resume endpoint on this server uses SSE (per-token frames), so
        chunk count == token count exactly when server stats are unavailable.
        """
        body = {
            "conv_id": conv_id,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if checkpoint_id:
            body["checkpoint_id"] = checkpoint_id
        if temperature is not None:
            body["temperature"] = temperature

        result = StreamResult()
        result._impl = self._stream_with_stats(
            "/v1/chat/resume",
            body=body,
            timeout=timeout or self.cfg.timeout,
            sse=True,
            _result=result,
        )
        return result

    def _read_sse_stream(
        self,
        resp,
        _state: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Parse Server‑Sent Events (SSE) from a streaming response,
        yielding only the content from `data: {...}` events.

        When `_state` is provided, updates:
          * `first_ts`, `last_ts`, `chunk_count` — same as `_read_stream`
          * `finish_reason` — extracted from the last non-null
            `finish_reason` in any frame
        """
        buffer = ""
        saw_done = False
        got_any = False
        try:
            if _HAS_URLLIB3 and hasattr(resp, "stream"):
                chunk_iter = resp.stream(4096)
            else:
                chunk_iter = self._iter_http_client(resp)

            for raw in chunk_iter:
                if not raw:
                    continue
                try:
                    text = raw.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    text = raw.decode("utf-8", errors="replace")
                buffer += text
                lines = buffer.splitlines(keepends=True)
                buffer = ""
                for line in lines:
                    if line.endswith("\n"):
                        buffer += line
                    else:
                        # Incomplete line, hold back
                        buffer = line
                        break
                # Process complete lines
                lines = buffer.splitlines()
                buffer = ""
                for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload == "[DONE]":
                            saw_done = True
                            return
                        try:
                            obj = json.loads(payload)
                            if "choices" in obj and obj["choices"]:
                                choice = obj["choices"][0]
                                # capture finish_reason
                                fr = choice.get("finish_reason")
                                if fr and _state is not None:
                                    _state["finish_reason"] = str(fr)
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if _state is not None:
                                        now = time.monotonic()
                                        if _state["first_ts"] is None:
                                            _state["first_ts"] = now
                                        _state["last_ts"] = now
                                        _state["chunk_count"] += 1
                                    got_any = True
                                    yield content
                            elif "error" in obj:
                                raise StreamError(obj["error"])
                        except json.JSONDecodeError:
                            # Not JSON – yield as‑is (fallback)
                            yield payload
            # Flush any leftover
            if buffer.strip():
                got_any = True
                yield buffer
            if not saw_done and not got_any:
                raise StreamError(
                    "server closed the stream before sending any tokens"
                )
        except StreamError:
            raise
        except Exception as exc:
            raise StreamError(f"SSE stream read error: {exc}") from exc
        finally:
            try:
                if hasattr(resp, "close"):
                    resp.close()
                if hasattr(resp, "connection") and resp.connection:
                    resp.connection.close()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Logs
    # --------------------------------------------------------------------------
    def get_logs(self, n: int = 100, since: float = 0.0) -> Dict[str, Any]:
        return self._request("GET", f"/v1/logs?n={n}&since={since}")

    # --------------------------------------------------------------------------
    # API Key management
    # --------------------------------------------------------------------------
    def create_key(
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
        return self._request("POST", "/admin/keys/create", body=body, admin=True)

    def revoke_key(self, key_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/admin/keys/revoke/{key_id}", admin=True)

    def activate_key(self, key_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/admin/keys/activate/{key_id}", admin=True)

    def delete_key(self, key_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/admin/keys/delete/{key_id}", admin=True)

    def list_keys(self, active_only: bool = True) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/admin/keys/list?active_only={str(active_only).lower()}", admin=True)
        return data.get("keys", [])

    def key_stats(self, key_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/admin/keys/stats/{key_id}", admin=True)

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Word-count HEURISTIC  — for BUDGET sizing only, not display.

        For real token counting on the client:
          • `ChunkTokenCounter`  (from `_accel.py`) — streamed output
          • `TokenizerAdapter`   (from `_accel.py`) — arbitrary text
          • `StreamResult.stats` (this module)      — after chat_stream /
                                                     stream / resume_stream
          • `GET /v1/requests/{id}/stats`           — authoritative,
                                                     post-request
        """
        return estimate_tokens_fast(text)

    def close(self) -> None:
        """Close the connection pool."""
        try:
            if self._pool and hasattr(self._pool, "close"):
                self._pool.close()
        except Exception:
            pass

    def __enter__(self) -> "LLMCloudClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
