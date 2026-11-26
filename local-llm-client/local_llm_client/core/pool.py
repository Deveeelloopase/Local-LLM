"""
local_llm_client/core/pool.py

Production‑ready thread‑safe and async HTTP connection pool with retry,
backoff, authentication, and comprehensive error handling.

Features:
- Bounded concurrency (pool_size limit) for both sync and async operations
- Exponential backoff with jitter on transient errors
- Automatic fallback from urllib3 to urllib.request if urllib3 missing
- Full aiohttp support for async operations with fallback to sync
- API key and master key injection for admin endpoints
- Streaming response support
- Detailed per‑pool statistics
- No hardcoded paths – all configuration passed via constructor
- Integrates seamlessly with the core exception hierarchy
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from queue import Empty, Queue
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Union

# Core exception hierarchy (shared with client classes)
from .exceptions import (
    AuthError,
    ConnectionError as CoreConnectionError,
    RateLimitError,
    ServerError,
    TimeoutError as CoreTimeoutError,
)

# ------------------------------------------------------------------------------
# HTTP backend detection
# ------------------------------------------------------------------------------
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _HAS_URLLIB3 = True
except ImportError:
    _HAS_URLLIB3 = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    _HAS_AIOHTTP = False

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ConnectionPool (synchronous)
# ------------------------------------------------------------------------------
class ConnectionPool:
    """
    Fixed‑size pool of reusable synchronous HTTP connections.

    Uses urllib3 if available, otherwise falls back to urllib.request.
    Thread‑safe.  Supports authentication via API key and master key
    (injected into headers automatically for admin paths).

    All configuration is passed via constructor; no global state.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        master_key: str = "",
        pool_size: int = 8,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        retry_jitter: float = 0.1,
    ) -> None:
        """
        Args:
            base_url: Server endpoint (e.g., http://localhost:8080)
            api_key: Optional API key for regular endpoints
            master_key: Optional master key for admin endpoints
            pool_size: Maximum number of concurrent connections
            timeout: Default request timeout in seconds
            max_retries: Number of retry attempts on transient errors
            retry_delay: Initial retry delay in seconds
            retry_backoff: Exponential backoff multiplier
            retry_jitter: Random jitter fraction (0.0 to 1.0)
        """
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._master_key = master_key

        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._retry_jitter = retry_jitter

        # Pool slot management (semaphore via Queue)
        self._pool: Queue = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._pool.put(True)
        self._pool_size = pool_size

        # Stats
        self._lock = threading.Lock()
        self._total_reqs = 0
        self._errors = 0

        # urllib3 pool (if available)
        self._urllib3_pool: Optional[Union[urllib3.HTTPConnectionPool, urllib3.HTTPSConnectionPool]] = None
        if _HAS_URLLIB3:
            self._init_urllib3_pool()

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _init_urllib3_pool(self) -> None:
        """Initialise urllib3 connection pool."""
        scheme = "https" if self.base_url.startswith("https") else "http"
        url_part = self.base_url.split("://", 1)[1]
        host_port = url_part.split("/")[0]
        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            port = 443 if scheme == "https" else 80
        cls = urllib3.HTTPSConnectionPool if scheme == "https" else urllib3.HTTPConnectionPool
        self._urllib3_pool = cls(
            host,
            port=port,
            maxsize=self._pool_size,
            retries=False,
            timeout=urllib3.Timeout(connect=self._timeout, read=self._timeout),
        )

    def _headers(self, extra: Optional[Dict] = None, admin: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        if admin and self._master_key:
            h["X-Master-Key"] = self._master_key
        if extra:
            h.update(extra)
        return h

    def _check_status(self, status: int, body: bytes, url: str) -> None:
        """Translate HTTP status codes into typed exceptions."""
        msg = body.decode(errors="replace")[:300] if body else ""
        if status == 200:
            return
        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}): {msg}")
        if status == 429:
            retry_after = 0.0
            if body:
                try:
                    data = json.loads(body)
                    retry_after = float(data.get("retry_after", 0))
                except Exception:
                    pass
            raise RateLimitError(f"Rate limited – retry after {retry_after:.1f}s", retry_after=retry_after)
        if status == 503 and ("No model" in msg or "not loaded" in msg.lower()):
            raise AuthError(f"Model not loaded ({status}): {msg}")
        if 500 <= status < 600:
            raise ServerError(f"Server error {status} on {url}: {msg}")
        raise ServerError(f"HTTP {status} on {url}: {msg}")

    # --------------------------------------------------------------------------
    # Slot management
    # --------------------------------------------------------------------------
    @contextmanager
    def acquire(self, timeout: Optional[float] = None) -> Generator[None, None, None]:
        """
        Context manager that acquires a slot from the pool.
        Blocks until a slot is available or timeout expires.
        """
        t = timeout if timeout is not None else self._timeout
        try:
            self._pool.get(timeout=t)
        except Empty as exc:
            raise CoreConnectionError(
                f"Pool exhausted ({self._pool_size} slots all in use) after {t:.1f}s wait"
            ) from exc
        try:
            yield
        finally:
            self._pool.put(True)

    # --------------------------------------------------------------------------
    # Core request method
    # --------------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        admin: bool = False,
        stream: bool = False,
    ) -> Any:
        """
        Perform an HTTP request with retries and backoff.
        Returns parsed JSON body (or raw response object if stream=True).
        """
        url = self.base_url + path
        headers = self._headers(extra=extra_headers, admin=admin)
        payload = json.dumps(body).encode() if body else None
        t_out = timeout if timeout is not None else self._timeout
        last_exc: Exception = RuntimeError("No request made")

        with self.acquire():
            for attempt in range(self._max_retries + 1):
                try:
                    resp = self._do_request(method, url, payload, headers, t_out, stream)
                    with self._lock:
                        self._total_reqs += 1
                    if stream:
                        return resp
                    status = self._get_status(resp)
                    raw = self._read_body(resp)
                    self._check_status(status, raw, url)
                    return json.loads(raw) if raw else {}
                except (AuthError, RateLimitError, CoreConnectionError, CoreTimeoutError, ServerError) as exc:
                    if isinstance(exc, (AuthError, RateLimitError)):
                        # Non‑retryable
                        with self._lock:
                            self._errors += 1
                        raise
                    last_exc = exc
                    with self._lock:
                        self._errors += 1
                    if attempt < self._max_retries:
                        delay = self._retry_delay * (self._retry_backoff ** attempt)
                        jitter = random.uniform(-self._retry_jitter, self._retry_jitter)
                        delay = delay * (1 + jitter)
                        log.warning(
                            "Request %s %s failed (attempt %d/%d): %s – retrying in %.2fs",
                            method, path, attempt + 1, self._max_retries + 1, exc, delay
                        )
                        time.sleep(delay)
                    else:
                        raise
                except Exception as exc:
                    with self._lock:
                        self._errors += 1
                    raise

        raise last_exc

    def _do_request(
        self,
        method: str,
        url: str,
        payload: Optional[bytes],
        headers: Dict[str, str],
        timeout: float,
        stream: bool,
    ) -> Any:
        """Execute a single HTTP request."""
        if _HAS_URLLIB3 and self._urllib3_pool is not None:
            import urllib.parse as up
            parsed = up.urlsplit(url)
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            try:
                resp = self._urllib3_pool.request(
                    method=method,
                    url=path,
                    body=payload,
                    headers=headers,
                    timeout=timeout,
                    preload_content=not stream,
                )
                return resp
            except urllib3.exceptions.MaxRetryError as exc:
                raise CoreConnectionError(f"Cannot connect to {url}: {exc.reason}") from exc
            except urllib3.exceptions.TimeoutError as exc:
                raise CoreTimeoutError(f"Request to {url} timed out ({timeout}s)") from exc

        # Fallback to urllib.request
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        req = Request(url, data=payload, headers=headers, method=method)
        try:
            return urlopen(req, timeout=timeout)
        except HTTPError as exc:
            raw_body = exc.read() if exc.fp else b""
            self._check_status(exc.code, raw_body, url)
            raise
        except (URLError, OSError) as exc:
            raise CoreConnectionError(f"Cannot reach {url}: {exc}") from exc
        except TimeoutError as exc:
            raise CoreTimeoutError(f"Request to {url} timed out after {timeout}s") from exc

    @staticmethod
    def _get_status(resp) -> int:
        return resp.status if hasattr(resp, "status") else resp.status  # http.client fallback

    @staticmethod
    def _read_body(resp) -> bytes:
        return resp.data if hasattr(resp, "data") else resp.read()

    # --------------------------------------------------------------------------
    # Streaming helper
    # --------------------------------------------------------------------------
    def stream_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        timeout: Optional[float] = None,
        admin: bool = False,
    ):
        """
        Returns a raw response object for streaming.
        Caller is responsible for reading and closing it.
        """
        return self.request(method, path, body=body, timeout=timeout, admin=admin, stream=True)

    # --------------------------------------------------------------------------
    # Statistics & cleanup
    # --------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            return {
                "pool_size": self._pool_size,
                "available": self._pool.qsize(),
                "total_reqs": self._total_reqs,
                "errors": self._errors,
                "error_rate": round(self._errors / max(self._total_reqs, 1), 4),
            }

    def close(self) -> None:
        """Close the underlying urllib3 pool if present."""
        if _HAS_URLLIB3 and self._urllib3_pool is not None:
            try:
                self._urllib3_pool.close()
            except Exception:
                pass


# ------------------------------------------------------------------------------
# AsyncConnectionPool (asynchronous)
# ------------------------------------------------------------------------------
class AsyncConnectionPool:
    """
    Async version of ConnectionPool using aiohttp with semaphore‑based
    concurrency control.  Falls back to the synchronous ConnectionPool
    run in an executor if aiohttp is unavailable.

    Thread‑safe and asyncio‑safe.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        master_key: str = "",
        pool_size: int = 50,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        retry_jitter: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._master_key = master_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._retry_jitter = retry_jitter

        # Semaphore for limiting concurrency
        self._semaphore = asyncio.Semaphore(pool_size)
        self._pool_size = pool_size

        # Stats
        self._lock = asyncio.Lock()
        self._total_reqs = 0
        self._errors = 0

        # aiohttp session (lazy)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Synchronous fallback pool (lazy)
        self._sync_pool: Optional[ConnectionPool] = None

    # --------------------------------------------------------------------------
    # Session management
    # --------------------------------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        if not _HAS_AIOHTTP:
            raise ImportError("aiohttp is required for async operations. Install it: pip install aiohttp")
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(limit=self._pool_size)
                timeout = aiohttp.ClientTimeout(total=self._timeout)
                self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            return self._session

    def _get_sync_pool(self) -> ConnectionPool:
        if self._sync_pool is None:
            self._sync_pool = ConnectionPool(
                base_url=self.base_url,
                api_key=self._api_key,
                master_key=self._master_key,
                pool_size=self._pool_size,
                timeout=self._timeout,
                max_retries=self._max_retries,
                retry_delay=self._retry_delay,
                retry_backoff=self._retry_backoff,
                retry_jitter=self._retry_jitter,
            )
        return self._sync_pool

    # --------------------------------------------------------------------------
    # Headers
    # --------------------------------------------------------------------------
    def _headers(self, extra: Optional[Dict] = None, admin: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        if admin and self._master_key:
            h["X-Master-Key"] = self._master_key
        if extra:
            h.update(extra)
        return h

    # --------------------------------------------------------------------------
    # Core async request
    # --------------------------------------------------------------------------
    async def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        admin: bool = False,
    ) -> Dict[str, Any]:
        """Perform an asynchronous HTTP request with retries."""
        if not _HAS_AIOHTTP:
            # Fallback to sync pool in executor
            loop = asyncio.get_running_loop()
            sync_pool = self._get_sync_pool()
            return await loop.run_in_executor(
                None,
                sync_pool.request,
                method, path, body, extra_headers, timeout, admin
            )

        url = self.base_url + path
        headers = self._headers(extra=extra_headers, admin=admin)
        t_out = timeout if timeout is not None else self._timeout
        last_exc: Exception = RuntimeError("No attempt made")

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    session = await self._get_session()
                    async with session.request(
                        method, url,
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=t_out),
                    ) as resp:
                        status = resp.status
                        raw = await resp.read()
                        self._check_status(status, raw, path)
                        async with self._lock:
                            self._total_reqs += 1
                        return json.loads(raw) if raw else {}
                except (AuthError, RateLimitError):
                    async with self._lock:
                        self._errors += 1
                    raise
                except (CoreConnectionError, CoreTimeoutError, ServerError) as exc:
                    last_exc = exc
                    async with self._lock:
                        self._errors += 1
                    if attempt < self._max_retries:
                        delay = self._retry_delay * (self._retry_backoff ** attempt)
                        jitter = random.uniform(-self._retry_jitter, self._retry_jitter)
                        delay = delay * (1 + jitter)
                        log.warning(
                            "Async request %s %s failed (attempt %d/%d): %s – retrying in %.2fs",
                            method, path, attempt + 1, self._max_retries + 1, exc, delay
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
                except Exception as exc:
                    async with self._lock:
                        self._errors += 1
                    raise

        raise last_exc

    def _check_status(self, status: int, body: bytes, path: str) -> None:
        msg = body.decode(errors="replace")[:300] if body else ""
        if status == 200:
            return
        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}): {msg}")
        if status == 429:
            retry_after = 0.0
            if body:
                try:
                    data = json.loads(body)
                    retry_after = float(data.get("retry_after", 0))
                except Exception:
                    pass
            raise RateLimitError(f"Rate limited – retry after {retry_after:.1f}s", retry_after=retry_after)
        if status == 503 and ("No model" in msg or "not loaded" in msg.lower()):
            raise AuthError(f"Model not loaded ({status}): {msg}")
        if status >= 500:
            raise ServerError(f"Server error {status} on {path}: {msg}")
        raise ServerError(f"HTTP {status} on {path}: {msg}")

    # --------------------------------------------------------------------------
    # Async streaming
    # --------------------------------------------------------------------------
    async def stream_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        timeout: Optional[float] = None,
        admin: bool = False,
    ) -> AsyncIterator[bytes]:
        """
        Stream a response body in chunks.
        Yields raw bytes.
        """
        if not _HAS_AIOHTTP:
            loop = asyncio.get_running_loop()
            sync_pool = self._get_sync_pool()
            # In sync fallback, read whole response and yield once (not optimal)
            resp_body = await loop.run_in_executor(
                None,
                sync_pool.request,
                method, path, body, None, timeout, admin
            )
            if resp_body:
                yield json.dumps(resp_body).encode()
            return

        url = self.base_url + path
        headers = self._headers(admin=admin)
        t_out = timeout if timeout is not None else self._timeout

        async with self._semaphore:
            session = await self._get_session()
            async with session.request(
                method, url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=t_out),
            ) as resp:
                status = resp.status
                if status != 200:
                    raw = await resp.read()
                    self._check_status(status, raw, path)
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        yield chunk

    # --------------------------------------------------------------------------
    # Statistics & cleanup
    # --------------------------------------------------------------------------
    async def stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        async with self._lock:
            return {
                "pool_size": self._pool_size,
                "total_reqs": self._total_reqs,
                "errors": self._errors,
                "error_rate": round(self._errors / max(self._total_reqs, 1), 4),
            }

    async def close(self) -> None:
        """Close the underlying aiohttp session and sync pool."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
        if self._sync_pool:
            self._sync_pool.close()
            self._sync_pool = None
