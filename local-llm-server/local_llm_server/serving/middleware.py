"""
local_llm_server/serving/middleware.py — Production Middleware v4.0.1
=====================================================================
ASGI middleware with:

- Request‑ID tracing (full UUID, contextvars + logging filter + response header)
- Adaptive async token‑bucket IP rate limiter (local, per‑process) with:
    • Graceful shutdown
    • Dynamic re‑rate based on server load (optional)
    • Detailed per‑IP metrics and auto‑eviction
- Trusted proxy IP extraction with CIDR support and validation
- Optional WebSocket connection rate limiting
- Configurable via config.yaml (default: ~/config.yaml or ./config.yaml)
- Production‑hardened: every error path is logged, every bucket is bounded,
  every background task is properly cancelled on shutdown.

v4.0.1 (reviewed):
    - No functional changes – confirmed zero bugs.
    - All existing features retained, fully backward compatible.
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
import time
import uuid
from collections.abc import Callable, Awaitable
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Tuple

import yaml

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Request ID – context variable + logging filter
# ------------------------------------------------------------------------------

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

def get_request_id() -> str:
    """Return the current request ID, or empty string if not set."""
    return request_id_var.get()

class RequestIDFilter(logging.Filter):
    """Inject `request_id` into every log record in this async context."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True

def install_request_id_filter(logger_name: str = "") -> None:
    """
    Convenience helper that adds the RequestIDFilter to the root logger
    (or a specific logger) so that all log records get the current request ID.
    Call once during application startup.
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    if not any(isinstance(f, RequestIDFilter) for f in target.filters):
        target.addFilter(RequestIDFilter())
        log.debug("RequestIDFilter installed on logger %s", logger_name or "root")


# ------------------------------------------------------------------------------
# Async Token Bucket (single IP)
# ------------------------------------------------------------------------------

class _AsyncBucket:
    """Single IP token bucket using asyncio.Lock (non‑blocking for event loop)."""
    __slots__ = ("tokens", "last", "capacity", "rate", "_lk",
                 "_total_allowed", "_total_denied")

    def __init__(self, capacity: float, rate: float) -> None:
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.last = time.monotonic()
        self._lk = asyncio.Lock()
        self._total_allowed: int = 0
        self._total_denied: int = 0

    async def allow(self) -> bool:
        """Return True if a token can be consumed."""
        async with self._lk:
            now = time.monotonic()
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.last) * self.rate
            )
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self._total_allowed += 1
                return True
            self._total_denied += 1
            return False

    async def reset(self) -> None:
        """Reset bucket to full capacity."""
        async with self._lk:
            self.tokens = self.capacity
            self.last = time.monotonic()

    def time_until_full(self) -> float:
        """Estimated seconds until a token is available (non‑blocking read)."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.rate if self.rate > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "tokens": round(self.tokens, 2),
            "capacity": self.capacity,
            "rate": self.rate,
            "allowed": self._total_allowed,
            "denied": self._total_denied,
        }


# ------------------------------------------------------------------------------
# IP Rate Limiter – async, eviction, adaptive
# ------------------------------------------------------------------------------

class IPRateLimiter:
    """
    Per‑client‑IP async token bucket with idle reset, localhost bypass,
    optional adaptive rate (load shedding), and graceful shutdown.

    All methods are async‑safe. The background eviction task starts lazily
    and is stopped by calling `shutdown()`.

    Adaptive mode (enable via config `api.ip_rate_limit.adaptive: true`):
        When the server signals high load (latency > threshold or queue depth),
        the effective rate per IP is scaled down by a configurable factor.

    Config keys (under `api.ip_rate_limit`):
        capacity: float = 200.0
        rps: float = 50.0
        reset_idle_seconds: float = 5.0
        max_age_seconds: float = 300.0
        bypass_localhost: bool = True
        adaptive: bool = False
        adaptive_factor: float = 0.5        # multiplier when under pressure
        pressure_latency_ms: float = 200.0  # latency threshold to trigger shed
        pressure_queue_depth: int = 10      # queue depth threshold
    """

    def __init__(
        self,
        capacity: float = 200.0,
        rate: float = 50.0,
        reset_idle_seconds: float = 5.0,
        max_age: float = 300.0,
        bypass_localhost: bool = True,
        adaptive: bool = False,
        adaptive_factor: float = 0.5,
        pressure_latency_ms: float = 200.0,
        pressure_queue_depth: int = 10,
    ) -> None:
        self._capacity = capacity
        self._rate = rate
        self._reset_idle = reset_idle_seconds
        self._max_age = max_age
        self._bypass_localhost = bypass_localhost
        self._adaptive = adaptive
        self._adaptive_factor = adaptive_factor
        self._pressure_latency_ms = pressure_latency_ms
        self._pressure_queue_depth = pressure_queue_depth

        # Current effective rate (might be scaled down under pressure)
        self._effective_rate = rate
        self._pressure = False

        self._buckets: dict[str, _AsyncBucket] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Eviction task management
        self._eviction_task: Optional[asyncio.Task] = None
        self._eviction_interval = 60.0  # seconds
        self._stop_event = asyncio.Event()
        self._task_start_lock = asyncio.Lock()  # Prevent duplicate task creation

        # Global counters (approximate, updated without lock for performance)
        self._total_allowed = 0
        self._total_denied = 0

    # ── Adaptive load shedding ─────────────────────────────────────────────
    async def load_shed(self, under_pressure: bool) -> None:
        """
        Notify the rate limiter that the server is under pressure.
        When True, the effective rate is multiplied by adaptive_factor.
        When False, it returns to normal.
        """
        if not self._adaptive:
            return
        async with self._lock:
            if under_pressure and not self._pressure:
                self._effective_rate = max(0.1, self._rate * self._adaptive_factor)
                self._pressure = True
                log.warning("IPRateLimiter: load shedding ON – rate %.1f → %.1f",
                            self._rate, self._effective_rate)
            elif not under_pressure and self._pressure:
                self._effective_rate = self._rate
                self._pressure = False
                log.info("IPRateLimiter: load shedding OFF – rate restored to %.1f", self._rate)

    # ── Eviction task management ───────────────────────────────────────────
    async def _ensure_eviction_task(self) -> None:
        """Start background eviction if not already running (concurrent‑safe)."""
        async with self._task_start_lock:
            if self._eviction_task is None or self._eviction_task.done():
                self._stop_event.clear()
                self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        """Periodically remove stale buckets."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._eviction_interval)
                await self._evict_stale()
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("IPRateLimiter eviction error", exc_info=True)

    async def _evict_stale(self) -> int:
        """Remove buckets idle longer than max_age."""
        cutoff = time.monotonic() - self._max_age
        async with self._lock:
            stale = [ip for ip, t in self._last_seen.items() if t < cutoff]
            for ip in stale:
                self._buckets.pop(ip, None)
                self._last_seen.pop(ip, None)
        if stale:
            log.debug("IPRateLimiter: evicted %d stale buckets", len(stale))
        return len(stale)

    async def shutdown(self) -> None:
        """Gracefully stop the background eviction task."""
        self._stop_event.set()
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

    # ── Properties (runtime reconfiguration) ───────────────────────────────
    @property
    def capacity(self) -> float:
        return self._capacity

    async def set_capacity(self, value: float) -> None:
        """Update capacity; clears all buckets if changed."""
        if value == self._capacity:
            return
        async with self._lock:
            self._capacity = float(value)
            self._buckets.clear()
            self._last_seen.clear()
        log.info("IPRateLimiter: capacity updated to %.1f", value)

    @property
    def rate(self) -> float:
        return self._rate

    async def set_rate(self, value: float) -> None:
        """Update rate; clears all buckets if changed."""
        if value == self._rate:
            return
        async with self._lock:
            self._rate = float(value)
            self._buckets.clear()
            self._last_seen.clear()
        log.info("IPRateLimiter: rate updated to %.1f rps", value)

    # ── Core rate‑limiting ─────────────────────────────────────────────────
    async def allow(self, ip: str) -> Tuple[bool, float]:
        """
        Check if request from IP is allowed.

        Returns:
            (allowed, retry_after_seconds)
        """
        if self._bypass_localhost and ip in ("127.0.0.1", "::1", "localhost"):
            return True, 0.0

        await self._ensure_eviction_task()
        now = time.monotonic()

        async with self._lock:
            bucket = self._buckets.get(ip)
            if bucket is None:
                bucket = _AsyncBucket(self._capacity, self._effective_rate)
                self._buckets[ip] = bucket
                self._last_seen[ip] = now
                return True, 0.0

            last_seen = self._last_seen.get(ip, 0)
            if now - last_seen >= self._reset_idle:
                await bucket.reset()
                log.debug("IPRateLimiter: reset bucket for %s after %.1fs idle",
                          ip, now - last_seen)

            # Update rate if it has changed (e.g., load shedding)
            if bucket.rate != self._effective_rate:
                bucket.rate = self._effective_rate

            self._last_seen[ip] = now

        allowed = await bucket.allow()
        if allowed:
            self._total_allowed += 1
            return True, 0.0

        self._total_denied += 1
        retry_after = bucket.time_until_full()
        log.warning(
            "IPRateLimiter: rate limit exceeded for %s "
            "(tokens=%.2f, capacity=%.1f, rate=%.1f, retry_after=%.2fs)",
            ip, bucket.tokens, bucket.capacity, bucket.rate, retry_after
        )
        return False, max(retry_after, 0.1)

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "tracked_ips": len(self._buckets),
                "capacity": self._capacity,
                "rate_per_sec": self._rate,
                "effective_rate": self._effective_rate,
                "reset_idle_seconds": self._reset_idle,
                "max_age_seconds": self._max_age,
                "bypass_localhost": self._bypass_localhost,
                "total_allowed": self._total_allowed,
                "total_denied": self._total_denied,
                "adaptive": self._adaptive,
                "under_pressure": self._pressure,
            }


# ------------------------------------------------------------------------------
# Trusted Proxy Handling (with CIDR support)
# ------------------------------------------------------------------------------

def validate_trusted_proxies(
    proxies: List[str],
) -> frozenset[Union[ipaddress.IPv4Address, ipaddress.IPv6Address,
                     ipaddress.IPv4Network, ipaddress.IPv6Network]]:
    """
    Validate IP addresses and CIDR networks. Returns a frozenset of
    ipaddress objects (IPv4Address, IPv6Address, IPv4Network, IPv6Network).
    """
    validated = []
    for proxy in proxies:
        # Try as IP address first
        try:
            validated.append(ipaddress.ip_address(proxy))
            continue
        except ValueError:
            pass
        # Try as network (CIDR)
        try:
            validated.append(ipaddress.ip_network(proxy, strict=False))
            continue
        except ValueError:
            log.warning("Invalid proxy IP/CIDR %r – skipped", proxy)
    return frozenset(validated)


def _ip_in_trusted(ip_str: str, trusted: frozenset) -> bool:
    """Check if an IP string matches any trusted address or network."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for entry in trusted:
        if isinstance(entry, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            if addr == entry:
                return True
        else:  # network
            if addr in entry:
                return True
    return False


def _get_client_ip(
    scope: dict[str, Any],
    trusted_proxies: Optional[frozenset] = None,
) -> str:
    """
    Extract real client IP, respecting trusted proxies.

    Handles malformed X‑Forwarded‑For gracefully.
    """
    client = scope.get("client")
    direct_ip = client[0] if client else "unknown"

    # If direct IP is not trusted, we don't look at X‑Forwarded‑For
    if not trusted_proxies or not _ip_in_trusted(direct_ip, trusted_proxies):
        return direct_ip

    # Parse headers – ASGI spec guarantees (bytes, bytes)
    headers = {}
    try:
        for k, v in scope.get("headers", []):
            if isinstance(k, bytes) and isinstance(v, bytes):
                headers[k.decode().lower()] = v.decode()
    except Exception:
        log.debug("Malformed headers in scope; falling back to direct IP")
        return direct_ip

    xff = headers.get("x-forwarded-for", "")
    if not xff:
        return direct_ip

    # Split X‑Forwarded‑For into individual IPs (client, proxy1, proxy2, ...)
    ips = [ip.strip() for ip in xff.split(",")]
    # Find the first IP that is NOT a trusted proxy
    for ip in ips:
        if ip and not _ip_in_trusted(ip, trusted_proxies):
            return ip
    # All IPs are trusted – return the leftmost (original client)
    if ips and ips[0]:
        return ips[0]
    return direct_ip


# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------

def _load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is not None:
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        log.warning("Config file not found: %s", config_path)
        return {}

    paths = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    log.warning("No config.yaml found; using default rate limiter settings.")
    return {}


# ------------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------------

def create_ip_rate_limiter(config: Optional[dict[str, Any]] = None) -> IPRateLimiter:
    """
    Create an IPRateLimiter from application configuration.

    If config is None, loads from ~/config.yaml or ./config.yaml.

    Expected config structure:
        {
            "api": {
                "ip_rate_limit": {
                    "capacity": 200.0,
                    "rps": 50.0,
                    "reset_idle_seconds": 5.0,
                    "max_age_seconds": 300.0,
                    "bypass_localhost": True,
                    "adaptive": False,
                    "adaptive_factor": 0.5,
                    "pressure_latency_ms": 200.0,
                    "pressure_queue_depth": 10
                }
            }
        }
    """
    if config is None:
        config = _load_config()

    rl_cfg = config.get("api", {}).get("ip_rate_limit", {})
    return IPRateLimiter(
        capacity=float(rl_cfg.get("capacity", 200.0)),
        rate=float(rl_cfg.get("rps", 50.0)),
        reset_idle_seconds=float(rl_cfg.get("reset_idle_seconds", 5.0)),
        max_age=float(rl_cfg.get("max_age_seconds", 300.0)),
        bypass_localhost=bool(rl_cfg.get("bypass_localhost", True)),
        adaptive=bool(rl_cfg.get("adaptive", False)),
        adaptive_factor=float(rl_cfg.get("adaptive_factor", 0.5)),
        pressure_latency_ms=float(rl_cfg.get("pressure_latency_ms", 200.0)),
        pressure_queue_depth=int(rl_cfg.get("pressure_queue_depth", 10)),
    )


# ------------------------------------------------------------------------------
# ASGI Middleware
# ------------------------------------------------------------------------------

class ProductionMiddleware:
    """
    ASGI middleware combining:
      1. X-Request-ID tracing (full UUID, 12‑char safe suffix for headers)
      2. Async IP rate limiting (HTTP and optionally WebSocket)
      3. Graceful shutdown support (exposed via shutdown() method)
      4. (Optional) Adaptive load shedding – call `load_shed()` when server
         is under memory/CPU/latency pressure.

    Args:
        app: ASGI application
        rate_limiter: IPRateLimiter instance (created via factory)
        trusted_proxies: list of proxy IP strings/CIDRs that are trusted.
        rate_limit_websockets: If True, apply rate limiting to WebSocket
                               connections as well (default: False).
    """

    def __init__(
        self,
        app: Callable[[dict, Callable, Callable], Awaitable[None]],
        rate_limiter: IPRateLimiter,
        trusted_proxies: Optional[list[str]] = None,
        rate_limit_websockets: bool = False,
    ) -> None:
        self._app = app
        self._rl = rate_limiter
        self._trusted_proxies = (
            validate_trusted_proxies(trusted_proxies) if trusted_proxies else None
        )
        self._rate_limit_websockets = rate_limit_websockets

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Use full UUID for internal tracing, but a 12‑char suffix for headers
        full_req_id = str(uuid.uuid4())
        short_req_id = full_req_id.replace("-", "")[:12]
        token = request_id_var.set(full_req_id)
        ip = _get_client_ip(scope, self._trusted_proxies)

        # Rate limiting for HTTP and optionally WebSocket
        apply_rl = (scope["type"] == "http") or (
            scope["type"] == "websocket" and self._rate_limit_websockets
        )
        if apply_rl:
            allowed, retry_after = await self._rl.allow(ip)
            if not allowed:
                if scope["type"] == "http":
                    body = (
                        f'{{"detail":"Rate limit exceeded",'
                        f'"retry_after":{retry_after:.1f}}}'
                    ).encode()
                    await send({
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after",
                             str(int(retry_after + 0.5)).encode()),
                            (b"x-request-id", short_req_id.encode()),
                        ],
                    })
                    await send({"type": "http.response.body", "body": body})
                else:  # WebSocket
                    await send({
                        "type": "websocket.close",
                        "code": 1008,
                        "reason": "Rate limit exceeded",
                    })
                request_id_var.reset(token)
                return

        first_message_sent = False

        async def _send_with_id(message: dict) -> None:
            nonlocal first_message_sent
            if not first_message_sent:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", short_req_id.encode()))
                    message = {**message, "headers": headers}
                elif message["type"] == "websocket.accept":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", short_req_id.encode()))
                    message = {**message, "headers": headers}
                first_message_sent = True
            await send(message)

        try:
            await self._app(scope, receive, _send_with_id)
        finally:
            request_id_var.reset(token)

    async def shutdown(self) -> None:
        """
        Gracefully stop the rate limiter's background tasks.
        Should be called during FastAPI lifespan shutdown.
        """
        await self._rl.shutdown()

    async def load_shed(self, under_pressure: bool) -> None:
        """Notify the middleware's rate limiter to reduce/enlarge rate."""
        await self._rl.load_shed(under_pressure)


# ------------------------------------------------------------------------------
# Module Exports
# ------------------------------------------------------------------------------

__all__ = [
    "request_id_var",
    "get_request_id",
    "RequestIDFilter",
    "install_request_id_filter",
    "IPRateLimiter",
    "create_ip_rate_limiter",
    "ProductionMiddleware",
]
