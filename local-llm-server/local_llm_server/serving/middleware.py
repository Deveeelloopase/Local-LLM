"""
local_llm_server/serving/middleware.py — Production Middleware
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

version = 1.0.0
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
import threading
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

def set_request_id(request_id: str) -> contextvars.Token:
    """Set the request ID for the current context and return the reset token.

    helper for non-middleware callers. The middleware itself uses
    ``request_id_var.set()`` inline and resets with the returned token in a
    ``finally`` block. External callers (background tasks, startup hooks,
    log-correlation helpers in the CLI, etc.) can do the same via:

        token = set_request_id("req-abc123")
        try:
            ...  # any logs emitted here carry request_id=req-abc123
        finally:
            reset_request_id(token)

    Args:
        request_id: The ID string to set. Pass an empty string to clear
                    (though ``reset_request_id(token)`` is preferred for
                    restoring a prior value).

    Returns:
        The token returned by ``contextvars.ContextVar.set()``. Pass it to
        ``reset_request_id()`` in a ``finally`` block to restore the
        previous value (which may be ``""`` at the top level, or an
        outer request's ID when nested).
    """
    return request_id_var.set(str(request_id or ""))

def reset_request_id(token: contextvars.Token) -> None:
    """Restore the context var to the value it had before ``set_request_id``.

    Symmetric with ``set_request_id``. Silently ignores a ``None`` token —
    callers who forgot to capture the return value get a no-op rather than
    a TypeError, so a bug in their code does not break the log pipeline.
    """
    if token is None:
        return
    try:
        request_id_var.reset(token)
    except (ValueError, RuntimeError):
        # ValueError: token was created in a different context (a nested
        # task, wrong thread, etc.). RuntimeError: token already reset.
        # Either way, the safest action is to leave the current context
        # alone — logging continues with whatever ID is present.
        pass


class RequestIDFilter(logging.Filter):
    """Inject `request_id` into every log record in this async context.

    ``_TextFormatter`` renders ``[request_id]`` when the attribute
    is present, ``[-]`` when not. This filter sets it to the current
    contextvar value (or ``"-"`` when unset) so it is ALWAYS present —
    no formatter crashes on a missing attribute, no ``[-]`` unless the
    request really has no ID.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def install_request_id_filter(logger_name: str = "") -> None:
    """
    Convenience helper that adds the RequestIDFilter to the root logger
    (or a specific logger) so that all log records get the current request ID.
    Call once during application startup.

    Idempotent — repeated calls do not stack duplicate filters.
    """
    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    if not any(isinstance(f, RequestIDFilter) for f in target.filters):
        target.addFilter(RequestIDFilter())
        log.warning("RequestIDFilter installed on logger %s", logger_name or "root")


# ------------------------------------------------------------------------------
# Prometheus histograms — singleton, idempotent, no-op fallback
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Histogram
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False
    Histogram = None  # type: ignore[assignment]


class _NoOpMetric:
    """Silent fallback for a metric that could not be registered.

    Registration of a metric name that already exists in the process-global
    Prometheus REGISTRY raises ValueError. Rather than let that abort import
    (which would kill the whole middleware module), fall back to this no-op.
    Matches the pattern in log_handler.py, key_store.py, and
    metrics_collector.py.
    """
    def labels(self, *args, **kwargs): return self
    def inc(self, amount=1): pass
    def set(self, value): pass
    def observe(self, value): pass
    def time(self): return self


_PROM_MIDDLEWARE_METRICS: Dict[str, Any] = {}
_PROM_MIDDLEWARE_LOCK = threading.Lock()


def _setup_middleware_metrics() -> Dict[str, Any]:
    """Return the process-global Prometheus histogram dict.

    First call registers the metrics; subsequent calls return the cached
    dict — safe under uvicorn ``--reload`` and repeated middleware init.
    """
    global _PROM_MIDDLEWARE_METRICS
    if not _HAS_PROM:
        return {}
    with _PROM_MIDDLEWARE_LOCK:
        if _PROM_MIDDLEWARE_METRICS:
            return _PROM_MIDDLEWARE_METRICS

        def _make_hist(name: str, doc: str, labels: List[str], buckets: List[float]):
            try:
                return Histogram(name, doc, labels, buckets=buckets)
            except ValueError:
                # Duplicate — return a no-op so the recording helpers stay
                # safe (they only call .labels(...).observe(...)).
                return _NoOpMetric()
            except Exception:
                return _NoOpMetric()

        _PROM_MIDDLEWARE_METRICS = {
            # Covers 50 ms (fast GPU) to 60 s (slow APU prefill).
            "ttft_seconds": _make_hist(
                "llm_ttft_seconds",
                "Time to first token, in seconds",
                ["model", "route"],
                [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            ),
            # Matches the plan's tuning targets: 0.5 (slow) to 200 tps.
            "decode_tokens_per_second": _make_hist(
                "llm_decode_tokens_per_second",
                "Decode-only tokens per second (excludes prefill time)",
                ["model", "route"],
                [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
            ),
            # Covers instant admission (10 ms) through overload (300 s).
            "queue_wait_seconds": _make_hist(
                "llm_queue_wait_seconds",
                "Time spent waiting in the server queue, in seconds",
                ["route"],
                [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            ),
        }
        return _PROM_MIDDLEWARE_METRICS


def record_ttft_seconds(model: str, seconds: float, route: str = "") -> None:
    """Record TTFT (in seconds) into the ``llm_ttft_seconds`` histogram.

    Called once per completed inference request from
    ``api_server._log_request()``. Safe to call when Prometheus is
    not installed (no-op) or when the value is zero (some request types,
    e.g. cached prefix hits, have negligible TTFT — recording 0.0 is
    meaningful, so we do not skip it).

    Args:
        model:   Model ID, or "" for routes that do not know it.
        seconds: TTFT in seconds (from ``ttft_ms / 1000.0``). Negative
                 values are clamped to 0.
        route:   Route path (e.g. "/v1/chat/completions"). "" → "unknown".
    """
    m = _setup_middleware_metrics().get("ttft_seconds")
    if not m:
        return
    try:
        m.labels(
            model=str(model or "unknown"),
            route=str(route or "unknown"),
        ).observe(max(0.0, float(seconds)))
    except Exception:
        pass


def record_decode_tps(model: str, tps: float, route: str = "") -> None:
    """Record decode TPS into the ``llm_decode_tokens_per_second`` histogram.

    Called once per completed inference request. Uses the
    decode-only TPS (``decode_tps`` from ``/v1/requests/{id}/stats``), NOT
    the end-to-end TPS — end-to-end conflates prefill and is 30-40 % lower
    on this box, which would make the histogram useless for tuning.

    Values <= 0 are skipped (a zero means "not measured", not "0 tps").
    """
    m = _setup_middleware_metrics().get("decode_tokens_per_second")
    if not m:
        return
    v = float(tps)
    if v <= 0:
        return
    try:
        m.labels(
            model=str(model or "unknown"),
            route=str(route or "unknown"),
        ).observe(v)
    except Exception:
        pass


def record_queue_wait_seconds(seconds: float, route: str = "") -> None:
    """Record queue-wait (in seconds) into ``llm_queue_wait_seconds``.

    Called once per completed request. A queue wait of 0 (the common
    case when there is spare capacity) is recorded, not skipped — the
    histogram is a distribution, and "we admitted immediately" is a real
    observation that affects the mean and p95.
    """
    m = _setup_middleware_metrics().get("queue_wait_seconds")
    if not m:
        return
    try:
        m.labels(route=str(route or "unknown")).observe(max(0.0, float(seconds)))
    except Exception:
        pass


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

def _is_loopback(ip: str) -> bool:
    """True for any address that `ipaddress.is_loopback` accepts.

    the previous test `ip in ("127.0.0.1", "::1", "localhost")`
    missed:
      * `127.0.1.1` — many distros map the hostname there. Kali maps
        the machine hostname to this address, so `curl http://deve-oops:8080`
        was being rate-limited.
      * `::ffff:127.0.0.1` — the IPv4-mapped IPv6 form some stacks use.
      * `127.0.0.2` … `127.255.255.254` — the whole 127.0.0.0/8 block.

    Parses with `ipaddress`, unwraps IPv4-mapped IPv6, and defers to
    `is_loopback()`. Empty strings and unparseable values return False.
    """
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


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
        self._eviction_interval = 60.0
        self._stop_event = asyncio.Event()
        self._task_start_lock = asyncio.Lock()

        # Global counters (approximate, updated without lock for performance)
        self._total_allowed = 0
        self._total_denied = 0

    # ── Adaptive load shedding ─────────────────────────────────────────────
    async def load_shed(self, under_pressure: bool) -> None:
        """Notify the rate limiter that the server is under pressure."""
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
        async with self._task_start_lock:
            if self._eviction_task is None or self._eviction_task.done():
                self._stop_event.clear()
                self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._eviction_interval)
                await self._evict_stale()
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("IPRateLimiter eviction error", exc_info=True)

    async def _evict_stale(self) -> int:
        cutoff = time.monotonic() - self._max_age
        async with self._lock:
            stale = [ip for ip, t in self._last_seen.items() if t < cutoff]
            for ip in stale:
                self._buckets.pop(ip, None)
                self._last_seen.pop(ip, None)
        if stale:
            log.warning("IPRateLimiter: evicted %d stale buckets", len(stale))
        return len(stale)

    async def shutdown(self) -> None:
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

        loopback bypass now uses `ipaddress.is_loopback`, which
        covers `127.0.0.0/8`, `::1`, and the IPv4-mapped form
        `::ffff:127.0.0.1` — not just the two literal strings the old
        check accepted.
        """
        if self._bypass_localhost and _is_loopback(ip):
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
                log.warning("IPRateLimiter: reset bucket for %s after %.1fs idle",
                          ip, now - last_seen)

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
        try:
            validated.append(ipaddress.ip_address(proxy))
            continue
        except ValueError:
            pass
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
        else:
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

    if not trusted_proxies or not _ip_in_trusted(direct_ip, trusted_proxies):
        return direct_ip

    headers = {}
    try:
        for k, v in scope.get("headers", []):
            if isinstance(k, bytes) and isinstance(v, bytes):
                headers[k.decode().lower()] = v.decode()
    except Exception:
        log.warning("Malformed headers in scope; falling back to direct IP")
        return direct_ip

    xff = headers.get("x-forwarded-for", "")
    if not xff:
        return direct_ip

    ips = [ip.strip() for ip in xff.split(",")]
    for ip in ips:
        if ip and not _ip_in_trusted(ip, trusted_proxies):
            return ip
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

        # ── ONE request id, everywhere ──────────────────────────────
        # The middleware is the single source of truth. Handlers read
        # `request.state.request_id` (populated below via scope["state"]),
        # use it as `InferenceRequest.request_id`, and the header
        # `X-Request-ID` carries the same string. `_log_request` keys
        # `_request_stats_map` on this value, so a client that captured
        # the header can call `GET /v1/requests/{id}/stats` and find its
        # entry.
        #
        # The old version generated a 12-hex-char id for the header and
        # every handler generated a *separate* full uuid4 for
        # `ir.request_id` — the map was keyed on the handler's uuid, the
        # client looked it up by the header's 12-char value, and got a
        # 404 every time.
        full_req_id = str(uuid.uuid4())

        # Expose the id to FastAPI handlers. Starlette builds
        # `Request.state` on exactly this dict, so a handler doing
        # `req.state.request_id` reads the value below.
        _state = scope.setdefault("state", {})
        _state["request_id"] = full_req_id

        token = request_id_var.set(full_req_id)
        ip = _get_client_ip(scope, self._trusted_proxies)
        _state["client_ip"] = ip

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
                            (b"x-request-id", full_req_id.encode()),
                        ],
                    })
                    await send({"type": "http.response.body", "body": body})
                else:
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
                    # Full UUID (36 chars with hyphens) — matches
                    # `request.state.request_id` and the stats map key.
                    # Older 12-char-prefix clients still work because
                    # /v1/requests/{id}/stats accepts any unique prefix
                    # of a known id (see api_server's prefix-match fix).
                    headers.append((b"x-request-id", full_req_id.encode()))
                    message = {**message, "headers": headers}
                elif message["type"] == "websocket.accept":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", full_req_id.encode()))
                    message = {**message, "headers": headers}
                first_message_sent = True
            await send(message)

        try:
            await self._app(scope, receive, _send_with_id)
        finally:
            request_id_var.reset(token)

    async def shutdown(self) -> None:
        """Gracefully stop the rate limiter's background tasks."""
        await self._rl.shutdown()

    async def load_shed(self, under_pressure: bool) -> None:
        """Notify the middleware's rate limiter to reduce/enlarge rate."""
        await self._rl.load_shed(under_pressure)


# ------------------------------------------------------------------------------
# Module Exports
# ------------------------------------------------------------------------------

__all__ = [
    # Request ID
    "request_id_var",
    "get_request_id",
    "set_request_id",
    "reset_request_id",
    "RequestIDFilter",
    "install_request_id_filter",
    # Rate limiting & middleware
    "IPRateLimiter",
    "create_ip_rate_limiter",
    "ProductionMiddleware",
    # Prometheus histograms
    "record_ttft_seconds",
    "record_decode_tps",
    "record_queue_wait_seconds",
]
