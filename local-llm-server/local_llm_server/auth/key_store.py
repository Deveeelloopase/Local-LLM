from __future__ import annotations

"""
local_llm_server/auth/key_store.py — Multi‑Database API Key Management v1.0.0
===============================================================================
Enterprise‑grade API key store with pluggable backends: SQLite, PostgreSQL, MySQL.

"""

import asyncio
import hashlib
import json
import logging
import random
import re
import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Optional Prometheus client
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Module-level singleton — prometheus_client registers metrics in a global
# CollectorRegistry for the lifetime of the process.  Re-registering the same
# metric name (e.g. after uvicorn --reload reimports this module, or when
# create_key_store() is called more than once) raises ValueError and crashes
# the server.  We keep one dict and reuse it on every subsequent call.
_PROM_METRICS: Dict[str, Any] = {}


def _setup_prometheus_metrics() -> Dict[str, Any]:
    global _PROM_METRICS
    if not HAS_PROMETHEUS:
        return {}
    if _PROM_METRICS:
        # Already registered in this process — return the existing objects.
        return _PROM_METRICS
    _PROM_METRICS = {
        "validate_total": Counter("key_store_validate_total", "Total validations"),
        "validate_ok":    Counter("key_store_validate_ok",    "Successful validations"),
        "validate_fail":  Counter("key_store_validate_fail",  "Failed validations"),
        "rate_limited":   Counter("key_store_rate_limited",   "Rate‑limited requests"),
        "create_total":   Counter("key_store_create_total",   "Keys created"),
        "revoke_total":   Counter("key_store_revoke_total",   "Keys revoked"),
        "db_errors":      Counter("key_store_db_errors",      "Database errors"),
        "db_latency_ms":  Histogram("key_store_db_latency_ms", "Query latency (ms)",
                                    buckets=[1, 5, 10, 25, 50, 100, 250, 500]),
    }
    return _PROM_METRICS


# ------------------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------------------
class KeyStatus(str, Enum):
    ACTIVE  = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    PENDING = "pending"

# ------------------------------------------------------------------------------
# Key Exist
# ------------------------------------------------------------------------------
class KeyAlreadyExistsError(Exception):
    """Raised by create_key() when key_id is already taken. Maps to HTTP 409
    in the admin endpoint — was previously an unhandled DB uniqueness-
    constraint exception surfacing as a generic 500 with no indication of
    what actually went wrong (confirmed in production: retrying
    `llm keys create --id cli` after it had already succeeded once)."""
    def __init__(self, key_id: str):
        self.key_id = key_id
        super().__init__(f"API key with id '{key_id}' already exists")

# ------------------------------------------------------------------------------
# Rate Limiter Token Bucket (Thread‑Safe, burst‑capable)
# ------------------------------------------------------------------------------
class TokenBucket:
    """Per-key token bucket for rate limiting. Thread-safe.

    ``allow()`` consumes one token if available. ``stats()`` reports current
    state for observability endpoints.

    NOTE the original class also tracked ``concurrency_limit``
    and ``active_requests``, incrementing on ``allow()`` and decrementing
    on a separate ``release()`` method. But **nothing ever called
    release()**: the bucket returned by ``validate_key()`` is discarded by
    the caller (``verify_api_key``), and no endpoint holds a reference.
    That made ``active_requests`` a monotonically increasing counter:
    a bucket created with ``concurrency_limit > 0`` would accept the
    first N requests, then permanently refuse every subsequent one
    (``if active_requests >= concurrency_limit: return False``).
    Confirmed dead today — ``api.api_key_concurrency_limit: 0`` in
    config and no ``CreateKeyRequest`` field sets it — but a real latent
    bug the moment someone wires it up via the DB admin path.

    Removed rather than patched: a correct per-key concurrency limiter
    needs a proper async semaphore released in a ``try/finally`` around
    every request, not a token counter. If you need that, add a
    ``_ConcurrencyGate`` keyed on key_id with an async context manager,
    and wire it in ``verify_api_key``'s success path — a change large
    enough to deserve its own review, not a silent tweak here.

    The 3rd ``concurrency_limit`` arg is accepted-and-ignored so existing
    callers (``SQLKeyStore._get_bucket``) need not change.
    """

    __slots__ = ("rps", "capacity", "tokens", "last", "_lock")

    def __init__(self, rps: float, capacity: float,
                 concurrency_limit: int = 0) -> None:
        self.rps     = rps
        self.capacity = capacity
        self.tokens  = capacity
        self.last    = time.monotonic()
        self._lock   = threading.Lock()
        # concurrency_limit is intentionally ignored — see docstring.
        # Accepting it here keeps SQLKeyStore._get_bucket unchanged.

    def allow(self) -> bool:
        """Consume one token if available. Returns True if allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + self.rps * elapsed)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "tokens":   round(self.tokens, 2),
                "capacity": self.capacity,
                "rps":      self.rps,
                # Kept for backward compat — any client that reads
                # active_requests on /admin/keys/stats/{id} sees 0
                # (accurate: no concurrency accounting is performed).
                "active_requests": 0,
            }


# ------------------------------------------------------------------------------
# Abstract Base Key Store
# ------------------------------------------------------------------------------
class BaseKeyStore(ABC):
    @abstractmethod
    async def create_key(
        self,
        key_id: str,
        rate_limit_rps: float = 10.0,
        rate_limit_capacity: float = 20.0,
        description: str = "",
        expires_in_seconds: Optional[float] = None,
        active: bool = True,
        concurrency_limit: int = 0,
        ip_whitelist: Optional[List[str]] = None,
        # Per-key quotas. All 0 = no per-key cap.
        max_output_tokens: int = 0,
        max_request_seconds: float = 0.0,
        max_requests_per_day: int = 0,
    ) -> Tuple[str, str]:
        pass

    @abstractmethod
    async def validate_key(
        self, key: str, client_ip: str = "", tokens: int = 0
    ) -> Tuple[bool, Optional[str], Optional[TokenBucket]]:
        pass

    @abstractmethod
    async def log_request(
        self,
        key_id: Optional[str],
        path: str,
        method: str,
        status: int,
        latency_ms: float,
        tokens: int = 0,
        client_ip: str = "",
        request_id: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    async def revoke_key(self, key_id: str) -> bool:
        pass

    @abstractmethod
    async def activate_key(self, key_id: str) -> bool:
        pass

    @abstractmethod
    async def update_key_limits(self, key_id: str, rps: float, capacity: float) -> bool:
        pass

    @abstractmethod
    async def update_key_description(self, key_id: str, description: str) -> bool:
        pass

    @abstractmethod
    async def list_keys(self, active_only: bool = True) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_key_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_key(self, key_id: str, soft: bool = False) -> bool:
        pass

    @abstractmethod
    async def prune_expired(self) -> int:
        pass

    @abstractmethod
    async def start_background_tasks(self) -> None:
        pass

    @abstractmethod
    async def stop_background_tasks(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    @abstractmethod
    def metrics(self) -> Dict[str, int]:
        pass


# ------------------------------------------------------------------------------
# Base SQL Key Store with Common Logic (v1.0.0)
# ------------------------------------------------------------------------------
class SQLKeyStore(BaseKeyStore):
    """Base class for SQL backends with connection pooling, retry, and new features."""

    def __init__(
        self,
        enable_logging: bool = True,
        max_cache_size: int = 1000,
        expected_concurrency: int = 100,
        batch_usage_updates: bool = False,
        usage_batch_interval: float = 1.0,
        usage_batch_size: int = 100,
        prune_interval_hours: int = 1,
        request_log_retention_days: int = 30,
        pool_size: int = 10,
        max_overflow: int = 5,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        retry_attempts: int = 3,
        retry_backoff_base: float = 0.1,
        retry_jitter: bool = True,
        sqlite_cache_mb: int = 64,
        soft_delete_retention_days: int = 90,
    ) -> None:
        self._enable_logging = enable_logging
        self._max_cache = max_cache_size
        self._expected_concurrency = expected_concurrency
        self._batch_usage_updates = batch_usage_updates
        self._usage_batch_interval = usage_batch_interval
        self._usage_batch_size = usage_batch_size
        self._prune_interval = prune_interval_hours * 3600
        self._request_log_retention_days = request_log_retention_days
        self._soft_delete_retention_days = soft_delete_retention_days

        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._retry_attempts = retry_attempts
        self._retry_backoff_base = retry_backoff_base
        self._retry_jitter = retry_jitter
        self._sqlite_cache_mb = sqlite_cache_mb

        self._pool = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Bucket cache: key_id -> TokenBucket (LRU)
        self._bucket_cache: OrderedDict[str, TokenBucket] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Background tasks
        self._prune_task: Optional[asyncio.Task] = None
        self._usage_worker_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Bounded queue for usage updates
        self._usage_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._pending_usage: Dict[str, Tuple[int, str, float]] = {}
        self._last_batch_flush = time.monotonic()

        # Prometheus metrics
        self._prom = _setup_prometheus_metrics()

        # Store metrics
        self._metrics = {
            "validate_total": 0,
            "validate_success": 0,
            "validate_failures": 0,
            "rate_limited": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "db_errors": 0,
            "usage_updates_queued": 0,
            "usage_updates_dropped": 0,
        }
        self._metrics_lock = threading.Lock()

    # --------------------------------------------------------------------------
    # Database Initialization (implemented by subclasses)
    # --------------------------------------------------------------------------
    @abstractmethod
    async def _init_pool(self) -> None:
        pass

    @abstractmethod
    async def _create_tables(self) -> None:
        pass

    @abstractmethod
    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        pass

    @abstractmethod
    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        pass

    @abstractmethod
    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        pass

    # --------------------------------------------------------------------------
    # Key Generation & Hashing
    # --------------------------------------------------------------------------
    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_key(prefix: str = "") -> str:
        raw = secrets.token_urlsafe(32)
        return f"{prefix}_{raw}" if prefix else raw

    def _backoff(self, attempt: int) -> float:
        sleep = self._retry_backoff_base * (2 ** attempt)
        if self._retry_jitter:
            sleep *= random.uniform(0.8, 1.2)
        return min(sleep, 5.0)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    async def create_key(
        self,
        key_id: str,
        rate_limit_rps: float = 10.0,
        rate_limit_capacity: float = 20.0,
        description: str = "",
        expires_in_seconds: Optional[float] = None,
        active: bool = True,
        concurrency_limit: int = 0,
        ip_whitelist: Optional[List[str]] = None,
        max_output_tokens: int = 0,
        max_request_seconds: float = 0.0,
        max_requests_per_day: int = 0,
    ) -> Tuple[str, str]:
        if not key_id.strip():
            raise ValueError("key_id must be non-empty")

        # Fail with a clear, expected error instead of letting a raw DB
        # uniqueness-constraint exception surface as a generic 500.
        existing = await self._fetchone(
            "SELECT id FROM api_keys WHERE id = ?", (key_id,)
        )
        if existing:
            raise KeyAlreadyExistsError(key_id)

        plain_key = self._generate_key(prefix=key_id)
        key_hash = self._hash_key(plain_key)
        created = time.time()
        expires = created + expires_in_seconds if expires_in_seconds else None
        ip_whitelist_json = json.dumps(ip_whitelist) if ip_whitelist else None

        is_active_int = 1 if active else 0
        key_status = KeyStatus.ACTIVE.value if active else KeyStatus.PENDING.value

        try:
            await self._execute(
                """
                INSERT INTO api_keys
                (id, key_hash, created_at, expires_at, rate_limit_rps, rate_limit_capacity,
                 description, is_active, key_status, concurrency_limit, ip_whitelist,
                 max_output_tokens, max_request_seconds, max_requests_per_day,
                 daily_requests_date, daily_requests_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    key_hash,
                    created,
                    expires,
                    rate_limit_rps,
                    rate_limit_capacity,
                    description,
                    is_active_int,
                    key_status,
                    concurrency_limit,
                    ip_whitelist_json,
                    int(max_output_tokens or 0),
                    float(max_request_seconds or 0.0),
                    int(max_requests_per_day or 0),
                    None,   # daily_requests_date — reset on first request
                    0,      # daily_requests_count
                ),
                commit=True,
            )
        except Exception as e:
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg or "pkey" in msg:
                raise KeyAlreadyExistsError(key_id) from e
            raise
        if self._prom.get("create_total"):
            self._prom["create_total"].inc()
        log.info(
            "Created API key '%s' (hash=%s..., max_out=%d, max_secs=%.1f, max_per_day=%d)",
            key_id, key_hash[:8], max_output_tokens, max_request_seconds, max_requests_per_day,
        )
        return plain_key, key_hash

    async def validate_key(
        self, key: str, client_ip: str = "", tokens: int = 0
    ) -> Tuple[bool, Optional[str], Optional[TokenBucket]]:
        with self._metrics_lock:
            self._metrics["validate_total"] += 1
        if self._prom.get("validate_total"):
            self._prom["validate_total"].inc()

        key_hash = self._hash_key(key)
        t0 = time.monotonic()
        try:
            row = await self._fetchone(
                """
                SELECT id, is_active, expires_at, rate_limit_rps, rate_limit_capacity,
                       concurrency_limit, ip_whitelist,
                       max_output_tokens, max_request_seconds, max_requests_per_day,
                       daily_requests_date, daily_requests_count
                FROM api_keys WHERE key_hash = ?
                """,
                (key_hash,),
            )
        except Exception as e:
            with self._metrics_lock:
                self._metrics["db_errors"] += 1
                self._metrics["validate_failures"] += 1
            if self._prom.get("db_errors"):
                self._prom["db_errors"].inc()
            if self._prom.get("validate_fail"):
                self._prom["validate_fail"].inc()
            log.error("DB error in validate_key: %s", e)
            return False, None, None
        finally:
            latency = (time.monotonic() - t0) * 1000.0
            if self._prom.get("db_latency_ms"):
                self._prom["db_latency_ms"].observe(latency)

        if not row:
            with self._metrics_lock:
                self._metrics["validate_failures"] += 1
            if self._prom.get("validate_fail"):
                self._prom["validate_fail"].inc()
            log.debug("Key not found")
            return False, None, None

        # is_active is INTEGER (0/1)
        (key_id, is_active_int, expires_at, rps, capacity, concurrency_limit,
         ip_whitelist_json,
         max_output_tokens, max_request_seconds, max_requests_per_day,
         daily_date, daily_count) = row
        if not is_active_int:
            with self._metrics_lock:
                self._metrics["validate_failures"] += 1
            if self._prom.get("validate_fail"):
                self._prom["validate_fail"].inc()
            log.debug("Key '%s' inactive", key_id)
            return False, key_id, None

        if expires_at and expires_at < time.time():
            with self._metrics_lock:
                self._metrics["validate_failures"] += 1
            if self._prom.get("validate_fail"):
                self._prom["validate_fail"].inc()
            log.debug("Key '%s' expired", key_id)
            return False, key_id, None

        # IP whitelist check
        if ip_whitelist_json:
            try:
                allowed_ips = json.loads(ip_whitelist_json)
                if client_ip and allowed_ips and client_ip not in allowed_ips:
                    with self._metrics_lock:
                        self._metrics["validate_failures"] += 1
                    log.debug("Key '%s' IP not in whitelist", key_id)
                    return False, key_id, None
            except Exception:
                pass

        # ── Daily request quota ──────────────────────────────────
        # Enforced HERE (at auth) so a key over its daily cap cannot even
        # start a request. The counter is rolling-calendar-day in server-
        # local time: the first request after midnight resets the count.
        #
        # Trade-off: the SELECT-then-UPDATE sequence is not atomic across
        # concurrent requests. Two simultaneous requests could both pass
        # the check, pushing the count 1-2 over the cap on the boundary
        # request. That is acceptable for a quota (not a security
        # boundary); the counter is monotonic within a day, so subsequent
        # requests see the exceeded count and are rejected.
        if max_requests_per_day and max_requests_per_day > 0:
            today = time.strftime("%Y-%m-%d")
            already_hit_cap = (
                daily_date == today
                and int(daily_count or 0) >= int(max_requests_per_day)
            )
            if already_hit_cap:
                with self._metrics_lock:
                    self._metrics["validate_failures"] += 1
                log.warning(
                    "Daily quota exceeded for key '%s' (%d/%d on %s)",
                    key_id, int(daily_count or 0), int(max_requests_per_day), today,
                )
                return False, key_id, None
            try:
                await self._execute(
                    """
                    UPDATE api_keys
                    SET daily_requests_date = ?,
                        daily_requests_count = CASE
                            WHEN daily_requests_date = ? THEN daily_requests_count + 1
                            ELSE 1
                        END
                    WHERE id = ?
                    """,
                    (today, today, key_id),
                    commit=True,
                )
            except Exception as exc:
                # Do NOT fail the request on quota-counter write failure.
                # The request is already authenticated; the counter is
                # advisory and self-heals on next restart. Log and move on.
                log.warning(
                    "Daily quota increment failed for key '%s': %s",
                    key_id, exc,
                )

        bucket = self._get_bucket(key_id, rps, capacity, concurrency_limit)
        if not bucket.allow():
            with self._metrics_lock:
                self._metrics["rate_limited"] += 1
            if self._prom.get("rate_limited"):
                self._prom["rate_limited"].inc()
            log.warning("Rate limit exceeded for key '%s' (rps=%.1f)", key_id, rps)
            return False, key_id, None

        with self._metrics_lock:
            self._metrics["validate_success"] += 1
        if self._prom.get("validate_ok"):
            self._prom["validate_ok"].inc()

        await self._enqueue_usage_update(key_id, tokens, client_ip)
        return True, key_id, bucket

    async def record_usage(self, key_id: str, tokens: int, client_ip: str = "") -> None:
        """Accumulate post-hoc token usage for a key.

        ``validate_key()`` runs at the START of each request and always
        passes ``tokens=0`` because the actual generation count is not yet
        known. Without this method, ``api_keys.total_tokens`` stays at 0
        forever and ``/admin/keys/stats/{id}`` always reports
        ``total_tokens: 0`` regardless of how many tokens the key has
        actually generated.

        Safe no-op on backpressure: the underlying queue is bounded
        (``api.usage_batch_size``, default 2000) and drops rather than
        blocking, so this cannot stall the request path. Failures are
        swallowed by the caller's ``with suppress(Exception)`` wrapper.
        """
        await self._enqueue_usage_update(key_id, int(tokens), client_ip)

    async def get_key_quotas(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Return the per-request quota for a key 

        Called by ``verify_api_key`` after a successful auth to stash the
        caps on ``request.state.key_quotas``, where the chat handlers read
        them to clamp ``max_tokens`` / ``timeout_s``.

        A missing key returns None (which the caller treats as "no caps" —
        safe default; the key was validated microseconds earlier, so a
        missing row is almost certainly a concurrent delete).
        """
        row = await self._fetchone(
            """
            SELECT max_output_tokens, max_request_seconds, max_requests_per_day,
                   daily_requests_date, daily_requests_count
            FROM api_keys WHERE id = ?
            """,
            (key_id,),
        )
        if not row:
            return None
        return {
            "max_output_tokens": int(row[0] or 0),
            "max_request_seconds": float(row[1] or 0.0),
            "max_requests_per_day": int(row[2] or 0),
            "daily_requests_date": row[3],
            "daily_requests_count": int(row[4] or 0),
        }

    async def update_key_quotas(
        self,
        key_id: str,
        max_output_tokens: Optional[int] = None,
        max_request_seconds: Optional[float] = None,
        max_requests_per_day: Optional[int] = None,
    ) -> bool:
        """Update per-key quotas. ``None`` leaves the field unchanged.

        Setting a field to 0 clears the cap (unlimited for that dimension).
        """
        fields: List[str] = []
        params: List[Any] = []
        if max_output_tokens is not None:
            fields.append("max_output_tokens = ?")
            params.append(int(max_output_tokens))
        if max_request_seconds is not None:
            fields.append("max_request_seconds = ?")
            params.append(float(max_request_seconds))
        if max_requests_per_day is not None:
            fields.append("max_requests_per_day = ?")
            params.append(int(max_requests_per_day))
        if not fields:
            return True
        params.append(key_id)
        try:
            await self._execute(
                f"UPDATE api_keys SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
                commit=True,
            )
            log.info(
                "Updated quotas for key '%s' (out=%s secs=%s per_day=%s)",
                key_id, max_output_tokens, max_request_seconds, max_requests_per_day,
            )
            return True
        except Exception as e:
            log.error("Failed to update quotas for key '%s': %s", key_id, e)
            return False

    async def _enqueue_usage_update(self, key_id: str, tokens: int, client_ip: str) -> None:
        try:
            self._usage_queue.put_nowait((key_id, tokens, client_ip, time.time()))
            with self._metrics_lock:
                self._metrics["usage_updates_queued"] += 1
        except asyncio.QueueFull:
            with self._metrics_lock:
                self._metrics["usage_updates_dropped"] += 1
            log.warning("Usage update queue full; dropping update for key '%s'", key_id)

    async def _record_usage(self, key_id: str, tokens: int, client_ip: str) -> None:
        try:
            await self._execute(
                """
                UPDATE api_keys
                SET last_used = ?, total_requests = total_requests + 1,
                    total_tokens = total_tokens + ?, last_ip = ?
                WHERE id = ?
                """,
                (time.time(), tokens, client_ip, key_id),
                commit=True,
            )
        except Exception as e:
            log.debug("Failed to update usage for key '%s': %s", key_id, e)

    async def log_request(
        self,
        key_id: Optional[str],
        path: str,
        method: str,
        status: int,
        latency_ms: float,
        tokens: int = 0,
        client_ip: str = "",
        request_id: Optional[str] = None,
    ) -> None:
        if not self._enable_logging:
            return
        try:
            await self._execute(
                """
                INSERT INTO api_requests
                (key_id, timestamp, path, method, status, latency_ms, tokens, client_ip, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    time.time(),
                    path[:512],
                    method,
                    status,
                    latency_ms,
                    tokens,
                    client_ip,
                    request_id,
                ),
                commit=True,
            )
        except Exception as e:
            log.debug("Failed to log request [%s]: %s", request_id, e)

    async def revoke_key(self, key_id: str) -> bool:
        try:
            await self._execute(
                "UPDATE api_keys SET is_active = 0, key_status = ? WHERE id = ?",
                (KeyStatus.REVOKED.value, key_id),
                commit=True,
            )
            with self._cache_lock:
                self._bucket_cache.pop(key_id, None)
            if self._prom.get("revoke_total"):
                self._prom["revoke_total"].inc()
            log.info("Revoked key '%s'", key_id)
            return True
        except Exception as e:
            log.error("Failed to revoke key '%s': %s", key_id, e)
            return False

    async def activate_key(self, key_id: str) -> bool:
        try:
            await self._execute(
                "UPDATE api_keys SET is_active = 1, key_status = ? WHERE id = ?",
                (KeyStatus.ACTIVE.value, key_id),
                commit=True,
            )
            log.info("Activated key '%s'", key_id)
            return True
        except Exception as e:
            log.error("Failed to activate key '%s': %s", key_id, e)
            return False

    async def update_key_limits(self, key_id: str, rps: float, capacity: float) -> bool:
        try:
            await self._execute(
                "UPDATE api_keys SET rate_limit_rps = ?, rate_limit_capacity = ? WHERE id = ?",
                (rps, capacity, key_id),
                commit=True,
            )
            with self._cache_lock:
                self._bucket_cache.pop(key_id, None)
            log.info("Updated rate limits for '%s': rps=%.1f, cap=%.1f", key_id, rps, capacity)
            return True
        except Exception as e:
            log.error("Failed to update key '%s': %s", key_id, e)
            return False

    async def update_key_description(self, key_id: str, description: str) -> bool:
        try:
            await self._execute(
                "UPDATE api_keys SET description = ? WHERE id = ?",
                (description, key_id),
                commit=True,
            )
            log.info("Updated description for '%s'", key_id)
            return True
        except Exception as e:
            log.error("Failed to update description for '%s': %s", key_id, e)
            return False

    async def list_keys(self, active_only: bool = True) -> List[Dict[str, Any]]:
        query = """
            SELECT id, created_at, expires_at, rate_limit_rps, rate_limit_capacity,
                   description, is_active, key_status, last_used, total_requests, total_tokens, last_ip,
                   concurrency_limit, ip_whitelist,
                   max_output_tokens, max_request_seconds, max_requests_per_day,
                   daily_requests_date, daily_requests_count
            FROM api_keys
        """
        if active_only:
            query += " WHERE is_active = 1"
        rows = await self._fetchall(query)
        keys = []
        for row in rows:
            ip_whitelist = None
            if row[12] is not None:
                try:
                    ip_whitelist = json.loads(row[12])
                except Exception:
                    pass
            keys.append({
                "id": row[0],
                "created_at": row[1],
                "expires_at": row[2],
                "rate_limit_rps": row[3],
                "rate_limit_capacity": row[4],
                "description": row[5],
                "is_active": bool(row[6]),
                "key_status": row[7],
                "last_used": row[8],
                "total_requests": row[9],
                "total_tokens": row[10],
                "last_ip": row[11],
                "concurrency_limit": row[13] if row[13] else 0,
                "ip_whitelist": ip_whitelist,
                "max_output_tokens":   int(row[14] or 0),
                "max_request_seconds": float(row[15] or 0.0),
                "max_requests_per_day": int(row[16] or 0),
                "daily_requests_date": row[17],
                "daily_requests_count": int(row[18] or 0),
            })
        return keys

    async def get_key_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetchone(
            """
            SELECT created_at, expires_at, rate_limit_rps, rate_limit_capacity,
                   description, is_active, key_status, last_used, total_requests, total_tokens, last_ip,
                   concurrency_limit, ip_whitelist,
                   max_output_tokens, max_request_seconds, max_requests_per_day,
                   daily_requests_date, daily_requests_count
            FROM api_keys WHERE id = ?
            """,
            (key_id,),
        )
        if not row:
            return None
        ip_whitelist = None
        if row[11] is not None:
            try:
                ip_whitelist = json.loads(row[11])
            except Exception:
                pass
        return {
            "created_at": row[0],
            "expires_at": row[1],
            "rate_limit_rps": row[2],
            "rate_limit_capacity": row[3],
            "description": row[4],
            "is_active": bool(row[5]),
            "key_status": row[6],
            "last_used": row[7],
            "total_requests": row[8],
            "total_tokens": row[9],
            "last_ip": row[10],
            "concurrency_limit": row[12] if row[12] else 0,
            "ip_whitelist": ip_whitelist,
            # Per-key quotas
            "max_output_tokens":   int(row[13] or 0),
            "max_request_seconds": float(row[14] or 0.0),
            "max_requests_per_day": int(row[15] or 0),
            "daily_requests_date": row[16],
            "daily_requests_count": int(row[17] or 0),
        }

    async def delete_key(self, key_id: str, soft: bool = False) -> bool:
        try:
            if soft:
                await self._execute(
                    "UPDATE api_keys SET is_active = 0, key_status = ?, deleted_at = ? WHERE id = ?",
                    (KeyStatus.REVOKED.value, time.time(), key_id),
                    commit=True,
                )
            else:
                await self._execute("DELETE FROM api_keys WHERE id = ?", (key_id,), commit=True)
            with self._cache_lock:
                self._bucket_cache.pop(key_id, None)
            log.info("Deleted key '%s' (soft=%s)", key_id, soft)
            return True
        except Exception as e:
            log.error("Failed to delete key '%s': %s", key_id, e)
            return False

    async def prune_expired(self) -> int:
        now = time.time()
        # Mark expired keys as inactive
        await self._execute(
            "UPDATE api_keys SET is_active = 0, key_status = ? "
            "WHERE expires_at IS NOT NULL AND expires_at < ? AND is_active = 1",
            (KeyStatus.EXPIRED.value, now),
            commit=True,
        )
        # Delete soft‑deleted keys beyond retention
        if self._soft_delete_retention_days > 0:
            cutoff = now - self._soft_delete_retention_days * 86400
            await self._execute(
                "DELETE FROM api_keys WHERE is_active = 0 AND key_status = ? "
                "AND deleted_at IS NOT NULL AND deleted_at < ?",
                (KeyStatus.REVOKED.value, cutoff),
                commit=True,
            )
        rows = await self._fetchall(
            "SELECT id FROM api_keys WHERE key_status = ?", (KeyStatus.EXPIRED.value,)
        )
        affected = len(rows)
        if affected:
            log.info("Pruned %d expired keys", affected)
        return affected

    # --------------------------------------------------------------------------
    # Rate Limiter Cache
    # --------------------------------------------------------------------------
    def _get_bucket(self, key_id: str, rps: float, capacity: float, concurrency_limit: int = 0) -> TokenBucket:
        with self._cache_lock:
            if key_id in self._bucket_cache:
                existing = self._bucket_cache[key_id]
                if existing.rps != rps or existing.capacity != capacity:
                    log.debug("Rate limits changed for key '%s' – recreating bucket", key_id)
                    new_bucket = TokenBucket(rps, capacity, concurrency_limit)
                    self._bucket_cache[key_id] = new_bucket
                    self._bucket_cache.move_to_end(key_id)
                    with self._metrics_lock:
                        self._metrics["cache_misses"] += 1
                    return new_bucket
                self._bucket_cache.move_to_end(key_id)
                with self._metrics_lock:
                    self._metrics["cache_hits"] += 1
                return existing
            bucket = TokenBucket(rps, capacity, concurrency_limit)
            self._bucket_cache[key_id] = bucket
            if len(self._bucket_cache) > self._max_cache:
                self._bucket_cache.popitem(last=False)
            with self._metrics_lock:
                self._metrics["cache_misses"] += 1
            return bucket

    def metrics(self) -> Dict[str, int]:
        with self._metrics_lock:
            return self._metrics.copy()

    # --------------------------------------------------------------------------
    # Background Tasks
    # --------------------------------------------------------------------------
    async def start_background_tasks(self) -> None:
        if self._prune_task is None or self._prune_task.done():
            self._prune_task = asyncio.create_task(self._prune_loop(), name="key_prune")
        if self._usage_worker_task is None or self._usage_worker_task.done():
            self._usage_worker_task = asyncio.create_task(self._usage_worker(), name="usage_worker")
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop(), name="key_health")
        log.debug("KeyStore background tasks started")

    async def stop_background_tasks(self) -> None:
        tasks = []
        if self._prune_task and not self._prune_task.done():
            tasks.append(self._prune_task)
            self._prune_task.cancel()
        if self._usage_worker_task and not self._usage_worker_task.done():
            self._usage_queue.put_nowait(None)
            tasks.append(self._usage_worker_task)
            self._usage_worker_task.cancel()
        if self._health_check_task and not self._health_check_task.done():
            tasks.append(self._health_check_task)
            self._health_check_task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._flush_pending_usage()
        log.debug("KeyStore background tasks stopped")

    async def close(self) -> None:
        await self.stop_background_tasks()
        if self._pool:
            if hasattr(self._pool, 'close'):
                await self._pool.close()
            self._pool = None
            log.info("KeyStore database connection closed")

    async def health_check(self) -> bool:
        try:
            await self._fetchone("SELECT 1", ())
            return True
        except Exception:
            return False

    async def _prune_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._prune_interval)
                await self.prune_expired()
                if self._enable_logging:
                    await self._prune_logs_batched()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Prune loop error: %s", e)
                await asyncio.sleep(5)

    async def _prune_logs_batched(self, batch_size: int = 1000) -> None:
        cutoff = time.time() - (self._request_log_retention_days * 86400)
        while True:
            try:
                result = await self._execute(
                    "DELETE FROM api_requests WHERE timestamp < ? LIMIT ?",
                    (cutoff, batch_size),
                    commit=True,
                )
                break
            except Exception as e:
                log.warning("Log prune error: %s", e)
                break

    async def _usage_worker(self) -> None:
        while True:
            try:
                try:
                    item = await asyncio.wait_for(
                        self._usage_queue.get(), timeout=self._usage_batch_interval
                    )
                except asyncio.TimeoutError:
                    if self._batch_usage_updates:
                        await self._flush_pending_usage()
                    continue
                if item is None:
                    break
                key_id, tokens, client_ip, ts = item
                if self._batch_usage_updates:
                    prev = self._pending_usage.get(key_id, (0, "", 0))
                    self._pending_usage[key_id] = (
                        tokens + prev[0],
                        client_ip,
                        max(ts, prev[2]),
                    )
                    if len(self._pending_usage) >= self._usage_batch_size:
                        await self._flush_pending_usage()
                else:
                    await self._record_usage(key_id, tokens, client_ip)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Usage worker error: %s", e)
        await self._flush_pending_usage()

    async def _flush_pending_usage(self) -> None:
        if not self._pending_usage:
            return
        items = list(self._pending_usage.items())
        self._pending_usage.clear()
        for key_id, (tokens, client_ip, _) in items:
            try:
                await self._record_usage(key_id, tokens, client_ip)
            except Exception as e:
                log.error("Failed to flush usage for key '%s': %s", key_id, e)

    async def _health_check_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                ok = await self.health_check()
                if not ok:
                    log.warning("KeyStore health check failed – attempting reconnect...")
            except asyncio.CancelledError:
                break
            except Exception:
                pass


# ------------------------------------------------------------------------------
# SQLite Implementation
# ------------------------------------------------------------------------------
class SQLiteKeyStore(SQLKeyStore):
    """SQLite backend using aiosqlite."""

    def __init__(self, db_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._db_conn = None
        self._db_write_lock = asyncio.Lock()

    async def _init_pool(self) -> None:
        try:
            import aiosqlite
        except ImportError:
            raise ImportError("aiosqlite not installed. Run: pip install aiosqlite")
        self._db_conn = await aiosqlite.connect(self._db_path)
        await self._db_conn.execute("PRAGMA journal_mode=WAL")
        await self._db_conn.execute("PRAGMA synchronous=NORMAL")
        await self._db_conn.execute(f"PRAGMA cache_size=-{self._sqlite_cache_mb * 1024}")
        await self._db_conn.execute("PRAGMA busy_timeout=5000")
        await self._create_tables()
        await self._migrate_schema()

        self._initialized = True

    async def _create_tables(self) -> None:
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                expires_at REAL,
                rate_limit_rps REAL NOT NULL DEFAULT 10.0,
                rate_limit_capacity REAL NOT NULL DEFAULT 20.0,
                description TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                key_status TEXT NOT NULL DEFAULT 'active',
                last_used REAL,
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                last_ip TEXT,
                concurrency_limit INTEGER NOT NULL DEFAULT 0,
                ip_whitelist TEXT,
                -- Per-key quotas. All 0 = no per-key cap (the
                -- server-wide inference.max_total_tokens still applies).
                max_output_tokens INTEGER NOT NULL DEFAULT 0,
                max_request_seconds REAL NOT NULL DEFAULT 0.0,
                max_requests_per_day INTEGER NOT NULL DEFAULT 0,
                -- Rolling daily counter. daily_requests_date is "YYYY-MM-DD"
                -- in server-local time; on the first request of a new day
                -- the count resets to 1 (see validate_key).
                daily_requests_date TEXT,
                daily_requests_count INTEGER NOT NULL DEFAULT 0,
                deleted_at REAL
            )
        """)
        if self._enable_logging:
            await self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS api_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT,
                    timestamp REAL NOT NULL,
                    path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    latency_ms REAL,
                    tokens INTEGER,
                    client_ip TEXT,
                    request_id TEXT,
                    FOREIGN KEY(key_id) REFERENCES api_keys(id) ON DELETE SET NULL
                )
            """)
            await self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_key ON api_requests(key_id)"
            )
            await self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_ts ON api_requests(timestamp)"
            )
            await self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_reqid ON api_requests(request_id)"
            )
        await self._db_conn.commit()

    async def _migrate_schema(self) -> None:
        """Add columns that may be missing from older DB versions.

        also migrates the 5 per-key-quota columns added in Stage 8
        — same bug as the Postgres version.
        """
        cursor = await self._db_conn.execute("PRAGMA table_info(api_keys)")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}

        alters = []
        if "key_status" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN key_status TEXT NOT NULL DEFAULT 'active'")
        if "deleted_at" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN deleted_at REAL")
        if "concurrency_limit" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN concurrency_limit INTEGER NOT NULL DEFAULT 0")
        if "ip_whitelist" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN ip_whitelist TEXT")
        # ── per-key quotas ────────────────────────────────────────
        if "max_output_tokens" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN max_output_tokens INTEGER NOT NULL DEFAULT 0")
        if "max_request_seconds" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN max_request_seconds REAL NOT NULL DEFAULT 0.0")
        if "max_requests_per_day" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN max_requests_per_day INTEGER NOT NULL DEFAULT 0")
        if "daily_requests_date" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN daily_requests_date TEXT")
        if "daily_requests_count" not in existing:
            alters.append("ALTER TABLE api_keys ADD COLUMN daily_requests_count INTEGER NOT NULL DEFAULT 0")

        for stmt in alters:
            await self._db_conn.execute(stmt)
        if alters:
            await self._db_conn.commit()

    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                if commit:
                    async with self._db_write_lock:
                        cursor = await self._db_conn.execute(query, params)
                        await self._db_conn.commit()
                        return cursor
                else:
                    cursor = await self._db_conn.execute(query, params)
                    return cursor
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                with self._metrics_lock:
                    self._metrics["db_errors"] += 1
                raise

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        cursor = await self._execute(query, params, commit=False)
        return await cursor.fetchone() if cursor else None

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        cursor = await self._execute(query, params, commit=False)
        return await cursor.fetchall() if cursor else []

    async def close(self) -> None:
        await super().close()
        if self._db_conn:
            await self._db_conn.close()
            self._db_conn = None


# ------------------------------------------------------------------------------
# PostgreSQL Implementation
# ------------------------------------------------------------------------------
class PostgreSQLKeyStore(SQLKeyStore):
    """PostgreSQL backend using asyncpg."""

    RETRY_SQLSTATES = frozenset({
        "40001", "40P01", "08006", "08001", "57P03",
    })

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password

    async def _init_pool(self) -> None:
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg not installed. Run: pip install asyncpg")
        self._pool = await asyncpg.create_pool(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._username,
            password=self._password,
            min_size=self._pool_size,
            max_size=self._pool_size + self._max_overflow,
            timeout=self._pool_timeout,
            max_inactive_connection_lifetime=self._pool_recycle,
        )
        await self._create_tables()
        await self._migrate_schema()

        self._initialized = True

    async def _create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id VARCHAR(255) PRIMARY KEY,
                    key_hash VARCHAR(64) NOT NULL UNIQUE,
                    created_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    rate_limit_rps DOUBLE PRECISION NOT NULL DEFAULT 10.0,
                    rate_limit_capacity DOUBLE PRECISION NOT NULL DEFAULT 20.0,
                    description TEXT,
                    is_active SMALLINT NOT NULL DEFAULT 1,
                    key_status VARCHAR(10) NOT NULL DEFAULT 'active',
                    last_used DOUBLE PRECISION,
                    total_requests INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    last_ip VARCHAR(45),
                    concurrency_limit INTEGER NOT NULL DEFAULT 0,
                    ip_whitelist JSONB,
                    max_output_tokens INTEGER NOT NULL DEFAULT 0,
                    max_request_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    max_requests_per_day INTEGER NOT NULL DEFAULT 0,
                    daily_requests_date VARCHAR(10),
                    daily_requests_count INTEGER NOT NULL DEFAULT 0,
                    deleted_at DOUBLE PRECISION
                )
            """)
            if self._enable_logging:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_requests (
                        id SERIAL PRIMARY KEY,
                        key_id VARCHAR(255),
                        timestamp DOUBLE PRECISION NOT NULL,
                        path TEXT NOT NULL,
                        method VARCHAR(10) NOT NULL,
                        status INTEGER NOT NULL,
                        latency_ms DOUBLE PRECISION,
                        tokens INTEGER,
                        client_ip VARCHAR(45),
                        request_id VARCHAR(36),
                        FOREIGN KEY(key_id) REFERENCES api_keys(id) ON DELETE SET NULL
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_key ON api_requests(key_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON api_requests(timestamp)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_reqid ON api_requests(request_id)")

    async def _migrate_schema(self) -> None:
        """Add columns if missing (safe for repeated runs).

        also migrates the 5 per-key-quota columns added in Stage 8
        — previously only `_create_tables` (fresh installs) got them, so
        upgrading an existing DB left every request 403'ing with
        'column "max_output_tokens" does not exist'.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_status VARCHAR(10) NOT NULL DEFAULT 'active'"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS deleted_at DOUBLE PRECISION"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS concurrency_limit INTEGER NOT NULL DEFAULT 0"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS ip_whitelist JSONB"
            )
            # ── per-key quotas ────────────────────────────────────
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS max_output_tokens INTEGER NOT NULL DEFAULT 0"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS max_request_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS max_requests_per_day INTEGER NOT NULL DEFAULT 0"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS daily_requests_date VARCHAR(10)"
            )
            await conn.execute(
                "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS daily_requests_count INTEGER NOT NULL DEFAULT 0"
            )

    def _convert_placeholders(self, query: str, params: tuple) -> str:
        count = 1
        def repl(match):
            nonlocal count
            res = f"${count}"
            count += 1
            return res
        return re.sub(r"\?", repl, query)

    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        pg_query = self._convert_placeholders(query, params)
        max_retries = self._retry_attempts
        last_exc = None
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    try:
                        if commit:
                            async with conn.transaction():
                                return await conn.execute(pg_query, *params)
                        else:
                            return await conn.execute(pg_query, *params)
                    except asyncio.CancelledError:
                        # Ensure transaction is rolled back before the connection is released
                        try:
                            await conn.rollback()
                        except Exception:
                            pass
                        raise
            except Exception as e:
                last_exc = e
                if hasattr(e, "sqlstate") and e.sqlstate in self.RETRY_SQLSTATES:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                with self._metrics_lock:
                    self._metrics["db_errors"] += 1
                raise
        raise last_exc

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        pg_query = self._convert_placeholders(query, params)
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(pg_query, *params)
                    return tuple(row) if row else None
            except Exception as e:
                if hasattr(e, "sqlstate") and e.sqlstate in self.RETRY_SQLSTATES:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                raise
        return None

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        pg_query = self._convert_placeholders(query, params)
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(pg_query, *params)
                    return [tuple(r) for r in rows]
            except Exception as e:
                if hasattr(e, "sqlstate") and e.sqlstate in self.RETRY_SQLSTATES:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                raise
        return []

    # --------------------------------------------------------------------------
    # 🔥 Override _prune_logs_batched for PostgreSQL
    # --------------------------------------------------------------------------
    async def _prune_logs_batched(self, batch_size: int = 1000) -> None:
        """
        PostgreSQL does not support LIMIT on a DELETE statement.
        This uses a subquery to delete rows in batches safely.
        """
        cutoff = time.time() - (self._request_log_retention_days * 86400)
        while True:
            try:
                # Use a subquery with LIMIT inside the WHERE clause
                await self._execute(
                    """
                    DELETE FROM api_requests
                    WHERE id IN (
                        SELECT id FROM api_requests
                        WHERE timestamp < ?
                        ORDER BY timestamp
                        LIMIT ?
                    )
                    """,
                    (cutoff, batch_size),
                    commit=True,
                )
                # Break after a single batch (like the base implementation)
                # The original code had a while True but broke immediately.
                # We keep the same behaviour: one batch per call.
                break
            except Exception as e:
                log.warning("Log prune error (PostgreSQL): %s", e)
                break


# ------------------------------------------------------------------------------
# MySQL Implementation
# ------------------------------------------------------------------------------
class MySQLKeyStore(SQLKeyStore):
    """MySQL backend using aiomysql."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password

    async def _init_pool(self) -> None:
        try:
            import aiomysql
        except ImportError:
            raise ImportError("aiomysql not installed. Run: pip install aiomysql")
        self._pool = await aiomysql.create_pool(
            host=self._host,
            port=self._port,
            db=self._database,
            user=self._username,
            password=self._password,
            minsize=self._pool_size,
            maxsize=self._pool_size + self._max_overflow,
            pool_recycle=self._pool_recycle,
            autocommit=True,
        )
        await self._create_tables()
        await self._migrate_schema()

        self._initialized = True

    async def _create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id VARCHAR(255) PRIMARY KEY,
                        key_hash VARCHAR(64) NOT NULL UNIQUE,
                        created_at DOUBLE PRECISION NOT NULL,
                        expires_at DOUBLE PRECISION,
                        rate_limit_rps DOUBLE PRECISION NOT NULL DEFAULT 10.0,
                        rate_limit_capacity DOUBLE PRECISION NOT NULL DEFAULT 20.0,
                        description TEXT,
                        is_active TINYINT NOT NULL DEFAULT 1,
                        key_status VARCHAR(10) NOT NULL DEFAULT 'active',
                        last_used DOUBLE PRECISION,
                        total_requests INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        last_ip VARCHAR(45),
                        concurrency_limit INTEGER NOT NULL DEFAULT 0,
                        ip_whitelist JSON,
                        deleted_at DOUBLE PRECISION
                    )
                """)
                if self._enable_logging:
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS api_requests (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            key_id VARCHAR(255),
                            timestamp DOUBLE PRECISION NOT NULL,
                            path TEXT NOT NULL,
                            method VARCHAR(10) NOT NULL,
                            status INTEGER NOT NULL,
                            latency_ms DOUBLE PRECISION,
                            tokens INTEGER,
                            client_ip VARCHAR(45),
                            request_id VARCHAR(36),
                            FOREIGN KEY(key_id) REFERENCES api_keys(id) ON DELETE SET NULL
                        )
                    """)
                    await cur.execute("CREATE INDEX idx_requests_key ON api_requests(key_id)")
                    await cur.execute("CREATE INDEX idx_requests_ts ON api_requests(timestamp)")
                    await cur.execute("CREATE INDEX idx_requests_reqid ON api_requests(request_id)")

    async def _migrate_schema(self) -> None:
        """Add columns if missing (MySQL compatible check)."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW COLUMNS FROM api_keys LIKE 'key_status'")
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE api_keys ADD COLUMN key_status VARCHAR(10) NOT NULL DEFAULT 'active'"
                    )
                await cur.execute("SHOW COLUMNS FROM api_keys LIKE 'deleted_at'")
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE api_keys ADD COLUMN deleted_at DOUBLE PRECISION"
                    )
                await cur.execute("SHOW COLUMNS FROM api_keys LIKE 'concurrency_limit'")
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE api_keys ADD COLUMN concurrency_limit INTEGER NOT NULL DEFAULT 0"
                    )
                await cur.execute("SHOW COLUMNS FROM api_keys LIKE 'ip_whitelist'")
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE api_keys ADD COLUMN ip_whitelist JSON"
                    )

    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        query = query.replace("?", "%s")
        max_retries = self._retry_attempts
        last_exc = None
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params)
                        if commit:
                            await conn.commit()
                        return cur
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if ("deadlock" in msg or "lock wait timeout" in msg or "connection" in msg):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                with self._metrics_lock:
                    self._metrics["db_errors"] += 1
                raise
        raise last_exc

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        query = query.replace("?", "%s")
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params)
                        row = await cur.fetchone()
                        return row
            except Exception as e:
                msg = str(e).lower()
                if ("deadlock" in msg or "lock wait timeout" in msg or "connection" in msg):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                raise
        return None

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        query = query.replace("?", "%s")
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params)
                        rows = await cur.fetchall()
                        return rows
            except Exception as e:
                msg = str(e).lower()
                if ("deadlock" in msg or "lock wait timeout" in msg or "connection" in msg):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                raise
        return []


# ------------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------------
async def create_key_store(cfg: Optional[Dict[str, Any]] = None) -> Optional[BaseKeyStore]:
    if cfg is None:
        cfg = _load_default_config()

    key_cfg = cfg.get("api", {})
    if not key_cfg.get("key_store_enabled", False):
        log.info("API key store disabled (fallback to static key).")
        return None

    storage = key_cfg.get("storage", "sqlite")
    if isinstance(storage, str):
        storage = storage.strip().strip('"').strip("'").lower()
    else:
        storage = "sqlite"

    log_requests = key_cfg.get("log_requests", True)
    cache_size = key_cfg.get("key_cache_size", 1000)
    batch_updates = key_cfg.get("batch_usage_updates", False)
    batch_interval = key_cfg.get("usage_batch_interval", 1.0)
    batch_size = key_cfg.get("usage_batch_size", 100)
    expected_concurrency = key_cfg.get("expected_concurrency", 100)
    prune_interval_hours = key_cfg.get("prune_interval_hours", 1)
    request_log_retention_days = key_cfg.get("request_log_retention_days", 30)
    pool_size = key_cfg.get("pool_size", 10)
    max_overflow = key_cfg.get("max_overflow", 5)
    pool_timeout = key_cfg.get("pool_timeout", 30)
    pool_recycle = key_cfg.get("pool_recycle", 3600)
    retry_attempts = key_cfg.get("retry_attempts", 3)
    retry_backoff_base = key_cfg.get("retry_backoff_base", 0.1)
    retry_jitter = key_cfg.get("retry_jitter", True)
    sqlite_cache_mb = key_cfg.get("sqlite_cache_mb", 64)
    soft_delete_retention_days = key_cfg.get("soft_delete_retention_days", 90)

    common_kwargs = {
        "enable_logging": log_requests,
        "max_cache_size": cache_size,
        "expected_concurrency": expected_concurrency,
        "batch_usage_updates": batch_updates,
        "usage_batch_interval": batch_interval,
        "usage_batch_size": batch_size,
        "prune_interval_hours": prune_interval_hours,
        "request_log_retention_days": request_log_retention_days,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "retry_attempts": retry_attempts,
        "retry_backoff_base": retry_backoff_base,
        "retry_jitter": retry_jitter,
        "sqlite_cache_mb": sqlite_cache_mb,
        "soft_delete_retention_days": soft_delete_retention_days,
    }

    store: Optional[BaseKeyStore] = None
    try:
        if storage == "sqlite":
            db_path = key_cfg.get("key_store_db", "/tmp/llm_keys.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            store = SQLiteKeyStore(db_path=db_path, **common_kwargs)
        elif storage == "postgresql":
            host = key_cfg.get("host", "localhost")
            port = key_cfg.get("port", 5432)
            database = key_cfg.get("database", "llm_keys")
            username = key_cfg.get("username", "postgres")
            password = key_cfg.get("password", "")
            store = PostgreSQLKeyStore(
                host=host, port=int(port), database=database,
                username=username, password=password, **common_kwargs,
            )
        elif storage == "mysql":
            host = key_cfg.get("host", "localhost")
            port = key_cfg.get("port", 3306)
            database = key_cfg.get("database", "llm_keys")
            username = key_cfg.get("username", "root")
            password = key_cfg.get("password", "")
            store = MySQLKeyStore(
                host=host, port=int(port), database=database,
                username=username, password=password, **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown key store storage type: {storage}")

        await store._init_pool()
        await store.start_background_tasks()
        log.info("KeyStore initialized: backend=%s", storage)
        return store
    except Exception as db_err:
        # Fail-fast by default. The previous behavior — silently falling
        # back to an in-memory SQLite store on ANY asyncpg / Postgres error
        # (including auth failure, missing DB, schema mismatch, timeout) —
        # was a production bug every existing API key becomes
        # invalid because the in-memory store starts empty, but /health
        # and /ready keep returning 200 because they check loader/engine,
        # not the key store. The only visible symptom was clients getting
        # 403 with no clear cause.
        #
        # Opt-in escape hatch: set api.key_store_allow_memory_fallback: true
        # in config.yaml to restore the old behavior (only useful for local
        # dev where losing keys across restarts is acceptable).
        allow_mem = bool(key_cfg.get("key_store_allow_memory_fallback", False))
        if not allow_mem:
            log.error(
                "KeyStore primary storage (%s) failed: %s. Refusing to start — "
                "set api.key_store_allow_memory_fallback: true to override.",
                storage, db_err,
            )
            raise
        log.warning("Primary storage failed (%s): %s", storage, db_err)
        log.warning(
            "⚠️  Falling back to FALLBACK IN-MEMORY KEYSTORE — "
            "all existing keys will be invalid until restart."
        )
        store = SQLiteKeyStore(db_path=":memory:")
        await store._init_pool()
        await store.start_background_tasks()
        return store


def _load_default_config() -> Dict[str, Any]:
    paths = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    log.warning("No config.yaml found; using empty configuration.")
    return {}


# Backward‑compatible alias
KeyStore = BaseKeyStore

__all__ = [
    "TokenBucket",
    "BaseKeyStore",
    "KeyStore",
    "SQLiteKeyStore",
    "PostgreSQLKeyStore",
    "MySQLKeyStore",
    "create_key_store",
]
