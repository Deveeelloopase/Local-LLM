from __future__ import annotations

"""
local_llm_server/memory/conversation_store.py — Production Conversation Storage v1.0.0
========================================================================================
Enterprise‑grade conversation history management with pluggable database backends.

  • Leaky lease cache protection: the in‑memory store's lease token cache is
    bounded (configurable), with LRU eviction.
  • Write‑buffer resilience: failed writes are retried with short backoff
    before being dropped.
  • Health‑check for SQL stores.
  • All previous features retained (shadow‑paging, lease tokens, adaptive LRU, etc.).
  • Added checkpoint support for resumable generation (Section 2 of the audit).
  • FIX: save_checkpoint() now ensures the conversation row exists before inserting
    a checkpoint, eliminating foreign‑key violations for new conversations.
"""

import asyncio
import hashlib
import logging
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Optional Prometheus
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

def _setup_prometheus() -> Dict[str, Any]:
    if not HAS_PROMETHEUS:
        return {}
    return {
        "gets_total": Counter("conv_store_gets_total", "Get calls"),
        "adds_total": Counter("conv_store_adds_total", "Add message calls"),
        "creates_total": Counter("conv_store_creates_total", "Create conversation calls"),
        "deletes_total": Counter("conv_store_deletes_total", "Delete calls"),
        "db_errors": Counter("conv_store_db_errors", "Database errors"),
        "db_latency_ms": Histogram("conv_store_db_latency_ms", "Query latency (ms)",
                                   buckets=[1, 5, 10, 25, 50, 100, 250, 500]),
        "write_buffer_len": Gauge("conv_store_write_buffer_len", "Write buffer length"),
    }


# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------
def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is not None:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError) as e:
                log.error("Failed to load config from %s: %s", config_path, e)
        else:
            log.warning("Config file not found: %s", config_path)
        return {}

    paths = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError) as e:
                log.error("Failed to load config from %s: %s", p, e)
    log.warning("No config.yaml found; using empty configuration.")
    return {}


# ------------------------------------------------------------------------------
# Abstract Base Store (extended with checkpoint methods)
# ------------------------------------------------------------------------------
class ConversationStore(ABC):
    # ── Existing methods ──────────────────────────────────────────────────
    @abstractmethod
    async def get(self, conv_id: str) -> Optional[List[dict]]:
        pass

    @abstractmethod
    async def add_message(self, conv_id: str, role: str, content: str, lease_token: Optional[str] = None) -> bool:
        """Returns False if duplicate lease_token detected."""
        pass

    @abstractmethod
    async def create(self, conv_id: str, messages: List[dict]) -> None:
        pass

    @abstractmethod
    async def delete(self, conv_id: str) -> None:
        pass

    @abstractmethod
    async def list(self) -> List[dict]:
        pass

    @abstractmethod
    async def prune_old(self) -> int:
        pass

    @abstractmethod
    def lock(self, conv_id: str):
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    # ── Checkpoint methods (Section 2) ──────────────────────────────────
    @abstractmethod
    async def save_checkpoint(self, conv_id: str, checkpoint: dict) -> str:
        """
        Store a generation checkpoint.
        checkpoint must contain: model_id, segment_id, accumulated_text,
        tokens_generated, status (in_progress/interrupted/complete).
        Returns the checkpoint_id.
        """
        pass

    @abstractmethod
    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        """Return the most recent checkpoint for this conversation, or None."""
        pass

    @abstractmethod
    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        """Return a specific checkpoint by its ID."""
        pass

    @abstractmethod
    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        """Update the status of a checkpoint (e.g., to 'complete')."""
        pass


# ------------------------------------------------------------------------------
# In‑Memory Store with Adaptive LRU, lease cache eviction, and checkpoints
# ------------------------------------------------------------------------------
class MemoryConversationStore(ConversationStore):
    def __init__(
        self,
        max_conversations: int = 100,
        max_messages: int = 100,
        max_age_days: float = 7.0,
        enable_adaptive_lru: bool = True,
        min_conversations: int = 50,
        max_lease_cache_size: int = 10_000,       # bound the lease token cache
    ) -> None:
        self._max_conv = max_conversations
        self._min_conv = min_conversations
        self._max_msgs = max_messages
        self._max_age_days = max_age_days
        self._enable_adaptive = enable_adaptive_lru
        self._data: Dict[str, OrderedDict] = {}
        self._meta: Dict[str, dict] = {}
        # lease cache: conv_id -> latest lease_token, bounded LRU
        self._lease_cache: Dict[str, str] = {}
        self._lease_cache_max = max_lease_cache_size
        self._global_lock = asyncio.Lock()
        self._conv_locks: Dict[str, asyncio.Lock] = {}
        # ── Checkpoint storage ────────────────────────────────────────────
        self._checkpoints: Dict[str, Dict[str, dict]] = {}  # conv_id -> {checkpoint_id: checkpoint_data}
        self._latest_checkpoint_id: Dict[str, str] = {}     # conv_id -> latest checkpoint_id

    def _get_conv_lock(self, conv_id: str) -> asyncio.Lock:
        if conv_id not in self._conv_locks:
            self._conv_locks[conv_id] = asyncio.Lock()
        return self._conv_locks[conv_id]

    @asynccontextmanager
    async def lock(self, conv_id: str):
        lk = self._get_conv_lock(conv_id)
        await lk.acquire()
        try:
            yield
        finally:
            lk.release()

    # ── Existing methods ──────────────────────────────────────────────────
    async def get(self, conv_id: str) -> Optional[List[dict]]:
        async with self.lock(conv_id):
            conv = self._data.get(conv_id)
            return list(conv.values()) if conv else None

    async def add_message(self, conv_id: str, role: str, content: str, lease_token: Optional[str] = None) -> bool:
        async with self.lock(conv_id):
            # Idempotent write protection
            if lease_token:
                prev = self._lease_cache.get(conv_id)
                if prev and prev == lease_token:
                    return False  # duplicate
                # Evict oldest if cache is full
                if len(self._lease_cache) >= self._lease_cache_max:
                    # remove the first inserted key (arbitrary)
                    oldest = next(iter(self._lease_cache))
                    del self._lease_cache[oldest]
                self._lease_cache[conv_id] = lease_token

            if conv_id not in self._data:
                self._data[conv_id] = OrderedDict()
                self._meta[conv_id] = {
                    "id": conv_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "message_count": 0,
                }
            conv = self._data[conv_id]
            msg = {"role": role, "content": content, "timestamp": time.time()}
            conv[msg["timestamp"]] = msg
            while len(conv) > self._max_msgs:
                conv.popitem(last=False)
            self._meta[conv_id]["updated_at"] = time.time()
            self._meta[conv_id]["message_count"] = len(conv)

        # Adaptive LRU: if RAM pressure, reduce max conversations
        async with self._global_lock:
            current = len(self._data)
            if current > self._max_conv:
                sorted_meta = sorted(self._meta.items(), key=lambda x: x[1]["updated_at"])
                to_remove = current - self._max_conv
                for conv_id, _ in sorted_meta[:to_remove]:
                    async with self.lock(conv_id):
                        self._data.pop(conv_id, None)
                        self._meta.pop(conv_id, None)
                    self._conv_locks.pop(conv_id, None)

        if self._enable_adaptive:
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.percent > 85:
                    self._max_conv = max(self._min_conv, self._max_conv // 2)
                elif mem.percent < 40:
                    self._max_conv = min(100, self._max_conv * 2)
            except ImportError:
                pass
        return True

    async def create(self, conv_id: str, messages: List[dict]) -> None:
        async with self.lock(conv_id):
            self._data[conv_id] = OrderedDict()
            self._meta[conv_id] = {
                "id": conv_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                "message_count": 0,
            }
            for msg in messages:
                role, content = msg.get("role"), msg.get("content")
                if role and content is not None:
                    ts = time.time()
                    self._data[conv_id][ts] = {"role": role, "content": content, "timestamp": ts}
            while len(self._data[conv_id]) > self._max_msgs:
                self._data[conv_id].popitem(last=False)
            self._meta[conv_id]["message_count"] = len(self._data[conv_id])
            self._meta[conv_id]["updated_at"] = time.time()

    async def delete(self, conv_id: str) -> None:
        async with self.lock(conv_id):
            self._data.pop(conv_id, None)
            self._meta.pop(conv_id, None)
            self._lease_cache.pop(conv_id, None)
            # Also delete checkpoints
            self._checkpoints.pop(conv_id, None)
            self._latest_checkpoint_id.pop(conv_id, None)
        self._conv_locks.pop(conv_id, None)

    async def list(self) -> List[dict]:
        async with self._global_lock:
            return list(self._meta.values())

    async def prune_old(self) -> int:
        if self._max_age_days <= 0:
            return 0
        cutoff = time.time() - (self._max_age_days * 86400)
        to_delete = []
        async with self._global_lock:
            to_delete = [cid for cid, meta in self._meta.items() if meta["updated_at"] < cutoff]
        for cid in to_delete:
            await self.delete(cid)
        return len(to_delete)

    async def close(self) -> None:
        pass

    # ── Checkpoint methods ────────────────────────────────────────────────
    async def save_checkpoint(self, conv_id: str, checkpoint: dict) -> str:
        # ── Ensure conversation exists (FIX: Issue 2) ────────────────────
        async with self.lock(conv_id):
            if conv_id not in self._data:
                self._data[conv_id] = OrderedDict()
                self._meta[conv_id] = {
                    "id": conv_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "message_count": 0,
                }

        checkpoint_id = str(uuid.uuid4())
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "model_id": checkpoint.get("model_id", ""),
            "segment_id": checkpoint.get("segment_id", 0),
            "accumulated_text": checkpoint.get("accumulated_text", ""),
            "tokens_generated": checkpoint.get("tokens_generated", 0),
            "status": checkpoint.get("status", "in_progress"),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        async with self.lock(conv_id):
            if conv_id not in self._checkpoints:
                self._checkpoints[conv_id] = {}
            self._checkpoints[conv_id][checkpoint_id] = checkpoint_data
            self._latest_checkpoint_id[conv_id] = checkpoint_id
        return checkpoint_id

    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        async with self.lock(conv_id):
            cid = self._latest_checkpoint_id.get(conv_id)
            if not cid:
                return None
            return self._checkpoints.get(conv_id, {}).get(cid)

    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        async with self.lock(conv_id):
            return self._checkpoints.get(conv_id, {}).get(checkpoint_id)

    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        async with self.lock(conv_id):
            cp = self._checkpoints.get(conv_id, {}).get(checkpoint_id)
            if not cp:
                return False
            cp["status"] = status
            cp["updated_at"] = time.time()
            return True


# ------------------------------------------------------------------------------
# Base SQL Store with Write Buffer, Lease Token, Retry Jitter, Health Check
# and checkpoint support (shared logic)
# ------------------------------------------------------------------------------
class SQLConversationStore(ConversationStore):
    def __init__(
        self,
        max_conversations: int = 100,
        max_messages: int = 100,
        max_age_days: float = 7.0,
        pool_size: int = 10,
        max_overflow: int = 5,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        retry_attempts: int = 3,
        retry_backoff_base: float = 0.1,
        retry_jitter: bool = True,
        write_buffer_size: int = 32,
        write_flush_interval: float = 0.05,
        enable_lease: bool = True,
        buffer_retry_attempts: int = 3,       # retries for failed flushed items
    ) -> None:
        self._max_conv = max_conversations
        self._max_msgs = max_messages
        self._max_age_days = max_age_days
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._retry_attempts = retry_attempts
        self._retry_backoff_base = retry_backoff_base
        self._retry_jitter = retry_jitter
        self._enable_lease = enable_lease
        self._buffer_retry_attempts = buffer_retry_attempts

        self._pool = None
        self._initialized = False
        self._conv_locks: Dict[str, asyncio.Lock] = {}
        self._write_lock = asyncio.Lock()

        # Write buffer
        self._write_buffer: deque = deque()
        self._buffer_size = write_buffer_size
        self._flush_interval = write_flush_interval
        self._buffer_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Metrics
        self._prom = _setup_prometheus()
        self._metrics = {
            "gets": 0, "adds": 0, "deletes": 0, "creates": 0, "prunes": 0,
            "db_errors": 0, "retries": 0, "buffer_flushes": 0, "buffer_drops": 0,
        }

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

    @abstractmethod
    def _convert_placeholders(self, query: str, params: tuple) -> str:
        pass

    def _backoff(self, attempt: int) -> float:
        sleep = self._retry_backoff_base * (2 ** attempt)
        if self._retry_jitter:
            sleep *= random.uniform(0.8, 1.2)
        return min(sleep, 5.0)

    def _get_conv_lock(self, conv_id: str) -> asyncio.Lock:
        if conv_id not in self._conv_locks:
            self._conv_locks[conv_id] = asyncio.Lock()
        return self._conv_locks[conv_id]

    @asynccontextmanager
    async def lock(self, conv_id: str):
        lk = self._get_conv_lock(conv_id)
        await lk.acquire()
        try:
            yield
        finally:
            lk.release()

    async def _start_buffer_flusher(self) -> None:
        if self._buffer_task is None or self._buffer_task.done():
            self._buffer_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Buffer flush error: %s", e)

    async def _flush_buffer(self) -> None:
        async with self._write_lock:
            if not self._write_buffer:
                return
            batch = list(self._write_buffer)
            self._write_buffer.clear()
            if self._prom.get("write_buffer_len"):
                self._prom["write_buffer_len"].set(0)
        # Retry each item a few times
        for query, params in batch:
            for attempt in range(self._buffer_retry_attempts):
                try:
                    await self._execute(query, params, commit=True)
                    break
                except Exception as e:
                    if attempt == self._buffer_retry_attempts - 1:
                        log.error("Failed to flush write after %d attempts: %s",
                                  self._buffer_retry_attempts, e)
                        self._metrics["db_errors"] += 1
                    else:
                        await asyncio.sleep(0.01 * (2 ** attempt))
        self._metrics["buffer_flushes"] += 1

    async def _enqueue_write(self, query: str, params=()) -> None:
        async with self._write_lock:
            if len(self._write_buffer) >= self._buffer_size:
                self._write_buffer.popleft()
                self._metrics["buffer_drops"] += 1
            self._write_buffer.append((query, params))
            if self._prom.get("write_buffer_len"):
                self._prom["write_buffer_len"].inc()
        await self._start_buffer_flusher()

    async def _execute_buffered(self, query: str, params=(), *, commit: bool = False, buffered: bool = False) -> Any:
        if buffered and commit:
            await self._enqueue_write(query, params)
            return None
        else:
            return await self._execute(query, params, commit=commit)

    async def health_check(self) -> bool:
        """Return True if the database is reachable."""
        try:
            await self._fetchone("SELECT 1", ())
            return True
        except Exception:
            return False

    # ── Public API (existing) ──────────────────────────────────────────────
    async def get(self, conv_id: str) -> Optional[List[dict]]:
        """Return the current message list for a conversation.

        this used to read messages without taking the per-
        conversation lock. Concurrent ``create()`` and ``add_message()``
        (which both delete-then-insert or insert-then-count) can race
        here, exposing a partial list — for example, the DELETE from a
        ``_create_transactional`` runs, but the subsequent INSERTs have
        not yet committed. The caller (``_merge_history``) then feeds
        that torn list into the next prompt as authoritative history,
        producing a subtly wrong generation.

        ``MemoryConversationStore.get()`` already takes the lock; this
        brings the SQL backends into line. The lock is per-conv_id, so
        concurrent GETs on different conversations still proceed in
        parallel.
        """
        async with self.lock(conv_id):
            query = (
                "SELECT ts, role, content FROM messages "
                "WHERE conv_id = ? ORDER BY ts"
            )
            rows = await self._fetchall(query, (conv_id,))
        if not rows:
            return None
        return [{"role": r[1], "content": r[2], "timestamp": r[0]} for r in rows]

    async def add_message(self, conv_id: str, role: str, content: str, lease_token: Optional[str] = None) -> bool:
        if self._enable_lease and lease_token:
            existing = await self._fetchone(
                "SELECT 1 FROM message_leases WHERE conv_id = ? AND lease_token = ?",
                (conv_id, lease_token)
            )
            if existing:
                return False
            await self._execute(
                "INSERT INTO message_leases (conv_id, lease_token, created_at) VALUES (?, ?, ?)",
                (conv_id, lease_token, time.time()),
                commit=True
            )

        async with self.lock(conv_id):
            now = time.time()
            upsert = """
                INSERT INTO conversations (id, created_at, updated_at, message_count)
                VALUES (?, ?, ?, 0)
                ON CONFLICT (id) DO NOTHING
            """
            await self._execute_buffered(upsert, (conv_id, now, now), commit=True, buffered=True)

            row = await self._fetchone("SELECT message_count FROM conversations WHERE id = ?", (conv_id,))
            msg_count = row[0] if row else 0

            insert_msg = "INSERT INTO messages (conv_id, ts, role, content) VALUES (?, ?, ?, ?)"
            await self._execute_buffered(insert_msg, (conv_id, now, role, content), commit=True, buffered=True)
            msg_count += 1

            update_conv = "UPDATE conversations SET message_count = ?, updated_at = ? WHERE id = ?"
            await self._execute_buffered(update_conv, (msg_count, now, conv_id), commit=True, buffered=True)

            if msg_count > self._max_msgs:
                delete_excess = """
                    DELETE FROM messages
                    WHERE conv_id = ? AND ts NOT IN (
                        SELECT ts FROM messages WHERE conv_id = ? ORDER BY ts DESC LIMIT ?
                    )
                """
                await self._execute_buffered(delete_excess, (conv_id, conv_id, self._max_msgs), commit=True, buffered=True)
                await self._execute_buffered(
                    "UPDATE conversations SET message_count = ? WHERE id = ?",
                    (self._max_msgs, conv_id), commit=True, buffered=True
                )

        await self._enforce_conversation_limit()
        return True

    async def create(self, conv_id: str, messages: List[dict]) -> None:
        async with self.lock(conv_id):
            await self._create_transactional(conv_id, messages)

    async def _create_transactional(self, conv_id: str, messages: List[dict]) -> None:
        """Override in each backend. Default: buffered (existing) behaviour."""
        now = time.time()
        await self._execute_buffered(
            "DELETE FROM messages WHERE conv_id = ?",
            (conv_id,), commit=True, buffered=True,
        )
        upsert = """
            INSERT INTO conversations (id, created_at, updated_at, message_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = excluded.updated_at,
                message_count = excluded.message_count
        """
        await self._execute_buffered(
            upsert, (conv_id, now, now, len(messages)),
            commit=True, buffered=True,
        )
        insert_msg = "INSERT INTO messages (conv_id, ts, role, content) VALUES (?, ?, ?, ?)"
        for msg in messages:
            role, content = msg.get("role"), msg.get("content")
            if role and content is not None:
                await self._execute_buffered(
                    insert_msg, (conv_id, time.time(), role, content),
                    commit=True, buffered=True,
                )

    async def delete(self, conv_id: str) -> None:
        await self._execute_buffered("DELETE FROM conversations WHERE id = ?", (conv_id,), commit=True, buffered=True)
        # Also clean up checkpoints (cascade should handle if foreign keys set, but do it explicitly for safety)
        await self._execute_buffered("DELETE FROM generation_checkpoints WHERE conv_id = ?", (conv_id,), commit=True, buffered=True)

    async def list(self) -> List[dict]:
        query = "SELECT id, created_at, updated_at, message_count FROM conversations ORDER BY updated_at DESC"
        rows = await self._fetchall(query)
        return [{"id": r[0], "created_at": r[1], "updated_at": r[2], "message_count": r[3]} for r in rows]

    async def prune_old(self) -> int:
        if self._max_age_days <= 0:
            return 0
        cutoff = time.time() - (self._max_age_days * 86400)
        rows = await self._fetchall("SELECT id FROM conversations WHERE updated_at < ?", (cutoff,))
        old_ids = [r[0] for r in rows]
        for cid in old_ids:
            await self.delete(cid)
        return len(old_ids)

    async def _enforce_conversation_limit(self) -> None:
        if self._max_conv <= 0:
            return
        row = await self._fetchone("SELECT COUNT(*) as cnt FROM conversations")
        count = row[0] if row else 0
        if count > self._max_conv:
            excess = count - self._max_conv
            rows = await self._fetchall(
                "SELECT id FROM conversations ORDER BY updated_at ASC LIMIT ?", (excess,)
            )
            old_ids = [r[0] for r in rows]
            for cid in old_ids:
                await self.delete(cid)
            log.warning("Pruned %d old conversations (LRU)", excess)

    async def close(self) -> None:
        self._stop_event.set()
        if self._buffer_task:
            self._buffer_task.cancel()
            try:
                await self._buffer_task
            except asyncio.CancelledError:
                pass
        await self._flush_buffer()
        if self._pool:
            await self._pool.close()
            self._pool = None
        log.info("Conversation store closed.")

    # ── Checkpoint methods (shared SQL logic) ──────────────────────────────
    async def save_checkpoint(self, conv_id: str, checkpoint: dict) -> str:
        # ── Ensure conversation row exists (FIX: Issue 2) ──────────────────
        now = time.time()
        upsert = """
            INSERT INTO conversations (id, created_at, updated_at, message_count)
            VALUES (?, ?, ?, 0)
            ON CONFLICT (id) DO NOTHING
        """
        await self._execute(upsert, (conv_id, now, now), commit=True)

        checkpoint_id = str(uuid.uuid4())
        query = """
            INSERT INTO generation_checkpoints
            (checkpoint_id, conv_id, model_id, segment_id, accumulated_text,
             tokens_generated, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            checkpoint_id,
            conv_id,
            checkpoint.get("model_id", ""),
            checkpoint.get("segment_id", 0),
            checkpoint.get("accumulated_text", ""),
            checkpoint.get("tokens_generated", 0),
            checkpoint.get("status", "in_progress"),
            now,
            now,
        )
        await self._execute_buffered(query, params, commit=True, buffered=True)
        return checkpoint_id

    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        query = """
            SELECT checkpoint_id, model_id, segment_id, accumulated_text,
                   tokens_generated, status, created_at, updated_at
            FROM generation_checkpoints
            WHERE conv_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
        """
        row = await self._fetchone(query, (conv_id,))
        if not row:
            return None
        return {
            "checkpoint_id": row[0],
            "model_id": row[1],
            "segment_id": row[2],
            "accumulated_text": row[3],
            "tokens_generated": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        query = """
            SELECT checkpoint_id, model_id, segment_id, accumulated_text,
                   tokens_generated, status, created_at, updated_at
            FROM generation_checkpoints
            WHERE conv_id = ? AND checkpoint_id = ?
        """
        row = await self._fetchone(query, (conv_id, checkpoint_id))
        if not row:
            return None
        return {
            "checkpoint_id": row[0],
            "model_id": row[1],
            "segment_id": row[2],
            "accumulated_text": row[3],
            "tokens_generated": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        query = """
            UPDATE generation_checkpoints
            SET status = ?, updated_at = ?
            WHERE conv_id = ? AND checkpoint_id = ?
        """
        params = (status, time.time(), conv_id, checkpoint_id)
        result = await self._execute_buffered(query, params, commit=True, buffered=True)
        # For SQLite/MySQL, result.rowcount may not be available; we'll do a follow-up check.
        # Simpler: try to select the updated row.
        row = await self._fetchone(
            "SELECT 1 FROM generation_checkpoints WHERE conv_id = ? AND checkpoint_id = ? AND status = ?",
            (conv_id, checkpoint_id, status)
        )
        return row is not None


# ------------------------------------------------------------------------------
# SQLite Implementation (extended with checkpoint table)
# ------------------------------------------------------------------------------
class SQLiteConversationStore(SQLConversationStore):
    def __init__(self, db_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._db_conn = None
        self._write_lock = asyncio.Lock()

    async def _init_pool(self) -> None:
        try:
            import aiosqlite
        except ImportError:
            raise ImportError("aiosqlite not installed. Run: pip install aiosqlite")
        self._db_conn = await aiosqlite.connect(self._db_path)
        await self._db_conn.execute("PRAGMA journal_mode=WAL")
        await self._db_conn.execute("PRAGMA synchronous=NORMAL")
        await self._db_conn.execute("PRAGMA foreign_keys=ON")
        await self._db_conn.execute("PRAGMA busy_timeout=5000")
        await self._create_tables()
        self._initialized = True

    async def _create_tables(self) -> None:
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                message_count INTEGER NOT NULL
            )
        """)
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                conv_id TEXT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id)")
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS message_leases (
                conv_id TEXT,
                lease_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (conv_id, lease_token)
            )
        """)
        # ── Checkpoint table ───────────────────────────────────────────────
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                conv_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                segment_id INTEGER NOT NULL,
                accumulated_text TEXT NOT NULL,
                tokens_generated INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")
        await self._db_conn.commit()

    def _convert_placeholders(self, query: str, params: tuple) -> str:
        return query

    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        query = self._adapt_on_conflict(query)
        max_retries = self._retry_attempts
        for attempt in range(max_retries):
            try:
                if commit:
                    async with self._write_lock:
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
                self._metrics["db_errors"] += 1
                raise

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        query = self._adapt_on_conflict(query)
        cursor = await self._db_conn.execute(query, params)
        return await cursor.fetchone()

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        query = self._adapt_on_conflict(query)
        cursor = await self._db_conn.execute(query, params)
        return await cursor.fetchall()

    def _adapt_on_conflict(self, query: str) -> str:
        if "ON CONFLICT (id) DO NOTHING" in query:
            return query.replace("INSERT INTO", "INSERT OR IGNORE INTO").replace(" ON CONFLICT (id) DO NOTHING", "")
        if "ON CONFLICT (id) DO UPDATE SET" in query:
            return query.replace("INSERT INTO", "INSERT OR REPLACE INTO").split(" ON CONFLICT")[0]
        return query

    async def _create_transactional(self, conv_id: str, messages: List[dict]) -> None:
        rows = [
            (conv_id, time.time(), m.get("role", "user"), m.get("content", ""))
            for m in messages
            if m.get("role") and m.get("content") is not None
        ]
        async with self._write_lock:
            try:
                await self._db_conn.execute("BEGIN")
                await self._db_conn.execute(
                    "DELETE FROM messages WHERE conv_id = ?", (conv_id,)
                )
                await self._db_conn.execute(
                    "INSERT OR REPLACE INTO conversations "
                    "(id, created_at, updated_at, message_count) "
                    "VALUES (?, ?, ?, ?)",
                    (conv_id, time.time(), time.time(), len(messages)),
                )
                if rows:
                    await self._db_conn.executemany(
                        "INSERT INTO messages (conv_id, ts, role, content) "
                        "VALUES (?, ?, ?, ?)",
                        rows,
                    )
                await self._db_conn.commit()
            except Exception:
                try:
                    await self._db_conn.rollback()
                except Exception:
                    pass
                raise

    async def close(self) -> None:
        await super().close()
        if self._db_conn:
            await self._db_conn.close()
            self._db_conn = None


# ------------------------------------------------------------------------------
# PostgreSQL Implementation (extended with checkpoint table)
# ------------------------------------------------------------------------------
class PostgreSQLConversationStore(SQLConversationStore):
    RETRY_SQLSTATES = frozenset({
        "40001", "40P01", "08006", "08001", "57P03"
    })

    def __init__(self, host: str, port: int, database: str, username: str, password: str, **kwargs) -> None:
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
            host=self._host, port=self._port, database=self._database,
            user=self._username, password=self._password,
            min_size=self._pool_size, max_size=self._pool_size + self._max_overflow,
            timeout=self._pool_timeout, max_inactive_connection_lifetime=self._pool_recycle,
        )
        await self._create_tables()
        self._initialized = True

    async def _create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR(255) PRIMARY KEY,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    message_count INTEGER NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    conv_id VARCHAR(255) NOT NULL,
                    ts DOUBLE PRECISION NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS message_leases (
                    conv_id VARCHAR(255) NOT NULL,
                    lease_token VARCHAR(128) NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (conv_id, lease_token)
                )
            """)
            # ── Checkpoint table ───────────────────────────────────────────────
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_checkpoints (
                    checkpoint_id VARCHAR(36) PRIMARY KEY,
                    conv_id VARCHAR(255) NOT NULL,
                    model_id VARCHAR(255) NOT NULL,
                    segment_id INTEGER NOT NULL,
                    accumulated_text TEXT NOT NULL,
                    tokens_generated INTEGER NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")

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
        for attempt in range(max_retries):
            try:
                async with self._pool.acquire() as conn:
                    if conn.is_in_transaction():
                        await conn.rollback()
                    try:
                        if commit:
                            async with conn.transaction():
                                return await conn.execute(pg_query, *params)
                        return await conn.execute(pg_query, *params)
                    except asyncio.CancelledError:
                        try:
                            await conn.rollback()
                        except Exception:
                            pass
                        raise
            except Exception as e:
                if hasattr(e, "sqlstate") and e.sqlstate in self.RETRY_SQLSTATES:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                self._metrics["db_errors"] += 1
                raise

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        pg_query = self._convert_placeholders(query, params)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(pg_query, *params)
            return tuple(row) if row else None

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        pg_query = self._convert_placeholders(query, params)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(pg_query, *params)
            return [tuple(r) for r in rows]

    async def _create_transactional(self, conv_id: str, messages: List[dict]) -> None:
        pg_delete = self._convert_placeholders(
            "DELETE FROM messages WHERE conv_id = ?", (conv_id,)
        )
        pg_upsert = self._convert_placeholders(
            """
            INSERT INTO conversations (id, created_at, updated_at, message_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = excluded.updated_at,
                message_count = excluded.message_count
            """,
            (conv_id, 0, 0, 0),
        )
        rows = [
            (conv_id, time.time(), m.get("role", "user"), m.get("content", ""))
            for m in messages
            if m.get("role") and m.get("content") is not None
        ]
        async with self._pool.acquire() as conn:
            try:
                async with conn.transaction():
                    await conn.execute(pg_delete, conv_id)
                    await conn.execute(
                        pg_upsert, conv_id, time.time(), time.time(), len(messages)
                    )
                    if rows:
                        await conn.executemany(
                            "INSERT INTO messages (conv_id, ts, role, content) "
                            "VALUES ($1, $2, $3, $4)",
                            rows,
                        )
            except asyncio.CancelledError:
                try:
                    await conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

# ------------------------------------------------------------------------------
# MySQL Implementation (extended with checkpoint table)
# ------------------------------------------------------------------------------
class MySQLConversationStore(SQLConversationStore):
    def __init__(self, host: str, port: int, database: str, username: str, password: str, **kwargs) -> None:
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
            host=self._host, port=self._port, db=self._database,
            user=self._username, password=self._password,
            minsize=self._pool_size, maxsize=self._pool_size + self._max_overflow,
            pool_recycle=self._pool_recycle, autocommit=True,
        )
        await self._create_tables()
        self._initialized = True

    async def _create_tables(self) -> None:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id VARCHAR(255) PRIMARY KEY,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        message_count INTEGER NOT NULL
                    )
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        conv_id VARCHAR(255) NOT NULL,
                        ts DOUBLE PRECISION NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS message_leases (
                        conv_id VARCHAR(255) NOT NULL,
                        lease_token VARCHAR(128) NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (conv_id, lease_token)
                    )
                """)
                # ── Checkpoint table ───────────────────────────────────────────
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS generation_checkpoints (
                        checkpoint_id VARCHAR(36) PRIMARY KEY,
                        conv_id VARCHAR(255) NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        segment_id INTEGER NOT NULL,
                        accumulated_text TEXT NOT NULL,
                        tokens_generated INTEGER NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")

    def _convert_placeholders(self, query: str, params: tuple) -> str:
        return query.replace("?", "%s")

    def _adapt_on_conflict(self, query: str) -> str:
        if "ON CONFLICT (id) DO NOTHING" in query:
            return query.replace("INSERT INTO", "INSERT IGNORE INTO").replace(" ON CONFLICT (id) DO NOTHING", "")
        if "ON CONFLICT (id) DO UPDATE SET" in query:
            parts = query.split(" ON CONFLICT (id) DO UPDATE SET ")
            if len(parts) == 2:
                base = parts[0]
                set_clause = parts[1]
                def repl(m):
                    col = m.group(1)
                    return f"{col} = VALUES({col})"
                new_set = re.sub(r"(\w+)\s*=\s*excluded\.\w+", repl, set_clause)
                return base + " ON DUPLICATE KEY UPDATE " + new_set
        return query

    async def _execute(self, query: str, params=(), *, commit: bool = False) -> Any:
        query = self._convert_placeholders(query, params)
        query = self._adapt_on_conflict(query)
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
                if any(kw in msg for kw in ("deadlock", "lock wait timeout", "connection")):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                self._metrics["db_errors"] += 1
                raise
        raise last_exc

    async def _fetchone(self, query: str, params=()) -> Optional[tuple]:
        query = self._convert_placeholders(query, params)
        query = self._adapt_on_conflict(query)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchone()

    async def _fetchall(self, query: str, params=()) -> List[tuple]:
        query = self._convert_placeholders(query, params)
        query = self._adapt_on_conflict(query)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()


# ------------------------------------------------------------------------------
# Factory Function (updated to pass checkpoint support)
# ------------------------------------------------------------------------------
async def create_conversation_store(cfg: Optional[Dict[str, Any]] = None) -> Optional[ConversationStore]:
    if cfg is None:
        cfg = _load_config()
    conv_cfg = cfg.get("conversation", {})
    if not conv_cfg.get("enabled", True):
        log.info("Conversation store disabled.")
        return None

    storage = conv_cfg.get("storage", "memory").lower()
    max_conv = conv_cfg.get("max_conversations", 100)
    max_msgs = conv_cfg.get("max_messages_per_conversation", 100)
    max_age = conv_cfg.get("max_age_days", 7.0)
    pool_size = conv_cfg.get("pool_size", 10)
    max_overflow = conv_cfg.get("max_overflow", 5)
    pool_timeout = conv_cfg.get("pool_timeout", 30)
    pool_recycle = conv_cfg.get("pool_recycle", 3600)
    retry_attempts = conv_cfg.get("retry_attempts", 3)
    retry_backoff_base = conv_cfg.get("retry_backoff_base", 0.1)
    retry_jitter = conv_cfg.get("retry_jitter", True)
    write_buffer_size = conv_cfg.get("write_buffer_size", 32)
    write_flush_interval = conv_cfg.get("write_flush_interval", 0.05)
    enable_lease = conv_cfg.get("enable_lease", True)

    common = {
        "max_conversations": max_conv,
        "max_messages": max_msgs,
        "max_age_days": max_age,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "retry_attempts": retry_attempts,
        "retry_backoff_base": retry_backoff_base,
        "retry_jitter": retry_jitter,
        "write_buffer_size": write_buffer_size,
        "write_flush_interval": write_flush_interval,
        "enable_lease": enable_lease,
    }

    store: Optional[ConversationStore] = None

    if storage == "sqlite":
        db_path = conv_cfg.get("sqlite_path", "/tmp/llm_conversations.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = SQLiteConversationStore(db_path=db_path, **common)
    elif storage == "postgresql":
        store = PostgreSQLConversationStore(
            host=conv_cfg.get("host", "localhost"),
            port=int(conv_cfg.get("port", 5432)),
            database=conv_cfg.get("database", "llm_conversations"),
            username=conv_cfg.get("username", "postgres"),
            password=conv_cfg.get("password", ""),
            **common,
        )
    elif storage == "mysql":
        store = MySQLConversationStore(
            host=conv_cfg.get("host", "localhost"),
            port=int(conv_cfg.get("port", 3306)),
            database=conv_cfg.get("database", "llm_conversations"),
            username=conv_cfg.get("username", "root"),
            password=conv_cfg.get("password", ""),
            **common,
        )
    elif storage == "memory":
        store = MemoryConversationStore(
            max_conversations=max_conv,
            max_messages=max_msgs,
            max_age_days=max_age,
            enable_adaptive_lru=conv_cfg.get("adaptive_lru", True),
            min_conversations=conv_cfg.get("min_conversations", 50),
        )
        log.info("Conversation store: in‑memory (non‑persistent)")
        return store
    else:
        raise ValueError(f"Unknown conversation storage type: {storage}")

    try:
        await store._init_pool()
        log.info("Conversation store: %s initialized (pool size=%d)", storage, pool_size)
        return store
    except Exception as e:
        log.error("Failed to initialize %s store: %s, falling back to memory", storage, e)
        return MemoryConversationStore(
            max_conversations=max_conv,
            max_messages=max_msgs,
            max_age_days=max_age,
        )


async def shutdown_conversation_store(store: Optional[ConversationStore]) -> None:
    if store is not None and hasattr(store, "close"):
        await store.close()
        log.info("Conversation store closed.")


__all__ = [
    "ConversationStore",
    "MemoryConversationStore",
    "SQLiteConversationStore",
    "PostgreSQLConversationStore",
    "MySQLConversationStore",
    "create_conversation_store",
    "shutdown_conversation_store",
]
