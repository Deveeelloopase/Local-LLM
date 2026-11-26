from __future__ import annotations

"""
local_llm_server/memory/conversation_store.py — Production Conversation Storage
==============================================================================================
Enterprise‑grade conversation history management with pluggable database backends.

  • Leaky lease cache protection: the in‑memory store's lease token cache is
    bounded (configurable), with LRU eviction.
  • Write‑buffer resilience: failed writes are retried with short backoff
    before being dropped.
  • Health‑check for SQL stores.
  • All previous features retained (shadow‑paging, lease tokens, adaptive LRU, etc.).
  • Checkpoint support for resumable generation.

  NEW — `attach_request_id(request_id, checkpoint_id)`
  ─────────────────────────────────────────────────────
  A process-local, LRU-bounded index from `request_id` → `checkpoint_id`
  so a future `_REQUEST_RING` can correlate a stats record to
  the checkpoint persisted for the same request WITHOUT adding a
  `request_id` column to `generation_checkpoints`.

  `save_checkpoint()` auto-registers the mapping if the caller includes a
  `"request_id"` key in the checkpoint dict (future api_server change) OR
  the caller can call `conv_store.attach_request_id()` explicitly.

  Limitations (documented, not hidden):
    • Process-local only — NOT shared across multi-worker uvicorn, and
      cleared on restart. In `workers: 1` mode (this deployment's config)
      this is a non-issue. If you ever run workers > 1, a checkpoint can
      only be correlated back to its originating request within the
      worker that created it.
    • LRU-bounded at 4096 entries (defensive against unbounded growth in
      long-running processes). Older mappings are silently evicted.
    • Deleting a conversation does NOT invalidate the request_id →
      checkpoint_id mapping; a subsequent lookup returns the ID, and
      `get_checkpoint_by_id` returns None because the row is gone. This
      is the correct behavior (idempotent miss) and costs nothing.

Dependencies on `api_server.py`
───────────────────────────────
  The version of `api_server.py` in the current tree ALREADY expects
  this contract:
    • `checkpoint_data = {...}` on save — includes `original_prompt` and
      `merged_history` when truthy.
    • `checkpoint_data.get("original_prompt") or ""` on read — tolerant
      of absent / None.
    • `checkpoint_data.get("merged_history") or []` on read — same.
    • `checkpoint_data["status"] == "complete"` → 400 "already complete".
    • `checkpoint_data["segment_id"] + 1` — the resume handler bumps it.

version = 1.0.0
"""

import asyncio
import json
import logging
import random
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Checkpoint status contract
# ------------------------------------------------------------------------------
# Valid values for `generation_checkpoints.status`. Kept as a module constant
# so callers (and validators) compare against one authoritative set.
VALID_CHECKPOINT_STATUSES = frozenset({"in_progress", "interrupted", "complete"})

# Legal status transitions. `complete` is terminal — no caller, however
# privileged, should be able to "re-open" a completed conversation via this
# API. `in_progress` ↔ `interrupted` is a legitimate retry/retry-again
# pattern; both may go to `complete`.
_ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    "in_progress": frozenset({"interrupted", "complete"}),
    "interrupted": frozenset({"in_progress", "complete"}),
    "complete":    frozenset(),
}


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
# Request → Checkpoint index (_REQUEST_RING correlation support)
# ------------------------------------------------------------------------------
class _RequestCheckpointIndex:
    """Process-local, LRU-bounded map from `request_id` → `checkpoint_id`.

    Lets a future `_REQUEST_RING` in api_server.py correlate a completed
    request's stats record to the checkpoint that was persisted for it,
    without adding a `request_id` column to `generation_checkpoints`.

    Thread-safe (a `threading.Lock` guards the OrderedDict; all ops are
    O(1) dict touches, so lock hold time is microseconds and this class
    is safe to call from both sync and async contexts without introducing
    event-loop stalls).

    LRU eviction: when `max_size` is exceeded, the least-recently-looked-up
    entry is dropped. The default 4096 supports 4096 concurrent in-flight
    requests — far more than this deployment ever runs.
    """

    __slots__ = ("_map", "_lock", "_max")

    def __init__(self, max_size: int = 4096) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._map: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = int(max_size)

    def put(self, request_id: str, checkpoint_id: str) -> None:
        rid = str(request_id or "")
        cid = str(checkpoint_id or "")
        if not rid or not cid:
            return
        with self._lock:
            self._map[rid] = cid
            self._map.move_to_end(rid)
            while len(self._map) > self._max:
                self._map.popitem(last=False)

    def get(self, request_id: str) -> Optional[str]:
        rid = str(request_id or "")
        if not rid:
            return None
        with self._lock:
            cid = self._map.get(rid)
            if cid is not None:
                self._map.move_to_end(rid)
            return cid

    def discard(self, request_id: str) -> None:
        rid = str(request_id or "")
        if not rid:
            return
        with self._lock:
            self._map.pop(rid, None)

    def clear(self) -> None:
        with self._lock:
            self._map.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)


_REQUEST_CHECKPOINT_INDEX = _RequestCheckpointIndex()


def attach_request_id(request_id: str, checkpoint_id: str) -> None:
    """Record that `checkpoint_id` was created for `request_id`.

    Module-level convenience form of `ConversationStore.attach_request_id`.
    Both write to the same process-local index — use whichever fits the
    call site. Sync (returns None); do NOT `await` this.

    Idempotent: re-attaching the same request_id overwrites the mapping
    and bumps its LRU position.
    """
    _REQUEST_CHECKPOINT_INDEX.put(request_id, checkpoint_id)


def get_checkpoint_id_for_request(request_id: str) -> Optional[str]:
    """Look up the checkpoint_id associated with `request_id`.

    Module-level convenience form of
    `ConversationStore.get_checkpoint_id_for_request`. Returns None for
    unknown or already-evicted request IDs.
    """
    return _REQUEST_CHECKPOINT_INDEX.get(request_id)


# ------------------------------------------------------------------------------
# Abstract Base Store
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

    # ── Checkpoint methods ────────────────────────────────────────────────
    @abstractmethod
    async def save_checkpoint(self, conv_id: str, checkpoint: dict) -> str:
        """Store a generation checkpoint and return its `checkpoint_id`.

        `checkpoint` must contain (api_server contract):
            model_id            : str
            segment_id          : int
            accumulated_text    : str  — generated text only (prompt excluded)
            tokens_generated    : int
            status              : str ∈ {in_progress, interrupted, complete}

        Optional fields, persisted when provided:
            original_prompt     : str  — the full templated prompt from the
                                        originating request; combined with
                                        `accumulated_text` on resume.
            merged_history      : list[dict] — the merged conversation history
                                        from the originating request; used
                                        to re-save the completed conversation.
            request_id          : str  — the originating request id; used to
                                        index request_id → checkpoint_id in a
                                        process-local map (see
                                        `attach_request_id`). Not persisted
                                        to the DB.
        """
        pass

    @abstractmethod
    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        """Return the most-recently-*touched* checkpoint for this conversation.

        "Touched" means `updated_at` (creation OR status update). Matches
        the SQL `ORDER BY updated_at DESC`. Returns None if the conversation
        has no checkpoints at all.

        NOTE: the return value can legitimately be a `status="complete"`
        checkpoint. api_server's `/v1/chat/resume` handles that case with a
        400 ("already complete"); this method does NOT filter it out. If
        it filtered, a conversation with checkpoints [CP1=interrupted,
        CP2=complete] would return CP1 — actively worse, because resuming
        from CP1 would replay content the user already has.
        """
        pass

    @abstractmethod
    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        """Return a specific checkpoint by ID, or None if missing."""
        pass

    @abstractmethod
    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        """Transition a checkpoint's status.

        Returns True if the transition was applied or was a no-op
        (idempotent success). Returns False if:
          • the checkpoint does not exist,
          • `status` is not one of VALID_CHECKPOINT_STATUSES,
          • the transition is illegal (see `_ALLOWED_TRANSITIONS`), or
          • the underlying store is unavailable.

        Illegal transitions are logged at WARNING, never raised — callers
        currently invoke this inside `suppress(Exception)` blocks in
        api_server, so raising would not add information.
        """
        pass

    # ── Request-ID correlation (process-local) ──────────────────────
    # Default implementations. Both concrete stores inherit these. Kept
    # sync (not async) because they're trivial in-memory dict ops with no
    # I/O — wrapping them in `async def` would only add per-call overhead.
    def attach_request_id(self, request_id: str, checkpoint_id: str) -> None:
        """Record that `checkpoint_id` was created for `request_id`.

        Sync; do NOT `await`. Thread-safe. LRU-bounded at 4096 entries.
        """
        _REQUEST_CHECKPOINT_INDEX.put(request_id, checkpoint_id)

    def get_checkpoint_id_for_request(self, request_id: str) -> Optional[str]:
        """Look up the checkpoint_id associated with `request_id`.

        Sync; returns None for unknown / evicted request IDs.
        """
        return _REQUEST_CHECKPOINT_INDEX.get(request_id)


# ------------------------------------------------------------------------------
# Internal helpers shared by both memory and SQL stores
# ------------------------------------------------------------------------------
def _validate_checkpoint_payload(checkpoint: dict) -> Dict[str, Any]:
    """Normalise and validate a checkpoint dict before persisting it.

    Raises:
        TypeError:  if `checkpoint` is not a dict.
        ValueError: if `status` is not a known value.

    Returns a normalised dict with all expected keys present (missing
    optional keys set to None so downstream callers see a stable shape).
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"checkpoint must be a dict, got {type(checkpoint).__name__}"
        )

    status = checkpoint.get("status", "in_progress")
    if status not in VALID_CHECKPOINT_STATUSES:
        raise ValueError(
            f"checkpoint.status must be one of "
            f"{sorted(VALID_CHECKPOINT_STATUSES)}, got {status!r}"
        )

    try:
        segment_id = int(checkpoint.get("segment_id", 0) or 0)
    except (TypeError, ValueError):
        segment_id = 0

    try:
        tokens_generated = int(checkpoint.get("tokens_generated", 0) or 0)
    except (TypeError, ValueError):
        tokens_generated = 0

    # Preserve exactly what the caller provided for the two fields that are
    # optional on the wire. Empty string / None for `original_prompt` and
    # None for `merged_history` are both valid "not supplied" sentinels;
    # api_server's `.get(...) or default` handles either.
    original_prompt = checkpoint.get("original_prompt")
    if original_prompt is not None and not isinstance(original_prompt, str):
        original_prompt = str(original_prompt)

    merged_history = checkpoint.get("merged_history")
    if merged_history is not None and not isinstance(merged_history, list):
        # Defensive: refuse to guess. A non-list merged_history will only
        # corrupt the resume path later; better to drop it than store
        # something json.loads will choke on.
        log.warning(
            "checkpoint.merged_history has unexpected type %s — dropping",
            type(merged_history).__name__,
        )
        merged_history = None

    # request_id is NOT persisted — only used by save_checkpoint to register
    # the process-local index entry. Extract here so the caller's dict is
    # not mutated and the DB paths never see it.
    request_id = checkpoint.get("request_id")
    if request_id is not None:
        request_id = str(request_id)

    return {
        "model_id":          str(checkpoint.get("model_id") or ""),
        "segment_id":        segment_id,
        "accumulated_text":  str(checkpoint.get("accumulated_text") or ""),
        "tokens_generated":  tokens_generated,
        "status":            status,
        "original_prompt":   original_prompt,
        "merged_history":    merged_history,
        "_request_id":       request_id,  # internal-only, stripped before DB write
    }


def _copy_checkpoint_for_return(cp: dict) -> dict:
    """Shallow-copy a stored checkpoint and detach any mutable inner list.

    Callers of `get_checkpoint_by_id` / `get_latest_checkpoint` receive a
    dict they can safely hold without the risk that a subsequent in-store
    mutation (e.g. `update_checkpoint_status`) changes what they see, or
    that their own mutations corrupt the store.

    `merged_history` (a list of small role/content dicts) is deep-copied
    one level so a caller's `cp["merged_history"].append(...)` does not
    mutate the stored list.
    """
    out = dict(cp)
    mh = out.get("merged_history")
    if isinstance(mh, list):
        out["merged_history"] = [
            dict(m) if isinstance(m, dict) else m for m in mh
        ]
    return out


def _resolve_status_transition(current_status: str, new_status: str) -> Optional[str]:
    """Return `new_status` if the transition is legal, else None.

    - Same-status is treated as a legal no-op (idempotent).
    - `complete` is terminal.
    - `in_progress` ↔ `interrupted` are legal both ways.
    """
    if new_status not in VALID_CHECKPOINT_STATUSES:
        return None
    if current_status == new_status:
        return new_status
    if new_status in _ALLOWED_TRANSITIONS.get(current_status, frozenset()):
        return new_status
    return None


# ------------------------------------------------------------------------------
# In-Memory Store
# ------------------------------------------------------------------------------
class MemoryConversationStore(ConversationStore):
    def __init__(
        self,
        max_conversations: int = 100,
        max_messages: int = 100,
        max_age_days: float = 7.0,
        enable_adaptive_lru: bool = True,
        min_conversations: int = 50,
        max_lease_cache_size: int = 10_000,
    ) -> None:
        self._max_conv = max_conversations
        self._min_conv = min_conversations
        self._max_msgs = max_messages
        self._max_age_days = max_age_days
        self._enable_adaptive = enable_adaptive_lru
        self._data: Dict[str, OrderedDict] = {}
        self._meta: Dict[str, dict] = {}
        self._lease_cache: Dict[str, str] = {}
        self._lease_cache_max = max_lease_cache_size
        self._global_lock = asyncio.Lock()
        self._conv_locks: Dict[str, asyncio.Lock] = {}
        # Checkpoint state — conv_id → {checkpoint_id → checkpoint_data}
        self._checkpoints: Dict[str, Dict[str, dict]] = {}
        # Convenience pointer to the most-recently-touched checkpoint per
        # conv. Kept for O(1) lookup; MUST be updated by every method that
        # changes a checkpoint's `updated_at`.
        self._latest_checkpoint_id: Dict[str, str] = {}
        self._seq: int = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

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
            if lease_token:
                prev = self._lease_cache.get(conv_id)
                if prev and prev == lease_token:
                    return False
                if len(self._lease_cache) >= self._lease_cache_max:
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
            seq = self._next_seq()
            msg = {"role": role, "content": content,
                   "timestamp": time.time(), "seq": seq}
            conv[seq] = msg
            while len(conv) > self._max_msgs:
                conv.popitem(last=False)
            self._meta[conv_id]["updated_at"] = time.time()
            self._meta[conv_id]["message_count"] = len(conv)

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
                    seq = self._next_seq()
                    self._data[conv_id][seq] = {
                        "role": role, "content": content,
                        "timestamp": ts, "seq": seq,
                    }
            while len(self._data[conv_id]) > self._max_msgs:
                self._data[conv_id].popitem(last=False)
            self._meta[conv_id]["message_count"] = len(self._data[conv_id])
            self._meta[conv_id]["updated_at"] = time.time()

    async def delete(self, conv_id: str) -> None:
        async with self.lock(conv_id):
            self._data.pop(conv_id, None)
            self._meta.pop(conv_id, None)
            self._lease_cache.pop(conv_id, None)
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
        # Normalise + validate up-front so bad input raises before any
        # state mutation.
        norm = _validate_checkpoint_payload(checkpoint)
        request_id = norm.pop("_request_id", None)

        checkpoint_id = str(uuid.uuid4())
        now = time.time()
        checkpoint_data = {
            "checkpoint_id":    checkpoint_id,
            "model_id":         norm["model_id"],
            "segment_id":       norm["segment_id"],
            "accumulated_text": norm["accumulated_text"],
            "tokens_generated": norm["tokens_generated"],
            "status":           norm["status"],
            "original_prompt":  norm["original_prompt"],
            "merged_history":   norm["merged_history"],
            "created_at":       now,
            "updated_at":       now,
        }

        async with self.lock(conv_id):
            # Ensure the conversation exists — checkpoints may be saved
            # before any message was written (e.g. mid-first-generation).
            if conv_id not in self._data:
                self._data[conv_id] = OrderedDict()
                self._meta[conv_id] = {
                    "id": conv_id,
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                }
            bucket = self._checkpoints.setdefault(conv_id, {})
            bucket[checkpoint_id] = checkpoint_data
            self._latest_checkpoint_id[conv_id] = checkpoint_id

        # Register the request_id → checkpoint_id mapping AFTER the write
        # succeeds, so callers can rely on "if the index has the entry, the
        # checkpoint is (at least in this process) durable".
        if request_id:
            self.attach_request_id(request_id, checkpoint_id)

        return checkpoint_id

    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        async with self.lock(conv_id):
            bucket = self._checkpoints.get(conv_id)
            if not bucket:
                return None
            # Return the checkpoint with the highest `updated_at` — matches
            # the SQL store's `ORDER BY updated_at DESC, created_at DESC`.
            # Using `_latest_checkpoint_id` alone would go stale after an
            # `update_checkpoint_status` on an older checkpoint.
            latest = max(bucket.values(), key=lambda cp: cp.get("updated_at", 0))
            return _copy_checkpoint_for_return(latest)

    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        async with self.lock(conv_id):
            bucket = self._checkpoints.get(conv_id, {})
            cp = bucket.get(checkpoint_id)
            if cp is None:
                return None
            return _copy_checkpoint_for_return(cp)

    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        if status not in VALID_CHECKPOINT_STATUSES:
            log.warning(
                "update_checkpoint_status: invalid status %r (conv=%s, cp=%s)",
                status, conv_id, checkpoint_id,
            )
            return False

        async with self.lock(conv_id):
            bucket = self._checkpoints.get(conv_id, {})
            cp = bucket.get(checkpoint_id)
            if cp is None:
                return False
            current = cp.get("status", "in_progress")
            if _resolve_status_transition(current, status) is None:
                log.warning(
                    "update_checkpoint_status: illegal transition %s → %s "
                    "(conv=%s, cp=%s)",
                    current, status, conv_id, checkpoint_id,
                )
                return False
            cp["status"] = status
            cp["updated_at"] = time.time()
            # Keep _latest_checkpoint_id aligned with "most recently touched"
            # so the fast path (used by any future caller) stays correct.
            self._latest_checkpoint_id[conv_id] = checkpoint_id
            return True


# ------------------------------------------------------------------------------
# Base SQL Store
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
        buffer_retry_attempts: int = 3,
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

        self._write_buffer: deque = deque()
        self._buffer_size = write_buffer_size
        self._flush_interval = write_flush_interval
        self._buffer_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

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

    async def _migrate_checkpoint_schema(self) -> None:
        """Add `original_prompt` / `merged_history` columns to existing DBs.

        Called by each backend's `_init_pool` AFTER `_create_tables` (which
        is a no-op on an existing table thanks to `CREATE TABLE IF NOT
        EXISTS`). Default implementation is a no-op; each backend
        overrides it with the appropriate `ALTER TABLE` dialect.
        """
        return None

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
        try:
            await self._fetchone("SELECT 1", ())
            return True
        except Exception:
            return False

    # ── Public API ────────────────────────────────────────────────────────
    async def get(self, conv_id: str) -> Optional[List[dict]]:
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

    # ── Checkpoint methods (shared SQL logic) ────────────────────────────
    async def save_checkpoint(self, conv_id: str, checkpoint: dict) -> str:
        norm = _validate_checkpoint_payload(checkpoint)
        request_id = norm.pop("_request_id", None)

        now = time.time()
        # Ensure the conversation row exists — checkpoints can be saved
        # before any message. Committed directly (not buffered) so the FK
        # is satisfied for the checkpoint INSERT below.
        upsert = """
            INSERT INTO conversations (id, created_at, updated_at, message_count)
            VALUES (?, ?, ?, 0)
            ON CONFLICT (id) DO NOTHING
        """
        await self._execute(upsert, (conv_id, now, now), commit=True)

        checkpoint_id = str(uuid.uuid4())
        merged_history_json: Optional[str] = None
        if norm["merged_history"] is not None:
            try:
                merged_history_json = json.dumps(
                    norm["merged_history"], ensure_ascii=False
                )
            except (TypeError, ValueError) as exc:
                log.warning(
                    "save_checkpoint: could not serialise merged_history "
                    "for conv=%s request=%s: %s — dropping the field",
                    conv_id, request_id, exc,
                )
                merged_history_json = None

        query = """
            INSERT INTO generation_checkpoints
            (checkpoint_id, conv_id, model_id, segment_id, accumulated_text,
             tokens_generated, status, original_prompt, merged_history,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            checkpoint_id,
            conv_id,
            norm["model_id"],
            norm["segment_id"],
            norm["accumulated_text"],
            norm["tokens_generated"],
            norm["status"],
            norm["original_prompt"],
            merged_history_json,
            now,
            now,
        )
        # DIRECT write, not buffered — see module docstring §3. The caller
        # (`api_server._save_checkpoint_callback`) needs to know whether the
        # checkpoint actually landed, and any subsequent read must see it.
        await self._execute(query, params, commit=True)

        if request_id:
            self.attach_request_id(request_id, checkpoint_id)

        return checkpoint_id

    async def get_latest_checkpoint(self, conv_id: str) -> Optional[dict]:
        query = """
            SELECT checkpoint_id, model_id, segment_id, accumulated_text,
                   tokens_generated, status, original_prompt, merged_history,
                   created_at, updated_at
            FROM generation_checkpoints
            WHERE conv_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
        """
        row = await self._fetchone(query, (conv_id,))
        if not row:
            return None
        return self._checkpoint_row_to_dict(row)

    async def get_checkpoint_by_id(self, conv_id: str, checkpoint_id: str) -> Optional[dict]:
        query = """
            SELECT checkpoint_id, model_id, segment_id, accumulated_text,
                   tokens_generated, status, original_prompt, merged_history,
                   created_at, updated_at
            FROM generation_checkpoints
            WHERE conv_id = ? AND checkpoint_id = ?
        """
        row = await self._fetchone(query, (conv_id, checkpoint_id))
        if not row:
            return None
        return self._checkpoint_row_to_dict(row)

    @staticmethod
    def _checkpoint_row_to_dict(row: tuple) -> dict:
        """Convert a `generation_checkpoints` SELECT row to the canonical
        checkpoint dict shape returned by this API.

        Column order is fixed in the SELECT statements above; if you add a
        column, update both SELECTs and this method in one change.
        """
        original_prompt = row[6]
        merged_history_raw = row[7]
        merged_history: Optional[List[dict]] = None
        if merged_history_raw:
            try:
                loaded = json.loads(merged_history_raw)
                if isinstance(loaded, list):
                    merged_history = loaded
            except (json.JSONDecodeError, TypeError):
                log.warning(
                    "get_checkpoint_by_id: merged_history for checkpoint %s "
                    "is not valid JSON — returning None",
                    row[0],
                )
                merged_history = None

        return {
            "checkpoint_id":    row[0],
            "model_id":         row[1],
            "segment_id":       row[2],
            "accumulated_text": row[3],
            "tokens_generated": row[4],
            "status":           row[5],
            "original_prompt":  original_prompt,
            "merged_history":   merged_history,
            "created_at":       row[8],
            "updated_at":       row[9],
        }

    async def update_checkpoint_status(self, conv_id: str, checkpoint_id: str, status: str) -> bool:
        if status not in VALID_CHECKPOINT_STATUSES:
            log.warning(
                "update_checkpoint_status: invalid status %r (conv=%s, cp=%s)",
                status, conv_id, checkpoint_id,
            )
            return False

        # Read current status first so we can validate the transition
        # (cheaper than attempting an UPDATE that silently no-ops, and lets
        # us emit a clear diagnostic for illegal transitions).
        current_row = await self._fetchone(
            "SELECT status FROM generation_checkpoints "
            "WHERE conv_id = ? AND checkpoint_id = ?",
            (conv_id, checkpoint_id),
        )
        if not current_row:
            return False
        current_status = current_row[0] or "in_progress"
        if _resolve_status_transition(current_status, status) is None:
            log.warning(
                "update_checkpoint_status: illegal transition %s → %s "
                "(conv=%s, cp=%s)",
                current_status, status, conv_id, checkpoint_id,
            )
            return False

        # Same-status: skip the write entirely (idempotent no-op).
        if current_status == status:
            return True

        # DIRECT write, not buffered — the previous buffered version raced
        # the caller's subsequent read (`api_server.chat_resume` calls this
        # then re-reads via /v1/requests/{id}/stats if any). See docstring §3.
        await self._execute(
            "UPDATE generation_checkpoints "
            "SET status = ?, updated_at = ? "
            "WHERE conv_id = ? AND checkpoint_id = ?",
            (status, time.time(), conv_id, checkpoint_id),
            commit=True,
        )
        return True


# ------------------------------------------------------------------------------
# SQLite Implementation
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
        await self._migrate_checkpoint_schema()
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
        # ── Checkpoint table (original_prompt + merged_history) ─────
        await self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                conv_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                segment_id INTEGER NOT NULL,
                accumulated_text TEXT NOT NULL,
                tokens_generated INTEGER NOT NULL,
                status TEXT NOT NULL,
                original_prompt TEXT,
                merged_history TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
        await self._db_conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")
        await self._db_conn.commit()

    async def _migrate_checkpoint_schema(self) -> None:
        """Add columns to generation_checkpoints tables."""
        cursor = await self._db_conn.execute("PRAGMA table_info(generation_checkpoints)")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        alters = []
        if "original_prompt" not in existing:
            alters.append("ALTER TABLE generation_checkpoints ADD COLUMN original_prompt TEXT")
        if "merged_history" not in existing:
            alters.append("ALTER TABLE generation_checkpoints ADD COLUMN merged_history TEXT")
        for stmt in alters:
            await self._db_conn.execute(stmt)
        if alters:
            await self._db_conn.commit()
            log.info(
                "Migrated generation_checkpoints schema (SQLite): added %d column(s)",
                len(alters),
            )

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
# PostgreSQL Implementation
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
        await self._migrate_checkpoint_schema()
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_checkpoints (
                    checkpoint_id VARCHAR(36) PRIMARY KEY,
                    conv_id VARCHAR(255) NOT NULL,
                    model_id VARCHAR(255) NOT NULL,
                    segment_id INTEGER NOT NULL,
                    accumulated_text TEXT NOT NULL,
                    tokens_generated INTEGER NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    original_prompt TEXT,
                    merged_history TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")

    async def _migrate_checkpoint_schema(self) -> None:
        """PostgreSQL's ADD COLUMN IF NOT EXISTS is idempotent — no PRAGMA needed."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "ALTER TABLE generation_checkpoints "
                "ADD COLUMN IF NOT EXISTS original_prompt TEXT"
            )
            await conn.execute(
                "ALTER TABLE generation_checkpoints "
                "ADD COLUMN IF NOT EXISTS merged_history TEXT"
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
# MySQL Implementation
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
        await self._migrate_checkpoint_schema()
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
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS generation_checkpoints (
                        checkpoint_id VARCHAR(36) PRIMARY KEY,
                        conv_id VARCHAR(255) NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        segment_id INTEGER NOT NULL,
                        accumulated_text TEXT NOT NULL,
                        tokens_generated INTEGER NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        original_prompt TEXT,
                        merged_history TEXT,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_conv ON generation_checkpoints(conv_id)")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON generation_checkpoints(updated_at)")

    async def _migrate_checkpoint_schema(self) -> None:
        """MySQL lacks ADD COLUMN IF NOT EXISTS pre-8.0.29 — use SHOW COLUMNS."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SHOW COLUMNS FROM generation_checkpoints LIKE 'original_prompt'"
                )
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE generation_checkpoints "
                        "ADD COLUMN original_prompt TEXT"
                    )
                await cur.execute(
                    "SHOW COLUMNS FROM generation_checkpoints LIKE 'merged_history'"
                )
                if not await cur.fetchone():
                    await cur.execute(
                        "ALTER TABLE generation_checkpoints "
                        "ADD COLUMN merged_history TEXT"
                    )

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
# Factory Function
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
    "VALID_CHECKPOINT_STATUSES",
    "attach_request_id",
    "get_checkpoint_id_for_request",
]
