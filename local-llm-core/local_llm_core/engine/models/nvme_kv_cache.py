from __future__ import annotations

"""
engine/models/nvme_kv_cache.py — Production‑Ready NVMe KV State Cache
=====================================================================
v1.0.0 – Config injected via constructor; library‑safe.

Persistent prefix cache on NVMe SSD for KV states.
All parameters are provided through the `config` argument (dict or
NVMeKVCacheConfig model), with optional constructor overrides.

Features:
- Atomic writes (temp file + os.replace)
- Sharded directory structure to avoid inode exhaustion
- O(1) LRU eviction using OrderedDict
- Optional Prometheus metrics
- Graceful handling of stale or corrupted entries
- Model‑specific invalidation
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Prometheus Metrics (optional, non‑invasive)
# -----------------------------------------------------------------------------
def _setup_metrics() -> Dict[str, Any]:
    try:
        from prometheus_client import Counter, Gauge
        return {
            "hits": Counter("llm_kvcache_hits_total", "KV cache hits"),
            "misses": Counter("llm_kvcache_misses_total", "KV cache misses"),
            "evictions": Counter("llm_kvcache_evictions_total", "KV cache evictions"),
            "size_bytes": Gauge("llm_kvcache_size_bytes", "Current cache size in bytes"),
            "entries": Gauge("llm_kvcache_entries", "Number of cached entries"),
        }
    except ImportError:
        return {}

_METRICS = _setup_metrics()


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class KVCacheEntry:
    """Metadata for one cached KV state."""
    key: str
    model_id: str
    prompt_len: int
    size_bytes: int
    created_ts: float
    last_used_ts: float
    hit_count: int = 0


# -----------------------------------------------------------------------------
# Main Cache Class
# -----------------------------------------------------------------------------
class NVMeKVCache:
    """
    Persistent KV state cache on NVMe SSD.

    Configuration is taken from the `config` argument (a dict or
    NVMeKVCacheConfig model).  Constructor parameters `cache_dir`, `max_gb`,
    `max_entries`, `prefix_len`, and `enabled` can override config values.

    The cache uses atomic writes and an in‑memory index with LRU eviction
    for O(1) operations.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        max_gb: Optional[float] = None,
        max_entries: Optional[int] = None,
        prefix_len: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Normalise configuration
        # ------------------------------------------------------------------
        cfg: Dict[str, Any] = {}
        if config is not None:
            try:
                # Prefer the central Pydantic model for validation
                from local_llm_core.engine.utils.config_loader import NVMeKVCacheConfig
                if isinstance(config, dict):
                    cfg = NVMeKVCacheConfig(**config).model_dump()
                else:
                    cfg = config.model_dump()
            except Exception as exc:
                log.warning("NVMe KV cache config validation failed: %s; using defaults", exc)
                cfg = {}  # will rely on constructor overrides and defaults

        # Apply values in order: explicit kwargs > config > hardcoded default
        self._dir = Path(
            cache_dir or cfg.get("dir", "/tmp/llm_kv_cache")
        )
        max_gb_val = max_gb if max_gb is not None else cfg.get("max_size_gb", 10.0)
        self._max_bytes = int(max_gb_val * 1024 ** 3)
        self._max_entries = max_entries if max_entries is not None else cfg.get("max_entries", 0)
        self._pfx_len = prefix_len if prefix_len is not None else cfg.get("prefix_len", 512)
        self._enabled = enabled if enabled is not None else cfg.get("enabled", True)

        self._lock = threading.RLock()
        self._index: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._used_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        if self._enabled:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._load_index()
                log.info(
                    "NVMeKVCache: %s (max=%.1f GB, %d entries, %.1f MB used)",
                    self._dir,
                    max_gb_val,
                    len(self._index),
                    self._used_bytes / 1024 ** 2,
                )
            except Exception as exc:
                log.warning("NVMeKVCache init failed: %s — disabling", exc)
                self._enabled = False

    # -------------------------------------------------------------------------
    # Key management
    # -------------------------------------------------------------------------
    def _make_key(self, model_id: str, prompt: str) -> str:
        """SHA‑256 of (model_id + ':' + prompt[:prefix_len])."""
        raw = f"{model_id}:{prompt[:self._pfx_len]}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _key_path(self, key: str) -> Path:
        """Directory‑sharded path: <cache>/XX/fullhash.kv"""
        return self._dir / key[:2] / f"{key}.kv"

    # -------------------------------------------------------------------------
    # Public API – get / put / invalidate
    # -------------------------------------------------------------------------
    def get(self, model_id: str, prompt: str) -> Optional[bytes]:
        """
        Return cached KV state bytes for a model + prompt prefix.
        Returns None on cache miss or if disabled.
        """
        if not self._enabled:
            return None
        key = self._make_key(model_id, prompt)
        with self._lock:
            if key not in self._index:
                self._misses += 1
                _METRICS.get("misses", lambda: None) and _METRICS["misses"].inc()  # type: ignore[func-returns-value]
                return None
            entry = self._index[key]
            self._index.move_to_end(key)  # mark most recently used

        path = self._key_path(key)
        try:
            data = path.read_bytes()
            with self._lock:
                entry.last_used_ts = time.time()
                entry.hit_count += 1
                self._hits += 1
            if _METRICS.get("hits"):
                _METRICS["hits"].inc()
            log.debug(
                "KVCache HIT: %s prefix=%d size=%.1f MB (hit #%d)",
                model_id, len(prompt[:self._pfx_len]), len(data) / 1024 ** 2, entry.hit_count,
            )
            return data
        except Exception as exc:
            log.debug("KVCache read error for %s: %s", key[:8], exc)
            with self._lock:
                self._misses += 1
                if _METRICS.get("misses"):
                    _METRICS["misses"].inc()
                # Remove stale entry
                if key in self._index:
                    self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
                    del self._index[key]
            return None

    def put(self, model_id: str, prompt: str, state_bytes: bytes) -> bool:
        """
        Atomically store KV state bytes. Returns True on success.
        Already existing entries have their timestamp updated.
        """
        if not self._enabled:
            return False
        key = self._make_key(model_id, prompt)
        size = len(state_bytes)

        with self._lock:
            if key in self._index:
                entry = self._index[key]
                entry.last_used_ts = time.time()
                self._index.move_to_end(key)
                log.debug("KVCache PUT: key %s already exists, updated timestamp", key[:8])
                return True

        # Create directory if missing
        path = self._key_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.debug("KVCache mkdir failed for %s: %s", key[:8], exc)
            return False

        # Atomic data write
        temp_fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".kv")
        try:
            with os.fdopen(temp_fd, "wb") as f:
                f.write(state_bytes)
            os.replace(temp_path, path)
        except Exception as exc:
            os.unlink(temp_path)
            log.debug("KVCache write error: %s", exc)
            return False

        # Atomic metadata write
        meta = {
            "key": key,
            "model_id": model_id,
            "prompt_len": len(prompt[:self._pfx_len]),
            "size_bytes": size,
            "created_ts": time.time(),
            "last_used_ts": time.time(),
        }
        meta_path = path.with_suffix(".meta")
        temp_meta_fd, temp_meta_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".meta")
        try:
            with os.fdopen(temp_meta_fd, "w") as f:
                json.dump(meta, f)
            os.replace(temp_meta_path, meta_path)
        except Exception as exc:
            os.unlink(temp_meta_path)
            # Rollback kv file
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            log.debug("KVCache meta write error: %s", exc)
            return False

        # Insert into index with eviction if necessary
        with self._lock:
            if key in self._index:
                # Another thread beat us – our file will be orphaned but that's harmless
                log.debug("KVCache PUT: key %s inserted concurrently, discarding duplicate", key[:8])
                return True

            self._evict_to_fit(size)
            if self._used_bytes + size > self._max_bytes:
                # Still no space
                remove_temp_kv(path)
                remove_temp_meta(meta_path)
                log.debug("KVCache: no space for %.1f MB entry", size / 1024 ** 2)
                return False
            if self._max_entries > 0 and len(self._index) >= self._max_entries:
                self._evict_one()
                if len(self._index) >= self._max_entries:
                    remove_temp_kv(path)
                    remove_temp_meta(meta_path)
                    log.debug("KVCache: max entries reached")
                    return False

            entry = KVCacheEntry(
                key=key,
                model_id=model_id,
                prompt_len=meta["prompt_len"],
                size_bytes=size,
                created_ts=meta["created_ts"],
                last_used_ts=meta["last_used_ts"],
            )
            self._index[key] = entry
            self._index.move_to_end(key)  # mark as most recent
            self._used_bytes += size

        # Update Prometheus gauges
        if _METRICS.get("size_bytes"):
            _METRICS["size_bytes"].set(self._used_bytes)
        if _METRICS.get("entries"):
            _METRICS["entries"].set(len(self._index))

        log.debug(
            "KVCache PUT: %s size=%.1f MB total=%.1f MB",
            model_id, size / 1024 ** 2, self._used_bytes / 1024 ** 2,
        )
        return True

    def invalidate_model(self, model_id: str) -> int:
        """Remove all cached entries for the given model. Returns number of removed entries."""
        with self._lock:
            keys_to_del = [k for k, e in self._index.items() if e.model_id == model_id]
            freed = 0
            for key in keys_to_del:
                entry = self._index.pop(key, None)
                if entry:
                    freed += entry.size_bytes
                    self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
                    try:
                        self._key_path(key).unlink(missing_ok=True)
                        self._key_path(key).with_suffix(".meta").unlink(missing_ok=True)
                    except Exception:
                        pass
            if _METRICS.get("size_bytes"):
                _METRICS["size_bytes"].set(self._used_bytes)
            if _METRICS.get("entries"):
                _METRICS["entries"].set(len(self._index))
        if freed:
            log.info("KVCache: invalidated %d entries for %s (freed %.1f MB)",
                     len(keys_to_del), model_id, freed / 1024 ** 2)
        return len(keys_to_del)

    # -------------------------------------------------------------------------
    # LRU Eviction (O(1) using OrderedDict)
    # -------------------------------------------------------------------------
    def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict least recently used entries until `needed_bytes` will fit. Must hold lock."""
        if self._used_bytes + needed_bytes <= self._max_bytes:
            return
        keys_to_remove = []
        for key, entry in self._index.items():
            if self._used_bytes + needed_bytes <= self._max_bytes:
                break
            keys_to_remove.append(key)
            self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
        for key in keys_to_remove:
            entry = self._index.pop(key)
            self._remove_on_disk(key)
            self._evictions += 1
            if _METRICS.get("evictions"):
                _METRICS["evictions"].inc()

    def _evict_one(self) -> None:
        """Evict the single least recently used entry. Must hold lock."""
        if not self._index:
            return
        key, entry = next(iter(self._index.items()))
        self._remove_on_disk(key)
        self._used_bytes = max(0, self._used_bytes - entry.size_bytes)
        del self._index[key]
        self._evictions += 1
        if _METRICS.get("evictions"):
            _METRICS["evictions"].inc()

    def _remove_on_disk(self, key: str) -> None:
        """Delete .kv and .meta files for a key, ignoring errors."""
        try:
            self._key_path(key).unlink(missing_ok=True)
            self._key_path(key).with_suffix(".meta").unlink(missing_ok=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def _load_index(self) -> None:
        """Rebuild the in‑memory index by scanning all .meta files."""
        count = 0
        total_bytes = 0
        for meta_path in self._dir.rglob("*.meta"):
            try:
                meta = json.loads(meta_path.read_text())
                key = meta["key"]
                kv_path = meta_path.with_suffix(".kv")
                if not kv_path.exists():
                    meta_path.unlink(missing_ok=True)
                    continue
                stat = kv_path.stat()
                size = meta.get("size_bytes", stat.st_size)
                if size != stat.st_size:
                    log.debug("KVCache: size mismatch for %s, using file size", key[:8])
                    size = stat.st_size
                entry = KVCacheEntry(
                    key=key,
                    model_id=meta.get("model_id", "unknown"),
                    prompt_len=meta.get("prompt_len", 0),
                    size_bytes=size,
                    created_ts=meta.get("created_ts", 0.0),
                    last_used_ts=meta.get("last_used_ts", 0.0),
                )
                self._index[key] = entry
                total_bytes += size
                count += 1
            except Exception as exc:
                log.debug("KVCache: skipping corrupted meta %s: %s", meta_path, exc)

        # Restore LRU order (oldest first)
        sorted_items = sorted(self._index.items(), key=lambda kv: kv[1].last_used_ts)
        self._index = OrderedDict(sorted_items)
        self._used_bytes = total_bytes
        log.debug("KVCache index loaded: %d entries, %.1f MB", count, total_bytes / 1024 ** 2)

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all cached entries (data + metadata)."""
        with self._lock:
            for key in list(self._index):
                self._remove_on_disk(key)
            self._index.clear()
            self._used_bytes = 0
            self._evictions = 0
            self._hits = 0
            self._misses = 0
        if _METRICS.get("size_bytes"):
            _METRICS["size_bytes"].set(0)
        if _METRICS.get("entries"):
            _METRICS["entries"].set(0)
        log.info("NVMeKVCache: cleared")

    def cleanup_orphans(self) -> int:
        """Remove .kv files without .meta and vice versa. Returns number of removed files."""
        removed = 0
        for kv_path in self._dir.rglob("*.kv"):
            if not kv_path.with_suffix(".meta").exists():
                kv_path.unlink(missing_ok=True)
                removed += 1
        for meta_path in self._dir.rglob("*.meta"):
            if not meta_path.with_suffix(".kv").exists():
                meta_path.unlink(missing_ok=True)
                removed += 1
        if removed:
            log.info("KVCache cleanup: removed %d orphaned files", removed)
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": self._enabled,
                "entries": len(self._index),
                "used_mb": round(self._used_bytes / 1024 ** 2, 1),
                "max_gb": round(self._max_bytes / 1024 ** 3, 1),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
                "evictions": self._evictions,
            }


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def remove_temp_kv(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def remove_temp_meta(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
