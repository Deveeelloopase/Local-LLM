from __future__ import annotations

"""
memory/swap_handler.py — Production‑Ready NVMe Swap Handler
===========================================================
– Config injected via constructor; no file I/O at import.

NVMe SSD‑backed swap for model layers with:
- Atomic writes (fsync before rename)
- LZ4 compression (configurable level)
- CRC32 integrity checks
- LRU eviction
- Persistent index recovery on restart

All parameters are taken from the `config` dict (or a `SwapConfig`
model). Sensible defaults are used when config is omitted.

version = 1.0.0
"""

import asyncio
import hashlib
import logging
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants (header format, magic, etc.)
# -----------------------------------------------------------------------------
_MAGIC = b"LLMSWP\x02"          # 7 bytes + version byte
_HDR_BASE = "7s B Q I"          # magic, flags, payload_size, crc32 (20 B)
_HDR_KEY = "I"                  # key_len (4 B)
_BASE_SIZE = struct.calcsize(_HDR_BASE)
_KEY_LEN_SIZE = struct.calcsize(_HDR_KEY)

_F_COMPRESS = 0x01


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _pack_header(payload: bytes, flags: int, key: str) -> bytes:
    """Pack full header (magic + flags + size + crc + key_len + key)."""
    key_b = key.encode("utf-8", errors="replace")
    base_hdr = struct.pack(_HDR_BASE, _MAGIC, flags, len(payload), _crc32(payload))
    key_hdr = struct.pack(_HDR_KEY, len(key_b))
    return base_hdr + key_hdr + key_b


def _unpack_header(raw: bytes) -> Optional[Tuple[int, bytes, str]]:
    """
    Returns (flags, payload_bytes, original_key) or None on parse failure.
    Handles both v1 (no key) and v2 (with key) headers.
    """
    if len(raw) < _BASE_SIZE:
        log.warning("Swap header too short: %d bytes", len(raw))
        return None
    try:
        magic, flags, size, crc = struct.unpack(_HDR_BASE, raw[:_BASE_SIZE])
    except struct.error as e:
        log.error("Swap header unpack failed: %s", e)
        return None
    if magic != _MAGIC:
        log.warning("Swap invalid magic: %s (expected %s)", magic.hex(), _MAGIC.hex())
        return None

    payload_start = _BASE_SIZE
    original_key = ""

    if len(raw) >= _BASE_SIZE + _KEY_LEN_SIZE:
        try:
            key_len = struct.unpack(_HDR_KEY, raw[_BASE_SIZE:_BASE_SIZE + _KEY_LEN_SIZE])[0]
            key_start = _BASE_SIZE + _KEY_LEN_SIZE
            if key_len > 0 and len(raw) >= key_start + key_len:
                original_key = raw[key_start:key_start + key_len].decode(
                    "utf-8", errors="replace",
                )
                payload_start = key_start + key_len
        except (struct.error, UnicodeDecodeError) as e:
            log.error("Swap key parsing failed: %s", e)

    payload = raw[payload_start:]
    if _crc32(payload) != crc:
        log.error("Swap CRC mismatch for key '%s'", original_key or "unknown")
        return None

    if flags & _F_COMPRESS:
        try:
            import lz4.frame as lz4
            payload = lz4.decompress(payload)
        except Exception as exc:
            log.error("Swap decompression failed: %s", exc)
            return None

    return flags, payload, original_key


# -----------------------------------------------------------------------------
# SwapHandler
# -----------------------------------------------------------------------------
class SwapHandler:
    """
    Async SSD swap. Keys map to flat files under swap_dir.
    All configuration is injected via the `config` argument.

    Thread safety:
      * `_write_lock` (asyncio.Lock) serialises `write()` operations so
        two writers cannot race the same atomic-rename target.
      * `_index_lock` (threading.RLock) guards `_index` and `_used`.
        Every dict op and every arithmetic on `_used` happens under this
        lock; file I/O happens outside it. This gives every reader
        (`stats`, `has`, `read`, `delete`) a coherent view of the index
        without ever blocking the event loop on a filesystem call.
      * Lock order is strictly `_write_lock` → `_index_lock` → release.
        Never the reverse. This lets a `write()` holding `_write_lock`
        call `delete()` (which briefly takes `_index_lock`) without
        deadlocking against an external caller that only touches
        `_index_lock`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # ---- normalise config ------------------------------------------------
        mem_cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import SwapConfig
                if isinstance(config, dict):
                    cfg_model = SwapConfig(**config.get("memory", {}))
                else:
                    cfg_model = SwapConfig(**config.get("memory", {}))
                mem_cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("Swap config validation failed: %s; using defaults", exc)
                mem_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        swap_path = mem_cfg.get("ssd_swap_path", "/tmp/llm_swap")
        self.swap_dir = Path(swap_path).resolve()
        self.max_bytes = int(mem_cfg.get("ssd_swap_max_gb", 32.0) * (1024**3))
        self._compress = mem_cfg.get("ssd_compression_enabled", True)
        self._compression_level = mem_cfg.get("ssd_compression_level", 0)

        self._index: Dict[str, Dict[str, Any]] = {}
        self._used = 0
        # asyncio.Lock is not safe to construct before an event loop
        # exists — keep the lazy pattern but note it must only be
        # constructed on the loop. `_get_write_lock` handles that.
        self._write_lock: Optional[asyncio.Lock] = None
        # RLock because it's acquired from both sync
        # (`stats`, `has`, `shutdown`) and async (`write`, `read`, `delete`)
        # contexts. Held only for dict ops — never across an await.
        self._index_lock = threading.RLock()

        try:
            import lz4.frame  # noqa: F401
            self._lz4_available = True
        except ImportError:
            self._lz4_available = False
            log.warning("SwapHandler: lz4 not available — compression disabled")

        self.swap_dir.mkdir(parents=True, exist_ok=True)
        self._scan_existing()

        log.info(
            "SwapHandler: dir=%s max=%.1fGB compress=%s recovered=%d entries",
            self.swap_dir,
            self.max_bytes / (1024**3),
            self._compress and self._lz4_available,
            len(self._index),
        )

    def _get_write_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def write(self, key: str, data: bytes) -> bool:
        """Store data atomically on SSD. Returns True on success."""
        compressed, flags = self._maybe_compress(data)
        header = _pack_header(compressed, flags, key)
        payload = header + compressed
        size = len(compressed)

        async with self._get_write_lock():
            # Try to free room. `_evict_lru_locked` iterates under
            # `_index_lock` for each candidate, releasing it to do the
            # actual unlink — see its docstring.
            await self._evict_lru_locked(size)

            with self._index_lock:
                if self._used + size > self.max_bytes:
                    log.warning("Swap quota exceeded for '%s'", key)
                    return False

            # Atomic file write — outside the index lock so a slow
            # filesystem does not block every concurrent index read.
            path = self._key_path(key)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _atomic_write, path, payload)

            # Register in the index — under the lock, so a concurrent
            # `stats()` / `has()` sees either the pre-write state or the
            # post-write state, never a mid-update.
            with self._index_lock:
                if key in self._index:
                    self._used -= self._index[key]["size"]
                self._index[key] = {
                    "path": str(path),
                    "size": size,
                    "ts": time.monotonic(),
                }
                self._used += size
            return True

    async def read(self, key: str) -> Optional[bytes]:
        """Read data from SSD. Returns None if not found or corrupted.

        when the index has an entry but the file has vanished
        (external `rm`, cleanup script, or unlink failure elsewhere),
        the entry is reaped **and `_used` is decremented**. The previous
        code popped the index entry without decrementing `_used`,
        permanently leaking quota until process restart.
        """
        with self._index_lock:
            meta = self._index.get(key)
            if not meta:
                return None
            path_str = meta["path"]

        path = Path(path_str)
        if not path.exists():
            with self._index_lock:
                cur = self._index.get(key)
                # Re-check under lock: another thread may have replaced
                # the entry (rewritten the key) between our exists() call
                # and now. Only reap if it's still the same stale entry.
                if cur is not None and cur["path"] == path_str:
                    self._index.pop(key, None)
                    self._used = max(0, self._used - cur["size"])
            return None

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, path.read_bytes)
        result = _unpack_header(raw)
        if result is None:
            return None
        _, payload, _ = result
        with self._index_lock:
            if key in self._index:
                self._index[key]["ts"] = time.monotonic()
        return payload

    async def delete(self, key: str) -> bool:
        """Remove entry from SSD. Returns True if existed."""
        with self._index_lock:
            meta = self._index.pop(key, None)
            if not meta:
                return False
            self._used = max(0, self._used - meta["size"])

        # File I/O outside the index lock — unlink is fast but can
        # occasionally block on contended filesystems and we don't want
        # to serialise every concurrent reader behind it.
        try:
            Path(meta["path"]).unlink(missing_ok=True)
        except OSError:
            pass
        return True

    def has(self, key: str) -> bool:
        with self._index_lock:
            return key in self._index

    def stats(self) -> Dict[str, Any]:
        """Return a coherent snapshot of swap state.

        every field is read under the same `_index_lock`
        hold. Previously `_used` and `len(self._index)` were read
        unlocked, so a concurrent `write()` mid-update could yield a
        snapshot where `used_mb` and `num_entries` disagree.
        """
        with self._index_lock:
            return {
                "swap_dir": str(self.swap_dir),
                "max_gb": round(self.max_bytes / 1024**3, 2),
                "used_mb": round(self._used / (1024**2), 2),
                "free_mb": round((self.max_bytes - self._used) / (1024**2), 2),
                "num_entries": len(self._index),
                "utilization": round(
                    self._used / max(self.max_bytes, 1) * 100, 2,
                ),
            }

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    def _key_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.swap_dir / f"{h}.llmswap"

    def _maybe_compress(self, data: bytes) -> Tuple[bytes, int]:
        if self._compress and self._lz4_available:
            try:
                import lz4.frame as lz4
                c = lz4.compress(data, compression_level=self._compression_level)
                if len(c) < len(data):
                    return c, _F_COMPRESS
            except Exception:
                pass
        return data, 0

    async def _evict_lru_locked(self, needed: int) -> None:
        """Evict LRU entries until `needed` bytes fit. Caller MUST hold _write_lock.

        index reads are wrapped in `_index_lock` and the
        `_used + needed <= max_bytes` check runs under the same hold.
        The previous version iterated `self._index` with no lock — a
        concurrent `delete()` from an external caller (or from `write()`
        itself finishing for a different key) could remove an entry
        mid-loop, producing a stale `oldest_key` and a wasted delete
        attempt that silently failed while the loop thought it had
        freed space.

        Returns silently when either enough room is available or there
        is nothing left to evict — the caller's quota check produces
        the correct return value in that case.
        """
        while True:
            with self._index_lock:
                if self._used + needed <= self.max_bytes:
                    return
                if not self._index:
                    # Nothing to free — caller will see quota exceeded.
                    return
                oldest_key = min(
                    self._index, key=lambda k: self._index[k]["ts"],
                )
            # delete() re-acquires `_index_lock` briefly for the pop,
            # then does I/O outside the lock.
            await self.delete(oldest_key)

    def _scan_existing(self) -> None:
        """
        Rebuild index from existing files.
        v2 files embed the original key; v1 files fall back to stem.
        Corrupted files are skipped.
        """
        for f in self.swap_dir.glob("*.llmswap"):
            try:
                raw = f.read_bytes()
                result = _unpack_header(raw)
                if result is None:
                    continue
                _, _, original_key = result
                index_key = original_key if original_key else f.stem
                try:
                    _, _, size, _ = struct.unpack(_HDR_BASE, raw[:_BASE_SIZE])
                except struct.error:
                    size = f.stat().st_size
                with self._index_lock:
                    self._index[index_key] = {
                        "path": str(f),
                        "size": size,
                        "ts": f.stat().st_mtime,
                    }
                    self._used += size
            except Exception as exc:
                log.debug("Swap scan skipped %s: %s", f, exc)

    def shutdown(self) -> None:
        """Release resources; files persist for next start."""
        with self._index_lock:
            n = len(self._index)
            used = self._used
            self._index.clear()
            self._used = 0
        log.info(
            "SwapHandler shutdown: %d entries, %.1f MB used",
            n,
            used / (1024**2),
        )


def _atomic_write(path: Path, data: bytes) -> None:
    """Atomic write with fsync for durability (power‑loss safe)."""
    import os
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        tmp.rename(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
