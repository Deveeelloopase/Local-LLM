from __future__ import annotations

"""
memory/swap_handler.py — Production‑Ready NVMe Swap Handler
===========================================================
v1.0.0 – Config injected via constructor; no file I/O.

NVMe SSD‑backed swap for model layers with:
- Atomic writes (fsync before rename)
- LZ4 compression (configurable level)
- CRC32 integrity checks
- LRU eviction
- Persistent index recovery on restart

All parameters are taken from the `config` dict (or a `SwapConfig`
model).  Sensible defaults are used when config is omitted.
"""

import asyncio
import hashlib
import logging
import struct
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
_BASE_SIZE = struct.calcsize(_HDR_BASE)       # 20
_KEY_LEN_SIZE = struct.calcsize(_HDR_KEY)      # 4

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
        log.debug("Swap header too short: %d bytes", len(raw))
        return None
    try:
        magic, flags, size, crc = struct.unpack(_HDR_BASE, raw[:_BASE_SIZE])
    except struct.error as e:
        log.debug("Swap header unpack failed: %s", e)
        return None
    if magic != _MAGIC:
        log.debug("Swap invalid magic: %s (expected %s)", magic.hex(), _MAGIC.hex())
        return None

    payload_start = _BASE_SIZE
    original_key = ""

    if len(raw) >= _BASE_SIZE + _KEY_LEN_SIZE:
        try:
            key_len = struct.unpack(_HDR_KEY, raw[_BASE_SIZE:_BASE_SIZE + _KEY_LEN_SIZE])[0]
            key_start = _BASE_SIZE + _KEY_LEN_SIZE
            if key_len > 0 and len(raw) >= key_start + key_len:
                original_key = raw[key_start:key_start + key_len].decode("utf-8", errors="replace")
                payload_start = key_start + key_len
        except (struct.error, UnicodeDecodeError) as e:
            log.debug("Swap key parsing failed: %s", e)

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
        self._write_lock: Optional[asyncio.Lock] = None

        # LZ4 availability
        try:
            import lz4.frame
            self._lz4_available = True
        except ImportError:
            self._lz4_available = False
            log.debug("SwapHandler: lz4 not available — compression disabled")

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
            if self._used + size > self.max_bytes:
                await self._evict_lru(size)
            if self._used + size > self.max_bytes:
                log.warning("Swap quota exceeded for '%s'", key)
                return False

            path = self._key_path(key)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _atomic_write, path, payload)

            if key in self._index:
                self._used -= self._index[key]["size"]
            self._index[key] = {"path": str(path), "size": size, "ts": time.monotonic()}
            self._used += size
            return True

    async def read(self, key: str) -> Optional[bytes]:
        """Read data from SSD. Returns None if not found or corrupted."""
        meta = self._index.get(key)
        if not meta:
            return None
        path = Path(meta["path"])
        if not path.exists():
            self._index.pop(key, None)
            return None

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, path.read_bytes)
        result = _unpack_header(raw)
        if result is None:
            return None
        _, payload, _ = result
        meta["ts"] = time.monotonic()
        return payload

    async def delete(self, key: str) -> bool:
        """Remove entry from SSD. Returns True if existed."""
        meta = self._index.pop(key, None)
        if not meta:
            return False
        self._used -= meta["size"]
        try:
            Path(meta["path"]).unlink(missing_ok=True)
        except OSError:
            pass
        return True

    def has(self, key: str) -> bool:
        return key in self._index

    def stats(self) -> Dict[str, Any]:
        return {
            "swap_dir": str(self.swap_dir),
            "max_gb": round(self.max_bytes / 1024**3, 2),
            "used_mb": round(self._used / (1024**2), 2),
            "free_mb": round((self.max_bytes - self._used) / (1024**2), 2),
            "num_entries": len(self._index),
            "utilization": round(self._used / max(self.max_bytes, 1) * 100, 2),
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

    async def _evict_lru(self, needed: int) -> None:
        by_age = sorted(self._index, key=lambda k: self._index[k]["ts"])
        for k in by_age:
            if self._used + needed <= self.max_bytes:
                break
            await self.delete(k)

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
        log.info(
            "SwapHandler shutdown: %d entries, %.1f MB used",
            len(self._index),
            self._used / (1024**2),
        )
        self._index.clear()
        self._used = 0


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
