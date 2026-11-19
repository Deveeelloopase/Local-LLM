from __future__ import annotations

"""
memory/hybrid_manager.py — Production‑Ready Hybrid Memory Manager
=================================================================
– Config injected via constructor; no filesystem reads.

Three‑tier memory hierarchy: VRAM → RAM → SSD swap.
- LRU caching with promotion/demotion
- Atomic SSD writes with fsync and CRC validation
- LZ4 compression (configurable)
- Safe deserialization (pickle whitelist)
- Configurable promotion threshold

All parameters are taken from the `config` dict (or a `HybridConfig`
model).  Sensible defaults are used when config is omitted.

version = 1.0.0
"""

import gc
import hashlib
import io
import logging
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants (Header Format)
# -----------------------------------------------------------------------------
_MAGIC = b"HYB\x02"
_HDR_FMT = "!4s B H I I"  # big‑endian: magic, flags, key_len, crc32, payload_len
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # 15 bytes
_F_COMPRESSED = 0x01


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# -----------------------------------------------------------------------------
# Enums and Data Classes
# -----------------------------------------------------------------------------
class MemoryTier(Enum):
    VRAM = auto()
    RAM = auto()
    SSD = auto()


@dataclass
class HybridBlock:
    key: str
    tier: MemoryTier
    size_bytes: int  # actual bytes on‑tier (compressed size for SSD)
    data: Any  # tensor | bytes | str(path) for SSD
    compressed: bool = False
    last_used: float = field(default_factory=time.monotonic)


# -----------------------------------------------------------------------------
# HybridMemoryManager
# -----------------------------------------------------------------------------
class HybridMemoryManager:
    """
    Three‑tier LRU memory hierarchy for LLM weight caching.
    All SSD filenames are SHA256‑hashed. Writes are atomic with fsync.

    Concurrency model:

      * `self._lock` (RLock) guards:
          - `_blocks` dict
          - `_vram_used` / `_ram_used` / `_ssd_used` counters
          - block metadata (`tier`, `size_bytes`, `last_used`)

      * `self._lock` is NEVER held during:
          - SSD file reads (`path.read_bytes()`)
          - SSD file writes (`tmp.write_bytes() + fsync + rename`)
          - SSD file unlinks (`path.unlink()`)
          - `gc.collect()`

      * Every public method (`store` / `load` / `evict` /
        `migrate_down` / `shutdown`) is structured as a sequence of
        short locked phases interleaved with unlocked I/O phases.
    """

    def __init__(
        self,
        vram_bytes: int,
        ram_bytes: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ---- normalise config ------------------------------------------------
        # Capture the caller's raw `memory` dict so we can recover any keys
        # the Pydantic validator silently drops (HybridConfig has no
        # `hybrid` field in some releases — the sub-dict is discarded on
        # validation, and every hybrid default then fires).
        _raw_mem: Dict[str, Any] = (
            config.get("memory", {}) or {}
            if isinstance(config, dict) else {}
        )

        mem_cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import HybridConfig as HybridConfigModel
                if isinstance(config, dict):
                    cfg_model = HybridConfigModel(**_raw_mem)
                else:
                    cfg_model = HybridConfigModel(**config.get("memory", {}))
                mem_cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("Hybrid config validation failed: %s; using defaults", exc)
                mem_cfg = dict(_raw_mem) if _raw_mem else {}

        # Merge back any keys the validator dropped — the caller's input
        # always wins over a hardcoded default (§1: real values only).
        for _k, _v in _raw_mem.items():
            mem_cfg.setdefault(_k, _v)

        # Accept BOTH nested `hybrid.*` (config.yaml) and flat `hybrid_*`
        # (api_server.py's historical call shape) — one source of truth.
        hybrid_cfg: Dict[str, Any] = dict(mem_cfg.get("hybrid", {}) or {})
        if "ssd_path" not in hybrid_cfg and "hybrid_ssd_path" in mem_cfg:
            hybrid_cfg["ssd_path"] = mem_cfg["hybrid_ssd_path"]
        if "ssd_max_gb" not in hybrid_cfg and "hybrid_ssd_max_gb" in mem_cfg:
            hybrid_cfg["ssd_max_gb"] = mem_cfg["hybrid_ssd_max_gb"]
        # ----------------------------------------------------------------------

        self.vram_bytes = vram_bytes
        self.ram_bytes = ram_bytes

        self.ssd_path = Path(
            hybrid_cfg.get("ssd_path", "/tmp/llm_hybrid")
        ).resolve()
        self.ssd_max_bytes = int(
            float(hybrid_cfg.get("ssd_max_gb", 50.0)) * (1024**3)
        )
        self._compression = hybrid_cfg.get("compression_enabled", True)
        self._promotion_threshold = hybrid_cfg.get("promotion_threshold", 0.95)

        self._vram_used = 0
        self._ram_used = 0
        self._ssd_used = 0
        self._read_errors = 0
        self._write_errors = 0
        self._blocks: Dict[str, HybridBlock] = {}
        self._lock = threading.RLock()

        self.ssd_path.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphan_tmps()
        self._compress_fn, self._decompress_fn = self._setup_compression()

        log.info(
            "HybridMemoryManager: vram=%dMB ram=%dMB ssd=%dMB compress=%s",
            vram_bytes >> 20, ram_bytes >> 20,
            self.ssd_max_bytes >> 20, self._compression,
        )

    def _cleanup_orphan_tmps(self) -> None:
        for f in self.ssd_path.glob("*.tmp"):
            try:
                f.unlink(missing_ok=True)
                log.debug("HybridManager: removed orphan tmp: %s", f.name)
            except Exception:
                pass

    def _setup_compression(self) -> Tuple[Callable, Callable]:
        try:
            import lz4.frame
            return lz4.frame.compress, lz4.frame.decompress
        except (ImportError, AttributeError):
            pass
        try:
            import zlib
            return zlib.compress, zlib.decompress
        except Exception:
            return (lambda x, **kw: x), (lambda x: x)

    def _path_for_key(self, key: str) -> Path:
        stem = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]
        return self.ssd_path / f"{stem}.hybrid"

    # =========================================================================
    # store — phased locking; SSD file I/O outside the lock
    # =========================================================================
    def store(
        self,
        key: str,
        data: Any,
        preferred_tier: MemoryTier = MemoryTier.VRAM,
    ) -> MemoryTier:
        size = self._estimate_size(data)
        cleanup_paths: List[Path] = []

        # ── Phase 1: lock — pop old block, pick tier ────────────────────
        needs_ssd_write = False
        with self._lock:
            old_ssd_path: Optional[Path] = None
            if key in self._blocks:
                old = self._free_block_locked(key)
                if old is not None:
                    self._release_block_data(old)
                    if old.tier == MemoryTier.SSD:
                        old_ssd_path = self._path_for_key(key)

            tier = self._find_tier(size, preferred_tier)
            needs_ssd_write = (tier == MemoryTier.SSD)

            if tier == MemoryTier.VRAM:
                self._blocks[key] = HybridBlock(
                    key=key, tier=tier, size_bytes=size,
                    data=self._to_vram(data),
                )
                self._vram_used += size
            elif tier == MemoryTier.RAM:
                self._blocks[key] = HybridBlock(
                    key=key, tier=tier, size_bytes=size,
                    data=self._to_ram(data),
                )
                self._ram_used += size
            # else: SSD — registered in Phase 3 after the (unlocked) write.

            # Drop the previous SSD file only when the new tier is NOT
            # also SSD. If both are SSD, `_to_ssd()` in Phase 2 overwrites
            # the exact same path; unlinking here would destroy the file
            # we are about to write.
            if old_ssd_path is not None and not needs_ssd_write:
                cleanup_paths.append(old_ssd_path)

        # ── VRAM/RAM fast path: cleanup outside the lock ───────────────
        if not needs_ssd_write:
            self._do_cleanup_paths(cleanup_paths)
            return tier

        # ── Phase 2: no lock — SSD write (slow) ────────────────────────
        path_str, actual_bytes = self._to_ssd(key, data)
        if actual_bytes == 0:
            # `_to_ssd` already bumped `_write_errors` exactly once.
            self._do_cleanup_paths(cleanup_paths)
            raise IOError(
                f"SSD write failed for key '{key}' — "
                f"check disk space/permissions"
            )

        # ── Phase 3: lock — register + LRU-evict if needed ─────────────
        with self._lock:
            # A concurrent `store()` on the same key may have registered
            # between our Phase 1 pop and now. Pop it under the lock —
            # its file at `path_str` is exactly the file we just wrote.
            if key in self._blocks:
                stale = self._free_block_locked(key)
                if stale is not None:
                    self._release_block_data(stale)

            while (self._ssd_used + actual_bytes > self.ssd_max_bytes
                   and self._blocks):
                oldest_key = min(
                    self._blocks,
                    key=lambda k: self._blocks[k].last_used,
                )
                evicted = self._free_block_locked(oldest_key)
                if evicted is not None:
                    self._release_block_data(evicted)
                    if evicted.tier == MemoryTier.SSD:
                        cleanup_paths.append(self._path_for_key(oldest_key))

            if self._ssd_used + actual_bytes > self.ssd_max_bytes:
                self._do_cleanup_paths(cleanup_paths)
                try:
                    Path(path_str).unlink(missing_ok=True)
                except Exception:
                    pass
                raise MemoryError(
                    f"HybridManager: SSD tier full "
                    f"({actual_bytes} bytes needed, "
                    f"{self.ssd_max_bytes - self._ssd_used} available)"
                )

            self._blocks[key] = HybridBlock(
                key=key, tier=MemoryTier.SSD, size_bytes=actual_bytes,
                data=path_str, compressed=self._compression,
            )
            self._ssd_used += actual_bytes

        # ── Phase 4: no lock — cleanup ─────────────────────────────────
        self._do_cleanup_paths(cleanup_paths)
        log.info(
            "HybridManager: stored '%s' on %s (%d bytes)",
            key, tier.name, actual_bytes if needs_ssd_write else size,
        )
        return tier

    def load(self, key: str) -> Optional[Any]:
        cleanup_paths: List[Path] = []
        data: Optional[Any] = None
        path_str: Optional[str] = None

        with self._lock:
            if key not in self._blocks:
                return None
            block = self._blocks[key]
            block.last_used = time.monotonic()

            if block.tier != MemoryTier.SSD:
                data = block.data
                self._maybe_promote_to_vram_locked(
                    block, key, data, cleanup_paths,
                )
            else:
                path_str = block.data

        if data is None and path_str is not None:
            data = self._read_ssd_payload(key, path_str)
            if data is None:
                with self._lock:
                    cur = self._blocks.get(key)
                    if cur is not None and cur.tier == MemoryTier.SSD:
                        blk = self._free_block_locked(key)
                        if blk is not None:
                            self._release_block_data(blk)
                            cleanup_paths.append(self._path_for_key(key))
                self._do_cleanup_paths(cleanup_paths)
                return None

            with self._lock:
                cur = self._blocks.get(key)
                if cur is not None and cur.tier == MemoryTier.SSD:
                    cur.last_used = time.monotonic()
                    self._maybe_promote_to_vram_locked(
                        cur, key, data, cleanup_paths,
                    )

        self._do_cleanup_paths(cleanup_paths)
        return data

    def evict(self, key: str) -> bool:
        with self._lock:
            block = self._free_block_locked(key)
        if block is None:
            return False
        self._release_block_data(block)
        if block.tier == MemoryTier.SSD:
            try:
                self._path_for_key(key).unlink(missing_ok=True)
            except Exception:
                log.error("HybridManager: unlink failed for '%s'", key)
        return True

    def migrate_down(self, key: str) -> bool:
        data: Optional[Any] = None
        path_str: Optional[str] = None
        next_tier: Optional[MemoryTier] = None

        with self._lock:
            if key not in self._blocks:
                return False
            block = self._blocks[key]

            next_tier = {
                MemoryTier.VRAM: MemoryTier.RAM,
                MemoryTier.RAM: MemoryTier.SSD,
                MemoryTier.SSD: MemoryTier.SSD,
            }[block.tier]

            if next_tier == block.tier:
                return True

            if block.tier != MemoryTier.SSD:
                data = block.data
            else:
                path_str = block.data

        if data is None and path_str is not None:
            data = self._read_ssd_payload(key, path_str)
            if data is None:
                with self._lock:
                    blk = self._free_block_locked(key)
                if blk is not None:
                    self._release_block_data(blk)
                    try:
                        self._path_for_key(key).unlink(missing_ok=True)
                    except Exception:
                        pass
                return False

        with self._lock:
            blk = self._free_block_locked(key)
        if blk is not None:
            self._release_block_data(blk)
            if blk.tier == MemoryTier.SSD and next_tier != MemoryTier.SSD:
                try:
                    self._path_for_key(key).unlink(missing_ok=True)
                except Exception:
                    pass

        try:
            self.store(key, data, next_tier)
        except Exception as exc:
            log.warning(
                "HybridManager: migrate_down store failed for '%s': %s",
                key, exc,
            )
            return False
        return True

    def utilization(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "vram_pct": 100 * self._vram_used / max(self.vram_bytes, 1),
                "ram_pct": 100 * self._ram_used / max(self.ram_bytes, 1),
                "ssd_pct": 100 * self._ssd_used / max(self.ssd_max_bytes, 1),
                "vram_used_mb": self._vram_used >> 20,
                "ram_used_mb": self._ram_used >> 20,
                "ssd_used_mb": self._ssd_used >> 20,
                "vram_total_mb": self.vram_bytes >> 20,
                "ram_total_mb": self.ram_bytes >> 20,
                "ssd_total_mb": self.ssd_max_bytes >> 20,
                "block_count": len(self._blocks),
                "read_errors": self._read_errors,
                "write_errors": self._write_errors,
            }

    def shutdown(self) -> None:
        with self._lock:
            blocks = list(self._blocks.values())
            self._blocks.clear()
            self._vram_used = 0
            self._ram_used = 0
            self._ssd_used = 0

        for blk in blocks:
            self._release_block_data(blk)
            if blk.tier == MemoryTier.SSD:
                try:
                    self._path_for_key(blk.key).unlink(missing_ok=True)
                except Exception:
                    log.error(
                        "HybridManager.shutdown: unlink failed for '%s'",
                        blk.key,
                    )

        if blocks:
            try:
                gc.collect()
            except Exception:
                pass

        log.info(
            "HybridMemoryManager shutdown complete "
            "(read_errors=%d write_errors=%d)",
            self._read_errors, self._write_errors,
        )

    def _find_tier(self, size: int, preferred: MemoryTier) -> MemoryTier:
        tiers = [preferred] + [t for t in MemoryTier if t != preferred]
        caps = {
            MemoryTier.VRAM: self.vram_bytes - self._vram_used,
            MemoryTier.RAM: self.ram_bytes - self._ram_used,
            MemoryTier.SSD: self.ssd_max_bytes - self._ssd_used,
        }
        for t in tiers:
            if caps[t] >= size:
                return t
        raise MemoryError(
            f"HybridManager: no tier has {size >> 10} KB free "
            f"(vram_free={caps[MemoryTier.VRAM] >> 20}MB "
            f"ram_free={caps[MemoryTier.RAM] >> 20}MB "
            f"ssd_free={caps[MemoryTier.SSD] >> 20}MB)"
        )

    def _free_block_locked(self, key: str) -> Optional[HybridBlock]:
        if key not in self._blocks:
            return None
        block = self._blocks.pop(key)
        if block.tier == MemoryTier.VRAM:
            self._vram_used = max(0, self._vram_used - block.size_bytes)
        elif block.tier == MemoryTier.RAM:
            self._ram_used = max(0, self._ram_used - block.size_bytes)
        else:
            self._ssd_used = max(0, self._ssd_used - block.size_bytes)
        return block

    def _maybe_promote_to_vram_locked(
        self,
        block: HybridBlock,
        key: str,
        data: Any,
        cleanup_paths: List[Path],
    ) -> None:
        if block.tier == MemoryTier.VRAM:
            return

        size = self._estimate_size(data)
        if self._vram_used + size > self.vram_bytes * self._promotion_threshold:
            return

        old_tier = block.tier
        old_size = block.size_bytes
        old_data = block.data

        if old_tier == MemoryTier.RAM:
            self._ram_used = max(0, self._ram_used - old_size)
            try:
                del old_data
            except Exception:
                pass
        elif old_tier == MemoryTier.SSD:
            self._ssd_used = max(0, self._ssd_used - old_size)
            cleanup_paths.append(self._path_for_key(key))

        vram_data = self._to_vram(data)
        block.tier = MemoryTier.VRAM
        block.size_bytes = size
        block.data = vram_data
        block.compressed = False
        self._vram_used += size

    @staticmethod
    def _release_block_data(block: HybridBlock) -> None:
        if block.tier != MemoryTier.SSD:
            try:
                del block.data
            except Exception:
                pass

    @staticmethod
    def _do_cleanup_paths(paths: List[Path]) -> None:
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                log.error("HybridManager: unlink failed for %s", p)

    def _read_ssd_payload(self, key: str, path_str: str) -> Optional[Any]:
        try:
            path = Path(path_str) if path_str else self._path_for_key(key)
            raw = path.read_bytes()
            if len(raw) < _HDR_SIZE:
                log.warning(
                    "HybridManager: truncated file for '%s' (%d bytes)",
                    key, len(raw),
                )
                with self._lock:
                    self._read_errors += 1
                return None

            magic, flags, key_len, crc32_stored, payload_len = struct.unpack(
                _HDR_FMT, raw[:_HDR_SIZE]
            )
            if magic != _MAGIC:
                log.warning("HybridManager: bad magic for '%s'", key)
                with self._lock:
                    self._read_errors += 1
                return None

            header_total = _HDR_SIZE + key_len
            if len(raw) < header_total + payload_len:
                log.warning("HybridManager: file too short for '%s'", key)
                with self._lock:
                    self._read_errors += 1
                return None

            payload = raw[header_total : header_total + payload_len]
            if _crc32(payload) != crc32_stored:
                log.warning("HybridManager: CRC mismatch for '%s'", key)
                with self._lock:
                    self._read_errors += 1
                return None

            if flags & _F_COMPRESSED:
                payload = self._decompress_fn(payload)

            result = self._deserialize(payload)
            if result is None:
                with self._lock:
                    self._read_errors += 1
            return result
        except Exception:
            log.warning(
                "HybridManager: read failed for '%s'", key, exc_info=True,
            )
            with self._lock:
                self._read_errors += 1
            return None

    def _to_ssd(self, key: str, data: Any) -> Tuple[str, int]:
        raw = self._serialize(data)
        flags = 0
        if self._compression:
            compressed = self._compress_fn(raw)
            if len(compressed) < len(raw):
                raw = compressed
                flags = _F_COMPRESSED

        key_b = key.encode("utf-8", errors="replace")
        header = struct.pack(
            _HDR_FMT, _MAGIC, flags, len(key_b), _crc32(raw), len(raw)
        )
        payload = header + key_b + raw
        path = self._path_for_key(key)
        tmp = path.with_suffix(".tmp")

        try:
            tmp.write_bytes(payload)
            with tmp.open("rb") as fh:
                os.fsync(fh.fileno())
            tmp.rename(path)
            return str(path), len(payload)
        except Exception:
            log.warning(
                "HybridManager: SSD write failed for '%s'", key, exc_info=True,
            )
            with self._lock:
                self._write_errors += 1
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return str(path), 0

    def _serialize(self, data: Any) -> bytes:
        try:
            import torch
            if hasattr(torch, "Tensor") and isinstance(data, torch.Tensor):
                buf = io.BytesIO()
                torch.save(data, buf)
                return buf.getvalue()
        except ImportError:
            pass
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, data)
                return buf.getvalue()
        except ImportError:
            pass
        import pickle
        return pickle.dumps(data)

    def _deserialize(self, raw: bytes) -> Optional[Any]:
        """Decode `raw` into the original Python object.

        dispatches by magic bytes so each payload goes straight to the
        correct decoder without spurious WARNING/ERROR stacks from the
        two loaders that don't apply. Previous behaviour tried
        torch → numpy → safe-pickle for every payload, so every pickled
        bytes/str/dict SSD read (the common non-tensor case) logged two
        scary-looking tracebacks before succeeding at stage three.

        Magic prefixes:
          * b"PK\\x03\\x04" — torch.save (ZIP container, torch >= 1.6)
          * b"\\x93NUMPY"   — numpy .npy format
          * b"\\x80"        — pickle protocol >= 2 (plain pickled bytes,
                              str, dict, list, etc.)
          * anything else   — try safe-pickle → numpy → torch; only the
                              final failure surfaces as a WARNING.
        """
        if raw[:4] == b"PK\x03\x04":
            return self._deserialize_torch(raw)
        if raw[:6] == b"\x93NUMPY":
            return self._deserialize_numpy(raw)
        if raw[:1] == b"\x80":
            return self._deserialize_pickle(raw)
        # Unknown magic — probe in cheapest-first order, DEBUG-level on
        # intermediate failures (they are expected probes, not bugs).
        result = self._deserialize_pickle(raw)
        if result is not None:
            return result
        result = self._deserialize_numpy(raw)
        if result is not None:
            return result
        return self._deserialize_torch(raw)

    def _deserialize_torch(self, raw: bytes) -> Optional[Any]:
        try:
            import torch
        except ImportError:
            log.warning(
                "HybridManager: torch payload on SSD but torch is not "
                "installed — dropping. Install torch or clear the SSD tier."
            )
            return None
        try:
            return torch.load(
                io.BytesIO(raw), map_location="cpu", weights_only=True,
            )
        except TypeError:
            log.warning(
                "HybridManager: torch %s does not support weights_only=True "
                "— refusing to deserialize tensor payload from SSD. Upgrade "
                "torch to >= 1.13 to re-enable SSD tier for tensor data.",
                getattr(torch, "__version__", "?"),
            )
            return None
        except Exception:
            log.warning(
                "HybridManager: torch deserialize failed", exc_info=True,
            )
            return None

    def _deserialize_numpy(self, raw: bytes) -> Optional[Any]:
        try:
            import numpy as np
        except ImportError:
            log.warning(
                "HybridManager: numpy payload on SSD but numpy is not "
                "installed — dropping."
            )
            return None
        try:
            return np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception:
            log.warning(
                "HybridManager: numpy deserialize failed", exc_info=True,
            )
            return None

    def _deserialize_pickle(self, raw: bytes) -> Optional[Any]:
        import pickle
        try:
            class _SafeUnpickler(pickle.Unpickler):
                _SAFE = frozenset({
                    "numpy", "numpy.core", "numpy.core.multiarray",
                    "torch", "torch.storage", "torch._utils",
                    "builtins", "__builtin__", "collections", "_codecs",
                })
                def find_class(self, module, name):
                    root = module.split(".")[0]
                    if root in self._SAFE:
                        return super().find_class(module, name)
                    raise pickle.UnpicklingError(
                        f"Blocked: {module}.{name} not in whitelist"
                    )
            return _SafeUnpickler(io.BytesIO(raw)).load()
        except pickle.UnpicklingError as exc:
            log.error("HybridManager: pickle blocked: %s", exc)
            return None
        except Exception:
            log.warning(
                "HybridManager: pickle deserialize failed", exc_info=True,
            )
            return None

    def _to_vram(self, data: Any) -> Any:
        try:
            import torch
            if (
                hasattr(torch, "Tensor")
                and isinstance(data, torch.Tensor)
                and torch.cuda.is_available()
            ):
                return data.cuda()
        except ImportError:
            pass
        return data

    def _to_ram(self, data: Any) -> Any:
        try:
            import torch
            if (
                hasattr(torch, "Tensor")
                and isinstance(data, torch.Tensor)
                and data.device.type != "cpu"
            ):
                return data.cpu()
        except ImportError:
            pass
        return data

    @staticmethod
    def _estimate_size(data: Any) -> int:
        try:
            import torch
            if hasattr(torch, "Tensor") and isinstance(data, torch.Tensor):
                return data.nelement() * data.element_size()
        except (ImportError, Exception):
            pass
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return data.nbytes
        except ImportError:
            pass
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        if hasattr(data, "__len__"):
            return max(len(data), 1)
        return 1024

    def estimate_model_fit(
        self,
        model_path: Union[str, Path],
        model_size_mb: float,
        context_len: int,
        target_tps: float,
        kv_quant: str = "q8_0",
    ) -> Dict[str, Any]:
        ram_mb = self.ram_bytes // (1024 * 1024)
        overhead_mb = ram_mb * 0.20
        fast_mb = (self.vram_bytes + self.ram_bytes) // (1024 * 1024) - overhead_mb
        swap_mb = self.ssd_max_bytes // (1024 * 1024)
        usable_mb = fast_mb + swap_mb

        try:
            from local_llm_core.engine.pipelines.layer_streamer import read_gguf_meta
            meta = read_gguf_meta(Path(model_path))
        except Exception:
            log.warning(
                "Could not read GGUF metadata for '%s' – using defaults",
                model_path,
            )
            meta = {
                "n_layer": 32, "n_head": 32,
                "n_embd": 4096, "n_kv_heads": 8,
            }

        n_layers = meta.get("n_layer", 32)
        n_head = meta.get("n_head", 32)
        n_embd = meta.get("n_embd", 4096)
        n_kv_heads = meta.get("n_kv_heads", 0)
        if n_kv_heads == 0:
            n_kv_heads = max(1, n_head // 4)
        head_dim = max(1, n_embd // n_head)

        kv_factor = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}.get(kv_quant, 0.5)
        kv_cache_bytes = (
            (2 * n_kv_heads * head_dim * context_len * 2 * kv_factor) * n_layers
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        required_mb = model_size_mb + kv_cache_mb
        fits = required_mb <= usable_mb
        fits_fast = required_mb <= fast_mb

        if not fits:
            estimated_tps = 0.0
            status = "will_not_fit"
        elif not fits_fast:
            status = "slow" if (required_mb / usable_mb) < 0.8 else "very_slow"
            estimated_tps = target_tps * (0.35 if status == "slow" else 0.10)
        else:
            usage_ratio = required_mb / fast_mb
            if usage_ratio < 0.6:
                status = "comfortable"
                estimated_tps = target_tps * 0.90
            elif usage_ratio < 0.8:
                status = "slow"
                estimated_tps = target_tps * 0.60
            else:
                status = "very_slow"
                estimated_tps = target_tps * 0.20

            if estimated_tps < 0.5:
                status = "very_slow"
                estimated_tps = 0.0

        return {
            "fits": fits,
            "fits_without_swap": fits_fast,
            "usable_memory_mb": int(usable_mb),
            "fast_memory_mb": int(fast_mb),
            "required_memory_mb": int(required_mb),
            "estimated_tps": round(estimated_tps, 2),
            "status": status,
        }

    def total_usable_mb(self) -> int:
        total_mb = (self.vram_bytes + self.ram_bytes) // (1024 * 1024)
        swap_mb = self.ssd_max_bytes // (1024 * 1024)
        total_mb += swap_mb
        ram_mb = self.ram_bytes // (1024 * 1024)
        overhead_mb = ram_mb * 0.20
        return int(total_mb - overhead_mb)
