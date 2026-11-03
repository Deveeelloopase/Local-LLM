from __future__ import annotations

"""
memory/hybrid_manager.py — Production‑Ready Hybrid Memory Manager
=================================================================
v1.0.0 – Config injected via constructor; no filesystem reads.

Three‑tier memory hierarchy: VRAM → RAM → SSD swap.
- LRU caching with promotion/demotion
- Atomic SSD writes with fsync and CRC validation
- LZ4 compression (configurable)
- Safe deserialization (pickle whitelist)
- Configurable promotion threshold

All parameters are taken from the `config` dict (or a `HybridConfig`
model).  Sensible defaults are used when config is omitted.
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
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
    """

    def __init__(
        self,
        vram_bytes: int,
        ram_bytes: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ---- normalise config ------------------------------------------------
        mem_cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import HybridConfig as HybridConfigModel
                if isinstance(config, dict):
                    cfg_model = HybridConfigModel(**config.get("memory", {}))
                else:
                    cfg_model = HybridConfigModel(**config.get("memory", {}))
                mem_cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("Hybrid config validation failed: %s; using defaults", exc)
                mem_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        self.vram_bytes = vram_bytes
        self.ram_bytes = ram_bytes

        # Configurable parameters (from the `hybrid` sub‑section)
        hybrid_cfg = mem_cfg.get("hybrid", {})
        self.ssd_path = Path(
            hybrid_cfg.get("ssd_path", "/tmp/llm_hybrid")
        ).resolve()
        self.ssd_max_bytes = int(
            hybrid_cfg.get("ssd_max_gb", 50.0) * (1024**3)
        )
        self._compression = hybrid_cfg.get("compression_enabled", True)
        self._promotion_threshold = hybrid_cfg.get("promotion_threshold", 0.95)

        # State
        self._vram_used = 0
        self._ram_used = 0
        self._ssd_used = 0
        self._read_errors = 0
        self._write_errors = 0
        self._blocks: Dict[str, HybridBlock] = {}
        self._lock = threading.RLock()

        # Initialize SSD directory and compression
        self.ssd_path.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphan_tmps()
        self._compress_fn, self._decompress_fn = self._setup_compression()

        log.info(
            "HybridMemoryManager: vram=%dMB ram=%dMB ssd=%dGB compress=%s",
            vram_bytes >> 20,
            ram_bytes >> 20,
            self.ssd_max_bytes >> 30,
            self._compression,
        )

    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------
    def _cleanup_orphan_tmps(self) -> None:
        """Remove .tmp files left by interrupted writes."""
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

    # -------------------------------------------------------------------------
    # Path Generation
    # -------------------------------------------------------------------------
    def _path_for_key(self, key: str) -> Path:
        stem = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]
        return self.ssd_path / f"{stem}.hybrid"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def store(
        self,
        key: str,
        data: Any,
        preferred_tier: MemoryTier = MemoryTier.VRAM,
    ) -> MemoryTier:
        """Store data, spilling to lower tiers as needed."""
        size = self._estimate_size(data)
        with self._lock:
            if key in self._blocks:
                self._free_block(key)
            tier = self._find_tier(size, preferred_tier)
            block = HybridBlock(key=key, tier=tier, size_bytes=size, data=None)

            if tier == MemoryTier.VRAM:
                block.data = self._to_vram(data)
                self._vram_used += size
            elif tier == MemoryTier.RAM:
                block.data = self._to_ram(data)
                self._ram_used += size
            else:
                path, actual_bytes = self._to_ssd(key, data)
                if actual_bytes == 0:
                    log.error(
                        "HybridManager: SSD write failed for '%s' — not storing", key
                    )
                    self._write_errors += 1
                    raise IOError(
                        f"SSD write failed for key '{key}' — check disk space/permissions"
                    )
                block.data = path
                block.size_bytes = actual_bytes
                block.compressed = self._compression
                self._ssd_used += actual_bytes

            self._blocks[key] = block
            log.debug(
                "HybridManager: stored '%s' on %s (%d bytes)",
                key,
                tier.name,
                block.size_bytes,
            )
            return tier

    def load(self, key: str) -> Optional[Any]:
        """Load data and promote toward VRAM if room allows."""
        with self._lock:
            if key not in self._blocks:
                return None
            block = self._blocks[key]
            block.last_used = time.monotonic()
            data = self._read_block(block)

            if data is None:
                log.warning(
                    "HybridManager: read failed for '%s' — not promoting", key
                )
                return None

            size = self._estimate_size(data)
            if (
                block.tier != MemoryTier.VRAM
                and self._vram_used + size <= self.vram_bytes * self._promotion_threshold
            ):
                # Promote in‑place
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
                    path = self._path_for_key(key)
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass

                vram_data = self._to_vram(data)
                block.tier = MemoryTier.VRAM
                block.size_bytes = size
                block.data = vram_data
                block.compressed = False
                self._vram_used += size
                return vram_data

            return data

    def evict(self, key: str) -> bool:
        with self._lock:
            return self._free_block(key)

    def migrate_down(self, key: str) -> bool:
        """Force a block one tier down (VRAM→RAM→SSD)."""
        with self._lock:
            if key not in self._blocks:
                return False
            block = self._blocks[key]
            data = self._read_block(block)
            if data is None:
                log.warning(
                    "HybridManager: migrate_down read failed for '%s'", key
                )
                self._free_block(key)
                return False

            next_tier = {
                MemoryTier.VRAM: MemoryTier.RAM,
                MemoryTier.RAM: MemoryTier.SSD,
                MemoryTier.SSD: MemoryTier.SSD,
            }[block.tier]
            self._free_block(key)
            self.store(key, data, next_tier)
            return True

    def utilization(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "vram_pct": 100 * self._vram_used / max(self.vram_bytes, 1),
                "ram_pct": 100 * self._ram_used / max(self.ram_bytes, 1),
                "ssd_pct": 100 * self._ssd_used / max(self.ssd_max_bytes, 1),
                "read_errors": self._read_errors,
                "write_errors": self._write_errors,
            }

    def shutdown(self) -> None:
        """Release all blocks and SSD files cleanly."""
        with self._lock:
            for key in list(self._blocks.keys()):
                try:
                    self._free_block(key)
                except Exception:
                    log.debug(
                        "HybridManager.shutdown error for '%s'", key, exc_info=True
                    )
        log.info(
            "HybridMemoryManager shutdown complete (read_errors=%d write_errors=%d)",
            self._read_errors,
            self._write_errors,
        )

    # -------------------------------------------------------------------------
    # Tier Resolution
    # -------------------------------------------------------------------------
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
            f"(vram_free={caps[MemoryTier.VRAM]>>20}MB "
            f"ram_free={caps[MemoryTier.RAM]>>20}MB "
            f"ssd_free={caps[MemoryTier.SSD]>>20}MB)"
        )

    # -------------------------------------------------------------------------
    # Block I/O
    # -------------------------------------------------------------------------
    def _read_block(self, block: HybridBlock) -> Optional[Any]:
        if block.tier != MemoryTier.SSD:
            return block.data

        path = self._path_for_key(block.key)
        try:
            raw = path.read_bytes()
            if len(raw) < _HDR_SIZE:
                log.warning(
                    "HybridManager: truncated file for '%s' (%d bytes)",
                    block.key,
                    len(raw),
                )
                self._read_errors += 1
                return None

            magic, flags, key_len, crc32_stored, payload_len = struct.unpack(
                _HDR_FMT, raw[:_HDR_SIZE]
            )
            if magic != _MAGIC:
                log.warning("HybridManager: bad magic for '%s'", block.key)
                self._read_errors += 1
                return None

            header_total = _HDR_SIZE + key_len
            if len(raw) < header_total + payload_len:
                log.warning("HybridManager: file too short for '%s'", block.key)
                self._read_errors += 1
                return None

            payload = raw[header_total : header_total + payload_len]
            if _crc32(payload) != crc32_stored:
                log.warning("HybridManager: CRC mismatch for '%s'", block.key)
                self._read_errors += 1
                return None

            if flags & _F_COMPRESSED:
                payload = self._decompress_fn(payload)

            return self._deserialize(payload)
        except Exception:
            log.warning(
                "HybridManager: _read_block failed for '%s'",
                block.key,
                exc_info=True,
            )
            self._read_errors += 1
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

    def _to_ssd(self, key: str, data: Any) -> Tuple[str, int]:
        """Write data atomically with fsync. Returns (path, actual_bytes_written)."""
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
            actual = len(payload)
            return str(path), actual
        except Exception:
            log.warning(
                "HybridManager: SSD write failed for '%s'", key, exc_info=True
            )
            self._write_errors += 1
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return str(path), 0

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
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
        try:
            import torch
            buf = io.BytesIO(raw)
            return torch.load(buf, map_location="cpu")
        except ImportError:
            pass
        except Exception:
            log.debug("HybridManager: torch deserialize failed", exc_info=True)
        try:
            import numpy as np
            buf = io.BytesIO(raw)
            return np.load(buf, allow_pickle=False)
        except ImportError:
            pass
        except Exception:
            log.debug("HybridManager: numpy deserialize failed", exc_info=True)
        try:
            import pickle
            import io as _io

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

            return _SafeUnpickler(_io.BytesIO(raw)).load()
        except pickle.UnpicklingError as exc:
            log.error("HybridManager: pickle blocked: %s", exc)
            self._read_errors += 1
            return None
        except Exception:
            log.debug("HybridManager: pickle deserialize failed", exc_info=True)
            return None

    # -------------------------------------------------------------------------
    # Free Block
    # -------------------------------------------------------------------------
    def _free_block(self, key: str) -> bool:
        if key not in self._blocks:
            return False
        block = self._blocks.pop(key)

        if block.tier == MemoryTier.VRAM:
            self._vram_used = max(0, self._vram_used - block.size_bytes)
            try:
                del block.data
            except Exception:
                pass
            gc.collect()
        elif block.tier == MemoryTier.RAM:
            self._ram_used = max(0, self._ram_used - block.size_bytes)
            try:
                del block.data
            except Exception:
                pass
            gc.collect()
        else:
            self._ssd_used = max(0, self._ssd_used - block.size_bytes)
            path = self._path_for_key(key)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                log.debug("HybridManager: unlink failed for '%s'", key, exc_info=True)
        return True

    # -------------------------------------------------------------------------
    # Size Estimation
    # -------------------------------------------------------------------------
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

    # =========================================================================
    # Pre‑flight model fit estimation
    # =========================================================================
    def estimate_model_fit(
        self,
        model_path: Union[str, Path],
        model_size_mb: float,
        context_len: int,
        target_tps: float,
        kv_quant: str = "q8_0",
    ) -> Dict[str, Any]:
        """
        Estimate whether a model can run on this hardware.

        Reads GGUF metadata to compute KV cache size, then compares against
        total usable memory (VRAM + RAM + swap). Returns a dict with:
            - "fits": bool
            - "usable_memory_mb": total memory available (VRAM + RAM + swap) minus OS overhead
            - "required_memory_mb": estimated memory required for model + KV cache
            - "estimated_tps": projected TPS (0 if below a threshold)
            - "status": "comfortable", "slow", "very_slow", "will_not_fit"
        """
        # ── 1. Total usable memory ────────────────────────────────────────
        # Two figures, not one: "fast" (VRAM+RAM, what the model can run in
        # without touching swap) decides the comfortable/slow/very_slow
        # label; "with swap" only decides whether it can load at all.
        # Blending swap into the same number the "comfortable" label reads
        # from was the bug — a model that only fits by counting tens of GB
        # of swap was reporting the same status as one that fits in RAM.
        ram_mb = self.ram_bytes // (1024 * 1024)
        overhead_mb = ram_mb * 0.20
        fast_mb = (self.vram_bytes + self.ram_bytes) // (1024 * 1024) - overhead_mb
        swap_mb = self.ssd_max_bytes // (1024 * 1024)
        usable_mb = fast_mb + swap_mb   # kept for the will_not_fit / fits boundary below

        # ── 2. Read model metadata ────────────────────────────────────────
        try:
            from local_llm_core.engine.pipelines.layer_streamer import read_gguf_meta
            meta = read_gguf_meta(Path(model_path))
        except Exception:
            log.warning("Could not read GGUF metadata for '%s' – using defaults", model_path)
            meta = {
                "n_layer": 32,
                "n_head": 32,
                "n_embd": 4096,
                "n_kv_heads": 8,
            }

        n_layers = meta.get("n_layer", 32)
        n_head = meta.get("n_head", 32)
        n_embd = meta.get("n_embd", 4096)
        n_kv_heads = meta.get("n_kv_heads", 0)
        if n_kv_heads == 0:
            n_kv_heads = max(1, n_head // 4)
        head_dim = max(1, n_embd // n_head)

        # ── 3. Compute KV cache size ──────────────────────────────────────
        # Same formula as layer_streamer.py
        kv_factor = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}.get(kv_quant, 0.5)
        kv_cache_bytes = (
            (2 * n_kv_heads * head_dim * context_len * 2 * kv_factor) * n_layers
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        required_mb = model_size_mb + kv_cache_mb
        fits = required_mb <= usable_mb          # swap-inclusive: can it load at all
        fits_fast = required_mb <= fast_mb        # RAM+VRAM only: can it run without swap

        # ── 4. Estimate TPS ───────────────────────────────────────────────
        if not fits:
            estimated_tps = 0.0
            status = "will_not_fit"
        elif not fits_fast:
            # Only fits by counting swap — this WILL cause the kind of
            # thrashing/stalling swap-dependent inference produces. Never
            # "comfortable": that word should mean "runs entirely in
            # RAM/VRAM," not "technically loadable if we page constantly."
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

            # If estimated TPS < 0.5, treat as "very_slow"
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
        """Return total usable memory in MB (VRAM + RAM + SSD swap minus overhead)."""
        total_mb = (self.vram_bytes + self.ram_bytes) // (1024 * 1024)
        swap_mb = self.ssd_max_bytes // (1024 * 1024)
        total_mb += swap_mb
        ram_mb = self.ram_bytes // (1024 * 1024)
        overhead_mb = ram_mb * 0.20
        return int(total_mb - overhead_mb)
    