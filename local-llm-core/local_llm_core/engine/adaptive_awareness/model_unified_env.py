from __future__ import annotations

# engine/adaptive_awareness/model_unified_env.py  — AQAMR  v1.0.0
# ====================================================================
# Adaptive Quantisation-Aware Model Router (AQAMR)
# Unified environment validator, state manager, and backend selector.
#
# v1.0.0
# ====================================================================

import json
import logging
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── SPECTRE: apply gguf-py PYTHONPATH before any gguf import (BUG 1) ────────
def _apply_spectre() -> None:
    """Apply SPECTRE PYTHONPATH so local gguf-py (TL1/TL2) is importable."""
    try:
        # Try compatibility layer first (full SPECTRE support)
        from local_llm_core.compatibility.transformers_compat import StateFileReader
        StateFileReader.apply_spectre_pythonpath()
        return
    except Exception:
        pass
    # Fallback: manual scan for gguf-py path in environment / common locations
    _spec_pp = os.environ.get("SPECTRE_GGUF_PY_PATH", "")
    if _spec_pp and Path(_spec_pp).is_dir() and _spec_pp not in sys.path:
        sys.path.insert(0, _spec_pp)
        return
    _pp = os.environ.get("PYTHONPATH", "")
    for entry in _pp.split(os.pathsep):
        entry = entry.strip()
        if entry and Path(entry).name == "gguf-py" and entry not in sys.path:
            sys.path.insert(0, entry)
            return

_apply_spectre()

# =============================================================================
# Enumerations
# =============================================================================
class InferenceBackend(Enum):
    BITNET_CPP_SERVER   = "bitnet_cpp_server"
    BITNET_CPP_PROCESS  = "bitnet_cpp_process"
    BITNET_TRANSFORMERS = "bitnet_transformers"
    LLAMA_CPP_PYTHON    = "llama_cpp_python"
    UNKNOWN             = "unknown"


class GGUFModelType(Enum):
    BITNET   = "bitnet"
    STANDARD = "standard"
    UNKNOWN  = "unknown"


# =============================================================================
# GGUF header parser
# =============================================================================
_GGUF_MAGIC = b"GGUF"

# BitNet quantisation types.
# PRODUCTION FIX (root cause #2 — confirmed, not guessed): this set used to
# be {30, 31, 32}, picked by assuming I2_S/TL1/TL2 were the "next available"
# ggml_ftype enum slots after Q8_K=19. That assumption was wrong. Verified
# directly against a real BitNet-b1.58-2B-4T i2_s GGUF's own
# `llama_load_print_meta` output ("model ftype = I2_S - 2 bpw ternary",
# `general.file_type` u32 = 40): the real value is 40, not 30. TL1/TL2
# numeric values are NOT independently confirmed by any source available
# here — do not extend this set with more guessed numbers. The
# architecture-string check below (_BITNET_ARCHITECTURE_PREFIXES) is the
# primary, fork-version-agnostic signal; this numeric set is now only a
# secondary/defensive check for builds where the architecture string is
# for some reason missing or non-standard.
_BITNET_QUANT_TYPES: frozenset = frozenset({40})
# GGUF file-type field names that indicate BitNet (kept broad — matches by
# name even if a given fork's numeric id differs from the one above)
_BITNET_FILE_TYPE_NAMES: frozenset = frozenset({
    "I2_S", "TL1", "TL2", "GGUF_TYPE_I2_S",
    "GGUF_TYPE_TL1", "GGUF_TYPE_TL2",
})
# general.architecture string prefixes that indicate a BitNet model.
# Confirmed directly against a real model file: general.architecture =
# "bitnet-b1.58". This is the durable signal: it doesn't depend on which
# ggml_ftype integer a particular BitNet.cpp fork/llama.cpp version chose,
# so it keeps working even if that numeric enum gets renumbered upstream.
_BITNET_ARCHITECTURE_PREFIXES: tuple = ("bitnet",)


@dataclass
class GGUFHeader:
    magic:          bytes = b""
    version:        int   = 0
    tensor_count:   int   = 0
    metadata_count: int   = 0
    model_type:     GGUFModelType = GGUFModelType.UNKNOWN
    file_type:      int   = -1
    file_type_name: str   = "unknown"
    architecture:   str   = ""     # PRODUCTION FIX: general.architecture, e.g. "bitnet-b1.58"
    parse_error:    Optional[str] = None
    source_path:    str   = ""


class GGUFTypeDetector:
    """
    Reads the GGUF binary header to classify a .gguf file.
    v9 fix: detect() is now a @staticmethod so classify_gguf() works
    without instantiation (BUG 4 fix).
    """

    # GGUF metadata value type sizes (for skipping metadata).
    # Per the GGUF spec: 0 UINT8=1 1 INT8=1 2 UINT16=2 3 INT16=2 4 UINT32=4
    # 5 INT32=4 6 FLOAT32=4 7 BOOL=1 10 UINT64=8 11 INT64=8 12 FLOAT64=8.
    # (8 STRING and 9 ARRAY are variable-length and handled separately.)
    # PRODUCTION FIX: previous map had wrong sizes for INT16/INT32/BOOL
    # (3,5,7) and was missing the 64-bit types (10,11,12) entirely. Any
    # GGUF file with one of those value types in its metadata BEFORE the
    # "general.file_type" key (e.g. a "tokenizer.ggml.add_bos_token" BOOL,
    # or a "general.parameter_count" UINT64 — both common, and present in
    # the official BitNet i2_s GGUF) desynced the byte scan, causing
    # classify_gguf() to silently fall back to GGUFModelType.STANDARD
    # instead of BITNET. That misclassification is why BitNet GGUFs were
    # routed through the standard llama.cpp path instead of BitNet.cpp.
    _VAL_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                  10: 8, 11: 8, 12: 8}

    @staticmethod
    def detect(model_path: str) -> GGUFHeader:
        """
        Parse the GGUF binary header.  Safe: never raises.
        If the file is not a valid .gguf, returns a header with parse_error set.
        """
        p = Path(model_path)
        hdr = GGUFHeader(source_path=model_path)

        # If path is a directory, look for GGUF files inside it
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
            if ggufs:
                return GGUFTypeDetector.detect(str(ggufs[0]))
            hdr.parse_error = f"No .gguf files in directory: {model_path}"
            return hdr

        if not p.is_file():
            hdr.parse_error = f"Path not found: {model_path}"
            return hdr

        try:
            with p.open("rb") as f:
                magic = f.read(4)
                if magic != _GGUF_MAGIC:
                    hdr.parse_error = f"Not a GGUF file (magic={magic!r})"
                    return hdr
                hdr.magic = magic

                raw_ver = f.read(4)
                if len(raw_ver) < 4:
                    hdr.parse_error = "Truncated version field"
                    return hdr
                hdr.version = struct.unpack("<I", raw_ver)[0]

                if hdr.version not in (1, 2, 3):
                    hdr.parse_error = f"Unsupported GGUF version: {hdr.version}"
                    return hdr

                if hdr.version == 1:
                    hdr.tensor_count   = struct.unpack("<I", f.read(4))[0]
                    hdr.metadata_count = struct.unpack("<I", f.read(4))[0]
                else:
                    hdr.tensor_count   = struct.unpack("<Q", f.read(8))[0]
                    hdr.metadata_count = struct.unpack("<Q", f.read(8))[0]

                # PRODUCTION FIX: previously scanned only for
                # "general.file_type" and classified BITNET purely on a
                # numeric ftype match — which silently failed for this
                # exact BitNet.cpp build (confirmed file_type=40, not the
                # previously-assumed 30). Now also captures
                # "general.architecture" in the same single pass and
                # classifies BITNET if EITHER signal matches. Architecture
                # is the durable one (a fork-specific ftype enum can be
                # renumbered; "bitnet-b1.58" as a string is not going to
                # change).
                file_type, architecture = GGUFTypeDetector._scan_metadata_for_classification(
                    f, hdr.metadata_count, hdr.version
                )
                hdr.architecture = architecture or ""

                if file_type is not None:
                    hdr.file_type      = file_type
                    hdr.file_type_name = GGUFTypeDetector._file_type_name(file_type)

                _is_bitnet_by_ftype = (
                    file_type is not None
                    and (file_type in _BITNET_QUANT_TYPES
                         or hdr.file_type_name in _BITNET_FILE_TYPE_NAMES)
                )
                _is_bitnet_by_arch = (
                    bool(architecture)
                    and architecture.lower().startswith(_BITNET_ARCHITECTURE_PREFIXES)
                )
                hdr.model_type = (
                    GGUFModelType.BITNET
                    if (_is_bitnet_by_ftype or _is_bitnet_by_arch)
                    else GGUFModelType.STANDARD
                )

        except (struct.error, OSError, ValueError) as exc:
            hdr.parse_error = f"GGUF parse error: {exc}"
        return hdr

    @staticmethod
    def _scan_metadata_for_classification(f, count: int, version: int):
        """
        Single pass over GGUF metadata key-value pairs, capturing BOTH
        "general.file_type" (UINT32) and "general.architecture" (STRING)
        if either is present.

        Returns (file_type: Optional[int], architecture: Optional[str]).

        PRODUCTION FIX: the old version returned the instant it found
        general.file_type, so general.architecture (which in this file's
        actual layout comes FIRST, at kv index 0, with file_type at index
        13) was never looked at. Capturing both in one pass costs nothing
        extra and gives detect() a second, more durable signal to check.
        """
        UINT32_TYPE = 4
        STRING_TYPE = 8
        file_type: Optional[int] = None
        architecture: Optional[str] = None
        try:
            for _ in range(min(count, 256)):
                # Read key
                if version == 1:
                    key_len_raw = f.read(4)
                    if not key_len_raw:
                        return file_type, architecture
                    key_len = struct.unpack("<I", key_len_raw)[0]
                else:
                    key_len_raw = f.read(8)
                    if not key_len_raw:
                        return file_type, architecture
                    key_len = struct.unpack("<Q", key_len_raw)[0]
                if key_len > 256:
                    return file_type, architecture  # sanity — no metadata key is this long
                key = f.read(key_len).decode("utf-8", errors="replace")

                # Read value type
                val_type_raw = f.read(4)
                if not val_type_raw:
                    return file_type, architecture
                val_type = struct.unpack("<I", val_type_raw)[0]

                if key == "general.file_type" and val_type == UINT32_TYPE:
                    val_raw = f.read(4)
                    if not val_raw:
                        return file_type, architecture
                    file_type = struct.unpack("<I", val_raw)[0]
                    continue

                if key == "general.architecture" and val_type == STRING_TYPE:
                    if version == 1:
                        n_raw = f.read(4)
                        if not n_raw:
                            return file_type, architecture
                        n = struct.unpack("<I", n_raw)[0]
                    else:
                        n_raw = f.read(8)
                        if not n_raw:
                            return file_type, architecture
                        n = struct.unpack("<Q", n_raw)[0]
                    architecture = f.read(n).decode("utf-8", errors="replace")
                    continue

                # Not a key we care about — skip its value and keep going.
                if not GGUFTypeDetector._skip_value(f, val_type, version):
                    return file_type, architecture

                # Stop early once both signals are found — no need to
                # scan through the (often huge) tokenizer vocab arrays.
                if file_type is not None and architecture is not None:
                    return file_type, architecture
        except Exception:
            return file_type, architecture
        return file_type, architecture

    @staticmethod
    def _skip_value(f, val_type: int, version: int) -> bool:
        """Skip a GGUF metadata value. Returns False on failure.

        the previous version parsed only the
        first 1024 array elements then returned True — leaving the file
        pointer desynchronised whenever a real array (tokenizer vocab is
        100K+ strings, scores and token_type are the same length) had
        more elements. The next KV-key read would consume raw float/byte
        data as a length, then attempt a multi-GB ``f.read()`` — the
        same bug class that was already fixed in
        ``layer_streamer.read_gguf_meta`` (which caused a 2-minute stall
        + SIGKILL in production; see that file's docstring).

        Masked today because real GGUFs put ``general.architecture`` at
        KV index 0 and ``general.file_type`` at ~13, so the classifier
        stops before reaching tokenizer arrays. Any future fork with a
        different metadata order would trigger it.

        Correct approach:
          * fixed-size element types: one ``f.seek(count * elem_size, 1)``
            — O(1) regardless of count
          * variable-size elements (STRING / nested ARRAY): parse each
            element's length prefix and advance; O(count) but no
            allocation
        """
        try:
            if val_type == 8:  # string
                if version == 1:
                    n = struct.unpack("<I", f.read(4))[0]
                else:
                    n = struct.unpack("<Q", f.read(8))[0]
                f.seek(n, 1)
            elif val_type == 9:  # array
                elem_type = struct.unpack("<I", f.read(4))[0]
                if version == 1:
                    count = struct.unpack("<I", f.read(4))[0]
                else:
                    count = struct.unpack("<Q", f.read(8))[0]
                esz = GGUFTypeDetector._VAL_SIZES.get(elem_type, 0)
                if esz > 0 and elem_type not in (8, 9):
                    # Fixed-size elements: single seek past all of them.
                    f.seek(count * esz, 1)
                else:
                    # Variable-size elements: must walk each one so the
                    # file pointer ends at the correct byte offset.
                    for _ in range(count):
                        if not GGUFTypeDetector._skip_value(f, elem_type, version):
                            return False
            else:
                sz = GGUFTypeDetector._VAL_SIZES.get(val_type)
                if sz is None:
                    return False
                f.seek(sz, 1)
        except Exception:
            return False
        return True

    @staticmethod
    def _file_type_name(ft: int) -> str:
        # PRODUCTION FIX: 30/31/32 removed (were guessed, confirmed wrong —
        # see _BITNET_QUANT_TYPES comment above). 40 added (confirmed I2_S).
        # TL1/TL2 are intentionally NOT mapped to a guessed number here —
        # an unconfirmed number is worse than no number, since it could
        # silently mislabel some future, unrelated quant type that happens
        # to reuse that integer. They're still caught via
        # _BITNET_ARCHITECTURE_PREFIXES regardless of file_type.
        _NAMES = {
            0:  "F32",  1:  "F16",  2:  "Q4_0", 3:  "Q4_1",
            6:  "Q5_0", 7:  "Q5_1", 8:  "Q8_0", 9:  "Q8_1",
            10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
            14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K", 19: "Q8_K", 40: "I2_S",
        }
        return _NAMES.get(ft, f"TYPE_{ft}")


# =============================================================================
# PRISM: inline model classifier for non-GGUF paths (BUG 2 fix)
# =============================================================================
_PACKED_ARCHITECTURES = frozenset({
    "BitNetForCausalLM",
    "BitNetModel",
    "BitNetBEForCausalLM",
})

_CONVERTIBLE_ARCHITECTURES = frozenset({
    "LlamaForCausalLM", "LLaMAForCausalLM", "MistralForCausalLM",
    "FalconForCausalLM", "MptForCausalLM", "GPT2LMHeadModel",
    "BloomForCausalLM", "OPTForCausalLM", "RWForCausalLM", "PhiForCausalLM",
})


class PRISMRouter:
    """
    Inline PRISM classification for HF model directories.
    Reads config.json + safetensors header without external dependencies.
    Returns routing hints used by AdaptiveModelRouter.
    """

    @staticmethod
    def classify(model_path: str) -> Tuple[str, str]:
        """
        Returns (prism_class, architecture).
        prism_class: "packed" | "bf16" | "pre-built" | "gguf" | "unknown"
        architecture: "BitNetForCausalLM" | "LlamaForCausalLM" | "" | ...
        """
        p = Path(model_path)

        # Direct .gguf file
        if p.is_file() and p.suffix.lower() == ".gguf":
            return "gguf", ""

        # Directory with built GGUF
        if p.is_dir() and list(p.glob("ggml-model-*.gguf")):
            return "pre-built", ""

        cfg_path = p / "config.json" if p.is_dir() else p.parent / "config.json"
        if not cfg_path.exists():
            return "unknown", ""

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return "unknown", ""

        arch = cfg.get("architectures", [""])[0]

        # Packed model detection
        qc = cfg.get("quantization_config", {})
        if qc and ("bitlinear" in json.dumps(cfg).lower()
                   or qc.get("quant_type") or qc.get("bits")):
            return "packed", arch

        if arch in _PACKED_ARCHITECTURES:
            return "packed", arch

        # Safetensors dtype check
        if p.is_dir():
            st_files = sorted(p.glob("*.safetensors"))
            if st_files:
                dtype = PRISMRouter._read_dtype(st_files[0])
                if dtype == "U8":
                    return "packed", arch
                if dtype in ("BF16", "F32", "F16"):
                    return "bf16", arch

        return "bf16", arch

    @staticmethod
    def _read_dtype(path: Path) -> Optional[str]:
        try:
            with path.open("rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                if hdr_len > 100 * 1024 * 1024:
                    return None
                hdr = json.loads(f.read(hdr_len))
            counts: Dict[str, int] = {}
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                dt = v.get("dtype", "").upper()
                if dt:
                    counts[dt] = counts.get(dt, 0) + 1
            if not counts:
                return None
            if "U8" in counts:
                return "U8"
            return max(counts, key=lambda k: counts[k])
        except Exception:
            return None


# =============================================================================
# Environment status
# =============================================================================
@dataclass
class EnvironmentStatus:
    overall_ok:          bool  = False
    transformers_ok:     bool  = False
    transformers_version: str  = ""
    transformers_error:  Optional[str] = None
    torch_ok:            bool  = False
    torch_version:       str   = ""
    torch_error:         Optional[str] = None
    llama_cpp_ok:        bool  = False
    llama_cpp_version:   str   = ""
    bitlinear_available: bool  = False
    spectre_mode:        str   = "none"
    gguf_tl1_available:  bool  = False


class UnifiedEnvironmentValidator:
    """Validates the Python environment for inference readiness."""

    # Cached result + timestamp for lazy re-validation
    _cache: Optional[EnvironmentStatus] = None
    _cache_t: float = 0.0
    _cache_ttl: float = 300.0   # re-validate every 5 minutes
    _lock = threading.Lock()

    @classmethod
    def validate(cls, raise_on_error: bool = False) -> EnvironmentStatus:
        with cls._lock:
            if cls._cache and (time.monotonic() - cls._cache_t) < cls._cache_ttl:
                return cls._cache

        status = EnvironmentStatus()

        # ── transformers (BUG 7 fix: accept dev/pinned commit versions) ──
        try:
            import transformers
            ver = transformers.__version__
            status.transformers_version = ver
            # Accept released >=4.46.3,<5.0 OR dev/pinned-commit builds
            is_dev = any(x in ver for x in ("dev", "post", "+"))
            if is_dev:
                status.transformers_ok = True
            else:
                from packaging.version import Version
                v = Version(ver)
                status.transformers_ok = (
                    Version("4.46.3") <= v < Version("5.0.0")
                )
            if not status.transformers_ok:
                status.transformers_error = (
                    f"transformers {ver} outside >=4.46.3,<5.0.0 "
                    f"(not a dev/pinned build either)"
                )
        except ImportError as exc:
            status.transformers_error = str(exc)
        except Exception as exc:
            status.transformers_error = f"transformers check error: {exc}"

        # ── torch ────────────────────────────────────────────────────────
        try:
            import torch
            status.torch_version = torch.__version__
            status.torch_ok      = True
        except ImportError as exc:
            status.torch_error = str(exc)

        # ── llama-cpp-python ─────────────────────────────────────────────
        try:
            import llama_cpp
            status.llama_cpp_ok      = True
            status.llama_cpp_version = getattr(llama_cpp, "__version__", "?")
        except ImportError:
            pass

        # ── gguf TL1/TL2 ─────────────────────────────────────────────────
        try:
            import gguf
            _ = gguf.GGMLQuantizationType.TL1
            _ = gguf.GGMLQuantizationType.TL2
            status.gguf_tl1_available = True
        except (ImportError, AttributeError):
            pass

        # ── BitLinear ────────────────────────────────────────────────────
        try:
            from local_llm_core.compatibility.transformers_compat import (
                BitLinearCompatibilityLayer, StateFileReader,
            )
            status.bitlinear_available = BitLinearCompatibilityLayer.is_bitlinear_importable()
            state_data = StateFileReader.load() or {}
            status.spectre_mode = state_data.get("spectre_mode", "none")
        except Exception:
            pass

        status.overall_ok = status.transformers_ok and status.torch_ok

        with cls._lock:
            cls._cache   = status
            cls._cache_t = time.monotonic()

        if raise_on_error and not status.overall_ok:
            errors = [e for e in (
                status.transformers_error, status.torch_error
            ) if e]
            raise RuntimeError(
                "Environment validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )
        return status

    @classmethod
    def invalidate_cache(cls) -> None:
        with cls._lock:
            cls._cache = None


# =============================================================================
# BitNetState + BitNetStateManager
# =============================================================================
@dataclass
class BitNetState:
    available:           bool  = False
    bitnet_dir:          str   = ""
    model_path:          str   = ""
    model_name:          str   = ""
    gguf_path:           str   = ""
    quant_type:          str   = ""
    quant_embd:          bool  = False
    llama_cli_binary:    str   = ""
    llama_server_binary: str   = ""
    run_inference_script: str  = ""
    verify_passed:       bool  = False
    setup_version:       str   = ""
    setup_timestamp:     str   = ""
    prism_class:         str   = ""
    prism_arch_compat:   str   = ""
    spectre_mode:        str   = ""
    spectre_gguf_py_path: str  = ""
    atlas_threads:       int   = 4
    atlas_gpu_backend:   str   = "cpu"
    python_bin:          str   = ""


class BitNetStateManager:
    """
    Loads and caches bitnet_state.json written by bitnet_setup.sh.
    v9 fix: real-home-aware path (BUG 11), mtime-based cache invalidation
    so stale state is never served after local-llm-setup reruns (BUG 10).
    """

    _cache: Optional[BitNetState] = None
    _cache_path: Optional[str]    = None
    _cache_mtime: float           = 0.0
    _lock = threading.Lock()

    @classmethod
    def _get_candidate_paths(cls) -> List[Path]:
        """Real-home-aware state file search (matches bitnet_setup.sh logic)."""
        candidates: List[Path] = []

        # Real home detection (mirrors install.sh / bitnet_setup.sh)
        sudo_user = os.environ.get("SUDO_USER", "")
        real_home_env = os.environ.get("REAL_HOME", "")
        if real_home_env and Path(real_home_env).is_dir():
            candidates.append(Path(real_home_env) / ".config" / "local-llm" / "bitnet_state.json")
        if sudo_user and sudo_user != "root":
            try:
                import pwd
                home = Path(pwd.getpwnam(sudo_user).pw_dir)
                candidates.append(home / ".config" / "local-llm" / "bitnet_state.json")
            except Exception:
                pass
        candidates.append(Path.home() / ".config" / "local-llm" / "bitnet_state.json")
        candidates.append(Path("/root") / ".config" / "local-llm" / "bitnet_state.json")

        return candidates

    @classmethod
    def load(cls, force_reload: bool = False) -> BitNetState:
        with cls._lock:
            # Find first existing candidate
            state_path: Optional[Path] = None
            for c in cls._get_candidate_paths():
                if c.is_file():
                    state_path = c
                    break

            if state_path is None:
                return BitNetState()

            # BUG 10 fix: mtime-based invalidation
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                return BitNetState()

            if (not force_reload
                    and cls._cache is not None
                    and cls._cache_path == str(state_path)
                    and cls._cache_mtime == mtime):
                return cls._cache

            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.debug("Failed to read bitnet_state.json: %s", exc)
                return BitNetState()

            def _path_ok(val: str) -> bool:
                return bool(val and Path(val).exists())

            def _exec_ok(val: str) -> bool:
                return bool(val and Path(val).is_file() and os.access(val, os.X_OK))

            state = BitNetState(
                available            = bool(data.get("verify_passed")),
                bitnet_dir           = data.get("bitnet_dir", ""),
                model_path           = data.get("model_path", ""),
                model_name           = data.get("model_name", ""),
                gguf_path            = data.get("gguf_path", ""),
                quant_type           = data.get("quant_type", ""),
                quant_embd           = bool(data.get("quant_embd", False)),
                llama_cli_binary     = data.get("llama_cli_binary", ""),
                llama_server_binary  = data.get("llama_server_binary", ""),
                run_inference_script = data.get("run_inference_script", ""),
                verify_passed        = bool(data.get("verify_passed")),
                setup_version        = data.get("setup_version", ""),
                setup_timestamp      = data.get("setup_timestamp", ""),
                prism_class          = data.get("prism_class", ""),
                prism_arch_compat    = data.get("prism_arch_compat", ""),
                spectre_mode         = data.get("spectre_mode", ""),
                spectre_gguf_py_path = data.get("spectre_gguf_py_path", ""),
                atlas_threads        = int(data.get("atlas_threads") or 4),
                atlas_gpu_backend    = data.get("atlas_gpu_backend", "cpu"),
                python_bin           = data.get("python_bin", ""),
            )

            # Validate critical binary paths
            if state.llama_server_binary and not _exec_ok(state.llama_server_binary):
                log.debug("llama_server_binary not executable: %s", state.llama_server_binary)
                state.llama_server_binary = ""
            if state.llama_cli_binary and not _exec_ok(state.llama_cli_binary):
                log.debug("llama_cli_binary not executable: %s", state.llama_cli_binary)
                state.llama_cli_binary = ""
            if state.gguf_path and not _path_ok(state.gguf_path):
                log.debug("gguf_path not found: %s", state.gguf_path)
                state.gguf_path = ""

            # Re-evaluate availability after path validation
            state.available = (
                state.verify_passed
                and bool(state.gguf_path)
                and (bool(state.llama_server_binary) or bool(state.llama_cli_binary)
                     or bool(state.run_inference_script))
            )

            cls._cache      = state
            cls._cache_path = str(state_path)
            cls._cache_mtime = mtime
            log.debug("BitNetState loaded from %s (available=%s)", state_path, state.available)
            return state

    @classmethod
    def invalidate(cls) -> None:
        with cls._lock:
            cls._cache = None
            cls._cache_mtime = 0.0


# =============================================================================
# AdaptiveModelRouter — AQAMR v9.0.0
# =============================================================================
class AdaptiveModelRouter:
    """
    Routes a model path to the best available inference backend.

    Routing priority (v9):
      1. State-file GGUF path (if state matches and GGUF exists) → CPP_SERVER
      2. GGUF file/dir → parse header → BitNet or Standard
      3. HF directory → PRISM → packed (transformers) or bf16 (convertible)
      4. Unknown → best available fallback
    """

    def __init__(self, bitnet_state: Optional[BitNetState] = None) -> None:
        self._bitnet_state = bitnet_state
        self._detector     = GGUFTypeDetector()
        # cache keyed on (path, state_mtime) — auto-invalidated
        self._cache: Dict[str, Tuple[GGUFHeader, InferenceBackend, float]] = {}
        self._lock = threading.Lock()

    def _get_state(self) -> BitNetState:
        if self._bitnet_state is not None:
            return self._bitnet_state
        return BitNetStateManager.load()

    def route(
        self, model_path: str
    ) -> Tuple[GGUFHeader, InferenceBackend, Dict[str, Any]]:
        """
        Route model_path to the best backend.
        Returns (header, backend, params).
        """
        path_str = str(Path(model_path).expanduser().resolve())
        state    = self._get_state()
        state_mtime = BitNetStateManager._cache_mtime

        with self._lock:
            cached = self._cache.get(path_str)
            # invalidate if state file changed
            if cached and cached[2] == state_mtime:
                hdr, backend, _ = cached
                params = self._build_params(hdr, backend, path_str, state)
                return hdr, backend, params

        # ── Priority 1: state-file GGUF shortcut ──────────────────────────
        # If the state file has a valid GGUF for the same model, use it
        # directly — avoids re-scanning directories.
        shortcut_gguf = self._resolve_gguf_from_state(path_str, state)
        if shortcut_gguf:
            hdr = GGUFTypeDetector.detect(shortcut_gguf)
            if not hdr.parse_error:
                backend = self._select_cpp_backend(state)
                with self._lock:
                    self._cache[path_str] = (hdr, backend, state_mtime)
                params = self._build_params(hdr, backend, path_str, state,
                                            gguf_override=shortcut_gguf)
                log.info("AQAMR (state-file shortcut): %s → %s via %s",
                         Path(path_str).name, backend.value, Path(shortcut_gguf).name)
                return hdr, backend, params

        # ── Priority 2: GGUF file or directory with GGUF files ────────────
        p = Path(path_str)
        is_gguf = (p.is_file() and p.suffix.lower() == ".gguf") or \
                  (p.is_dir() and bool(list(p.glob("*.gguf"))))

        if is_gguf:
            hdr = GGUFTypeDetector.detect(path_str)
            if not hdr.parse_error:
                backend = self._select_backend(hdr, path_str, state)
                with self._lock:
                    self._cache[path_str] = (hdr, backend, state_mtime)
                params = self._build_params(hdr, backend, path_str, state)
                log.info("AQAMR (GGUF): %s → type=%s backend=%s",
                         Path(path_str).name, hdr.model_type.value, backend.value)
                return hdr, backend, params

        # ── Priority 3: HF directory → PRISM ─────────────────────────────
        if p.is_dir() or "/" in model_path or path_str.startswith(str(Path.home())):
            hdr, backend = self._route_hf_directory(path_str, state)
            with self._lock:
                self._cache[path_str] = (hdr, backend, state_mtime)
            params = self._build_params(hdr, backend, path_str, state)
            log.info("AQAMR (PRISM): %s → prism=%s backend=%s",
                     Path(path_str).name,
                     hdr.file_type_name, backend.value)
            return hdr, backend, params

        # ── Priority 4: HF model ID (download) ───────────────────────────
        # Treat as transformers path
        hdr     = GGUFHeader(source_path=path_str, model_type=GGUFModelType.UNKNOWN,
                              file_type_name="hf_model_id")
        backend = InferenceBackend.BITNET_TRANSFORMERS
        with self._lock:
            self._cache[path_str] = (hdr, backend, state_mtime)
        params = self._build_params(hdr, backend, path_str, state)
        log.info("AQAMR (HF ID): %s → %s", model_path, backend.value)
        return hdr, backend, params

    def _resolve_gguf_from_state(
        self, model_path: str, state: BitNetState
    ) -> Optional[str]:
        """
        BUG 3 fix: resolve the state GGUF for a given model_path.
        The original compared Path(state.gguf_path).resolve() == Path(model_path).resolve()
        which almost never matches (GGUF file ≠ safetensors directory).
        Now uses name-based and directory-containment matching.
        """
        if not state.available or not state.gguf_path:
            return None
        if not Path(state.gguf_path).is_file():
            return None

        mp   = Path(model_path)
        sguf = Path(state.gguf_path)

        # Exact match (both are the same file)
        try:
            if mp.is_file() and mp.resolve() == sguf.resolve():
                return state.gguf_path
        except Exception:
            pass

        # Directory containment: state GGUF lives inside model_path dir
        try:
            sguf.relative_to(mp)
            return state.gguf_path
        except ValueError:
            pass

        # Name-based: model directory name matches state model_name
        mp_name = mp.name.lower().replace("-", "").replace("_", "")
        st_name = (state.model_name or "").lower().replace("-", "").replace("_", "")
        if st_name and mp_name and (mp_name in st_name or st_name in mp_name):
            return state.gguf_path

        # State model_path is a parent of model_path
        try:
            if state.model_path:
                mp.resolve().relative_to(Path(state.model_path).resolve())
                return state.gguf_path
        except (ValueError, Exception):
            pass

        return None

    def _route_hf_directory(
        self, model_path: str, state: BitNetState
    ) -> Tuple[GGUFHeader, InferenceBackend]:
        """
        PRISM classification for HF directories.
        Packed BitNetForCausalLM → BITNET_TRANSFORMERS (not UNKNOWN).
        Pre-built GGUF inside dir → CPP_SERVER/CPP_PROCESS.
        """
        prism_class, arch = PRISMRouter.classify(model_path)

        hdr = GGUFHeader(
            source_path=model_path,
            model_type=GGUFModelType.UNKNOWN,
            file_type_name=f"prism:{prism_class}",
        )

        if prism_class == "pre-built":
            # Directory has a built GGUF — use it
            built_ggufs = sorted(
                Path(model_path).glob("ggml-model-*.gguf"),
                key=lambda f: f.stat().st_size, reverse=True,
            )
            if built_ggufs:
                sub_hdr = GGUFTypeDetector.detect(str(built_ggufs[0]))
                if not sub_hdr.parse_error:
                    backend = self._select_cpp_backend(state)
                    sub_hdr.file_type_name = f"prism:{prism_class}+{sub_hdr.file_type_name}"
                    return sub_hdr, backend

        if prism_class == "packed":
            hdr.file_type_name = f"prism:packed arch:{arch}"
            hdr.model_type     = GGUFModelType.BITNET
            backend = InferenceBackend.BITNET_TRANSFORMERS
            log.info(
                "AQAMR PRISM: packed model (%s) → BITNET_TRANSFORMERS "
                "(converter not compatible, needs pinned transformers commit)",
                arch,
            )
            return hdr, backend

        if prism_class == "bf16":
            # Standard HF model — use llama-cpp-python or transformers
            hdr.file_type_name = f"prism:bf16 arch:{arch}"
            hdr.model_type     = GGUFModelType.STANDARD
            try:
                import llama_cpp   # noqa: F401
                return hdr, InferenceBackend.LLAMA_CPP_PYTHON
            except ImportError:
                pass
            return hdr, InferenceBackend.BITNET_TRANSFORMERS

        # Unknown
        hdr.file_type_name = "prism:unknown"
        return hdr, InferenceBackend.UNKNOWN

    def _select_cpp_backend(self, state: BitNetState) -> InferenceBackend:
        """Select BITNET_CPP_SERVER or BITNET_CPP_PROCESS from state."""
        if state.llama_server_binary and Path(state.llama_server_binary).is_file():
            return InferenceBackend.BITNET_CPP_SERVER
        if state.llama_cli_binary and Path(state.llama_cli_binary).is_file():
            return InferenceBackend.BITNET_CPP_PROCESS
        if state.run_inference_script and Path(state.run_inference_script).is_file():
            return InferenceBackend.BITNET_CPP_PROCESS
        return InferenceBackend.BITNET_TRANSFORMERS

    def _select_backend(
        self, header: GGUFHeader, model_path: str, state: BitNetState
    ) -> InferenceBackend:
        """Select backend for a parsed GGUF file."""

        if header.model_type == GGUFModelType.BITNET:
            if state and state.available:
                return self._select_cpp_backend(state)
            log.warning(
                "BitNet GGUF found but bitnet.cpp not available — "
                "falling back to transformers (no efficiency gains)"
            )
            return InferenceBackend.BITNET_TRANSFORMERS

        elif header.model_type == GGUFModelType.STANDARD:
            try:
                import llama_cpp   # noqa: F401
                return InferenceBackend.LLAMA_CPP_PYTHON
            except ImportError:
                pass
            # fall back to any usable binary the
            # state file exposes. Previously only checked llama_server,
            # skipping llama_cli and run_inference — which _select_cpp_backend
            # already handles for BitNet models. A standard GGUF can be
            # served by any of those binaries too, so both branches should
            # agree on which state-file options they consider.
            if state and state.available:
                return self._select_cpp_backend(state)
            return InferenceBackend.UNKNOWN

        else:
            try:
                import llama_cpp   # noqa: F401
                return InferenceBackend.LLAMA_CPP_PYTHON
            except ImportError:
                return InferenceBackend.UNKNOWN

    def _build_params(
        self,
        header:        GGUFHeader,
        backend:       InferenceBackend,
        model_path:    str,
        state:         BitNetState,
        gguf_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the params dict for a routing result.
        uses gguf_override when state GGUF was resolved,
                   avoids the broken == comparison.
        BitNet transformers always uses "cpu" device.
        """
        params: Dict[str, Any] = {
            "model_path":    model_path,
            "model_type":    header.model_type.value,
            "quant_type":    header.file_type_name,
            "backend":       backend.value,
            "atlas_threads": state.atlas_threads if state else 4,
        }

        actual_gguf = gguf_override or model_path

        if backend == InferenceBackend.BITNET_CPP_SERVER:
            params.update({
                "server_binary": state.llama_server_binary if state else "",
                "gguf_path":     actual_gguf,
                "port":          int(os.environ.get("BITNET_PORT", "18080")),
                "host":          os.environ.get("BITNET_HOST", "127.0.0.1"),
                "n_threads":     state.atlas_threads if state else 4,
            })

        elif backend == InferenceBackend.BITNET_CPP_PROCESS:
            params.update({
                "llama_cli_binary":    state.llama_cli_binary if state else "",
                "run_inference_script": state.run_inference_script if state else "",
                "gguf_path":           actual_gguf,
                "bitnet_dir":          state.bitnet_dir if state else "",
                "n_threads":           state.atlas_threads if state else 4,
            })

        elif backend == InferenceBackend.LLAMA_CPP_PYTHON:
            params.update({
                "llama_kwargs": {
                    "model_path":  actual_gguf,
                    "n_ctx":       int(os.environ.get("LLM_CTX_SIZE", "4096")),
                    "n_gpu_layers": int(os.environ.get("LLM_GPU_LAYERS", "-1")),
                    "verbose":     False,
                    "n_threads":   state.atlas_threads if state else 4,
                }
            })

        elif backend == InferenceBackend.BITNET_TRANSFORMERS:
            # always CPU — BitNet ternary kernels are CPU SIMD
            #
            # previously this unconditionally used state.model_path
            # whenever it was set, even if the state file pointed at a
            # COMPLETELY DIFFERENT BitNet install than the one the caller
            # asked for. That silently loaded the wrong checkpoint — the
            # caller passed one model_path and got another model's weights.
            # Mirrors the containment logic already used by
            # _resolve_gguf_from_state().
            hf_path = str(Path(model_path))
            if state and state.model_path:
                try:
                    Path(model_path).resolve().relative_to(
                        Path(state.model_path).resolve()
                    )
                    # state.model_path is an ancestor of the requested path
                    # — using it is safe (it's the same model, just a
                    # broader root).
                    hf_path = state.model_path
                except (ValueError, OSError):
                    # state points at a different model tree — ignore it.
                    pass
            params.update({
                "hf_model_path": hf_path,
                "device":        "cpu",
                "n_threads":     state.atlas_threads if state else 4,
            })

        return params

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_paths": list(self._cache.keys()),
                "cache_size":   len(self._cache),
            }


# =============================================================================
# Module-level singleton router
# =============================================================================
_router: Optional[AdaptiveModelRouter] = None
_router_lock = threading.Lock()


def get_router() -> AdaptiveModelRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = AdaptiveModelRouter()
        return _router


def _invalidate_router_cache() -> None:
    """Call after local-llm-setup reruns to force fresh routing."""
    global _router
    with _router_lock:
        if _router:
            _router.clear_cache()
    BitNetStateManager.invalidate()
    UnifiedEnvironmentValidator.invalidate_cache()


# =============================================================================
# Convenience functions (module-level API)
# =============================================================================

def classify_gguf(model_path: str) -> GGUFHeader:
    """BUG 4 fix: detect() is now @staticmethod — no instantiation needed."""
    return GGUFTypeDetector.detect(model_path)


def route_model(model_path: str) -> Tuple[InferenceBackend, Dict[str, Any]]:
    _, backend, params = get_router().route(model_path)
    return backend, params


def auto_detect_backend() -> str:
    """
    Return a string identifying the best available BitNet backend.
    BUG 12 fix: returns a STRING (not dict) to match bitnet_loader.auto_detect_backend().
    The dict version is available as auto_detect_backend_info().
    """
    state = BitNetStateManager.load()
    if state.available:
        if state.llama_server_binary and Path(state.llama_server_binary).is_file():
            return "bitnet_cpp"
        if state.llama_cli_binary and Path(state.llama_cli_binary).is_file():
            return "bitnet_cpp"

    try:
        from local_llm_core.compatibility.transformers_compat import BitLinearCompatibilityLayer
        if BitLinearCompatibilityLayer.is_bitlinear_importable():
            return "transformers_bitlinear"
    except Exception:
        pass

    try:
        import transformers  # noqa: F401
        return "transformers_standard"
    except ImportError:
        pass

    return "none"


def auto_detect_backend_info() -> Dict[str, Any]:
    """
    Return a rich dict of all backend availability info.
    Renamed from auto_detect_backend() to avoid collision with
    bitnet_loader.auto_detect_backend() which returns a string (BUG 12 fix).
    """
    state      = BitNetStateManager.load()
    env_status = UnifiedEnvironmentValidator.validate()
    llama_cpp_ok  = False
    llama_cpp_ver = None
    try:
        import llama_cpp
        llama_cpp_ok  = True
        llama_cpp_ver = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        pass

    return {
        "best_backend": auto_detect_backend(),
        "bitnet_cpp": {
            "available":       state.available,
            "bitnet_dir":      state.bitnet_dir,
            "gguf_path":       state.gguf_path,
            "quant_type":      state.quant_type,
            "quant_embd":      state.quant_embd,
            "llama_cli":       state.llama_cli_binary,
            "llama_server":    state.llama_server_binary,
            "verify_passed":   state.verify_passed,
            "setup_timestamp": state.setup_timestamp,
            "prism_class":     state.prism_class,
            "spectre_mode":    state.spectre_mode,
        },
        "llama_cpp_python": {
            "available": llama_cpp_ok,
            "version":   llama_cpp_ver,
        },
        "transformers": {
            "version": env_status.transformers_version,
            "ok":      env_status.transformers_ok,
            "error":   env_status.transformers_error,
        },
        "torch": {
            "version": env_status.torch_version,
            "ok":      env_status.torch_ok,
        },
        "bitlinear": {
            "available": env_status.bitlinear_available,
        },
        "gguf_tl1": env_status.gguf_tl1_available,
        "unified_constraint": "transformers>=4.46.3,<5.0.0",
        "overall_ok":         env_status.overall_ok,
    }


def get_bitnet_state() -> BitNetState:
    return BitNetStateManager.load()


def validate_environment(raise_on_error: bool = False) -> EnvironmentStatus:
    return UnifiedEnvironmentValidator.validate(raise_on_error)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# Module initialisation — BUG 9 fix: lazy, non-blocking
# =============================================================================
def _on_import() -> None:
    """
    BUG 9 fix: only load state file (fast, JSON read).
    Do NOT import transformers/torch here — too slow for startup.
    Environment validation is deferred until first actual call.
    """
    try:
        BitNetStateManager.load()
        # Apply SPECTRE again after state file is loaded
        _apply_spectre()
    except Exception as exc:
        log.debug("model_unified_env init: %s", exc)


_on_import()
