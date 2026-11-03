from __future__ import annotations

# engine/bitnet/bitnet_loader.py — Production BitNet Model Loader  v1.0.0
# =========================================================================
# v1.0.0 — Complete rewrite fixing all critical bugs from v1.0.0
#
# Addd a documentation about the whoe script
#
# New features in v1.0.0:
#   • auto_detect_backend() module-level function (used by startup checks)
#   • GGUFFirstWrapper — lightweight wrapper for bitnet.cpp path (no RAM model)
#   • PRISMModelClassifier — inline safetensors dtype + config.json analysis
#   • BitLinear guard with actionable error message + auto-install option
#   • State-file-driven device selection (atlas_gpu_backend from bitnet_state.json)
#   • GGUF memory check (file size * 1.1, not safetensors * 1.5)
#   • Full SPECTRE PYTHONPATH application at loader init
#   • Packed model loads correctly when pinned transformers is installed
#   • bf16/float model loads use bfloat16 (correct for BitNet training dtype)
#   • load() returns consistent LoadResult namedtuple for type safety
# =========================================================================

import asyncio
import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import psutil
import torch

log = logging.getLogger(__name__)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CRITICAL: Import compatibility layer BEFORE any transformers imports       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
from ...compatibility.transformers_compat import (
    TransformersCompatibilityManager,
    BitLinearCompatibilityLayer,
    StateFileReader,
    BitNetModelLoader,
    LoadResult,
)

# Apply SPECTRE PYTHONPATH immediately at module import time so that any
# subsequent gguf imports (including TL1/TL2 checks) work correctly
StateFileReader.apply_spectre_pythonpath()

# ── Transformers — Tier 2/3 only (Tier 1 bitnet.cpp path never needs this) ──
# Wrapped in try/except because:
#   • transformers 4.52.x dev builds fail with PYTHONOPTIMIZE=1 (strips docstrings,
#     @replace_return_docstrings calls func.__doc__.split() → NoneType.split)
#   • The pinned commit (096f25ae) may not be installed
#   • Tier 1 (bitnet.cpp + GGUF) is fully independent — it must remain available
#     even when the transformers import fails completely.
# If import fails, _TRANSFORMERS_AVAILABLE=False and Tier 2/3 is disabled with a
# clear error; Tier 1 continues unaffected.
try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    _TRANSFORMERS_AVAILABLE = True
except Exception as _tf_import_exc:        # noqa: BLE001
    _TRANSFORMERS_AVAILABLE = False
    # Stub out names so module-level code referencing them doesn't NameError.
    AutoConfig          = None             # type: ignore[assignment,misc]
    AutoModelForCausalLM = None           # type: ignore[assignment,misc]
    AutoTokenizer       = None            # type: ignore[assignment,misc]
    BitsAndBytesConfig  = None            # type: ignore[assignment,misc]
    PreTrainedModel     = None            # type: ignore[assignment,misc]
    PreTrainedTokenizer = None            # type: ignore[assignment,misc]
    _tf_import_exc_msg  = str(_tf_import_exc)
    log.warning(
        "transformers import failed — BitNet Tier 1 (bitnet.cpp / GGUF) unaffected; "
        "Tier 2/3 (transformers BitLinear path) unavailable. "
        "Cause: %s. "
        "If you need Tier 2/3, ensure transformers is installed without "
        "PYTHONOPTIMIZE and with the pinned commit (see local-llm-setup).",
        _tf_import_exc_msg,
    )

from .bitnet_engine import BitNetInferenceWrapper
from ..utils.config_loader import BitNetSystemConfig, BitNetProfileConfig

# ── State-aware router ─────────────────────────────────────────────────────
from ..adaptive_awareness.model_unified_env import (
    AdaptiveModelRouter,
    BitNetState,
    BitNetStateManager,
    InferenceBackend,
    get_bitnet_state,
)

# Optional ONNX support
try:
    import onnxruntime as ort   # noqa: F401
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# =============================================================================
# Prometheus Metrics — SINGLETON (fix for duplicate registration)
# =============================================================================
_PROM_METRICS: Dict[str, Any] = {}

def _setup_metrics() -> Dict[str, Any]:
    """
    Create or return cached Prometheus metrics (singleton).
    This prevents duplicate metric registration when multiple BitNetLoader
    instances are created (e.g., during server startup or model reloads).
    """
    global _PROM_METRICS
    if _PROM_METRICS:
        return _PROM_METRICS
    try:
        from prometheus_client import Counter, Gauge, Histogram
        _PROM_METRICS = {
            "load_total":    Counter("bitnet_load_total", "BitNet model loads",
                                    ["model_id", "status"]),
            "unload_total":  Counter("bitnet_unload_total", "BitNet model unloads",
                                    ["model_id"]),
            "loaded_models": Gauge("bitnet_loaded_models", "Number of loaded BitNet models"),
            "vram_used_mb":  Gauge("bitnet_vram_used_mb", "VRAM used by BitNet models"),
            "load_time_s":   Histogram("bitnet_load_time_seconds", "Model load time",
                                       buckets=(1, 5, 10, 30, 60, 120, 300)),
        }
    except ImportError:
        _PROM_METRICS = {}
    return _PROM_METRICS

_METRICS: Dict[str, Any] = _setup_metrics()


# =============================================================================
# Backend determination constants
# =============================================================================
BACKEND_BITNET_CPP  = "bitnet_cpp"
BACKEND_TRANSFORMERS_BITLINEAR = "transformers_bitlinear"   # packed + pinned commit
BACKEND_TRANSFORMERS_STANDARD  = "transformers_standard"   # bf16 / standard
BACKEND_GGUF = "gguf"                                       # llama-cpp-python direct

#: These architectures cannot be converted by convert-hf-to-gguf-bitnet.py
#: and require the transformers BitLinear path instead of bitnet.cpp
PACKED_ARCHITECTURES = frozenset({
    "BitNetForCausalLM",
    "BitNetModel",
    "BitNetBEForCausalLM",
})

#: Architectures known to work with the converter (bf16/float weights)
CONVERTIBLE_ARCHITECTURES = frozenset({
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "MistralForCausalLM",
    "FalconForCausalLM",
    "MptForCausalLM",
    "GPT2LMHeadModel",
    "BloomForCausalLM",
    "OPTForCausalLM",
    "RWForCausalLM",
    "PhiForCausalLM",
})


# =============================================================================
# PRISM: Inline model classifier (no external script dependency)
# =============================================================================
class PRISMModelClassifier:
    """
    Inline model type classification for bitnet_loader.py.
    Reads safetensors header + config.json using only stdlib.
    Returns: "packed" | "bf16" | "pre-built" | "gguf" | "unknown"
    """

    @staticmethod
    def classify(model_path: str) -> str:
        p = Path(model_path)

        # Direct .gguf file
        if p.is_file() and p.suffix.lower() == ".gguf":
            return "gguf"

        # Already-built quantized GGUF in directory
        if p.is_dir() and list(p.glob("ggml-model-*.gguf")):
            return "pre-built"

        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        if not cfg_path.exists():
            return "unknown"

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return "unknown"

        # Check quantization_config (packed models have this with bitlinear keys)
        qc = cfg.get("quantization_config", {})
        content_lower = json.dumps(cfg).lower()
        if qc and ("bitlinear" in content_lower or qc.get("quant_type") or qc.get("bits")):
            return "packed"

        # Check architecture
        archs = cfg.get("architectures", [])
        if archs and archs[0] in PACKED_ARCHITECTURES:
            return "packed"

        # Read safetensors header
        if p.is_dir():
            st_files = sorted(p.glob("*.safetensors"))
            if st_files:
                dtype = PRISMModelClassifier._read_dominant_dtype(st_files[0])
                if dtype == "U8":
                    return "packed"
                if dtype in ("BF16", "F32", "F16"):
                    return "bf16"

        return "bf16"  # safe default for unrecognised HF checkpoints

    @staticmethod
    def _read_dominant_dtype(path: Path) -> Optional[str]:
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
            # U8 in ANY position indicates packed ternary weights
            if "U8" in counts:
                return "U8"
            return max(counts, key=lambda k: counts[k])
        except Exception:
            return None

    @staticmethod
    def get_architecture(model_path: str) -> str:
        p = Path(model_path)
        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return cfg.get("architectures", [""])[0]
        except Exception:
            return ""


# =============================================================================
# BitNetModelConfig (enhanced with state-aware defaults + PRISM routing)
# =============================================================================
@dataclass
class BitNetModelConfig:
    """BitNet model configuration with validation and state-aware defaults."""
    model_id: str
    model_name_or_path: Optional[str] = None
    quantization: Literal["1bit", "4bit", "8bit", "mixed"] = "1bit"
    default_model_id: str = "bitnet-default"
    quantized: bool = True
    device: str = "cpu"   # v1: default to cpu (BitNet ternary kernels are CPU-only)
    dtype: torch.dtype = torch.bfloat16   # v1: bfloat16 matches BitNet training dtype
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False   # v1 false by default (requires flash_attn)
    attn_implementation: Optional[str] = None
    max_position_embeddings: int = 4096
    rope_scaling: Optional[Dict[str, Any]] = None
    use_custom_kernels: bool = True
    kernel_backend: Optional[Literal["cuda", "rocm", "metal", "cpu"]] = None
    num_gpu_layers: int = 0    # v1: 0 for BitNet (always CPU SIMD kernels)
    gpu_memory_fraction: float = 0.9
    trust_remote_code: bool = True
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    offload_folder: Optional[str] = None
    autoload: bool = False
    priority: int = 10
    tags: List[str] = field(default_factory=list)
    # v1: explicit GGUF path (set when bitnet.cpp path is active)
    gguf_path: Optional[str] = None
    # v1.1: separate tokenizer directory for GGUF profiles.
    # When path is a .gguf file the safetensors dir cannot be used for the
    # tokenizer.  Set tokenizer_path to the HuggingFace checkpoint directory
    # (e.g. bitnet-b1.58-2B-4T) so AutoTokenizer can load from it separately.
    # None means: try model_name_or_path, then skip gracefully (llama-server
    # handles tokenization internally for all bitnet.cpp inference).
    tokenizer_path: Optional[str] = None
    # v1: loading backend decision
    loading_backend: str = BACKEND_TRANSFORMERS_STANDARD

    def __post_init__(self) -> None:
        if self.gpu_memory_fraction <= 0 or self.gpu_memory_fraction > 1.0:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        if self.num_gpu_layers < -1:
            raise ValueError("num_gpu_layers must be >= -1")
        if not self.default_model_id:
            self.default_model_id = self.model_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id":            self.model_id,
            "quantization":        self.quantization,
            "device":              self.device,
            "dtype":               str(self.dtype),
            "loading_backend":     self.loading_backend,
            "gguf_path":           self.gguf_path,
            "use_flash_attention_2": self.use_flash_attention_2,
            "num_gpu_layers":      self.num_gpu_layers,
            "trust_remote_code":   self.trust_remote_code,
        }

    @classmethod
    def from_state(cls, state: BitNetState) -> "BitNetModelConfig":
        """Create a BitNetModelConfig from a BitNetState (read from state file)."""
        # v9 FIX: prefer gguf_path when bitnet.cpp is verified
        if state.gguf_path and Path(state.gguf_path).is_file() and state.available:
            primary_path = state.gguf_path
            backend = BACKEND_BITNET_CPP
        else:
            primary_path = state.model_path or state.gguf_path or ""
            backend = BACKEND_TRANSFORMERS_STANDARD

        if not primary_path:
            raise ValueError("BitNet state has no model_path or gguf_path")

        quant = "1bit" if state.quant_type in ("i2_s", "tl1", "tl2") else "1bit"

        return cls(
            model_id=Path(primary_path).name,
            model_name_or_path=state.model_path or "",
            quantization=quant,
            device="cpu",   # BitNet always CPU
            dtype=torch.bfloat16,
            gguf_path=state.gguf_path if Path(state.gguf_path or "").is_file() else None,
            loading_backend=backend,
        )


# =============================================================================
# Hardware Detection
# =============================================================================
class HardwareDetector:
    @staticmethod
    def get_optimal_device(requested: str = "cpu") -> str:
        # BitNet ternary kernels are CPU-only (no GPU offload for BitNet weights)
        # Only the llama.cpp portion (non-BitNet layers) can use GPU
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested in ("mps", "cuda") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def detect_backend() -> Literal["cuda", "rocm", "metal", "cpu"]:
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                if "amd" in props.name.lower() or "gfx" in props.name.lower():
                    return "rocm"
            except Exception:
                pass
            return "cuda"
        if torch.backends.mps.is_available():
            return "metal"
        return "cpu"

    @staticmethod
    def get_vram_info(device_index: int = 0) -> Tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)
        try:
            total = torch.cuda.get_device_properties(device_index).total_memory
            free  = total - torch.cuda.memory_allocated(device_index)
            return (total // (1024 * 1024), free // (1024 * 1024))
        except Exception:
            return (0, 0)

    @staticmethod
    def can_fit_model(required_vram_mb: int) -> bool:
        _, free = HardwareDetector.get_vram_info()
        return free >= required_vram_mb

    @staticmethod
    def device_from_atlas_state() -> str:
        """Read atlas_gpu_backend from bitnet_state.json for device selection."""
        data = StateFileReader.load()
        if not data:
            return "cpu"
        backend = data.get("atlas_gpu_backend", "cpu")
        if backend in ("cuda",):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if backend in ("hip",):
            return "cuda" if torch.cuda.is_available() else "cpu"  # ROCm uses cuda device
        if backend in ("metal",):
            return "mps" if torch.backends.mps.is_available() else "cpu"
        return "cpu"


# =============================================================================
# GGUF path routing helpers
# =============================================================================
def _is_gguf_path(path: str) -> bool:
    """Return True if path is a .gguf file OR a dir with ggml-model-*.gguf inside."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return True
    if p.is_dir() and list(p.glob("ggml-model-*.gguf")):
        return True
    return False


def _resolve_gguf_from_dir(model_dir: str) -> Optional[str]:
    """Find the best quantized GGUF inside a model directory."""
    p = Path(model_dir)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)
    if not p.is_dir():
        return None
    # Priority: embed+quant > quant-only > any .gguf
    for pattern in ["ggml-model-*embed*.gguf", "ggml-model-*.gguf", "*.gguf"]:
        matches = sorted(p.glob(pattern), key=lambda f: f.stat().st_size, reverse=True)
        if matches:
            return str(matches[0])
    return None


def _check_bitlinear_available_or_raise(model_path: str) -> None:
    """
    Raise a clear, actionable error if BitLinear is not available
    and the model requires it (packed architecture).

    Also guards against the case where the transformers import failed
    entirely at module load time (e.g. PYTHONOPTIMIZE stripped docstrings
    and transformers.modeling_utils crashed on import).
    """
    # First guard: transformers itself failed to import
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "\n\n"
            "  transformers failed to import at server startup.\n"
            "  Tier 2/3 (transformers BitLinear / bf16 path) is unavailable.\n"
            "  Tier 1 (bitnet.cpp + GGUF) continues to work normally.\n"
            "\n"
            "  Most likely cause: PYTHONOPTIMIZE=1 strips docstrings; transformers\n"
            "  4.52.x dev builds call func.__doc__.split() at class-definition time.\n"
            "\n"
            "  Fix:\n"
            "    Remove PYTHONOPTIMIZE from /etc/systemd/system/local-llm-cloud.service\n"
            "    then: sudo systemctl daemon-reload && sudo systemctl restart local-llm-cloud\n"
            "\n"
            "  Alternatively use the bitnet.cpp path (recommended, no transformers needed):\n"
            "    Run local-llm-setup and use the bitnet-gguf profile.\n"
        )

    # Second guard: transformers available but BitLinear not in this build
    if BitLinearCompatibilityLayer.is_bitlinear_importable():
        return

    arch = PRISMModelClassifier.get_architecture(model_path)
    if arch not in PACKED_ARCHITECTURES:
        return  # Not a packed model — standard load will work

    from ...compatibility.transformers_compat import BITNET_PINNED_SPEC
    raise ImportError(
        f"\n\n"
        f"  BitLinear NOT available in transformers {BitLinearCompatibilityLayer._detection_cache}\n"
        f"  Model architecture '{arch}' requires BitLinear (packed 1.58-bit weights).\n"
        f"\n"
        f"  Fix — install the pinned transformers commit:\n"
        f"    pip install '{BITNET_PINNED_SPEC}'\n"
        f"\n"
        f"  Or auto-install via local-llm-core:\n"
        f"    from local_llm_core.compatibility.transformers_compat import ensure_bitlinear\n"
        f"    ensure_bitlinear()\n"
        f"\n"
        f"  Or use the bitnet.cpp path (recommended for performance):\n"
        f"    Download microsoft/BitNet-b1.58-2B-4T-gguf and re-run local-llm-setup\n"
    )


# =============================================================================
# Module-level convenience function (referenced by bitnet_setup.sh v9)
# =============================================================================
def auto_detect_backend() -> str:
    """
    Detect the best available BitNet backend.
    Returns one of: "bitnet_cpp" | "transformers_bitlinear" | "transformers_standard" | "none"

    Called by:  local-llm-core startup diagnostics
                bitnet_setup.sh summary: python -c "from local_llm_core.engine.bitnet import auto_detect_backend..."
    """
    # Tier 1: bitnet.cpp via state file
    if StateFileReader.is_bitnet_cpp_ready():
        return BACKEND_BITNET_CPP

    # Tier 2: transformers BitLinear (packed model + pinned commit)
    if BitLinearCompatibilityLayer.is_bitlinear_importable():
        return BACKEND_TRANSFORMERS_BITLINEAR

    # Tier 3: standard transformers (bf16/float models)
    try:
        import transformers  # noqa: F401
        return BACKEND_TRANSFORMERS_STANDARD
    except ImportError:
        pass

    return "none"


# =============================================================================
# Module-level retry decorator
# =============================================================================
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    log.warning(
                        "Retry %d/%d for %s: %s",
                        attempt + 1, max_attempts, func.__name__, e,
                    )
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


# =============================================================================
# _fix_linear_class — normalise quantization config linear class name
# =============================================================================
def _fix_linear_class(model_config: Any) -> None:
    qc = getattr(model_config, "quantization_config", None)
    if qc is None:
        return
    if isinstance(qc, dict):
        if qc.get("linear_class") == "autobitlinear":
            qc["linear_class"] = "bitlinear"
            log.info("BitNet: overrode linear_class autobitlinear → bitlinear")
    elif hasattr(qc, "linear_class") and qc.linear_class == "autobitlinear":
        qc.linear_class = "bitlinear"
        log.info("BitNet: overrode linear_class autobitlinear → bitlinear")


# =============================================================================
# BitNetLoader — state-aware, PRISM-routed, GGUF-first model loader
# =============================================================================
class BitNetLoader:
    """
    Production BitNet model loader with automatic backend selection.

    Loading priority (v1.0.0):
      1. GGUF-first: if bitnet_state.json has verify_passed=True and gguf_path
         is a valid .gguf file → create BitNetInferenceWrapper with
         force_backend="bitnet_cpp" and NO transformers model loaded (zero RAM waste)
      2. Packed model + BitLinear: if model is BitNetForCausalLM and the pinned
         transformers commit is installed → load via AutoModelForCausalLM with
         BitLinear layers
      3. Standard: bf16/float model → normal AutoModelForCausalLM load
    """

    KNOWN_MODELS: Dict[str, str] = {
        "bitnet-b1.58-7b":   "microsoft/BitNet-b1.58-7B",
        "bitnet-7b":         "microsoft/BitNet-7B",
        "bitnet-3b":         "microsoft/BitNet-3B",
        "bitnet-2b":         "microsoft/bitnet-b1.58-2B-4T",
        "bitnet-2b-bf16":    "microsoft/bitnet-b1.58-2B-4T-bf16",
        "bitnet-2b-gguf":    "microsoft/BitNet-b1.58-2B-4T-gguf",
        "bitnet-large":      "microsoft/BitNet-Large",
        "bitnet-medium":     "microsoft/BitNet-Medium",
        "bitnet-small":      "microsoft/BitNet-Small",
        "llama3-8b-bitnet":  "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
        "falcon3-1b-bitnet": "tiiuae/Falcon3-1B-Instruct-1.58bit",
        "falcon3-3b-bitnet": "tiiuae/Falcon3-3B-Instruct-1.58bit",
    }

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(
        self,
        config: BitNetSystemConfig,
        *,
        model_profiles:  Optional[List[BitNetProfileConfig]] = None,
        kernel_manager:  Optional[Any] = None,
        memory_manager:  Optional[Any] = None,
    ) -> None:
        if config is None:
            raise TypeError("config (BitNetSystemConfig) is required")
        self.config = config

        default_id    = (config.default_model_id
                         or (model_profiles[0].id if model_profiles else "bitnet-model"))
        default_quant = config.default_quantization

        self._fallback_model_config = BitNetModelConfig(
            model_id=default_id, quantization=default_quant
        )

        # v9: prefer CPU for BitNet (ternary SIMD kernels) — not cuda
        self.device  = HardwareDetector.device_from_atlas_state()
        self.backend = HardwareDetector.detect_backend()

        self.kernel_manager  = kernel_manager
        self.memory_manager  = memory_manager
        self._lifecycle_manager: Optional[Any] = None

        self._model_cache:  Dict[str, BitNetInferenceWrapper] = {}
        self._cache_order:  List[str] = []
        self._max_cached    = config.max_loaded_models
        self._cache_lock    = threading.RLock()

        self._loading_tasks: Dict[str, asyncio.Task] = {}
        self._loading_lock  = asyncio.Lock()

        self._profiles: Dict[str, BitNetModelConfig] = self._load_profiles(model_profiles or [])

        # Load state and apply SPECTRE PYTHONPATH
        self._state = BitNetStateManager.load()
        if self._state.available:
            log.info(
                "BitNet state loaded: model=%s gguf=%s verify=%s",
                self._state.model_path,
                self._state.gguf_path,
                self._state.available,
            )
        StateFileReader.apply_spectre_pythonpath()

        # Metrics (singleton – safe for multiple instances)
        self._metrics = _METRICS

        log.info(
            "BitNetLoader v1.0.0 init: device=%s backend=%s quant=%s max_cached=%d",
            self.device, self.backend, default_quant, self._max_cached,
        )

    # ── from_state_file classmethod ───────────────────────────────────────────
    @classmethod
    def from_state_file(
        cls,
        state_path:      Optional[str] = None,
        kernel_manager:  Optional[Any] = None,
        memory_manager:  Optional[Any] = None,
    ) -> "BitNetLoader":
        """
        Create a BitNetLoader from bitnet_state.json (written by bitnet_setup.sh).
        This is the zero-config path: no manual configuration needed.

        v9 FIX: Uses gguf_path when bitnet.cpp is verified (not model_path),
                preventing AutoModelForCausalLM being called on a .gguf file.
                Applies SPECTRE PYTHONPATH before any imports.
        """
        # Apply SPECTRE before anything else
        StateFileReader.apply_spectre_pythonpath()

        state = get_bitnet_state()
        if not state.available:
            raise FileNotFoundError(
                "BitNet state file not found or verify_passed=False.\n"
                "Run:  local-llm-setup\n"
                "  or: local-llm-setup --minimal-build  (for pre-built GGUF)"
            )

        # v9 FIX: Decide profile path based on what's actually available
        gguf_available = bool(state.gguf_path and Path(state.gguf_path).is_file())
        safetensors_available = bool(
            state.model_path and Path(state.model_path).is_dir()
            and list(Path(state.model_path).glob("*.safetensors"))
        )

        if gguf_available:
            # bitnet.cpp path — profile points to GGUF file
            profile_path    = state.gguf_path
            loading_backend = BACKEND_BITNET_CPP
            log.info("from_state_file: GGUF path → bitnet.cpp backend")
        elif safetensors_available:
            # Determine if packed or bf16
            prism = PRISMModelClassifier.classify(state.model_path)
            if prism == "packed" and BitLinearCompatibilityLayer.is_bitlinear_importable():
                profile_path    = state.model_path
                loading_backend = BACKEND_TRANSFORMERS_BITLINEAR
                log.info("from_state_file: packed model + BitLinear → transformers backend")
            else:
                profile_path    = state.model_path
                loading_backend = BACKEND_TRANSFORMERS_STANDARD
                log.info("from_state_file: bf16/standard model → transformers backend")
        else:
            raise FileNotFoundError(
                f"Neither GGUF ({state.gguf_path!r}) nor safetensors "
                f"({state.model_path!r}) found on disk.\n"
                f"Re-run: local-llm-setup"
            )

        # Build minimal BitNetSystemConfig from state
        config = BitNetSystemConfig(
            default_model_id  = Path(state.model_path or profile_path).name,
            default_quantization = "1bit",
            bitnet_cpp_port   = int(os.environ.get("BITNET_PORT", "18080")),
            bitnet_n_threads  = int(os.environ.get("BITNET_N_THREADS",
                                   str(StateFileReader.get("atlas_threads") or "4"))),
            bitnet_n_gpu_layers = 0,   # BitNet: always 0
            max_loaded_models = 1,
            load_retries      = 3,
        )

        # Create profile with the correct path for the chosen backend
        profile = BitNetProfileConfig(
            id             = "state_model",
            path           = profile_path,
            quantization   = "1bit",
            autoload       = True,
            priority       = 100,
        )
        # Attach backend decision to profile for _load_impl to consume
        profile._loading_backend = loading_backend
        profile._gguf_path       = state.gguf_path if gguf_available else None
        profile._model_path      = state.model_path or ""
        # v9.1 FIX: when using the GGUF path, set tokenizer_path to the
        # safetensors model dir so the tokenizer can be loaded separately.
        # This is the key linkage between ggml-model-i2_s.gguf and the
        # bitnet-b1.58-2B-4T directory for tokenization.
        if gguf_available and safetensors_available:
            profile.tokenizer_path = state.model_path
            log.info(
                "from_state_file: tokenizer_path set to safetensors dir: %s",
                state.model_path,
            )
        else:
            profile.tokenizer_path = None

        return cls(
            config         = config,
            model_profiles = [profile],
            kernel_manager = kernel_manager,
            memory_manager = memory_manager,
        )

    # ── Profile building ──────────────────────────────────────────────────────
    def _load_profiles(
        self, profile_list: List[BitNetProfileConfig]
    ) -> Dict[str, BitNetModelConfig]:
        profiles: Dict[str, BitNetModelConfig] = {}
        for prof in profile_list:
            try:
                # v9: check if profile has _loading_backend attached by from_state_file
                # v9.2 BUG FIX: profiles built via BitNetProfileConfig (the direct-path
                # load path in api_server._load_bitnet_model) have NO _loading_backend
                # attribute, so getattr falls back to BACKEND_TRANSFORMERS_STANDARD even
                # when the path is a .gguf file.  Call _determine_backend() as the
                # fallback so GGUF profiles correctly register as BACKEND_GGUF.
                _explicit_backend = getattr(prof, "_loading_backend", None)
                gguf_path         = getattr(prof, "_gguf_path", None)
                model_path_attr   = getattr(prof, "_model_path", None) or prof.path
                if _explicit_backend:
                    loading_backend = _explicit_backend
                else:
                    loading_backend = (
                        self._determine_backend(model_path_attr)
                        if model_path_attr
                        else BACKEND_TRANSFORMERS_STANDARD
                    )

                # v9.1 BUG 3 FIX: BitNetProfileConfig.device validator already enforces
                # cpu, but apply belt-and-suspenders here for any path that bypasses
                # the Pydantic validator (e.g. dataclass-style construction).
                raw_device = getattr(prof, "device", "cpu") or "cpu"
                safe_device = "cpu" if raw_device in ("cuda", "mps", "rocm") else raw_device

                # v9.1 BUG 3 FIX: num_gpu_layers validator already enforces 0, but
                # guard again at this level to prevent -1 (all-layers-on-GPU) slipping through.
                raw_gpu_layers = getattr(prof, "num_gpu_layers", 0)
                safe_gpu_layers = 0 if raw_gpu_layers > 0 else raw_gpu_layers

                cfg = BitNetModelConfig(
                    model_id               = prof.id,
                    model_name_or_path     = model_path_attr,
                    quantization           = prof.quantization or self.config.default_quantization,
                    device                 = safe_device,
                    dtype                  = torch.bfloat16,
                    use_flash_attention_2  = getattr(prof, "use_flash_attention_2", False),
                    max_position_embeddings= getattr(prof, "context_length", None) or 4096,
                    num_gpu_layers         = safe_gpu_layers,
                    autoload               = prof.autoload,
                    priority               = prof.priority,
                    tags                   = getattr(prof, "tags", []),
                    gguf_path              = gguf_path,
                    tokenizer_path         = getattr(prof, "tokenizer_path", None),
                    loading_backend        = loading_backend,
                )
                profiles[prof.id] = cfg
                log.info(
                    "Registered BitNet profile: %s  device=%s  gpu_layers=%d  backend=%s",
                    prof.id, safe_device, safe_gpu_layers, loading_backend,
                )
            except Exception as e:
                log.error("Failed to parse BitNet profile %s: %s", getattr(prof, "id", prof), e)
        return profiles

    def get_profile(self, model_id: str) -> Optional[BitNetModelConfig]:
        return self._profiles.get(model_id)

    def set_lifecycle_manager(self, manager: Any) -> None:
        self._lifecycle_manager = manager

    # ── Public model info ─────────────────────────────────────────────────────
    @staticmethod
    def estimate_model_size(model_path: str) -> int:
        """Estimate model memory requirement in MB."""
        p = Path(model_path)
        # GGUF file — use file size * 1.1 (much smaller than safetensors * 1.5)
        if p.is_file() and p.suffix.lower() == ".gguf":
            return int(p.stat().st_size * 1.1 / (1024 * 1024))
        if not p.is_dir():
            return 0
        # Check for built GGUF first
        ggufs = list(p.glob("ggml-model-*.gguf"))
        if ggufs:
            return int(max(f.stat().st_size for f in ggufs) * 1.1 / (1024 * 1024))
        # Safetensors
        safetensors = list(p.glob("*.safetensors"))
        if safetensors:
            return int(sum(f.stat().st_size for f in safetensors) * 1.5 / (1024 * 1024))
        return 2000  # 2 GB fallback

    @staticmethod
    def can_load_model_data(model_path: str, device: str = "cpu") -> bool:
        required_mb = BitNetLoader.estimate_model_size(model_path)
        if device in ("cuda", "mps"):
            if device == "cuda" and torch.cuda.is_available():
                free_vram_mb = (
                    torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                ) // (1024 * 1024)
                return free_vram_mb >= required_mb
            return True
        avail_mb = psutil.virtual_memory().available // (1024 * 1024)
        return avail_mb >= required_mb

    # ── Public load API ───────────────────────────────────────────────────────
    async def load(
        self,
        model_id:    str  = "state_model",
        force_reload: bool = False,
    ) -> BitNetInferenceWrapper:
        """
        Load a BitNet model by profile ID (or path).
        Returns a cached wrapper if already loaded.
        """
        # Return cached
        with self._cache_lock:
            if model_id in self._model_cache and not force_reload:
                log.debug("BitNet cache hit: %s", model_id)
                return self._model_cache[model_id]

        # Deduplicate concurrent loads for the same model
        resolved_path = self._resolve_model_path(model_id)
        async with self._loading_lock:
            if resolved_path in self._loading_tasks:
                log.debug("Awaiting in-progress load for %s", model_id)
                return await self._loading_tasks[resolved_path]

        profile_id = model_id
        load_config = self._profiles.get(model_id)
        if load_config is None:
            load_config = BitNetModelConfig(
                model_id           = model_id,
                model_name_or_path = resolved_path,
                loading_backend    = self._determine_backend(resolved_path),
            )
            profile_id = model_id

        t_start = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._load_impl(resolved_path, load_config, profile_id)
            )
            async with self._loading_lock:
                self._loading_tasks[resolved_path] = task

            wrapper = await task
            elapsed = time.monotonic() - t_start

            if self._metrics.get("load_time_s"):
                self._metrics["load_time_s"].observe(elapsed)
            if self._metrics.get("load_total"):
                self._metrics["load_total"].labels(model_id=model_id, status="success").inc()

            log.info("BitNet load complete: %s  %.1fs  backend=%s",
                     model_id, elapsed, load_config.loading_backend)
            return wrapper

        except Exception as e:
            if self._metrics.get("load_total"):
                self._metrics["load_total"].labels(model_id=model_id, status="failed").inc()
            raise
        finally:
            async with self._loading_lock:
                self._loading_tasks.pop(resolved_path, None)

    # ── Backend determination ─────────────────────────────────────────────────
    def _determine_backend(self, model_path: str) -> str:
        """
        Determine the correct loading backend for a given model path.

        Priority order (v9.1):
          1. Pre-built GGUF file (*.gguf) OR directory containing ggml-model-*.gguf
             → BACKEND_GGUF (bitnet.cpp subprocess — fastest, no transformers needed)
          2. State file has verify_passed=True AND gguf_path matches this model
             → BACKEND_BITNET_CPP (same as above but driven by state file)
          3. PRISM: packed model (BitNetForCausalLM/U8 weights) + BitLinear available
             → BACKEND_TRANSFORMERS_BITLINEAR (pinned transformers commit)
          4. Anything else (bf16 safetensors, unknown)
             → BACKEND_TRANSFORMERS_STANDARD
        """
        p = Path(model_path)

        # ── Priority 1: standalone .gguf file or dir-with-gguf (BUG 1/5 FIX) ──
        # This covers ggml-model-i2_s.gguf directly — no state file needed.
        if p.is_file() and p.suffix.lower() == ".gguf":
            log.info(
                "_determine_backend: '%s' is a .gguf file → BACKEND_GGUF (bitnet.cpp)",
                p.name,
            )
            return BACKEND_GGUF

        if p.is_dir() and list(p.glob("ggml-model-*.gguf")):
            log.info(
                "_determine_backend: dir '%s' contains pre-built GGUF → BACKEND_GGUF",
                p.name,
            )
            return BACKEND_GGUF

        # ── Priority 2: state file match → bitnet.cpp ──────────────────────
        if StateFileReader.is_bitnet_cpp_ready():
            state_gguf  = StateFileReader.get_gguf_path()
            state_model = StateFileReader.get("model_path") or ""
            if state_model and (
                p.resolve() == Path(state_model).resolve()
                or p.name == Path(state_model).name
            ):
                if state_gguf and Path(state_gguf).is_file():
                    log.info(
                        "_determine_backend: state-file match → BACKEND_BITNET_CPP "
                        "(gguf=%s)", state_gguf,
                    )
                    return BACKEND_BITNET_CPP

        # ── Priority 3 / 4: PRISM classification ───────────────────────────
        prism = PRISMModelClassifier.classify(model_path)
        log.debug("_determine_backend: PRISM class for '%s' → '%s'", p.name, prism)

        if prism == "pre-built":
            return BACKEND_BITNET_CPP  # directory already has built GGUF

        if prism == "packed":
            if BitLinearCompatibilityLayer.is_bitlinear_importable():
                log.info(
                    "_determine_backend: packed model + BitLinear available → "
                    "BACKEND_TRANSFORMERS_BITLINEAR"
                )
                return BACKEND_TRANSFORMERS_BITLINEAR
            # Packed model but no BitLinear — will fail at load time with a clear error
            log.warning(
                "_determine_backend: packed model '%s' detected but BitLinear is NOT "
                "available. Load will raise ImportError with install instructions. "
                "To avoid this, use the pre-built GGUF (ggml-model-i2_s.gguf) instead.",
                p.name,
            )
            return BACKEND_TRANSFORMERS_STANDARD  # triggers _check_bitlinear_available_or_raise

        # bf16 or unknown → standard transformers
        return BACKEND_TRANSFORMERS_STANDARD

    # ── Core load implementation (GGUF-first, backend-routed) ─────────────────
    async def _load_impl(
        self,
        model_path:  str,
        load_config: BitNetModelConfig,
        profile_id:  str,
    ) -> BitNetInferenceWrapper:
        """
        v9 GGUF-FIRST routing:
          If backend is bitnet_cpp → skip ALL transformers loading.
          Only load via transformers when actually needed.
        """
        log.info(
            "_load_impl: path=%s backend=%s",
            model_path, load_config.loading_backend,
        )

        loop = asyncio.get_running_loop()

        # ── PATH A: GGUF / bitnet.cpp (NO transformers model loaded) ──────────
        if load_config.loading_backend in (BACKEND_BITNET_CPP, BACKEND_GGUF):
            wrapper = await self._load_impl_gguf(model_path, load_config, profile_id, loop)

        # ── PATH B: transformers (packed BitLinear or standard bf16) ──────────
        else:
            # Check memory for transformers load (safetensors-based estimate)
            await self._check_memory_requirements_transformers(model_path, load_config)
            wrapper = await self._load_impl_transformers(model_path, load_config, profile_id, loop)

        # ── Cache management ──────────────────────────────────────────────────
        with self._cache_lock:
            if len(self._model_cache) >= self._max_cached and self._cache_order:
                oldest = self._cache_order.pop(0)
                old_wrapper = self._model_cache.pop(oldest, None)
                if old_wrapper:
                    await self._evict_wrapper(oldest, old_wrapper)

            self._model_cache[profile_id] = wrapper
            self._cache_order.append(profile_id)

        self._update_metrics()
        if self._lifecycle_manager:
            try:
                self._lifecycle_manager.on_model_loaded(profile_id, wrapper)
            except Exception as e:
                log.error("Lifecycle manager error after load: %s", e)

        log.info(
            "BitNet model ready: %s (profile=%s backend=%s)",
            model_path, profile_id, load_config.loading_backend,
        )
        return wrapper

    # ── PATH A: GGUF / bitnet.cpp loader ─────────────────────────────────────
    async def _load_impl_gguf(
        self,
        model_path:  str,
        load_config: BitNetModelConfig,
        profile_id:  str,
        loop:        asyncio.AbstractEventLoop,
    ) -> BitNetInferenceWrapper:
        """
        Create a BitNetInferenceWrapper backed by bitnet.cpp (llama-server/llama-cli).
        Does NOT load any model weights into RAM via transformers.

        The wrapper delegates ALL inference to the C++ binary via the
        BitNetCppBackend, which reads the GGUF file directly.
        """
        # Resolve the actual GGUF file path
        gguf_path = load_config.gguf_path
        if not gguf_path or not Path(gguf_path).is_file():
            # Try to find it
            gguf_path = _resolve_gguf_from_dir(model_path)
            if not gguf_path:
                gguf_path = StateFileReader.get_gguf_path()
            if not gguf_path:
                raise FileNotFoundError(
                    f"No .gguf file found for model at: {model_path}\n"
                    f"Run bitnet_setup.sh to build the quantized model, or use "
                    f"--minimal-build to download a pre-converted GGUF."
                )

        log.info("GGUF path resolved: %s", gguf_path)

        # Load tokenizer only (tiny, needed for prompt formatting + decode)
        # v9.1: prefer tokenizer_path (safetensors dir) over model_name_or_path
        # when set — this is critical for GGUF-only profiles where model_name_or_path
        # points to the .gguf file itself (not a tokenizer-loadable directory).
        tokenizer_source = (
            load_config.tokenizer_path
            or (load_config.model_name_or_path
                if load_config.model_name_or_path and Path(load_config.model_name_or_path).is_dir()
                else None)
        )
        tokenizer = None
        if tokenizer_source and Path(tokenizer_source).is_dir():
            try:
                tokenizer = await loop.run_in_executor(
                    None, self._load_tokenizer_sync, tokenizer_source, load_config
                )
                log.info("Tokenizer loaded for GGUF path: %s", tokenizer_source)
            except Exception as e:
                log.warning(
                    "Tokenizer load failed for GGUF path (%s): %s — "
                    "llama-server handles tokenization internally",
                    tokenizer_source, e,
                )
        else:
            log.debug(
                "No tokenizer directory available for GGUF path '%s' "
                "(tokenizer_path=%r model_name_or_path=%r) — "
                "llama-server handles tokenization.",
                gguf_path,
                load_config.tokenizer_path,
                load_config.model_name_or_path,
            )

        # Memory check for GGUF (much smaller than safetensors)
        gguf_size_mb = int(Path(gguf_path).stat().st_size * 1.1 / (1024 * 1024))
        avail_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
        if avail_ram_mb < gguf_size_mb:
            log.warning(
                "Low RAM: GGUF needs ~%d MB, only %d MB available. "
                "Proceeding anyway (llama.cpp uses mmap).",
                gguf_size_mb, avail_ram_mb,
            )

        n_threads = self.config.bitnet_n_threads or int(
            StateFileReader.get("atlas_threads") or "4"
        )

        wrapper = BitNetInferenceWrapper(
            model            = None,           # v9: NO transformers model for GGUF path
            tokenizer        = tokenizer,
            config           = load_config,
            device           = load_config.device,
            memory_manager   = self.memory_manager,
            kernel_manager   = self.kernel_manager,
            model_path       = model_path,
            bitnet_cpp_port  = self.config.bitnet_cpp_port,
            bitnet_n_threads = n_threads,
            bitnet_n_gpu_layers = 0,           # BitNet: always 0
            force_backend    = BACKEND_BITNET_CPP,
            gguf_path        = gguf_path,      # v9: explicit GGUF path
        )
        wrapper.load_time_s = 0.0
        wrapper.loaded_at   = time.monotonic()
        return wrapper

    # ── PATH B: transformers loader ───────────────────────────────────────────
    async def _load_impl_transformers(
        self,
        model_path:  str,
        load_config: BitNetModelConfig,
        profile_id:  str,
        loop:        asyncio.AbstractEventLoop,
    ) -> BitNetInferenceWrapper:
        """
        Load via AutoModelForCausalLM (packed + BitLinear, or standard bf16).
        Includes BitLinear guard for packed models.
        """
        # BitLinear guard — raise BEFORE attempting load for packed models
        if load_config.loading_backend == BACKEND_TRANSFORMERS_BITLINEAR:
            _check_bitlinear_available_or_raise(model_path)
        else:
            # Check if model is actually packed (PRISM late check)
            arch = PRISMModelClassifier.get_architecture(model_path)
            if arch in PACKED_ARCHITECTURES:
                _check_bitlinear_available_or_raise(model_path)
                # If we get here, BitLinear is available
                load_config.loading_backend = BACKEND_TRANSFORMERS_BITLINEAR

        try:
            tokenizer = await loop.run_in_executor(
                None, self._load_tokenizer_sync, model_path, load_config
            )
            model = await loop.run_in_executor(
                None, self._load_model_sync, model_path, load_config
            )
        except Exception as e:
            log.exception("Failed to load BitNet model %s via transformers", model_path)
            raise RuntimeError(
                f"Failed to load BitNet model {model_path}: {e}\n"
                f"Tip: For bitnet.cpp (faster), use pre-built GGUF: "
                f"local-llm-setup --minimal-build"
            ) from e

        # Use router only as a secondary signal (model already loaded correctly)
        force_backend = load_config.loading_backend
        try:
            router  = AdaptiveModelRouter()
            header, backend_enum, _ = router.route(model_path)
            if backend_enum in (InferenceBackend.BITNET_CPP_SERVER,
                                 InferenceBackend.BITNET_CPP_PROCESS):
                # Router wants bitnet.cpp but we loaded via transformers —
                # use transformers backend for this session, log advisory
                log.info(
                    "Router preferred bitnet.cpp for %s but model was loaded via "
                    "transformers (no GGUF available). Performance: normal (not optimised).",
                    model_path,
                )
        except Exception as e:
            log.debug("Router error (non-fatal): %s", e)

        n_threads = self.config.bitnet_n_threads or int(
            StateFileReader.get("atlas_threads") or "4"
        )

        wrapper = BitNetInferenceWrapper(
            model            = model,
            tokenizer        = tokenizer,
            config           = load_config,
            device           = load_config.device,
            memory_manager   = self.memory_manager,
            kernel_manager   = self.kernel_manager,
            model_path       = model_path,
            bitnet_cpp_port  = self.config.bitnet_cpp_port,
            bitnet_n_threads = n_threads,
            bitnet_n_gpu_layers = 0,
            force_backend    = force_backend,
        )
        wrapper.load_time_s = 0.0
        wrapper.loaded_at   = time.monotonic()
        return wrapper

    # ── Tokenizer loading ─────────────────────────────────────────────────────
    def _load_tokenizer_sync(
        self,
        model_path:  str,
        load_config: BitNetModelConfig,
    ) -> PreTrainedTokenizer:
        tokenizer_kwargs = TransformersCompatibilityManager.get_safe_tokenizer_kwargs(
            trust_remote_code=load_config.trust_remote_code,
            use_fast=True,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.eos_token is None:
                log.warning("Tokenizer has no eos_token — generation may not terminate")
            log.debug("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)
            return tokenizer
        except Exception as exc:
            msg = str(exc).lower()
            if "gated" in msg or "authentication" in msg:
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if not hf_token:
                    raise RuntimeError(
                        f"Model '{model_path}' requires HuggingFace authentication.\n"
                        "Set the HF_TOKEN environment variable and retry."
                    ) from exc
            raise RuntimeError(
                f"Failed to load tokenizer from '{model_path}': {exc}"
            ) from exc

    # ── Model loading (transformers path) ─────────────────────────────────────
    def _load_model_sync(
        self,
        model_path:  str,
        load_config: BitNetModelConfig,
    ) -> PreTrainedModel:
        """
        Load model via AutoModelForCausalLM.
        v9: Uses bfloat16 (matches BitNet training dtype).
            BitLinear guard prevents confusing errors on packed models.
        """
        quant_config = self._get_quantization_config(load_config)
        load_kwargs  = TransformersCompatibilityManager.get_bitnet_load_kwargs(
            model_path, device=load_config.device
        )

        # v9: bfloat16 is the correct dtype for BitNet (matches training)
        # Only override if explicitly set in config
        if load_config.dtype is not None:
            load_kwargs["torch_dtype"] = load_config.dtype
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        if quant_config:
            for key in ("load_in_8bit", "load_in_4bit"):
                if key in quant_config:
                    load_kwargs[key] = quant_config[key]
            if "quantization_config" in quant_config:
                load_kwargs["quantization_config"] = quant_config["quantization_config"]

        # Load AutoConfig first to get architecture name
        model_config = None
        trust_remote = load_config.trust_remote_code
        for attempt_trust in ([trust_remote, False] if trust_remote else [False]):
            try:
                model_config = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=attempt_trust
                )
                trust_remote = attempt_trust
                break
            except OSError as exc:
                err_s = str(exc)
                if attempt_trust and (
                    "configuration_bitnet" in err_s
                    or "does not appear to have a file" in err_s
                ):
                    log.warning("BitNet custom config missing — using built-in; retrying")
                    continue
                raise
            except Exception:
                if not attempt_trust:
                    raise
                continue

        if model_config is None:
            raise RuntimeError(f"Could not load model config from '{model_path}'")

        # v9: Check architecture against packed set (late guard)
        arch = getattr(model_config, "architectures", [""])[0] if hasattr(
            model_config, "architectures") else ""
        if arch in PACKED_ARCHITECTURES:
            _check_bitlinear_available_or_raise(model_path)

        _fix_linear_class(model_config)
        load_kwargs["config"]           = model_config
        load_kwargs["trust_remote_code"] = trust_remote

        # CPU-only path: device_map auto → cpu
        if load_config.device == "cpu" and load_kwargs.get("device_map") == "auto":
            load_kwargs["device_map"] = "cpu"

        max_retries = getattr(self.config, "load_retries", 3)
        for attempt in range(max_retries):
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load BitNet model '{model_path}' after "
                        f"{max_retries} attempts. Last error: {exc}"
                    ) from exc
                log.warning(
                    "Load attempt %d/%d failed: %s", attempt + 1, max_retries, exc
                )
                time.sleep(2 ** attempt)

        if load_kwargs.get("device_map") is None:
            model = model.to(load_config.device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters()) // 1_000_000
        log.info(
            "BitNet model loaded via transformers: device=%s dtype=%s params=%dM arch=%s",
            load_config.device,
            str(next(model.parameters()).dtype),
            param_count,
            arch,
        )
        return model

    # ── Quantization config builder ───────────────────────────────────────────
    def _get_quantization_config(self, load_config: BitNetModelConfig) -> Dict[str, Any]:
        if load_config.quantization == "1bit":
            return {}  # BitNet 1.58-bit is handled by the model architecture itself
        elif load_config.quantization == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
            return {"load_in_4bit": True, "quantization_config": bnb}
        elif load_config.quantization == "8bit":
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            return {"load_in_8bit": True, "quantization_config": bnb}
        elif load_config.quantization == "mixed":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            return {"load_in_4bit": True, "quantization_config": bnb}
        return {}

    # ── Memory checks ─────────────────────────────────────────────────────────
    async def _check_memory_requirements_transformers(
        self, model_path: str, load_config: BitNetModelConfig
    ) -> None:
        """Memory check for transformers path (safetensors-based estimate)."""
        p = Path(model_path)
        if p.exists() and p.is_dir():
            safetensors = list(p.glob("*.safetensors"))
            file_size_gb = (
                sum(f.stat().st_size for f in safetensors) / (1024 ** 3)
                if safetensors else 2.0
            )
        else:
            file_size_gb = 2.0
        required_gb = file_size_gb * 1.5

        if load_config.device in ("cuda", "mps"):
            if load_config.device == "cuda" and torch.cuda.is_available():
                free_vram_mb = (
                    torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                ) / 1024
                free_vram_gb = free_vram_mb / 1024
            else:
                free_vram_gb = required_gb + 1.0  # MPS: assume enough
            if free_vram_gb < required_gb:
                raise MemoryError(
                    f"Insufficient {load_config.device.upper()} memory: "
                    f"need {required_gb:.2f}GB, have {free_vram_gb:.2f}GB.\n"
                    f"Tip: Use bitnet.cpp path (GGUF) which needs only "
                    f"{file_size_gb * 1.1:.2f}GB via mmap."
                )
        else:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_ram_gb < required_gb:
                raise MemoryError(
                    f"Insufficient CPU RAM: need {required_gb:.2f}GB, "
                    f"have {available_ram_gb:.2f}GB.\n"
                    f"Tip: Use bitnet.cpp path (GGUF, ~{file_size_gb * 0.3:.2f}GB via mmap)."
                )

    # ── Model path resolution ─────────────────────────────────────────────────
    def _resolve_model_path(self, model_id: str) -> str:
        # ── Check registered profiles FIRST ──────────────────────────────────
        # Profile IDs (e.g. "bitnet-gguf") are user-defined labels — they are
        # NOT HuggingFace IDs and do NOT have to be filesystem paths themselves.
        # The actual path lives inside the registered BitNetModelConfig.
        # Without this check, load("bitnet-gguf") falls through to the
        # KNOWN_MODELS / filesystem checks and raises "Unknown model: 'bitnet-gguf'"
        # even though the profile with a valid .gguf path is registered.
        if model_id in self._profiles:
            cfg = self._profiles[model_id]
            registered_path = (
                getattr(cfg, "model_name_or_path", None)
                or getattr(cfg, "gguf_path",        None)
            )
            if registered_path:
                log.debug(
                    "_resolve_model_path: '%s' resolved via registered profile → %s",
                    model_id, registered_path,
                )
                return registered_path

        lower_id = model_id.lower()
        if lower_id in self.KNOWN_MODELS:
            return self.KNOWN_MODELS[lower_id]
        path = Path(model_id)
        if path.exists():
            if path.is_file() and path.suffix.lower() == ".gguf":
                return str(path.resolve())  # v9: accept .gguf files directly
            if not path.is_dir():
                raise ValueError(f"BitNet model must be a directory or .gguf file: {model_id}")
            if not (path / "config.json").exists():
                # Could be a dir with only .gguf files (minimal download)
                if list(path.glob("*.gguf")):
                    return str(path.resolve())
                raise ValueError(
                    f"Model directory missing config.json: {model_id}\n"
                    f"Expected a HuggingFace model directory or .gguf file."
                )
            return str(path.resolve())
        if "/" not in model_id and model_id not in self.KNOWN_MODELS:
            raise ValueError(
                f"Unknown model: {model_id!r}.\n"
                f"Provide a HuggingFace model ID (org/model), local path, or "
                f"one of: {list(self.KNOWN_MODELS.keys())}"
            )
        return model_id  # HF model ID — will be downloaded

    # ── Cache eviction ────────────────────────────────────────────────────────
    async def _evict_wrapper(self, model_id: str, wrapper: BitNetInferenceWrapper) -> None:
        """Safely evict a wrapper from cache."""
        if hasattr(wrapper, "shutdown"):
            try:
                await wrapper.shutdown()
            except Exception as e:
                log.warning("Error shutting down BitNet backend on eviction: %s", e)
        # v9 FIX: model may be None for GGUF path — guard del
        if getattr(wrapper, "model", None) is not None:
            try:
                del wrapper.model
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._lifecycle_manager:
            try:
                self._lifecycle_manager.on_model_unloaded(model_id)
            except Exception:
                pass
        log.info("Evicted BitNet model from cache: %s", model_id)

    # ── Unload ────────────────────────────────────────────────────────────────
    def unload(self, model_id: str) -> None:
        """Unload a model from the cache and free its resources."""
        with self._cache_lock:
            if model_id not in self._model_cache:
                log.debug("unload: %s not in cache", model_id)
                return
            wrapper = self._model_cache.pop(model_id)
            if model_id in self._cache_order:
                self._cache_order.remove(model_id)

        if hasattr(wrapper, "shutdown"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(wrapper.shutdown())
                else:
                    loop.run_until_complete(wrapper.shutdown())
            except Exception as e:
                log.warning("Error during wrapper shutdown on unload: %s", e)

        # v9 FIX: model=None for GGUF path — guard against AttributeError
        model = getattr(wrapper, "model", None)
        if model is not None:
            try:
                del wrapper.model
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._lifecycle_manager:
            try:
                self._lifecycle_manager.on_model_unloaded(model_id)
            except Exception:
                pass

        if self._metrics.get("unload_total"):
            self._metrics["unload_total"].labels(model_id=model_id).inc()

        self._update_metrics()
        log.info("BitNet model unloaded: %s", model_id)

    def unload_all(self) -> None:
        for model_id in list(self._model_cache.keys()):
            self.unload(model_id)

    # ── Cache queries ─────────────────────────────────────────────────────────
    def list_loaded(self) -> List[str]:
        with self._cache_lock:
            return list(self._model_cache.keys())

    def is_loaded(self, model_id: str) -> bool:
        with self._cache_lock:
            return model_id in self._model_cache

    def model_memory_usage(self, model_id: str) -> Dict[str, Any]:
        wrapper = self._model_cache.get(model_id)
        if wrapper and hasattr(wrapper, "get_memory_usage"):
            return wrapper.get_memory_usage()
        return {"vram_mb": 0, "ram_mb": 0}

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _update_metrics(self) -> None:
        if self._metrics.get("loaded_models"):
            self._metrics["loaded_models"].set(len(self._model_cache))
        if self._metrics.get("vram_used_mb") and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() // (1024 * 1024)
            self._metrics["vram_used_mb"].set(used)

    # ── Autoload ──────────────────────────────────────────────────────────────
    async def autoload_profiles(self) -> List[str]:
        autoload = sorted(
            [(pid, cfg) for pid, cfg in self._profiles.items() if cfg.autoload],
            key=lambda x: -x[1].priority,
        )
        loaded = []
        for pid, cfg in autoload:
            try:
                await self.load(pid)
                loaded.append(pid)
            except Exception as e:
                log.error("Autoload failed for %s: %s", pid, e)
        return loaded

    # ── ONNX export ───────────────────────────────────────────────────────────
    async def export_to_onnx(
        self,
        model_id:     str,
        output_path:  str,
        opset_version: int = 17,
    ) -> None:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        wrapper = await self.load(model_id)
        if getattr(wrapper, "model", None) is None:
            raise RuntimeError(
                "ONNX export is only available for transformers-loaded models, "
                "not bitnet.cpp GGUF path. Use the transformers backend for export."
            )
        dummy = wrapper.tokenizer(
            "Hello", return_tensors="pt"
        )["input_ids"].to(wrapper.device)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: torch.onnx.export(
                wrapper.model, dummy, output_path,
                opset_version=opset_version,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
            ),
        )
        log.info("Model exported to ONNX: %s", output_path)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Return a full status dict for health endpoints and diagnostics."""
        state_data = StateFileReader.load() or {}
        return {
            "loaded_models":     self.list_loaded(),
            "device":            self.device,
            "backend":           self.backend,
            "bitlinear_available": BitLinearCompatibilityLayer.is_bitlinear_importable(),
            "bitnet_cpp_ready":  StateFileReader.is_bitnet_cpp_ready(),
            "gguf_path":         StateFileReader.get_gguf_path(),
            "llama_cli":         StateFileReader.get_llama_cli(),
            "llama_server":      StateFileReader.get_llama_server(),
            "spectre_mode":      state_data.get("spectre_mode"),
            "prism_class":       state_data.get("prism_class"),
            "auto_backend":      auto_detect_backend(),
        }
    