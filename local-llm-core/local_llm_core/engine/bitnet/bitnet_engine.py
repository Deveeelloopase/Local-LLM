from __future__ import annotations

# engine/bitnet/bitnet_engine.py — Production BitNet Inference Engine  v1.0.0
# ============================================================================
#
# New features in v1.0.0:
#   • GGUF-first init path (no router needed when gguf_path + binary explicit)
#   • Chat template support for BitNet packed models (BitLinear path)
#   • Circuit breaker for C++ backend (configurable)
#   • Per-request correlation IDs propagated through all log lines
#   • `get_status()` rich diagnostic dict for health endpoints
#   • `warm_up()` async method — run a 1-token probe to pre-load llama-server
#   • `create_chat_completion()` OpenAI-style chat API (messages[] → string)
#   • Consistent n_ctx default across both paths (4096)
#   • SPECTRE PYTHONPATH applied at module import
#   • All original public API preserved unchanged
# ============================================================================

import asyncio
import logging
import os
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch

# ── Transformers — guarded import (same pattern as bitnet_loader.py) ─────────
# bitnet_engine.py is imported by bitnet_loader.py at module level.
# If transformers fails to import (e.g. PYTHONOPTIMIZE=1 strips docstrings and
# transformers.modeling_utils crashes), this guard ensures the entire bitnet
# subsystem doesn't go down. Tier 1 (bitnet.cpp / GGUF) never uses these.
try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
    _TF_ENGINE_AVAILABLE = True
except Exception as _tf_engine_exc:      # noqa: BLE001
    _TF_ENGINE_AVAILABLE = False
    PreTrainedModel       = None         # type: ignore[assignment,misc]
    PreTrainedTokenizer   = None         # type: ignore[assignment,misc]
    StoppingCriteria      = None         # type: ignore[assignment,misc]
    StoppingCriteriaList  = None         # type: ignore[assignment,misc]
    TextIteratorStreamer   = None         # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

# ── Apply SPECTRE PYTHONPATH at module import (gguf TL1/TL2 availability) ──
try:
    from ...compatibility.transformers_compat import (
        StateFileReader,
        BitLinearCompatibilityLayer,
    )
    StateFileReader.apply_spectre_pythonpath()
    _COMPAT_AVAILABLE = True
except Exception:
    _COMPAT_AVAILABLE = False

# ── BitNet.cpp backend bindings (optional, gracefully absent) ───────────────
try:
    from .bitnet_cpp_backend import (
        BitNetCppBackend,
        BitNetCppChatBackend,
        BitNetRunInferenceBackend,
        BitNetCppConfig,
        BackendResult,
        create_best_backend,
    )
    _BITNET_CPP_AVAILABLE = True
except ImportError:
    _BITNET_CPP_AVAILABLE = False
    BitNetCppBackend       = None   # type: ignore[assignment]
    BitNetRunInferenceBackend = None  # type: ignore[assignment]
    BitNetCppChatBackend   = None   # type: ignore[assignment]
    BitNetCppConfig        = None   # type: ignore[assignment]
    BackendResult          = None   # type: ignore[assignment]

# ── Unified environment & model router ──────────────────────────────────────
from ..adaptive_awareness.model_unified_env import (
    AdaptiveModelRouter,
    BitNetState,
    BitNetStateManager,
    EnvironmentStatus,
    InferenceBackend,
    GGUFHeader,
    GGUFModelType,
    UnifiedEnvironmentValidator,
    classify_gguf,
    route_model,
    auto_detect_backend,
    get_bitnet_state,
    validate_environment,
)

# ── Packed model architectures (need BitLinear / pinned commit) ─────────────
_PACKED_ARCHITECTURES = frozenset({
    "BitNetForCausalLM",
    "BitNetModel",
    "BitNetBEForCausalLM",
})

# ── BitNet b1.58 chat template (applied when tokenizer has no template) ─────
_BITNET_SYSTEM_TPL   = "<|system|>\n{system}\n<|end|>\n"
_BITNET_USER_TPL     = "<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
_BITNET_ASST_TPL     = "{assistant}\n<|end|>\n"
_BITNET_STOP_TOKENS  = ["<|end|>", "<|user|>", "<|system|>", "<|endoftext|>"]


# =============================================================================
# Prometheus metrics (optional)
# =============================================================================
def _setup_metrics() -> Dict[str, Any]:
    """
    Register Prometheus metrics for BitNet inference.

    Uses a try/except around each metric to handle the case where
    bitnet_engine.py is imported twice in the same Python process
    (e.g. when the first import partially fails and Python evicts the
    module from sys.modules before the second import attempt).

    In that scenario Counter/Gauge/Histogram raises:
        ValueError: Duplicated timeseries in CollectorRegistry

    We catch it and retrieve the already-registered collector from the
    global registry so the rest of the code gets a valid object.
    """
    try:
        from prometheus_client import Counter, Gauge, Histogram
        from prometheus_client import REGISTRY as _REG
    except ImportError:
        return {}

    def _safe(factory: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create metric or return existing one if already registered."""
        try:
            return factory(name, *args, **kwargs)
        except ValueError:
            # Module re-imported in same process — retrieve existing collector.
            # prometheus_client stores collectors keyed by their first name.
            existing = _REG._names_to_collectors.get(name)
            if existing is not None:
                log.debug(
                    "Prometheus metric '%s' already registered — reusing existing collector",
                    name,
                )
                return existing
            # Name not found directly — search by prefix
            for registered_name, collector in _REG._names_to_collectors.items():
                if registered_name.startswith(name):
                    log.debug(
                        "Prometheus metric '%s' reused via prefix match '%s'",
                        name, registered_name,
                    )
                    return collector
            log.warning(
                "Prometheus metric '%s' registration failed and could not be "
                "retrieved from registry — metric will be unavailable",
                name,
            )
            return None

    return {
        "bitnet_requests_total": _safe(
            Counter,
            "bitnet_requests_total",
            "Total BitNet inference requests",
            ["backend", "status"],
        ),
        "bitnet_tokens_total": _safe(
            Counter,
            "bitnet_tokens_total",
            "Total tokens generated by BitNet",
            ["backend"],
        ),
        "bitnet_ttft_ms": _safe(
            Histogram,
            "bitnet_ttft_ms",
            "Time to first token (ms)",
            buckets=(5, 20, 50, 100, 250, 500, 1000, 2000, 5000),
        ),
        "bitnet_tps": _safe(
            Histogram,
            "bitnet_tps",
            "Tokens per second",
            buckets=(0.5, 1, 2, 5, 10, 20, 50, 100, 200),
        ),
        "bitnet_active": _safe(
            Gauge,
            "bitnet_active_generations",
            "Active BitNet generations in progress",
        ),
    }

_METRICS: Dict[str, Any] = _setup_metrics()


# =============================================================================
# Enums and dataclasses
# =============================================================================
class BitNetBackendType(Enum):
    BITNET_CPP_SERVER  = "bitnet_cpp_server"
    BITNET_CPP_PROCESS = "bitnet_cpp_process"
    TRANSFORMERS_BITLINEAR = "transformers_bitlinear"   # packed + pinned commit
    TRANSFORMERS       = "transformers"                 # standard bf16


class GenerationPhase(Enum):
    TOKENIZATION = "tokenization"
    FIRST_TOKEN  = "first_token"
    STREAMING    = "streaming"
    COMPLETE     = "complete"


@dataclass
class BitNetInferenceConfig:
    """Per-request inference configuration. 0.0 timeouts use adaptive defaults."""
    max_tokens:         int   = 512
    temperature:        float = 0.7
    top_p:              float = 0.95
    top_k:              int   = 40
    repetition_penalty: float = 1.1
    do_sample:          bool  = True
    num_beams:          int   = 1
    early_stopping:     bool  = False
    length_penalty:     float = 1.0
    first_token_timeout: float = 0.0    # 0 = use adaptive default
    total_timeout:      float = 0.0     # 0 = use adaptive default
    stream_interval:    float = 0.0     # minimum seconds between yields (0=off)
    stop:     Optional[List[str]] = None
    seed:     int   = -1
    n_ctx:    int   = 4096


@dataclass
class GenerationMetrics:
    request_id:         Optional[str] = None
    model_id:           str  = ""
    backend:            str  = ""
    input_tokens:       int  = 0
    output_tokens:      int  = 0
    ttft_ms:            float = 0.0
    total_time_ms:      float = 0.0
    tokens_per_second:  float = 0.0
    peak_vram_mb:       float = 0.0
    error:              Optional[str] = None
    phase_times:        Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id":       self.request_id,
            "model_id":         self.model_id,
            "backend":          self.backend,
            "input_tokens":     self.input_tokens,
            "output_tokens":    self.output_tokens,
            "ttft_ms":          round(self.ttft_ms, 2),
            "total_time_ms":    round(self.total_time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "peak_vram_mb":     round(self.peak_vram_mb, 2),
            "error":            self.error,
            "phases":           self.phase_times,
        }


# =============================================================================
# Stop-string stopping criteria (transformers backend only)
# =============================================================================
# Safe base class: when transformers is unavailable StoppingCriteria=None,
# using it directly as a base class raises TypeError: NoneType takes no arguments.
_StoppingCriteriaBase: Any = StoppingCriteria if (StoppingCriteria is not None) else object


class StopStringCriteria(_StoppingCriteriaBase):
    """Stop generation when any stop string appears in decoded output.

    Only used on the Tier 2/3 transformers path. When StoppingCriteria is
    unavailable (transformers import failed), this class inherits from object
    as a harmless stub — it will never be instantiated on Tier 1 (bitnet.cpp).
    """

    def __init__(self, stop_strings: List[str], tokenizer: Any) -> None:
        if StoppingCriteria is None:
            raise RuntimeError(
                "StopStringCriteria requires transformers StoppingCriteria — "
                "unavailable on this path (transformers import failed). "
                "Use the bitnet.cpp / GGUF Tier 1 path instead."
            )
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer    = tokenizer

    def __call__(
        self,
        input_ids: Any,
        scores:    Any,
        **kwargs:  Any,
    ) -> bool:
        if StoppingCriteria is None:
            return False
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in decoded for s in self.stop_strings)


# =============================================================================
# Simple circuit breaker for backend failure isolation
# =============================================================================
class _CircuitBreaker:
    """
    Prevents hammering a failing backend.
    CLOSED → OPEN (after threshold failures) → HALF_OPEN (after recovery_s).
    """
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, recovery_s: float = 60.0) -> None:
        self._state     = self.CLOSED
        self._failures  = 0
        self._threshold = threshold
        self._recovery  = recovery_s
        self._opened_at = 0.0
        self._lock      = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = self.HALF_OPEN
                    log.info("CircuitBreaker → HALF_OPEN (probing)")
            return self._state

    def is_open(self) -> bool:
        return self.state == self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state    = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                if self._state != self.OPEN:
                    log.warning("CircuitBreaker OPEN after %d failures", self._failures)
                self._state    = self.OPEN
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state    = self.CLOSED


# =============================================================================
# Chat template utilities
# =============================================================================
def _format_bitnet_chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> str:
    """Format OpenAI-style messages into BitNet b1.58 prompt template."""
    parts: List[str] = []
    if system:
        parts.append(_BITNET_SYSTEM_TPL.format(system=system))
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(_BITNET_SYSTEM_TPL.format(system=content))
        elif role == "user":
            parts.append(_BITNET_USER_TPL.format(user=content))
        elif role == "assistant":
            parts.append(_BITNET_ASST_TPL.format(assistant=content))
    return "".join(parts)


def _apply_chat_template_if_available(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply tokenizer's chat template if available (pinned commit has it).
    Fall back to BitNet b1.58 manual template.
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    except Exception as e:
        log.debug("apply_chat_template failed: %s — using manual template", e)
    return _format_bitnet_chat(messages)


def _check_bitlinear_before_load(model_path: str) -> None:
    """Raise clear ImportError if model is packed and BitLinear unavailable."""
    if not _COMPAT_AVAILABLE:
        return
    arch = _get_config_architecture(model_path)
    if arch not in _PACKED_ARCHITECTURES:
        return
    if BitLinearCompatibilityLayer.is_bitlinear_importable():
        return
    try:
        from ...compatibility.transformers_compat import BITNET_PINNED_SPEC
        spec = BITNET_PINNED_SPEC
    except Exception:
        spec = "git+https://github.com/huggingface/transformers.git@096f25ae..."
    raise ImportError(
        f"\n\n"
        f"  BitLinear NOT available in installed transformers.\n"
        f"  Model '{model_path}' uses {arch} (packed 1.58-bit weights).\n"
        f"\n"
        f"  Fix: pip install '{spec}'\n"
        f"\n"
        f"  Or use the bitnet.cpp GGUF path (recommended, no BitLinear needed):\n"
        f"    local-llm-setup --minimal-build\n"
    )


def _get_config_architecture(model_path: str) -> str:
    """Read architectures[0] from config.json without importing transformers."""
    import json as _json
    p = Path(model_path)
    cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
    try:
        cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("architectures", [""])[0]
    except Exception:
        return ""


# =============================================================================
# BitNetInferenceWrapper — main inference engine
# =============================================================================
class BitNetInferenceWrapper:
    """
    Production BitNet inference wrapper with automatic backend selection.

    Initialization modes (v9):
      A) gguf_path provided + force_backend="bitnet_cpp":
         Fastest path — directly creates BitNetCppBackend with no router,
         no transformers model load, no RAM wasted on weights.

      B) model_path provided (safetensors dir):
         Routes via AdaptiveModelRouter → bitnet.cpp if available,
         otherwise transformers (with BitLinear guard for packed models).

      C) Explicit model= + tokenizer= provided:
         Forces the transformers backend. model may be None for GGUF path
         (bitnet_loader v9 passes model=None for bitnet.cpp path).

    Public API (unchanged from v3):
      agenerate(prompt, cfg, stop, request_id) → AsyncIterator[str]
      create_completion(prompt, ...) → Iterator[dict]
      create_chat_completion(messages, ...) → str  [NEW in v9]
      get_memory_usage() → dict
      backend_info() → dict
      shutdown() → None
    """

    def __init__(
        self,
        *,
        # Unified routing
        model_path:  Optional[str] = None,
        config:      Optional[Any] = None,
        device:      str = "cpu",          #FIX: was "cuda" — BitNet kernels are CPU-only
        # Explicit model (may be None for GGUF path)
        model:       Optional[PreTrainedModel]  = None,
        tokenizer:   Optional[PreTrainedTokenizer] = None,
        # Managers
        memory_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        # bitnet.cpp overrides
        bitnet_cpp_port:      int = 18080,
        bitnet_n_threads:     int = 0,
        bitnet_n_gpu_layers:  int = 0,
        #NEW: explicit GGUF path from bitnet_loader
        gguf_path:   Optional[str] = None,
        # Backend selection
        force_backend: Optional[str] = None,  # "transformers" | "bitnet_cpp" | None (auto)
        # Circuit breaker
        cb_failure_threshold: int   = 5,
        cb_recovery_s:        float = 60.0,
    ) -> None:
        self.device          = device
        self.memory_manager  = memory_manager
        self._kernel_manager = kernel_manager
        self._custom_matmul_fn = (
            kernel_manager.get_matmul_fn()
            if kernel_manager and hasattr(kernel_manager, "get_matmul_fn")
            else None
        )
        self.bitnet_cpp_port     = bitnet_cpp_port
        self.bitnet_n_threads    = bitnet_n_threads or self._default_threads()
        self.bitnet_n_gpu_layers = bitnet_n_gpu_layers
        self.force_backend       = force_backend

        # Resolve model_path
        self.model_path = model_path
        if self.model_path is None and config is not None:
            self.model_path = getattr(config, "model_name_or_path",
                                      getattr(config, "model_id", None))
        if self.model_path is not None:
            self.model_path = str(Path(self.model_path).resolve())

        # Resolve gguf_path
        self.gguf_path: Optional[str] = None
        if gguf_path and Path(gguf_path).is_file():
            self.gguf_path = str(Path(gguf_path).resolve())
        elif gguf_path:
            log.warning("gguf_path provided but file not found: %s", gguf_path)

        self.config = config

        # Backend state
        self._backend_type: BitNetBackendType = BitNetBackendType.TRANSFORMERS
        self._cpp_backend:  Optional[Any]     = None
        self._backend_info: Dict[str, Any]    = {}

        # Circuit breaker
        self._circuit_breaker = _CircuitBreaker(cb_failure_threshold, cb_recovery_s)

        # Model / tokenizer (may both be None for GGUF path)
        self.model:     Optional[PreTrainedModel]    = model
        self.tokenizer: Optional[PreTrainedTokenizer] = tokenizer

        # Async semaphore for streaming (one generation at a time)
        self._async_semaphore    = asyncio.Semaphore(1)
        # Sync semaphore for create_completion
        self._generation_semaphore = threading.BoundedSemaphore(1)

        # Shutdown flag with lock
        self._shutdown      = False
        self._shutdown_lock = threading.Lock()

        # Default context size
        self.n_ctx  = 4096
        self._n_ctx = 4096

        # ── Backend initialization ────────────────────────────────────────
        self._initialize_backend()

        # Environment validation (advisory only)
        self._env_status = EnvironmentStatus()
        try:
            self._env_status = validate_environment(raise_on_error=False)
            if (not self._env_status.transformers_ok
                    and self._backend_type in (
                        BitNetBackendType.TRANSFORMERS,
                        BitNetBackendType.TRANSFORMERS_BITLINEAR)):
                log.warning(
                    "Transformers constraint: %s — bitnet.cpp recommended for performance.",
                    self._env_status.transformers_error,
                )
        except Exception as e:
            log.debug("Environment validation skipped: %s", e)

    @staticmethod
    def _default_threads() -> int:
        """Read ATLAS thread count from state file, fall back to CPU count."""
        if _COMPAT_AVAILABLE:
            try:
                t = StateFileReader.get("atlas_threads")
                if t and isinstance(t, int) and t > 0:
                    return t
            except Exception:
                pass
        import os
        return min(os.cpu_count() or 4, 8)

    # ── Backend initialization router ─────────────────────────────────────
    def _initialize_backend(self) -> None:
        """
        Route to the correct initialization path.
        Priority:
          1. gguf_path + force_backend="bitnet_cpp" → direct GGUF init (fastest)
          2. force_backend="transformers" → transformers only
          3. model_path + auto-routing via AdaptiveModelRouter
          4. Explicit model= + tokenizer= passed (model may be None for GGUF path)
        """
        # Priority 1: explicit GGUF path + bitnet_cpp request → fastest path
        if self.gguf_path and (
            self.force_backend == "bitnet_cpp"
            or self.force_backend == "gguf"
        ):
            log.info("Init: explicit GGUF path → bitnet.cpp backend")
            self._init_gguf_backend()
            return

        # Priority 2: force transformers (or model=None with no GGUF path)
        if self.force_backend == "transformers":
            if self.model is None and self.model_path is None:
                raise ValueError(
                    "force_backend='transformers' requires model= or model_path="
                )
            if self.model is not None:
                self._init_transformers_attributes()
            elif self.model_path:
                self._init_transformers_from_path()
            return

        # Priority 3: model_path provided → use router
        if self.model_path is not None and self.force_backend != "transformers":
            self._init_router_backend()
            return

        # Priority 4: explicit model= passed (may be None for GGUF path)
        if self.model is not None:
            self._backend_type = BitNetBackendType.TRANSFORMERS
            self._init_transformers_attributes()
            return

        # Priority 4b: model=None, gguf_path provided (bitnet_loader v9 GGUF path)
        if self.model is None and self.gguf_path:
            log.info("Init: model=None with gguf_path → bitnet.cpp backend")
            self.force_backend = "bitnet_cpp"
            self._init_gguf_backend()
            return

        # Last resort: error
        raise ValueError(
            "BitNetInferenceWrapper requires one of:\n"
            "  • gguf_path= (for bitnet.cpp)\n"
            "  • model_path= (for auto-routing)\n"
            "  • model= + tokenizer= (for transformers)\n"
            "  • model_path= + force_backend='transformers'"
        )

    # ── GGUF / bitnet.cpp initialization ──────────────────────────────────
    def _init_gguf_backend(self) -> None:
        """
        Initialize the bitnet.cpp backend directly with the explicit GGUF path.
        No router, no transformers loading — fastest possible path.
        """
        if not _BITNET_CPP_AVAILABLE:
            log.warning("bitnet.cpp bindings not available — falling back to transformers")
            if self.model is not None:
                self._init_transformers_attributes()
            elif self.model_path:
                self._init_transformers_from_path()
            else:
                raise RuntimeError(
                    "bitnet.cpp not available and no transformers model or path provided"
                )
            return

        gguf_path = self.gguf_path or self.model_path
        if not gguf_path:
            raise ValueError("_init_gguf_backend: no gguf_path or model_path")

        # Try state file for binary paths
        server_bin: Optional[Path] = None
        run_script_path: Optional[str] = None
        if _COMPAT_AVAILABLE:
            srv = StateFileReader.get_llama_server()
            if srv:
                server_bin = Path(srv)
            run_script = StateFileReader.get("run_inference_script")
            if run_script and Path(run_script).is_file():
                run_script_path = run_script

        gguf_p = Path(gguf_path)

        if server_bin and server_bin.is_file() and os.access(server_bin, os.X_OK):
            try:
                self._cpp_backend = BitNetCppBackend(
                    server_binary    = server_bin,
                    gguf_model       = gguf_p,
                    port             = self.bitnet_cpp_port,
                    n_threads        = self.bitnet_n_threads,
                    n_gpu_layers     = self.bitnet_n_gpu_layers,
                    bitnet_dir       = str(server_bin.parent.parent.parent),
                    spectre_gguf_py  = (StateFileReader.get_spectre_pythonpath()
                                        if _COMPAT_AVAILABLE else None),
                )
                self._backend_type = BitNetBackendType.BITNET_CPP_SERVER
                log.info("GGUF backend: llama-server at %s", server_bin)
                return
            except Exception as e:
                log.warning("Failed to create llama-server backend: %s", e)

        if run_script_path:
            try:
                self._cpp_backend = BitNetRunInferenceBackend(
                    bitnet_repo_dir  = str(Path(run_script_path).parent),
                    gguf_model       = str(gguf_p),
                    n_threads        = self.bitnet_n_threads,
                    spectre_gguf_py  = (StateFileReader.get_spectre_pythonpath()
                                        if _COMPAT_AVAILABLE else None),
                )
                self._backend_type = BitNetBackendType.BITNET_CPP_PROCESS
                log.info("GGUF backend: run_inference.py at %s", run_script_path)
                return
            except Exception as e:
                log.warning("Failed to create run_inference backend: %s", e)

        # Factory fallback
        try:
            result = create_best_backend(
                model_dir        = str(gguf_p) if gguf_p.is_dir() else str(gguf_p.parent),
                port             = self.bitnet_cpp_port,
                n_threads        = self.bitnet_n_threads,
                n_gpu_layers     = self.bitnet_n_gpu_layers,
            )
            if result.available:
                self._cpp_backend = result.backend
                self._backend_type = (
                    BitNetBackendType.BITNET_CPP_SERVER
                    if result.tier == "server"
                    else BitNetBackendType.BITNET_CPP_PROCESS
                )
                log.info("GGUF backend via factory: tier=%s", result.tier)
                return
        except Exception as e:
            log.warning("Factory backend creation failed: %s", e)

        # ── bitnet.cpp not found — resolve a valid transformers fallback path ─
        # self.model_path may be the raw .gguf FILE path.
        # AutoModelForCausalLM.from_pretrained() requires a *directory* (or HF
        # repo ID), so passing a .gguf file raises:
        #   "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"
        # Resolution order:
        #   1. config.tokenizer_path  — explicit HF model dir in config.yaml
        #   2. config.model_name_or_path if it's a directory (not the .gguf)
        #   3. Parent dir of the .gguf file if it contains config.json
        #   4. Give up with a clear, actionable error (don't call from_pretrained)
        log.warning(
            "Could not create bitnet.cpp backend for GGUF. "
            "Falling back to transformers (no performance benefit)."
        )

        if self.model is not None:
            # Model already in memory — use it directly, no path needed
            self._init_transformers_attributes()
            return

        _tf_path: Optional[str] = self.model_path
        _is_gguf = _tf_path and Path(_tf_path).suffix.lower() == ".gguf"

        if _is_gguf:
            _cfg = self.config  # BitNetModelConfig or None
            _candidates = [
                getattr(_cfg, "tokenizer_path",       None) if _cfg else None,
                getattr(_cfg, "model_name_or_path",   None) if _cfg else None,
                str(Path(_tf_path).parent) if _tf_path else None,
            ]
            _resolved: Optional[str] = None
            for _cand in _candidates:
                if _cand and Path(_cand).is_dir() and (Path(_cand) / "config.json").exists():
                    _resolved = _cand
                    break

            if _resolved:
                log.info(
                    "GGUF→transformers fallback: no bitnet.cpp binary found.\n"
                    "  GGUF path    : %s\n"
                    "  HF model dir : %s  (performance: significantly reduced vs bitnet.cpp)\n"
                    "  Run `local-llm-setup` to restore bitnet.cpp performance.",
                    _tf_path, _resolved,
                )
                _saved          = self.model_path
                self.model_path = _resolved
                try:
                    self._init_transformers_from_path()
                except Exception:
                    self.model_path = _saved
                    raise
                return
            else:
                # No valid HF model directory anywhere — cannot load
                log.error(
                    "bitnet.cpp binary not found AND no HF model directory available "
                    "for transformers fallback.\n"
                    "  GGUF path : %s\n"
                    "  Fix A     : run `local-llm-setup` to build bitnet.cpp binary\n"
                    "  Fix B     : set `tokenizer_path` in config.yaml to the HuggingFace "
                    "model directory (e.g. /path/to/BitNet-b1.58-2B-4T)",
                    _tf_path,
                )
                self._backend_type = BitNetBackendType.TRANSFORMERS
                return

        # model_path is a directory or HF ID — standard transformers path
        if _tf_path:
            self._init_transformers_from_path()
        else:
            self._backend_type = BitNetBackendType.TRANSFORMERS
            log.error("No transformers fallback available — wrapper will be non-functional")

    # ── Router-based backend initialization ───────────────────────────────
    def _init_router_backend(self) -> None:
        """Use AdaptiveModelRouter to select the best backend for model_path."""
        router = AdaptiveModelRouter()
        try:
            header, backend_enum, params = router.route(self.model_path)
        except Exception as e:
            log.error("Router failed: %s — falling back to transformers", e)
            self._fallback_to_transformers(params=None)
            return

        if getattr(header, "parse_error", None):
            log.warning("GGUF header parse error: %s — falling back to transformers",
                        header.parse_error)
            self._fallback_to_transformers(params=None)
            return

        backend_map = {
            InferenceBackend.BITNET_CPP_SERVER:   BitNetBackendType.BITNET_CPP_SERVER,
            InferenceBackend.BITNET_CPP_PROCESS:  BitNetBackendType.BITNET_CPP_PROCESS,
            InferenceBackend.BITNET_TRANSFORMERS: BitNetBackendType.TRANSFORMERS,
            InferenceBackend.LLAMA_CPP_PYTHON:    BitNetBackendType.TRANSFORMERS,
            InferenceBackend.UNKNOWN:             BitNetBackendType.TRANSFORMERS,
        }
        self._backend_type = backend_map.get(backend_enum, BitNetBackendType.TRANSFORMERS)

        if self._backend_type in (
            BitNetBackendType.BITNET_CPP_SERVER,
            BitNetBackendType.BITNET_CPP_PROCESS,
        ):
            self._init_cpp_backend_from_params(params)
        else:
            self._fallback_to_transformers(params)

    def _init_cpp_backend_from_params(self, params: Dict[str, Any]) -> None:
        if not _BITNET_CPP_AVAILABLE:
            log.warning("bitnet.cpp not available — falling back to transformers")
            self._fallback_to_transformers(params)
            return

        gguf_path = str(params.get("gguf_path") or self.gguf_path or self.model_path)

        if self._backend_type == BitNetBackendType.BITNET_CPP_SERVER:
            try:
                server_binary = params.get("server_binary")
                port   = params.get("port", self.bitnet_cpp_port)
                host   = params.get("host", "127.0.0.1")
                if server_binary and Path(server_binary).exists():
                    self._cpp_backend = BitNetCppBackend(
                        server_binary    = Path(server_binary),
                        gguf_model       = Path(gguf_path),
                        host             = host,
                        port             = port,
                        n_threads        = self.bitnet_n_threads,
                        n_gpu_layers     = self.bitnet_n_gpu_layers,
                    )
                else:
                    result = create_best_backend(
                        model_dir  = gguf_path,
                        host       = host,
                        port       = port,
                        n_threads  = self.bitnet_n_threads,
                        n_gpu_layers = self.bitnet_n_gpu_layers,
                    )
                    if result.available:
                        self._cpp_backend = result.backend
                    else:
                        raise RuntimeError("create_best_backend returned no backend")
                log.info("C++ server backend ready (gguf=%s)", gguf_path)
            except Exception as e:
                log.error("C++ server backend init failed: %s — falling back", e)
                self._fallback_to_transformers(params)

        elif self._backend_type == BitNetBackendType.BITNET_CPP_PROCESS:
            try:
                run_script = params.get("run_inference_script")
                if run_script and Path(run_script).is_file():
                    self._cpp_backend = BitNetRunInferenceBackend(
                        bitnet_repo_dir = str(Path(run_script).parent),
                        gguf_model      = gguf_path,
                        n_threads       = self.bitnet_n_threads,
                    )
                    log.info("C++ process backend ready (run_inference.py)")
                else:
                    result = create_best_backend(
                        model_dir  = gguf_path,
                        n_threads  = self.bitnet_n_threads,
                    )
                    if result.available:
                        self._cpp_backend = result.backend
                    else:
                        raise RuntimeError("No process backend available")
            except Exception as e:
                log.error("C++ process backend init failed: %s — falling back", e)
                self._fallback_to_transformers(params)

    def _fallback_to_transformers(self, params: Optional[Dict[str, Any]]) -> None:
        """
        Fall back to transformers backend.
        v9 FIX: If model= was already passed, use it — don't reload from disk.
        """
        self._backend_type = BitNetBackendType.TRANSFORMERS
        if self.model is not None:
            # Model already loaded by bitnet_loader — use it directly
            log.info("Transformers fallback: using already-loaded model")
            self._init_transformers_attributes()
        elif self.model_path:
            self._init_transformers_from_path()
        else:
            log.error("No model or model_path for transformers fallback — inference will fail")

    # ── Transformers initialization helpers ───────────────────────────────
    def _init_transformers_from_path(self) -> None:
        """
        Load transformers model from model_path.
        v9: Uses bfloat16 + BitLinear guard + explicit device_map (not "auto").
        """
        if not self.model_path:
            raise ValueError("model_path required for transformers init from path")

        #FIX: BitLinear guard before loading
        _check_bitlinear_before_load(self.model_path)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            arch = _get_config_architecture(self.model_path)
            is_packed = arch in _PACKED_ARCHITECTURES

            device_map = self.device if self.device != "auto" else "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype   = torch.bfloat16,   # v9 FIX: was "auto"
                device_map    = device_map,        # v9 FIX: was "auto" always
                trust_remote_code = True,
                low_cpu_mem_usage = True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model.eval()
            self.config = self.model.config

            if is_packed:
                self._backend_type = BitNetBackendType.TRANSFORMERS_BITLINEAR
                log.info("Transformers BitLinear backend from path: %s", self.model_path)
            else:
                self._backend_type = BitNetBackendType.TRANSFORMERS
                log.info("Transformers standard backend from path: %s", self.model_path)

        except ImportError:
            raise  # BitLinear error — propagate as-is
        except Exception as e:
            raise RuntimeError(
                f"Failed to load transformers model from '{self.model_path}': {e}"
            ) from e

        self._init_transformers_attributes()

    def _init_transformers_attributes(self) -> None:
        """
        Set derived attributes for the transformers backend.
        v9 FIX: All model-access code guarded for model=None (GGUF path).
        """
        if self.model is not None:
            model_config = getattr(self.model, "config", None)
            if model_config is None and self.config is not None:
                model_config = self.config

            self.n_ctx  = getattr(model_config, "max_position_embeddings",
                          getattr(model_config, "context_length", 4096)) if model_config else 4096
            self._n_ctx = self.n_ctx
            self.bits   = self._detect_bits()
            self.is_quantized   = getattr(model_config, "quantized", True) if model_config else True
            self.quantization_type = getattr(model_config, "quantization", "1bit") if model_config else "1bit"

            if hasattr(self.model, "to") and self.device not in ("auto", "cpu"):
                try:
                    self.model.to(self.device)
                except Exception as e:
                    log.warning("model.to(%s) failed: %s — keeping current device", self.device, e)
            self.model.eval()
        else:
            # GGUF path — model is None, set safe defaults
            self.n_ctx             = 4096
            self._n_ctx            = 4096
            self.bits              = 1
            self.is_quantized      = True
            self.quantization_type = "1bit"

        # Adaptive timeouts
        if "cpu" in str(self.device).lower():
            self._adaptive_first_timeout = 120.0
            self._adaptive_total_timeout = 600.0
        else:
            self._adaptive_first_timeout = 30.0
            self._adaptive_total_timeout = 300.0
        self._adaptive_per_token_timeout = 90.0

    def _detect_bits(self) -> int:
        """Detect weight precision from model parameters."""
        if self.model is None:
            return 1  # BitNet GGUF
        if hasattr(self.config, "quantization_bits"):
            return self.config.quantization_bits
        try:
            first_param = next(self.model.parameters())
            dtype = first_param.dtype
            if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
                return 8
            if dtype == torch.int4:
                return 4
            if dtype in (torch.float16, torch.bfloat16):
                return 16
            if dtype == torch.float32:
                return 32
        except StopIteration:
            pass
        return 1  # Default BitNet

    # ── Public API ────────────────────────────────────────────────────────
    def create_completion(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
        top_p:       Optional[float] = None,
        top_k:       Optional[int]   = None,
        stop:        Optional[List[str]] = None,
        stream:      bool = True,
        **kwargs: Any,
    ) -> Any:
        """llama.cpp-style synchronous completion interface."""
        cfg = self._build_runtime_config(locals())
        self._resolve_timeouts(cfg)
        if stream:
            yield from self._stream_completion(prompt, cfg, stop, **kwargs)
        else:
            text = self._sync_completion(prompt, cfg, stop, **kwargs)
            yield {"choices": [{"text": text}]}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.create_completion(*args, **kwargs)

    def create_chat_completion(
        self,
        messages:    List[Dict[str, str]],
        system:      Optional[str] = None,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        top_k:       int   = 40,
        stop:        Optional[List[str]] = None,
    ) -> str:
        """
        NEW in v9: OpenAI-style chat completion interface.
        Applies BitNet b1.58 chat template (or tokenizer template if available).
        """
        if self.tokenizer is not None:
            prompt = _apply_chat_template_if_available(self.tokenizer, messages)
        else:
            prompt = _format_bitnet_chat(messages, system=system)

        effective_stop = list(stop or []) + _BITNET_STOP_TOKENS
        seen: set = set()
        effective_stop = [s for s in effective_stop if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        tokens: List[str] = []
        for chunk in self.create_completion(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=effective_stop, stream=True,
        ):
            tokens.append(chunk["choices"][0]["text"])
        return "".join(tokens)

    # ── Async generation (routes to active backend) ───────────────────────
    async def agenerate(
        self,
        prompt:     str,
        cfg:        Optional[BitNetInferenceConfig] = None,
        stop:       Optional[List[str]] = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Main async token streaming interface.
        Routes to bitnet.cpp or transformers based on active backend.
        """
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Wrapper is shutting down — cannot generate")

        if not request_id:
            request_id = str(uuid.uuid4())
        if cfg is None:
            cfg = BitNetInferenceConfig()
        self._resolve_timeouts(cfg)

        backend_name = self._backend_type.value
        log.debug(
            "agenerate start | id=%s prompt_len=%d max_tokens=%d backend=%s",
            request_id[:8], len(prompt), cfg.max_tokens, backend_name,
        )

        if _METRICS.get("bitnet_active"):
            _METRICS["bitnet_active"].inc()

        t_start      = time.monotonic()
        token_count  = 0
        first_token_t: Optional[float] = None
        error_str:   Optional[str] = None

        try:
            # ── C++ backend path (bitnet.cpp) ────────────────────────────
            if self._cpp_backend is not None:
                if self._circuit_breaker.is_open():
                    log.warning("CircuitBreaker OPEN — falling back to transformers for id=%s",
                                request_id[:8])
                    async for tok in self._agenerate_transformers(
                        prompt, cfg, stop, request_id, **kwargs
                    ):
                        if first_token_t is None:
                            first_token_t = time.monotonic()
                        token_count += 1
                        yield tok
                    return

                # Start server if needed (idempotent)
                if hasattr(self._cpp_backend, "start"):
                    started = await self._cpp_backend.start()
                    if not started:
                        log.warning(
                            "C++ backend start failed — falling back | id=%s",
                            request_id[:8]
                        )
                        self._circuit_breaker.record_failure()
                        async for tok in self._agenerate_transformers(
                            prompt, cfg, stop, request_id, **kwargs
                        ):
                            if first_token_t is None:
                                first_token_t = time.monotonic()
                            token_count += 1
                            yield tok
                        return

                cpp_cfg = self._to_cpp_config(cfg)
                try:
                    async for tok in self._cpp_backend.agenerate(
                        prompt, cfg=cpp_cfg, stop=stop or cfg.stop
                    ):
                        if first_token_t is None:
                            first_token_t = time.monotonic()
                        token_count += 1
                        yield tok
                    self._circuit_breaker.record_success()
                except Exception as exc:
                    error_str = str(exc)[:200]
                    self._circuit_breaker.record_failure()
                    log.warning(
                        "C++ backend error: %s — falling back to transformers | id=%s",
                        exc, request_id[:8],
                    )
                    # Fallback to transformers if possible
                    if self.model is not None:
                        async for tok in self._agenerate_transformers(
                            prompt, cfg, stop, request_id, **kwargs
                        ):
                            if first_token_t is None:
                                first_token_t = time.monotonic()
                            token_count += 1
                            yield tok
                        error_str = None  # recovered
                    else:
                        raise

            # ── Transformers path ────────────────────────────────────────
            else:
                async for tok in self._agenerate_transformers(
                    prompt, cfg, stop, request_id, **kwargs
                ):
                    if first_token_t is None:
                        first_token_t = time.monotonic()
                    token_count += 1
                    yield tok

        except asyncio.CancelledError:
            error_str = "Cancelled"
            raise
        except Exception as exc:
            error_str = str(exc)[:200]
            raise
        finally:
            if _METRICS.get("bitnet_active"):
                _METRICS["bitnet_active"].dec()

            total_ms = (time.monotonic() - t_start) * 1000
            ttft_ms  = ((first_token_t - t_start) * 1000) if first_token_t else 0.0
            tps      = token_count / max(total_ms / 1000, 1e-9)

            self._record_metrics(
                backend    = backend_name,
                prompt_toks = 0,
                output_toks = token_count,
                ttft_ms     = ttft_ms,
                total_ms    = total_ms,
                error       = error_str,
            )
            log.info(
                "agenerate done | id=%s tokens=%d ttft=%.1fms total=%.1fms "
                "tps=%.2f backend=%s error=%s",
                request_id[:8], token_count, ttft_ms, total_ms, tps,
                backend_name, error_str or "None",
            )

    def _to_cpp_config(self, cfg: BitNetInferenceConfig) -> Optional[Any]:
        if BitNetCppConfig is None:
            return None
        return BitNetCppConfig(
            max_tokens          = cfg.max_tokens,
            temperature         = cfg.temperature,
            top_p               = cfg.top_p,
            top_k               = cfg.top_k,
            repetition_penalty  = cfg.repetition_penalty,
            first_token_timeout = cfg.first_token_timeout,
            total_timeout       = cfg.total_timeout,
            stop                = cfg.stop,
            seed                = cfg.seed,
            n_ctx               = cfg.n_ctx,
        )

    # ── Transformers async generation ─────────────────────────────────────
    async def _agenerate_transformers(
        self,
        prompt:     str,
        cfg:        BitNetInferenceConfig,
        stop:       Optional[List[str]],
        request_id: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async token streaming via transformers TextIteratorStreamer.
        v9 FIX: Guards model is not None before all model-access.
        v9 NEW: Applies chat template detection.
        """
        # v9 FIX: model=None guard
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Transformers backend requested but model/tokenizer not loaded. "
                "Use the bitnet.cpp backend (gguf_path) or provide model= at init."
            )

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        cfg.max_tokens = min(max(cfg.max_tokens, 1), 32768)
        if cfg.total_timeout < cfg.first_token_timeout:
            cfg.total_timeout = cfg.first_token_timeout + 10.0

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Union[str, Exception, None]] = asyncio.Queue(maxsize=1024)
        metrics_input_tokens = 0
        start_time  = time.monotonic()
        token_count = 0

        effective_stop = list(stop or cfg.stop or [])

        def producer() -> None:
            try:
                inputs       = self.tokenizer(prompt, return_tensors="pt")
                input_ids    = inputs["input_ids"].to(self.device)
                attn_mask    = inputs.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)
                input_tokens = input_ids.shape[1]
                nonlocal metrics_input_tokens
                metrics_input_tokens = input_tokens

                if input_tokens > self.n_ctx:
                    raise RuntimeError(
                        f"Prompt exceeds context window: {input_tokens} > {self.n_ctx}"
                    )

                stopping_criteria = None
                if effective_stop:
                    try:
                        stopping_criteria = StoppingCriteriaList(
                            [StopStringCriteria(effective_stop, self.tokenizer)]
                        )
                    except Exception as exc:
                        log.warning("StopCriteria init failed: %s", exc)

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt         = True,
                    skip_special_tokens = True,
                    decode_kwargs       = {"skip_special_tokens": True},
                )

                _do_sample = cfg.temperature > 0.01
                gen_kwargs: Dict[str, Any] = {
                    "input_ids":          input_ids,
                    "max_new_tokens":     cfg.max_tokens,
                    "do_sample":          _do_sample,
                    "streamer":           streamer,
                    "pad_token_id":       self.tokenizer.eos_token_id,
                    "repetition_penalty": cfg.repetition_penalty,
                    "length_penalty":     cfg.length_penalty,
                }
                if _do_sample:
                    gen_kwargs["temperature"] = max(cfg.temperature, 0.05)
                    gen_kwargs["top_p"]       = cfg.top_p
                    gen_kwargs["top_k"]       = cfg.top_k
                if attn_mask is not None:
                    gen_kwargs["attention_mask"] = attn_mask
                if cfg.num_beams > 1:
                    gen_kwargs["num_beams"]    = cfg.num_beams
                    gen_kwargs["early_stopping"] = cfg.early_stopping
                if stopping_criteria:
                    gen_kwargs["stopping_criteria"] = stopping_criteria
                safe_extra = {
                    k: v for k, v in kwargs.items()
                    if k not in gen_kwargs and not k.startswith("_")
                }
                gen_kwargs.update(safe_extra)

                with self._generation_semaphore:
                    gen_thread = threading.Thread(
                        target = self._safe_generate,
                        kwargs = gen_kwargs,
                        daemon = True,
                        name   = f"bitnet-gen-{request_id[:8]}",
                    )
                    gen_thread.start()
                    for token in streamer:
                        if token:
                            asyncio.run_coroutine_threadsafe(
                                token_queue.put(token), loop
                            ).result(timeout=10.0)
                    gen_thread.join()

            except Exception as exc:
                try:
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put(exc), loop
                    ).result(timeout=2.0)
                except Exception:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put(None), loop
                    ).result(timeout=2.0)
                except Exception:
                    pass

        loop.run_in_executor(None, producer)

        # First-token wait
        try:
            token_or_exc = await asyncio.wait_for(
                token_queue.get(), timeout=cfg.first_token_timeout
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"First token timeout ({cfg.first_token_timeout:.0f}s) for request {request_id[:8]}"
            )

        if isinstance(token_or_exc, Exception):
            raise token_or_exc
        if token_or_exc is None:
            return

        token_count = 1
        yield token_or_exc

        # Stream remaining tokens
        deadline = time.monotonic() + cfg.total_timeout
        _per_tok_t = self._adaptive_per_token_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"Total generation timeout ({cfg.total_timeout:.0f}s) exceeded"
                )
            try:
                token_or_exc = await asyncio.wait_for(
                    token_queue.get(), timeout=min(remaining, _per_tok_t)
                )
            except asyncio.TimeoutError:
                log.warning(
                    "Per-token timeout (%.1fs) hit | id=%s token=%d",
                    _per_tok_t, request_id[:8], token_count,
                )
                break
            if token_or_exc is None:
                break
            if isinstance(token_or_exc, Exception):
                raise token_or_exc
            token_count += 1
            yield token_or_exc

    def _safe_generate(self, **gen_kwargs: Any) -> torch.LongTensor:
        """
        Wrapper around model.generate() with BitNet 1-bit stability fixes.
        v9 FIX: Guards model is not None.
        """
        if self.model is None:
            raise RuntimeError("_safe_generate: model is None (GGUF path should not reach here)")

        quant_type = getattr(self, "quantization_type", "1bit")
        if quant_type == "1bit" and gen_kwargs.get("do_sample", False):
            temp = gen_kwargs.get("temperature", 0.7)
            gen_kwargs["temperature"] = max(0.05, min(float(temp), 0.9))
            if gen_kwargs.get("top_p", 1.0) > 0.95:
                gen_kwargs["top_p"] = 0.95
            if gen_kwargs["temperature"] < 0.1:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)

        max_retries = 3
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                with torch.no_grad():
                    return self.model.generate(**gen_kwargs)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if any(kw in msg for kw in ("nan", "inf", "overflow", "numerical")):
                    log.warning(
                        "BitNet numerical instability attempt %d/%d — falling back to greedy",
                        attempt + 1, max_retries,
                    )
                    gen_kwargs["do_sample"] = False
                    gen_kwargs.pop("temperature", None)
                    gen_kwargs.pop("top_p", None)
                    gen_kwargs.pop("top_k", None)
                    last_exc = exc
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    log.warning("generate attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(0.1)
                else:
                    raise
        raise RuntimeError(f"generate failed after {max_retries} attempts") from last_exc

    # ── Synchronous streaming wrappers ────────────────────────────────────
    def _build_runtime_config(self, overrides: Dict[str, Any]) -> BitNetInferenceConfig:
        cfg = BitNetInferenceConfig()
        for key in ("max_tokens", "temperature", "top_p", "top_k"):
            val = overrides.get(key)
            if val is not None:
                setattr(cfg, key, val)
        return cfg

    def _resolve_timeouts(self, cfg: BitNetInferenceConfig) -> None:
        if cfg.first_token_timeout <= 0:
            cfg.first_token_timeout = getattr(self, "_adaptive_first_timeout", 120.0)
        if cfg.total_timeout <= 0:
            cfg.total_timeout = getattr(self, "_adaptive_total_timeout", 600.0)

    def _stream_completion(
        self, prompt: str, cfg: BitNetInferenceConfig,
        stop: Optional[List[str]], **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        loop = asyncio.new_event_loop()
        try:
            async def _gen():
                async for tok in self.agenerate(prompt, cfg=cfg, stop=stop, **kwargs):
                    yield tok
            agen = _gen()
            while True:
                try:
                    tok = loop.run_until_complete(agen.__anext__())
                    yield {"choices": [{"text": tok}]}
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def _sync_completion(
        self, prompt: str, cfg: BitNetInferenceConfig,
        stop: Optional[List[str]], **kwargs: Any,
    ) -> str:
        return "".join(
            chunk["choices"][0]["text"]
            for chunk in self._stream_completion(prompt, cfg, stop, **kwargs)
        )

    # ── Metrics recording ─────────────────────────────────────────────────
    def _record_metrics(
        self,
        backend:    str,
        prompt_toks: int,
        output_toks: int,
        ttft_ms:    float,
        total_ms:   float,
        error:      Optional[str],
    ) -> None:
        try:
            status = "error" if error else "success"
            tps = output_toks / max(total_ms / 1000, 1e-9)
            if _METRICS.get("bitnet_requests_total"):
                _METRICS["bitnet_requests_total"].labels(
                    backend=backend, status=status
                ).inc()
            if _METRICS.get("bitnet_tokens_total"):
                _METRICS["bitnet_tokens_total"].labels(backend=backend).inc(output_toks)
            if _METRICS.get("bitnet_ttft_ms") and ttft_ms > 0:
                _METRICS["bitnet_ttft_ms"].observe(ttft_ms)
            if _METRICS.get("bitnet_tps") and tps > 0:
                _METRICS["bitnet_tps"].observe(tps)
        except Exception as e:
            log.debug("Metrics error: %s", e)

    # ── Warm-up ───────────────────────────────────────────────────────────
    async def warm_up(self, n_tokens: int = 1) -> bool:
        """
        NEW in v9: Run a 1-token probe generation to pre-load llama-server
        and warm up the model. Returns True if successful.
        """
        try:
            cfg = BitNetInferenceConfig(max_tokens=n_tokens, temperature=0.0)
            self._resolve_timeouts(cfg)
            tokens: List[str] = []
            async for tok in self.agenerate("Hello", cfg=cfg, request_id="warmup"):
                tokens.append(tok)
            log.info("Warm-up complete: %d token(s) generated", len(tokens))
            return True
        except Exception as e:
            log.warning("Warm-up failed: %s", e)
            return False

    # ── Utility methods ───────────────────────────────────────────────────
    def get_memory_usage(self) -> Dict[str, Any]:
        """v9 FIX: Handles model=None for GGUF path."""
        if self._backend_type in (
            BitNetBackendType.BITNET_CPP_SERVER,
            BitNetBackendType.BITNET_CPP_PROCESS,
        ):
            if hasattr(self._cpp_backend, "stats"):
                return self._cpp_backend.stats()
            gguf_size_mb = 0
            if self.gguf_path and Path(self.gguf_path).is_file():
                gguf_size_mb = Path(self.gguf_path).stat().st_size // (1024 * 1024)
            return {
                "active_backend": self._backend_type.value,
                "gguf_size_mb":   gguf_size_mb,
                "vram_mb":        0,
                "total_parameters": 0,
            }
        # Transformers path
        if self.model is None:
            return {"active_backend": self._backend_type.value, "model": "not_loaded"}
        total_params = sum(p.numel() for p in self.model.parameters())
        vram_mb = 0
        if torch.cuda.is_available():
            try:
                vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            except Exception:
                pass
        return {
            "total_parameters": total_params,
            "vram_mb":          vram_mb,
            "model_dtype":      str(next(self.model.parameters()).dtype),
            "quantization":     getattr(self, "quantization_type", "1bit"),
            "bits":             getattr(self, "bits", 1),
            "active_backend":   self._backend_type.value,
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        if self.tokenizer is not None:
            try:
                inputs = self.tokenizer(text, return_tensors="pt")
                return inputs["input_ids"].shape[1]
            except Exception:
                pass
        return max(1, len(text) // 4)

    def backend_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "backend_type":    self._backend_type.value,
            "has_cpp_backend": self._cpp_backend is not None,
            "device":          self.device,
            "model_path":      self.model_path,
            "gguf_path":       self.gguf_path,
            "n_ctx":           self.n_ctx,
            "n_threads":       self.bitnet_n_threads,
            "circuit_breaker": self._circuit_breaker.state,
        }
        if self._backend_type in (
            BitNetBackendType.TRANSFORMERS,
            BitNetBackendType.TRANSFORMERS_BITLINEAR,
        ):
            info["quantization"]  = getattr(self, "quantization_type", "1bit")
            info["model_loaded"]  = self.model is not None
        if self._cpp_backend and hasattr(self._cpp_backend, "stats"):
            info.update(self._cpp_backend.stats())
        return info

    def get_status(self) -> Dict[str, Any]:
        """Rich diagnostic dict for health endpoints and logging."""
        return {
            **self.backend_info(),
            **self.get_memory_usage(),
            "shutdown":           self._shutdown,
            "tokenizer_loaded":   self.tokenizer is not None,
            "bitlinear_available": (
                BitLinearCompatibilityLayer.is_bitlinear_importable()
                if _COMPAT_AVAILABLE else None
            ),
        }

    async def shutdown(self) -> None:
        """
        Graceful shutdown. v9 FIX: atomic flag + waits for C++ backend stop.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        log.info("BitNetInferenceWrapper shutting down (backend=%s)", self._backend_type.value)

        if self._cpp_backend is not None and hasattr(self._cpp_backend, "stop"):
            try:
                await self._cpp_backend.stop()
            except Exception as e:
                log.warning("C++ backend stop error: %s", e)

        # Free transformers model memory
        if self.model is not None:
            try:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        log.info("BitNetInferenceWrapper shutdown complete")

    @asynccontextmanager
    async def generation_session(self, request_id: Optional[str] = None):
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - start
            log.debug("Request %s completed in %.2fs", request_id, elapsed)


# =============================================================================
# Module-level convenience
# =============================================================================
def get_router() -> AdaptiveModelRouter:
    """Expose the singleton router for external usage."""
    return AdaptiveModelRouter()
