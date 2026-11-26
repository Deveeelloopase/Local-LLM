"""
engine/bitnet/__init__.py  v1.0.0
==================================
BitNet integration module — production-ready lazy-loading API.
Public API:
  Model loading:     BitNetLoader, BitNetModelConfig, BitNetInferenceWrapper,
                     BitNetInferenceConfig, GenerationMetrics
  Classification:    PRISMModelClassifier, HardwareDetector
  State / routing:   BitNetState, get_bitnet_state, AdaptiveModelRouter,
                     InferenceBackend, GGUFModelType, classify_gguf,
                     route_model, auto_detect_backend, validate_environment,
                     EnvironmentStatus
  C++ backend:       BitNetCppBackend, BitNetCppChatBackend,
                     BitNetRunInferenceBackend, BitNetCppConfig,
                     BackendResult, create_best_backend
  Compat layer:      LoadResult
  Managers:          BitNetMemoryManager, BitNetKernelManager (alias)
  Utilities:         initialize, get_status
"""

from __future__ import annotations

__version__ = "1.0.0"

import importlib
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Apply SPECTRE PYTHONPATH at package-import time ──────────────────────────
# Ensures the local gguf-py (with TL1/TL2) is on sys.path before any
# sub-module within engine.bitnet imports `gguf`. This prevents the
# AttributeError: type object 'GGMLQuantizationType' has no attribute 'TL1'
# from appearing in any module that imports from this package.
try:
    from ...compatibility.transformers_compat import StateFileReader as _SFR
    _SFR.apply_spectre_pythonpath()
    del _SFR
except Exception:
    pass  # SPECTRE is best-effort — never block package import

# =============================================================================
# Lazy Import Registry
# =============================================================================
# Format: "PublicName": ("relative.module.path", "AttributeName")
#
# Rules for relative paths:
#   ".module"    → local sibling  (engine/bitnet/module.py)
#   "..module"   → parent package (engine/module.py)
#   "..pkg.mod"  → parent pkg     (engine/pkg/mod.py)
# NOTE: importlib.import_module uses Python dotted notation, NOT filesystem
# slashes. The original "../adaptive_awareness.model_unified_env" was broken.
# =============================================================================

_LAZY_IMPORTS: Dict[str, tuple] = {

    # ── Core loader + engine ─────────────────────────────────────────────────
    "BitNetLoader":
        (".bitnet_loader", "BitNetLoader"),
    "BitNetModelConfig":
        (".bitnet_loader", "BitNetModelConfig"),
    "PRISMModelClassifier":          # v9 NEW: inline PRISM classifier
        (".bitnet_loader", "PRISMModelClassifier"),
    "HardwareDetector":              # v9 NEW: device + backend detector
        (".bitnet_loader", "HardwareDetector"),

    "BitNetInferenceWrapper":
        (".bitnet_engine", "BitNetInferenceWrapper"),
    "BitNetInferenceConfig":         # v9 NEW: per-request inference config
        (".bitnet_engine", "BitNetInferenceConfig"),
    "GenerationMetrics":             # v9 NEW: per-request metrics dataclass
        (".bitnet_engine", "GenerationMetrics"),
    "get_router":                    # v9 NEW: expose singleton router
        (".bitnet_engine", "get_router"),

    # ── Memory + kernel managers ─────────────────────────────────────────────
    "BitNetMemoryManager":
        (".bitnet_memory_manager", "BitNetMemoryManager"),
    "BitNetKernelManager":           # alias for engine.kernels.manager
        (".bitnet_kernels", "BitNetKernelManager"),

    # ── C++ backend (all optional — gracefully absent if not built) ──────────
    "BitNetCppBackend":
        (".bitnet_cpp_backend", "BitNetCppBackend"),
    "BitNetCppChatBackend":          # v9 NEW: chat-formatted C++ wrapper
        (".bitnet_cpp_backend", "BitNetCppChatBackend"),
    "BitNetRunInferenceBackend":     # v9 NEW: run_inference.py subprocess backend
        (".bitnet_cpp_backend", "BitNetRunInferenceBackend"),
    "BitNetCppConfig":
        (".bitnet_cpp_backend", "BitNetCppConfig"),
    "BackendResult":                 # v9 NEW: typed factory result
        (".bitnet_cpp_backend", "BackendResult"),
    "create_best_backend":
        (".bitnet_cpp_backend", "create_best_backend"),

    # ── Compatibility layer (LoadResult for type-safe loading) ───────────────
    "LoadResult":                    # v9 NEW: typed load result from transformers_compat
        ("...compatibility.transformers_compat", "LoadResult"),

    # ── State-aware router + utilities ───────────────────────────────────────
    # NOTE: auto_detect_backend comes from bitnet_loader (v9) not model_unified_env.
    # The bitnet_loader version checks: state file → BitLinear → standard transformers.
    "auto_detect_backend":           # v9 FIX: from bitnet_loader, not model_unified_env
        (".bitnet_loader", "auto_detect_backend"),

    "BitNetState":
        ("..adaptive_awareness.model_unified_env", "BitNetState"),
    "get_bitnet_state":
        ("..adaptive_awareness.model_unified_env", "get_bitnet_state"),
    "AdaptiveModelRouter":
        ("..adaptive_awareness.model_unified_env", "AdaptiveModelRouter"),
    "InferenceBackend":
        ("..adaptive_awareness.model_unified_env", "InferenceBackend"),
    "GGUFModelType":
        ("..adaptive_awareness.model_unified_env", "GGUFModelType"),
    "classify_gguf":
        ("..adaptive_awareness.model_unified_env", "classify_gguf"),
    "route_model":
        ("..adaptive_awareness.model_unified_env", "route_model"),
    "validate_environment":
        ("..adaptive_awareness.model_unified_env", "validate_environment"),
    "EnvironmentStatus":
        ("..adaptive_awareness.model_unified_env", "EnvironmentStatus"),
}

# Attributes that may be absent in older installs — import failures are
# logged as warnings rather than hard errors.
_OPTIONAL_IMPORTS = frozenset({
    "BitNetKernelManager",
    "BitNetMemoryManager",
    "BitNetCppBackend",
    "BitNetCppChatBackend",
    "BitNetRunInferenceBackend",
    "BitNetCppConfig",
    "BackendResult",
    "create_best_backend",
    "LoadResult",
    "get_router",
})


# =============================================================================
# Lazy __getattr__ with proper relative-module resolution
# =============================================================================
def __getattr__(name: str) -> Any:
    """
    Resolve any registered lazy import on first access.
    Caches the result in globals() for O(1) subsequent access.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(
            f"module 'local_llm_core.engine.bitnet' has no attribute {name!r}.\n"
            f"Available: {sorted(_LAZY_IMPORTS.keys())}"
        )

    module_path, attr_name = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, __package__)
        attr   = getattr(module, attr_name)
        # Cache in module globals — avoids repeated importlib overhead
        globals()[name] = attr
        return attr

    except ImportError as exc:
        if name in _OPTIONAL_IMPORTS:
            log.warning(
                "Optional BitNet component '%s' not available: %s",
                name, exc,
            )
            raise AttributeError(
                f"Optional component '{name}' is not available in this install. "
                f"Run 'local-llm-setup' to build BitNet.cpp binaries."
            ) from exc
        log.exception(
            "Failed to lazy-import '%s' from '%s'", name, module_path
        )
        raise ImportError(
            f"Could not import '{name}' from '{module_path}'.\n"
            f"Original error: {exc}\n"
            f"Tip: Run 'local-llm-setup' to ensure all dependencies are installed."
        ) from exc

    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_path}' has no attribute '{attr_name}'.\n"
            f"The module loaded correctly but the symbol is missing — "
            f"this may indicate a version mismatch. Original error: {exc}"
        ) from exc


# =============================================================================
# __dir__ — IDE / tab-completion support
# =============================================================================
def __dir__() -> List[str]:
    return sorted(list(_LAZY_IMPORTS.keys()) + [
        "__version__", "initialize", "get_status",
    ])


# =============================================================================
# initialize() — eager pre-load for production servers
# =============================================================================
def initialize(
    skip_optional: bool = True,
    apply_spectre: bool = True,
) -> Dict[str, bool]:
    """
    Eagerly load all registered BitNet components.

    Call this at server startup to:
      • Pre-warm all lazy imports (zero latency on first request)
      • Validate all components are importable
      • Apply SPECTRE PYTHONPATH before any inference

    Args:
        skip_optional:  If True (default), silently skip optional components
                        that require bitnet.cpp to be built (C++ backends).
        apply_spectre:  If True (default), apply SPECTRE PYTHONPATH fix.

    Returns:
        Dict mapping component name → True (loaded) / False (failed)
    """
    if apply_spectre:
        try:
            from ...compatibility.transformers_compat import StateFileReader
            StateFileReader.apply_spectre_pythonpath()
        except Exception as e:
            log.debug("SPECTRE apply failed in initialize(): %s", e)

    results: Dict[str, bool] = {}
    for name in sorted(_LAZY_IMPORTS.keys()):
        try:
            getattr(__import__(__name__), name)
            results[name] = True
        except (ImportError, AttributeError) as exc:
            if skip_optional and name in _OPTIONAL_IMPORTS:
                log.debug("Optional component '%s' skipped: %s", name, exc)
                results[name] = False
            else:
                log.warning("Failed to load '%s': %s", name, exc)
                results[name] = False

    loaded   = sum(v for v in results.values())
    total    = len(results)
    optional = sum(1 for k, v in results.items() if not v and k in _OPTIONAL_IMPORTS)
    required_failed = sum(
        1 for k, v in results.items()
        if not v and k not in _OPTIONAL_IMPORTS
    )

    if required_failed:
        log.error(
            "BitNet initialize: %d/%d components loaded "
            "(%d optional unavailable, %d REQUIRED FAILED)",
            loaded, total, optional, required_failed,
        )
    else:
        log.info(
            "BitNet initialize: %d/%d components loaded "
            "(%d optional unavailable — run local-llm-setup to build C++ backends)",
            loaded, total, optional,
        )
    return results


# =============================================================================
# get_status() — module-level diagnostic for health endpoints
# =============================================================================
def get_status() -> Dict[str, Any]:
    """
    Return a health/diagnostic dict for this BitNet package.
    Safe to call at any time — never raises.

    Includes:
      • version and backend availability
      • bitnet_state.json summary (gguf_path, verify_passed, etc.)
      • SPECTRE mode
      • auto_detect_backend() result
      • Which lazy imports are cached vs not yet loaded
    """
    status: Dict[str, Any] = {
        "version": __version__,
        "cached_imports": sorted(
            k for k in _LAZY_IMPORTS if k in globals()
        ),
        "uncached_imports": sorted(
            k for k in _LAZY_IMPORTS if k not in globals()
        ),
    }

    # auto_detect_backend
    try:
        from .bitnet_loader import auto_detect_backend as _adb
        status["auto_detect_backend"] = _adb()
    except Exception as e:
        status["auto_detect_backend"] = f"error: {e}"

    # State file summary
    try:
        from ...compatibility.transformers_compat import StateFileReader
        state_data = StateFileReader.load() or {}
        status["state_file"] = {
            "loaded":        bool(state_data),
            "verify_passed": state_data.get("verify_passed"),
            "gguf_path":     state_data.get("gguf_path"),
            "model_name":    state_data.get("model_name"),
            "prism_class":   state_data.get("prism_class"),
            "spectre_mode":  state_data.get("spectre_mode"),
            "atlas_threads": state_data.get("atlas_threads"),
        }
    except Exception as e:
        status["state_file"] = {"error": str(e)}

    # BitLinear availability
    try:
        from ...compatibility.transformers_compat import BitLinearCompatibilityLayer
        status["bitlinear_available"] = BitLinearCompatibilityLayer.is_bitlinear_importable()
        status["bitlinear_backend"]   = BitLinearCompatibilityLayer.get_backend_path()
    except Exception as e:
        status["bitlinear_available"] = None
        status["bitlinear_backend"]   = f"error: {e}"

    # C++ backend availability
    try:
        from .bitnet_cpp_backend import is_bitnet_cpp_available, get_state  # type: ignore[attr-defined]
        status["bitnet_cpp_available"] = is_bitnet_cpp_available()
        cpp_state = get_state()
        if cpp_state:
            status["llama_cli"]    = cpp_state.get("llama_cli_binary")
            status["llama_server"] = cpp_state.get("llama_server_binary")
    except Exception:
        status["bitnet_cpp_available"] = False

    # GGUF TL1/TL2
    try:
        import gguf
        _ = gguf.GGMLQuantizationType.TL1
        _ = gguf.GGMLQuantizationType.TL2
        status["gguf_tl1_available"] = True
        status["gguf_version"] = getattr(gguf, "__version__", "?")
    except (ImportError, AttributeError):
        status["gguf_tl1_available"] = False

    return status


# =============================================================================
# Public API list (affects `from engine.bitnet import *`)
# =============================================================================
__all__: List[str] = sorted(
    list(_LAZY_IMPORTS.keys()) + ["initialize", "get_status", "__version__"]
)
