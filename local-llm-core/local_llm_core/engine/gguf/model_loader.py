from __future__ import annotations

# engine/gguf/model_loader.py — Production‑Ready Model Loader  v1.0.0
# ===================================================================
# v1.0.0 – Integrated with Unified BitNet + GGUF environment:
#   • BitNet GGUF guard – prevents accidental loading of I2_S / TL1 / TL2
#     models via llama‑cpp‑python (which would produce garbage).
#   • Adaptive routing – optionally delegates BitNet models to the
#     correct backend (bitnet.cpp server, subprocess, or transformers)
#     via AdaptiveModelRouter.
#   • Environment validation – checks transformers version compatibility
#     on init and exposes status for monitoring.
#   • All v1.0.0 features preserved (memory‑budget helpers, circuit breaker,
#     hardware‑adaptive config, async pipelines, etc.).
#   • Pre‑flight capacity check – rejects models that don't fit in
#     VRAM + RAM + swap with a clear message.

# ===================================================================

import asyncio
import gc
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Imports from unified environment module
# -----------------------------------------------------------------------------
try:
    from local_llm_core.engine.adaptive_awareness.model_unified_env import (
        classify_gguf,
        route_model,
        get_bitnet_state,
        validate_environment,
        GGUFModelType,
        InferenceBackend,
        GGUFHeader,
        BitNetState,
        EnvironmentStatus,
        AdaptiveModelRouter,
        UnifiedEnvironmentValidator,
        BitNetStateManager,
    )
    _UNIFIED_ENV_AVAILABLE = True
except ImportError:
    _UNIFIED_ENV_AVAILABLE = False
    log.warning("model_unified_env not available – BitNet guard disabled")

# -----------------------------------------------------------------------------
# BitNet.cpp backend (binary detection: state file + direct search)
# -----------------------------------------------------------------------------

try:
    from local_llm_core.engine.bitnet.bitnet_cpp_backend import (
        BitNetCliBackend,
        BitNetCppBackend,
        BitNetRunInferenceBackend,
        _find_binary as _bn_find_binary,
        _load_bitnet_state as _bn_load_state,
        _LLAMA_SERVER_NAMES,
        _LLAMA_CLI_NAMES,
    )
    _BITNET_CPP_BACKEND_AVAILABLE = True
except ImportError:
    _BITNET_CPP_BACKEND_AVAILABLE = False

# -----------------------------------------------------------------------------
# Exceptions (if not already imported)
# -----------------------------------------------------------------------------
try:
    from local_llm_core.engine.exceptions import ModelError
except ImportError:
    # fallback local definition
    class ModelError(Exception):
        pass

# -----------------------------------------------------------------------------
# Prometheus Metrics (optional)
# -----------------------------------------------------------------------------
def _setup_metrics():
    try:
        from prometheus_client import Counter, Gauge
        return {
            "load_total": Counter("llm_model_load_total", "Model load attempts", ["model_id", "status"]),
            "unload_total": Counter("llm_model_unload_total", "Model unloads", ["model_id"]),
            "loaded_models": Gauge("llm_loaded_models", "Number of currently loaded models"),
            "vram_used_mb": Gauge("llm_model_vram_used_mb", "VRAM used by loaded models"),
            "load_time_s": Gauge("llm_model_load_time_s", "Last model load time", ["model_id"]),
            "oom": Counter("llm_model_oom_total", "Out of memory events during load"),
        }
    except ImportError:
        return {}

_METRICS = _setup_metrics()

# -----------------------------------------------------------------------------
# Circuit Breaker for Model Loading
# -----------------------------------------------------------------------------
class _LoaderBreaker:
    """Circuit breaker for model loading with configurable thresholds."""

    def __init__(self, threshold: int = 3, recovery_s: float = 60.0) -> None:
        self._threshold = threshold
        self._recovery = recovery_s
        self._failures = 0
        self._opened = 0.0
        self._state = "closed"
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def can_attempt(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened >= self._recovery:
                    if not self._probe_in_flight:
                        self._state = "half-open"
                        self._probe_in_flight = True
                        log.info("ModelLoader breaker → HALF-OPEN")
                        return True
                return False
            return not self._probe_in_flight

    def record_success(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            self._failures = 0
            if self._state != "closed":
                self._state = "closed"
                log.info("ModelLoader breaker → CLOSED")

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            self._failures += 1
            if self._state in ("open", "half-open"):
                self._state = "open"
                self._opened = time.monotonic()
                log.error("Breaker OPEN (probe failed)")
                return
            if self._failures >= self._threshold:
                self._state = "open"
                self._opened = time.monotonic()
                log.error("Breaker OPEN (%d failures)", self._failures)

    def stats(self) -> dict:
        with self._lock:
            return {"state": self._state, "failures": self._failures, "probe_in_flight": self._probe_in_flight}

# -----------------------------------------------------------------------------
# Configuration Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model_path: str
    model_id: str
    quantization: str = "auto"
    context_length: int = 0          # 0 = use hardware profile default
    num_gpu_layers: int = -1         # -1 = auto
    rope_scaling: float = 1.0
    use_mmap: bool = True
    use_mlock: bool = False
    gpu_split: Optional[List[float]] = None
    n_threads: int = 0  # 0 = hardware profile default
    # New field: allow BitNet fallback (if True, BitNet GGUFs are routed to appropriate backend)
    allow_bitnet_fallback: bool = False
    # Per-model KV-cache type ("f16", "q8_0", "q4_0"); empty = the measured
    # type for this model when a benchmark exists, else the hardware profile's.
    kv_cache_quant: str = ""
    # Measured decode speed for this model (tokens/s); 0 = look it up.
    target_tps: float = 0.0


@dataclass
class LoadedModel:
    """Represents a loaded model with metadata (covers both GGUF and BitNet models)."""

    config: ModelConfig
    llm: Any                         # llama‑cpp model or BitNetInferenceWrapper
    loaded_at: float = field(default_factory=time.monotonic)
    total_layers: int = 0
    gpu_layers: int = 0
    cpu_layers: int = 0
    vram_used_mb: int = 0
    load_time_s: float = 0.0
    request_count: int = 0
    last_used: float = field(default_factory=time.monotonic)
    strategy: str = "unknown"
    backend: str = "unknown"
    nvme_advised: bool = False
    cpu_threads: int = 0
    profile_name: str = "unknown"
    pipeline_registered: bool = False
    # New fields for BitNet support
    model_type: str = "gguf"         # "gguf" or "bitnet"
    # Actual context length the model was loaded with, AFTER hardware-profile
    # capping. This is the value clients (CLI, Web UI) must use for history
    # budgeting / max_tokens — NOT the (possibly larger, uncapped) value from
    # the model's config.yaml entry. Using the uncapped config value causes
    # clients to send prompts/budgets that exceed the model's real n_ctx,
    # leading to silent truncation mid-response or empty generations once
    # the prompt alone exceeds the actual context window.
    loaded_context_length: int = 0
    backend_info: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Quantization Bits‑Per‑Weight Table
# -----------------------------------------------------------------------------
_BPW = {
    "Q2_K": 2.0, "Q3_K_S": 3.0, "Q3_K_M": 3.35, "Q4_0": 4.0,
    "Q4_K_S": 4.0, "Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q6_K": 6.0,
    "Q8_0": 8.0, "F16": 16.0,
}

def _select_quantization(params_b: float, vram_gb: float) -> str:
    """Select best quantization that fits in available VRAM."""
    overhead_gb = 0.3
    for quant, bpw in sorted(_BPW.items(), key=lambda x: -x[1]):
        req = params_b * 1e9 * bpw / (8 * 1024**3) + overhead_gb
        if req <= vram_gb * 0.95:
            return quant
    return "Q2_K"

# -----------------------------------------------------------------------------
# Main ModelLoader Class
# -----------------------------------------------------------------------------
class ModelLoader:
    """
    Full‑lifecycle model manager with hardware‑adaptive configuration and
    automatic backend detection for BitNet GGUF files.

    Parameters
    ----------
    vram_mgr : VRAMManager
    ram_mgr : RAMOptimizer
    swap_mgr : SwapHandler
    hybrid : HybridMemoryManager, optional
        Hybrid manager for pre‑flight capacity checks and tiered caching.
    gpu_infos : list of GPUInfo
    config : dict, optional
    """

    _sysctl_applied = False
    _sysctl_lock = threading.Lock()

    def __init__(
        self,
        vram_mgr: Any,
        ram_mgr: Any,
        swap_mgr: Any,
        gpu_infos: List[Any],
        config: Optional[Dict[str, Any]] = None,
        hybrid: Optional[Any] = None,
        system_info: Optional[Any] = None,
    ) -> None:
        self._vram = vram_mgr
        self._ram = ram_mgr
        self._swap = swap_mgr
        self._hybrid = hybrid
        self._gpus = gpu_infos
        self._cfg = config or {}
        self._system_info = system_info

        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._loading: set[str] = set()
        self._loading_events: Dict[str, asyncio.Event] = {}
        self._loading_lock: Optional[asyncio.Lock] = None
        # dedicated lock for hot_swap. Cannot reuse `_loading_lock`
        # — `hot_swap` calls `self.load()` which re-acquires `_loading_lock`,
        # and asyncio.Lock is not reentrant (deadlock).
        self._hot_swap_lock: Optional[asyncio.Lock] = None

        # Circuit breaker
        breaker_cfg = self._cfg.get("model_loader", {}).get("circuit_breaker", {})
        self._breaker = _LoaderBreaker(
            threshold=breaker_cfg.get("failure_threshold", 3),
            recovery_s=breaker_cfg.get("recovery_timeout_s", 60.0),
        )

        self._shutdown_event = threading.Event()

        # Security
        security_cfg = self._cfg.get("security", {})
        self._allowed_models = set(security_cfg.get("allowed_models", []))
        self._model_base_dir = security_cfg.get("model_base_dir")
        if self._model_base_dir:
            self._model_base_dir = Path(self._model_base_dir).resolve()

        # ── hardware profile resolution ───────────────────────────
        # Two modes:
        #   1. Standalone (parent process): `self._cfg` is the full
        #      config.yaml dict, `hardware_profiles.overrides` is present,
        #      run local `detect_profile()`.
        #   2. Worker subprocess: `self._cfg` is the parent's `_load_spec`,
        #      which does NOT include `hardware_profiles` — but DOES include
        #      `hardware_profile` (the parent's resolved profile with all
        #      operator overrides applied). Use it directly; running local
        #      detection here produces a profile WITHOUT the operator's
        #      overrides, triggering `worker/main.py`'s "profile drift
        #      detected" warning on every respawn.
        from local_llm_core.engine.hardware.hardware_profiles import (
            detect_profile, HardwareProfile,
        )

        _parent_profile_dict = self._cfg.get("hardware_profile")
        _used_parent_profile = False
        if _parent_profile_dict and isinstance(_parent_profile_dict, dict):
            try:
                self._profile = HardwareProfile(**_parent_profile_dict)
                _used_parent_profile = True
                log.info(
                    "ModelLoader: using parent-resolved profile '%s' "
                    "from _load_spec.hardware_profile (skipping local "
                    "detection — operator overrides already applied)",
                    self._profile.name,
                )
            except Exception as _exc:
                log.warning(
                    "ModelLoader: could not reconstruct parent "
                    "hardware_profile (%s) — falling back to local detection",
                    _exc,
                )

        if not _used_parent_profile:
            _hp_cfg = self._cfg.get("hardware_profiles", {}) or {}
            _hp_overrides = _hp_cfg.get("overrides", {}) or {}
            hw_config = {"profile_overrides": _hp_overrides}
            self._profile = detect_profile(
                gpu_infos,
                ram_total_mb=self._ram_total_mb(),
                hardware_config=hw_config,
                system_info=system_info,
            )
            # The profile before any measurement.  load() resolves the record
            # of the model actually being loaded starting from this one, so a
            # measurement made for one model is never applied to another.
            self._profile_base = self._profile
            # Single-model deployment: when exactly one model has been
            # measured, apply it at construction so `target_tps_source` reads
            # `bench_file` from the first log line.
            try:
                from local_llm_core.engine.hardware.hardware_profiles import (
                    single_bench_record, _apply_measured_tps,
                )
                _bd = single_bench_record()
                if _bd:
                    _bt = float(_bd.get("decode_tps", 0) or 0)
                    if _bt > 0:
                        _old = self._profile.target_tps_source
                        self._profile = _apply_measured_tps(
                            self._profile, _bt, "bench_file",
                        )
                        log.info(
                            "ModelLoader: bench file present at construction "
                            "(tps=%.2f) — applied to profile (source: %s → "
                            "bench_file)",
                            _bt, _old,
                        )
            except Exception:
                pass

        log.info(
            "ModelLoader profile=%s vram_budget=%dMB",
            self._profile.name,
            self._profile.vram_budget_mb(self._vram_total_mb()),
        )

        # ── CPUThrottler resolution ───────────────────────────────
        # Three sources, in priority order:
        #   1. `drive_pipeline.cpu_throttler.thread_limit` (explicit count)
        #   2. `drive_pipeline.cpu_throttler.thread_ratio` (fraction of cores)
        #   3. `profile.n_threads()` (profile's `cpu_thread_ratio` × cores)
        #
        # The previous code passed the full `cpu_throttler` dict to
        # `CPUThrottler`, which defaults to a 50 % ratio when no key is
        # present. On workers whose `_load_spec` does not include
        # `drive_pipeline`, this collapsed to 2 threads on a 4-core box
        # (parent side correctly used 4). The `if thread_limit <= 0:`
        # fallback was dead code — `CPUThrottler` always clamps to ≥ 1.
        # The priority above is now explicit and profile-driven when no
        # config is available.
        from local_llm_core.engine.pipelines.drive_pipeline import CPUThrottler
        _dp_cfg = self._cfg.get("drive_pipeline", {}) or {}
        _ct_cfg = _dp_cfg.get("cpu_throttler", {}) or {}
        _explicit_limit = int(_ct_cfg.get("thread_limit", 0) or 0)
        _has_ratio = "thread_ratio" in _ct_cfg

        if _explicit_limit > 0:
            # (1) Explicit count wins.
            self._cpu_throttler = CPUThrottler(thread_limit=_explicit_limit)
        elif _has_ratio:
            # (2) Explicit ratio: pass through, CPUThrottler computes.
            self._cpu_throttler = CPUThrottler(
                thread_limit=None,
                config={"cpu_throttler": _ct_cfg},
            )
        else:
            # (3) No config — derive from the resolved profile's
            # `cpu_thread_ratio`. This is the operator's intent when
            # config.yaml has no `drive_pipeline.cpu_throttler` section
            # (typical case for a worker subprocess on this deployment).
            self._cpu_throttler = CPUThrottler(
                thread_limit=self._profile.n_threads(),
            )
        self._cpu_throttler.set_openmp_threads()

        # Lifecycle manager hook (set externally)
        self._lifecycle_manager: Optional[Any] = None

        # ── AsyncWeightPipeline (madvise WILLNEED prefetch) ───────────
        # The prefetch thread estimates each layer's byte range from the
        # GGUF file size and calls madvise(WILLNEED) on that range while
        # llama.cpp is decoding. There is no signal from llama.cpp about
        # which layer is active, so the guess is right only by accident.
        # On a bandwidth-bound machine (shared-memory APU, single-channel
        # DDR4, any UMA laptop) the madvise hints compete with the decode
        # stream for exactly the resource decode is bound by — memory
        # bandwidth — and deliver no measurable page-fault reduction to
        # compensate. On this class of hardware the net effect is a
        # small but consistent decode slowdown.
        #
        # Default: disabled. Enable per-deployment only after benchmarking
        # a measured win. The switch lives under `model_loader` so a worker
        # subprocess (`_load_spec`) inherits the same setting as the
        # parent — previously each worker spawned its own pipeline thread,
        # multiplying the effect by the worker count.
        self._pipeline_manager = None
        _ml_cfg_init = (self._cfg.get("model_loader", {}) or {})
        _awp_enabled = bool(_ml_cfg_init.get("async_weight_pipeline_enabled", False))
        if _awp_enabled:
            try:
                from local_llm_core.engine.inference.async_weight_pipe import (
                    get_pipeline_manager,
                )
                self._pipeline_manager = get_pipeline_manager()
                log.info(
                    "AsyncWeightPipeline ENABLED (async_weight_pipeline_enabled=true) — "
                    "madvise(WILLNEED) prefetch will run against layer regions "
                    "estimated from file size. Verify a measured win on this hardware "
                    "before leaving enabled."
                )
            except Exception as exc:
                log.warning(
                    "AsyncWeightPipeline enabled but could not be loaded: %s — "
                    "continuing without prefetch", exc,
                )
        else:
            log.info(
                "AsyncWeightPipeline disabled (model_loader.async_weight_pipeline_enabled=false) "
                "— no madvise prefetch thread will run. The prefetch guesses layer "
                "byte ranges from file size with no feedback from llama.cpp, so on "
                "memory-bandwidth-bound hardware it competes with decode rather than "
                "helping it. See config.yaml's model_loader.async_weight_pipeline_enabled "
                "comment for how to benchmark a change."
            )

        # Total VRAM/RAM usage tracking (MB) for external queries
        self._total_vram_used_mb = 0
        self._total_ram_used_mb = 0

        self._update_loaded_models_metric()

        # ── Unified environment validation (non‑blocking) ────────────────
        if _UNIFIED_ENV_AVAILABLE:
            try:
                self._env_status = validate_environment(raise_on_error=False)
                if not self._env_status.overall_ok:
                    log.warning("Unified environment: not fully ready (%s)", self._env_status)
            except Exception as e:
                log.warning("Environment validation failed: %s", e)
        else:
            self._env_status = None

        # Lazy router for BitNet delegation
        self._router: Optional[AdaptiveModelRouter] = None

        # Warn if hybrid manager is missing
        if self._hybrid is None:
            log.warning("HybridMemoryManager not provided – pre‑flight capacity checks disabled")

    # -------------------------------------------------------------------------
    # Lifecycle hook
    # -------------------------------------------------------------------------
    def set_lifecycle_manager(self, manager) -> None:
        self._lifecycle_manager = manager

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def estimate_model_size(model_path: str) -> int:
        """
        Return an approximate RAM usage in MB for a GGUF model file.
        This is the file size plus a fixed 10% overhead for runtime structures
        (KV cache, etc.).  It is suitable for quick memory‑budget checks
        before attempting a load.
        """
        path = Path(model_path)
        if not path.exists():
            return 0
        file_mb = path.stat().st_size // (1024 * 1024)
        return int(file_mb * 1.1)

    @staticmethod
    def _estimate_model_size_mb(path: Path) -> float:
        """
        Estimate model size in MB for a GGUF file or HF directory.
        For GGUF, returns file size. For directory, sums safetensors/.bin files.
        """
        if not path.exists():
            return 0.0
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        # Directory: sum safetensors and .bin files
        total = 0
        for ext in (".safetensors", ".bin"):
            for f in path.glob(f"*{ext}"):
                total += f.stat().st_size
        return total / (1024 * 1024)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _vram_total_mb(self) -> int:
        return sum(g.vram_total_mb for g in self._gpus if g.supported) if self._gpus else 0

    @staticmethod
    def _ram_total_mb() -> int:
        try:
            return psutil.virtual_memory().total >> 20
        except Exception:
            return 8192

    def _update_loaded_models_metric(self) -> None:
        if _METRICS.get("loaded_models"):
            _METRICS["loaded_models"].set(len(self._models))
        if _METRICS.get("vram_used_mb"):
            total_vram = sum(m.vram_used_mb for m in self._models.values())
            _METRICS["vram_used_mb"].set(total_vram)

    # -------------------------------------------------------------------------
    # Security: Path Validation
    # -------------------------------------------------------------------------
    def _validate_model_config(self, mc: ModelConfig) -> None:
        """Prevent path traversal, symlink attacks, and unauthorized models."""
        if self._allowed_models and mc.model_id not in self._allowed_models:
            raise ValueError(f"Model ID '{mc.model_id}' not in allowed_models. Allowed: {list(self._allowed_models)}")

        path = Path(mc.model_path)

        if path.is_symlink():
            raise ValueError(f"Model path is a symlink (security risk): {path}. Use the actual file path.")

        resolved_path = path.resolve()
        if not resolved_path.is_file() and not resolved_path.is_dir():
            # GGUF files are files; BitNet directories are directories
            raise FileNotFoundError(f"Model path does not exist or is not a regular file/directory: {resolved_path}")

        if self._model_base_dir:
            base_resolved = self._model_base_dir.resolve()
            try:
                resolved_path.relative_to(base_resolved)
            except ValueError:
                raise ValueError(f"Model path '{resolved_path}' not inside base directory '{base_resolved}'")

        if self._model_base_dir and self._model_base_dir.is_symlink():
            log.warning("Model base directory is a symlink; this may be intentional")

        if not os.access(resolved_path, os.R_OK):
            raise PermissionError(f"Model file not readable: {resolved_path}")

    # -------------------------------------------------------------------------
    # Async Lock Lazy Initializer
    # -------------------------------------------------------------------------
    def _get_loading_lock(self) -> asyncio.Lock:
        if self._loading_lock is None:
            self._loading_lock = asyncio.Lock()
        return self._loading_lock

    # -------------------------------------------------------------------------
    # BitNet.cpp backend for GGUF files (i2_s/tl1/tl2) — direct, no ceremony
    # -------------------------------------------------------------------------
    def _bn_persist_discovery(
        self,
        server_bin: Optional[Path] = None,
        cli_bin: Optional[Path] = None,
    ) -> None:
        """
        ("auto-adapt / auto-fix"): once a BitNet.cpp binary
        is found via ANY discovery tier (PATH, default search list, etc.),
        write it into bitnet_state.json. The next restart then hits
        Priority-0 of _find_binary()'s own priority order immediately and
        deterministically, instead of re-running the same environment-
        dependent search and risking the same failure mode again if HOME/
        PATH ever gets misconfigured a second time.

        Best-effort and self-contained: any failure here is logged at
        debug level only and never blocks the model load that already
        succeeded. Uses a write-to-temp-then-atomic-rename pattern so a
        crash mid-write can't corrupt an existing valid state file.
        """
        try:
            from local_llm_core.engine.bitnet.bitnet_cpp_backend import _get_state_file_path
            state_path = _get_state_file_path()
            existing: Dict[str, Any] = {}
            if state_path.is_file():
                try:
                    loaded = json.loads(state_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except Exception:
                    existing = {}  # corrupt/unreadable — start fresh rather than fail
            if server_bin is not None:
                existing["llama_server_binary"] = str(server_bin)
            if cli_bin is not None:
                existing["llama_cli_binary"] = str(cli_bin)
            existing["verify_passed"] = True
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = state_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            tmp_path.replace(state_path)  # atomic on POSIX
            log.info("Persisted BitNet.cpp binary discovery to %s", state_path)
        except Exception as exc:
            log.warning("Could not persist BitNet.cpp binary discovery: %s", exc)

    def _try_bitnet_cpp_backend(self, mc: ModelConfig, ctx_len: int) -> Optional[LoadedModel]:
        """
        For a BitNet-quantized GGUF file, try to use a BUILT BitNet.cpp
        binary directly — no state file dependency required, no router, no
        SPECTRE/PRISM/FORGE/ATLAS.

        Tier order (PRODUCTION FIX — reordered): direct llama-cli
        (one process per request) is now Tier 0, the primary path.
        run_inference.py is Tier 1. llama-server (a persistent daemon) is
        now Tier 2, the LAST resort.

        This is a deliberate reversal of the original order (which had
        llama-server first). Rationale, confirmed across repeated
        real-world testing of this exact deployment: llama-server shares
        one mutable generation loop across every request (more acute here
        under --parallel 1, a single shared slot) — when that loop gets
        stuck, a real, currently-open class of upstream llama.cpp bug
        (server freezes while the OS process stays alive, particularly on
        CPU-only hardware), every subsequent request blocks until
        something notices and restarts it. In practice, every direct
        llama-cli invocation succeeded cleanly and predictably, while
        llama-server was the one component that repeatedly stalled. A
        one-process-per-request architecture has zero shared mutable
        state to get stuck in — a hung process here affects exactly one
        request and is simply killed.

        Uses the EXACT configured `mc.model_path` rather than searching a
        directory for "a" GGUF file — the model directory may contain
        other, unrelated GGUF models (e.g. Qwen3-8B-Q4_K_M.gguf alongside
        ggml-model-i2_s.gguf), and a directory search could silently pick
        the wrong one.

        Returns None (not an error) if no built binary of any kind is
        found — the caller falls through to attempting the standard GGUF
        loader, which will either work (if the installed llama-cpp-python
        happens to support the quant type) or fail with the clear,
        already-classified error in layer_streamer.py.
        """
        if not _BITNET_CPP_BACKEND_AVAILABLE:
            return None

        gguf_path = Path(mc.model_path)
        n_threads = max(1, (os.cpu_count() or 4) // 2)

        # load the state file once and pass it into every
        # lookup below, so Priority-0 of _find_binary()'s own documented
        # order is actually exercised for all three tiers (previously
        # called with neither state= nor extra_paths=, silently skipping
        # straight to the weaker PATH/default-list tiers every time).
        _bn_state = _bn_load_state()

        # ── Tier 0: direct llama-cli, one process per request (PRIMARY) ────
        cli_bin = _bn_find_binary(_LLAMA_CLI_NAMES, state=_bn_state)
        if cli_bin is not None:
            log.info(
                "Load [%s]: BitNet.cpp llama-cli found at %s — using "
                "directly (process-per-request, no persistent server)",
                mc.model_id, cli_bin,
            )
            self._bn_persist_discovery(cli_bin=cli_bin)
            bitnet_dir = cli_bin.parent.parent.parent
            backend = BitNetCliBackend(
                cli_bin, gguf_path,
                n_threads=n_threads, n_ctx=ctx_len,
                bitnet_dir=bitnet_dir,
            )
            return LoadedModel(
                config=mc, llm=backend,
                total_layers=0, gpu_layers=0, cpu_layers=0,
                vram_used_mb=0, load_time_s=0.0,
                strategy="bitnet_cli_direct", backend="bitnet_cli_direct",
                model_type="bitnet", loaded_context_length=ctx_len,
            )

        # ── Tier 1: run_inference.py (thin wrapper around the same llama-cli
        #            binary, kept as a fallback for the case where
        #            _find_binary's PATH/default-search-list discovery
        #            missed the llama-cli binary directly, but the BitNet.cpp
        #            repo root — which run_inference.py resolves its own
        #            llama-cli path from internally — is still locatable) ──
        # previously re-derived the repo root from `cli_bin`,
        # which is impossible to reach as non-None here (Tier 0 above
        # already returns whenever cli_bin is found) — that made this
        # entire tier dead code. auto_detect() performs its OWN
        # independent search (state file's bitnet_dir field, then default
        # candidate paths) rather than depending on Tier 0's already-failed
        # binary search, so it can genuinely succeed in cases Tier 0 can't.
        run_inference_backend = BitNetRunInferenceBackend.auto_detect(
            str(gguf_path.parent), n_threads=n_threads, n_ctx=ctx_len,
        )
        if run_inference_backend is not None:
            log.info(
                "Load [%s]: no llama-cli binary found directly, but "
                "run_inference.py was located at %s — using it",
                mc.model_id, run_inference_backend.repo_dir,
            )
            return LoadedModel(
                config=mc, llm=run_inference_backend,
                total_layers=0, gpu_layers=0, cpu_layers=0,
                vram_used_mb=0, load_time_s=0.0,
                strategy="bitnet_run_inference", backend="bitnet_run_inference",
                model_type="bitnet", loaded_context_length=ctx_len,
            )

        # ── Tier 2 (LAST RESORT): llama-server, persistent HTTP daemon ─────
        # demoted from Tier 1 (was checked first) to last.
        # See method docstring for the full rationale. Only reached if
        # neither a direct llama-cli nor a usable run_inference.py install
        # could be found at all.
        server_bin = _bn_find_binary(_LLAMA_SERVER_NAMES, state=_bn_state)
        if server_bin is not None:
            log.info(
                "Load [%s]: no llama-cli found; falling back to "
                "llama-server (persistent daemon, last resort) at %s",
                mc.model_id, server_bin,
            )
            self._bn_persist_discovery(server_bin=server_bin)
            backend = BitNetCppBackend(
                server_bin, gguf_path,
                n_threads=n_threads, n_ctx=ctx_len,
            )
            return LoadedModel(
                config=mc, llm=backend,
                total_layers=0, gpu_layers=0, cpu_layers=0,
                vram_used_mb=0, load_time_s=0.0,
                strategy="bitnet_cpp_server", backend="bitnet_cpp_server",
                model_type="bitnet", loaded_context_length=ctx_len,
            )

        # ── Nothing built yet — one clear, accurate instruction ─────────────
        log.warning(
            "Load [%s]: '%s' is a BitNet-quantized GGUF (i2_s/tl1/tl2), but no "
            "built BitNet.cpp binary (llama-cli or llama-server) was found. "
            "Standard llama-cpp-python cannot read this quantization format — "
            "the kernels live in Microsoft's BitNet.cpp fork, not upstream "
            "llama.cpp. Build it once (~2 min): clone "
            "https://github.com/microsoft/BitNet, then inside that directory run: "
            "  python setup_env.py --hf-repo microsoft/bitnet-b1.58-2B-4T-gguf -q i2_s\n"
            "Falling back to attempt standard loading now (will likely fail cleanly).",
            mc.model_id, gguf_path.name,
        )
        return None

    # -------------------------------------------------------------------------
    # BitNet delegation (for adaptive loading)
    # -------------------------------------------------------------------------
    def _load_bitnet_model(self, mc: ModelConfig) -> LoadedModel:
        """Load a BitNet model using the unified router and BitNetInferenceWrapper."""
        if not _UNIFIED_ENV_AVAILABLE:
            raise ModelError(
                "model_unified_env not available, cannot load BitNet model. "
                "Install local-llm-core with full dependencies."
            )
        # Lazy initialize router
        if self._router is None:
            self._router = AdaptiveModelRouter()

        header, backend, params = self._router.route(mc.model_path)
        if backend == InferenceBackend.BITNET_TRANSFORMERS:
            # transformers fallback – use the BitNetInferenceWrapper directly
            # We need to create a wrapper that loads the HF model.
            try:
                from local_llm_core.engine.bitnet.bitnet_engine import BitNetInferenceWrapper
                wrapper = BitNetInferenceWrapper(
                    model_path=mc.model_path,
                    device=params.get("device", "cuda"),
                    force_backend="transformers",
                    config=getattr(mc, 'config', None),  # may be missing
                )
                wrapper.estimate_tokens("warm")  # force tokenizer load
            except Exception as e:
                raise ModelError(f"Failed to load BitNet model with transformers: {e}") from e

            loaded = LoadedModel(
                config=mc,
                llm=wrapper,
                total_layers=0,
                gpu_layers=0,
                cpu_layers=0,
                vram_used_mb=wrapper.get_memory_usage().get("vram_mb", 0),
                load_time_s=0.0,
                strategy="bitnet_transformers",
                backend="transformers",
                model_type="bitnet",
                backend_info=wrapper.backend_info(),
                loaded_context_length=mc.context_length or 4096,
            )
        else:
            # BitNet.cpp backend (server or subprocess) – handled by the same wrapper
            try:
                from local_llm_core.engine.bitnet.bitnet_engine import BitNetInferenceWrapper
                wrapper = BitNetInferenceWrapper(
                    model_path=mc.model_path,
                    force_backend=None,  # auto (use router params)
                    bitnet_cpp_port=params.get("port", 18080),
                )
            except Exception as e:
                raise ModelError(f"Failed to start BitNetCpp backend: {e}") from e

            loaded = LoadedModel(
                config=mc,
                llm=wrapper,
                total_layers=0,
                gpu_layers=0,
                cpu_layers=0,
                vram_used_mb=0,   # VRAM tracked by C++ server
                load_time_s=0.0,
                strategy=backend.value,
                backend=backend.value,
                model_type="bitnet",
                backend_info=wrapper.backend_info(),
                loaded_context_length=mc.context_length or 4096,
            )
        return loaded

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def _profile_for(self, mc: ModelConfig) -> "HardwareProfile":
        """Hardware profile specialised for one model.

        Order of precedence for the KV-cache type and the decode speed:
        values carried by ``mc`` (set from the computed model profile), then
        the model's own benchmark record, then the machine profile.
        """
        base = getattr(self, "_profile_base", None) or self._profile
        prof = base
        try:
            from local_llm_core.engine.hardware.hardware_profiles import apply_bench_for_model

            prof, _rec = apply_bench_for_model(base, mc.model_path)
        except Exception as exc:
            log.debug("bench lookup for %s skipped: %s", mc.model_id, exc)
        overrides: Dict[str, Any] = {}
        kv = str(getattr(mc, "kv_cache_quant", "") or "").lower()
        if kv in ("f16", "q8_0", "q4_0") and kv != prof.kv_cache_quant:
            overrides["kv_cache_quant"] = kv
        tps = float(getattr(mc, "target_tps", 0.0) or 0.0)
        if tps > 0:
            overrides["target_tps"] = tps
            overrides["target_tps_source"] = "bench_file"
        return prof.apply_overrides(overrides) if overrides else prof

    async def load(self, mc: ModelConfig) -> LoadedModel:
        """
        Load a model asynchronously.

        Routing is decided by path type: a .gguf file always uses the
        standard GGUF loader below (identical to any other model); a
        directory is treated as an HF transformers checkpoint and routed
        to the BitNet transformers backend.

        Returns
        -------
        LoadedModel
        """
        start_time = time.monotonic()
        load_id = str(uuid.uuid4())[:8]


        log.info("Load [%s]: model_id='%s' path='%s'", load_id, mc.model_id, mc.model_path)

        # ── Per-model measured speed and KV-cache type ────────────────
        # The profile built in __init__ describes the machine.  What a model
        # actually achieves, and which KV-cache type is fastest for it on
        # this back-end, is a property of the model, so it is resolved here
        # from the model's own benchmark record (or from the values the
        # server passed in `mc`).  The shared profile is never modified:
        # several models can be loaded one after another, or concurrently,
        # without carrying each other's numbers.
        try:
            _lp = self._profile_for(mc)
            log.info(
                "ModelLoader: '%s' target_tps=%.2f [%s] KV=%s",
                mc.model_id, _lp.target_tps, _lp.target_tps_source, _lp.kv_cache_quant,
            )
        except Exception as _bench_exc:
            log.warning(
                "ModelLoader: per-model profile for '%s' failed: %s "
                "(machine profile is used as is)", mc.model_id, _bench_exc,
            )

        # 1. Validation
        try:
            self._validate_model_config(mc)
        except Exception as e:
            log.warning("Load [%s]: validation failed for '%s': %s", load_id, mc.model_id, e)
            if _METRICS.get("load_total"):
                _METRICS["load_total"].labels(model_id=mc.model_id, status="validation_failed").inc()
            raise

        # ── 1a. Pre‑flight capacity check ──────────────────────────
        if self._hybrid is not None:
            path = Path(mc.model_path)
            if path.exists():
                model_size_mb = self._estimate_model_size_mb(path)
                # Get context length from config or profile
                ctx_len = mc.context_length or self._profile.context_length
                # For BitNet models (directories), we may not have a GGUF header; use profile target TPS
                _lp = self._profile_for(mc)
                target_tps = _lp.target_tps
                # Use this model's KV-cache type
                kv_quant = _lp.kv_cache_quant

                fit_est = None
                try:
                    fit_est = self._hybrid.estimate_model_fit(
                        model_path=mc.model_path,
                        model_size_mb=model_size_mb,
                        context_len=ctx_len,
                        target_tps=target_tps,
                        kv_quant=kv_quant,
                    )
                except Exception as e:
                    log.warning("Pre‑flight capacity check failed: %s – proceeding with load", e)
                    fit_est = None

                # NOTE (pre-flight dead-code fix): the block below used to sit
                # INSIDE the `except` clause above (one indent level deeper),
                # which produced two bugs:
                #   1. Happy path (estimate_model_fit returns a dict): the
                #      entire honest-ceiling adjustment AND the will_not_fit
                #      rejection were silently skipped — models that exceed
                #      available RAM+VRAM would attempt to load and fail with
                #      a raw llama.cpp OOM instead of the clean, actionable
                #      "Model X requires Y MB but only Z available" message.
                #   2. Exception path (estimate_model_fit raises): fit_est is
                #      set to None in the except block above, so the very next
                #      line would crash on fit_est["required_memory_mb"] with
                #      "TypeError: 'NoneType' object is not subscriptable".
                #
                # Dedenting to the level of `try:` above fixes both: the
                # block now runs whenever the estimator succeeded, and is
                # skipped cleanly when the estimator failed.
                if fit_est is not None:
                    try:
                        # ── Honest RAM+VRAM-only fit check ───────────────
                        _path_obj = Path(mc.model_path)
                        _model_mb = (
                            _path_obj.stat().st_size >> 20
                            if _path_obj.is_file()
                            else int(model_size_mb)
                        )
                        _vram_budget = self._profile.vram_budget_mb(
                            self._vram_total_mb()
                        )
                        _free_ram_mb = psutil.virtual_memory().available >> 20
                        _load_overhead_mb = int(self._cfg.get(
                            "model_loader", {}
                        ).get("preflight_load_overhead_mb", 1500))
                        _honest_fit_mb = max(
                            1, _vram_budget + _free_ram_mb - _load_overhead_mb,
                        )

                        # Thresholds (fraction of the honest RAM+VRAM ceiling):
                        #   ≤ 0.75  → comfortable (has genuine headroom)
                        #   ≤ 1.00  → slow        (tight; KV cache may churn)
                        #   ≤ 1.50  → very_slow   (will page from swap/SSD)
                        #   > 1.50  → leave as-is (the estimator's own
                        #                         will_not_fit is the honest
                        #                         hard limit — swap at least
                        #                         prevents an OOM kill)
                        if _model_mb <= _honest_fit_mb * 0.75:
                            _verdict = "comfortable"
                        elif _model_mb <= _honest_fit_mb:
                            _verdict = "slow"
                        elif _model_mb <= _honest_fit_mb * 1.50:
                            _verdict = "very_slow"
                        else:
                            _verdict = None

                        _order = {"comfortable": 0, "slow": 1,
                                  "very_slow": 2, "will_not_fit": 3}
                        _current = fit_est.get("status", "comfortable")

                        if (_verdict is not None
                                and _order[_verdict] > _order.get(_current, 0)):
                            _ratio = _honest_fit_mb / max(_model_mb, 1)
                            _new_tps = max(
                                0.3,
                                float(fit_est.get("estimated_tps", 8.0)) * _ratio,
                            )
                            log.warning(
                                "Pre‑flight '%s': DOWNGRADING '%s' → '%s' — "
                                "model=%.0f MB exceeds honest RAM+VRAM ceiling "
                                "%.0f MB (VRAM %d + free RAM %d − overhead %d); "
                                "expect mmap page-faults from swap/SSD per "
                                "token; est TPS %.2f → %.2f.",
                                mc.model_id, _current, _verdict,
                                _model_mb, _honest_fit_mb,
                                _vram_budget, _free_ram_mb, _load_overhead_mb,
                                float(fit_est.get("estimated_tps", 0)),
                                _new_tps,
                            )
                            fit_est["status"] = _verdict
                            fit_est["estimated_tps"] = _new_tps
                    except Exception as _xcheck_exc:
                        log.warning(
                            "Pre-flight honest-classification skipped for "
                            "'%s': %s", mc.model_id, _xcheck_exc,
                        )

                    log.info(
                        "Pre‑flight: model='%s' size=%.1fMB required=%.0fMB usable=%.0fMB est TPS=%.2f status=%s",
                        mc.model_id,
                        model_size_mb,
                        fit_est["required_memory_mb"],
                        fit_est["usable_memory_mb"],
                        fit_est["estimated_tps"],
                        fit_est["status"],
                    )
                    if fit_est["status"] == "will_not_fit":
                        raise RuntimeError(
                            f"Model '{mc.model_id}' ({model_size_mb:.1f} MB) exceeds available memory "
                            f"{fit_est['usable_memory_mb']} MB (VRAM+RAM+swap). "
                            f"Required: {fit_est['required_memory_mb']} MB. "
                            "Cannot load. Increase swap size (ssd_max_gb in config.yaml) or use a smaller model."
                        )
                    elif fit_est["status"] == "very_slow":
                        log.warning(
                            "Model '%s' is expected to run very slowly (~%.2f TPS). "
                            "Consider using a smaller model or increasing swap size.",
                            mc.model_id,
                            fit_est["estimated_tps"],
                        )
                    # For "slow" and "comfortable", we just log and continue.
        else:
            log.warning("Pre‑flight capacity check skipped (HybridMemoryManager not available)")

        # 2. BitNet routing — decided by PATH TYPE, not by GGUF quant detection.
        _bn_path = Path(mc.model_path)
        if _bn_path.is_dir():
            log.info("Load [%s]: HF transformer checkpoint directory detected – using transformers backend", load_id)
            loaded = self._load_bitnet_model(mc)
            loaded.load_time_s = time.monotonic() - start_time
            with self._lock:
                self._models[mc.model_id] = loaded
                loaded.request_count = 1
                loaded.last_used = time.monotonic()
            self._update_loaded_models_metric()
            if _METRICS.get("load_total"):
                _METRICS["load_total"].labels(model_id=mc.model_id, status="success").inc()
            log.info("BitNet model '%s' loaded via transformers backend", mc.model_id)
            return loaded

        # 3. Circuit breaker
        if not self._breaker.can_attempt():
            if _METRICS.get("load_total"):
                _METRICS["load_total"].labels(model_id=mc.model_id, status="circuit_open").inc()
            raise RuntimeError("Circuit breaker open – model loader temporarily unavailable")

        # 4. Concurrent load coordination
        my_evt = asyncio.Event()
        should_load = False
        wait_evt = None

        async with self._get_loading_lock():
            with self._lock:
                if mc.model_id in self._models:
                    existing = self._models[mc.model_id]
                    existing.request_count += 1
                    existing.last_used = time.monotonic()
                    if _METRICS.get("load_total"):
                        _METRICS["load_total"].labels(model_id=mc.model_id, status="cached").inc()
                    return existing

                if mc.model_id in self._loading:
                    wait_evt = self._loading_events.get(mc.model_id)
                    log.info("Load [%s]: waiting for concurrent load", load_id)
                else:
                    self._loading.add(mc.model_id)
                    self._loading_events[mc.model_id] = my_evt
                    should_load = True
                    log.info("Load [%s]: initiating load", load_id)

        # 5. Wait for concurrent load (if another coroutine is loading)
        if not should_load:
            load_timeout = self._cfg.get("model_loader", {}).get("load_timeout_s", 600.0)
            try:
                await asyncio.wait_for(wait_evt.wait(), timeout=load_timeout)
                with self._lock:
                    m = self._models.get(mc.model_id)
                if m is not None:
                    m.request_count += 1
                    m.last_used = time.monotonic()
                    return m
                raise RuntimeError(f"Concurrent load of '{mc.model_id}' failed")
            except asyncio.TimeoutError:
                raise RuntimeError(f"Timed out waiting for concurrent load of '{mc.model_id}'")

        # 6. VRAM budget calculation (standard GGUF only)
        try:
            vram_budget_mb = self._profile.vram_budget_mb(self._vram_total_mb())
        except Exception as e:
            raise RuntimeError(f"Failed to compute VRAM budget: {e}") from e

        _ml_cfg = self._cfg.get("model_loader", {})
        if not _ml_cfg and self._cfg.get("hardware_profile"):
            # worker mode with a thin `_load_spec`. The parent's
            # config.yaml carries tuned margins (50/100/50/100 on this
            # deployment), but `_load_spec` doesn't include the
            # `model_loader` section. Worker falls back to defaults
            # (500/300/100/100) which under-report available VRAM by
            # ~700 MB vs. the parent's intent. api_server
            # should add `model_loader: cfg.get("model_loader", {})`
            # to `_load_spec` to close this gap; the warning below
            # surfaces it in the meantime.
            log.warning(
                "ModelLoader: worker `_load_spec` is missing the "
                "`model_loader` config section — using defaults "
                "(safety=500, runtime=300, system_prompt=100 MB). "
                "The parent may have tighter margins. "
                "api_server should include `model_loader` in `_load_spec`."
            )
        safety_margin_mb = _ml_cfg.get("vram_safety_margin_mb", 500)
        runtime_overhead_mb = _ml_cfg.get("vram_runtime_overhead_mb", 300)
        system_prompt_overhead_mb = _ml_cfg.get("vram_system_prompt_overhead_mb", 100)
        min_vram = _ml_cfg.get("min_vram_budget_mb", 100)

        with self._lock:
            already_used_vram = sum(m.vram_used_mb for m in self._models.values() if m.model_type == "gguf")
            total_overhead = safety_margin_mb + runtime_overhead_mb + system_prompt_overhead_mb + already_used_vram
            vram_available = vram_budget_mb - total_overhead

            if vram_available < min_vram:
                raise RuntimeError(
                    f"Insufficient VRAM: available {vram_available}MB < minimum {min_vram}MB. "
                    f"Unload other models or adjust memory settings."
                )

        # 7. Perform load (in executor)
        model = None
        try:
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(None, self._load_sync, mc, vram_available)
            self._breaker.record_success()
        except MemoryError as me:
            self._breaker.record_failure()
            if _METRICS.get("oom"):
                _METRICS["oom"].inc()
            raise RuntimeError(f"Out of memory loading '{mc.model_id}': {me}") from me
        except SystemExit as se:
            self._breaker.record_failure()
            raise RuntimeError(f"llama_cpp exited during load of '{mc.model_id}': {se}") from se
        except Exception:
            self._breaker.record_failure()
            raise
        finally:
            with self._lock:
                self._loading.discard(mc.model_id)
                evt = self._loading_events.pop(mc.model_id, None)
                if evt and not evt.is_set():
                    try:
                        evt.set()
                    except RuntimeError:
                        pass

        # 8. Register model
        with self._lock:
            if mc.model_id in self._models:
                # another thread loaded it in the meantime
                existing = self._models[mc.model_id]
                existing.request_count += 1
                existing.last_used = time.monotonic()
                # clean up our duplicate
                if hasattr(model.llm, "close"):
                    model.llm.close()
                return existing
            self._models[mc.model_id] = model
            model.request_count = 1
            model.last_used = time.monotonic()
            self._total_vram_used_mb += model.vram_used_mb
            self._total_ram_used_mb += (model.cpu_layers * 100)  # rough estimate

        self._update_loaded_models_metric()

        # 9. Auto‑start AsyncWeightPipeline (only for GGUF models)
        if model.model_type == "gguf" and self._pipeline_manager:
            try:
                self._pipeline_manager.register(
                    model_id=mc.model_id,
                    model_path=mc.model_path,
                )
                model.pipeline_registered = True
                log.info("AsyncWeightPipeline started for %s", mc.model_id)
            except Exception as e:
                log.warning("Failed to register pipeline for %s: %s", mc.model_id, e)

        # 10. Notify lifecycle manager
        if self._lifecycle_manager:
            try:
                self._lifecycle_manager.on_model_loaded(mc.model_id, model)
            except Exception as e:
                log.warning("Lifecycle manager error: %s", e)

        elapsed_s = time.monotonic() - start_time
        log.info("Model '%s' loaded in %.2fs | gpu=%d/%d", mc.model_id, elapsed_s, model.gpu_layers, model.total_layers)
        if _METRICS.get("load_total"):
            _METRICS["load_total"].labels(model_id=mc.model_id, status="success").inc()
        if _METRICS.get("load_time_s"):
            _METRICS["load_time_s"].labels(model_id=mc.model_id).set(elapsed_s)

        return model

    async def unload(self, model_id: str) -> bool:
        with self._lock:
            model = self._models.pop(model_id, None)
        if not model:
            return False

        loop = asyncio.get_running_loop()
        if model.model_type == "bitnet":
            # BitNet wrapper has async shutdown
            if hasattr(model.llm, "shutdown"):
                await model.llm.shutdown()
        else:
            await loop.run_in_executor(None, self._unload_sync, model)

        # Stop pipeline
        if self._pipeline_manager and model.pipeline_registered:
            self._pipeline_manager.unregister(model_id)

        # Notify lifecycle
        if self._lifecycle_manager:
            self._lifecycle_manager.on_model_unloaded(model_id)

        if _METRICS.get("unload_total"):
            _METRICS["unload_total"].labels(model_id=model_id).inc()

        with self._lock:
            if model.model_type == "gguf":
                self._total_vram_used_mb = max(0, self._total_vram_used_mb - model.vram_used_mb)
                self._total_ram_used_mb = max(0, self._total_ram_used_mb - (model.cpu_layers * 100))

        self._update_loaded_models_metric()
        return True

    def _get_hot_swap_lock(self) -> asyncio.Lock:
        """dedicated lock for hot_swap. Cannot reuse `_loading_lock`
        — `hot_swap` calls `self.load()` internally, and `load()` re-acquires
        `_loading_lock`. asyncio.Lock is not reentrant, so the previous
        `async with self._get_loading_lock(): await self.load(...)`
        deadlocked on every call. This lock serialises concurrent
        hot-swaps without touching the loading lock.
        """
        if self._hot_swap_lock is None:
            self._hot_swap_lock = asyncio.Lock()
        return self._hot_swap_lock

    async def hot_swap(self, old_id: str, new_mc: ModelConfig) -> LoadedModel:
        """Atomic hot-swap: load new, then unload old.

        uses `_get_hot_swap_lock()` (dedicated), not
        `_get_loading_lock()`. The previous implementation deadlocked:
        `async with _get_loading_lock():` then `await self.load(new_mc)`
        which re-acquires `_get_loading_lock` inside `load()` — a
        non-reentrant asyncio.Lock. Concurrent hot-swaps are serialised
        by the new lock; `load()` and `unload()` continue to guard their
        own state as before.
        """
        async with self._get_hot_swap_lock():
            old = self.get(old_id)
            if not old:
                raise ValueError(f"Model '{old_id}' not loaded")
            try:
                new_model = await self.load(new_mc)
            except Exception:
                log.error(
                    "hot_swap: load of '%s' failed, keeping '%s'",
                    new_mc.model_id, old_id,
                )
                raise
            unloaded = await self.unload(old_id)
            if not unloaded:
                log.warning(
                    "hot_swap: unload of '%s' failed – both models may be loaded",
                    old_id,
                )
            return new_model

    def get(self, model_id: str) -> Optional[LoadedModel]:
        with self._lock:
            m = self._models.get(model_id)
            if m:
                m.last_used = time.monotonic()
                m.request_count += 1
            return m

    def list_models(self) -> List[Dict[str, Any]]:
        """Return metadata for all currently‑loaded models."""
        with self._lock:
            result = []
            for m in self._models.values():
                entry = {
                    "id": m.config.model_id,
                    "path": m.config.model_path,
                    "quantization": m.config.quantization if m.model_type == "gguf" else "bitnet",
                    "gpu_layers": m.gpu_layers,
                    "cpu_layers": m.cpu_layers,
                    "vram_mb": m.vram_used_mb,
                    "requests": m.request_count,
                    "load_time_s": round(m.load_time_s, 2),
                    "age_s": round(time.monotonic() - m.loaded_at),
                    "strategy": m.strategy,
                    "backend": m.backend,
                    "nvme_advised": m.nvme_advised,
                    "cpu_threads": m.cpu_threads,
                    "profile": m.profile_name,
                    "model_type": m.model_type,
                    "status": "ready",   # loaded models are ready
                    # Actual context the model was loaded with, AFTER hardware
                    # profile capping. Clients MUST use this for max_tokens /
                    # history-budget decisions, not the model's raw config
                    # value, which may be larger than what the loaded n_ctx
                    # actually supports.
                    "context_length": m.loaded_context_length,
                }
                if m.model_type == "bitnet":
                    entry.update(m.backend_info)
                result.append(entry)
            return result

    def model_memory_usage(self, model_id: str) -> dict:
        with self._lock:
            m = self._models.get(model_id)
            if not m:
                return {}
            usage = {
                "vram_mb": m.vram_used_mb,
                "ram_mb": self._ram.safe_allocation_bytes() // (1024 * 1024) if self._ram else 0,
                "strategy": m.strategy,
            }
            if m.model_type == "bitnet":
                # Add BitNet wrapper's own memory report
                wrapper_usage = m.llm.get_memory_usage() if hasattr(m.llm, "get_memory_usage") else {}
                usage.update(wrapper_usage)
            return usage

    def loader_breaker_stats(self) -> dict:
        return self._breaker.stats()

    def cpu_throttler_stats(self) -> dict:
        """Return current CPU throttler thread limit."""
        limit = self._cpu_throttler.thread_limit if self._cpu_throttler else 0
        return {"thread_limit": limit}

    def total_memory_usage(self) -> Dict[str, int]:
        """Return current total VRAM and RAM usage across all loaded models."""
        with self._lock:
            return {
                "vram_mb": self._total_vram_used_mb,
                "ram_mb": self._total_ram_used_mb,
            }

    # New method: expose unified environment status
    def get_environment_status(self) -> Optional[EnvironmentStatus]:
        return self._env_status

    # -------------------------------------------------------------------------
    # Graceful Shutdown
    # -------------------------------------------------------------------------
    async def shutdown(self, timeout_s: Optional[float] = None) -> None:
        if timeout_s is None:
            timeout_s = self._cfg.get("model_loader", {}).get("shutdown_timeout_s", 10.0)
        log.info("ModelLoader: shutting down...")
        self._shutdown_event.set()
        await self.unload_all()
        log.info("ModelLoader: shutdown complete")

    # -------------------------------------------------------------------------
    # Synchronous Load Implementation (GGUF only)
    # -------------------------------------------------------------------------
    def _load_sync(self, mc: ModelConfig, vram_budget_mb: int) -> LoadedModel:
        if self._shutdown_event.is_set():
            raise RuntimeError("Model load aborted due to server shutdown")

        from local_llm_core.engine.pipelines.layer_streamer import (
            FLPConfig, AdaptiveLoader, calc_optimal_gpu_layers, read_gguf_meta,
        )
        from local_llm_core.engine.pipelines.drive_pipeline import NVMeModelAdvisor

        path = Path(mc.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {mc.model_path}")

        with ModelLoader._sysctl_lock:
            if not ModelLoader._sysctl_applied:
                # NVMeModelAdvisor.set_sysctl_for_nvme writes
                # vm.dirty_bytes, vm.dirty_background_bytes,
                # vm.vfs_cache_pressure and kernel.numa_balancing — host-
                # wide sysctls, applied as root, that survive process exit.
                # They also FIGHT with SystemOptimizer.tune_vm, which writes
                # vm.dirty_ratio / vm.dirty_background_ratio — the two are
                # mutually exclusive at the kernel level (writing one zeroes
                # the other), so the observable value depended on which
                # component ran last.
                #
                # Gate on the same master switch that governs the rest of
                # host tuning. When allow_kernel_tuning is false (the new
                # default), this code path is a no-op.
                _allow_tuning = bool(
                    (self._cfg.get("system", {}) or {}).get(
                        "allow_kernel_tuning", False
                    )
                )
                if _allow_tuning:
                    log.info(
                        "Applying NVMe sysctls (system.allow_kernel_tuning=true): "
                        "dirty_bytes / dirty_background_bytes / vfs_cache_pressure"
                    )
                    NVMeModelAdvisor.set_sysctl_for_nvme()
                else:
                    log.info(
                        "Skipping NVMe sysctls (system.allow_kernel_tuning=false) "
                        "— dirty_* / vfs_cache_pressure left at host defaults"
                    )
                ModelLoader._sysctl_applied = True

        vm = psutil.virtual_memory()
        total_ram_free = max(0, int(vm.total * self._profile.ram_target) - vm.used) >> 20

        mc.quantization = mc.quantization or "auto"
        mc.context_length = int(mc.context_length) if mc.context_length else 0
        mc.num_gpu_layers = int(mc.num_gpu_layers) if mc.num_gpu_layers is not None else -1
        mc.n_threads = int(mc.n_threads) if mc.n_threads else 0

        # Profile specialised for this model (KV-cache type, measured speed).
        _prof = self._profile_for(mc)

        # ── Pre-read GGUF metadata ───────────────────────────────────────────
        # compute_context_length() needs n_ctx_train, which lives in the GGUF
        # header. Reading it here (cheap — GGUF header only) lets us honour
        # the model's trained context when sizing n_ctx instead of blindly
        # assuming the hardware profile default. calc_optimal_gpu_layers will
        # re-read the same header internally below; that's fine, the parse
        # is microseconds.
        try:
            _early_meta = read_gguf_meta(path) or {}
        except Exception as _rme:
            log.warning(
                "read_gguf_meta failed for %s: %s — using profile default",
                mc.model_id, _rme,
            )
            _early_meta = {}

        # ── Context length: explicit override, else memory-derived up to the trained window ──
        #
        # The old implementation clamped n_ctx to the hardware profile's
        # static `context_length` (2048 on APU tiers), which pinned every
        # reply's headroom to ~768 tokens regardless of the model's real
        # capacity. The KV-cache cost estimate fed into that clamp was also
        # wrong by ~4 orders of magnitude: it used
        # {"q4_0": 0.5, "q8_0": 1.0, "f16": 2.0} bytes-per-token, which are
        # per-element multipliers, not the real per-token KV footprint. For
        # Qwen3-8B q4_0 the true cost is 36 layers × 8 KV heads × 128 head
        # dim × (K+V) × 0.5625 B/elem = 41,472 bytes/token.
        #
        # The planner below uses the real numbers (via stream_guards, which
        # reads head_dim / head_dim_v / n_kv_heads from the GGUF header we
        # just parsed), budgets a configurable fraction of *available* RAM
        # for KV, and returns the largest 256-token-aligned n_ctx that fits
        # — never above the model's trained window, never below
        # model_loader.min_context_tokens.
        #
        # Per-model override semantics:
        #   mc.context_length > 0  → that exact value is used, no planner.
        #   mc.context_length == 0 → planner decides (this is the default
        #                            in config.yaml for qwen3-8b/4b).
        from local_llm_core.engine.inference.stream_guards import (
            kv_bytes_per_token, kv_budget_mb, plan_context_length,
        )
        profile_ctx = self._profile.context_length
        _ml_cfg = self._cfg.get("model_loader", {})
        _n_ctx_train = int(
            _early_meta.get("n_ctx_train") or _early_meta.get("context_length") or 0
        )

        if mc.context_length:
            # Explicit per-model override always wins, no planner.
            ctx_len = int(mc.context_length)
            computed = ctx_len
            _kv_bpt = 0.0
        else:
            # Real per-token KV cost from GGUF metadata. Qwen3-4B/8B both
            # report head_dim=128 from GGUF key_length, which is essential:
            # naively deriving head_dim = n_embd // n_head gives 80 for the
            # 4B and would under-size KV by 37 %.
            # kv_layers: layers that own a KV cache (fewer than n_layer for
            # hybrid attention + state-space models).
            _kv_bpt = kv_bytes_per_token(
                int(_early_meta.get("kv_layers") or _early_meta.get("n_layer") or 32),
                int(_early_meta.get("n_kv_heads") or _early_meta.get("n_head") or 8),
                int(_early_meta.get("head_dim")   or 128),
                int(_early_meta.get("head_dim_v") or 0) or None,
                kv_quant_k=_prof.kv_cache_quant,
            )
            _ram_avail_mb = int(psutil.virtual_memory().available >> 20)
            _kv_budget_mb = kv_budget_mb(
                _ram_avail_mb,
                float(_ml_cfg.get("kv_ram_share", 0.35)),
            )
            computed = plan_context_length(
                n_ctx_train=_n_ctx_train,
                kv_per_token_bytes=_kv_bpt,
                kv_budget_mb=_kv_budget_mb,
                floor=max(
                    int(profile_ctx),
                    int(_ml_cfg.get("min_context_tokens", 4096)),
                ),
                ceiling=int(_ml_cfg.get("max_context_tokens", 0) or 0),
            )
            ctx_len = computed

        log.info(
            "Context sizing for '%s': n_ctx_train=%d, profile_floor=%d, "
            "kv_bytes/token=%s, computed=%d, chosen=%d (mc.context_length=%d)",
            mc.model_id, _n_ctx_train, profile_ctx,
            "override" if mc.context_length else "%.0f" % _kv_bpt,
            computed, ctx_len, mc.context_length,
        )

        # ── BitNet-quantized GGUF: try a built BitNet.cpp binary first ──────
        # i2_s/tl1/tl2 quantization kernels live only in Microsoft's BitNet.cpp
        # fork of llama.cpp, not in upstream llama.cpp/llama-cpp-python. If a
        # built llama-server or llama-cli binary is found, use it directly —
        # no state file, no router, no SPECTRE/PRISM/FORGE/ATLAS. Regular GGUF
        # models (Q4_K_M, Q8_0, etc.) are unaffected by this check and always
        # fall through to the standard loader below.
        #
        # Previously gated ONLY on classify_gguf()'s binary
        # GGUF-header parse. A parser bug (now fixed) silently misclassified
        # real BitNet GGUFs as STANDARD, which skipped this branch entirely
        # with no error — the model then loaded via plain llama-cpp-python
        # and failed downstream. As defense-in-depth (a binary metadata
        # parser is the single most fragile link in this chain), the same
        # filename/config heuristic already used in layer_streamer.py as a
        # last-resort safety net is also checked here, so a future parser
        # regression degrades to a slower path instead of silently routing
        # a BitNet model through the wrong backend again.
        _bn_is_bitnet_hint = (
            mc.quantization in ("1bit", "i2_s", "tl1", "tl2")
            or any(x in path.name.lower() for x in ("i2_s", "tl1", "tl2", "bitnet"))
        )
        if _UNIFIED_ENV_AVAILABLE:
            try:
                _hdr = classify_gguf(mc.model_path)
                _is_bitnet = _hdr.model_type == GGUFModelType.BITNET or _bn_is_bitnet_hint
                if _is_bitnet:
                    if _hdr.model_type != GGUFModelType.BITNET:
                        log.warning(
                            "Load [%s]: GGUF header parse did not flag '%s' as BitNet, "
                            "but filename/quantization config indicates it is. "
                            "Routing via the BitNet.cpp path anyway.",
                            mc.model_id, path.name,
                        )
                    _bn_loaded = self._try_bitnet_cpp_backend(mc, ctx_len)
                    if _bn_loaded is not None:
                        return _bn_loaded
                    # else: no binary built yet — warning already logged;
                    # fall through and attempt the standard loader below,
                    # which will fail with the clear, classified error in
                    # layer_streamer.py if the format truly isn't supported.
            except Exception as _cls_exc:
                log.warning("GGUF classification failed for %s: %s", mc.model_id, _cls_exc)
                if _bn_is_bitnet_hint:
                    _bn_loaded = self._try_bitnet_cpp_backend(mc, ctx_len)
                    if _bn_loaded is not None:
                        return _bn_loaded
        elif _bn_is_bitnet_hint:
            # model_unified_env unavailable entirely — still attempt the
            # BitNet.cpp binary path using the filename/config heuristic.
            _bn_loaded = self._try_bitnet_cpp_backend(mc, ctx_len)
            if _bn_loaded is not None:
                return _bn_loaded

        # ── Overcommit factor: conservative for APUs (shared VRAM) ──────────
        # APU VRAM is carved from system RAM.  The actual allocatable VRAM is
        # lower than the reported total due to display compositor buffers and
        # memory fragmentation.  Using overcommit on shared-memory devices
        # pushes estimated usage past real capacity and triggers hipMalloc
        # failure → abort().  Use 0.90 (10% safety margin) for APU profiles.
        _is_apu = self._profile.name.startswith("apu_")
        _overcommit = 0.90 if _is_apu else 1.05

        if mc.num_gpu_layers == -1:
            budget = calc_optimal_gpu_layers(
                path,
                vram_budget_mb=vram_budget_mb,
                ram_free_mb=max(total_ram_free, 1024),
                context_len=ctx_len,
                kv_quant=_prof.kv_cache_quant,
                layer_overcommit_factor=_overcommit,
                # honour the profile's context-vs-layers priority.
                # Without these two kwargs, the planner defaults to "balanced",
                # which grows context back to 4× the base value (32768 from
                # 8192) while computing the layer plan — then those layers
                # are what actually gets loaded. Result: fewer GPU layers
                # (7 vs ~11) than necessary for the 8192-token context we
                # actually load with.
                priority=_prof.context_vs_layers_priority,
                hardware_profile=_prof,
            )
            n_gpu = budget["n_gpu_layers"]
            strategy = budget["strategy"]
            meta = budget["meta"]
        else:
            n_gpu = mc.num_gpu_layers
            strategy = "manual"
            meta = read_gguf_meta(path)

        if mc.quantization == "auto":
            params_b = self._estimate_params_b(path)
            mc.quantization = _select_quantization(params_b, max(self._vram_total_mb(), 1) / 1024)

        n_threads = mc.n_threads or self._cpu_throttler.thread_limit
        # Batch threads: physical cores, never logical. The profile's
        # `n_threads()` already returns physical × ratio; for prefill we
        # want the same count, not SMT siblings.
        try:
            import psutil as _ps2
            _physical = _ps2.cpu_count(logical=False) or n_threads
        except Exception:
            _physical = n_threads
        n_threads_batch = max(1, int(_physical))

        # ── Continuous batching: build the llama context ONCE, at load ──
        # layer_streamer already forwards n_seq_max/kv_unified to Llama(**kw),
        # but nothing populated them, so the batched engine always had to free
        # the context and build a second one (10+ s, and the step that
        # ggml_abort()ed). No-op when n_seq_max <= 1 (e.g. the 2 GB APU).
        _n_seq_load, _kv_uni_load, _guard_fp = 1, True, None
        try:
            _cb_cfg = ((self._cfg.get("inference") or {}).get("continuous_batching") or {})
            if bool(_cb_cfg.get("enabled", False)):
                _n_seq_load = max(1, int(_cb_cfg.get("n_seq_max", 1)))
                if bool(_cb_cfg.get("pf_enabled", False)):
                    _n_seq_load += 2        # BatchedInferenceEngine._PF_RESERVED
                _kv_uni_load = bool(_cb_cfg.get("kv_unified", True))
            if _n_seq_load > 1:
                from local_llm_core.engine.inference import ctx_guard as _cg
                _guard_fp = _cg.fingerprint(
                    mc.model_path, ctx_len, _n_seq_load, _kv_uni_load,
                )
                if _cg.previous_start_crashed(_guard_fp):
                    log.critical(
                        "Previous start died while building a context with "
                        "n_seq_max=%d — loading with n_seq_max=1 (see %s).",
                        _n_seq_load, _cg.crashed_path(),
                    )
                    _n_seq_load, _guard_fp = 1, None
        except Exception as _lg_exc:
            log.warning("load-time n_seq_max wiring skipped: %s", _lg_exc)
            _n_seq_load, _guard_fp = 1, None

        cfg = FLPConfig(
            n_seq_max=_n_seq_load,
            kv_unified=_kv_uni_load,
            model_path=mc.model_path,
            # llama.cpp counts the output (lm_head) layer as layer n_layer + 1:
            # asking for exactly n_layer leaves it on the CPU.  When every
            # repeating layer is offloaded, offload the output layer too.
            n_gpu_layers=(
                n_gpu + 1
                if n_gpu > 0 and n_gpu >= int(meta.get("n_layer", 0) or 0) > 0
                else n_gpu
            ),
            n_ctx=ctx_len,
            n_batch=self._profile.n_batch,
            n_ubatch=self._profile.n_ubatch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            use_mmap=mc.use_mmap,
            use_mlock=mc.use_mlock,
            flash_attn=self._profile.flash_attn,
            rope_freq_base=mc.rope_scaling if mc.rope_scaling != 1.0 else 0.0,
            tensor_split=mc.gpu_split,
            cache_type_k=_prof.kv_cache_quant,
            cache_type_v=_prof.kv_cache_quant,
        )

        NVMeModelAdvisor.advise(mc.model_path, ram_free_mb=total_ram_free)

        # RAM sanity check
        _vm_now = psutil.virtual_memory()
        _actual_free_mb = max(0, _vm_now.available >> 20)
        _model_file_mb = max(1, path.stat().st_size >> 20)
        _gpu_frac = n_gpu / max(meta.get("n_layer", 32), 1) if n_gpu > 0 else 0.0
        _cpu_weight_mb = int(_model_file_mb * (1.0 - _gpu_frac))
        if _cpu_weight_mb > _actual_free_mb and not cfg.use_mmap:
            raise RuntimeError(f"Low RAM: need ~{_cpu_weight_mb}MB for CPU layers, only {_actual_free_mb}MB free. Use mmap or reduce gpu_layers.")
        elif _cpu_weight_mb > _actual_free_mb and cfg.use_mmap:
            log.warning("Low RAM: model CPU layers = ~%dMB, available = %dMB – streaming from NVMe (slower but safe).",
                        _cpu_weight_mb, _actual_free_mb)

        if self._shutdown_event.is_set():
            raise RuntimeError("Model load aborted due to server shutdown")

        t0 = time.monotonic()
        _ls_cfg = dict(self._cfg.get("layer_streamer", {}) or {})
        _ls_cfg.setdefault("n_threads_batch", n_threads_batch)
        loader = AdaptiveLoader(cfg, streamer_config=_ls_cfg)
        llm = None
        if _guard_fp is not None:
            from local_llm_core.engine.inference import ctx_guard as _cg
            _cg.arm(_guard_fp)
        try:
            llm = loader.load()
            if _guard_fp is not None:
                _cg.disarm()
        except Exception:
            if _guard_fp is not None:
                _cg.disarm()        # a Python exception is not a C abort
            if llm is not None:
                try:
                    if hasattr(llm, "close"):
                        llm.close()
                    elif hasattr(llm, "free"):
                        llm.free()
                except Exception as ce:
                    log.warning("llm cleanup on load failure: %s", ce)
            raise

        elapsed = time.monotonic() - t0
        n_layers = meta.get("n_layer", 32)
        file_mb = path.stat().st_size >> 20
        vram_mb = int(file_mb * n_gpu / max(n_layers, 1))

        # VRAM bookkeeping on a GTT-inclusive APU:
        #
        # `vram_budget_mb` in the profile is the hardware carve-out (2 GB
        # on this box) minus OS reserve. The model can be larger than that
        # and still load because llama.cpp uses GTT (system RAM addressable
        # through the iGPU). Trying to track the full model size against
        # the carve-out budget produces the 3 × "VRAM full — cannot
        # allocate 5 GB" warnings you saw at every load even though the
        # load succeeded.
        #
        # Detect that case and skip bookkeeping. The real GPU consumption
        # is visible in /v1/metrics (vram.used_bytes) via the drm sysfs
        # query, which reads what the driver actually allocated.
        try:
            _budget_mb = self._profile.vram_budget_mb(self._vram_total_mb())
            _gtt_mb = int(
                getattr(self._system_info, "vram_gtt_total_mb", 0) or 0
            ) if self._system_info is not None else 0
            _is_gtt_apu = _gtt_mb > 0
            if _is_gtt_apu and vram_mb > _budget_mb * 1.2:
                log.info(
                    "VRAM tracker: skipping bookkeeping for '%s' "
                    "(est %d MB, carve-out budget %d MB, GTT total %d MB). "
                    "Model is running from GTT-addressable RAM, not the "
                    "carve-out — see /v1/metrics for the driver-reported "
                    "allocation.",
                    mc.model_id, vram_mb, _budget_mb, _gtt_mb,
                )
            else:
                self._vram.allocate(
                    key=mc.model_id, data=None,
                    size_bytes=vram_mb * 1024 * 1024,
                )
        except Exception:
            log.info(
                "VRAM allocation tracking skipped for '%s'", mc.model_id,
            )

        log.info("Loaded '%s' %.1fs | backend=%s | gpu=%d/%d | strategy=%s | vram=%dMB",
                 mc.model_id, elapsed, loader.backend, n_gpu, n_layers, strategy, vram_mb)

        return LoadedModel(
            config=mc,
            llm=llm,
            total_layers=n_layers,
            gpu_layers=n_gpu,
            cpu_layers=n_layers - n_gpu,
            vram_used_mb=vram_mb,
            load_time_s=elapsed,
            strategy=strategy,
            backend=loader.backend,
            nvme_advised=True,
            cpu_threads=n_threads,
            profile_name=self._profile.name,
            model_type="gguf",
            loaded_context_length=ctx_len,
        )

    # -------------------------------------------------------------------------
    # Synchronous Unload (Direct close, no fork)
    # -------------------------------------------------------------------------
    def _unload_sync(self, model: LoadedModel) -> None:
        """Release model resources. Simple close/free – safe in separate thread."""
        mid = model.config.model_id
        log.info("Unloading model '%s'", mid)
        freed = False
        if hasattr(model.llm, "close"):
            try:
                model.llm.close()
                freed = True
            except SystemExit:
                freed = True
                log.warning("llama.cpp triggered SystemExit during unload – suppressed")
            except Exception as e:
                log.warning("llm.close() error for '%s': %s", mid, e)
        elif hasattr(model.llm, "free"):
            try:
                model.llm.free()
                freed = True
            except SystemExit:
                freed = True
                log.warning("llama.cpp triggered SystemExit during unload – suppressed")
            except Exception as e:
                log.warning("llm.free() error for '%s': %s", mid, e)
        if not freed:
            log.warning("No close/free method on model '%s' – relying on GC", mid)
        try:
            del model.llm
        except Exception:
            pass
        gc.collect()

        try:
            self._vram.free(model.config.model_id)
        except Exception:
            log.warning("VRAM free tracking failed for '%s'", mid)

        log.info("Model '%s' unloaded (clean=%s)", mid, freed)

    async def unload_all(self) -> int:
        with self._lock:
            ids = list(self._models.keys())
        count = 0
        for mid in ids:
            try:
                if await self.unload(mid):
                    count += 1
            except Exception as e:
                log.error("unload_all '%s' failed: %s", mid, e)
        return count

    @staticmethod
    def _estimate_params_b(path: Path) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[bB]", path.name)
        if m:
            return float(m.group(1))
        return path.stat().st_size / 1024**3 / 0.56
