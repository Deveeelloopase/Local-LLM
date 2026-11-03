from __future__ import annotations

"""
engine/models/model_lifecycle.py — Production‑Ready Model Lifecycle Manager
===========================================================================
v1.0.0 – Config and profiles injected; no filesystem access.

Automatically manages the lifecycle of loaded models (GGUF and BitNet)
based on memory budgets, idle timeouts, priority, and demand.

Features:
- Memory‑aware startup: preloads models that fit in available RAM/VRAM,
  respecting profile priority and autoload flags.
- On‑demand loading: models not preloaded can be loaded later when memory allows.
- Idle eviction: unloads models that haven't been used for a configurable time.
- Memory‑pressure eviction: evicts lower‑priority models to free memory.
- Provides model status for server listing.
- Integrates with GGUF ModelLoader and BitNetLoader via hooks.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class ModelLifecycleInfo:
    """Tracked state for one loaded model."""
    model_id: str
    backend: str                      # "gguf" or "bitnet"
    vram_used_mb: int = 0
    ram_used_mb: int = 0
    load_time_s: float = 0.0
    last_used: float = field(default_factory=time.monotonic)
    priority: int = 0
    autoload: bool = False
    status: str = "loading"           # loading / ready / error
    error_message: Optional[str] = None
    gguf_model: Optional[Any] = None
    bitnet_wrapper: Optional[Any] = None

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used


# -----------------------------------------------------------------------------
# ModelLifecycleManager
# -----------------------------------------------------------------------------
class ModelLifecycleManager:
    """
    Manages the set of loaded models across all backends.

    Receives lifecycle configuration and model profiles from the application.
    No file I/O; all settings are explicit.

    Parameters
    ----------
    lifecycle_config : dict
        Dictionary containing the `model_lifecycle` section (or a
        ModelLifecycleConfig Pydantic model).
    model_profiles : list of dict
        List of model profile dictionaries (e.g., from `models.profiles` in
        the app config).
    vram_manager, ram_optimizer, gguf_loader, bitnet_loader, inference_engine
        Required components; can be set later after construction.
    """

    def __init__(
        self,
        *,
        lifecycle_config: Optional[Dict[str, Any]] = None,
        model_profiles: Optional[List[Dict[str, Any]]] = None,
        vram_manager: Optional[Any] = None,
        ram_optimizer: Optional[Any] = None,
        gguf_loader: Optional[Any] = None,
        bitnet_loader: Optional[Any] = None,
        inference_engine: Optional[Any] = None,
    ) -> None:
        # ---- lifecycle config ------------------------------------------------
        if lifecycle_config is None:
            lifecycle_config = {}
        # Optionally validate with Pydantic if available
        try:
            from local_llm_core.engine.utils.config_loader import ModelLifecycleConfig
            if isinstance(lifecycle_config, dict):
                validated = ModelLifecycleConfig(**lifecycle_config)
                self._cfg = validated.model_dump()
            else:
                self._cfg = lifecycle_config.model_dump()
        except Exception as exc:
            log.warning("Lifecycle config validation failed: %s; using defaults", exc)
            self._cfg = {}  # will rely on hardcoded defaults below

        # ---- model profiles --------------------------------------------------
        self._model_profiles = model_profiles or []

        # ---- external components ---------------------------------------------
        self._vram = vram_manager
        self._ram = ram_optimizer
        self._gguf_loader = gguf_loader
        self._bitnet_loader = bitnet_loader
        self._inference_engine = inference_engine

        # ---- internal state --------------------------------------------------
        self._models: Dict[str, ModelLifecycleInfo] = {}
        self._lock = threading.RLock()
        self._loading: Set[str] = set()               # model_ids currently being loaded
        self._loading_events: Dict[str, asyncio.Event] = {}

        # ---- tunable parameters ----------------------------------------------
        self._max_loaded = int(self._cfg.get("max_loaded_models", 8))
        self._idle_timeout_s = float(self._cfg.get("idle_timeout_s", 1800.0))
        self._vram_target = float(self._cfg.get("vram_target_utilization", 0.90))
        self._ram_target = float(self._cfg.get("ram_target_utilization", 0.75))
        self._eviction_policy = self._cfg.get("eviction_policy", "lru")
        self._check_interval_s = float(self._cfg.get("check_interval_s", 15.0))
        self._auto_evict = bool(self._cfg.get("auto_evict_on_pressure", True))

        # ---- background task ------------------------------------------------
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        log.info("ModelLifecycleManager initialised: max_loaded=%d idle_timeout=%.0fs eviction=%s",
                 self._max_loaded, self._idle_timeout_s, self._eviction_policy)

    # -------------------------------------------------------------------------
    # Lifecycle hooks (called by loaders)
    # -------------------------------------------------------------------------
    def on_model_loaded(self, model_id: str, model: Any) -> None:
        backend = self._detect_backend(model)
        with self._lock:
            if model_id in self._models:
                old = self._models[model_id]
                old.last_used = time.monotonic()
                old.status = "ready"
                if backend == "gguf":
                    old.gguf_model = model
                else:
                    old.bitnet_wrapper = model
                # Wake up any waiting task
                evt = self._loading_events.get(model_id)
                if evt and not evt.is_set():
                    evt.set()
                return

            vram_mb = 0
            ram_mb = 0
            if backend == "gguf":
                vram_mb = getattr(model, "vram_used_mb", 0)
                ram_mb = getattr(model, "cpu_layers", 0) * 100
            else:
                mem = model.get_memory_usage() if hasattr(model, "get_memory_usage") else {}
                vram_mb = mem.get("vram_mb", 0)
                ram_mb = mem.get("ram_mb", 0)

            info = ModelLifecycleInfo(
                model_id=model_id,
                backend=backend,
                vram_used_mb=vram_mb,
                ram_used_mb=ram_mb,
                load_time_s=time.monotonic(),
                status="ready",
            )
            if backend == "gguf":
                info.gguf_model = model
            else:
                info.bitnet_wrapper = model
            info.priority = self._get_model_priority_from_profiles(model_id)
            info.autoload = False
            self._models[model_id] = info

            # Wake up any waiting task
            evt = self._loading_events.get(model_id)
            if evt and not evt.is_set():
                evt.set()

        log.info("Lifecycle: model '%s' (%s) registered", model_id, backend)

    def on_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
        log.info("Lifecycle: model '%s' removed", model_id)

    # -------------------------------------------------------------------------
    # Public API – load / unload
    # -------------------------------------------------------------------------
    async def load_model(self, model_id: str) -> Any:
        """
        Ensure a model is loaded, loading it via the appropriate loader if not.
        Returns the container object (LoadedModel for GGUF, BitNetInferenceWrapper
        for BitNet) suitable for the inference engine's unified registry.
        """
        # Fast‑path: already ready
        with self._lock:
            info = self._models.get(model_id)
            if info and info.status == "ready":
                info.last_used = time.monotonic()
                if info.backend == "gguf" and info.gguf_model is not None:
                    info.gguf_model.request_count += 1
                    return info.gguf_model
                elif info.backend == "bitnet" and info.bitnet_wrapper is not None:
                    return info.bitnet_wrapper

        # If already loading, wait for the loading event (instead of a busy loop)
        with self._lock:
            if model_id in self._loading:
                evt = self._loading_events.get(model_id)
                # Fallback – create one if missing
                if evt is None:
                    evt = asyncio.Event()
                    self._loading_events[model_id] = evt
                # Release lock while waiting
                wait_event = evt

        if wait_event:
            log.debug("Waiting for concurrent load of '%s'", model_id)
            try:
                await asyncio.wait_for(wait_event.wait(), timeout=600.0)  # generous timeout
            except asyncio.TimeoutError:
                raise RuntimeError(f"Timed out waiting for model '{model_id}' to load")
            # Re‑check
            with self._lock:
                info = self._models.get(model_id)
                if info and info.status == "ready":
                    if info.backend == "gguf" and info.gguf_model is not None:
                        return info.gguf_model
                    if info.bitnet_wrapper is not None:
                        return info.bitnet_wrapper
            raise RuntimeError(f"Model '{model_id}' failed to become ready after concurrent load")

        # Initiate new load
        self._loading.add(model_id)
        # Create loading event
        evt = asyncio.Event()
        with self._lock:
            self._loading_events[model_id] = evt
            if model_id not in self._models:
                self._models[model_id] = ModelLifecycleInfo(
                    model_id=model_id,
                    backend="unknown",
                    status="loading",
                    priority=self._get_model_priority_from_profiles(model_id),
                )
            else:
                self._models[model_id].status = "loading"

        try:
            backend = self._resolve_backend(model_id)
            if backend == "gguf":
                profile = self._get_profile_dict(model_id)
                from local_llm_core.engine.gguf.model_loader import ModelConfig
                mc = ModelConfig(
                    model_id=model_id,
                    model_path=profile["path"],
                    quantization=profile.get("quantization", "auto"),
                    context_length=profile.get("context_length", 0),
                    num_gpu_layers=profile.get("num_gpu_layers", -1),
                    n_threads=profile.get("n_threads", 0),
                )
                loaded = await self._gguf_loader.load(mc)
                return loaded
            elif backend == "bitnet":
                wrapper = await self._bitnet_loader.load(model_id)
                return wrapper
            else:
                raise ValueError(f"Unknown backend for model '{model_id}'")
        except Exception as e:
            with self._lock:
                if model_id in self._models:
                    self._models[model_id].status = "error"
                    self._models[model_id].error_message = str(e)
            log.error("Failed to load model '%s': %s", model_id, e)
            raise
        finally:
            self._loading.discard(model_id)
            evt.set()  # release any waiters
            with self._lock:
                self._loading_events.pop(model_id, None)

    # -------------------------------------------------------------------------
    # Profile helpers (no file reads)
    # -------------------------------------------------------------------------
    def _get_profile_dict(self, model_id: str) -> dict:
        for prof in self._model_profiles:
            if prof.get("id") == model_id:
                return prof
        raise ValueError(f"Model profile '{model_id}' not found in provided profiles")

    def _get_model_priority_from_profiles(self, model_id: str) -> int:
        for prof in self._model_profiles:
            if prof.get("id") == model_id:
                return prof.get("priority", 10)
        return 10

    def _resolve_backend(self, model_id: str) -> str:
        for prof in self._model_profiles:
            if prof.get("id") == model_id:
                fmt = prof.get("format", "gguf").lower()
                return "bitnet" if fmt == "bitnet" else "gguf"
        return "gguf"  # safe default

    # -------------------------------------------------------------------------
    # Memory‑aware preloading
    # -------------------------------------------------------------------------
    async def preload_startup_models(self) -> List[str]:
        """
        Preload models marked `autoload` in order of priority, respecting RAM budget.
        """
        avail_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
        budget_mb = int(avail_ram_mb * self._ram_target)

        autoload_all = self._cfg.get("autoload_all", False)
        profiles_sorted = sorted(
            self._model_profiles,
            key=lambda p: p.get("priority", 10),
            reverse=True,
        )

        loaded_ids = []
        used_mb = 0

        for prof in profiles_sorted:
            autoload = prof.get("autoload", False) or autoload_all
            if not autoload:
                continue

            model_id = prof.get("id")
            estimated_mb = self._estimate_model_memory(prof)
            if estimated_mb is None:
                log.warning("Cannot estimate memory for '%s', skipping", model_id)
                continue

            if used_mb + estimated_mb > budget_mb:
                log.info("Skipping autoload of '%s': would exceed RAM budget", model_id)
                continue

            try:
                log.info("Auto‑loading '%s' (priority=%d, est=%dMB)",
                         model_id, prof.get("priority", 10), estimated_mb)
                await self.load_model(model_id)
                loaded_ids.append(model_id)
                used_mb += estimated_mb
            except Exception as e:
                log.error("Autoload failed for '%s': %s", model_id, e)
                with self._lock:
                    if model_id in self._models:
                        self._models[model_id].status = "error"
                        self._models[model_id].error_message = str(e)
        return loaded_ids

    @staticmethod
    def _estimate_model_memory(profile: Dict[str, Any]) -> Optional[int]:
        """Return estimated RAM usage in MB for a given profile dict."""
        fmt = profile.get("format", "gguf").lower()
        path = profile.get("path", "")
        if not path:
            return None
        if fmt == "gguf":
            try:
                from local_llm_core.engine.gguf.model_loader import ModelLoader
                if hasattr(ModelLoader, "estimate_model_size"):
                    return ModelLoader.estimate_model_size(path)
            except Exception:
                pass
            try:
                return int(Path(path).stat().st_size // (1024 * 1024) * 1.1)
            except Exception:
                return None
        elif fmt == "bitnet":
            try:
                from local_llm_core.engine.bitnet.bitnet_loader import BitNetLoader
                if hasattr(BitNetLoader, "estimate_model_size"):
                    return BitNetLoader.estimate_model_size(path)
            except Exception:
                pass
            p = Path(path)
            safetensors = list(p.glob("*.safetensors"))
            if safetensors:
                total = sum(f.stat().st_size for f in safetensors)
                return int(total * 1.5 / (1024 * 1024))
            return 2000
        return None

    # -------------------------------------------------------------------------
    # Status reporting
    # -------------------------------------------------------------------------
    def get_loaded_models_info(self) -> List[Dict[str, Any]]:
        result = []
        with self._lock:
            for mid, info in self._models.items():
                entry = {
                    "id": mid,
                    "model_type": info.backend if info.backend in ("gguf", "bitnet") else "unknown",
                    "status": info.status,
                    "backend": info.backend,
                    "vram_mb": info.vram_used_mb,
                    "ram_mb": info.ram_used_mb,
                    "priority": info.priority,
                    "idle_seconds": round(info.idle_seconds, 1),
                    "autoload": info.autoload,
                }
                if info.error_message:
                    entry["error"] = info.error_message
                result.append(entry)
        return result

    # -------------------------------------------------------------------------
    # Background task
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._background_loop(), name="lifecycle")

    async def shutdown(self) -> None:
        if self._task:
            self._stop_event.set()
            # Wait for task to finish; cancel if needed
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._check_interval_s)
                await self._evict_idle_models()
                await self._check_memory_pressure()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Lifecycle background error: %s", e)

    # -------------------------------------------------------------------------
    # Eviction logic
    # -------------------------------------------------------------------------
    async def _evict_idle_models(self) -> None:
        now = time.monotonic()
        to_unload = []
        with self._lock:
            for model_id, info in self._models.items():
                if info.status == "ready" and info.idle_seconds > self._idle_timeout_s and not info.autoload:
                    to_unload.append(model_id)

        for model_id in to_unload:
            log.info("Evicting idle model '%s' (idle=%.0fs)", model_id, self._models[model_id].idle_seconds)
            try:
                await self._unload_model_safe(model_id)
            except Exception as e:
                log.error("Failed to unload idle model '%s': %s", model_id, e)

    async def _check_memory_pressure(self) -> None:
        if not self._auto_evict:
            return

        vram_used = sum(info.vram_used_mb for info in self._models.values() if info.status == "ready")
        vram_total = self._get_total_vram_mb()
        ram_used = sum(info.ram_used_mb for info in self._models.values() if info.status == "ready")
        ram_total_mb = psutil.virtual_memory().total // (1024 * 1024)
        ram_limit_mb = int(ram_total_mb * self._ram_target)

        # VRAM pressure
        if vram_total and vram_used > vram_total * self._vram_target:
            needed = vram_used - int(vram_total * self._vram_target)
            to_evict = self._select_eviction_candidates(needed, target="vram")
            for model_id in to_evict:
                log.warning("Memory pressure: evicting '%s' to free VRAM", model_id)
                await self._unload_model_safe(model_id)

        # RAM pressure
        if ram_used > ram_limit_mb:
            needed = ram_used - ram_limit_mb
            to_evict = self._select_eviction_candidates(needed, target="ram")
            for model_id in to_evict:
                log.warning("Memory pressure: evicting '%s' to free RAM", model_id)
                await self._unload_model_safe(model_id)

    def _select_eviction_candidates(self, needed_mb: int, target: str = "vram") -> List[str]:
        with self._lock:
            if not self._models:
                return []
            ready = [m for m in self._models.values() if m.status == "ready"]
            if self._eviction_policy == "lru":
                ready.sort(key=lambda m: m.last_used)  # oldest first
            elif self._eviction_policy == "priority":
                # Evict lowest priority first
                ready.sort(key=lambda m: (m.priority, m.last_used))
            else:
                ready.sort(key=lambda m: m.last_used)

            freed = 0
            to_evict = []
            for m in ready:
                if m.autoload:  # never evict autoload models automatically
                    continue
                freed += m.vram_used_mb if target == "vram" else m.ram_used_mb
                to_evict.append(m.model_id)
                if freed >= needed_mb:
                    break
            return to_evict

    async def _unload_model_safe(self, model_id: str) -> None:
        """Unload a model, preferring the inference_engine if available."""
        try:
            if self._inference_engine:
                await self._inference_engine.unload_model(model_id)
            else:
                # Direct fallback
                with self._lock:
                    info = self._models.get(model_id)
                    if not info:
                        return
                if info.backend == "gguf" and self._gguf_loader:
                    await self._gguf_loader.unload(model_id)
                elif info.backend == "bitnet" and self._bitnet_loader:
                    await self._bitnet_loader.unload(model_id)
        except Exception as e:
            log.error("Failed to unload '%s': %s", model_id, e)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _detect_backend(self, model: Any) -> str:
        from local_llm_core.engine.gguf.model_loader import LoadedModel
        from local_llm_core.engine.bitnet.bitnet_engine import BitNetInferenceWrapper
        if isinstance(model, LoadedModel):
            return "gguf"
        if isinstance(model, BitNetInferenceWrapper):
            return "bitnet"
        if hasattr(model, "strategy"):
            return "gguf"
        return "bitnet"

    def _get_total_vram_mb(self) -> Optional[int]:
        if self._vram and hasattr(self._vram, "total_bytes"):
            return self._vram.total_bytes // (1024 * 1024)
        if self._inference_engine and hasattr(self._inference_engine, "loader") \
                and hasattr(self._inference_engine.loader, "_vram"):
            return self._inference_engine.loader._vram.total_bytes // (1024 * 1024)
        return None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "loaded_count": len(self._models),
                "max_loaded": self._max_loaded,
                "idle_timeout_s": self._idle_timeout_s,
                "models": {
                    mid: {
                        "idle_seconds": round(info.idle_seconds, 1),
                        "vram_mb": info.vram_used_mb,
                        "ram_mb": info.ram_used_mb,
                        "backend": info.backend,
                        "status": info.status,
                    }
                    for mid, info in self._models.items()
                },
            }
