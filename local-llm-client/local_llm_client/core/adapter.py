"""
local_llm_client/core/adapter.py

Production‑grade drop‑in adapter for ai_agent.py and model_manager.py.

Provides:
- LocalLLMCloudAdapter: exact interface as BaseLLMAdapter
- TokenMetrics: dataclass matching model_manager.TokenMetrics
- AdaptiveContextManager: context window management with token budgeting
- get_model_manager(): singleton accessor

Enhancements over previous version:
- generate_full() returns a GenerationResult with complete performance data.
- Uses the accelerated token counter from _accel when available.
- Thread‑safe context window updates with asyncio‑friendly locking.
- Graceful handling of transient failures during initialisation.
- Built‑in circuit‑breaker reset on re‑initialisation (via public method).
- Backward‑compatible – all existing calls continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .client import LLMCloudClient
from .async_client import AsyncLLMCloudClient
from .exceptions import LLMCloudError, ModelNotLoadedError, CircuitOpenError
from .models import ClientConfig, GenerationResult
from ._accel import estimate_tokens_fast

log = logging.getLogger(__name__)

# Thread‑local / context var for singleton config (allows per‑thread reuse in tests)
_current_adapter_config: ContextVar[Optional[ClientConfig]] = ContextVar(
    "adapter_config", default=None
)

# ------------------------------------------------------------------------------
# Fast token estimator – always available via _accel
# ------------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Fast heuristic token count (uses C accelerator if available)."""
    if not text:
        return 0
    return estimate_tokens_fast(text)


# ------------------------------------------------------------------------------
# TokenMetrics – matches model_manager.TokenMetrics
# ------------------------------------------------------------------------------
@dataclass
class TokenMetrics:
    """
    Token budget metrics – identical interface to model_manager.TokenMetrics.

    This dataclass is deliberately duplicated from local_llm_client.core.models
    to avoid a hard dependency on that module from ai_agent.py (which only
    imports adapter.py).  Both definitions are kept in sync.
    """
    prompt_tokens: int
    available_for_generation: int
    recommended_max_gen: int
    total_budget: int
    remaining: int
    efficiency_ratio: float
    is_feasible: bool
    warnings: List[str] = field(default_factory=list)
    recommended_timeout: float = 120.0
    max_prompt_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"TokenMetrics(prompt={self.prompt_tokens}, "
            f"available_gen={self.available_for_generation}, "
            f"recommended={self.recommended_max_gen}, "
            f"timeout={self.recommended_timeout:.1f}s, "
            f"feasible={self.is_feasible})"
        )


# ------------------------------------------------------------------------------
# AdaptiveContextManager – singleton for context management
# ------------------------------------------------------------------------------
class AdaptiveContextManager:
    """
    Compatibility shim for model_manager.AdaptiveModelManager.

    Manages a dynamic context window based on server metrics and
    memory pressure.  Singleton per process (thread‑safe).

    The context window starts at the model's default context size
    (or the configured max_tokens) and can be reduced under load,
    then slowly recovered after successful generations.
    """

    _instance: Optional["AdaptiveContextManager"] = None
    _inst_lock: threading.Lock = threading.Lock()
    _init_config: Optional[ClientConfig] = None

    def __new__(cls, config: Optional[ClientConfig] = None) -> "AdaptiveContextManager":
        with cls._inst_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialized = False
                cls._instance = obj
                cls._init_config = config
            elif config is not None and config != cls._init_config:
                log.warning(
                    "AdaptiveContextManager already initialized with different config. "
                    "Singleton reused; new config ignored. Use reset_singleton() for testing."
                )
        return cls._instance

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        with self.__class__._inst_lock:
            if self._initialized:
                return
            self.cfg = config or ClientConfig()
            self._ensure_defaults()
            self._sync_client = LLMCloudClient(self.cfg)
            self._async_client = AsyncLLMCloudClient(self.cfg)
            self._context_window = self.cfg.max_tokens or 4096
            self._initial_ctx = self._context_window
            self._lock = threading.Lock()
            self._tps_cache: Optional[float] = None
            self._tps_cache_time: float = 0.0
            self._initialized = True
            log.debug("AdaptiveContextManager initialized with config: %s", self.cfg)

    def _ensure_defaults(self) -> None:
        defaults = {
            "health_check_retries": 3,
            "health_check_retry_delay_s": 2.0,
            "wait_for_model": False,
            "wait_timeout_s": 300.0,
            "tps_estimate_fallback": 7.0,
            "tps_cache_ttl_s": 60.0,
        }
        for attr, default in defaults.items():
            if not hasattr(self.cfg, attr):
                setattr(self.cfg, attr, default)

    @classmethod
    def reset_singleton(cls) -> None:
        with cls._inst_lock:
            cls._instance = None
            cls._init_config = None

    async def _get_server_tps(self) -> float:
        """Query the server for average TPS, with caching."""
        now = time.time()
        ttl = getattr(self.cfg, "tps_cache_ttl_s", 60.0)
        if self._tps_cache is not None and (now - self._tps_cache_time) < ttl:
            return self._tps_cache
        try:
            metrics = await self._async_client.metrics()
            # Try multiple possible locations for the TPS value
            tps = None
            for key in ("tokens_per_second", "avg_tps"):
                if "performance" in metrics:
                    tps = metrics["performance"].get(key)
                if tps is None and "engine" in metrics:
                    tps = metrics["engine"].get(key)
                if tps is None:
                    tps = metrics.get(key)
                if tps is not None:
                    break
            if tps is not None and tps > 0:
                self._tps_cache = float(tps)
                self._tps_cache_time = now
                return self._tps_cache
        except Exception:
            pass
        return self.cfg.tps_estimate_fallback

    def calculate_token_budget(
        self,
        prompt_text: str,
        estimated_prompt_tokens: Optional[int] = None,
        max_generation_tokens: int = 512,
    ) -> TokenMetrics:
        """
        Calculate how many tokens remain for generation and a recommended
        timeout based on current server performance.

        Args:
            prompt_text: The full prompt text.
            estimated_prompt_tokens: Pre‑computed token count (optional).
            max_generation_tokens: Desired maximum generation length.

        Returns:
            TokenMetrics with budget details.
        """
        prompt_toks = estimated_prompt_tokens or _estimate_tokens(prompt_text)
        with self._lock:
            budget = self._context_window
        available = max(0, budget - prompt_toks - 64)
        rec_gen = min(max_generation_tokens, available)
        feasible = available > 50

        # Try to get TPS estimate
        fallback = getattr(self.cfg, "tps_estimate_fallback", 7.0)
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                tps = fallback
            else:
                tps = loop.run_until_complete(self._get_server_tps())
        except RuntimeError:
            tps = fallback
        except Exception:
            tps = fallback

        rec_timeout = max(30.0, rec_gen / max(tps, 0.1) + 15.0)
        warnings = []
        if not feasible:
            warnings.append(f"Prompt ({prompt_toks} tokens) leaves only {available} for generation")
        if prompt_toks > budget * 0.8:
            warnings.append("Prompt uses >80% of context window – consider compressing")

        return TokenMetrics(
            prompt_tokens=prompt_toks,
            available_for_generation=available,
            recommended_max_gen=rec_gen,
            total_budget=budget,
            remaining=available,
            efficiency_ratio=round(available / max(budget, 1), 3),
            is_feasible=feasible,
            warnings=warnings,
            recommended_timeout=rec_timeout,
            max_prompt_tokens=budget,
        )

    def adapt_prompt_to_context(
        self, original_prompt: str, target_tokens: int
    ) -> str:
        """
        Compress a prompt to fit within a token budget, prioritising
        important lines (tool calls, goals, numbered steps).
        """
        if _estimate_tokens(original_prompt) <= target_tokens:
            return original_prompt

        lines = original_prompt.split("\n")
        scored = []
        for line in lines:
            score = self._score_line(line)
            toks = _estimate_tokens(line)
            scored.append((line, score, toks))

        result = []
        used = 0
        for priority in (2, 1, 0):
            for line, score, toks in scored:
                if score != priority:
                    continue
                if used + toks <= target_tokens:
                    result.append(line)
                    used += toks
                elif priority == 2 and result:
                    remaining = target_tokens - used
                    words = line.split()
                    keep = max(1, int(remaining / 1.3))
                    result.append(" ".join(words[:keep]) + "…")
                    used = target_tokens
                    break
        return "\n".join(result) or original_prompt[:target_tokens * 4]

    @staticmethod
    def _score_line(line: str) -> int:
        ll = line.lower()
        if re.search(
            r"tool\s*:|use\s*:|call\s*:|must|required|^\d+\.\s+|"
            r"goal|objective|step\s*[:#]",
            ll,
        ):
            return 2
        if re.search(r"example|note|hint|priority|recommend", ll):
            return 1
        return 0

    async def adapt_to_memory_pressure(self) -> None:
        """Detect high VRAM / RAM usage and log a warning."""
        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, self._sync_client.metrics
            )
            vram_pct = data.get("vram", {}).get("utilization_pct", 0.0)
            ram_pct = data.get("system", {}).get("ram_pct", 0.0)
            if vram_pct > 95:
                log.warning("VRAM pressure %.1f%% – server may slow down", vram_pct)
            if ram_pct > 90:
                log.warning(
                    "RAM pressure %.1f%% – consider clearing context cache", ram_pct
                )
        except Exception:
            pass

    def reduce_context_window(self, factor: float = 0.85) -> None:
        with self._lock:
            self._context_window = max(512, int(self._context_window * factor))
        log.info("Context window reduced to %d tokens", self._context_window)

    async def attempt_context_recovery(self) -> None:
        with self._lock:
            if self._context_window < self._initial_ctx:
                self._context_window = min(
                    self._initial_ctx, int(self._context_window * 1.05)
                )
                log.debug("Context window recovered to %d", self._context_window)

    def get_full_report(self) -> Dict[str, Any]:
        try:
            sys_data = self._sync_client.system_info()
        except Exception:
            sys_data = {}
        return {
            "system": {
                "server": self.cfg.base_url,
                "model_id": self.cfg.model,
                **sys_data,
            },
            "context_window": self._context_window,
            "timestamp": time.time(),
        }

    @property
    def context_window(self) -> int:
        with self._lock:
            return self._context_window

    async def close(self) -> None:
        try:
            await self._async_client.close()
        except Exception:
            pass
        try:
            self._sync_client.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------------------
_SINGLETON_CFG: Optional[ClientConfig] = None


def get_model_manager(config: Optional[ClientConfig] = None) -> AdaptiveContextManager:
    """
    Return the global AdaptiveContextManager singleton.
    Thread‑safe lazy initialisation.
    """
    global _SINGLETON_CFG
    if config is not None:
        _SINGLETON_CFG = config
    return AdaptiveContextManager(config or _SINGLETON_CFG)


# ------------------------------------------------------------------------------
# LocalLLMCloudAdapter – drop‑in for ai_agent.py BaseLLMAdapter
# ------------------------------------------------------------------------------
class LocalLLMCloudAdapter:
    """
    Adapter wrapping Local LLM Cloud in the exact interface expected by
    ai_agent.py (BaseLLMAdapter).  Also usable standalone.

    Features:
    - Health checks and automatic model readiness wait on initialisation.
    - Synchronous and asynchronous generation.
    - Full‑response metadata (generate_full).
    - Context‑window adaptation and prompt compression.
    - Circuit breaker reset on re‑initialisation.
    """

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self.config = config or ClientConfig()
        self._ensure_defaults()
        self._sync_client = LLMCloudClient(self.config)
        self._async_client = AsyncLLMCloudClient(self.config)
        self._initialized = False
        self._model_loaded = False
        self._shutting_down = False
        self._last_used: Optional[float] = None
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._total_tokens = 0
        self.context_manager = get_model_manager(self.config)
        # Override circuit breaker reset method (public)
        self._async_client.reset_circuit_breaker = self._reset_cb  # type: ignore[attr-defined]

    def _ensure_defaults(self) -> None:
        defaults = {
            "health_check_retries": 3,
            "health_check_retry_delay_s": 2.0,
            "wait_for_model": False,
            "wait_timeout_s": 300.0,
        }
        for attr, default in defaults.items():
            if not hasattr(self.config, attr):
                setattr(self.config, attr, default)

    def _reset_cb(self) -> None:
        """Reset the async client's internal circuit breaker state."""
        self._async_client._cb_failures = 0
        self._async_client._cb_state = "closed"

    @classmethod
    def from_model_config(cls, model_config: Any) -> "LocalLLMCloudAdapter":
        """Build an adapter from an ai_agent ModelConfig object."""
        base_url = getattr(model_config, "base_url", None) or "http://localhost:8080"
        api_key = getattr(model_config, "api_key", None)
        model_id = getattr(model_config, "model_name", "qwen3-8b")
        max_tokens = getattr(model_config, "max_tokens", 0) or 0
        temperature = getattr(model_config, "temperature", 0.7)
        timeout_s = float(getattr(model_config, "timeout", 300) or 300)
        cfg = ClientConfig(
            base_url=base_url,
            api_key=api_key or "",
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_s,
        )
        return cls(config=cfg)

    async def initialize(self) -> bool:
        """
        Connect to the server, verify health, and optionally wait for a model
        to be loaded.

        Returns True if ready to serve inference requests.
        """
        with self._lock:
            if self._initialized:
                return True

        # Reset circuit breaker before trying fresh connections
        self._reset_cb()

        healthy = False
        retries = getattr(self.config, "health_check_retries", 3)
        delay = getattr(self.config, "health_check_retry_delay_s", 2.0)
        for attempt in range(retries):
            try:
                healthy = await self._async_client.is_healthy()
                if healthy:
                    log.info(
                        "✅ Server healthy at %s (attempt %d)",
                        self.config.base_url,
                        attempt + 1,
                    )
                    break
                log.warning("Server not healthy (attempt %d)", attempt + 1)
            except Exception as exc:
                log.warning("Health check failed: %s (attempt %d)", exc, attempt + 1)
            if attempt < retries - 1:
                await asyncio.sleep(delay)

        if not healthy:
            log.error(
                "❌ Server not healthy after %d retries", retries
            )
            return False

        model_ready = False
        try:
            model_ready = await self._async_client.ready()
            if model_ready:
                log.info("✅ Model loaded and ready")
            else:
                log.warning("Model not ready – server may be autoloading")
        except ModelNotLoadedError:
            model_ready = False
            log.warning("Model not loaded on server")
        except Exception as exc:
            log.debug(
                "Could not check /ready: %s – assuming model may load later", exc
            )
            model_ready = True

        wait_for_model = getattr(self.config, "wait_for_model", False)
        if not model_ready and wait_for_model:
            timeout = getattr(self.config, "wait_timeout_s", 300.0)
            log.info(
                "Waiting for model to load (timeout: %.0fs)", timeout
            )
            start = time.time()
            while time.time() - start < timeout:
                try:
                    if await self._async_client.ready():
                        log.info("✅ Model now ready")
                        model_ready = True
                        break
                except ModelNotLoadedError:
                    pass
                except Exception:
                    pass
                await asyncio.sleep(1.0)
            if not model_ready:
                log.error(
                    "Model not ready after waiting %.0fs", timeout
                )
                return False

        with self._lock:
            self._initialized = True
            self._model_loaded = model_ready
            self._last_used = time.time()
        log.info(
            "✅ LocalLLMCloudAdapter initialized → %s [model=%s]",
            self.config.base_url,
            self.config.model,
        )
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a complete response for a prompt.

        Returns the assistant's text content.
        """
        if self._shutting_down:
            return "ERROR: Adapter is shutting down"
        if not self._initialized:
            ok = await self.initialize()
            if not ok:
                return f"ERROR: Could not connect to {self.config.base_url}"
        try:
            self._last_used = time.time()
            result = await self._async_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                system_prompt=system_prompt,
            )
            with self._lock:
                self._total_requests += 1
                self._total_tokens += len(result.split())
            return result
        except ModelNotLoadedError:
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return "ERROR: No model loaded on server – load a model first"
        except LLMCloudError as exc:
            log.error("LocalLLMCloudAdapter.generate error: %s", exc)
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return f"ERROR: {exc}"
        except Exception as exc:
            log.error(
                "Unexpected error in generate: %s", exc, exc_info=True
            )
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return f"ERROR: {exc}"

    async def generate_full(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a response and return a GenerationResult with full metadata
        (tokens, timing, finish reason, etc.).
        """
        if self._shutting_down:
            return GenerationResult(
                text="ERROR: Adapter is shutting down",
                finish_reason="error",
                error="Adapter shutting down",
            )
        if not self._initialized:
            ok = await self.initialize()
            if not ok:
                return GenerationResult(
                    text=f"ERROR: Could not connect to {self.config.base_url}",
                    finish_reason="error",
                    error="Connection failed",
                )
        try:
            self._last_used = time.time()
            # Use the raw chat_completions to get the full response dict
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = await self._async_client.chat_completions(
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                timeout=timeout,
            )
            result = GenerationResult.from_chat_response(
                resp, server_url=self.config.base_url
            )
            with self._lock:
                self._total_requests += 1
                self._total_tokens += result.tokens_generated
            return result
        except ModelNotLoadedError:
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return GenerationResult(
                text="ERROR: No model loaded on server – load a model first",
                finish_reason="error",
                error="Model not loaded",
            )
        except LLMCloudError as exc:
            log.error("generate_full error: %s", exc)
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return GenerationResult(
                text=f"ERROR: {exc}",
                finish_reason="error",
                error=str(exc),
            )
        except Exception as exc:
            log.error("Unexpected error in generate_full: %s", exc, exc_info=True)
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return GenerationResult(
                text=f"ERROR: {exc}",
                finish_reason="error",
                error=str(exc),
            )

    async def generate_with_adaptation(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_degradation: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with automatic prompt compression and timeout tuning.
        Returns (text, metadata).
        """
        meta: Dict[str, Any] = {
            "original_strategy": "direct",
            "strategy_degraded": False,
            "compression_applied": False,
            "attempt_number": 1,
            "max_attempts": 3,
            "ttft_ms": 0.0,
            "tps": 0.0,
        }
        for attempt in range(1, meta["max_attempts"] + 1):
            meta["attempt_number"] = attempt
            try:
                await self.context_manager.adapt_to_memory_pressure()
                budget = self.context_manager.calculate_token_budget(
                    prompt_text=prompt,
                    max_generation_tokens=max_tokens
                    or self.config.max_tokens
                    or 512,
                )
                prompt_to_use = prompt
                use_max_tokens = max_tokens
                if not budget.is_feasible and allow_degradation:
                    target = int(budget.total_budget * 0.6)
                    prompt_to_use = self.context_manager.adapt_prompt_to_context(
                        prompt, target
                    )
                    meta["compression_applied"] = True
                    use_max_tokens = budget.recommended_max_gen
                    log.info(
                        "✂️ Prompt compressed to %d tokens for attempt %d",
                        target,
                        attempt,
                    )
                use_timeout = timeout or budget.recommended_timeout

                # Use generate_full to capture metrics
                gen_result = await self.generate_full(
                    prompt_to_use,
                    max_tokens=use_max_tokens,
                    temperature=temperature,
                    timeout=use_timeout,
                )
                text = gen_result.text
                meta["ttft_ms"] = gen_result.ttft_ms
                meta["tps"] = gen_result.tps

                if gen_result.success:
                    await self.context_manager.attempt_context_recovery()
                    return text, meta

                # Generation failed
                if attempt < meta["max_attempts"] and allow_degradation:
                    await asyncio.sleep(1.0)
                else:
                    return text, meta

            except asyncio.TimeoutError:
                self.context_manager.reduce_context_window()
                if attempt < meta["max_attempts"] and allow_degradation:
                    meta["strategy_degraded"] = True
                    await asyncio.sleep(2.0)
                else:
                    return f"ERROR: Timeout after {attempt} attempts", meta
            except Exception as exc:
                if attempt < meta["max_attempts"]:
                    await asyncio.sleep(1.0)
                else:
                    return f"ERROR: {exc}", meta

        return "ERROR: Exhausted all adaptation attempts", meta

    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> List[str]:
        """Generate responses for a list of prompts sequentially."""
        results = []
        for prompt in prompts:
            try:
                res = await self.generate(
                    prompt, max_tokens, temperature, timeout
                )
                results.append(res)
            except Exception as exc:
                results.append(f"ERROR: {exc}")
        return results

    def stream_sync(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """Synchronous streaming generator (word‑buffered)."""
        yield from self._sync_client.stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
        )

    def is_healthy(self) -> bool:
        """Return True if the server is reachable and responsive."""
        try:
            return self._sync_client.is_healthy()
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return adapter usage counters and server metrics snapshot."""
        base = {
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "last_used": self._last_used,
            "provider": "local_llm_cloud",
            "model_name": self.config.model,
            "base_url": self.config.base_url,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(
                self._total_errors / max(self._total_requests, 1), 4
            ),
        }
        try:
            base["server"] = self._sync_client.metrics()
        except Exception:
            pass
        return base

    def set_shutdown_flag(self) -> None:
        """Signal that the adapter is shutting down."""
        self._shutting_down = True

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Release all resources (connections, pools)."""
        self._shutting_down = True
        try:
            self._sync_client.close()
        except Exception:
            pass
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._async_client.close())
                asyncio.ensure_future(self.context_manager.close())
            else:
                loop.run_until_complete(self._async_client.close())
                loop.run_until_complete(self.context_manager.close())
        except Exception:
            pass
        log.info("LocalLLMCloudAdapter shutdown complete")
        return True

    def count_tokens(self, text: str) -> int:
        """Fast token count estimate (uses C accelerator if available)."""
        return _estimate_tokens(text)
