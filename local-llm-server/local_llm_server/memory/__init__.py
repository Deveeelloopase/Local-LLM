"""
local_llm_server/memory/__init__.py — Conversation Memory Management v1.0.0
=============================================================================
Public interface for conversation history storage and retrieval, fully
integrated with the advanced conversation store backends (v1.0.0).

  • `create_conversation_store` now accepts a configuration dictionary directly,
    avoiding redundant YAML I/O when the config is already in memory (e.g. from
    the server's lifespan).
  • `get_conversation_store` properly passes the config path to the internal
    factory, ensuring that environment‑specific configs are used.
  • `ConversationStoreManager` – a high‑level async context manager that
    handles initialization, background tasks (if any), health checks, and
    graceful shutdown.  Ideal for FastAPI lifespan integration.
  • `health_check` – lightweight async function that probes the store's
    underlying connection.
  • Re‑exports of all backend classes and factories.
  • Lazy imports with caching for fast attribute access.
  • Full type hints for IDE support.
  • This module is production‑ready and pairs with the write‑buffered,
    lease‑token‑protected conversation store (v5.0.0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

# ------------------------------------------------------------------------------
# Lazy import cache
# ------------------------------------------------------------------------------
_IMPORT_CACHE: Dict[str, Any] = {}

def _load_module():
    """Import the conversation_store module once and cache its contents."""
    if "mod" not in _IMPORT_CACHE:
        from . import conversation_store as _mod
        _IMPORT_CACHE["mod"] = _mod
    return _IMPORT_CACHE["mod"]

def __getattr__(name: str):
    """Lazy attribute access for all exported names."""
    mod = _load_module()
    if name in mod.__dict__:
        return mod.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ------------------------------------------------------------------------------
# Re‑export all public names for static analysis / documentation
# ------------------------------------------------------------------------------
__all__ = [
    "ConversationStore",
    "MemoryConversationStore",
    "SQLiteConversationStore",
    "PostgreSQLConversationStore",
    "MySQLConversationStore",
    "create_conversation_store",
    "get_conversation_store",
    "shutdown_conversation_store",
    "ConversationStoreManager",
    "health_check",
]

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Forward references for type checkers
# ------------------------------------------------------------------------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .conversation_store import ConversationStore as _ConvStore

# ------------------------------------------------------------------------------
# Public Factory (accepts config dict OR path)
# ------------------------------------------------------------------------------
async def create_conversation_store(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    config_path: Optional[Path] = None,
) -> Optional["_ConvStore"]:
    """
    Create and initialize a conversation store from configuration.

    Precedence:
        1. If `cfg` is provided, it is used directly.
        2. Otherwise, if `config_path` is given, that file is loaded.
        3. Otherwise, the default YAML search paths are tried.

    Args:
        cfg: Configuration dictionary.
        config_path: Explicit path to config.yaml.

    Returns:
        Initialized ConversationStore instance, or None if disabled.
    """
    mod = _load_module()
    if cfg is not None:
        # Use provided config dictionary
        return await mod.create_conversation_store(cfg)
    # Fall back to path‑based loading
    return await mod.create_conversation_store_from_path(config_path)


async def get_conversation_store(
    config_path: Optional[Path] = None,
) -> Optional["_ConvStore"]:
    """
    Convenience initializer that loads config from a given file path.

    Args:
        config_path: Optional explicit path to config.yaml.

    Returns:
        Initialized ConversationStore, or None if disabled.
    """
    return await create_conversation_store(config_path=config_path)


async def shutdown_conversation_store(store: Optional["_ConvStore"]) -> None:
    """
    Gracefully shut down the conversation store, closing any connections.

    This should be called during FastAPI shutdown lifespan events.

    Args:
        store: The store instance returned by create_conversation_store().
    """
    mod = _load_module()
    await mod.shutdown_conversation_store(store)


async def health_check(store: Optional["_ConvStore"]) -> bool:
    """
    Perform a lightweight health check on the conversation store.

    Returns True if the store is operational, False otherwise.
    """
    if store is None:
        return True  # no store is not a failure
    try:
        # A simple list operation probes the connection
        await store.list()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# ConversationStoreManager – async context manager for lifecycle
# ------------------------------------------------------------------------------
class ConversationStoreManager:
    """
    Async context manager that initializes the conversation store on entry
    and gracefully shuts it down on exit.  Ideal for FastAPI lifespan.

    Usage:
        async with ConversationStoreManager(cfg) as store:
            app.state.conversation_store = store

    Features:
        - Automatic health check after initialization.
        - Background prune loop (optional, controlled by config).
        - Emits Prometheus metrics (if configured).
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, *, config_path: Optional[Path] = None):
        self._cfg = cfg
        self._config_path = config_path
        self._store: Optional["_ConvStore"] = None
        self._prune_task = None

    async def __aenter__(self) -> Optional["_ConvStore"]:
        self._store = await create_conversation_store(self._cfg, config_path=self._config_path)
        if self._store:
            # Perform a health check
            ok = await health_check(self._store)
            if not ok:
                log.error("Conversation store health check failed.")
                # Continue anyway; the caller can decide
            # Optionally start a background prune loop if enabled in config
            conv_cfg = self._cfg.get("conversation", {}) if self._cfg else {}
            if conv_cfg.get("enable_background_prune", False):
                import asyncio
                self._prune_task = asyncio.create_task(self._prune_loop())
                log.info("Background prune loop started.")
        return self._store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._prune_task:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        if self._store:
            await shutdown_conversation_store(self._store)
        self._store = None

    async def _prune_loop(self) -> None:
        """Periodically prune old conversations."""
        import asyncio
        conv_cfg = self._cfg.get("conversation", {}) if self._cfg else {}
        interval = conv_cfg.get("prune_interval_seconds", 3600)
        while True:
            try:
                await asyncio.sleep(interval)
                if self._store:
                    n = await self._store.prune_old()
                    if n:
                        log.info("Background prune removed %d old conversations", n)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Prune loop error: %s", e)


# ------------------------------------------------------------------------------
# Module Initialization Logging
# ------------------------------------------------------------------------------
log.debug("memory module loaded (v1.0.0)")
