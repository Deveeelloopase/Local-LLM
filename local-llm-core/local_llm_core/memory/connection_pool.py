from __future__ import annotations

"""
memory/connection_pool.py — Production‑Ready SQLite Connection Pool
===================================================================
v1.0.0 – Config injected via constructor; no filesystem reads.

Async SQLite connection pool using aiosqlite with:
- Configurable pool size and timeouts
- Proper async context manager
- Connection health checks
- Graceful shutdown

All parameters are taken from the `config` dict (or a `SQLiteConfig`
model) or explicit constructor arguments. Sensible defaults are used when
config is omitted.
"""

import asyncio
import logging
from typing import Any, AsyncContextManager, Dict, Optional

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SimpleConnectionPool
# -----------------------------------------------------------------------------
class SimpleConnectionPool:
    """
    Async SQLite connection pool using asyncio.Semaphore.

    Usage:
        pool = SimpleConnectionPool()
        await pool.initialize()
        async with pool.acquire() as conn:
            cursor = await conn.execute("SELECT * FROM table")
            rows = await cursor.fetchall()
        await pool.shutdown()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_connections: Optional[int] = None,
        connection_timeout: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ---- normalise config ------------------------------------------------
        cfg: Dict[str, Any] = {}
        if config:
            try:
                from local_llm_core.memory.config_loader import SQLiteConfig
                if isinstance(config, dict):
                    # Accept either a plain dict or the full memory section
                    section_cfg = config.get("memory", config)
                    cfg_model = SQLiteConfig(**section_cfg)
                else:
                    cfg_model = SQLiteConfig(**config.get("memory", config))
                cfg = cfg_model.model_dump()
            except Exception as exc:
                log.warning("SQLite config validation failed: %s; using defaults", exc)
                cfg = config.get("memory", config) if isinstance(config, dict) else {}
        # ----------------------------------------------------------------------

        self._db_path = db_path if db_path is not None else cfg.get("db_path", "/tmp/llm_keys.db")
        self._max_connections = max_connections if max_connections is not None else cfg.get("max_connections", 10)
        self._connection_timeout = connection_timeout if connection_timeout is not None else cfg.get("connection_timeout_s", 5.0)

        self._semaphore = asyncio.Semaphore(self._max_connections)
        self._aiosqlite = None
        self._initialized = False
        self._shutdown = False
        self._active_connections = 0
        self._lock = asyncio.Lock()

        log.info(
            "SimpleConnectionPool: db=%s max_conn=%d timeout=%.1fs",
            self._db_path,
            self._max_connections,
            self._connection_timeout,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def initialize(self) -> None:
        """Initialize the pool and import aiosqlite."""
        async with self._lock:
            if self._initialized:
                return
            try:
                import aiosqlite
                self._aiosqlite = aiosqlite
                self._initialized = True
                log.debug("SimpleConnectionPool initialized")
            except ImportError as e:
                log.error("aiosqlite not installed: %s", e)
                raise ImportError("aiosqlite required for SQLite connection pool") from e

    async def shutdown(self) -> None:
        """Cleanly shutdown the pool."""
        async with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._initialized = False
        log.debug("SimpleConnectionPool shutdown")

    # -------------------------------------------------------------------------
    # Connection Acquisition
    # -------------------------------------------------------------------------
    class _ConnContextManager:
        """Proper async context manager for database connections."""

        def __init__(self, pool: "SimpleConnectionPool", conn) -> None:
            self.pool = pool
            self.conn = conn
            self._released = False

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
            if self._released:
                log.warning("Connection already released (double-exit?)")
                return False

            try:
                if self.conn:
                    try:
                        await asyncio.wait_for(self.conn.close(), timeout=2.0)
                    except asyncio.TimeoutError:
                        log.warning("Connection close timed out")
                    except Exception as e:
                        log.warning("Error closing connection: %s", e)
            finally:
                try:
                    async with self.pool._lock:
                        self.pool._active_connections = max(
                            0, self.pool._active_connections - 1
                        )
                    self.pool._semaphore.release()
                except Exception as e:
                    log.error("Error releasing connection resources: %s", e)
                finally:
                    self._released = True

            return False  # Don't suppress exceptions

    async def acquire(self) -> AsyncContextManager:
        """
        Acquire a connection from the pool.

        Returns:
            An async context manager that yields the connection.
        """
        if self._shutdown:
            raise RuntimeError("Connection pool is shut down")

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._connection_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out after {self._connection_timeout}s waiting for connection"
            )

        try:
            if not self._initialized:
                await self.initialize()

            conn = await self._aiosqlite.connect(self._db_path)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-2000")
            await conn.execute("PRAGMA temp_store=MEMORY")

            async with self._lock:
                self._active_connections += 1

            return self._ConnContextManager(self, conn)
        except Exception as e:
            self._semaphore.release()
            log.error("Failed to acquire connection: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Check if the pool and database are healthy."""
        if not self._initialized or self._shutdown:
            return False
        try:
            async with self.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            log.warning("Health check failed: %s", e)
            return False

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "db_path": self._db_path,
            "max_connections": self._max_connections,
            "active_connections": self._active_connections,
            "available_slots": self._max_connections - self._active_connections,
            "initialized": self._initialized,
            "shutdown": self._shutdown,
        }
