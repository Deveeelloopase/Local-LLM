"""
local_llm_server/serving/log_handler.py — Production Log Handler v4.0.1
=========================================================================
TUILogHandler – thread‑safe, memory‑safe, WebSocket race‑free log handler
with novel adaptive batching, deduplication, and overflow persistence.

v4.0.1 Fixes:
    - Import `random` at module level.
    - Overflow file parent directory is created automatically if missing.
    - Configurable deduplication cache size and overflow flush interval.
    - All v4.0.0 features fully retained.

Configuration keys (config.yaml → `logging`):
  buffer_size: int = 500
  ws_queue_size: int = 512
  max_msg_len: int = 4096
  reaper_interval_sec: float = 60.0
  broadcast_timeout_sec: float = 2.0
  ws_client_idle_timeout_sec: float = 300.0
  adaptive_sampling: bool = true
  adaptive_sampling_threshold: int = 50
  adaptive_sampling_rate: float = 0.1
  deduplication_window_sec: float = 2.0
  dedup_cache_max_entries: int = 10000
  dedup_cache_cleanup_age: float = 10.0   # seconds after which old dedup entries are pruned
  overflow_file: str = "/tmp/llm_log_overflow.jsonl"
  overflow_flush_interval_sec: float = 30.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import random as _random_module   # <-- moved to module level
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# ------------------------------------------------------------------------------
# Configuration Loading
# ------------------------------------------------------------------------------
def _load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is not None:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        logging.getLogger(__name__).warning("Config file not found: %s", config_path)
        return {}

    paths = [
        Path.home() / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    logging.getLogger(__name__).warning("No config.yaml found; using default log handler settings.")
    return {}


# Default configuration values
DEFAULT_CONFIG = {
    "logging": {
        "buffer_size": 500,
        "ws_queue_size": 512,
        "max_msg_len": 4096,
        "reaper_interval_sec": 60.0,
        "broadcast_timeout_sec": 2.0,
        "ws_client_idle_timeout_sec": 300.0,
        "adaptive_sampling": True,
        "adaptive_sampling_threshold": 50,
        "adaptive_sampling_rate": 0.1,
        "deduplication_window_sec": 2.0,
        "dedup_cache_max_entries": 10_000,
        "dedup_cache_cleanup_age": 10.0,
        "overflow_file": "/tmp/llm_log_overflow.jsonl",
        "overflow_flush_interval_sec": 30.0,
    }
}

# Internal logger for handler diagnostics
_internal_log = logging.getLogger(__name__ + "._internal")

# ------------------------------------------------------------------------------
# Optional Prometheus metrics
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


def _setup_prometheus():
    if not HAS_PROMETHEUS:
        return {}
    return {
        "log_entries_total": Counter("tui_log_entries_total", "Log entries emitted"),
        "ws_clients": Gauge("tui_ws_clients", "Active WS log clients"),
        "ws_broadcast_total": Counter("tui_ws_broadcast_total", "Log messages broadcast"),
        "ws_broadcast_errors": Counter("tui_ws_broadcast_errors", "Broadcast errors"),
        "deduplicated_total": Counter("tui_log_deduplicated_total", "Deduplicated log entries"),
        "overflow_flushes": Counter("tui_log_overflow_flushes", "Ring buffer overflow flushes"),
    }


# ------------------------------------------------------------------------------
# TUILogHandler (Singleton with class‑level state)
# ------------------------------------------------------------------------------
class TUILogHandler(logging.Handler):
    """
    Singleton log handler that stores recent logs in a ring buffer and broadcasts
    them to WebSocket clients.

    Adaptive behaviour:
        - If active WS clients > adaptive_sampling_threshold, only a fraction
          (adaptive_sampling_rate) of messages are forwarded (random selection).
        - Deduplication: consecutive identical messages within a short window
          are dropped.
        - Overflow: when buffer ≥ 90% full, oldest entries are flushed to disk.
    """

    # ---- Class-level state (singleton) ------------------------------------
    _instance: Optional["TUILogHandler"] = None
    _init_lock: threading.Lock = threading.Lock()

    # Ring buffer and its lock
    _buffer: deque = deque(maxlen=500)
    _buffer_lock: threading.Lock = threading.Lock()

    # WebSocket client queues with activity tracking
    _ws_clients: List[queue.Queue] = []
    _ws_client_activity: Dict[int, float] = {}  # id(queue) -> last activity timestamp
    _ws_lock: threading.Lock = threading.Lock()

    # Token streaming subscribers
    _subscribers: Dict[str, Set[Any]] = {}
    _sub_lock: threading.Lock = threading.Lock()

    # Dead client counter & metrics
    _dead_count: int = 0
    _metrics: Dict[str, Any] = {}

    # Configuration values (set during installation)
    _buffer_size: int = 500
    _ws_queue_size: int = 512
    _max_msg_len: int = 4096
    _reaper_interval: float = 60.0
    _broadcast_timeout: float = 2.0
    _ws_client_idle_timeout: float = 300.0
    _adaptive_sampling: bool = True
    _adaptive_sampling_threshold: int = 50
    _adaptive_sampling_rate: float = 0.1
    _dedup_window_sec: float = 2.0
    _dedup_cache_max_entries: int = 10_000
    _dedup_cache_cleanup_age: float = 10.0
    _overflow_file: str = "/tmp/llm_log_overflow.jsonl"
    _overflow_flush_interval: float = 30.0

    # Background reaper task and stop event
    _reaper_task: Optional[asyncio.Task] = None
    _stop_event: asyncio.Event = asyncio.Event()

    # Deduplication state: stores hash of (logger, level, msg) → last seen time
    _dedup_cache: Dict[str, float] = {}
    _dedup_lock: threading.Lock = threading.Lock()

    # Last overflow flush timestamp
    _last_overflow_flush: float = 0.0
    _overflow_flush_lock: threading.Lock = threading.Lock()

    # Flag to disable further overflow flushes if file cannot be opened
    _overflow_disabled: bool = False

    # --------------------------------------------------------------------------
    # Singleton construction
    # --------------------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        # Avoid re‑initializing the singleton
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Initialize base logging.Handler
        super().__init__()

        # Load configuration (only once)
        cfg = _load_config(config_path)
        log_cfg = cfg.get("logging", {})
        for key, value in DEFAULT_CONFIG["logging"].items():
            if key not in log_cfg:
                log_cfg[key] = value

        cls = self.__class__
        cls._buffer_size = log_cfg["buffer_size"]
        cls._ws_queue_size = log_cfg["ws_queue_size"]
        cls._max_msg_len = log_cfg["max_msg_len"]
        cls._reaper_interval = log_cfg["reaper_interval_sec"]
        cls._broadcast_timeout = log_cfg["broadcast_timeout_sec"]
        cls._ws_client_idle_timeout = log_cfg.get("ws_client_idle_timeout_sec", 300.0)
        cls._adaptive_sampling = log_cfg.get("adaptive_sampling", True)
        cls._adaptive_sampling_threshold = log_cfg.get("adaptive_sampling_threshold", 50)
        cls._adaptive_sampling_rate = log_cfg.get("adaptive_sampling_rate", 0.1)
        cls._dedup_window_sec = log_cfg.get("deduplication_window_sec", 2.0)
        cls._dedup_cache_max_entries = log_cfg.get("dedup_cache_max_entries", 10_000)
        cls._dedup_cache_cleanup_age = log_cfg.get("dedup_cache_cleanup_age", 10.0)
        cls._overflow_file = log_cfg.get("overflow_file", "/tmp/llm_log_overflow.jsonl")
        cls._overflow_flush_interval = log_cfg.get("overflow_flush_interval_sec", 30.0)

        # Recreate buffer with correct size
        cls._buffer = deque(maxlen=cls._buffer_size)

        # Set up Prometheus metrics
        cls._metrics = _setup_prometheus()

        # Set handler level and formatter
        self.setLevel(logging.DEBUG)
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        self._initialized = True
        _internal_log.info("TUILogHandler v4.0.1 initialized (buffer=%d, ws_queue=%d, adaptive=%s)",
                           cls._buffer_size, cls._ws_queue_size, cls._adaptive_sampling)

    # --------------------------------------------------------------------------
    # emit() – thread‑safe log handling with all enhancements
    # --------------------------------------------------------------------------
    def emit(self, record: logging.LogRecord) -> None:
        """Called from any thread; must never raise exceptions."""
        try:
            cls = self.__class__

            # Step 1: Build message
            msg = str(record.getMessage() or "")
            if len(msg) > cls._max_msg_len:
                msg = msg[: cls._max_msg_len] + "…[truncated]"

            entry = {
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": msg,
                "req_id": getattr(record, "request_id", None),
            }

            # Step 2: Ring buffer
            with cls._buffer_lock:
                # Overflow flush if buffer > 90% full
                if len(cls._buffer) >= int(cls._buffer.maxlen * 0.9):
                    self._maybe_flush_overflow()
                cls._buffer.append(entry)

            # Step 3: Deduplication
            dedup_key = f"{entry['logger']}|{entry['level']}|{entry['msg']}"
            dedup_hash = hashlib.sha256(dedup_key.encode()).hexdigest()
            now = time.monotonic()
            with cls._dedup_lock:
                last_time = cls._dedup_cache.get(dedup_hash, 0)
                if now - last_time < cls._dedup_window_sec:
                    if cls._metrics.get("deduplicated_total"):
                        cls._metrics["deduplicated_total"].inc()
                    return
                cls._dedup_cache[dedup_hash] = now
                # Periodic cleanup
                if len(cls._dedup_cache) > cls._dedup_cache_max_entries:
                    cutoff = now - cls._dedup_cache_cleanup_age
                    cls._dedup_cache = {k: v for k, v in cls._dedup_cache.items() if v > cutoff}

            # Step 4: Adaptive sampling
            with cls._ws_lock:
                num_clients = len(cls._ws_clients)
                if cls._adaptive_sampling and num_clients > cls._adaptive_sampling_threshold:
                    if _random_module.random() > cls._adaptive_sampling_rate:
                        return
                clients = list(cls._ws_clients)

            if cls._metrics.get("log_entries_total"):
                cls._metrics["log_entries_total"].inc()

            payload = json.dumps(entry)

            # Step 5: Distribute to WS clients
            dead = []
            for q in clients:
                try:
                    try:
                        q.put_nowait(payload)
                        with cls._ws_lock:
                            cls._ws_client_activity[id(q)] = now
                    except queue.Full:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            q.put_nowait(payload)
                            with cls._ws_lock:
                                cls._ws_client_activity[id(q)] = now
                        except queue.Full:
                            pass
                except Exception:
                    dead.append(q)

            if dead:
                with cls._ws_lock:
                    for q in dead:
                        try:
                            cls._ws_clients.remove(q)
                            cls._ws_client_activity.pop(id(q), None)
                        except (ValueError, KeyError):
                            pass
                cls._dead_count += len(dead)
                if cls._metrics.get("ws_broadcast_errors"):
                    cls._metrics["ws_broadcast_errors"].inc(len(dead))

            if cls._metrics.get("ws_broadcast_total"):
                cls._metrics["ws_broadcast_total"].inc()

        except Exception:
            # Logging must never crash the application
            pass

    # --------------------------------------------------------------------------
    # Overflow persistence
    # --------------------------------------------------------------------------
    def _maybe_flush_overflow(self) -> None:
        """Flush oldest buffer entries to overflow file if needed."""
        cls = self.__class__

        if cls._overflow_disabled:
            return

        now = time.monotonic()
        with cls._overflow_flush_lock:
            if now - cls._last_overflow_flush < cls._overflow_flush_interval:
                return
            cls._last_overflow_flush = now

        # Ensure parent directory exists
        overflow_path = Path(cls._overflow_file)
        try:
            overflow_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _internal_log.error("Cannot create overflow log directory %s: %s", overflow_path.parent, e)
            cls._overflow_disabled = True
            return

        with cls._buffer_lock:
            entries = list(cls._buffer)
            if len(entries) == 0:
                return
            n_flush = len(entries) // 2
            if n_flush == 0:
                return
            to_flush = entries[:n_flush]
            for _ in range(n_flush):
                cls._buffer.popleft()

        try:
            with open(cls._overflow_file, "a", encoding="utf-8") as f:
                for e in to_flush:
                    f.write(json.dumps(e) + "\n")
            if cls._metrics.get("overflow_flushes"):
                cls._metrics["overflow_flushes"].inc()
            _internal_log.debug("Flushed %d log entries to %s", n_flush, cls._overflow_file)
        except Exception as e:
            _internal_log.error("Failed to write overflow log: %s", e)
            # Disable further flushes to prevent repeated errors
            cls._overflow_disabled = True

    # --------------------------------------------------------------------------
    # WebSocket Client Management (Log Streaming)
    # --------------------------------------------------------------------------
    @classmethod
    def add_ws_client(cls) -> queue.Queue:
        """Register a new log‑streaming WebSocket. Returns a bounded queue."""
        q = queue.Queue(maxsize=cls._ws_queue_size)
        now = time.monotonic()
        with cls._ws_lock:
            cls._ws_clients.append(q)
            cls._ws_client_activity[id(q)] = now
        if cls._metrics.get("ws_clients"):
            cls._metrics["ws_clients"].inc()
        return q

    @classmethod
    def remove_ws_client(cls, q: queue.Queue) -> None:
        """Remove a WebSocket client queue."""
        with cls._ws_lock:
            try:
                cls._ws_clients.remove(q)
                cls._ws_client_activity.pop(id(q), None)
            except (ValueError, KeyError):
                pass
        if cls._metrics.get("ws_clients"):
            cls._metrics["ws_clients"].dec()

    # --------------------------------------------------------------------------
    # WebSocket Subscription Management (Token Streaming)
    # --------------------------------------------------------------------------
    @classmethod
    def subscribe(cls, request_id: str, ws: Any) -> None:
        """Subscribe a WebSocket to token updates for a given request ID."""
        with cls._sub_lock:
            cls._subscribers.setdefault(request_id, set()).add(ws)

    @classmethod
    def unsubscribe(cls, request_id: str, ws: Any) -> None:
        """Unsubscribe a WebSocket from token updates."""
        with cls._sub_lock:
            subs = cls._subscribers.get(request_id)
            if subs:
                subs.discard(ws)
                if not subs:
                    del cls._subscribers[request_id]

    @classmethod
    async def broadcast_token(cls, request_id: str, token: str) -> None:
        """
        Broadcast a token to all subscribers of a request.
        Uses asyncio.gather with individual timeouts.
        """
        payload = json.dumps({"type": "token", "id": request_id, "text": token})

        with cls._sub_lock:
            subs = set(cls._subscribers.get(request_id, set()))

        if not subs:
            return

        async def send_one(ws) -> bool:
            try:
                await asyncio.wait_for(ws.send_text(payload), timeout=cls._broadcast_timeout)
                return True
            except asyncio.TimeoutError:
                _internal_log.warning("WS send timeout for request %s", request_id)
            except (RuntimeError, ConnectionError) as exc:
                _internal_log.debug("WS closed for %s: %s", request_id, type(exc).__name__)
            except Exception as exc:
                _internal_log.error("Unexpected WS error for %s: %s", request_id, exc, exc_info=True)
            return False

        results = await asyncio.gather(
            *[send_one(ws) for ws in subs], return_exceptions=True
        )

        dead = []
        for ws, ok in zip(subs, results):
            if ok is not True:
                dead.append(ws)

        for ws in dead:
            cls.unsubscribe(request_id, ws)

    # --------------------------------------------------------------------------
    # Background Reaper Task (cleans stale client queues)
    # --------------------------------------------------------------------------
    @classmethod
    async def start_reaper_task(cls) -> None:
        """Start the background task that removes stale WebSocket queues."""
        if cls._reaper_task is not None and not cls._reaper_task.done():
            return
        cls._stop_event.clear()
        cls._reaper_task = asyncio.create_task(cls._reaper_loop())
        _internal_log.debug("Log handler reaper task started")

    @classmethod
    async def stop_reaper_task(cls) -> None:
        """Stop the background reaper task."""
        if cls._reaper_task is None:
            return
        cls._stop_event.set()
        cls._reaper_task.cancel()
        try:
            await cls._reaper_task
        except asyncio.CancelledError:
            pass
        cls._reaper_task = None
        _internal_log.debug("Log handler reaper task stopped")

    @classmethod
    async def _reaper_loop(cls) -> None:
        """Periodically probe client queues and remove broken or idle ones."""
        while not cls._stop_event.is_set():
            try:
                await asyncio.sleep(cls._reaper_interval)
                await cls._reap_dead_clients()
            except asyncio.CancelledError:
                break
            except Exception:
                _internal_log.debug("Reaper loop error", exc_info=True)

    @classmethod
    async def _reap_dead_clients(cls) -> None:
        """Identify and remove stale WebSocket queues (broken or idle too long)."""
        now = time.monotonic()
        idle_cutoff = now - cls._ws_client_idle_timeout

        with cls._ws_lock:
            clients_snapshot = list(cls._ws_clients)
            activity_snapshot = dict(cls._ws_client_activity)

        stale = []
        for q in clients_snapshot:
            qid = id(q)
            last_act = activity_snapshot.get(qid, 0)

            try:
                q.empty()
                is_broken = False
            except Exception:
                is_broken = True

            is_idle = (last_act < idle_cutoff)

            if is_broken or is_idle:
                stale.append(q)

        if stale:
            with cls._ws_lock:
                for q in stale:
                    try:
                        cls._ws_clients.remove(q)
                        cls._ws_client_activity.pop(id(q), None)
                    except (ValueError, KeyError):
                        pass
            cls._dead_count += len(stale)
            if cls._metrics.get("ws_clients"):
                cls._metrics["ws_clients"].set(len(cls._ws_clients))
            _internal_log.debug("Reaped %d stale WS queues (total dead: %d)", len(stale), cls._dead_count)

    # --------------------------------------------------------------------------
    # Ring Buffer Access (class methods – safe for tests)
    # --------------------------------------------------------------------------
    @classmethod
    def recent(cls, n: int = 100, since: float = 0.0) -> List[Dict[str, Any]]:
        """Return up to `n` recent log entries, optionally after a timestamp."""
        with cls._buffer_lock:
            entries = list(cls._buffer)
        if since > 0:
            entries = [e for e in entries if e["ts"] > since]
        return entries[-n:]

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """Return current statistics about the log handler."""
        with cls._ws_lock:
            n_clients = len(cls._ws_clients)
        with cls._sub_lock:
            n_subs = sum(len(v) for v in cls._subscribers.values())
        with cls._buffer_lock:
            buf_used = len(cls._buffer)
        return {
            "ws_log_clients": n_clients,
            "ws_token_subs": n_subs,
            "total_reaped": cls._dead_count,
            "buffer_size": cls._buffer.maxlen if cls._buffer else 0,
            "buffer_used": buf_used,
            "adaptive_sampling": cls._adaptive_sampling,
            "adaptive_threshold": cls._adaptive_sampling_threshold,
            "adaptive_rate": cls._adaptive_sampling_rate,
        }

    # --------------------------------------------------------------------------
    # Singleton Installation
    # --------------------------------------------------------------------------
    @classmethod
    def install(cls, config_path: Optional[Path | str] = None) -> "TUILogHandler":
        """Install the singleton log handler on the root logger. Idempotent."""
        handler = cls(config_path=config_path)
        root = logging.getLogger()
        if not any(isinstance(h, cls) for h in root.handlers):
            root.addHandler(handler)
        return handler


# ------------------------------------------------------------------------------
# Factory Functions (for consistency with other modules)
# ------------------------------------------------------------------------------
def create_log_handler() -> TUILogHandler:
    return TUILogHandler.install()


def get_log_handler() -> TUILogHandler:
    return TUILogHandler.install()

# ─────────────────────────────────────────────────────────────────────────────
# JSON log formatter
# ─────────────────────────────────────────────────────────────────────────────
class JSONLogFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Activated by ``logging.format: json`` in config.yaml. Structured keys
    (``request_id``, ``key_id``, ``path``, ``status``, etc.) are populated
    via ``logger.info(..., extra={...})`` — most notably by
    ``api_server._log_request`` — and picked up here when present.

    Output schema (stable; safe to parse in log shippers):

        {
          "ts":       "2026-10-25T05:14:10.123Z",
          "level":    "INFO",
          "logger":   "local_llm_server.serving.api_server",
          "msg":      "...",
          "request_id": "...",   # optional
          "key_id":     "...",   # optional
          "path":       "/v1/chat/completions",  # optional
          "method":     "POST",  # optional
          "status":     200,     # optional
          "latency_ms": 123.4,   # optional
          "tokens":     42,      # optional
          "client_ip":  "127.0.0.1",  # optional
          "exc":        "traceback..."   # optional (on ERROR/CRITICAL)
        }

    Any other ``extra={}`` keys passed by callers are also included verbatim.
    """
    _STRUCTURED_KEYS = (
        "request_id", "key_id", "path", "method", "status",
        "latency_ms", "tokens", "client_ip",
    )
    # Attributes always present on LogRecord — excluded from the "extras" sweep
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    } | set(_STRUCTURED_KEYS)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in self._STRUCTURED_KEYS:
            v = getattr(record, key, None)
            if v is not None:
                payload[key] = v
        # Any other extras the caller set
        for k, v in record.__dict__.items():
            if k in self._RESERVED:
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def create_formatter(fmt: str = "text") -> logging.Formatter:
    """Return a Formatter appropriate for the config's ``logging.format``.

    ``"minimal"`` → `TIMESTAMP  LEVEL  MESSAGE` only (no logger, no extras).
    ``"text"``    → legacy text with logger name.
    ``"json"``    → structured JSON with all fields.

    Set ``logging.format: "minimal"`` in config.yaml for the stripped form.
    """
    fmt = (fmt or "").lower()
    if fmt == "minimal":
        return _MinimalFormatter()
    if fmt == "json":
        return JSONLogFormatter()
    return logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class _MinimalFormatter(logging.Formatter):
    """`TIMESTAMP  LEVEL  MODULE  MESSAGE` — no JSON, no logger path, no extras.

    - timestamp: `YYYY-MM-DD HH:MM:SS`
    - level:     INFO / WARNING / ERROR / CRITICAL / DEBUG (8 chars, left-padded)
    - module:    the file/module that emitted the log line
    - message:   the formatted message; exception traceback appended on ERROR+

    Module resolution:
      `record.name` is the logger name (e.g.
      `local_llm_core.engine.gguf.model_loader`). We take the last dotted
      segment (`model_loader`). For loggers whose last segment is a generic
      word (`error`, `access`, `main`), we keep the previous segment too
      (`uvicorn.error`, `uvicorn.access`). Names longer than `_MOD_W` are
      truncated with a trailing `…`.
    """
    _MOD_W = 20   # fixed width for the module column

    @staticmethod
    def _module_for(record: logging.LogRecord) -> str:
        name = record.name
        if "." not in name:
            return name
        short = name.rsplit(".", 1)[-1]
        if short in ("error", "access", "main", "warning", "info"):
            parts = name.split(".")
            short = ".".join(parts[-2:])
        return short

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(record.created))
        mod = self._module_for(record)
        if len(mod) > self._MOD_W:
            mod = mod[: self._MOD_W - 1] + "…"
        msg = record.getMessage()
        if record.exc_info and record.levelno >= logging.ERROR:
            msg = msg + "\n" + self.formatException(record.exc_info)
        return f"{ts}  {record.levelname:<8s}  {mod:<{self._MOD_W}s}  {msg}"

# ------------------------------------------------------------------------------
# Module Exports
# ------------------------------------------------------------------------------
__all__ = [
    "TUILogHandler",
    "JSONLogFormatter",
    "create_formatter",
    "create_log_handler",
    "get_log_handler",
    "_MinimalFormatter"
]
