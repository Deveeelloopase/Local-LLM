"""
handle.py — parent-side proxy for one worker subprocess.

Owns the TCP loopback connection, dispatches frames to per-request async
queues, tracks live status, and exposes the streaming API the pool consumes.

Design invariants:
  * Every outbound write goes through `_write_lock`; frames cannot
    interleave their wire bytes.
  * Every inbound frame refreshes `_last_heartbeat_ts` — including `hb`,
    which is why the heartbeat-timeout check sees activity even while a
    worker is idle.
  * `kill()` marks the handle dead before touching the socket, so any
    concurrent reader/dispatch that wakes up on cancellation sees a dead
    handle rather than a half-torn-down one.
  * Child stderr is parsed line-by-line and re-emitted at the child's
    original level; the pid and timestamp are stripped so the parent log
    shows one timestamp, one level, and one message per line (see P1-7).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from local_llm_core.accel.protocol import read_frame, write_frame

log = logging.getLogger(__name__)


# Match the worker's default log line format:
#   2026-10-31 02:42:12,544 [worker 1252233] INFO __main__ worker starting
# Anything that does NOT match (tracebacks, indented lines) is a
# continuation of the previous record and re-emitted at the same level.
_WORKER_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)"
    r"\s+\[worker\s+(?P<pid>\d+)\]\s+"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?P<logger>\S+)\s+"
    r"(?P<msg>.*)$"
)

_LEVEL_MAP = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_STDERR_RING_MAX = 20  # tail lines kept for the single death ERROR


@dataclass
class WorkerStatusSnapshot:
    worker_id: int
    alive: bool
    busy: bool
    resident_prefix_len: int
    resident_prefix_head: list
    last_ready_ts: float
    last_heartbeat_ts: float


@dataclass
class _ActiveRequest:
    request_id: str
    queue: asyncio.Queue
    sent_ts: float = field(default_factory=time.monotonic)
    finished: bool = False


class WorkerHandle:
    """
    Not thread-safe. Must be driven from the parent's asyncio loop.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        load_spec: Dict[str, Any],
        python_exe: Optional[str] = None,
    ) -> None:
        self._id = worker_id
        self._load_spec = load_spec
        self._python = python_exe or sys.executable
        self._secret = os.urandom(32).hex()

        self._server: Optional[asyncio.base_events.Server] = None
        self._port: int = 0
        self._conn_ready = asyncio.Event()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._proc: Optional[asyncio.subprocess.Process] = None

        # Single writer lock — see class docstring.
        self._write_lock = asyncio.Lock()

        self._reader_task: Optional[asyncio.Task] = None
        self._pump_tasks: list = []
        self._active: Optional[_ActiveRequest] = None

        # Death diagnostics — one ERROR line per death with reason + tail.
        self._stderr_ring: deque = deque(maxlen=_STDERR_RING_MAX)
        self._last_emitted_level: int = logging.INFO
        self._death_reason: str = "unknown"

        # Status
        self._resident_prefix_len: int = 0
        self._resident_prefix_head: list = []
        self._busy: bool = False
        self._last_ready_ts: float = 0.0
        self._last_heartbeat_ts: float = 0.0
        self._alive: bool = False
        self._expected_exit: bool = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_connection, host="127.0.0.1", port=0,
        )
        self._port = int(self._server.sockets[0].getsockname()[1])

        env = os.environ.copy()
        env["LLM_WORKER_PORT"] = str(self._port)
        env["LLM_WORKER_SECRET"] = self._secret
        env["LLM_WORKER_CONFIG"] = base64.b64encode(
            json.dumps(self._load_spec).encode("utf-8")
        ).decode("ascii")
        env["PYTHONUNBUFFERED"] = "1"

        # Propagate the parent's log level so the worker's library loggers
        # (transformers, hardware detector, etc.) stay quiet by default.
        try:
            parent_level = logging.getLogger().level
            if parent_level and parent_level > 0:
                env["LLM_WORKER_LOG_LEVEL"] = logging.getLevelName(parent_level)
        except Exception:
            pass

        self._proc = await asyncio.create_subprocess_exec(
            self._python, "-m", "local_llm_core.engine.worker.main",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Track pump tasks so kill() can cancel them. Previously these
        # tasks were fire-and-forget; a stuck one survived kill() and
        # leaked until process exit.
        self._pump_tasks = [
            asyncio.create_task(self._pump_stderr(), name=f"w{self._id}_stderr"),
            asyncio.create_task(self._pump_stdout(), name=f"w{self._id}_stdout"),
        ]

        try:
            await asyncio.wait_for(self._conn_ready.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            self._death_reason = "load_failed"
            await self.kill()
            raise RuntimeError(f"worker {self._id}: handshake timeout")
        except asyncio.CancelledError:
            await self.kill()
            raise

        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"w{self._id}_reader",
        )
        self._alive = True
        log.info("worker %d ready (pid=%s)", self._id, self._proc.pid)

    async def _on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            header, _ = await read_frame(reader)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass
            return
        if header.get("t") != "hello" or header.get("secret") != self._secret:
            try:
                writer.close()
            except Exception:
                pass
            return
        async with self._write_lock:
            await write_frame(writer, {"t": "hello_ok"}, b"")
        self._reader = reader
        self._writer = writer
        self._last_heartbeat_ts = time.monotonic()
        self._conn_ready.set()

    async def _pump_stderr(self) -> None:
        """Parse child stderr lines and re-emit at their original level.

        Previously every line was re-emitted at INFO with a second
        timestamp, producing the doubled `[w0] 2026-10-31 02:42:11,645
        [worker 1252233] INFO ...` you saw in 87% of the log. Here we
        parse the worker's own format, drop the pid and timestamp, and
        emit under `worker.w{id}` at the same severity. Continuation
        lines (tracebacks, indented messages) attach to the previous
        record at the same level.
        """
        assert self._proc and self._proc.stderr
        worker_logger = logging.getLogger(f"worker.w{self._id}")
        async for raw in self._proc.stderr:
            try:
                line = raw.decode("utf-8", "replace").rstrip("\n")
            except Exception:
                continue
            if not line:
                continue
            self._stderr_ring.append(line[:500])
            m = _WORKER_LINE_RE.match(line)
            if m:
                level_no = _LEVEL_MAP.get(m.group("level"), logging.INFO)
                self._last_emitted_level = level_no
                worker_logger.log(
                    level_no, "%s", m.group("msg"),
                    extra={
                        "worker_id": self._id,
                        "child_pid": m.group("pid"),
                        "child_logger": m.group("logger"),
                    },
                )
            else:
                worker_logger.log(
                    self._last_emitted_level, "  | %s", line,
                    extra={"worker_id": self._id, "continuation": True},
                )

    async def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        worker_logger = logging.getLogger(f"worker.w{self._id}.out")
        async for raw in self._proc.stdout:
            try:
                line = raw.decode("utf-8", "replace").rstrip("\n")
            except Exception:
                continue
            if not line:
                continue
            worker_logger.debug("%s", line)

    # ── reader ─────────────────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                header, body = await read_frame(self._reader)
                # ANY inbound frame — including `hb` — proves liveness.
                self._last_heartbeat_ts = time.monotonic()
                await self._dispatch(header, body)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            log.info("worker %d connection lost", self._id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker %d reader error", self._id)
        finally:
            was_alive = self._alive
            self._alive = False
            if self._active is not None and not self._active.finished:
                try:
                    self._active.queue.put_nowait(
                        ("error", "worker died mid-request")
                    )
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
            if not self._expected_exit and was_alive:
                self._death_reason = self._classify_death()
                self._log_death_once()

    def _classify_death(self) -> str:
        proc = self._proc
        if proc is None:
            return "unknown"
        rc = proc.returncode
        if rc is None:
            # Process is still alive but the connection dropped. When the
            # supervisor killed this worker (heartbeat timeout) it sets
            # _death_reason = "heartbeat_timeout" BEFORE calling kill(),
            # so kill() preserves that reason and this code path is never
            # reached from the supervisor. Reaching here with the process
            # alive means the worker closed its own socket (internal crash
            # during shutdown), or a genuine hang where the parent-side
            # connection was torn down before the OS process died.
            if self._expected_exit:
                return "killed_by_parent"
            return "connection_lost_while_alive"
        if rc < 0:
            sig = -rc
            try:
                name = signal.Signals(sig).name
            except Exception:
                name = f"signal {sig}"
            return f"crashed({name})"
        if rc == 0:
            return "exited_cleanly"
        return f"crashed(exit={rc})"

    def _log_death_once(self) -> None:
        tail = list(self._stderr_ring)
        log.error(
            "worker w%d died reason=%s pid=%s expected=%s",
            self._id, self._death_reason,
            self._proc.pid if self._proc else "?",
            self._expected_exit,
            extra={"worker_id": self._id, "death_reason": self._death_reason},
        )
        if tail:
            log.error(
                "worker w%d last stderr:\n  %s",
                self._id, "\n  ".join(tail),
                extra={"worker_id": self._id},
            )

    async def _dispatch(self, header: dict, body: bytes) -> None:
        t = header.get("t")
        _rid = header.get("rid")
        _active_rid = self._active.request_id if self._active is not None else None

        # P1-4 (broadened): every frame that carries a `rid` must match
        # the currently active request. Without this, a stale `ready` or
        # a delayed `chunk` from a just-completed request can corrupt the
        # new request's state — clear `_busy` while it is actually still
        # generating (making the pool dispatch another request at a busy
        # worker), or splice tokens from request A into B's stream.
        #
        # Tolerance for missing `rid`: `worker/main.py`'s current build
        # only sets `rid` on chunk/done/error/cancelled — the ready frame
        # has no `rid` yet. Adding `rid` there is a separate file's fix.
        # If `_rid is None` we fall through (legacy behaviour, safe given
        # the pool-side FIFO serialises requests one-at-a-time per handle).
        # When the worker starts sending `rid` on ready, this check
        # activates for it too — no further change needed here.
        if _rid is not None and _active_rid is not None and _rid != _active_rid:
            log.debug(
                "worker %d: dropping stale %s frame (rid=%s, active=%s)",
                self._id, t, _rid, _active_rid,
            )
            return

        if t == "chunk":
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(
                        ("chunk", body.decode("utf-8", "replace"))
                    )
                except asyncio.QueueFull:
                    # Stream backpressure — cancel the request, do NOT
                    # kill the worker. The previous code let QueueFull
                    # escape and terminate the reader loop.
                    try:
                        self._active.queue.put_nowait(
                            ("error", "client stream backpressure overflow")
                        )
                    except asyncio.QueueFull:
                        pass
                    self._active.finished = True
        elif t == "done":
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(("done", None))
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
        elif t == "error":
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(
                        ("error", body.decode("utf-8", "replace") or "worker error")
                    )
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
        elif t == "cancelled":
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(("cancelled", None))
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
        elif t == "ready":
            self._busy = False
            self._resident_prefix_len = int(header.get("resident_prefix_len", 0) or 0)
            self._resident_prefix_head = list(header.get("resident_prefix_head", []) or [])
            self._last_ready_ts = time.monotonic()
        elif t == "hb":
            # Liveness was recorded by the reader loop; nothing else to
            # do. This branch exists so heartbeats do not log
            # "unknown frame" at WARNING.
            pass
        elif t == "pong":
            pass
        else:
            log.warning("worker %d: unknown frame %r", self._id, t)

    # ── submit / cancel ────────────────────────────────────────────────────
    async def submit(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        if self._active is not None and not self._active.finished:
            raise RuntimeError(f"worker {self._id} already busy")
        if not self._alive or self._writer is None:
            raise RuntimeError(f"worker {self._id} not alive")

        q: asyncio.Queue = asyncio.Queue(maxsize=4096)
        ar = _ActiveRequest(request_id=payload["request_id"], queue=q)
        self._active = ar
        self._busy = True

        # `_completed` tracks whether we exited the loop via a worker-
        # signalled terminal (`done` / `cancelled`). If we exit any other
        # way — caller cancelled the generator, upstream task died, worker
        # raised — the worker is still generating and needs an explicit
        # cancel frame.
        _completed = False
        try:
            async with self._write_lock:
                await write_frame(
                    self._writer,
                    {"t": "submit", "rid": payload["request_id"]},
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                )
            while True:
                kind, data = await q.get()
                if kind == "chunk":
                    yield data
                elif kind == "done":
                    _completed = True
                    return
                elif kind == "cancelled":
                    # Worker confirmed the stop — no cancel frame needed,
                    # the generation has already halted.
                    _completed = True
                    return
                elif kind == "error":
                    # Worker errored; its _run_one already sent `ready`
                    # and cleared its internal state. No cancel frame
                    # needed, but leave _completed False so the finally
                    # re-sends defensively (harmless, worker ignores
                    # cancel for a rid it no longer tracks).
                    raise RuntimeError(data)
        finally:
            # cancel-on-early-exit. Fires when the consumer stops
            # iterating before receiving `done` or `cancelled` — the
            # canonical cases are:
            #   * pool._run_on_worker was cancelled (shutdown, upstream
            #     cancellation propagation)
            #   * API-layer StreamingResponse was closed on client
            #     disconnect (Starlette calls aclose() on the generator)
            #   * an exception in the consumer path between yields
            # Without this, the single-slot worker kept generating for a
            # request nobody was reading, holding the slot until natural
            # completion or inference timeout. With it, the worker's
            # `_run_one` task receives `cancel`, generates `cancelled`,
            # and yields the slot immediately.
            if not _completed and self._writer is not None and self._alive:
                try:
                    async with self._write_lock:
                        await write_frame(
                            self._writer,
                            {"t": "cancel", "rid": payload["request_id"]},
                            b"",
                        )
                except Exception:
                    # Writer may have been torn down by kill() while we
                    # were suspended; nothing to do, cancellation of the
                    # worker is handled by kill()'s SIGTERM path.
                    pass

            # Guard the teardown: if this handle was somehow re-entered
            # (invariant violation, or a future change relaxing the top
            # guard), don't clobber a different active request.
            if self._active is ar:
                self._active = None
                self._busy = False

    async def cancel(self, request_id: str) -> None:
        if self._writer is None:
            return
        try:
            async with self._write_lock:
                await write_frame(
                    self._writer, {"t": "cancel", "rid": request_id}, b"",
                )
        except Exception:
            pass

    # ── status ─────────────────────────────────────────────────────────────

    def status(self) -> WorkerStatusSnapshot:
        return WorkerStatusSnapshot(
            worker_id=self._id,
            alive=self._alive,
            busy=self._busy,
            resident_prefix_len=self._resident_prefix_len,
            resident_prefix_head=self._resident_prefix_head,
            last_ready_ts=self._last_ready_ts,
            last_heartbeat_ts=self._last_heartbeat_ts,
        )

    def is_idle(self) -> bool:
        # defence in depth: `_busy` alone is not enough — a stale
        # `ready` frame could clear it while `_active` is still set.
        # Requiring `_active is None` makes "idle" mean exactly "no
        # request is in flight", matching what the pool's dispatch
        # loop needs to know.
        return self._alive and not self._busy and self._active is None

    @property
    def worker_id(self) -> int:
        return self._id

    @property
    def proc_pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    @property
    def death_reason(self) -> str:
        return self._death_reason

    # ── shutdown ───────────────────────────────────────────────────────────

    async def drain_and_stop(self, timeout_s: float = 30.0) -> None:
        self._expected_exit = True
        if self._writer is not None:
            try:
                async with self._write_lock:
                    await write_frame(self._writer, {"t": "shutdown"}, b"")
            except Exception:
                pass
        if self._proc is not None:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                await self.kill()

    async def kill(self) -> None:
        """Terminate the worker and release every owned resource.

        The old implementation waited on `self._server.wait_closed()`
        with no timeout. On Python 3.12 that call blocks on the
        parent-side connection which `_on_connection` never closes, so
        `kill()` could hang for >15s (reproduced in the diagnosis). The
        supervisor called `kill()` inline in its monitor loop, so a
        hanging kill froze the whole pool.

        Order here is defensive:
          1. mark dead FIRST so anything waking on cancellation sees a
             dead handle rather than a half-torn-down one,
          2. cancel the reader task,
          3. close the writer (may unblock a stuck `wait_closed`),
          4. cancel the stderr/stdout pump tasks,
          5. SIGTERM → wait 5s → SIGKILL → wait 5s (unchanged semantics),
          6. close the server socket with a bounded wait.
        """
        self._expected_exit = True
        self._alive = False
        if self._death_reason == "unknown":
            self._death_reason = "killed_by_parent"

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._writer.wait_closed(), timeout=2.0)
            except Exception:
                pass
            self._writer = None

        for t in self._pump_tasks:
            if t and not t.done():
                t.cancel()
        if self._pump_tasks:
            await asyncio.gather(*self._pump_tasks, return_exceptions=True)
            self._pump_tasks = []

        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass

        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except Exception:
                pass
            self._server = None
