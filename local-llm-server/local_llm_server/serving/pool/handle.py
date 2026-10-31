"""
handle.py — parent-side proxy for one worker subprocess.

Owns the TCP loopback connection, dispatches frames to per-request async
queues, tracks live status, and exposes the streaming API the pool consumes.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from local_llm_core.accel.protocol import read_frame, write_frame

log = logging.getLogger(__name__)


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

        # Server socket: TCP loopback, ephemeral port
        self._server: Optional[asyncio.base_events.Server] = None
        self._port: int = 0
        self._conn_ready = asyncio.Event()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._proc: Optional[asyncio.subprocess.Process] = None

        # Reader task handles all inbound frames; dispatches to _active
        self._reader_task: Optional[asyncio.Task] = None
        self._active: Optional[_ActiveRequest] = None

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

        self._proc = await asyncio.create_subprocess_exec(
            self._python, "-m", "local_llm_core.engine.worker.main",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,     # do not receive Ctrl-C meant for parent
        )
        # Pump child stdout/stderr into parent log with prefix
        asyncio.create_task(self._pump_stderr())
        asyncio.create_task(self._pump_stdout())

        # Wait for handshake
        try:
            await asyncio.wait_for(self._conn_ready.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            await self.kill()
            raise RuntimeError(f"worker {self._id}: handshake timeout")

        self._reader_task = asyncio.create_task(self._reader_loop())
        self._alive = True
        log.info("worker %d ready (pid=%s)", self._id, self._proc.pid)

    async def _on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        header, _ = await read_frame(reader)
        if header.get("t") != "hello" or header.get("secret") != self._secret:
            writer.close()
            return
        await write_frame(writer, {"t": "hello_ok"}, b"")
        self._reader = reader
        self._writer = writer
        self._last_heartbeat_ts = time.monotonic()
        self._conn_ready.set()

    async def _pump_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        async for line in self._proc.stderr:
            log.info("[w%d] %s", self._id, line.decode("utf-8", "replace").rstrip())

    async def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        async for line in self._proc.stdout:
            log.info("[w%d/out] %s", self._id, line.decode("utf-8", "replace").rstrip())

    # ── reader ─────────────────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                header, body = await read_frame(self._reader)
                self._last_heartbeat_ts = time.monotonic()
                await self._dispatch(header, body)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            log.info("worker %d connection lost", self._id)
        except Exception:
            log.exception("worker %d reader error", self._id)
        finally:
            self._alive = False
            if self._active is not None and not self._active.finished:
                self._active.queue.put_nowait(("error", "worker died mid-request"))
                self._active.finished = True

    async def _dispatch(self, header: dict, body: bytes) -> None:
        t = header.get("t")
        if t == "chunk":
            if self._active is not None:
                self._active.queue.put_nowait(("chunk", body.decode("utf-8", "replace")))
        elif t == "done":
            if self._active is not None:
                self._active.queue.put_nowait(("done", None))
                self._active.finished = True
        elif t == "error":
            if self._active is not None:
                self._active.queue.put_nowait(
                    ("error", body.decode("utf-8", "replace") or "worker error")
                )
                self._active.finished = True
        elif t == "cancelled":
            if self._active is not None:
                self._active.queue.put_nowait(("cancelled", None))
                self._active.finished = True
        elif t == "ready":
            self._busy = False
            self._resident_prefix_len = int(header.get("resident_prefix_len", 0) or 0)
            self._resident_prefix_head = list(header.get("resident_prefix_head", []) or [])
            self._last_ready_ts = time.monotonic()
        elif t == "pong":
            pass
        else:
            log.warning("worker %d: unknown frame %r", self._id, header)

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
        try:
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
                    return
                elif kind == "cancelled":
                    return
                elif kind == "error":
                    raise RuntimeError(data)
        finally:
            self._active = None
            self._busy = False

    async def cancel(self, request_id: str) -> None:
        if self._writer is None:
            return
        try:
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
        return self._alive and not self._busy

    @property
    def worker_id(self) -> int:
        return self._id

    @property
    def proc_pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    # ── shutdown ───────────────────────────────────────────────────────────

    async def drain_and_stop(self, timeout_s: float = 30.0) -> None:
        self._expected_exit = True
        if self._writer is not None:
            try:
                await write_frame(self._writer, {"t": "shutdown"}, b"")
            except Exception:
                pass
        if self._proc is not None:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                await self.kill()

    async def kill(self) -> None:
        self._expected_exit = True
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
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
