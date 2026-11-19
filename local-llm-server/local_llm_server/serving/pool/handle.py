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
  * `last_token_count` is the worker's *authoritative* generated-
    token count for the last request. It is 0 until `done` (or `cancelled`,
    or `error`) arrives, and 0 forever for workers predating which
    do not send the field. Callers MUST treat 0 as "not reported — fall
    back to per-chunk counting". The attribute is reset to 0 at the start
    of every `submit()` call so a stale count from a previous request
    cannot leak into a new one.

version = 1.0.0
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

class WorkerStartError(RuntimeError):
    """Raised by `WorkerHandle.start()` when the child process exits or is
    unloadable before the handshake completes.

    The exception carries the exit code and the tail of the child's stderr
    so lifespan can log a specific cause (`missing GGUF`, `ImportError:
    transformers`, `CUDA OOM`) instead of the previous behavior, which
    blocked on the handshake for the full 300 s timeout and then reported
    a generic `handshake timeout`.
    """

    def __init__(self, worker_id: int, exit_code: int, stderr_tail: str):
        self.worker_id = worker_id
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        super().__init__(
            f"worker {worker_id} exited during load (exit={exit_code}); "
            f"last stderr:\n  {stderr_tail}"
        )

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

# ── raw child-process (llama.cpp / ggml) line classification ────────
# These patterns only apply to lines that do NOT match _WORKER_LINE_RE —
# i.e., raw C-level stderr output, not the worker's own Python log lines.
# Order matters: the first match wins, so the list is sorted from most
# specific (ERROR keywords) to least (generic ggml_/llama_ prefix).
#
# The plan explicitly names these categories:
#   * "error / failed / abort / OOM"      → WARNING / ERROR
#   * "ggml_vulkan: Found/0 ="            → INFO   (device report)
#   * "llama_*" / "load_tensors"          → DEBUG  (load diagnostics)
#
# The `ggml_cuda:`, `ggml_metal:`, `ggml_sycl:`, `ggml_hip:` additions
# cover the same "device report" category on other backends — same intent,
# same level. `assert(ion)?` matches both "assert" and "assertion"
# because ggml's own abort messages read "GGML_ASSERT: ...".
_GGML_LEVEL_HINTS = [
    (re.compile(r"(?i)\b(error|failed|abort|assert(?:ion)?|oom|out\s+of\s+memory)\b"),
     logging.ERROR),
    (re.compile(r"(?i)^(ggml_vulkan|ggml_cuda|ggml_metal|ggml_sycl|ggml_hip|ggml_opencl):"),
     logging.INFO),
    (re.compile(r"(?i)\b(device|backend)\b.*\b(init|found|loaded|ready)\b"),
     logging.INFO),
    (re.compile(r"(?i)\b(load_tensors|llama_model_load|llama_kv_cache|llama_new_context)\b"),
     logging.DEBUG),
    (re.compile(r"(?i)\b(llama_|ggml_|gguf_)"),
     logging.DEBUG),
]


def _classify_child_line(line: str) -> int:
    """Return the logging level for a raw child-process stderr line.

    Called only from `_pump_stderr`'s else-branch — i.e., for lines that
    do NOT match `_WORKER_LINE_RE` (not wrapped by the worker's own
    Python logger). Returns `logging.WARNING` for unknown lines, matching
    the previous behavior as a safe fallback.
    """
    for pattern, level in _GGML_LEVEL_HINTS:
        if pattern.search(line):
            return level
    return logging.WARNING


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

        # the worker's authoritative generated-token count for the
        # last request. 0 means "not reported (old worker) or no request
        # has completed yet". Callers MUST fall back to per-chunk counting
        # when this is 0. Reset at the start of every submit() so a stale
        # count from a previous request cannot leak into a new one.
        self.last_token_count: int = 0

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
        # worker self-reports its load summary in the `hello`
        # frame (see worker/main.py:_amain). Stored here so the ready
        # log line and /v1/models metadata can include the actual
        # load time, gpu_layers, ctx, threads, and batch sizes rather
        # than "loaded" with no numbers.
        self._load_info: Dict[str, Any] = {}

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
            # asyncio StreamReader default limit is 64 KiB per
            # line. A long traceback or a llama.cpp error that embeds
            # prompt text can exceed that; the pump then dies on
            # LimitOverrunError, the stderr pipe fills, and the worker
            # blocks forever in write(2). 1 MiB gives ample headroom.
            limit=1 << 20,
        )

        self._pump_tasks = [
            asyncio.create_task(self._pump_stderr(), name=f"w{self._id}_stderr"),
            asyncio.create_task(self._pump_stdout(), name=f"w{self._id}_stdout"),
        ]

        # race the handshake against process exit. A worker that
        # dies during load (missing GGUF, OOM, import error) raises
        # WorkerStartError in seconds with its stderr tail, instead of
        # blocking for the full 300 s handshake timeout. The supervisor
        # then retries a bounded number of times and raises to lifespan.
        conn_task = asyncio.ensure_future(self._conn_ready.wait())
        proc_task = asyncio.ensure_future(self._proc.wait())
        try:
            done, _pending = await asyncio.wait_for(
                asyncio.wait(
                    {conn_task, proc_task},
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=300.0,
            )
        except asyncio.TimeoutError:
            self._death_reason = "load_failed"
            conn_task.cancel()
            proc_task.cancel()
            await self.kill()
            raise WorkerStartError(
                worker_id=self._id,
                exit_code=-1,
                stderr_tail="(handshake timeout after 300 s; last stderr below)\n"
                            + "\n  ".join(self._stderr_ring),
            ) from None
        except asyncio.CancelledError:
            conn_task.cancel()
            proc_task.cancel()
            await self.kill()
            raise

        if proc_task in done and not self._conn_ready.is_set():
            rc = self._proc.returncode
            tail = "\n  ".join(self._stderr_ring) if self._stderr_ring else "(no stderr)"
            self._death_reason = self._classify_death()
            if not conn_task.done():
                conn_task.cancel()
            try:
                await conn_task
            except (asyncio.CancelledError, Exception):
                pass
            await self.kill()
            raise WorkerStartError(
                worker_id=self._id,
                exit_code=rc if rc is not None else -1,
                stderr_tail=tail,
            )

        if not conn_task.done():
            conn_task.cancel()
        proc_task.cancel()

        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"w{self._id}_reader",
        )
        self._alive = True
        # full load summary in the parent log. This is the one
        # line an operator needs to answer "why is this worker slow /
        # how many GPU layers actually offloaded / what context did
        # the planner pick" without grepping through worker stderr.
        _req_gl = int(self._load_info.get("gpu_layers", -1))
        _actual_gl_raw = self._load_info.get("gpu_layers_actual")
        _actual_gl = (
            int(_actual_gl_raw)
            if _actual_gl_raw is not None and int(_actual_gl_raw) >= 0
            else "?"
        )
        _kv = str(self._load_info.get("kv_cache_quant", "?"))
        _fa = self._load_info.get("flash_attn", "?")
        # report `gpu_layers=req:N/actual:M` so a load that silently
        # reduced (loader OOM retry) is visible in the parent log without
        # requiring the operator to correlate against the worker's
        # `worker: model loaded` line. `kv` and `fa` were previously
        # invisible in pool mode because the internal loader/layer_streamer
        # INFO lines are suppressed at the worker's WARNING default.
        log.info(
            "worker %d ready pid=%s load=%.1fs gpu_layers=req:%d/actual:%s "
            "ctx=%d threads=%d batch=%d/%d kv=%s fa=%s",
            self._id, self._proc.pid,
            float(self._load_info.get("load_s", 0.0)),
            _req_gl,
            _actual_gl,
            int(self._load_info.get("ctx", 0)),
            int(self._load_info.get("threads", 0)),
            int(self._load_info.get("n_batch", 0)),
            int(self._load_info.get("n_ubatch", 0)),
            _kv,
            _fa,
        )

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
        # extract the load summary from the hello frame. Missing
        # info is not an error — older workers may not send it, in
        # which case the ready log falls back to zeros (clearly visible
        # as "load=0.0s gpu_layers=-1"), which is preferable to
        # silently omitting the fields and letting the reader assume
        # they were never measured.
        _info = header.get("info") if isinstance(header, dict) else None
        if isinstance(_info, dict):
            self._load_info = _info
        self._conn_ready.set()

    async def _pump_stderr(self) -> None:
        """Parse child stderr lines and re-emit at their original level.

        The reader was created with a 1 MiB line limit, but
        `readline()` still raises `ValueError` wrapping `LimitOverrunError`
        when a line exceeds it. The old code let that escape; the pump
        task died silently, the stderr pipe filled, and the worker
        blocked in `write(2)`. Two defences here:

          1. on overrun, drain and truncate the residual bytes so the
             pipe keeps flowing;
          2. wrap the whole loop so any truly unexpected exception logs
             and the pump resumes.

        lines that do NOT match `_WORKER_LINE_RE` (raw llama.cpp /
        ggml output) are classified via `_classify_child_line()`.
        Indented continuation lines inherit the previous line's level so
        a multi-line device report stays at INFO throughout.
        """
        assert self._proc and self._proc.stderr
        worker_logger = logging.getLogger(f"worker.w{self._id}")
        reader = self._proc.stderr
        while True:
            try:
                try:
                    raw = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    # Line exceeded the configured limit. Drain a chunk
                    # so the pipe keeps flowing, log a marker, continue.
                    try:
                        await reader.read(4096)
                    except Exception:
                        pass
                    worker_logger.warning(
                        "  | (stderr line exceeded buffer limit; truncated)",
                        extra={"worker_id": self._id, "continuation": True},
                    )
                    continue
                if not raw:
                    return
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
                    #  classify raw child-process output.
                    #   - Indented lines are continuations of the
                    #     preceding block (multi-line device report,
                    #     traceback body). They inherit `_last_emitted_level`
                    #     so a `ggml_vulkan: Found 1 Vulkan devices:`
                    #     header at INFO does not produce WARNING noise
                    #     from the per-device detail lines beneath it.
                    #   - Non-indented lines are new records; classify
                    #     them via `_classify_child_line()` and remember
                    #     the result so indented continuations inherit it.
                    _is_continuation = bool(line) and line[0] in (" ", "\t")
                    if _is_continuation:
                        _lvl = self._last_emitted_level
                    else:
                        _lvl = _classify_child_line(line)
                        self._last_emitted_level = _lvl
                    worker_logger.log(
                        _lvl, "  | %s", line,
                        extra={"worker_id": self._id, "continuation": True},
                    )
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception(
                    "worker %d stderr pump error — continuing", self._id,
                )
                await asyncio.sleep(0.1)

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
            worker_logger.warning("%s", line)

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

        # (broadened): every frame that carries a `rid` must match
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
            # a `ready` frame for a just-completed request can arrive
            # AFTER the pool has dispatched the next request to this same
            # worker — the reader loop drains in-order but the pool's
            # dispatch loop has already moved on. Dropping the frame is
            # correct for `_busy` (the worker IS busy with the new
            # request), but the prefix fields it carries describe what
            # is actually resident in the worker's KV cache. The router's
            # prefix-affinity score reads those two fields on every
            # dispatch, so dropping them makes affinity blind until the
            # next request on this worker finishes (a whole generation
            # later). Apply the prefix fields unconditionally and drop
            # only the `_busy`/`_last_ready_ts` side of the frame.
            #

            if t == "ready":
                self._resident_prefix_len = int(
                    header.get("resident_prefix_len", 0) or 0
                )
                self._resident_prefix_head = list(
                    header.get("resident_prefix_head", []) or []
                )
                log.info(
                    "worker %d: applying resident_prefix (len=%d) from "
                    "stale ready frame (rid=%s, active=%s) — "
                    "dropping _busy clear only",
                    self._id, self._resident_prefix_len, _rid, _active_rid,
                )
            else:
                log.info(
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
            # capture the worker's true generated-token count from
            # the frame. Old workers do not send the
            # field — default 0 tells the pool to fall back to per-chunk
            # counting. `tokens` may legitimately be 0 (empty response);
            # the pool treats 0 as "fall back", not "0 tokens emitted".
            _tokens = 0
            try:
                _tokens = int(header.get("tokens", 0) or 0)
            except (TypeError, ValueError):
                _tokens = 0
            self.last_token_count = max(0, _tokens)
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(("done", None))
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
        elif t == "error":
            # same as `done` — worker reports how many tokens it
            # emitted before failing, if it was able to. Default 0.
            _tokens = 0
            try:
                _tokens = int(header.get("tokens", 0) or 0)
            except (TypeError, ValueError):
                _tokens = 0
            self.last_token_count = max(0, _tokens)
            if self._active is not None:
                try:
                    self._active.queue.put_nowait(
                        ("error", body.decode("utf-8", "replace") or "worker error")
                    )
                except asyncio.QueueFull:
                    pass
                self._active.finished = True
        elif t == "cancelled":
            # same as `done` — worker reports how many tokens it
            # emitted before the cancel was processed. Default 0.
            _tokens = 0
            try:
                _tokens = int(header.get("tokens", 0) or 0)
            except (TypeError, ValueError):
                _tokens = 0
            self.last_token_count = max(0, _tokens)
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

        # reset the token count at the start of every request so a
        # stale count from the previous request cannot leak. The value is
        # set by `_dispatch()` when `done`/`cancelled`/`error` arrives;
        # callers read it AFTER the async-for completes (never during).
        self.last_token_count = 0

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
