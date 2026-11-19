"""
supervisor.py — spawn, monitor, respawn worker handles.

Design invariants:
  * The monitor loop never blocks on I/O. Respawn is a tracked background
    task; one slow handshake cannot stall monitoring of the other slots.
  * Time is sampled per-handle inside the iteration, not once per tick.
  * Backoff escalation survives a successful `start()`. It resets only
    after `stable_after_s` of health.
  * A parent-loop stall does not kill healthy workers. If the
    monitor tick gap exceeds 3 x tick, we skip the kill decision for one
    tick and let the reader loops drain buffered heartbeat frames. The
    old code killed every worker after a single GC pause or a blocking
    `time.sleep` in a sibling task.
  * Backoff is measured from the death and applied as a real
    `await asyncio.sleep(delay)` inside the respawn task. The old guard
    compared against `last_respawn_ts`, which was set at the START of the
    previous respawn — a worker that loaded for 15 s and died 1 s later
    saw `16 s > backoff` and was re-spawned immediately.
  * The old handle is `await kill()`-ed before replacement. The
    old code leaked the previous handle's `asyncio.Server`, `_writer`,
    stderr/stdout pumps and the subprocess fd, one set per crash.
  * `ensure_min_workers` is bounded. The old version retried
    forever, so an unloadable model hung lifespan startup until systemd
    `TimeoutStartSec` killed the unit and `StartLimitBurst` gave up.

version = 1.0.0
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .handle import WorkerHandle

log = logging.getLogger(__name__)


@dataclass
class _RespawnState:
    consecutive_failures: int = 0
    last_respawn_ts: float = 0.0
    next_backoff_s: float = 1.0
    degraded: bool = False
    respawn_times: List[float] = field(default_factory=list)
    healthy_since: Optional[float] = None  # monotonic ts of last stable start


class WorkerSupervisor:
    def __init__(
        self,
        *,
        load_spec: Dict[str, Any],
        python_exe: Optional[str] = None,
        heartbeat_s: float = 1.0,
        heartbeat_timeout_s: float = 15.0,
        respawn_backoff: Optional[List[float]] = None,
        max_respawns_per_5min: int = 3,
        stable_after_s: float = 60.0,
        spawn_timeout_s: float = 300.0,
        start_attempts: int = 3,
        start_attempt_backoff_s: float = 2.0,
    ) -> None:
        self._load_spec = load_spec
        self._python = python_exe

        self._heartbeat_s = float(heartbeat_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        if self._heartbeat_timeout_s <= self._heartbeat_s:
            clamped = max(3.0, self._heartbeat_s * 3.0)
            log.warning(
                "heartbeat_timeout_s=%.2f <= heartbeat_s=%.2f; clamping to %.2f",
                self._heartbeat_timeout_s, self._heartbeat_s, clamped,
            )
            self._heartbeat_timeout_s = clamped

        self._backoff = list(respawn_backoff or [1, 2, 4, 8, 30])
        self._max_respawns = int(max_respawns_per_5min)
        self._stable_after_s = float(stable_after_s)
        self._spawn_timeout_s = float(spawn_timeout_s)
        self._start_attempts = max(1, int(start_attempts))
        self._start_attempt_backoff_s = float(start_attempt_backoff_s)

        self._handles: Dict[int, WorkerHandle] = {}
        self._respawn: Dict[int, _RespawnState] = {}
        self._next_id: int = 0
        self._monitor_task: Optional[asyncio.Task] = None
        self._respawn_tasks: Dict[int, asyncio.Task] = {}
        self._shutting_down = False
        self._last_monitor_tick: float = 0.0

        # capacity_changed notification — set every time a worker
        # becomes idle, so the pool can wake its dispatch loop.
        self.on_worker_ready: Optional[Callable[[int], None]] = None

    def live_handles(self) -> List[WorkerHandle]:
        return self._live_handles()

    def _live_handles(self) -> List[WorkerHandle]:
        return [h for h in self._handles.values() if h.status().alive]

    def all_degraded(self) -> bool:
        """Return True when there are no workers OR every tracked worker
        is degraded.

        Callers MUST NOT treat this as a readiness signal by itself —
        use `live_worker_count()` + `degraded_worker_ids()` to
        distinguish "starting" (no workers yet) from "all degraded".
        Preserved unchanged from the previous version.
        """
        if not self._handles:
            return True
        return all(
            self._respawn.get(h.worker_id, _RespawnState()).degraded
            for h in self._handles.values()
        )

    async def spawn_worker(self) -> WorkerHandle:
        wid = self._next_id
        self._next_id += 1
        h = WorkerHandle(
            worker_id=wid,
            load_spec=self._load_spec,
            python_exe=self._python,
        )
        await asyncio.wait_for(h.start(), timeout=self._spawn_timeout_s)
        self._handles[wid] = h
        self._respawn[wid] = _RespawnState(healthy_since=time.monotonic())
        log.info("supervisor: worker %d spawned", wid)
        self._notify_ready(wid)
        return h

    async def ensure_min_workers(self, n: int) -> None:
        """Ensure `n` live workers, but do not block startup forever.

        bounded retries. A worker that cannot load (missing GGUF,
        OOM during load, import error) now fails its attempt in seconds
        (handle.start races conn_ready against proc.wait). The supervisor
        retries at most `start_attempts` times, then raises so lifespan
        can fail fast and systemd can report a real error instead of a
        generic "timeout".
        """
        consecutive_failures = 0
        last_exc: Optional[BaseException] = None
        while not self._shutting_down and len(self._live_handles()) < n:
            try:
                await self.spawn_worker()
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_failures += 1
                last_exc = exc
                log.exception(
                    "supervisor: spawn failed (attempt %d/%d): %s",
                    consecutive_failures, self._start_attempts, exc,
                )
                if consecutive_failures >= self._start_attempts:
                    raise RuntimeError(
                        f"supervisor: cannot spawn worker after "
                        f"{consecutive_failures} attempts: {last_exc}"
                    ) from last_exc
                await asyncio.sleep(self._start_attempt_backoff_s)

    async def start_monitor(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._last_monitor_tick = time.monotonic()
            self._monitor_task = asyncio.create_task(
                self._monitor_loop(), name="worker_supervisor",
            )

    async def shutdown(self, drain_timeout_s: float = 30.0) -> None:
        self._shutting_down = True
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        for t in list(self._respawn_tasks.values()):
            if t and not t.done():
                t.cancel()
        if self._respawn_tasks:
            await asyncio.gather(
                *self._respawn_tasks.values(), return_exceptions=True,
            )
            self._respawn_tasks.clear()
        await asyncio.gather(
            *[h.drain_and_stop(timeout_s=drain_timeout_s) for h in self._handles.values()],
            return_exceptions=True,
        )
        self._handles.clear()
        log.info("supervisor: shutdown complete")

    # ──— read-only accessors  ─────
    # All five methods below are additive observability contracts. They
    # read existing state; they never mutate supervisor state or send
    # anything to a worker. Safe to call at any time, including during
    # spawn/shutdown — a mid-mutation read yields a consistent snapshot
    # of the state as of the read, never a partial struct.

    def get_handle(self, worker_id: int) -> Optional[WorkerHandle]:
        """Return the live WorkerHandle for `worker_id`, or None.

        (pool.py) uses this to read `handle.last_token_count`
        after a request completes. Equivalent to `self._handles.get(wid)`
        — exposed as a public method so pool.py never reaches into the
        supervisor's private dict.

        Returns None for unknown or already-reaped worker IDs.
        """
        return self._handles.get(worker_id)

    def worker_load_info(self, worker_id: int) -> Optional[Dict[str, Any]]:
        """Return a live worker's load summary.

        The dict comes from the `hello` frame the worker sends on startup
        — see handle.py::_on_connection, which stores it in
        ``handle._load_info``. Fields:

            load_s      — seconds the worker spent loading the model
            gpu_layers  — actual n_gpu_layers the worker used
            ctx         — actual n_ctx
            threads     — actual n_threads
            n_batch     — actual n_batch
            n_ubatch    — actual n_ubatch

        These are the *effective* values after the parent's profile
        override (see worker/main.py's loader._profile replacement),
        not the raw config entries. api_server's startup banner
        uses this to print the MODEL LOADED block with truth rather than
        `gpu_layers=-1`.

        Returns None if the worker is unknown, or if the worker did not
        send an info dict (older worker).
        """
        h = self._handles.get(worker_id)
        if h is None:
            return None
        info = getattr(h, "_load_info", None)
        return dict(info) if isinstance(info, dict) else None

    def live_worker_count(self) -> int:
        """Return the number of currently-alive workers.

        "Alive" means the handle reports ``alive=True`` — the subprocess
        is running and its reader loop has not terminated. A worker that
        is running but degraded still counts as live (it may serve a
        request that has not yet failed). For "healthy and not degraded",
        check ``live_worker_count() > 0 and not all_degraded()``.

        Cheap: iterates the handles dict once. Suitable for `/readyz`
        which may be called by a load balancer every second.
        """
        return len(self._live_handles())

    def degraded_worker_ids(self) -> List[int]:
        """Return the worker IDs currently marked degraded.

        A worker becomes degraded when it has been respawned
        `max_respawns_per_5min` or more times in the last 5 minutes. It
        is not automatically killed — the pool's `PoolUnavailable`
        dispatch path will skip it — but it must not be trusted for
        readiness. `/readyz` in uses this to distinguish
        "no workers yet" (starting, Retry-After 5) from "all workers
        degraded" (failing, Retry-After 30).

        Returns a list sorted ascending by worker ID, so log lines are
        stable across calls.
        """
        out: List[int] = []
        for wid in self._handles.keys():
            state = self._respawn.get(wid)
            if state is not None and state.degraded:
                out.append(wid)
        out.sort()
        return out

    def worker_stats_snapshot(self) -> List[Dict[str, Any]]:
        """Return a list of per-worker stats dicts for observability.

        Consumed by:
          • `/v1/metrics` worker array — replaces the current
            truncated 4-field worker dict with the full per-worker state.
          • TUI Models panel — needs load_info, degraded state,
            heartbeat age to answer "why is this worker slow?".

        Each entry includes:

            worker_id             int
            alive                 bool       (subprocess running, reader up)
            busy                  bool       (request in flight)
            degraded              bool       (respawn rate limit hit)
            resident_prefix_len   int        (KV prefix cache size)
            last_ready_ts         float      (monotonic; 0 if never ready)
            last_heartbeat_ts     float      (monotonic; 0 if never seen)
            heartbeat_age_s       float|None (seconds since last hb; None if no hb yet)
            last_token_count      int        (worker's true generated count, 0 if not reported)
            death_reason          str        ("unknown" while alive, filled on death)
            load_info             dict       ({load_s, gpu_layers, ctx, threads, n_batch, n_ubatch} — empty if not reported)

        Read-only and snapshot-safe. A concurrent `kill()` or respawn of
        one worker during the snapshot may yield stale-but-consistent
        values for that worker; callers should treat the snapshot as
        best-effort telemetry, never as a lock. Per-handle exceptions are
        swallowed and that worker is skipped, so a single broken handle
        cannot abort the whole snapshot.
        """
        now = time.monotonic()
        out: List[Dict[str, Any]] = []
        for wid, h in list(self._handles.items()):
            try:
                st = h.status()
            except Exception:
                # A handle mid-kill() can transiently raise from status().
                # Skip rather than abort the whole snapshot.
                continue
            state = self._respawn.get(wid)
            _last_hb = float(st.last_heartbeat_ts)
            out.append({
                "worker_id": int(wid),
                "alive": bool(st.alive),
                "busy": bool(st.busy),
                "degraded": bool(state.degraded) if state else False,
                "resident_prefix_len": int(st.resident_prefix_len),
                "last_ready_ts": float(st.last_ready_ts),
                "last_heartbeat_ts": _last_hb,
                "heartbeat_age_s": (
                    round(now - _last_hb, 2) if _last_hb > 0 else None
                ),
                "last_token_count": int(getattr(h, "last_token_count", 0) or 0),
                "death_reason": str(getattr(h, "death_reason", "unknown")),
                "load_info": self.worker_load_info(wid) or {},
            })
        out.sort(key=lambda d: d["worker_id"])
        return out

    # ── monitor loop ───────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        tick = max(0.2, min(self._heartbeat_s * 4.0, 5.0))
        # track the interval between monitor ticks. If the parent
        # event loop stalled (GC pause, a blocking sync call in a sibling
        # coroutine), the first tick after the stall
        # sees every worker's last_heartbeat_ts as far in the past — but
        # the heartbeat frames ARE sitting in the socket buffer, and
        # will be processed as soon as the loop resumes. The old code
        # killed all workers in that window; one loop hiccup became a
        # full model reload. Now: if the tick gap exceeds 3 x tick, skip
        # the kill decision for ONE tick. Only a worker STILL silent on
        # the next tick is killed.
        stall_threshold = tick * 3.0
        while not self._shutting_down:
            try:
                await asyncio.sleep(tick)
                now = time.monotonic()
                tick_gap = now - (self._last_monitor_tick or now)
                self._last_monitor_tick = now
                if tick_gap > stall_threshold:
                    log.warning(
                        "supervisor: monitor tick gap %.1fs > %.1fs — "
                        "parent loop was stalled; skipping kill decisions "
                        "for one tick to let heartbeat frames drain",
                        tick_gap, stall_threshold,
                    )
                    continue

                for wid, h in list(self._handles.items()):
                    now = time.monotonic()
                    st = h.status()

                    # (1) Heartbeat liveness — only meaningful while alive.
                    if st.alive and (now - st.last_heartbeat_ts) > self._heartbeat_timeout_s:
                        idle_for = now - st.last_heartbeat_ts
                        h._death_reason = "heartbeat_timeout"
                        log.error(
                            "supervisor: worker %d missed heartbeat %.1fs "
                            "(timeout %.1fs) — killing",
                            wid, idle_for, self._heartbeat_timeout_s,
                            extra={
                                "worker_id": wid,
                                "idle_s": round(idle_for, 1),
                            },
                        )
                        try:
                            await h.kill()
                        except Exception:
                            log.exception(
                                "supervisor: kill failed for w%d", wid,
                            )
                        continue

                    if st.alive:
                        rs = self._respawn.setdefault(wid, _RespawnState())
                        if rs.healthy_since is None:
                            rs.healthy_since = now
                        elif (now - rs.healthy_since) >= self._stable_after_s:
                            if rs.consecutive_failures != 0 or rs.degraded:
                                log.info(
                                    "supervisor: worker %d stable for %.0fs — "
                                    "clearing degraded, resetting backoff",
                                    wid, now - rs.healthy_since,
                                )
                            rs.consecutive_failures = 0
                            rs.next_backoff_s = self._backoff[0]
                            rs.degraded = False
                        continue

                    # (2) Worker is dead — schedule a non-blocking respawn.
                    if self._shutting_down:
                        continue
                    self._maybe_schedule_respawn(wid, now)

            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("supervisor: monitor error")

    def _maybe_schedule_respawn(self, wid: int, now: float) -> None:
        # Never stack respawn tasks for the same slot.
        t = self._respawn_tasks.get(wid)
        if t is not None and not t.done():
            return
        state = self._respawn.setdefault(wid, _RespawnState())
        state.respawn_times = [x for x in state.respawn_times if now - x < 300.0]
        if len(state.respawn_times) >= self._max_respawns:
            if not state.degraded:
                state.degraded = True
                log.error(
                    "supervisor: worker %d respawned %d times in 5 min — "
                    "marking degraded; /ready will report not-ready",
                    wid, len(state.respawn_times),
                )
            return
        #  the real wait lives inside `_respawn_worker`. This guard
        # only prevents a second respawn task from being created while
        # one is already in its backoff sleep for the same worker.
        if now - state.last_respawn_ts < state.next_backoff_s:
            return
        self._respawn_tasks[wid] = asyncio.create_task(
            self._respawn_worker(wid),
            name=f"respawn_w{wid}",
        )

    async def _respawn_worker(self, wid: int) -> None:
        state = self._respawn.setdefault(wid, _RespawnState())
        state.consecutive_failures += 1
        backoff_idx = min(
            state.consecutive_failures - 1,
            len(self._backoff) - 1,
        )
        delay = self._backoff[backoff_idx]
        state.next_backoff_s = delay
        state.healthy_since = None  # only re-set on stable health

        log.warning(
            "supervisor: respawning worker %d after %.1fs backoff (attempt %d)",
            wid, delay, state.consecutive_failures,
        )
        # actual, awaited backoff from the DEATH of the previous
        # worker, not from the START of the previous respawn attempt.
        # Two crash-loops are now bounded by this sleep instead of
        # relying on the caller's 5-min rate limit.
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise

        state.last_respawn_ts = time.monotonic()
        state.respawn_times.append(state.last_respawn_ts)

        old_h = self._handles.get(wid)
        new_h: Optional[WorkerHandle] = None
        try:
            new_h = WorkerHandle(
                worker_id=wid,
                load_spec=self._load_spec,
                python_exe=self._python,
            )
            await asyncio.wait_for(new_h.start(), timeout=self._spawn_timeout_s)
            # reap the old handle before installing the new one so
            # its `asyncio.Server`, `_writer`, pump tasks, and the
            # subprocess fd are released now, not at interpreter exit.
            if old_h is not None:
                try:
                    await old_h.kill()
                except Exception:
                    log.exception(
                        "supervisor: old handle kill failed for w%d", wid,
                    )
            self._handles[wid] = new_h
            log.info("supervisor: worker %d respawned", wid)
            self._notify_ready(wid)
        except asyncio.CancelledError:
            if new_h is not None:
                try:
                    await new_h.kill()
                except Exception:
                    pass
            raise
        except Exception:
            log.exception("supervisor: respawn failed for worker %d", wid)
            if new_h is not None:
                try:
                    await new_h.kill()
                except Exception:
                    pass
        finally:
            self._respawn_tasks.pop(wid, None)

    def _notify_ready(self, wid: int) -> None:
        if self.on_worker_ready is not None:
            try:
                self.on_worker_ready(wid)
            except Exception:
                log.exception(
                    "on_worker_ready callback raised for w%d", wid,
                )
