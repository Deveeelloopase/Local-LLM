"""
supervisor.py — spawn, monitor, respawn worker handles.

Design invariants:
  * The monitor loop never blocks on I/O. Respawn is a tracked background
    task; one slow handshake cannot stall monitoring of the other slots.
  * Time is sampled per-handle inside the iteration, not once per tick.
    A slow `await` for one handle cannot make a later handle's "idle for"
    calculation use a stale `now`.
  * Backoff escalation survives a successful `start()`. It resets only
    after `stable_after_s` of health, not when the next attempt is
    scheduled. The previous code reset `consecutive_failures = 0` on the
    first successful `start()` return, so a worker that loaded in 15s
    then died 15s later always respawned at 1.0s — exactly what the
    committed log shows (every `respawning ... attempt 1 / 1.0s backoff`).
  * Degraded is sticky until a worker is stable. The previous code
    cleared `degraded = False` on the next successful `start()` return,
    which logged "marking degraded" 9 times in 19 minutes as the same
    slot flapped.
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
    ) -> None:
        self._load_spec = load_spec
        self._python = python_exe

        # Emit interval and kill threshold are SEPARATE named knobs.
        # `heartbeat_s * 10` and `heartbeat_s * 4` (the previous magic
        # multipliers) are gone; the parent kills only after
        # heartbeat_timeout_s of silence, and that timeout is at least
        # 3× the emit interval so a single missed heartbeat is not fatal.
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

        self._handles: Dict[int, WorkerHandle] = {}
        self._respawn: Dict[int, _RespawnState] = {}
        self._next_id: int = 0
        self._monitor_task: Optional[asyncio.Task] = None
        self._respawn_tasks: Dict[int, asyncio.Task] = {}
        self._shutting_down = False

        # P1-5: capacity_changed notification — set every time a worker
        # becomes idle, so the pool can wake its dispatch loop. Without
        # this a queued request could sit forever if no further traffic
        # arrives to trigger the next dispatch pass.
        self.on_worker_ready: Optional[Callable[[int], None]] = None

    def live_handles(self) -> List[WorkerHandle]:
        return self._live_handles()

    def _live_handles(self) -> List[WorkerHandle]:
        return [h for h in self._handles.values() if h.status().alive]

    def all_degraded(self) -> bool:
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
        while not self._shutting_down and len(self._live_handles()) < n:
            try:
                await self.spawn_worker()
            except Exception:
                log.exception("supervisor: spawn failed")
                await asyncio.sleep(1.0)

    async def start_monitor(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
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
        # Cancel in-flight respawn tasks
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

    # ── monitor loop ───────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        tick = max(0.2, min(self._heartbeat_s * 4.0, 5.0))
        while not self._shutting_down:
            try:
                await asyncio.sleep(tick)
                for wid, h in list(self._handles.items()):
                    # Per-handle `now`; never reused across awaits.
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
                        # The next tick will schedule respawn (h.alive is
                        # now False).
                        continue

                    if st.alive:
                        # Mark stable once healthy long enough.
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
        if now - state.last_respawn_ts < state.next_backoff_s:
            return
        self._respawn_tasks[wid] = asyncio.create_task(
            self._respawn_worker(wid),
            name=f"respawn_w{wid}",
        )

    async def _respawn_worker(self, wid: int) -> None:
        state = self._respawn.setdefault(wid, _RespawnState())
        # Capture and bump timing here, BEFORE the potentially slow
        # start(). If we bumped after start() returned, a successful
        # respawn followed by a fast death would see last_respawn_ts far
        # in the past and re-fire immediately.
        state.last_respawn_ts = time.monotonic()
        state.respawn_times.append(state.last_respawn_ts)
        state.consecutive_failures += 1
        backoff_idx = min(
            state.consecutive_failures - 1,
            len(self._backoff) - 1,
        )
        state.next_backoff_s = self._backoff[backoff_idx]
        state.healthy_since = None  # only re-set on stable health

        log.warning(
            "supervisor: respawning worker %d after %.1fs backoff (attempt %d)",
            wid, self._backoff[backoff_idx], state.consecutive_failures,
        )
        try:
            new_h = WorkerHandle(
                worker_id=wid,
                load_spec=self._load_spec,
                python_exe=self._python,
            )
            await asyncio.wait_for(new_h.start(), timeout=self._spawn_timeout_s)
            self._handles[wid] = new_h
            log.info("supervisor: worker %d respawned", wid)
            self._notify_ready(wid)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("supervisor: respawn failed for worker %d", wid)
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
