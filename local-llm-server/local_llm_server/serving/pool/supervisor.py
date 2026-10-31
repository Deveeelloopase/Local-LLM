"""
supervisor.py — spawn, monitor, respawn worker handles.

Respawn policy: exponential backoff up to 30 s, max 3 respawns per 5 minutes.
Beyond that, the slot is marked degraded and /ready reports not ready until
either a successful request flips it back or the operator restarts the server.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .handle import WorkerHandle

log = logging.getLogger(__name__)


@dataclass
class _RespawnState:
    consecutive_failures: int = 0
    last_respawn_ts: float = 0.0
    next_backoff_s: float = 1.0
    degraded: bool = False
    respawn_times: List[float] = field(default_factory=list)


class WorkerSupervisor:
    def __init__(
        self,
        *,
        load_spec: Dict[str, Any],
        python_exe: Optional[str] = None,
        heartbeat_s: float = 0.5,
        respawn_backoff: Optional[List[float]] = None,
        max_respawns_per_5min: int = 3,
    ) -> None:
        self._load_spec = load_spec
        self._python = python_exe
        self._heartbeat_s = float(heartbeat_s)
        self._backoff = list(respawn_backoff or [1, 2, 4, 8, 30])
        self._max_respawns = int(max_respawns_per_5min)

        self._handles: Dict[int, WorkerHandle] = {}
        self._respawn: Dict[int, _RespawnState] = {}
        self._next_id: int = 0
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutting_down = False

    # ── public ─────────────────────────────────────────────────────────────

    async def spawn_worker(self) -> WorkerHandle:
        wid = self._next_id
        self._next_id += 1
        h = WorkerHandle(
            worker_id=wid, load_spec=self._load_spec, python_exe=self._python,
        )
        await h.start()
        self._handles[wid] = h
        self._respawn[wid] = _RespawnState()
        log.info("supervisor: worker %d spawned", wid)
        return h

    async def ensure_min_workers(self, n: int) -> None:
        while len(self._live_handles()) < n:
            try:
                await self.spawn_worker()
            except Exception:
                log.exception("supervisor: spawn failed")
                await asyncio.sleep(1.0)

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
        await asyncio.gather(
            *[h.drain_and_stop(timeout_s=drain_timeout_s) for h in self._handles.values()],
            return_exceptions=True,
        )
        self._handles.clear()
        log.info("supervisor: shutdown complete")

    # ── monitor loop ───────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._heartbeat_s * 4)
                now = time.monotonic()
                for wid, h in list(self._handles.items()):
                    st = h.status()
                    # Missing heartbeat
                    if st.alive and (now - st.last_heartbeat_ts) > self._heartbeat_s * 10:
                        log.error(
                            "supervisor: worker %d missed heartbeat %.1fs — killing",
                            wid, now - st.last_heartbeat_ts,
                        )
                        await h.kill()
                    if st.alive:
                        continue
                    # Worker died unexpectedly → respawn policy
                    if self._shutting_down:
                        continue
                    await self._handle_death(wid, now)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("supervisor: monitor error")

    async def _handle_death(self, wid: int, now: float) -> None:
        state = self._respawn.setdefault(wid, _RespawnState())
        state.respawn_times = [t for t in state.respawn_times if now - t < 300.0]
        if len(state.respawn_times) >= self._max_respawns:
            if not state.degraded:
                log.error(
                    "supervisor: worker %d respawned %d times in 5 min — "
                    "marking degraded; /ready will report not-ready",
                    wid, len(state.respawn_times),
                )
                state.degraded = True
            return
        if now - state.last_respawn_ts < state.next_backoff_s:
            return
        log.warning(
            "supervisor: respawning worker %d after %.1fs backoff (attempt %d)",
            wid, state.next_backoff_s, state.consecutive_failures + 1,
        )
        state.last_respawn_ts = now
        state.respawn_times.append(now)
        state.consecutive_failures += 1
        state.next_backoff_s = self._backoff[
            min(state.consecutive_failures - 1, len(self._backoff) - 1)
        ]
        try:
            new_h = WorkerHandle(
                worker_id=wid, load_spec=self._load_spec, python_exe=self._python,
            )
            await new_h.start()
            self._handles[wid] = new_h
            state.consecutive_failures = 0
            state.next_backoff_s = self._backoff[0]
            state.degraded = False
            log.info("supervisor: worker %d respawned", wid)
        except Exception:
            log.exception("supervisor: respawn failed for worker %d", wid)

    