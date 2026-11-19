"""
router.py — CAAS (Cache-Affinity Admission Scheduler).

Among IDLE workers only. Busy workers are never routed to; requests queue in
the pool's global FIFO until a worker becomes idle.

Score of idle worker w for request r:

    score(w, r) = prefix_gain(w, r)
                - thermal_penalty(w)
                - staleness_penalty(w)

  prefix_gain       = shared_prefix_tokens(w, r) × prefix_weight
  thermal_penalty   = max(0, temp − throttle_start) × thermal_weight
  staleness_penalty = min(age_since_idle, 60) × 0.001 × staleness_weight

the previous docstring claimed this penalty "breaks ties
round-robin-style." It does not. `staleness_penalty` GROWS with idle
time, so the freshest idle worker scores highest and gets picked first.
That is most-recently-used, not round-robin.

Round-robin would need the opposite sign — a worker idle longer should
score higher to equalise load. That was tried and rejected: with 2+
workers the KV cache on a just-finished worker is warm, and a
round-robin pick forces a cache-miss prefill on the old worker for
every other request. Preferring the freshest idle worker keeps
steady-state TTFT low; the cost is that when two workers are idle
simultaneously (rare at this deployment's 1-worker ledger ceiling),
the same one tends to stay hot. Prefix_gain (once the parent holds a
tokenizer) will dominate the tie-break in multi-worker configurations,
so the practical impact of the freshest-wins choice is bounded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CAASWeights:
    prefix: float = 1.0
    thermal: float = 0.5
    staleness: float = 1.0
    throttle_start_c: float = 85.0


@dataclass(frozen=True)
class IdleCandidate:
    worker_id: int
    resident_prefix_head: Sequence[int]
    last_ready_ts: float
    temperature_c: Optional[float] = None


def _shared_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class CAASRouter:
    def __init__(self, weights: CAASWeights) -> None:
        self._w = weights

    def pick(
        self,
        *,
        request_token_ids: Sequence[int],
        candidates: List[IdleCandidate],
    ) -> Optional[IdleCandidate]:
        if not candidates:
            return None
        now = time.monotonic()
        best: Optional[IdleCandidate] = None
        best_score: float = float("-inf")
        for c in candidates:
            shared = _shared_prefix_len(
                c.resident_prefix_head, request_token_ids,
            )
            prefix_gain = shared * self._w.prefix

            thermal_pen = 0.0
            if c.temperature_c is not None:
                thermal_pen = max(
                    0.0, c.temperature_c - self._w.throttle_start_c,
                ) * self._w.thermal

            staleness_pen = 0.0
            if c.last_ready_ts > 0:
                staleness_pen = min(
                    now - c.last_ready_ts, 60.0,
                ) * 0.001 * self._w.staleness

            score = prefix_gain - thermal_pen - staleness_pen
            if score > best_score:
                best_score = score
                best = c
        return best
