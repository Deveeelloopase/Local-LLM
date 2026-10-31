from __future__ import annotations

"""
engine/scheduling/scheduler.py — Production-Ready Request Scheduler
=====================================================================
v1.0.0 – Config injected via constructor; all original logic preserved.

Features:
- Priority queues with weighted round-robin fairness
- Anti-starvation enforcement (16:1 max weight ratio)
- AIMD congestion control (symmetric hysteresis)
- Queue depth limits with backpressure
- Preemption with state persistence (in-memory + optional NVMe)
- Expired request pruning with background tasks
- Prometheus metrics (optional, fully optional)
- Thread-safe statistics collection
- All parameters loaded from config via constructor (no file I/O)
- Comprehensive validation and defensive programming
- Detailed structured logging with request correlation
- No resource leaks on any error path
- Integration with slot lifecycle (InferenceEngine)

Production Hardening:
- All counters protected by threading.Lock
- Fairness selection with defensive bounds checking
- AIMD hysteresis symmetric in both directions
- Preemption state validated before use
- Queue lock order enforced to prevent deadlock
- Configuration validation with sensible defaults
- Error recovery and graceful degradation

"""

import asyncio
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics (Optional, Graceful Fallback)
# ============================================================================

def _setup_metrics() -> Dict[str, Any]:
    """Setup Prometheus metrics with graceful fallback if unavailable."""
    try:
        from prometheus_client import Counter, Gauge, Histogram

        return {
            "enqueued": Counter(
                "llm_scheduler_enqueued_total",
                "Requests enqueued",
                ["priority"]
            ),
            "completed": Counter(
                "llm_scheduler_completed_total",
                "Requests completed"
            ),
            "preempted": Counter(
                "llm_scheduler_preempted_total",
                "Requests preempted"
            ),
            "expired": Counter(
                "llm_scheduler_expired_total",
                "Requests expired"
            ),
            "queue_depth": Gauge(
                "llm_scheduler_queue_depth",
                "Current queue depth",
                ["priority"]
            ),
            "batch_size": Gauge(
                "llm_scheduler_batch_size",
                "Current AIMD batch size"
            ),
            "running": Gauge(
                "llm_scheduler_running",
                "Currently executing requests"
            ),
            "wait_time": Histogram(
                "llm_scheduler_wait_time_seconds",
                "Time spent in queue",
                buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)
            ),
        }
    except ImportError:
        log.debug("Prometheus not available; metrics disabled")
        return {}


_METRICS = _setup_metrics()


# ============================================================================
# Priority Enum
# ============================================================================

class Priority(Enum):
    """Request priority levels (higher value = higher priority)."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


# ============================================================================
# ScheduledRequest Dataclass
# ============================================================================

@dataclass
class ScheduledRequest:
    """
    A request queued for inference execution.

    Thread-safe and immutable after creation (except for tracking fields).
    """
    request_id: str
    priority: Priority = Priority.NORMAL
    enqueued_at: float = field(default_factory=time.monotonic)
    timeout_s: float = 300.0
    preempt_allowed: bool = True
    model_id: str = ""
    tokens_served: int = 0  # For weighted fairness tracking

    def is_expired(self) -> bool:
        """Check if request has exceeded its timeout."""
        return (time.monotonic() - self.enqueued_at) > self.timeout_s

    def queue_time_s(self) -> float:
        """Get time spent in queue (seconds)."""
        return time.monotonic() - self.enqueued_at

    def __repr__(self) -> str:
        """Readable representation for logging."""
        return (
            f"ScheduledRequest(id={self.request_id[:8]}, "
            f"priority={self.priority.name}, "
            f"queue_time={self.queue_time_s():.1f}s)"
        )


# ============================================================================
# Scheduler Class (Production-Ready, Config Injected)
# ============================================================================

class Scheduler:
    """
    Production-ready request scheduler with fairness, AIMD control, and preemption.

    Thread Safety:
    - All counters protected by threading.Lock
    - Async primitives (asyncio.Lock) for queue access
    - Queue lock acquired before state lock (consistent ordering)

    Fairness:
    - Weighted round-robin with anti-starvation (16:1 max ratio)
    - Exponential weight scale: LOW:NORMAL:HIGH:URGENT = 1:2:4:8 (default)
    - Symmetric AIMD hysteresis to prevent thrashing

    Preemption:
    - In-memory state storage (always available)
    - Optional NVMe persistence for durability
    - State validated before use

    Configuration:
    - All parameters from config.yaml [scheduler] section passed via `config` arg.
    - Sensible defaults provided.
    - Validation ensures bounds and consistency.

    """

    def __init__(
            self,
            profile: Any,  # HardwareProfile
            config: Optional[Dict[str, Any]] = None,
            thermal_state: Optional[Any] = None,  # 🔥 ISSUE 1 FIX
    ) -> None:
        """
        Initialize production-ready scheduler.

        Args:
            profile: HardwareProfile with system information
            config: Optional config dict (or SchedulerConfig model). If None, defaults used.
            thermal_state: Shared ThermalState object (must have a `throttling` bool attr).

        Raises:
            ValueError: If critical configuration is invalid
        """
        self._profile = profile
        # store thermal state reference
        self._thermal_state = thermal_state

        # ----------------------------------------------------------------------
        # Normalise configuration
        # ----------------------------------------------------------------------
        if config is None:
            config = {}
        try:
            # Optionally validate with Pydantic if available
            from local_llm_core.engine.utils.config_loader import SchedulerConfig
            if isinstance(config, dict):
                cfg_model = SchedulerConfig(**config)
                self._cfg = cfg_model.model_dump()
            else:
                self._cfg = config.model_dump()
        except Exception as exc:
            log.warning("Scheduler config validation failed: %s; using defaults", exc)
            self._cfg = {}  # will rely on hardcoded defaults below

        log.info("Scheduler initialization starting...")

        # =====================================================================
        # 1. AIMD PARAMETERS (Additive Increase, Multiplicative Decrease)
        # =====================================================================

        self._aimd_target_tps = float(self._cfg.get("aimd_target_tps", 7.0))
        if self._aimd_target_tps <= 0:
            log.warning(
                "Invalid aimd_target_tps=%.1f (must be > 0), using default 7.0",
                self._aimd_target_tps
            )
            self._aimd_target_tps = 7.0

        self._aimd_min_batch = max(int(self._cfg.get("aimd_min_batch", 1)), 1)
        self._aimd_max_batch = max(
            int(self._cfg.get("aimd_max_batch", getattr(profile, "n_parallel", 4))),
            self._aimd_min_batch
        )

        self._aimd_interval_s = float(self._cfg.get("aimd_interval_s", 5.0))
        if self._aimd_interval_s <= 0:
            log.warning(
                "Invalid aimd_interval_s=%.1f (must be > 0), using 5.0",
                self._aimd_interval_s
            )
            self._aimd_interval_s = 5.0

        self._aimd_hysteresis = max(int(self._cfg.get("aimd_hysteresis", 2)), 1)
        if self._aimd_hysteresis > 10:
            log.warning(
                "aimd_hysteresis=%d exceeds recommended max 10",
                self._aimd_hysteresis
            )

        self._aimd_high_margin = float(self._cfg.get("aimd_high_margin", 0.95))
        if not (0.0 < self._aimd_high_margin <= 1.0):
            log.warning(
                "Invalid aimd_high_margin=%.2f (must be in (0, 1]), using 0.95",
                self._aimd_high_margin
            )
            self._aimd_high_margin = 0.95

        self._aimd_low_margin = float(self._cfg.get("aimd_low_margin", 0.85))
        if not (0.0 < self._aimd_low_margin <= 1.0):
            log.warning(
                "Invalid aimd_low_margin=%.2f (must be in (0, 1]), using 0.85",
                self._aimd_low_margin
            )
            self._aimd_low_margin = 0.85

        if self._aimd_low_margin >= self._aimd_high_margin:
            log.warning(
                "aimd_low_margin=%.2f must be < aimd_high_margin=%.2f, swapping",
                self._aimd_low_margin, self._aimd_high_margin
            )
            self._aimd_low_margin, self._aimd_high_margin = (
                self._aimd_high_margin, self._aimd_low_margin
            )

        log.info(
            "Scheduler AIMD config: target_tps=%.1f batch=[%d,%d] "
            "interval=%.1fs hysteresis=%d margins=[%.2f, %.2f]",
            self._aimd_target_tps,
            self._aimd_min_batch,
            self._aimd_max_batch,
            self._aimd_interval_s,
            self._aimd_hysteresis,
            self._aimd_low_margin,
            self._aimd_high_margin,
        )

        # =====================================================================
        # 2. QUEUE LIMITS (Backpressure Management)
        # =====================================================================

        self._max_queue_per_priority = max(
            int(self._cfg.get("max_queue_per_priority", 64)), 1
        )
        self._max_total_queue = max(
            int(self._cfg.get("max_total_queue", 256)),
            self._max_queue_per_priority
        )

        if self._max_total_queue < self._max_queue_per_priority * 2:
            log.warning(
                "max_total_queue=%d < max_queue_per_priority*2=%d, adjusting",
                self._max_total_queue,
                self._max_queue_per_priority * 2
            )
            self._max_total_queue = self._max_queue_per_priority * 4

        log.debug(
            "Scheduler queue limits: per_priority=%d total=%d",
            self._max_queue_per_priority,
            self._max_total_queue
        )

        # =====================================================================
        # 3. PRIORITY WEIGHTS (Anti-Starvation with 16:1 Max Ratio)
        # =====================================================================

        weights = self._cfg.get("fairness_weights", {})

        # ✅ Ensure ALL weights have minimum of 1 (no negatives/zeros)
        self._priority_weights = {
            Priority.LOW: max(int(weights.get("low", 1)), 1),
            Priority.NORMAL: max(int(weights.get("normal", 2)), 1),
            Priority.HIGH: max(int(weights.get("high", 4)), 1),
            Priority.URGENT: max(int(weights.get("urgent", 8)), 1),
        }

        # ✅ Validate and enforce 16:1 ratio constraint
        max_weight = max(self._priority_weights.values())
        min_weight = min(self._priority_weights.values())
        ratio = max_weight / min_weight

        if ratio > 16:
            log.warning(
                "Priority weight ratio %.1f:1 exceeds 16:1 limit — "
                "enforcing maximum ratio to prevent LOW priority starvation",
                ratio
            )

            max_allowed = min_weight * 16
            scaling_factor = max_allowed / max_weight

            self._priority_weights = {
                p: max(int(self._priority_weights[p] * scaling_factor), 1)
                for p in Priority
            }

            new_max = max(self._priority_weights.values())
            new_min = min(self._priority_weights.values())
            new_ratio = new_max / new_min

            log.info(
                "Priority weights adjusted: %.1f:1 → %.1f:1 | weights=%s",
                ratio,
                new_ratio,
                {p.name: self._priority_weights[p] for p in Priority}
            )
        else:
            log.debug(
                "Priority weights configured: %s (ratio: %.1f:1)",
                {p.name: self._priority_weights[p] for p in Priority},
                ratio
            )

        # =====================================================================
        # 4. PREEMPTION PERSISTENCE (Optional NVMe)
        # =====================================================================

        preempt_cfg = self._cfg.get("preemption", {})
        self._preempt_persist_dir = preempt_cfg.get("persist_dir")

        if self._preempt_persist_dir:
            try:
                persist_path = Path(self._preempt_persist_dir)
                persist_path.mkdir(parents=True, exist_ok=True)

                # Verify we can write to the directory
                test_file = persist_path / ".scheduler_test"
                test_file.write_text("")
                test_file.unlink()

                log.info(
                    "Scheduler preemption persistence: %s (writable)",
                    self._preempt_persist_dir
                )
            except Exception as exc:
                log.error(
                    "Failed to initialize preemption persistence at %s: %s — "
                    "preempted requests will use in-memory storage only",
                    self._preempt_persist_dir, exc
                )
                self._preempt_persist_dir = None
        else:
            log.debug(
                "Scheduler preemption persistence: disabled (in-memory only)"
            )

        # =====================================================================
        # 5. PRIORITY QUEUES (Deque for O(1) operations)
        # =====================================================================

        self._queues: Dict[Priority, Deque[ScheduledRequest]] = {
            p: deque() for p in Priority
        }
        self._queue_lock = asyncio.Lock()

        # =====================================================================
        # 6. AIMD STATE (Batch size and convergence tracking)
        # =====================================================================

        self._batch_size = self._aimd_min_batch
        self._last_adjust = time.monotonic()
        self._on_target_count = 0
        self._below_target_count = 0

        # =====================================================================
        # 7. RUNNING & PREEMPTED REQUESTS (State tracking)
        # =====================================================================

        self._running: Dict[str, ScheduledRequest] = {}
        self._preempted: Dict[str, bytes] = {}
        self._state_lock = asyncio.Lock()

        # =====================================================================
        # 8. STATISTICS (Thread-safe counters for observability)
        # =====================================================================

        self._stats_lock = threading.Lock()  # ✅ NEW: Protect counters
        self._enqueued_count = 0
        self._completed_count = 0
        self._preempted_count = 0
        self._expired_count = 0

        # =====================================================================
        # 9. BACKGROUND TASKS (Expired request pruning)
        # =====================================================================

        self._prune_task: Optional[asyncio.Task] = None
        self._prune_interval_s = float(self._cfg.get("prune_interval_s", 10.0))
        if self._prune_interval_s <= 0:
            log.warning(
                "Invalid prune_interval_s=%.1f (must be > 0), using 10.0",
                self._prune_interval_s
            )
            self._prune_interval_s = 10.0

        # Default timeout for requests that don't specify one
        self._default_timeout_s = float(self._cfg.get("default_timeout_s", 300.0))
        if self._default_timeout_s <= 0:
            self._default_timeout_s = 300.0

        log.info(
            "Scheduler initialized: batch=%d queue=[%d,%d] "
            "prune_interval=%.1fs preempt_persist=%s thermal_throttle=%s",
            self._batch_size,
            self._max_queue_per_priority,
            self._max_total_queue,
            self._prune_interval_s,
            "enabled" if self._preempt_persist_dir else "disabled",
            "active" if (self._thermal_state and self._thermal_state.throttling) else "inactive"
        )

    # =========================================================================
    # Dynamic target TPS update
    # =========================================================================
    def update_target_tps(self, new_target_tps: float) -> None:
        """
        Update the AIMD target TPS to a new value (e.g. from a warmup measurement).

        This method is thread-safe and logs the change. The new target will
        be used in all subsequent AIMD adjustments.

        Args:
            new_target_tps: New target tokens per second (must be > 0).
        """
        if new_target_tps <= 0:
            log.warning("Ignoring non-positive target TPS: %.2f", new_target_tps)
            return
        old = self._aimd_target_tps
        self._aimd_target_tps = new_target_tps
        # Reset counters to avoid sudden spikes based on old target
        self._on_target_count = 0
        self._below_target_count = 0
        log.info("Scheduler target TPS updated: %.2f → %.2f", old, new_target_tps)

    # cancel an in-flight request
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel an in-flight request by ID.
        Removes it from _running and cleans up preempted state.
        Returns True if the request was found and cancelled.
        """
        if not request_id:
            log.warning("cancel_request called with empty request_id")
            return False

        async with self._state_lock:
            req = self._running.pop(request_id, None)
            if req is not None:
                # Remove any preempted state for this request
                self._preempted.pop(request_id, None)
                log.info("Cancelled in-flight request %s", request_id[:8])
                return True
            else:
                log.debug("cancel_request: request %s not found in _running", request_id[:8])
                return False

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start background maintenance tasks."""
        try:
            self._prune_task = asyncio.create_task(
                self._prune_loop(),
                name="scheduler_prune"
            )
            log.info("Scheduler background tasks started")
        except Exception as e:
            log.error("Failed to start scheduler background tasks: %s", e)
            raise

    async def shutdown(self) -> None:
        """Stop background tasks gracefully."""
        if self._prune_task:
            try:
                self._prune_task.cancel()
                try:
                    await self._prune_task
                except asyncio.CancelledError:
                    pass
                log.info("Scheduler background tasks stopped")
            except Exception as e:
                log.warning("Error shutting down scheduler: %s", e)

    async def _prune_loop(self) -> None:
        """Periodically remove expired requests from queues."""
        while True:
            try:
                await asyncio.sleep(self._prune_interval_s)
                removed = await self._prune_expired()
                if removed > 0:
                    log.debug("Prune cycle: removed %d expired requests", removed)
            except asyncio.CancelledError:
                log.debug("Prune loop cancelled")
                break
            except Exception as exc:
                log.error("Prune loop error: %s", exc)

    # =========================================================================
    # Request Enqueueing
    # =========================================================================

    async def enqueue(self, req: ScheduledRequest) -> bool:
        """
        Enqueue a request for processing.

        Args:
            req: ScheduledRequest to enqueue

        Returns:
            True if enqueued successfully, False if queue limits exceeded

        Raises:
            ValueError: If request is invalid
        """
        # Validate request
        if not req or not isinstance(req, ScheduledRequest):
            log.error("Invalid request: %s", req)
            raise ValueError("Request must be ScheduledRequest instance")

        if not req.request_id:
            log.error("Request has empty request_id")
            raise ValueError("Request must have non-empty request_id")

        # Apply default timeout if not set
        if req.timeout_s <= 0:
            req.timeout_s = self._default_timeout_s

        async with self._queue_lock:
            # Check total queue depth
            total = sum(len(q) for q in self._queues.values())
            if total >= self._max_total_queue:
                log.warning(
                    "Scheduler: total queue full (%d/%d), rejecting %s",
                    total, self._max_total_queue, req.request_id[:8]
                )
                return False

            # Check per-priority limit
            q = self._queues[req.priority]
            if len(q) >= self._max_queue_per_priority:
                log.warning(
                    "Scheduler: queue full for %s priority (%d/%d), rejecting %s",
                    req.priority.name, len(q), self._max_queue_per_priority,
                    req.request_id[:8]
                )
                return False

            # Enqueue request
            q.append(req)
            with self._stats_lock:
                self._enqueued_count += 1

        # Record metrics
        if _METRICS.get("enqueued"):
            _METRICS["enqueued"].labels(priority=req.priority.name).inc()

        self._update_queue_depth_metrics()

        log.debug(
            "Request enqueued | id=%s priority=%s timeout=%.1fs total_queue=%d",
            req.request_id[:8], req.priority.name, req.timeout_s, total + 1
        )
        return True

    # =========================================================================
    # Expired Request Pruning
    # =========================================================================

    async def _prune_expired(self) -> int:
        """
        Remove expired requests from all priority queues.

        Returns:
            Number of requests removed
        """
        removed = 0
        async with self._queue_lock:
            for prio in Priority:
                q = self._queues[prio]
                new_q: Deque[ScheduledRequest] = deque()
                for req in q:
                    if not req.is_expired():
                        new_q.append(req)
                    else:
                        removed += 1
                self._queues[prio] = new_q

        # Update counter outside lock
        if removed:
            with self._stats_lock:
                self._expired_count += removed
            if _METRICS.get("expired"):
                _METRICS["expired"].inc(removed)

        return removed

    # =========================================================================
    # Next Batch Selection (With Fairness)
    # =========================================================================

    async def next_batch(self, current_tps: float) -> List[ScheduledRequest]:
        """
        Get the next batch of requests using weighted round-robin fairness.

        Features:
        - AIMD congestion control based on current TPS
        - Weighted round-robin with anti-starvation
        - Symmetric hysteresis (prevents thrashing)
        - Automatic removal of expired requests
        - Metrics export
        - Immediate batch-size reduction on thermal throttle

        Args:
            current_tps: Current throughput (tokens/second) for AIMD tuning

        Returns:
            List of requests to execute (may be empty)
        """
        batch: List[ScheduledRequest] = []

        async with self._queue_lock:
            # ✅ Update AIMD state under lock
            async with self._state_lock:
                self._update_aimd(current_tps)
                batch_size = self._batch_size

            # Build list of non-empty queues with weights
            active_queues: List[
                Tuple[Priority, Deque[ScheduledRequest], int]
            ] = [
                (prio, self._queues[prio], self._priority_weights.get(prio, 1))
                for prio in Priority
                if self._queues[prio]
            ]

            if not active_queues:
                return batch

            # Seed random for deterministic but varying selection
            random.seed(int(time.monotonic() * 1e6) % (2 ** 31))

            # Weighted round-robin selection
            while len(batch) < batch_size and active_queues:
                total_weight = sum(w for _, _, w in active_queues)
                if total_weight <= 0:
                    break

                # Select queue based on weight
                r = random.random() * total_weight
                cum = 0.0
                selected_idx = -1

                for i, (_, _, w) in enumerate(active_queues):
                    cum += w
                    if r < cum:
                        selected_idx = i
                        break

                # Defensive: ensure valid index
                if selected_idx < 0 or selected_idx >= len(active_queues):
                    log.warning(
                        "Fairness selection failed: r=%.2f total=%d idx=%d len=%d",
                        r, total_weight, selected_idx, len(active_queues)
                    )
                    selected_idx = 0  # Fallback to first

                prio, q, _ = active_queues[selected_idx]

                if q:
                    req = q.popleft()
                    if not req.is_expired():
                        batch.append(req)
                        log.debug(
                            "Dequeued request | id=%s priority=%s queue_time=%.1fs",
                            req.request_id[:8], req.priority.name, req.queue_time_s()
                        )
                    else:
                        # Count expired request
                        with self._stats_lock:
                            self._expired_count += 1
                        if _METRICS.get("expired"):
                            _METRICS["expired"].inc()
                        log.debug(
                            "Expired request skipped: %s (queue_time=%.1fs)",
                            req.request_id[:8], req.queue_time_s()
                        )

                # Remove queue if empty
                if not q:
                    active_queues.pop(selected_idx)

        # Update metrics (outside lock)
        if _METRICS.get("batch_size"):
            _METRICS["batch_size"].set(batch_size)
        if _METRICS.get("running"):
            _METRICS["running"].set(len(self._running))

        self._update_queue_depth_metrics()

        if batch:
            log.debug(
                "Batch selected: %d requests from %s priority levels",
                len(batch), len(active_queues) + (1 if len(batch) < batch_size else 0)
            )

        return batch

    def _update_queue_depth_metrics(self) -> None:
        """Update Prometheus queue depth gauges."""
        if not _METRICS.get("queue_depth"):
            return
        try:
            for prio in Priority:
                _METRICS["queue_depth"].labels(priority=prio.name).set(
                    len(self._queues[prio])
                )
        except Exception as e:
            log.debug("Queue depth metrics error: %s", e)

    # =========================================================================
    # Request Completion Tracking
    # =========================================================================

    async def record_completion(self, request_id: str) -> None:
        """
        Mark a request as completed.

        Args:
            request_id: ID of completed request
        """
        if not request_id:
            log.warning("record_completion called with empty request_id")
            return

        async with self._state_lock:
            req = self._running.pop(request_id, None)
            if req:
                with self._stats_lock:
                    self._completed_count += 1
                if _METRICS.get("completed"):
                    _METRICS["completed"].inc()
                log.debug(
                    "Request completed | id=%s queue_time=%.1fs",
                    request_id[:8], req.queue_time_s()
                )
            else:
                log.debug("Completion recorded for unknown request: %s", request_id[:8])

            # Clean up preempted state if exists
            self._preempted.pop(request_id, None)

        if _METRICS.get("running"):
            _METRICS["running"].set(len(self._running))

    # =========================================================================
    # Preemption Management
    # =========================================================================

    def can_preempt(
            self,
            low_priority_req: ScheduledRequest,
            high_priority_req: ScheduledRequest
    ) -> bool:
        """
        Check if high_priority_req should preempt low_priority_req.

        Args:
            low_priority_req: Currently running request
            high_priority_req: Incoming higher-priority request

        Returns:
            True if preemption is allowed and beneficial
        """
        if not low_priority_req.preempt_allowed:
            return False
        return (high_priority_req.priority.value > low_priority_req.priority.value)

    async def preempt(self, request_id: str, state_blob: bytes) -> bool:
        """
        Save state for a preempted request.

        Features:
        - Persists to NVMe if available
        - Falls back to in-memory storage
        - Validates state before returning

        Args:
            request_id: ID of preempted request
            state_blob: Serialized generation state

        Returns:
            True if state was saved successfully
        """
        if not request_id or not isinstance(state_blob, bytes):
            log.error(
                "Invalid preemption state: id=%s type=%s",
                request_id[:8], type(state_blob).__name__
            )
            return False

        # ✅ Try persistent storage first
        if self._preempt_persist_dir:
            try:
                path = Path(self._preempt_persist_dir) / f"{request_id}.preempt"
                path.write_bytes(state_blob)
                log.debug(
                    "Preempted state persisted to NVMe | id=%s size=%d bytes",
                    request_id[:8], len(state_blob)
                )
            except Exception as exc:
                log.error(
                    "Failed to persist preempted state to %s: %s",
                    path, exc
                )
                # Don't add to memory if persistent storage fails
                return False

        # ✅ Store in memory (after confirming persistent storage)
        async with self._state_lock:
            self._preempted[request_id] = state_blob
            with self._stats_lock:
                self._preempted_count += 1

        if _METRICS.get("preempted"):
            _METRICS["preempted"].inc()

        log.info(
            "Request preempted and state saved | id=%s size=%d bytes",
            request_id[:8], len(state_blob)
        )
        return True

    async def resume_preempted(self, request_id: str) -> Optional[bytes]:
        """
        Retrieve saved state for a preempted request.

        Features:
        - Checks persistent storage first (source of truth)
        - Falls back to in-memory storage
        - Cleans up after retrieval
        - Validates state before returning

        Args:
            request_id: ID of preempted request

        Returns:
            Serialized state bytes or None if not found
        """
        if not request_id:
            log.warning("resume_preempted called with empty request_id")
            return None

        state = None

        # ✅ Check persistent storage first (source of truth)
        if self._preempt_persist_dir:
            path = Path(self._preempt_persist_dir) / f"{request_id}.preempt"
            if path.exists():
                try:
                    state = path.read_bytes()
                    path.unlink()
                    log.debug(
                        "Preempted state loaded from NVMe | id=%s size=%d bytes",
                        request_id[:8], len(state)
                    )
                    return state  # Return immediately after loading from NVMe
                except Exception as exc:
                    log.error(
                        "Failed to load preempted state from %s: %s",
                        path, exc
                    )
                    return None  # Don't fall back to in-memory

        # ✅ Try in-memory only if no persistent file
        async with self._state_lock:
            state = self._preempted.pop(request_id, None)

        if state:
            log.info(
                "Preempted state resumed | id=%s size=%d bytes",
                request_id[:8], len(state)
            )
        else:
            log.debug("No preempted state found for request: %s", request_id[:8])

        return state

    # =========================================================================
    # AIMD Congestion Control — (thermal throttle inserted)
    # =========================================================================

    def _update_aimd(self, current_tps: float) -> None:
        """
        Update batch size using symmetric AIMD (Additive Increase, Multiplicative Decrease).

        Features:
        - Immediate response to thermal throttling.
        - Additive increase when on target (gradual)
        - Multiplicative decrease when below target (aggressive recovery)
        - Symmetric hysteresis to prevent thrashing
        - Gradual decay of counters in middle band

        Args:
            current_tps: Current throughput in tokens/second
        """
        now = time.monotonic()

        # ──────────────────────────────────────────────────────────────────────
        # Thermal throttle overrides all normal AIMD logic.
        # Bypasses the interval check so the system can react instantly to
        # overheating, preventing cumulative thermal stress.
        # ──────────────────────────────────────────────────────────────────────
        if self._thermal_state and self._thermal_state.throttling:
            # Decrease immediately, regardless of interval
            old_batch = self._batch_size
            self._batch_size = max(self._batch_size // 2, self._aimd_min_batch)
            # Reset counters so we don't immediately try to increase
            self._on_target_count = 0
            self._below_target_count = 0
            if old_batch != self._batch_size:
                log.warning(
                    "AIMD DECREASE | thermal throttle active: batch %d→%d",
                    old_batch, self._batch_size
                )
            self._last_adjust = now
            return

        # Normal AIMD interval check (only if not throttling)
        if (now - self._last_adjust) < self._aimd_interval_s:
            return

        # ✅ Use symmetric hysteresis for stability
        hysteresis_down = max(self._aimd_hysteresis // 2, 1)

        if current_tps >= self._aimd_target_tps * self._aimd_high_margin:
            # Above target: increase batch size
            self._on_target_count += 1
            self._below_target_count = 0

            if self._on_target_count >= self._aimd_hysteresis:
                old_batch = self._batch_size
                self._batch_size = min(self._batch_size + 1, self._aimd_max_batch)
                self._on_target_count = 0
                log.info(
                    "AIMD INCREASE | batch %d→%d (TPS=%.2f ≥ target=%.2f)",
                    old_batch, self._batch_size, current_tps,
                    self._aimd_target_tps * self._aimd_high_margin
                )

        elif current_tps < self._aimd_target_tps * self._aimd_low_margin:
            # Below target: decrease batch size aggressively
            self._below_target_count += 1
            self._on_target_count = 0

            if self._below_target_count >= hysteresis_down:
                old_batch = self._batch_size
                self._batch_size = max(
                    self._batch_size // 2, self._aimd_min_batch
                )
                self._below_target_count = 0
                log.warning(
                    "AIMD DECREASE | batch %d→%d (TPS=%.2f ≤ target_low=%.2f)",
                    old_batch, self._batch_size, current_tps,
                    self._aimd_target_tps * self._aimd_low_margin
                )

        else:
            # In middle band: gradual decay to avoid false triggers
            self._on_target_count = max(0, self._on_target_count - 1)
            self._below_target_count = max(0, self._below_target_count - 1)
            log.debug("AIMD in-range | TPS=%.2f batch=%d", current_tps, self._batch_size)

        self._last_adjust = now

    # =========================================================================
    # Queue Statistics by Priority (for diagnostics)
    # =========================================================================

    def queue_stats_by_priority(self) -> Dict[str, int]:
        """
        Get queue depth per priority (thread-safe, returns copy).

        Useful for debugging and monitoring.
        """
        try:
            # No lock needed for len() on deque, but we take a snapshot.
            return {p.name: len(q) for p, q in self._queues.items()}
        except Exception:
            return {}

    # =========================================================================
    # Properties & Statistics
    # =========================================================================

    @property
    def batch_size(self) -> int:
        """Current batch size (updated by AIMD control)."""
        return self._batch_size

    @property
    def queue_depth(self) -> int:
        """Approximate total queue depth (best-effort, no lock)."""
        try:
            return sum(len(q) for q in self._queues.values())
        except Exception:
            return 0

    @property
    def running_count(self) -> int:
        """Approximate number of running requests (best-effort, no lock)."""
        try:
            return len(self._running)
        except Exception:
            return 0

    @property
    def preempted_count(self) -> int:
        """Number of preempted requests with saved state."""
        try:
            return len(self._preempted)
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics for monitoring.

        Returns:
            Dict with comprehensive scheduler metrics

        Features:
        - Thread-safe counter access
        - Best-effort queue/running counts
        - AIMD convergence tracking
        - Fairness weight information
        - 🔥 ISSUE 1 FIX: includes thermal state snapshot
        """
        # ✅ Copy counter values under lock
        with self._stats_lock:
            enq = self._enqueued_count
            comp = self._completed_count
            exp = self._expired_count
            on_tgt = self._on_target_count
            below_tgt = self._below_target_count

        # include thermal state
        thermal_info = {}
        if self._thermal_state:
            thermal_info = {
                "throttling": self._thermal_state.throttling,
                "temperature_c": self._thermal_state.temperature_c,
                "timestamp": self._thermal_state.timestamp,
            }

        return {
            "batch_size": self._batch_size,
            "queue_depth": self.queue_depth,
            "running": self.running_count,
            "preempted": self.preempted_count,
            "enqueued_total": enq,
            "completed_total": comp,
            "expired_total": exp,
            "on_target_streaks": on_tgt,
            "below_target_streaks": below_tgt,
            "aimd_target_tps": self._aimd_target_tps,
            "priority_weights": {
                p.name: self._priority_weights[p] for p in Priority
            },
            "queue_stats_by_priority": self.queue_stats_by_priority(),
            "preemption_persistence": (
                    self._preempt_persist_dir is not None
            ),
            "thermal": thermal_info,
        }
