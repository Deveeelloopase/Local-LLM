"""
engine/scheduling/__init__.py — Production‑Ready Request Scheduling
===================================================================
Exports the scheduler components for request queuing, AIMD congestion control,
priority‑based preemption, and batch packing.

All configuration is read from config.yaml by the Scheduler class.
No hard‑coded values; everything is config‑driven.

Public API:
- Scheduler: Main scheduling class.
- ScheduledRequest: Dataclass for queued requests.
- Priority: Enum for request priority levels.
"""

__version__ = "1.0.0"

# -----------------------------------------------------------------------------
# Scheduler Components
# -----------------------------------------------------------------------------
try:
    from .scheduler import (
        Scheduler,
        ScheduledRequest,
        Priority,
    )
except ImportError as e:
    Scheduler = None
    ScheduledRequest = None
    Priority = None
    import logging
    logging.getLogger(__name__).warning("scheduler import failed: %s", e)

# -----------------------------------------------------------------------------
# Public Exports
# -----------------------------------------------------------------------------
__all__ = [
    "Scheduler",
    "ScheduledRequest",
    "Priority",
    "__version__",
]
