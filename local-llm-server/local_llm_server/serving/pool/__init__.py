"""serving.pool — shared-weight process pool for concurrency."""
__version__ = "1.0.0"

from .pool import SharedWeightProcessPool, PoolUnavailable
from .supervisor import WorkerSupervisor
from .router import CAASRouter, CAASWeights, IdleCandidate
from .handle import WorkerHandle, WorkerStatusSnapshot

__all__ = [
    "SharedWeightProcessPool",
    "PoolUnavailable",
    "WorkerSupervisor",
    "CAASRouter", "CAASWeights", "IdleCandidate",
    "WorkerHandle", "WorkerStatusSnapshot",
    "__version__",
]
