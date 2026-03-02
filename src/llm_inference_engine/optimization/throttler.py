"""Adaptive throttler for memory-aware request admission control.

The throttler tracks an estimate of currently committed memory and either
*accepts*, *queues* (soft limit), or *rejects* (hard limit) new requests
based on their predicted memory footprint versus the configured system
memory limit.
"""

import asyncio
from dataclasses import dataclass
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)

# Default hard memory ceiling for a 16 GB M2 Air (leaving ~2 GB for OS).
_DEFAULT_MEMORY_LIMIT_GB: float = 14.0


class AdmissionDecision(StrEnum):
    """Outcome of a throttler admission check.

    Attributes:
        ACCEPT: Request can be dispatched immediately.
        QUEUE: System is near capacity; request should wait.
        REJECT: System is at/over capacity; request should be dropped.
    """

    ACCEPT = "accept"
    QUEUE = "queue"
    REJECT = "reject"


@dataclass
class ThrottlerStats:
    """Snapshot of throttler state."""

    committed_gb: float
    available_gb: float
    memory_limit_gb: float
    soft_limit_gb: float
    active_requests: int


class AdaptiveThrottler:
    """Admission-control gate based on estimated memory pressure.

    The throttler maintains a *committed_gb* counter that tracks the
    sum of memory estimates for all active (in-flight) requests.  New
    requests are admitted when the projected total stays below the
    configured limits:

    - Below ``soft_limit_ratio × memory_limit_gb``:  **ACCEPT**
    - Between soft and hard limit:                  **QUEUE**
    - At or above ``memory_limit_gb``:              **REJECT**

    All methods are coroutine-safe.
    """

    def __init__(
        self,
        memory_limit_gb: float = _DEFAULT_MEMORY_LIMIT_GB,
        soft_limit_ratio: float = 0.85,
    ) -> None:
        """Initialise the throttler.

        Args:
            memory_limit_gb: Hard memory ceiling in gigabytes.
            soft_limit_ratio: Fraction of ``memory_limit_gb`` at which
                the throttler starts queuing new requests (default 0.85).
        """
        if memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        if not 0.0 < soft_limit_ratio < 1.0:
            raise ValueError("soft_limit_ratio must be between 0.0 and 1.0 (exclusive)")
        self._memory_limit_gb = memory_limit_gb
        self._soft_limit_gb = memory_limit_gb * soft_limit_ratio
        self._committed_gb: float = 0.0
        self._active_requests: int = 0
        self._lock = asyncio.Lock()
        logger.info(
            "throttler_initialized",
            memory_limit_gb=memory_limit_gb,
            soft_limit_gb=round(self._soft_limit_gb, 3),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, estimated_memory_gb: float) -> AdmissionDecision:
        """Evaluate whether a request with the given memory footprint can proceed.

        Args:
            estimated_memory_gb: Predicted memory consumption in GB for
                the new request (from :class:`~llm_inference_engine.\
optimization.memory.MemoryEstimator`).

        Returns:
            An :class:`AdmissionDecision` indicating whether to accept,
            queue, or reject the request.
        """
        if estimated_memory_gb < 0:
            raise ValueError("estimated_memory_gb cannot be negative")

        async with self._lock:
            projected = self._committed_gb + estimated_memory_gb

            if projected >= self._memory_limit_gb:
                decision = AdmissionDecision.REJECT
            elif projected >= self._soft_limit_gb:
                decision = AdmissionDecision.QUEUE
            else:
                decision = AdmissionDecision.ACCEPT

        logger.debug(
            "admission_check",
            estimated_gb=round(estimated_memory_gb, 3),
            committed_gb=round(self._committed_gb, 3),
            projected_gb=round(projected, 3),
            decision=decision,
        )
        return decision

    async def reserve(self, request_id: str, estimated_memory_gb: float) -> None:
        """Reserve memory for an admitted request.

        Should be called immediately after :meth:`check` returns
        :attr:`AdmissionDecision.ACCEPT` or :attr:`AdmissionDecision.QUEUE`
        once the request is ready to be dispatched.

        Args:
            request_id: Identifier of the request being dispatched.
            estimated_memory_gb: Memory amount to commit.
        """
        async with self._lock:
            self._committed_gb += estimated_memory_gb
            self._active_requests += 1
        logger.debug(
            "memory_reserved",
            request_id=request_id,
            reserved_gb=round(estimated_memory_gb, 3),
            committed_gb=round(self._committed_gb, 3),
        )

    async def release(self, request_id: str, estimated_memory_gb: float) -> None:
        """Release previously reserved memory when a request completes.

        Args:
            request_id: Identifier of the completed request.
            estimated_memory_gb: Memory amount to free.
        """
        async with self._lock:
            self._committed_gb = max(0.0, self._committed_gb - estimated_memory_gb)
            self._active_requests = max(0, self._active_requests - 1)
        logger.debug(
            "memory_released",
            request_id=request_id,
            released_gb=round(estimated_memory_gb, 3),
            committed_gb=round(self._committed_gb, 3),
        )

    @property
    def stats(self) -> ThrottlerStats:
        """Return a snapshot of the current throttler state."""
        return ThrottlerStats(
            committed_gb=self._committed_gb,
            available_gb=max(0.0, self._memory_limit_gb - self._committed_gb),
            memory_limit_gb=self._memory_limit_gb,
            soft_limit_gb=self._soft_limit_gb,
            active_requests=self._active_requests,
        )

    @property
    def committed_gb(self) -> float:
        """Currently committed memory in gigabytes."""
        return self._committed_gb

    @property
    def memory_limit_gb(self) -> float:
        """Hard memory ceiling in gigabytes."""
        return self._memory_limit_gb


__all__ = ["AdaptiveThrottler", "AdmissionDecision", "ThrottlerStats"]
