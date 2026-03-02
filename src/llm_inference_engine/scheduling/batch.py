"""Batch abstraction for grouping inference requests before dispatch."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import structlog

from llm_inference_engine.core.types import Request

logger = structlog.get_logger(__name__)

# Conservative bytes-per-token estimate used when no precise data is available.
_DEFAULT_BYTES_PER_TOKEN: int = 2


@dataclass
class Batch:
    """A group of requests dispatched together to the inference backend.

    The batch enforces configurable limits on the number of requests and
    total token budget.  Callers should use :meth:`can_add` before
    :meth:`add` to avoid violating constraints.

    Attributes:
        batch_id: Unique identifier for this batch.
        model: The Ollama model tag that all requests in this batch
            target.  Batches are always model-homogeneous.
        max_requests: Maximum number of requests this batch can hold.
        max_tokens: Maximum total token budget across all requests in
            the batch.  ``0`` means unlimited.
        requests: Ordered list of requests added to the batch.
    """

    batch_id: str
    model: str
    max_requests: int = 16
    max_tokens: int = 0  # 0 = unlimited
    requests: list[Request] = field(default_factory=list)
    _total_tokens: int = field(default=0, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of requests currently in the batch."""
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        """Sum of ``max_tokens`` across all requests in the batch.

        Maintained as an incremental counter in :meth:`add` — O(1) access
        instead of O(n) summation on every call.
        """
        return self._total_tokens

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no requests have been added."""
        return len(self.requests) == 0

    @property
    def is_full(self) -> bool:
        """Return ``True`` when the batch cannot accept any more requests."""
        return self.size >= self.max_requests

    @property
    def estimated_memory_bytes(self) -> int:
        """Rough memory footprint estimate based on token budgets.

        Uses a conservative ``2 bytes/token`` heuristic.  Phase 4 will
        replace this with the proper :class:`~llm_inference_engine.\
optimization.memory.MemoryEstimator`.

        Returns:
            Estimated memory in bytes.
        """
        return self.total_tokens * _DEFAULT_BYTES_PER_TOKEN

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def can_add(self, request: Request) -> bool:
        """Check whether *request* can be added without violating limits.

        Args:
            request: The candidate request.

        Returns:
            ``True`` if adding the request would not exceed
            ``max_requests`` or ``max_tokens`` (when set).
        """
        if self.size >= self.max_requests:
            return False
        if self.max_tokens > 0:
            new_total = self.total_tokens + request.generation_config.max_tokens
            if new_total > self.max_tokens:
                return False
        return True

    def add(self, request: Request) -> None:
        """Add *request* to the batch.

        Raises:
            ValueError: If adding the request would violate batch limits.
        """
        if not self.can_add(request):
            raise ValueError(
                f"Cannot add request {request.request_id!r} to batch "
                f"{self.batch_id!r}: limits exceeded "
                f"(size={self.size}/{self.max_requests}, "
                f"tokens={self.total_tokens}/{self.max_tokens or '∞'})"
            )
        self.requests.append(request)
        self._total_tokens += request.generation_config.max_tokens
        logger.debug(
            "request_added_to_batch",
            batch_id=self.batch_id,
            request_id=request.request_id,
            batch_size=self.size,
        )

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the requests in the batch."""
        return iter(self.requests)

    def __len__(self) -> int:
        return self.size


__all__ = ["Batch"]
