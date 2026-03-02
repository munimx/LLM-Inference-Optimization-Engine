"""Thread-safe, priority-aware request queue for the scheduling engine."""

import asyncio
from dataclasses import dataclass, field

import structlog

from llm_inference_engine.core.types import Request, RequestStatus

logger = structlog.get_logger(__name__)


@dataclass(order=True)
class _PriorityItem:
    """Internal wrapper for priority-queue ordering.

    Lower ``sort_key`` values are dequeued first.  We negate the user
    priority so that *higher* priority values are dequeued first, and
    use ``sequence`` as a tie-breaker that preserves FIFO order among
    equal priorities.
    """

    sort_key: tuple[int, int] = field(compare=True)
    request: Request = field(compare=False)


class RequestQueue:
    """Async, priority-aware queue for inference requests.

    Requests with a higher ``priority`` value are dequeued before lower-
    priority ones.  Within the same priority level, FIFO ordering is
    preserved via a monotonically increasing sequence counter.

    All public methods are coroutine-safe and can be called concurrently.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Initialise the request queue.

        Args:
            maxsize: Maximum number of items in the queue.  ``0`` means
                unbounded (default).
        """
        self._queue: asyncio.PriorityQueue[_PriorityItem] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._cancelled: set[str] = set()
        self._queued_ids: set[str] = set()
        self._sequence: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        logger.info("request_queue_initialized", maxsize=maxsize)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(self, request: Request) -> None:
        """Add a request to the queue.

        Args:
            request: The inference request to enqueue.  Its ``status``
                will be updated to :attr:`RequestStatus.PENDING`.
        """
        async with self._lock:
            seq = self._sequence
            self._sequence += 1
            self._queued_ids.add(request.request_id)

        # Higher user priority → smaller sort key → dequeued first.
        sort_key = (-request.priority, seq)
        item = _PriorityItem(sort_key=sort_key, request=request)
        request.status = RequestStatus.PENDING
        await self._queue.put(item)
        logger.debug(
            "request_enqueued",
            request_id=request.request_id,
            priority=request.priority,
            queue_size=self._queue.qsize(),
        )

    async def dequeue(self) -> Request | None:
        """Remove and return the highest-priority non-cancelled request.

        Cancelled requests encountered during dequeue are silently
        discarded.  Returns ``None`` only when the queue is empty *and*
        no further items are expected (i.e., caller should use
        :meth:`get_nowait` variant for non-blocking behaviour).

        Returns:
            The next eligible :class:`~llm_inference_engine.core.types.Request`,
            or ``None`` if the queue is empty.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

            if item.request.request_id in self._cancelled:
                self._cancelled.discard(item.request.request_id)
                self._queued_ids.discard(item.request.request_id)
                self._queue.task_done()
                logger.debug(
                    "cancelled_request_skipped",
                    request_id=item.request.request_id,
                )
                continue

            self._queue.task_done()
            self._queued_ids.discard(item.request.request_id)
            logger.debug(
                "request_dequeued",
                request_id=item.request.request_id,
                priority=item.request.priority,
            )
            return item.request

    async def dequeue_wait(self) -> Request:
        """Block until a non-cancelled request is available and return it.

        Returns:
            The next eligible :class:`~llm_inference_engine.core.types.Request`.
        """
        while True:
            item = await self._queue.get()
            if item.request.request_id in self._cancelled:
                self._cancelled.discard(item.request.request_id)
                self._queued_ids.discard(item.request.request_id)
                self._queue.task_done()
                continue
            self._queue.task_done()
            self._queued_ids.discard(item.request.request_id)
            return item.request

    def cancel(self, request_id: str) -> bool:
        """Mark a request as cancelled so it is skipped on dequeue.

        Only requests that are currently enqueued (tracked in
        ``_queued_ids``) are accepted.  This bounds the ``_cancelled``
        set to at most the current queue depth, preventing unbounded
        growth from stale or phantom cancellation calls.

        Args:
            request_id: The ID of the request to cancel.

        Returns:
            ``True`` if the request was newly marked as cancelled,
            ``False`` if it was already cancelled or not in the queue.
        """
        if request_id not in self._queued_ids:
            return False
        if request_id in self._cancelled:
            return False
        self._cancelled.add(request_id)
        logger.info("request_cancelled", request_id=request_id)
        return True

    @property
    def size(self) -> int:
        """Current number of items in the queue (including cancelled)."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Return ``True`` if the queue contains no items."""
        return self._queue.empty()

    def is_cancelled(self, request_id: str) -> bool:
        """Check whether a request has been cancelled.

        Args:
            request_id: The request ID to query.
        """
        return request_id in self._cancelled


__all__ = ["RequestQueue"]
