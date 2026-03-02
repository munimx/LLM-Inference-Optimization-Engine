"""ResultMapper: correlates async Ollama responses with originating requests."""

import asyncio
from dataclasses import dataclass, field

import structlog

from llm_inference_engine.core.types import Response

logger = structlog.get_logger(__name__)


@dataclass
class _PendingRequest:
    """Tracks a pending request waiting for its response."""

    future: asyncio.Future[Response]
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class ResultMapper:
    """Maps in-flight request IDs to async futures for response delivery.

    When a batch of requests is dispatched to Ollama concurrently, each
    request gets a :class:`asyncio.Future` registered here.  Once the
    response arrives the caller resolves the future; the original awaiter
    receives the result transparently.

    Usage::

        mapper = ResultMapper()

        # Register futures for all requests in a batch
        futures = {req.request_id: mapper.register(req.request_id) for req in batch}

        # Dispatch and resolve as responses arrive
        for response in await ollama_dispatch(batch):
            mapper.resolve(response.request_id, response)

        # Await individual results
        result = await futures["req-1"]
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingRequest] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register(self, request_id: str) -> asyncio.Future[Response]:
        """Create and register an :class:`asyncio.Future` for *request_id*.

        Args:
            request_id: Unique identifier of the request.

        Returns:
            A :class:`asyncio.Future` that will be resolved when the
            response is available.

        Raises:
            ValueError: If *request_id* is already registered.
        """
        async with self._lock:
            if request_id in self._pending:
                raise ValueError(f"Request {request_id!r} is already registered")
            loop = asyncio.get_event_loop()
            future: asyncio.Future[Response] = loop.create_future()
            self._pending[request_id] = _PendingRequest(future=future)
        logger.debug("result_mapper_registered", request_id=request_id)
        return future

    async def resolve(self, request_id: str, response: Response) -> bool:
        """Resolve the future for *request_id* with *response*.

        Args:
            request_id: The ID of the request whose response has arrived.
            response: The completed :class:`~llm_inference_engine.core.types.Response`.

        Returns:
            ``True`` if the future was resolved, ``False`` if not found.
        """
        async with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            logger.warning("result_mapper_unknown_request", request_id=request_id)
            return False
        if not entry.future.done():
            entry.future.set_result(response)
        logger.debug("result_mapper_resolved", request_id=request_id)
        return True

    async def reject(self, request_id: str, error: Exception) -> bool:
        """Reject the future for *request_id* with *error*.

        Args:
            request_id: The ID of the request that failed.
            error: The exception to set on the future.

        Returns:
            ``True`` if the future was rejected, ``False`` if not found.
        """
        async with self._lock:
            entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        if not entry.future.done():
            entry.future.set_exception(error)
        logger.warning("result_mapper_rejected", request_id=request_id, error=str(error))
        return True

    async def cancel_all(self) -> int:
        """Cancel all pending futures (e.g. during shutdown).

        Returns:
            Number of futures cancelled.
        """
        async with self._lock:
            count = 0
            for _request_id, entry in list(self._pending.items()):
                if not entry.future.done():
                    entry.future.cancel()
                    count += 1
            self._pending.clear()
        logger.info("result_mapper_all_cancelled", count=count)
        return count

    @property
    def pending_count(self) -> int:
        """Number of requests currently awaiting resolution."""
        return len(self._pending)

    def is_pending(self, request_id: str) -> bool:
        """Return ``True`` if *request_id* has an unresolved future."""
        return request_id in self._pending


__all__ = ["ResultMapper"]
