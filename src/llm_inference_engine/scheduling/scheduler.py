"""Scheduler: orchestrates queue → policy → batch formation → dispatch."""

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from llm_inference_engine.core.types import Request, RequestStatus, Response
from llm_inference_engine.scheduling.batch import Batch
from llm_inference_engine.scheduling.policies import SchedulingPolicy, get_policy
from llm_inference_engine.scheduling.queue import RequestQueue

logger = structlog.get_logger(__name__)

# Type alias for an async dispatch function that accepts a Batch.
DispatchFn = Callable[[Batch], Coroutine[Any, Any, list[Response]]]


class Scheduler:
    """Orchestrates request queuing, batch formation, and dispatch.

    The scheduler maintains one :class:`~llm_inference_engine.scheduling.\
queue.RequestQueue` per model and uses the selected
    :class:`~llm_inference_engine.scheduling.policies.SchedulingPolicy` to
    form :class:`~llm_inference_engine.scheduling.batch.Batch` objects which
    are then passed to the caller-supplied *dispatch_fn*.

    Usage example::

        async def my_dispatch(batch: Batch) -> list[Response]:
            ...  # call Ollama concurrently for each request in batch

        scheduler = Scheduler(dispatch_fn=my_dispatch)
        await scheduler.submit(request)
        responses = await scheduler.drain("my-model")
    """

    def __init__(
        self,
        dispatch_fn: DispatchFn,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
        max_requests_per_batch: int = 8,
        max_tokens_per_batch: int = 0,
        queue_maxsize: int = 0,
    ) -> None:
        """Initialise the scheduler.

        Args:
            dispatch_fn: Async callable that executes a :class:`Batch` and
                returns a list of :class:`~llm_inference_engine.core.types.\
Response` objects (one per request in the batch).
            policy: Batch formation policy to apply.
            max_requests_per_batch: Maximum requests per batch.
            max_tokens_per_batch: Maximum total token budget per batch
                (``0`` = unlimited).
            queue_maxsize: Maximum queue depth per model (``0`` = unbounded).
        """
        self._dispatch_fn = dispatch_fn
        self._policy = get_policy(policy)
        self._policy_name = policy
        self._max_requests = max_requests_per_batch
        self._max_tokens = max_tokens_per_batch
        self._queue_maxsize = queue_maxsize
        self._queues: dict[str, RequestQueue] = {}
        logger.info(
            "scheduler_initialized",
            policy=policy,
            max_requests_per_batch=max_requests_per_batch,
            max_tokens_per_batch=max_tokens_per_batch,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, request: Request) -> None:
        """Enqueue *request* for scheduling.

        A per-model queue is created automatically on first use.  Queue
        creation uses :meth:`dict.setdefault` which is atomic under
        CPython's GIL, avoiding any async lock suspension on the hot
        submit path.

        Args:
            request: The inference request to schedule.
        """
        queue = self._get_or_create_queue(request.model)
        await queue.enqueue(request)
        logger.debug(
            "request_submitted",
            request_id=request.request_id,
            model=request.model,
        )

    def cancel(self, request_id: str, model: str) -> bool:
        """Cancel a pending request.

        Args:
            request_id: ID of the request to cancel.
            model: Model queue to search in.

        Returns:
            ``True`` if the request was found and cancelled.
        """
        queue = self._queues.get(model)
        if queue is None:
            return False
        return queue.cancel(request_id)

    async def drain(self, model: str) -> list[Response]:
        """Drain one batch from the *model* queue and dispatch it.

        This is a *non-blocking* operation: if the queue is empty, an
        empty list is returned immediately.

        Args:
            model: The Ollama model tag whose queue should be drained.

        Returns:
            List of :class:`~llm_inference_engine.core.types.Response`
            objects produced by the dispatch function, or an empty list
            if the queue was empty.
        """
        queue = self._queues.get(model)
        if queue is None or queue.empty:
            return []

        requests = await self._collect_batch_requests(queue)
        if not requests:
            return []

        batch = self._form_batch(requests, model)
        if batch.is_empty:
            return []

        for req in batch:
            req.status = RequestStatus.SCHEDULED

        logger.info(
            "dispatching_batch",
            batch_id=batch.batch_id,
            model=model,
            batch_size=batch.size,
            total_tokens=batch.total_tokens,
        )

        responses = await self._dispatch_fn(batch)

        for req in batch:
            req.status = RequestStatus.COMPLETED

        return responses

    async def run_loop(self, model: str, interval_s: float = 0.05) -> None:
        """Continuously drain the *model* queue in a background loop.

        This coroutine runs indefinitely and should be run as an
        ``asyncio.Task``.  It polls the queue every *interval_s* seconds
        and dispatches batches as they become available.

        Args:
            model: Model whose queue to serve.
            interval_s: Sleep interval between empty-queue polls.
        """
        logger.info("scheduler_loop_started", model=model)
        while True:
            responses = await self.drain(model)
            if not responses:
                await asyncio.sleep(interval_s)

    def queue_size(self, model: str) -> int:
        """Return the current queue depth for *model*.

        Args:
            model: The model to query.

        Returns:
            Number of items in the queue (including cancelled items).
        """
        queue = self._queues.get(model)
        return queue.size if queue is not None else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_queue(self, model: str) -> RequestQueue:
        """Return the queue for *model*, creating it if needed.

        Uses :meth:`dict.setdefault` which is atomic under CPython's GIL —
        no async lock required.  Even if two coroutines call this
        concurrently for the same model, at most one extra ``RequestQueue``
        is constructed and immediately discarded; no data is corrupted.
        """
        if model not in self._queues:
            self._queues.setdefault(model, RequestQueue(maxsize=self._queue_maxsize))
            logger.info("model_queue_created", model=model)
        return self._queues[model]

    async def _collect_batch_requests(self, queue: RequestQueue) -> list[Request]:
        """Drain up to ``max_requests_per_batch`` items from the queue."""
        requests: list[Request] = []
        limit = self._max_requests
        while len(requests) < limit:
            request = await queue.dequeue()
            if request is None:
                break
            requests.append(request)
        return requests

    def _form_batch(self, requests: list[Request], model: str) -> Batch:
        batch_id = str(uuid.uuid4())
        return self._policy.form_batch(
            requests=requests,
            batch_id=batch_id,
            model=model,
            max_requests=self._max_requests,
            max_tokens=self._max_tokens,
        )


__all__ = ["Scheduler", "DispatchFn"]
