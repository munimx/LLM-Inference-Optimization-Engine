"""In-flight request coalescing.

When multiple callers submit identical ``(model, prompt)`` requests
concurrently, only one is actually dispatched to the inference backend.  All
other callers await the same result — saving GPU time and reducing queue
depth.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class RequestCoalescer:
    """Deduplicates in-flight inference requests.

    Usage::

        coalescer = RequestCoalescer()

        async def handle(model, prompt, do_inference):
            result = await coalescer.coalesce(model, prompt, do_inference)
            return result
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._in_flight: dict[str, asyncio.Future[Any]] = {}
        self._coalesced_count = 0

    async def coalesce(
        self,
        model: str,
        prompt: str,
        producer: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return the result of *producer*, deduplicating identical requests.

        If another coroutine is already awaiting a result for the same
        ``(model, prompt)`` pair, this call piggy-backs on that future
        instead of invoking *producer* again.

        Args:
            model: Model name.
            prompt: Prompt text (used to compute the dedup key).
            producer: Zero-arg async callable that performs the actual
                inference and returns the result.

        Returns:
            Whatever *producer* returns.
        """
        key = self._make_key(model, prompt)

        async with self._lock:
            if key in self._in_flight:
                self._coalesced_count += 1
                logger.debug("request_coalesced", model=model)
                future = self._in_flight[key]
            else:
                future = asyncio.get_event_loop().create_future()
                self._in_flight[key] = future
                # We're the "owner" — release the lock and run the producer
                asyncio.ensure_future(self._run(key, producer, future))

        return await future

    async def _run(
        self,
        key: str,
        producer: Callable[[], Awaitable[Any]],
        future: asyncio.Future[Any],
    ) -> None:
        """Execute *producer* and resolve the shared future."""
        try:
            result = await producer()
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)

    @property
    def coalesced_count(self) -> int:
        """Number of requests that were deduplicated."""
        return self._coalesced_count

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight unique requests."""
        return len(self._in_flight)

    @staticmethod
    def _make_key(model: str, prompt: str) -> str:
        h = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        return h


__all__ = ["RequestCoalescer"]
