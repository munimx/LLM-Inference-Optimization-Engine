"""Cross-worker in-flight request coalescing via Redis.

When multiple workers receive identical ``(model, prompt)`` requests
concurrently, only the first worker executes the inference.  All others
subscribe to a Redis pub/sub channel and receive the result when it is
published.

This extends the single-process coalescing concept to work across multiple
uvicorn workers or container replicas.

Usage::

    coalescer = RedisCoalescer(redis_client)

    async def handle(model, prompt, do_inference):
        result = await coalescer.coalesce(model, prompt, do_inference)
        return result
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# How long the winner's lock key stays in Redis (seconds).
# Requests that take longer than this will fall through to direct execution.
_LOCK_TTL_MS = 30_000  # 30 seconds in milliseconds

# How long waiters listen on the pub/sub channel before giving up and
# executing the inference themselves.
_WAIT_TIMEOUT_SECONDS = 25.0

# Redis key prefixes
_LOCK_PREFIX = "llm_coalesce:lock:"
_RESULT_CHANNEL_PREFIX = "llm_coalesce:result:"


def _request_hash(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class RequestCoalescer:
    """Deduplicates in-flight inference requests using Redis pub/sub.

    Args:
        redis_client: An async Redis client (``redis.asyncio.Redis``).
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._coalesced_count = 0

    async def coalesce(
        self,
        model: str,
        prompt: str,
        producer: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return the result of *producer*, deduplicating identical requests.

        If another worker is already executing an identical request, this
        method subscribes to its result channel and returns that result.
        Falls back to executing *producer* directly if the wait times out.

        Args:
            model: Model name.
            prompt: Prompt text.
            producer: Zero-arg async callable that performs the inference.

        Returns:
            Whatever *producer* returns.
        """
        req_hash = _request_hash(model, prompt)
        lock_key = f"{_LOCK_PREFIX}{req_hash}"
        channel = f"{_RESULT_CHANNEL_PREFIX}{req_hash}"

        # Attempt to claim ownership with SET NX PX (atomic)
        claimed = await self._redis.set(lock_key, "1", nx=True, px=_LOCK_TTL_MS)

        if claimed:
            return await self._execute_and_publish(producer, lock_key, channel)

        # Another worker owns this request — wait for the result
        logger.debug("request_coalesced_waiting", model=model)
        result = await self._wait_for_result(channel)
        if result is not None:
            self._coalesced_count += 1
            return result

        # Timed out waiting — fall back to executing directly
        logger.debug("coalesce_wait_timeout_fallback", model=model)
        return await producer()

    async def _execute_and_publish(
        self,
        producer: Callable[[], Awaitable[Any]],
        lock_key: str,
        channel: str,
    ) -> Any:
        """Execute *producer* and publish the serialised result to *channel*."""
        try:
            result = await producer()
            payload = json.dumps(result) if not isinstance(result, str) else result
            await self._redis.publish(channel, payload)
            return result
        except Exception as exc:
            # Publish an error sentinel so waiters don't hang
            await self._redis.publish(channel, json.dumps({"__error__": str(exc)}))
            raise
        finally:
            await self._redis.delete(lock_key)

    async def _wait_for_result(self, channel: str) -> Any | None:
        """Subscribe to *channel* and return the first message received.

        Returns ``None`` on timeout.
        """
        pubsub = self._redis.pubsub()
        try:
            await pubsub.subscribe(channel)
            deadline = asyncio.get_event_loop().time() + _WAIT_TIMEOUT_SECONDS
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining),
                    timeout=remaining + 0.1,
                )
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                raw = message.get("data", "")
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "__error__" in parsed:
                        raise RuntimeError(parsed["__error__"])
                    return parsed
                except (json.JSONDecodeError, TypeError):
                    return raw
        except TimeoutError:
            return None
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    @property
    def coalesced_count(self) -> int:
        """Number of requests that were deduplicated (in-process counter)."""
        return self._coalesced_count


__all__ = ["RequestCoalescer"]
