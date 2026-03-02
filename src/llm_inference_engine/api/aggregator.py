"""RequestAggregator: fan-out batched requests to Ollama, fan-in results.

The aggregator is the bridge between the FastAPI request handlers and the
:class:`~llm_inference_engine.scheduling.scheduler.Scheduler`.  It:

1. Converts API-layer request objects to core
   :class:`~llm_inference_engine.core.types.Request` objects.
2. Registers a :class:`~llm_inference_engine.api.result_mapper.ResultMapper`
   future for each request.
3. Submits the request to the :class:`~llm_inference_engine.scheduling.\
scheduler.Scheduler`.
4. Dispatches the formed :class:`~llm_inference_engine.scheduling.batch.Batch`
   to Ollama concurrently (one async task per request).
5. Fans in results and resolves the corresponding futures.
"""

import asyncio
import uuid
from typing import Any

import structlog

from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.api.coalescer import RequestCoalescer
from llm_inference_engine.api.result_mapper import ResultMapper
from llm_inference_engine.core.types import (
    GenerationConfig,
    GenerationResult,
    Request,
    RequestStatus,
    Response,
)
from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.scheduling.batch import Batch
from llm_inference_engine.scheduling.scheduler import Scheduler

logger = structlog.get_logger(__name__)


class RequestAggregator:
    """Orchestrates the full request lifecycle from API to Ollama and back.

    Usage::

        aggregator = RequestAggregator(
            ollama_client=client,
            scheduler=scheduler,
            cache=SemanticCache(),
        )
        response = await aggregator.complete(
            model="llama3.1:8b",
            prompt="Hello!",
            max_tokens=128,
        )
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        scheduler: Scheduler,
        cache: SemanticCache | None = None,
        drain_delay_seconds: float = 0.05,
        coalescer: RequestCoalescer | None = None,
    ) -> None:
        """Initialise the aggregator.

        Args:
            ollama_client: Connected :class:`~llm_inference_engine.integration.\
ollama_client.OllamaClient`.
            scheduler: Configured :class:`~llm_inference_engine.scheduling.\
scheduler.Scheduler`.
            cache: Optional :class:`~llm_inference_engine.api.cache.SemanticCache`
                for response caching.
            drain_delay_seconds: Time to wait before draining the scheduler,
                allowing concurrent requests to accumulate into a batch.
            coalescer: Optional :class:`~llm_inference_engine.api.coalescer.\
RequestCoalescer` for deduplicating identical in-flight requests.
        """
        self._client = ollama_client
        self._scheduler = scheduler
        self._cache = cache
        self._mapper = ResultMapper()
        self._total_requests = 0
        self._drain_delay = drain_delay_seconds
        self._drain_locks: dict[str, asyncio.Lock] = {}
        self._coalescer = coalescer
        logger.info(
            "request_aggregator_initialized",
            caching=cache is not None,
            drain_delay_seconds=drain_delay_seconds,
            coalescing=coalescer is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        priority: int = 0,
    ) -> Response:
        """Submit a completion request and wait for the response.

        Args:
            model: Ollama model tag.
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.
            priority: Request priority (higher = served sooner).

        Returns:
            A :class:`~llm_inference_engine.core.types.Response`.
        """
        self._total_requests += 1

        # Check cache first.
        if self._cache is not None:
            cached = await self._cache.get(model, prompt)
            if cached is not None:
                return self._make_cached_response(model, prompt, cached)

        request_id = str(uuid.uuid4())
        request = Request(
            request_id=request_id,
            prompt=prompt,
            model=model,
            generation_config=GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop or [],
            ),
            priority=priority,
        )

        # Register a future and submit to scheduler.
        future = await self._mapper.register(request_id)
        await self._scheduler.submit(request)

        # Wait briefly to allow concurrent requests to accumulate, then drain.
        if self._drain_delay > 0:
            await asyncio.sleep(self._drain_delay)

        # Use a per-model lock so only one coroutine drains at a time.
        if model not in self._drain_locks:
            self._drain_locks[model] = asyncio.Lock()
        async with self._drain_locks[model]:
            responses = await self._scheduler.drain(model)
            for resp in responses:
                await self._mapper.resolve(resp.request_id, resp)
                # Cache using this request's prompt (only valid for single-request drain)
                if self._cache is not None and resp.result is not None:
                    await self._cache.put(model, prompt, resp.result.text)

        # Wait for our specific future.
        try:
            return await asyncio.wait_for(future, timeout=300.0)
        except TimeoutError:
            await self._mapper.reject(request_id, TimeoutError(f"Request {request_id} timed out"))
            return Response(
                request_id=request_id,
                error="Request timed out",
                status=RequestStatus.FAILED,
            )

    async def chat_complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        priority: int = 0,
    ) -> Response:
        """Submit a chat completion request using Ollama's /api/chat endpoint.

        Args:
            model: Ollama model tag.
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.
            priority: Request priority (higher = served sooner).

        Returns:
            A :class:`~llm_inference_engine.core.types.Response`.
        """
        import time

        self._total_requests += 1

        # Build a cache key from the messages hash
        cache_key = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        if self._cache is not None:
            cached = await self._cache.get(model, cache_key)
            if cached is not None:
                return self._make_cached_response(model, cache_key, cached)

        # Coalesce identical in-flight chat requests
        if self._coalescer is not None:
            return await self._coalescer.coalesce(
                model,
                cache_key,
                lambda: self._do_chat(model, messages, cache_key, max_tokens, temperature, top_p, stop),
            )

        return await self._do_chat(model, messages, cache_key, max_tokens, temperature, top_p, stop)

    async def _do_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        cache_key: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> Response:
        """Perform the actual chat call to Ollama."""

        request_id = str(uuid.uuid4())
        start = time.monotonic()

        try:
            raw: dict[str, Any] = await self._client.chat(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop or None,
            )
            latency_ms = (time.monotonic() - start) * 1000

            # Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            message_data = raw.get("message", {})
            text = str(message_data.get("content", ""))

            result = GenerationResult(
                request_id=request_id,
                text=text,
                finish_reason=str(raw.get("done_reason", "stop")),
                tokens_used=int(raw.get("eval_count", 0)),
                latency_ms=latency_ms,
                model=model,
                metadata={
                    "prompt_eval_count": raw.get("prompt_eval_count"),
                    "eval_duration_ns": raw.get("eval_duration"),
                },
            )

            if self._cache is not None:
                await self._cache.put(model, cache_key, text)

            return Response(
                request_id=request_id,
                result=result,
                status=RequestStatus.COMPLETED,
            )
        except Exception as exc:
            logger.error("chat_complete_error", request_id=request_id, error=str(exc))
            return Response(
                request_id=request_id,
                error=str(exc),
                status=RequestStatus.FAILED,
            )

    @property
    def total_requests(self) -> int:
        """Total number of requests handled (including cache hits)."""
        return self._total_requests

    @property
    def pending_count(self) -> int:
        """Number of requests currently in-flight."""
        return self._mapper.pending_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cached_response(model: str, prompt: str, text: str) -> Response:
        """Build a synthetic Response from a cache hit."""
        request_id = str(uuid.uuid4())
        result = GenerationResult(
            request_id=request_id,
            text=text,
            finish_reason="cache_hit",
            tokens_used=0,
            latency_ms=0.0,
            model=model,
            metadata={"cache_hit": True},
        )
        return Response(request_id=request_id, result=result, status=RequestStatus.COMPLETED)


async def dispatch_batch(client: OllamaClient, batch: Batch) -> list[Response]:
    """Fan-out a :class:`~llm_inference_engine.scheduling.batch.Batch` to Ollama.

    Each request in the batch is dispatched as a separate concurrent
    :class:`asyncio.Task`.  Results are collected and returned in the
    same order as the batch.

    Args:
        client: Connected Ollama HTTP client.
        batch: The batch of requests to dispatch.

    Returns:
        List of :class:`~llm_inference_engine.core.types.Response` objects,
        one per request in the batch.
    """

    async def _dispatch_one(request: Request) -> Response:
        import time

        start = time.monotonic()
        try:
            raw: dict[str, Any] = await client.generate(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.generation_config.max_tokens,
                temperature=request.generation_config.temperature,
                top_p=request.generation_config.top_p,
                stop_sequences=request.generation_config.stop_sequences or None,
            )
            latency_ms = (time.monotonic() - start) * 1000
            result = GenerationResult(
                request_id=request.request_id,
                text=str(raw.get("response", "")),
                finish_reason=str(raw.get("done_reason", "stop")),
                tokens_used=int(raw.get("eval_count", 0)),
                latency_ms=latency_ms,
                model=request.model,
                metadata={
                    "prompt_eval_count": raw.get("prompt_eval_count"),
                    "eval_duration_ns": raw.get("eval_duration"),
                },
            )
            return Response(
                request_id=request.request_id,
                result=result,
                status=RequestStatus.COMPLETED,
            )
        except Exception as exc:
            logger.error(
                "dispatch_error",
                request_id=request.request_id,
                error=str(exc),
            )
            return Response(
                request_id=request.request_id,
                error=str(exc),
                status=RequestStatus.FAILED,
            )

    tasks = [asyncio.create_task(_dispatch_one(req)) for req in batch]
    return list(await asyncio.gather(*tasks))


__all__ = ["RequestAggregator", "dispatch_batch"]
