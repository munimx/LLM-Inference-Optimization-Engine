"""Unit tests for RequestAggregator and dispatch_batch."""

from unittest.mock import AsyncMock, MagicMock

from llm_inference_engine.api.aggregator import RequestAggregator, dispatch_batch
from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.core.types import (
    GenerationConfig,
    GenerationResult,
    Request,
    RequestStatus,
    Response,
)
from llm_inference_engine.scheduling.batch import Batch
from llm_inference_engine.scheduling.policies import SchedulingPolicy
from llm_inference_engine.scheduling.scheduler import Scheduler


def _make_response(request_id: str, text: str = "result") -> Response:
    result = GenerationResult(
        request_id=request_id,
        text=text,
        finish_reason="stop",
        tokens_used=10,
        latency_ms=50.0,
        model="llama3:8b",
    )
    return Response(request_id=request_id, result=result, status=RequestStatus.COMPLETED)


def _make_request(request_id: str = "req-1", model: str = "llama3:8b") -> Request:
    return Request(
        request_id=request_id,
        prompt="Hello",
        model=model,
        generation_config=GenerationConfig(max_tokens=64),
    )


def _make_ollama_client(text: str = "generated text") -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(
        return_value={
            "response": text,
            "eval_count": 10,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_duration": 1_000_000,
        }
    )
    return client


class TestRequestAggregatorInit:
    """Tests for RequestAggregator initialisation."""

    def test_total_requests_starts_at_zero(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler)
        assert agg.total_requests == 0

    def test_pending_count_starts_at_zero(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler)
        assert agg.pending_count == 0

    def test_cache_optional(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        # Should not raise
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler, cache=None)
        assert agg is not None


class TestRequestAggregatorCacheHit:
    """Tests for cache-hit fast path in RequestAggregator.complete()."""

    # Cache keys must include generation params (defaults: max_tokens=256, temperature=0.7)
    _default_suffix = "\x00mt=256\x00t=0.7"

    async def test_cache_hit_returns_cached_response(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        scheduler.submit = AsyncMock()
        scheduler.drain = AsyncMock(return_value=[])
        cache = SemanticCache()
        await cache.put("llama3:8b", f"Hello{self._default_suffix}", "cached response")

        agg = RequestAggregator(ollama_client=client, scheduler=scheduler, cache=cache)
        response = await agg.complete(model="llama3:8b", prompt="Hello")
        assert response.result is not None
        assert response.result.text == "cached response"
        assert response.result.metadata.get("cache_hit") is True

    async def test_cache_hit_does_not_call_scheduler(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        scheduler.submit = AsyncMock()
        scheduler.drain = AsyncMock(return_value=[])
        cache = SemanticCache()
        await cache.put("llama3:8b", f"Hello{self._default_suffix}", "cached response")

        agg = RequestAggregator(ollama_client=client, scheduler=scheduler, cache=cache)
        await agg.complete(model="llama3:8b", prompt="Hello")
        scheduler.submit.assert_not_awaited()

    async def test_cache_hit_increments_total_requests(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        cache = SemanticCache()
        await cache.put("llama3:8b", f"Hello{self._default_suffix}", "cached response")

        agg = RequestAggregator(ollama_client=client, scheduler=scheduler, cache=cache)
        await agg.complete(model="llama3:8b", prompt="Hello")
        assert agg.total_requests == 1

    async def test_cache_hit_status_completed(self) -> None:
        client = _make_ollama_client()
        scheduler = MagicMock(spec=Scheduler)
        cache = SemanticCache()
        await cache.put("llama3:8b", f"Hi{self._default_suffix}", "hello there")

        agg = RequestAggregator(ollama_client=client, scheduler=scheduler, cache=cache)
        resp = await agg.complete(model="llama3:8b", prompt="Hi")
        assert resp.status == RequestStatus.COMPLETED


class TestRequestAggregatorComplete:
    """Tests for cache-miss path in RequestAggregator.complete()."""

    async def test_complete_returns_response(self) -> None:
        client = _make_ollama_client("hello!")
        dispatch_responses = None

        async def fake_drain(model: str) -> list[Response]:
            nonlocal dispatch_responses
            # Simulate resolving via dispatch_batch logic
            return dispatch_responses or []

        scheduler = Scheduler(
            dispatch_fn=lambda batch: dispatch_batch(client, batch),
            policy=SchedulingPolicy.FCFS,
        )
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler)
        response = await agg.complete(model="llama3:8b", prompt="Hello world")
        assert response is not None

    async def test_complete_increments_total_requests(self) -> None:
        client = _make_ollama_client()
        scheduler = Scheduler(
            dispatch_fn=lambda batch: dispatch_batch(client, batch),
            policy=SchedulingPolicy.FCFS,
        )
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler)
        await agg.complete(model="llama3:8b", prompt="Hello")
        assert agg.total_requests == 1

    async def test_multiple_requests_increment_counter(self) -> None:
        client = _make_ollama_client()
        scheduler = Scheduler(
            dispatch_fn=lambda batch: dispatch_batch(client, batch),
            policy=SchedulingPolicy.FCFS,
        )
        agg = RequestAggregator(ollama_client=client, scheduler=scheduler)
        for _ in range(3):
            await agg.complete(model="llama3:8b", prompt="Hello")
        assert agg.total_requests == 3


class TestMakeCachedResponse:
    """Tests for RequestAggregator._make_cached_response."""

    def test_returns_completed_response(self) -> None:
        resp = RequestAggregator._make_cached_response("llama3:8b", "hello", "world")
        assert resp.status == RequestStatus.COMPLETED

    def test_result_text_correct(self) -> None:
        resp = RequestAggregator._make_cached_response("llama3:8b", "hello", "world")
        assert resp.result is not None
        assert resp.result.text == "world"

    def test_result_finish_reason_is_cache_hit(self) -> None:
        resp = RequestAggregator._make_cached_response("llama3:8b", "p", "r")
        assert resp.result is not None
        assert resp.result.finish_reason == "cache_hit"

    def test_metadata_cache_hit_true(self) -> None:
        resp = RequestAggregator._make_cached_response("llama3:8b", "p", "r")
        assert resp.result is not None
        assert resp.result.metadata.get("cache_hit") is True

    def test_model_stored_in_result(self) -> None:
        resp = RequestAggregator._make_cached_response("llama3:8b", "p", "r")
        assert resp.result is not None
        assert resp.result.model == "llama3:8b"


class TestDispatchBatch:
    """Tests for dispatch_batch()."""

    async def test_returns_list_of_responses(self) -> None:
        client = _make_ollama_client("output text")
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        batch.add(_make_request("r1"))
        responses = await dispatch_batch(client, batch)
        assert len(responses) == 1

    async def test_successful_response_status(self) -> None:
        client = _make_ollama_client("output")
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        batch.add(_make_request("r1"))
        responses = await dispatch_batch(client, batch)
        assert responses[0].status == RequestStatus.COMPLETED

    async def test_response_text_from_client(self) -> None:
        client = _make_ollama_client("expected output")
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        batch.add(_make_request("r1"))
        responses = await dispatch_batch(client, batch)
        assert responses[0].result is not None
        assert responses[0].result.text == "expected output"

    async def test_multiple_requests_in_batch(self) -> None:
        client = _make_ollama_client("out")
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=4096)
        for i in range(3):
            batch.add(_make_request(f"r{i}"))
        responses = await dispatch_batch(client, batch)
        assert len(responses) == 3

    async def test_client_error_produces_failed_response(self) -> None:
        client = MagicMock()
        client.generate = AsyncMock(side_effect=RuntimeError("Ollama down"))
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        batch.add(_make_request("r1"))
        responses = await dispatch_batch(client, batch)
        assert responses[0].status == RequestStatus.FAILED
        assert responses[0].error is not None

    async def test_empty_batch_returns_empty_list(self) -> None:
        client = _make_ollama_client()
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        responses = await dispatch_batch(client, batch)
        assert responses == []

    async def test_response_request_id_matches(self) -> None:
        client = _make_ollama_client()
        batch = Batch(batch_id="b1", model="llama3:8b", max_requests=4, max_tokens=1024)
        batch.add(_make_request("my-unique-id"))
        responses = await dispatch_batch(client, batch)
        assert responses[0].request_id == "my-unique-id"
