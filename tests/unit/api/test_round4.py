"""Round 4 tests: streaming parity, cache key variance, and admission rejection.

Tests verify that:
- Circuit breaker records success/failure for streaming requests
- Throttler reserve/release lifecycle works for streaming
- Different generation params produce different cache keys
- Throttler admission rejection returns 503
"""

import pytest

from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.api.circuit_breaker import CircuitBreaker, CircuitState


class TestCacheKeyVariance:
    """Verify different generation params produce different cache entries."""

    async def test_different_max_tokens_different_keys(self) -> None:
        cache = SemanticCache(max_size=100, ttl_seconds=300)
        from llm_inference_engine.api.aggregator import RequestAggregator

        key_a = RequestAggregator._cache_key("What is AI?", 256, 0.7)
        key_b = RequestAggregator._cache_key("What is AI?", 512, 0.7)
        assert key_a != key_b

        await cache.put("model", key_a, "response_256")
        await cache.put("model", key_b, "response_512")
        assert await cache.get("model", key_a) == "response_256"
        assert await cache.get("model", key_b) == "response_512"

    async def test_different_temperature_different_keys(self) -> None:
        cache = SemanticCache(max_size=100, ttl_seconds=300)
        from llm_inference_engine.api.aggregator import RequestAggregator

        key_a = RequestAggregator._cache_key("What is AI?", 256, 0.7)
        key_b = RequestAggregator._cache_key("What is AI?", 256, 0.0)
        assert key_a != key_b

        await cache.put("model", key_a, "creative")
        await cache.put("model", key_b, "deterministic")
        assert await cache.get("model", key_a) == "creative"
        assert await cache.get("model", key_b) == "deterministic"

    async def test_same_params_same_key(self) -> None:
        from llm_inference_engine.api.aggregator import RequestAggregator

        key_a = RequestAggregator._cache_key("Hello", 256, 0.7)
        key_b = RequestAggregator._cache_key("Hello", 256, 0.7)
        assert key_a == key_b


class TestStreamingCircuitBreaker:
    """Verify circuit breaker integration in streaming helpers."""

    async def test_stream_completion_records_success(self) -> None:
        from unittest.mock import MagicMock

        from llm_inference_engine.api.models import CompletionRequest
        from llm_inference_engine.api.server import _stream_completion

        mock_client = MagicMock()

        async def _fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            yield {"response": "hello", "done": False}
            yield {"response": "", "done": True, "eval_count": 5, "prompt_eval_count": 3}

        mock_client.generate_stream = _fake_stream

        cb = CircuitBreaker(failure_threshold=3)
        body = CompletionRequest(model="llama3:8b", prompt="test", stream=True)

        events = []
        async for event in _stream_completion(mock_client, body, circuit_breaker=cb):
            events.append(event)

        assert cb.state == CircuitState.CLOSED
        assert cb._consecutive_failures == 0
        assert len(events) >= 2  # at least data events + [DONE]

    async def test_stream_completion_records_failure(self) -> None:
        from unittest.mock import MagicMock

        from llm_inference_engine.api.models import CompletionRequest
        from llm_inference_engine.api.server import _stream_completion

        mock_client = MagicMock()

        async def _failing_stream(**kwargs):  # type: ignore[no-untyped-def]
            yield {"response": "partial", "done": False}
            raise ConnectionError("Ollama crashed")

        mock_client.generate_stream = _failing_stream

        cb = CircuitBreaker(failure_threshold=3)
        body = CompletionRequest(model="llama3:8b", prompt="test", stream=True)

        with pytest.raises(ConnectionError):
            async for _ in _stream_completion(mock_client, body, circuit_breaker=cb):
                pass

        assert cb._consecutive_failures == 1

    async def test_stream_chat_records_success(self) -> None:
        from unittest.mock import MagicMock

        from llm_inference_engine.api.models import ChatCompletionRequest
        from llm_inference_engine.api.server import _stream_chat_completion

        mock_client = MagicMock()

        async def _fake_chat_stream(**kwargs):  # type: ignore[no-untyped-def]
            yield {"message": {"content": "hi"}, "done": False}
            yield {"message": {"content": ""}, "done": True, "eval_count": 3, "prompt_eval_count": 2}

        mock_client.chat_stream = _fake_chat_stream

        cb = CircuitBreaker(failure_threshold=3)
        body = ChatCompletionRequest(
            model="llama3:8b",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        messages = [{"role": "user", "content": "hello"}]

        events = []
        async for event in _stream_chat_completion(mock_client, body, messages, circuit_breaker=cb):
            events.append(event)

        assert cb.state == CircuitState.CLOSED
        assert len(events) >= 2


class TestStreamingThrottler:
    """Verify throttler reserve/release in streaming path."""

    async def test_stream_completion_releases_throttler(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from llm_inference_engine.api.models import CompletionRequest
        from llm_inference_engine.api.server import _stream_completion

        mock_client = MagicMock()

        async def _fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            yield {"response": "hello", "done": True, "eval_count": 1, "prompt_eval_count": 1}

        mock_client.generate_stream = _fake_stream

        mock_throttler = MagicMock()
        mock_throttler.release = AsyncMock()

        body = CompletionRequest(model="llama3:8b", prompt="test", stream=True)

        async for _ in _stream_completion(
            mock_client, body,
            throttler=mock_throttler,
            request_id_for_throttler="req-123",
            memory_estimate_gb=0.5,
        ):
            pass

        mock_throttler.release.assert_called_once_with("req-123", 0.5)

    async def test_stream_completion_releases_on_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from llm_inference_engine.api.models import CompletionRequest
        from llm_inference_engine.api.server import _stream_completion

        mock_client = MagicMock()

        async def _failing_stream(**kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("down")
            yield  # make it a generator  # pragma: no cover

        mock_client.generate_stream = _failing_stream

        mock_throttler = MagicMock()
        mock_throttler.release = AsyncMock()

        body = CompletionRequest(model="llama3:8b", prompt="test", stream=True)

        with pytest.raises(ConnectionError):
            async for _ in _stream_completion(
                mock_client, body,
                throttler=mock_throttler,
                request_id_for_throttler="req-456",
                memory_estimate_gb=0.5,
            ):
                pass

        mock_throttler.release.assert_called_once_with("req-456", 0.5)

    async def test_stream_cache_hit_releases_throttler(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from llm_inference_engine.api.models import CompletionRequest
        from llm_inference_engine.api.server import _stream_completion

        mock_client = MagicMock()
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="cached response")

        mock_throttler = MagicMock()
        mock_throttler.release = AsyncMock()

        body = CompletionRequest(model="llama3:8b", prompt="test", stream=True)

        async for _ in _stream_completion(
            mock_client, body,
            cache=mock_cache,
            throttler=mock_throttler,
            request_id_for_throttler="req-789",
            memory_estimate_gb=0.5,
        ):
            pass

        mock_throttler.release.assert_called_once_with("req-789", 0.5)


class TestAdmissionRejection:
    """Verify throttler rejection returns 503."""

    async def test_throttler_reject_returns_503(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        from llm_inference_engine.api.dependencies import (
            get_aggregator,
            get_cache,
            get_ollama_client,
            get_throttler,
        )
        from llm_inference_engine.api.server import create_app
        from llm_inference_engine.config import InferenceConfig
        from llm_inference_engine.optimization.throttler import (
            AdaptiveThrottler,
            AdmissionDecision,
            ThrottlerStats,
        )

        config = InferenceConfig()
        app = create_app(config=config)

        mock_aggregator = MagicMock()
        mock_aggregator.pending_count = 0

        # Create a real throttler-like mock that returns REJECT
        mock_throttler = MagicMock(spec=AdaptiveThrottler)
        mock_throttler.check = AsyncMock(return_value=AdmissionDecision.REJECT)
        mock_throttler.stats = ThrottlerStats(
            committed_gb=14.0, available_gb=0.0, memory_limit_gb=14.0,
            soft_limit_gb=11.9, active_requests=28,
        )

        mock_cache = MagicMock()
        mock_cache.hits = 0
        mock_cache.misses = 0

        mock_ollama = MagicMock()
        mock_ollama.is_available = AsyncMock(return_value=True)

        app.dependency_overrides[get_aggregator] = lambda: mock_aggregator
        app.dependency_overrides[get_throttler] = lambda: mock_throttler
        app.dependency_overrides[get_cache] = lambda: mock_cache
        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

        with TestClient(app, raise_server_exceptions=False) as client:
            # Override app.state AFTER lifespan has run (inside the context)
            app.state.throttler = mock_throttler
            app.state.circuit_breaker = CircuitBreaker()

            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "hi"},
            )

        assert resp.status_code == 503
        assert "Memory limit" in resp.json()["detail"]


class TestCoalescerInCompletions:
    """Verify coalescer is wired into the completions path."""

    async def test_coalescer_called_for_completions(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from llm_inference_engine.api.aggregator import RequestAggregator
        from llm_inference_engine.core.types import (
            GenerationResult,
            RequestStatus,
            Response,
        )

        mock_client = MagicMock()
        mock_scheduler = MagicMock()
        mock_coalescer = MagicMock()

        expected_response = Response(
            request_id="coal-1",
            result=GenerationResult(
                request_id="coal-1", text="coalesced", finish_reason="stop",
                tokens_used=5, latency_ms=10.0, model="llama3:8b",
            ),
            status=RequestStatus.COMPLETED,
        )
        mock_coalescer.coalesce = AsyncMock(return_value=expected_response)

        aggregator = RequestAggregator(
            ollama_client=mock_client,
            scheduler=mock_scheduler,
            cache=None,
            coalescer=mock_coalescer,
        )

        result = await aggregator.complete(
            model="llama3:8b", prompt="What is AI?", max_tokens=256, temperature=0.7,
        )

        assert result.request_id == "coal-1"
        mock_coalescer.coalesce.assert_called_once()
