"""Tests for streaming, auth, Prometheus, and chat format.

Covers the critical paths that were untested after Round 1:
- SSE streaming endpoints
- API-key auth middleware
- Prometheus /metrics/prometheus endpoint
- Chat endpoint message structure
"""

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from llm_inference_engine.api.dependencies import (
    get_aggregator,
    get_cache,
    get_ollama_client,
    get_throttler,
)
from llm_inference_engine.api.server import create_app
from llm_inference_engine.config import AuthConfig, InferenceConfig
from llm_inference_engine.core.types import (
    GenerationResult,
    RequestStatus,
    Response,
)
from llm_inference_engine.optimization.throttler import ThrottlerStats


def _ok_response(text: str = "hello") -> Response:
    return Response(
        request_id="req-1",
        result=GenerationResult(
            request_id="req-1",
            text=text,
            finish_reason="stop",
            tokens_used=10,
            latency_ms=25.0,
            model="llama3:8b",
        ),
        status=RequestStatus.COMPLETED,
    )


def _make_app(*, auth_config: AuthConfig | None = None) -> tuple:
    """Create app + overrides for testing. Returns (app, mock_aggregator, mock_ollama)."""
    config = InferenceConfig()
    if auth_config is not None:
        config.auth = auth_config

    mock_aggregator = MagicMock()
    mock_aggregator.complete = AsyncMock(return_value=_ok_response())
    mock_aggregator.chat_complete = AsyncMock(return_value=_ok_response())
    mock_aggregator.pending_count = 0
    mock_aggregator.total_requests = 5

    mock_throttler = MagicMock()
    mock_throttler.stats = ThrottlerStats(
        committed_gb=2.0, available_gb=12.0, memory_limit_gb=14.0,
        soft_limit_gb=11.9, active_requests=1,
    )

    mock_cache = MagicMock()
    mock_cache.hits = 10
    mock_cache.misses = 3
    mock_cache.size = 5

    mock_ollama = MagicMock()
    mock_ollama.is_available = AsyncMock(return_value=True)
    mock_ollama.close = AsyncMock()

    app = create_app(config=config)
    app.dependency_overrides[get_aggregator] = lambda: mock_aggregator
    app.dependency_overrides[get_throttler] = lambda: mock_throttler
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

    # Set app.state for Prometheus endpoint
    app.state.throttler = mock_throttler
    app.state.cache = mock_cache

    return app, mock_aggregator, mock_ollama


# -----------------------------------------------------------------------
# Auth middleware tests
# -----------------------------------------------------------------------


class TestAuthMiddleware:
    """Test API-key authentication middleware."""

    def test_auth_disabled_allows_all_requests(self):
        app, _, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "hi"},
            )
            assert resp.status_code == 200

    def test_auth_enabled_rejects_missing_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "hi"},
            )
            assert resp.status_code == 401
            assert "API key" in resp.json()["detail"]

    def test_auth_enabled_rejects_invalid_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "hi"},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401

    def test_auth_enabled_accepts_valid_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "hi"},
                headers={"Authorization": "Bearer secret-key-123"},
            )
            assert resp.status_code == 200

    def test_auth_allows_health_without_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_auth_allows_docs_without_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/docs")
            assert resp.status_code == 200

    def test_auth_allows_prometheus_without_token(self):
        app, _, _ = _make_app(
            auth_config=AuthConfig(enabled=True, api_keys=["secret-key-123"]),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics/prometheus")
            assert resp.status_code == 200


# -----------------------------------------------------------------------
# Prometheus endpoint tests
# -----------------------------------------------------------------------


class TestPrometheusEndpoint:
    """Test GET /metrics/prometheus."""

    def test_returns_200(self):
        app, _, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics/prometheus")
            assert resp.status_code == 200

    def test_content_type_is_prometheus(self):
        app, _, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics/prometheus")
            assert "text/plain" in resp.headers["content-type"] or \
                   "text/plain" in resp.headers.get("content-type", "")

    def test_contains_metric_names(self):
        app, _, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics/prometheus")
            body = resp.text
            assert "llm_engine_requests_total" in body or "llm_engine" in body
            assert "llm_engine_cache_entries" in body


# -----------------------------------------------------------------------
# Chat format tests
# -----------------------------------------------------------------------


class TestChatMessageFormat:
    """Test that chat endpoint uses structured messages, not flattened strings."""

    def test_chat_calls_aggregator_chat_complete(self):
        app, mock_agg, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                    ],
                },
            )
            assert resp.status_code == 200
            # Verify aggregator.chat_complete was called (not complete)
            mock_agg.chat_complete.assert_awaited_once()

    def test_chat_passes_structured_messages(self):
        app, mock_agg, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            )
            call_kwargs = mock_agg.chat_complete.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            assert isinstance(messages, list)
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "What is 2+2?"

    def test_chat_response_has_message_field(self):
        app, _, _ = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            data = resp.json()
            assert "choices" in data
            assert "message" in data["choices"][0]
            assert "content" in data["choices"][0]["message"]


# -----------------------------------------------------------------------
# Streaming endpoint tests
# -----------------------------------------------------------------------


class TestStreamingEndpoints:
    """Test SSE streaming for /completions and /chat/completions."""

    def test_stream_completions_returns_event_stream(self):
        app, _, mock_ollama = _make_app()

        # Mock generate_stream to yield chunks
        async def mock_stream(*args, **kwargs):
            yield {"response": "Hello", "done": False}
            yield {"response": " world", "done": True, "eval_count": 2, "prompt_eval_count": 3}

        mock_ollama.generate_stream = mock_stream

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hi", "stream": True},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_completions_sse_format(self):
        app, _, mock_ollama = _make_app()

        async def mock_stream(*args, **kwargs):
            yield {"response": "Hi", "done": True, "eval_count": 1, "prompt_eval_count": 2}

        mock_ollama.generate_stream = mock_stream

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello", "stream": True},
            )
            lines = resp.text.strip().split("\n\n")
            # Should have at least one data: line and a [DONE]
            assert any(line.startswith("data: ") for line in lines)
            assert lines[-1] == "data: [DONE]"

    def test_stream_completions_final_event_has_usage(self):
        app, _, mock_ollama = _make_app()

        async def mock_stream(*args, **kwargs):
            yield {"response": "OK", "done": True, "eval_count": 5, "prompt_eval_count": 10}

        mock_ollama.generate_stream = mock_stream

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "test", "stream": True},
            )
            # Parse the data events
            for line in resp.text.strip().split("\n\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    if event["choices"][0]["finish_reason"] == "stop":
                        assert "usage" in event
                        assert event["usage"]["completion_tokens"] == 5
                        assert event["usage"]["prompt_tokens"] == 10

    def test_stream_chat_returns_event_stream(self):
        app, _, mock_ollama = _make_app()

        async def mock_stream(*args, **kwargs):
            yield {"message": {"role": "assistant", "content": "Hi"}, "done": True, "eval_count": 1, "prompt_eval_count": 2}

        mock_ollama.chat_stream = mock_stream

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_chat_sse_has_delta(self):
        app, _, mock_ollama = _make_app()

        async def mock_stream(*args, **kwargs):
            yield {"message": {"role": "assistant", "content": "Hey"}, "done": True, "eval_count": 1, "prompt_eval_count": 1}

        mock_ollama.chat_stream = mock_stream

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            for line in resp.text.strip().split("\n\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    assert "delta" in event["choices"][0]
                    assert event["object"] == "chat.completion.chunk"
