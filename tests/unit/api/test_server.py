"""Unit tests for FastAPI server routes and app creation."""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from llm_inference_engine.api.dependencies import get_aggregator, get_cache, get_ollama_client, get_throttler
from llm_inference_engine.api.server import VERSION, create_app
from llm_inference_engine.config import InferenceConfig
from llm_inference_engine.core.types import (
    GenerationResult,
    RequestStatus,
    Response,
)
from llm_inference_engine.optimization.throttler import ThrottlerStats


def _make_result(request_id: str = "req-1", text: str = "hello") -> Response:
    return Response(
        request_id=request_id,
        result=GenerationResult(
            request_id=request_id,
            text=text,
            finish_reason="stop",
            tokens_used=10,
            latency_ms=25.0,
            model="llama3:8b",
        ),
        status=RequestStatus.COMPLETED,
    )


def _failed_result(request_id: str = "req-fail") -> Response:
    return Response(
        request_id=request_id,
        result=None,
        error="Inference error",
        status=RequestStatus.FAILED,
    )


def _make_test_client(complete_response: Response | None = None) -> TestClient:
    """Create a TestClient with dependency-overridden aggregator + throttler."""
    if complete_response is None:
        complete_response = _make_result()

    mock_aggregator = MagicMock()
    mock_aggregator.complete = AsyncMock(return_value=complete_response)
    mock_aggregator.chat_complete = AsyncMock(return_value=complete_response)
    mock_aggregator.pending_count = 0
    mock_aggregator.total_requests = 5

    mock_throttler = MagicMock()
    mock_throttler.stats = ThrottlerStats(
        committed_gb=2.0,
        available_gb=12.0,
        memory_limit_gb=14.0,
        soft_limit_gb=11.9,
        active_requests=1,
    )

    mock_cache = MagicMock()
    mock_cache.hits = 10
    mock_cache.misses = 3

    app = create_app(config=InferenceConfig())
    app.dependency_overrides[get_aggregator] = lambda: mock_aggregator
    app.dependency_overrides[get_throttler] = lambda: mock_throttler
    app.dependency_overrides[get_cache] = lambda: mock_cache

    # Patch app.state.ollama_client so health check works without real Ollama
    mock_ollama = MagicMock()
    mock_ollama.is_available = AsyncMock(return_value=True)
    mock_ollama.close = AsyncMock()
    app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

    return TestClient(app, raise_server_exceptions=False)


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_returns_fastapi_instance(self) -> None:
        from fastapi import FastAPI
        app = create_app(InferenceConfig())
        assert isinstance(app, FastAPI)

    def test_app_has_correct_version(self) -> None:
        app = create_app(InferenceConfig())
        assert app.version == VERSION

    def test_app_with_none_config_uses_defaults(self) -> None:
        from fastapi import FastAPI
        app = create_app(None)
        assert isinstance(app, FastAPI)

    def test_version_string_format(self) -> None:
        parts = VERSION.split(".")
        assert len(parts) == 3


class TestCompletionsEndpoint:
    """Tests for POST /completions."""

    def test_valid_request_returns_200(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello world"},
            )
        assert resp.status_code == 200

    def test_response_has_choices(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello world"},
            ).json()
        assert "choices" in data
        assert len(data["choices"]) == 1

    def test_response_text_from_result(self) -> None:
        client = _make_test_client(_make_result(text="generated text"))
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello"},
            ).json()
        assert data["choices"][0]["text"] == "generated text"

    def test_response_has_usage(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello"},
            ).json()
        assert "usage" in data

    def test_failed_result_returns_503(self) -> None:
        client = _make_test_client(_failed_result())
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello"},
            )
        assert resp.status_code == 503

    def test_model_echoed_in_response(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello"},
            ).json()
        assert data["model"] == "llama3:8b"

    def test_response_has_latency_ms(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "Hello"},
            ).json()
        assert "latency_ms" in data

    def test_empty_prompt_returns_422(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3:8b", "prompt": "   "},
            )
        assert resp.status_code == 422


class TestChatCompletionsEndpoint:
    """Tests for POST /chat/completions."""

    def test_valid_request_returns_200(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200

    def test_response_has_choices(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            data = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            ).json()
        assert "choices" in data

    def test_chat_message_is_assistant_role(self) -> None:
        client = _make_test_client(_make_result(text="assistant response"))
        with client:
            data = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            ).json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_failed_result_returns_503(self) -> None:
        client = _make_test_client(_failed_result())
        with client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert resp.status_code == 503

    def test_empty_messages_returns_422(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.post(
                "/chat/completions",
                json={"model": "llama3:8b", "messages": []},
            )
        assert resp.status_code == 422

    def test_system_message_accepted(self) -> None:
        client = _make_test_client(_make_result())
        with client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3:8b",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hi"},
                    ],
                },
            )
        assert resp.status_code == 200


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_memory_fields(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/metrics").json()
        assert "committed_memory_gb" in data
        assert "available_memory_gb" in data
        assert "memory_limit_gb" in data

    def test_metrics_has_cache_fields(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/metrics").json()
        assert "cache_hits" in data
        assert "cache_misses" in data

    def test_metrics_has_total_requests(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/metrics").json()
        assert "total_requests" in data
