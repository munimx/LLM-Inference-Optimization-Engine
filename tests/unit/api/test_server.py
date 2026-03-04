"""Unit tests for FastAPI server routes and app creation."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_inference_engine.api.dependencies import get_pool, get_throttler
from llm_inference_engine.api.server import VERSION, create_app
from llm_inference_engine.config import InferenceConfig
from llm_inference_engine.integration.backend import BackendResult
from llm_inference_engine.optimization.throttler import AdmissionDecision, ThrottlerStats


def _make_backend_result(text: str = "hello world") -> BackendResult:
    return BackendResult(
        text=text,
        tokens_used=10,
        prompt_tokens=5,
        finish_reason="stop",
        latency_ms=25.0,
    )


def _make_test_client(
    result: BackendResult | None = None,
    throttle: AdmissionDecision = AdmissionDecision.ACCEPT,
) -> TestClient:
    """Build a TestClient with DI overrides and mock app.state values.

    Patches Redis and vLLM connections to prevent real network calls during tests.
    """
    if result is None:
        result = _make_backend_result()

    # Mock backend
    mock_backend = MagicMock()
    mock_backend.generate = AsyncMock(return_value=result)
    mock_backend.chat = AsyncMock(return_value=result)

    # Mock pool
    mock_pool = MagicMock()
    mock_pool.get_healthy_backend.return_value = mock_backend
    mock_pool.healthy_count.return_value = 1
    mock_pool._backends = [mock_backend]
    mock_pool.record_success = MagicMock()
    mock_pool.record_failure = MagicMock()
    mock_pool.close = AsyncMock()

    # Mock throttler
    mock_throttler = MagicMock()
    mock_throttler.check.return_value = throttle
    mock_throttler.stats = ThrottlerStats(
        kv_cache_usage=0.30,
        soft_limit=0.70,
        hard_limit=0.90,
        active_requests=0,
    )
    mock_throttler.increment_active = MagicMock()
    mock_throttler.decrement_active = MagicMock()
    mock_throttler.start = AsyncMock()
    mock_throttler.stop = AsyncMock()

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.hits = 0
    mock_cache.misses = 0
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.put = AsyncMock()
    mock_cache.close = AsyncMock()

    # Mock coalescer
    mock_coalescer = MagicMock()

    async def _passthrough(model, prompt, producer):
        return await producer()

    mock_coalescer.coalesce = _passthrough

    # Mock model router
    mock_model_router = MagicMock()
    mock_model_router.route.side_effect = lambda prompt, explicit_model=None: explicit_model or "llama3"
    mock_model_router.route_chat.side_effect = lambda msgs, explicit_model=None: explicit_model or "llama3"

    # Mock fallback router
    mock_fallback_router = MagicMock()

    app = create_app(config=InferenceConfig())
    app.dependency_overrides[get_pool] = lambda: mock_pool
    app.dependency_overrides[get_throttler] = lambda: mock_throttler

    # Patch lifespan to use our mocks instead of real Redis/vLLM
    @asynccontextmanager
    async def _mock_lifespan(app):
        app.state.pool = mock_pool
        app.state.cache = mock_cache
        app.state.throttler = mock_throttler
        app.state.coalescer = mock_coalescer
        app.state.model_router = mock_model_router
        app.state.fallback_router = mock_fallback_router
        yield

    app.router.lifespan_context = _mock_lifespan

    return TestClient(app, raise_server_exceptions=True)


class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        app = create_app(InferenceConfig())
        assert isinstance(app, FastAPI)

    def test_app_has_correct_version(self) -> None:
        app = create_app(InferenceConfig())
        assert app.version == VERSION

    def test_app_with_none_config_uses_defaults(self) -> None:
        app = create_app(None)
        assert isinstance(app, FastAPI)

    def test_version_string_format(self) -> None:
        parts = VERSION.split(".")
        assert len(parts) == 3


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_has_backend_available_field(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/health").json()
        assert "backend_available" in data
        assert data["backend_available"] is True

    def test_has_version(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/health").json()
        assert data["version"] == VERSION


class TestMetricsEndpoint:
    def test_returns_200(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_has_kv_cache_usage(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/metrics").json()
        assert "kv_cache_usage" in data

    def test_has_healthy_backends(self) -> None:
        client = _make_test_client()
        with client:
            data = client.get("/metrics").json()
        assert "healthy_backends" in data


class TestCompletionsEndpoint:
    def test_valid_request_returns_200(self) -> None:
        client = _make_test_client(_make_backend_result())
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "Hello world"},
            )
        assert resp.status_code == 200

    def test_response_has_choices(self) -> None:
        client = _make_test_client(_make_backend_result())
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "Hello world"},
            ).json()
        assert "choices" in data
        assert len(data["choices"]) == 1

    def test_response_text_matches(self) -> None:
        client = _make_test_client(_make_backend_result(text="generated text"))
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "Hello"},
            ).json()
        assert data["choices"][0]["text"] == "generated text"

    def test_response_has_usage(self) -> None:
        client = _make_test_client()
        with client:
            data = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "Hello"},
            ).json()
        assert "usage" in data

    def test_empty_prompt_returns_422(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "   "},
            )
        assert resp.status_code == 422

    def test_throttle_reject_returns_429(self) -> None:
        client = _make_test_client(throttle=AdmissionDecision.REJECT)
        with client:
            resp = client.post(
                "/completions",
                json={"model": "llama3", "prompt": "Hello"},
            )
        assert resp.status_code == 429


class TestChatCompletionsEndpoint:
    def test_valid_request_returns_200(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "llama3",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200

    def test_response_has_choices(self) -> None:
        client = _make_test_client()
        with client:
            data = client.post(
                "/chat/completions",
                json={
                    "model": "llama3",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ).json()
        assert "choices" in data

    def test_empty_messages_returns_422(self) -> None:
        client = _make_test_client()
        with client:
            resp = client.post(
                "/chat/completions",
                json={"model": "llama3", "messages": []},
            )
        assert resp.status_code == 422
