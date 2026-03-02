"""Integration tests for FastAPI endpoints with mocked Ollama."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
import httpx
from fastapi.testclient import TestClient

from llm_inference_engine.api.server import create_app
from llm_inference_engine.config import InferenceConfig


OLLAMA_GENERATE_RESPONSE: dict[str, Any] = {
    "model": "llama3.1:8b",
    "response": "Hello! How can I help you?",
    "done": True,
    "done_reason": "stop",
    "eval_count": 12,
    "prompt_eval_count": 5,
    "eval_duration": 500_000_000,
}


@pytest.fixture
def test_app() -> TestClient:
    """Create a test FastAPI app with a real (but patched) Ollama client."""
    config = InferenceConfig()

    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        # Health check response
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
        # Generate endpoint
        mock.post("/api/generate").mock(
            return_value=httpx.Response(200, json=OLLAMA_GENERATE_RESPONSE)
        )

        app = create_app(config)
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_app: TestClient) -> None:
        """Health endpoint should return 200."""
        response = test_app.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, test_app: TestClient) -> None:
        """Health response should contain required fields."""
        data = test_app.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "ollama_available" in data


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, test_app: TestClient) -> None:
        """Metrics endpoint should return 200."""
        response = test_app.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_structure(self, test_app: TestClient) -> None:
        """Metrics should contain expected fields."""
        data = test_app.get("/metrics").json()
        assert "committed_memory_gb" in data
        assert "total_requests" in data
        assert "cache_hits" in data


class TestAPIModels:
    """Tests for API model validation at the HTTP layer."""

    def test_completions_empty_prompt_returns_422(self, test_app: TestClient) -> None:
        """Empty prompt should result in a 422 Unprocessable Entity."""
        response = test_app.post(
            "/completions",
            json={"model": "llama3.1:8b", "prompt": "   "},
        )
        assert response.status_code == 422

    def test_chat_completions_empty_messages_returns_422(
        self, test_app: TestClient
    ) -> None:
        """Empty messages list should return 422."""
        response = test_app.post(
            "/chat/completions",
            json={"model": "llama3.1:8b", "messages": []},
        )
        assert response.status_code == 422


class TestOpenAPISpec:
    """Tests to ensure OpenAPI spec is generated correctly."""

    def test_openapi_json_available(self, test_app: TestClient) -> None:
        """OpenAPI schema endpoint should return 200."""
        response = test_app.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_available(self, test_app: TestClient) -> None:
        """Swagger UI should be accessible."""
        response = test_app.get("/docs")
        assert response.status_code == 200
