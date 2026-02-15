"""Unit tests for core types."""

import pytest
from llm_inference_engine.core.types import (
    GenerationConfig,
    Request,
    RequestStatus,
    Response,
    GenerationResult,
)


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.0
        assert config.stop_sequences == []

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.9,
            top_p=0.95,
            stop_sequences=["STOP"],
        )
        assert config.max_tokens == 512
        assert config.temperature == 0.9
        assert config.top_p == 0.95
        assert config.stop_sequences == ["STOP"]

    def test_validation_max_tokens(self) -> None:
        """Test max_tokens validation."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            GenerationConfig(max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            GenerationConfig(max_tokens=-1)

    def test_validation_temperature(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValueError, match="temperature must be between"):
            GenerationConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between"):
            GenerationConfig(temperature=2.1)

    def test_validation_top_p(self) -> None:
        """Test top_p validation."""
        with pytest.raises(ValueError, match="top_p must be between"):
            GenerationConfig(top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be between"):
            GenerationConfig(top_p=1.1)


class TestRequest:
    """Tests for Request."""

    def test_request_creation(self) -> None:
        """Test creating a request."""
        config = GenerationConfig()
        request = Request(
            request_id="test-123",
            prompt="Test prompt",
            model="mistral",
            generation_config=config,
        )

        assert request.request_id == "test-123"
        assert request.prompt == "Test prompt"
        assert request.model == "mistral"
        assert request.status == RequestStatus.PENDING
        assert request.priority == 0

    def test_request_validation_empty_prompt(self) -> None:
        """Test validation of empty prompt."""
        config = GenerationConfig()
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            Request(
                request_id="test-123",
                prompt="",
                model="mistral",
                generation_config=config,
            )

    def test_request_validation_empty_id(self) -> None:
        """Test validation of empty request_id."""
        config = GenerationConfig()
        with pytest.raises(ValueError, match="request_id cannot be empty"):
            Request(
                request_id="",
                prompt="Test prompt",
                model="mistral",
                generation_config=config,
            )


class TestResponse:
    """Tests for Response."""

    def test_successful_response(self) -> None:
        """Test successful response."""
        result = GenerationResult(
            request_id="test-123",
            text="Generated text",
            finish_reason="stop",
            tokens_used=10,
            latency_ms=100.0,
            model="mistral",
        )

        response = Response(
            request_id="test-123",
            result=result,
        )

        assert response.is_success is True
        assert response.error is None
        assert response.result == result

    def test_failed_response(self) -> None:
        """Test failed response."""
        response = Response(
            request_id="test-123",
            error="Something went wrong",
            status=RequestStatus.FAILED,
        )

        assert response.is_success is False
        assert response.error == "Something went wrong"
        assert response.result is None
