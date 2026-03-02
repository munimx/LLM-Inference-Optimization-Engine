"""Unit tests for API Pydantic models."""

import pytest
from pydantic import ValidationError

from llm_inference_engine.api.models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)


class TestCompletionRequest:
    """Tests for CompletionRequest."""

    def test_valid_request(self) -> None:
        """Valid data should construct without error."""
        req = CompletionRequest(model="llama3.1:8b", prompt="Hello!")
        assert req.model == "llama3.1:8b"
        assert req.max_tokens == 256
        assert req.temperature == 0.7

    def test_empty_prompt_raises(self) -> None:
        """Empty prompt should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            CompletionRequest(model="llama3.1:8b", prompt="  ")

    def test_max_tokens_bounds(self) -> None:
        """max_tokens below 1 should fail validation."""
        with pytest.raises(ValidationError):
            CompletionRequest(model="llama3.1:8b", prompt="Hi", max_tokens=0)

    def test_temperature_bounds(self) -> None:
        """Temperature outside [0, 2] should fail."""
        with pytest.raises(ValidationError):
            CompletionRequest(model="llama3.1:8b", prompt="Hi", temperature=3.0)

    def test_priority_bounds(self) -> None:
        """Priority outside [0, 10] should fail."""
        with pytest.raises(ValidationError):
            CompletionRequest(model="llama3.1:8b", prompt="Hi", priority=11)

    def test_default_stop_is_empty_list(self) -> None:
        """Default stop sequences should be an empty list."""
        req = CompletionRequest(model="m", prompt="hi")
        assert req.stop == []


class TestChatMessage:
    """Tests for ChatMessage."""

    def test_valid_roles(self) -> None:
        """system, user, assistant roles should all be valid."""
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self) -> None:
        """Unknown role should raise ValidationError."""
        with pytest.raises(ValidationError, match="role"):
            ChatMessage(role="bot", content="hello")


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest."""

    def test_valid_request(self) -> None:
        """Valid messages list should construct without error."""
        req = ChatCompletionRequest(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert len(req.messages) == 1

    def test_empty_messages_raises(self) -> None:
        """Empty messages list should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="llama3.1:8b", messages=[])


class TestCompletionResponse:
    """Tests for response model construction."""

    def test_completion_response_valid(self) -> None:
        from llm_inference_engine.api.models import CompletionChoice, CompletionResponse, UsageInfo
        choice = CompletionChoice(index=0, text="hello", finish_reason="stop")
        usage = UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        resp = CompletionResponse(
            id="req-1", model="llama3:8b", choices=[choice], usage=usage, latency_ms=20.0
        )
        assert resp.object == "text_completion"
        assert resp.model == "llama3:8b"

    def test_usage_info_totals(self) -> None:
        from llm_inference_engine.api.models import UsageInfo
        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30

    def test_chat_completion_message_default_role(self) -> None:
        from llm_inference_engine.api.models import ChatCompletionMessage
        msg = ChatCompletionMessage(content="hello")
        assert msg.role == "assistant"


class TestHealthAndMetricsModels:
    """Tests for HealthResponse and MetricsResponse."""

    def test_health_response_defaults(self) -> None:
        from llm_inference_engine.api.models import HealthResponse
        h = HealthResponse(status="ok", ollama_available=True)
        assert h.details == {}
        assert h.version == "0.1.0"

    def test_health_response_degraded(self) -> None:
        from llm_inference_engine.api.models import HealthResponse
        h = HealthResponse(status="degraded", ollama_available=False)
        assert h.ollama_available is False

    def test_metrics_response_fields(self) -> None:
        from llm_inference_engine.api.models import MetricsResponse
        m = MetricsResponse(
            committed_memory_gb=2.0,
            available_memory_gb=12.0,
            memory_limit_gb=14.0,
            active_requests=3,
            cache_hits=100,
            cache_misses=20,
            total_requests=500,
        )
        assert m.cache_hits == 100

    def test_error_response_minimal(self) -> None:
        from llm_inference_engine.api.models import ErrorResponse
        e = ErrorResponse(error="something went wrong")
        assert e.detail is None
        assert e.request_id is None

    def test_error_response_with_all_fields(self) -> None:
        from llm_inference_engine.api.models import ErrorResponse
        e = ErrorResponse(error="err", detail="details here", request_id="req-1")
        assert e.detail == "details here"
        assert e.request_id == "req-1"


class TestCompletionRequestExtras:
    """Additional edge-case tests for CompletionRequest."""

    def test_max_tokens_upper_bound(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", max_tokens=32_768)
        assert req.max_tokens == 32_768

    def test_max_tokens_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", max_tokens=32_769)

    def test_temperature_zero_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", temperature=0.0)
        assert req.temperature == 0.0

    def test_temperature_two_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", temperature=2.0)
        assert req.temperature == 2.0

    def test_top_p_zero_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", top_p=0.0)
        assert req.top_p == 0.0

    def test_top_p_one_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", top_p=1.0)
        assert req.top_p == 1.0

    def test_top_p_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", top_p=1.1)

    def test_priority_zero_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", priority=0)
        assert req.priority == 0

    def test_priority_ten_valid(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", priority=10)
        assert req.priority == 10

    def test_stop_sequences_accepted(self) -> None:
        req = CompletionRequest(model="m", prompt="hi", stop=["</s>", "\n\n"])
        assert req.stop == ["</s>", "\n\n"]

    def test_newline_only_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="\n\t\r")
