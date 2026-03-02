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
