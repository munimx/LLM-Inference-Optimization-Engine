"""Tests for circuit breaker, queue limits, and per-request timeout."""

import time

import pytest

from llm_inference_engine.api.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Unit tests for CircuitBreaker."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Manually simulate half-open
        cb._state = CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # After success, two more failures shouldn't open (count reset)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestRequestTimeoutField:
    """Test timeout_seconds on request models."""

    def test_completion_request_default_none(self):
        from llm_inference_engine.api.models import CompletionRequest
        req = CompletionRequest(model="llama3:8b", prompt="hi")
        assert req.timeout_seconds is None

    def test_completion_request_custom_timeout(self):
        from llm_inference_engine.api.models import CompletionRequest
        req = CompletionRequest(model="llama3:8b", prompt="hi", timeout_seconds=30.0)
        assert req.timeout_seconds == 30.0

    def test_chat_request_default_none(self):
        from llm_inference_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="llama3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.timeout_seconds is None

    def test_chat_request_custom_timeout(self):
        from llm_inference_engine.api.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="llama3:8b",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=10.0,
        )
        assert req.timeout_seconds == 10.0
