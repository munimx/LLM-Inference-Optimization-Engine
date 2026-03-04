"""Dedicated tests for CircuitBreaker state machine."""

import time
from unittest.mock import patch

from llm_inference_engine.api.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInit:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    def test_custom_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStateTransitions:
    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 failure after reset, not 3
        assert cb.state == CircuitState.CLOSED

    def test_open_to_half_open_after_cooldown(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_available is True

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_open_before_cooldown(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=100.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    def test_reset_clears_all(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    def test_consecutive_successes_keep_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(10):
            cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_name_attribute(self) -> None:
        cb = CircuitBreaker(name="test-backend")
        assert cb._name == "test-backend"

    def test_cooldown_timer_uses_monotonic(self) -> None:
        """Verify that cooldown uses time.monotonic (not wall clock)."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
        cb.record_failure()
        # Simulate time passing by patching monotonic
        with patch("llm_inference_engine.api.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 11.0
            assert cb.state == CircuitState.HALF_OPEN
