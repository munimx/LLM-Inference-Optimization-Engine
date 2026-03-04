"""Tests for FastAPI dependency injection providers."""

from unittest.mock import MagicMock

import pytest
from starlette.datastructures import State

from llm_inference_engine.api.dependencies import (
    get_aggregator,
    get_cache,
    get_memory_estimator,
    get_ollama_client,
    get_scheduler,
    get_throttler,
)


def _make_request_with_state(**attrs: object) -> MagicMock:
    """Create a mock FastAPI Request with the given app.state attributes."""
    mock_request = MagicMock()
    state = State()
    for key, value in attrs.items():
        setattr(state, key, value)
    mock_request.app.state = state
    return mock_request


class TestDependencyHappyPath:
    def test_get_ollama_client(self) -> None:
        sentinel = object()
        req = _make_request_with_state(ollama_client=sentinel)
        assert get_ollama_client(req) is sentinel

    def test_get_scheduler(self) -> None:
        sentinel = object()
        req = _make_request_with_state(scheduler=sentinel)
        assert get_scheduler(req) is sentinel

    def test_get_cache(self) -> None:
        sentinel = object()
        req = _make_request_with_state(cache=sentinel)
        assert get_cache(req) is sentinel

    def test_get_aggregator(self) -> None:
        sentinel = object()
        req = _make_request_with_state(aggregator=sentinel)
        assert get_aggregator(req) is sentinel

    def test_get_throttler(self) -> None:
        sentinel = object()
        req = _make_request_with_state(throttler=sentinel)
        assert get_throttler(req) is sentinel

    def test_get_memory_estimator(self) -> None:
        sentinel = object()
        req = _make_request_with_state(memory_estimator=sentinel)
        assert get_memory_estimator(req) is sentinel


class TestDependencyMissingAttribute:
    def test_missing_ollama_client_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="OllamaClient not initialized"):
            get_ollama_client(req)

    def test_missing_scheduler_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            get_scheduler(req)

    def test_missing_cache_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Cache not initialized"):
            get_cache(req)

    def test_missing_aggregator_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Aggregator not initialized"):
            get_aggregator(req)

    def test_missing_throttler_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Throttler not initialized"):
            get_throttler(req)

    def test_missing_memory_estimator_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="MemoryEstimator not initialized"):
            get_memory_estimator(req)
