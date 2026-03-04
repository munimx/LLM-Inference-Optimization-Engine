"""Tests for FastAPI dependency injection providers."""

from unittest.mock import MagicMock

import pytest
from starlette.datastructures import State

from llm_inference_engine.api.dependencies import (
    get_cache,
    get_coalescer,
    get_fallback_router,
    get_model_router,
    get_pool,
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
    def test_get_pool(self) -> None:
        sentinel = object()
        req = _make_request_with_state(pool=sentinel)
        assert get_pool(req) is sentinel

    def test_get_cache(self) -> None:
        sentinel = object()
        req = _make_request_with_state(cache=sentinel)
        assert get_cache(req) is sentinel

    def test_get_throttler(self) -> None:
        sentinel = object()
        req = _make_request_with_state(throttler=sentinel)
        assert get_throttler(req) is sentinel

    def test_get_coalescer(self) -> None:
        sentinel = object()
        req = _make_request_with_state(coalescer=sentinel)
        assert get_coalescer(req) is sentinel

    def test_get_model_router(self) -> None:
        sentinel = object()
        req = _make_request_with_state(model_router=sentinel)
        assert get_model_router(req) is sentinel

    def test_get_fallback_router(self) -> None:
        sentinel = object()
        req = _make_request_with_state(fallback_router=sentinel)
        assert get_fallback_router(req) is sentinel


class TestDependencyMissingAttribute:
    def test_missing_pool_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="BackendPool not initialized"):
            get_pool(req)

    def test_missing_cache_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Cache not initialized"):
            get_cache(req)

    def test_missing_throttler_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Throttler not initialized"):
            get_throttler(req)

    def test_missing_coalescer_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="Coalescer not initialized"):
            get_coalescer(req)

    def test_missing_model_router_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="ModelRouter not initialized"):
            get_model_router(req)

    def test_missing_fallback_router_raises(self) -> None:
        req = _make_request_with_state()
        with pytest.raises(RuntimeError, match="FallbackRouter not initialized"):
            get_fallback_router(req)
