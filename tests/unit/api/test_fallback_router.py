"""Unit tests for FallbackRouter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from llm_inference_engine.api.fallback_router import FallbackRouter
from llm_inference_engine.integration.backend import BackendResult


@pytest.fixture
def mock_pool() -> MagicMock:
    pool = MagicMock()
    pool.get_healthy_backend = MagicMock()
    pool.record_success = MagicMock()
    pool.record_failure = MagicMock()
    return pool


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.generate = AsyncMock(
        return_value=BackendResult(text="fallback response", tokens_used=5)
    )
    backend.chat = AsyncMock(
        return_value=BackendResult(text="fallback chat", tokens_used=3)
    )
    return backend


class TestFallbackRouterInit:
    def test_stores_pool_and_model(self, mock_pool: MagicMock) -> None:
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")
        assert router._pool is mock_pool
        assert router._fallback_model == "mistral"
        assert router._cache is None

    def test_stores_optional_cache(self, mock_pool: MagicMock) -> None:
        cache = MagicMock()
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral", cache=cache)
        assert router._cache is cache


class TestFallbackRouterRoute:
    async def test_uses_fallback_model_when_healthy_backend_available(
        self, mock_pool: MagicMock, mock_backend: MagicMock
    ) -> None:
        mock_pool.get_healthy_backend.return_value = mock_backend
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")

        result = await router.route("llama3", "hello")

        assert result.text == "fallback response"
        mock_pool.record_success.assert_called_once_with(mock_backend)

    async def test_records_failure_when_fallback_backend_raises(
        self, mock_pool: MagicMock, mock_backend: MagicMock
    ) -> None:
        mock_backend.generate = AsyncMock(side_effect=RuntimeError("backend down"))
        mock_pool.get_healthy_backend.return_value = mock_backend
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral", cache=mock_cache)

        with pytest.raises(HTTPException) as exc_info:
            await router.route("llama3", "hello")

        assert exc_info.value.status_code == 503
        mock_pool.record_failure.assert_called_once_with(mock_backend)

    async def test_uses_cache_when_no_healthy_backend(
        self, mock_pool: MagicMock
    ) -> None:
        mock_pool.get_healthy_backend.return_value = None
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="cached text")
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral", cache=mock_cache)

        result = await router.route("llama3", "hello")

        assert result.text == "cached text"
        assert result.metadata == {"cache_hit": True, "stale_fallback": True}

    async def test_raises_503_when_all_strategies_exhausted(
        self, mock_pool: MagicMock
    ) -> None:
        mock_pool.get_healthy_backend.return_value = None
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")

        with pytest.raises(HTTPException) as exc_info:
            await router.route("llama3", "hello")

        assert exc_info.value.status_code == 503

    async def test_skips_fallback_model_when_same_as_primary(
        self, mock_pool: MagicMock
    ) -> None:
        """When fallback_model == model, skip strategy 1."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="stale")
        router = FallbackRouter(pool=mock_pool, fallback_model="llama3", cache=mock_cache)

        result = await router.route("llama3", "hello")

        mock_pool.get_healthy_backend.assert_not_called()
        assert result.text == "stale"


class TestFallbackRouterRouteChat:
    async def test_uses_fallback_model_for_chat(
        self, mock_pool: MagicMock, mock_backend: MagicMock
    ) -> None:
        mock_pool.get_healthy_backend.return_value = mock_backend
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")

        result = await router.route_chat("llama3", [{"role": "user", "content": "hi"}])

        assert result.text == "fallback chat"
        mock_pool.record_success.assert_called_once_with(mock_backend)

    async def test_records_failure_when_chat_backend_raises(
        self, mock_pool: MagicMock, mock_backend: MagicMock
    ) -> None:
        mock_backend.chat = AsyncMock(side_effect=RuntimeError("down"))
        mock_pool.get_healthy_backend.return_value = mock_backend
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")

        with pytest.raises(HTTPException) as exc_info:
            await router.route_chat("llama3", [{"role": "user", "content": "hi"}])

        assert exc_info.value.status_code == 503
        mock_pool.record_failure.assert_called_once_with(mock_backend)

    async def test_raises_503_when_no_healthy_backend_for_chat(
        self, mock_pool: MagicMock
    ) -> None:
        mock_pool.get_healthy_backend.return_value = None
        router = FallbackRouter(pool=mock_pool, fallback_model="mistral")

        with pytest.raises(HTTPException) as exc_info:
            await router.route_chat("llama3", [{"role": "user", "content": "hi"}])

        assert exc_info.value.status_code == 503
