"""Unit tests for RedisCache."""

import fakeredis.aioredis
import pytest

from llm_inference_engine.api.cache import RedisCache


@pytest.fixture
async def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
async def cache(redis):
    return RedisCache(redis_client=redis, max_size=5, ttl_seconds=60.0)


class TestRedisCache:
    async def test_get_returns_none_on_miss(self, cache: RedisCache) -> None:
        result = await cache.get("llama3", "hello world")
        assert result is None
        assert cache.misses == 1

    async def test_put_and_get(self, cache: RedisCache) -> None:
        await cache.put("llama3", "hello world", "generated response")
        result = await cache.get("llama3", "hello world")
        assert result == "generated response"
        assert cache.hits == 1

    async def test_normalises_key(self, cache: RedisCache) -> None:
        await cache.put("llama3", "  Hello World  ", "response")
        result = await cache.get("llama3", "hello world")
        assert result == "response"

    async def test_different_models_separate_entries(self, cache: RedisCache) -> None:
        await cache.put("llama3", "question", "answer1")
        await cache.put("mistral", "question", "answer2")
        assert await cache.get("llama3", "question") == "answer1"
        assert await cache.get("mistral", "question") == "answer2"

    async def test_lru_eviction_at_max_size(self, cache: RedisCache) -> None:
        for i in range(6):
            await cache.put("m", f"prompt{i}", f"resp{i}")
        size = await cache.size
        assert size <= 5

    async def test_clear(self, cache: RedisCache) -> None:
        await cache.put("llama3", "test", "response")
        await cache.clear()
        result = await cache.get("llama3", "test")
        assert result is None

    async def test_hits_misses_tracking(self, cache: RedisCache) -> None:
        await cache.put("m", "p", "r")
        await cache.get("m", "p")   # hit
        await cache.get("m", "x")   # miss
        assert cache.hits == 1
        assert cache.misses == 1
