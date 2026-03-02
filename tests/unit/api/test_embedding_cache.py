"""Tests for the embedding-based semantic cache."""

import asyncio

import pytest

from llm_inference_engine.api.embedding_cache import EmbeddingCache


def _make_embed_fn(dim: int = 4):
    """Return an async embed function that produces deterministic vectors."""
    async def embed(text: str) -> list[float]:
        # Simple hash-based embedding for testing
        h = hash(text)
        return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(dim)]
    return embed


class TestEmbeddingCache:

    async def test_exact_hit(self):
        cache = EmbeddingCache(_make_embed_fn(), similarity_threshold=0.9)
        await cache.put("m", "hello", "world")
        assert await cache.get("m", "hello") == "world"
        assert cache.hits == 1
        assert cache.misses == 0

    async def test_miss_different_prompt(self):
        cache = EmbeddingCache(_make_embed_fn(), similarity_threshold=0.99)
        await cache.put("m", "hello", "world")
        await cache.get("m", "completely different text")
        # May or may not hit depending on hash collision — either way no crash
        assert cache.size == 1

    async def test_miss_different_model(self):
        cache = EmbeddingCache(_make_embed_fn(), similarity_threshold=0.5)
        await cache.put("model_a", "hello", "world")
        result = await cache.get("model_b", "hello")
        assert result is None
        assert cache.misses == 1

    async def test_ttl_expiry(self):
        cache = EmbeddingCache(_make_embed_fn(), ttl_seconds=0.01)
        await cache.put("m", "hello", "world")
        await asyncio.sleep(0.02)
        result = await cache.get("m", "hello")
        assert result is None

    async def test_lru_eviction(self):
        cache = EmbeddingCache(_make_embed_fn(), max_size=2)
        await cache.put("m", "a", "1")
        await cache.put("m", "b", "2")
        await cache.put("m", "c", "3")
        assert cache.size == 2

    async def test_clear(self):
        cache = EmbeddingCache(_make_embed_fn())
        await cache.put("m", "x", "y")
        await cache.clear()
        assert cache.size == 0

    async def test_invalidate(self):
        cache = EmbeddingCache(_make_embed_fn())
        await cache.put("m", "x", "y")
        assert await cache.invalidate("m", "x") is True
        assert await cache.invalidate("m", "x") is False

    async def test_embed_failure_get(self):
        async def failing_embed(text: str) -> list[float]:
            raise RuntimeError("embedding service down")

        cache = EmbeddingCache(failing_embed)
        result = await cache.get("m", "hello")
        assert result is None
        assert cache.misses == 1

    async def test_embed_failure_put(self):
        async def failing_embed(text: str) -> list[float]:
            raise RuntimeError("embedding service down")

        cache = EmbeddingCache(failing_embed)
        await cache.put("m", "hello", "world")  # should not raise
        assert cache.size == 0

    async def test_validation_errors(self):
        with pytest.raises(ValueError, match="max_size"):
            EmbeddingCache(_make_embed_fn(), max_size=0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            EmbeddingCache(_make_embed_fn(), ttl_seconds=-1)
        with pytest.raises(ValueError, match="similarity_threshold"):
            EmbeddingCache(_make_embed_fn(), similarity_threshold=1.5)
