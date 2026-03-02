"""Unit tests for SemanticCache."""

import time

import pytest

from llm_inference_engine.api.cache import SemanticCache


class TestSemanticCache:
    """Tests for SemanticCache."""

    async def test_cache_miss_returns_none(self) -> None:
        """Missing key should return None."""
        cache = SemanticCache()
        assert await cache.get("llama3:8b", "Hello") is None

    async def test_cache_hit_after_put(self) -> None:
        """A cached response should be returned on the next get."""
        cache = SemanticCache()
        await cache.put("llama3:8b", "Hello", "Hi there!")
        assert await cache.get("llama3:8b", "Hello") == "Hi there!"

    async def test_cache_miss_increments_counter(self) -> None:
        """Cache miss should increment misses counter."""
        cache = SemanticCache()
        await cache.get("m", "p")
        assert cache.misses == 1

    async def test_cache_hit_increments_counter(self) -> None:
        """Cache hit should increment hits counter."""
        cache = SemanticCache()
        await cache.put("m", "p", "response")
        await cache.get("m", "p")
        assert cache.hits == 1

    async def test_lru_eviction(self) -> None:
        """Least-recently-used entry should be evicted when max_size is reached."""
        cache = SemanticCache(max_size=2)
        await cache.put("m", "p1", "r1")
        await cache.put("m", "p2", "r2")
        await cache.put("m", "p3", "r3")  # evicts p1
        assert await cache.get("m", "p1") is None
        assert await cache.get("m", "p2") == "r2"
        assert await cache.get("m", "p3") == "r3"

    async def test_lru_access_updates_order(self) -> None:
        """Accessing an entry should protect it from LRU eviction."""
        cache = SemanticCache(max_size=2)
        await cache.put("m", "p1", "r1")
        await cache.put("m", "p2", "r2")
        await cache.get("m", "p1")  # p1 is now most-recently used
        await cache.put("m", "p3", "r3")  # should evict p2
        assert await cache.get("m", "p1") == "r1"
        assert await cache.get("m", "p2") is None

    async def test_ttl_expiry(self) -> None:
        """Entries older than TTL should be treated as misses."""
        cache = SemanticCache(ttl_seconds=0.01)
        await cache.put("m", "p", "r")
        time.sleep(0.05)
        assert await cache.get("m", "p") is None

    async def test_put_updates_existing_entry(self) -> None:
        """Putting with the same key should update the response."""
        cache = SemanticCache()
        await cache.put("m", "p", "v1")
        await cache.put("m", "p", "v2")
        assert await cache.get("m", "p") == "v2"
        assert cache.size == 1

    async def test_invalidate(self) -> None:
        """invalidate() should remove the specified entry."""
        cache = SemanticCache()
        await cache.put("m", "p", "r")
        result = await cache.invalidate("m", "p")
        assert result is True
        assert await cache.get("m", "p") is None

    async def test_invalidate_nonexistent_returns_false(self) -> None:
        """invalidating a nonexistent key should return False."""
        cache = SemanticCache()
        assert await cache.invalidate("m", "missing") is False

    async def test_clear(self) -> None:
        """clear() should remove all entries."""
        cache = SemanticCache()
        for i in range(5):
            await cache.put("m", f"p{i}", f"r{i}")
        await cache.clear()
        assert cache.size == 0

    def test_invalid_max_size(self) -> None:
        """Non-positive max_size should raise ValueError."""
        with pytest.raises(ValueError, match="max_size"):
            SemanticCache(max_size=0)

    def test_invalid_ttl(self) -> None:
        """Non-positive ttl_seconds should raise ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds"):
            SemanticCache(ttl_seconds=0.0)

    async def test_size_property(self) -> None:
        """size should reflect the number of stored entries."""
        cache = SemanticCache()
        assert cache.size == 0
        await cache.put("m", "p", "r")
        assert cache.size == 1

    async def test_different_models_are_separate_keys(self) -> None:
        """Same prompt for different models should be stored independently."""
        cache = SemanticCache()
        await cache.put("llama3:8b", "hi", "llama response")
        await cache.put("mistral:7b", "hi", "mistral response")
        assert await cache.get("llama3:8b", "hi") == "llama response"
        assert await cache.get("mistral:7b", "hi") == "mistral response"

    async def test_max_size_property(self) -> None:
        """max_size property should return constructor value."""
        cache = SemanticCache(max_size=42)
        assert cache.max_size == 42

    async def test_hits_misses_start_at_zero(self) -> None:
        """Counters should start at zero on a fresh cache."""
        cache = SemanticCache()
        assert cache.hits == 0
        assert cache.misses == 0

    async def test_multiple_hits_accumulate(self) -> None:
        """hits counter accumulates across multiple successful gets."""
        cache = SemanticCache()
        await cache.put("m", "p", "r")
        await cache.get("m", "p")
        await cache.get("m", "p")
        await cache.get("m", "p")
        assert cache.hits == 3

    async def test_multiple_misses_accumulate(self) -> None:
        """misses counter accumulates across multiple empty gets."""
        cache = SemanticCache()
        await cache.get("m", "p1")
        await cache.get("m", "p2")
        assert cache.misses == 2

    async def test_put_after_clear_works(self) -> None:
        """Entries can be added after a clear."""
        cache = SemanticCache()
        await cache.put("m", "p", "r")
        await cache.clear()
        await cache.put("m", "p", "new_r")
        assert await cache.get("m", "p") == "new_r"

    async def test_lru_eviction_exact_order_three_items(self) -> None:
        """With capacity 2, adding a third evicts the LRU (first inserted)."""
        cache = SemanticCache(max_size=2)
        await cache.put("m", "a", "r_a")
        await cache.put("m", "b", "r_b")
        # Access 'a' to make it most-recently-used
        await cache.get("m", "a")
        await cache.put("m", "c", "r_c")  # 'b' should be evicted
        assert await cache.get("m", "a") == "r_a"
        assert await cache.get("m", "b") is None
        assert await cache.get("m", "c") == "r_c"

    async def test_put_updates_ttl_on_overwrite(self) -> None:
        """Overwriting an existing key should reset its TTL."""
        import time
        cache = SemanticCache(ttl_seconds=0.05)
        await cache.put("m", "p", "v1")
        time.sleep(0.03)
        await cache.put("m", "p", "v2")  # should reset TTL
        time.sleep(0.03)
        # total elapsed ~0.06 from first put, but ~0.03 from second put
        assert await cache.get("m", "p") == "v2"

    async def test_invalidate_reduces_size(self) -> None:
        """Size should decrease after a successful invalidation."""
        cache = SemanticCache()
        await cache.put("m", "p1", "r1")
        await cache.put("m", "p2", "r2")
        await cache.invalidate("m", "p1")
        assert cache.size == 1

    async def test_concurrent_puts_no_exception(self) -> None:
        """Concurrent put operations should not raise."""
        import asyncio
        cache = SemanticCache(max_size=20)
        await asyncio.gather(
            *[cache.put("m", f"p{i}", f"r{i}") for i in range(10)]
        )
        assert cache.size <= 20

    async def test_concurrent_gets_no_exception(self) -> None:
        """Concurrent get operations should not raise."""
        import asyncio
        cache = SemanticCache()
        await cache.put("m", "p", "r")
        results = await asyncio.gather(
            *[cache.get("m", "p") for _ in range(10)]
        )
        assert all(r == "r" for r in results)

    async def test_size_after_eviction(self) -> None:
        """Size should never exceed max_size."""
        cache = SemanticCache(max_size=3)
        for i in range(10):
            await cache.put("m", f"p{i}", f"r{i}")
        assert cache.size == 3

    async def test_get_empty_string_prompt(self) -> None:
        """Empty string should be a valid cache key."""
        cache = SemanticCache()
        await cache.put("m", "", "empty prompt response")
        assert await cache.get("m", "") == "empty prompt response"

    async def test_model_name_case_sensitive(self) -> None:
        """Cache keys are case-sensitive for model names."""
        cache = SemanticCache()
        await cache.put("Llama3:8b", "hi", "upper")
        await cache.put("llama3:8b", "hi", "lower")
        assert await cache.get("Llama3:8b", "hi") == "upper"
        assert await cache.get("llama3:8b", "hi") == "lower"
