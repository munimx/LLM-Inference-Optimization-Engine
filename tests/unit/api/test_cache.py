"""Unit tests for SemanticCache."""

import time

import pytest

from llm_inference_engine.api.cache import SemanticCache


class TestSemanticCache:
    """Tests for SemanticCache."""

    def test_cache_miss_returns_none(self) -> None:
        """Missing key should return None."""
        cache = SemanticCache()
        assert cache.get("llama3:8b", "Hello") is None

    def test_cache_hit_after_put(self) -> None:
        """A cached response should be returned on the next get."""
        cache = SemanticCache()
        cache.put("llama3:8b", "Hello", "Hi there!")
        assert cache.get("llama3:8b", "Hello") == "Hi there!"

    def test_cache_miss_increments_counter(self) -> None:
        """Cache miss should increment misses counter."""
        cache = SemanticCache()
        cache.get("m", "p")
        assert cache.misses == 1

    def test_cache_hit_increments_counter(self) -> None:
        """Cache hit should increment hits counter."""
        cache = SemanticCache()
        cache.put("m", "p", "response")
        cache.get("m", "p")
        assert cache.hits == 1

    def test_lru_eviction(self) -> None:
        """Least-recently-used entry should be evicted when max_size is reached."""
        cache = SemanticCache(max_size=2)
        cache.put("m", "p1", "r1")
        cache.put("m", "p2", "r2")
        cache.put("m", "p3", "r3")  # evicts p1
        assert cache.get("m", "p1") is None
        assert cache.get("m", "p2") == "r2"
        assert cache.get("m", "p3") == "r3"

    def test_lru_access_updates_order(self) -> None:
        """Accessing an entry should protect it from LRU eviction."""
        cache = SemanticCache(max_size=2)
        cache.put("m", "p1", "r1")
        cache.put("m", "p2", "r2")
        cache.get("m", "p1")  # p1 is now most-recently used
        cache.put("m", "p3", "r3")  # should evict p2
        assert cache.get("m", "p1") == "r1"
        assert cache.get("m", "p2") is None

    def test_ttl_expiry(self) -> None:
        """Entries older than TTL should be treated as misses."""
        cache = SemanticCache(ttl_seconds=0.01)
        cache.put("m", "p", "r")
        time.sleep(0.05)
        assert cache.get("m", "p") is None

    def test_put_updates_existing_entry(self) -> None:
        """Putting with the same key should update the response."""
        cache = SemanticCache()
        cache.put("m", "p", "v1")
        cache.put("m", "p", "v2")
        assert cache.get("m", "p") == "v2"
        assert cache.size == 1

    def test_invalidate(self) -> None:
        """invalidate() should remove the specified entry."""
        cache = SemanticCache()
        cache.put("m", "p", "r")
        result = cache.invalidate("m", "p")
        assert result is True
        assert cache.get("m", "p") is None

    def test_invalidate_nonexistent_returns_false(self) -> None:
        """invalidating a nonexistent key should return False."""
        cache = SemanticCache()
        assert cache.invalidate("m", "missing") is False

    def test_clear(self) -> None:
        """clear() should remove all entries."""
        cache = SemanticCache()
        for i in range(5):
            cache.put("m", f"p{i}", f"r{i}")
        cache.clear()
        assert cache.size == 0

    def test_invalid_max_size(self) -> None:
        """Non-positive max_size should raise ValueError."""
        with pytest.raises(ValueError, match="max_size"):
            SemanticCache(max_size=0)

    def test_invalid_ttl(self) -> None:
        """Non-positive ttl_seconds should raise ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds"):
            SemanticCache(ttl_seconds=0.0)

    def test_size_property(self) -> None:
        """size should reflect the number of stored entries."""
        cache = SemanticCache()
        assert cache.size == 0
        cache.put("m", "p", "r")
        assert cache.size == 1

    def test_different_models_are_separate_keys(self) -> None:
        """Same prompt for different models should be stored independently."""
        cache = SemanticCache()
        cache.put("llama3:8b", "hi", "llama response")
        cache.put("mistral:7b", "hi", "mistral response")
        assert cache.get("llama3:8b", "hi") == "llama response"
        assert cache.get("mistral:7b", "hi") == "mistral response"
