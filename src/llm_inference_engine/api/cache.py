"""Redis-backed response cache for inference requests.

Provides a response cache with TTL expiry and LRU eviction backed by Redis.
Keys are ``(model, prompt)`` tuples — prompts are normalised via
``.lower().strip()`` before lookup.

The cache is safe to use across multiple workers: all state lives in Redis.

Usage::

    async with RedisCache.connect("redis://localhost:6379/0", max_size=256) as cache:
        hit = await cache.get("mistral-7b", "Hello!")
        if hit is None:
            result = await vllm_generate(...)
            await cache.put("mistral-7b", "Hello!", result)
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

# Redis key prefixes
_CACHE_VALUE_PREFIX = "llm_cache:v:"
_CACHE_LRU_ZSET = "llm_cache:lru"


def _cache_key(model: str, prompt: str) -> str:
    """Build a Redis key from model and normalised prompt."""
    normalised = f"{model}:{prompt.lower().strip()}"
    digest = hashlib.sha256(normalised.encode()).hexdigest()
    return f"{_CACHE_VALUE_PREFIX}{digest}"


class RedisCache:
    """Response cache backed by Redis with LRU eviction and TTL.

    Args:
        redis_client: An async Redis client (``redis.asyncio.Redis``).
        max_size: Maximum number of entries before LRU eviction.
        ttl_seconds: How long entries remain valid.
    """

    def __init__(
        self,
        redis_client: Any,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._redis = redis_client
        self._max_size = max_size
        self._ttl = int(ttl_seconds)
        self._hits = 0
        self._misses = 0
        logger.info("redis_cache_initialized", max_size=max_size, ttl_seconds=ttl_seconds)

    @classmethod
    async def connect(
        cls,
        redis_url: str,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
    ) -> RedisCache:
        """Create a RedisCache connected to *redis_url*."""
        client = await aioredis.from_url(redis_url, decode_responses=True)
        return cls(client, max_size=max_size, ttl_seconds=ttl_seconds)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, model: str, prompt: str) -> str | None:
        """Look up a cached response.

        Args:
            model: Model identifier.
            prompt: Prompt string (normalised internally).

        Returns:
            Cached response text, or ``None`` on a miss.
        """
        key = _cache_key(model, prompt)
        value: str | None = await self._redis.get(key)
        if value is None:
            self._misses += 1
            return None

        # Refresh LRU score on hit (sorted set score = current timestamp)
        await self._redis.zadd(_CACHE_LRU_ZSET, {key: time.time()})
        self._hits += 1
        logger.debug("cache_hit", model=model, total_hits=self._hits)
        return value

    async def put(self, model: str, prompt: str, response_text: str) -> None:
        """Store a response in the cache, evicting LRU entries if at capacity.

        Args:
            model: Model identifier.
            prompt: Prompt string (normalised internally).
            response_text: Generated text to cache.
        """
        key = _cache_key(model, prompt)
        score = time.time()

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(key, response_text, ex=self._ttl)
            pipe.zadd(_CACHE_LRU_ZSET, {key: score})
            await pipe.execute()

        await self._evict_if_needed()
        logger.debug("cache_put", model=model)

    async def invalidate(self, model: str, prompt: str) -> bool:
        """Remove a specific entry.

        Returns:
            ``True`` if an entry was removed, ``False`` otherwise.
        """
        key = _cache_key(model, prompt)
        removed = await self._redis.delete(key)
        await self._redis.zrem(_CACHE_LRU_ZSET, key)
        return bool(removed)

    async def clear(self) -> None:
        """Remove all cache entries tracked in the LRU set."""
        keys: list[str] = await self._redis.zrange(_CACHE_LRU_ZSET, 0, -1)
        if keys:
            await self._redis.delete(*keys)
        await self._redis.delete(_CACHE_LRU_ZSET)
        logger.info("cache_cleared")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total cache hits since creation (in-process counter)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation (in-process counter)."""
        return self._misses

    @property
    async def size(self) -> int:
        """Current number of entries tracked in the LRU sorted set."""
        return await self._redis.zcard(_CACHE_LRU_ZSET)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries when the cache exceeds max_size."""
        count: int = await self._redis.zcard(_CACHE_LRU_ZSET)
        if count <= self._max_size:
            return
        overflow = count - self._max_size
        # Oldest entries have the lowest scores
        lru_keys: list[str] = await self._redis.zrange(
            _CACHE_LRU_ZSET, 0, overflow - 1
        )
        if lru_keys:
            await self._redis.delete(*lru_keys)
            await self._redis.zrem(_CACHE_LRU_ZSET, *lru_keys)
            logger.debug("cache_lru_eviction", evicted=len(lru_keys))


# Backward-compatible alias used in server.py
ExactMatchCache = RedisCache
SemanticCache = RedisCache

__all__ = ["RedisCache", "ExactMatchCache", "SemanticCache"]

