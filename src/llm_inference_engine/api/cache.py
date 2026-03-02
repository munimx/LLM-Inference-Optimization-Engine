"""Exact-match response cache for inference requests.

Provides an exact-match cache with TTL expiry and LRU eviction.  Keys are
``(model, prompt)`` tuples — only character-identical prompts hit the cache.
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry."""

    response_text: str
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0


class ExactMatchCache:
    """Exact-match response cache with TTL and LRU eviction.

    The cache key is a ``(model, prompt)`` tuple — only character-identical
    prompts produce a hit.  Entries are evicted when they exceed *ttl_seconds*
    or when the cache reaches *max_size* (LRU eviction).

    All public methods are coroutine-safe: an internal :class:`asyncio.Lock`
    serialises concurrent ``get``/``put`` operations to prevent race
    conditions on the underlying :class:`~collections.OrderedDict`.

    Usage::

        cache = ExactMatchCache(max_size=512, ttl_seconds=300)
        hit = await cache.get("llama3.1:8b", "Hello!")
        if hit is None:
            result = await ollama_generate(...)
            await cache.put("llama3.1:8b", "Hello!", result)
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        """Initialise the cache.

        Args:
            max_size: Maximum number of entries (LRU eviction when full).
            ttl_seconds: Seconds before an entry is considered stale.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
        logger.info("exact_match_cache_initialized", max_size=max_size, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, model: str, prompt: str) -> str | None:
        """Look up a cached response.

        Args:
            model: Model tag.
            prompt: Exact prompt string (normalised internally).

        Returns:
            Cached response text, or ``None`` on a miss.
        """
        async with self._lock:
            key = (model, prompt.lower().strip())
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL expiry.
            if time.monotonic() - entry.created_at > self._ttl:
                del self._store[key]
                self._misses += 1
                logger.debug("cache_entry_expired", model=model)
                return None

            # Move to end (most-recently used).
            self._store.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            logger.debug("cache_hit", model=model, total_hits=self._hits)
            return entry.response_text

    async def put(self, model: str, prompt: str, response_text: str) -> None:
        """Store a response in the cache.

        If the cache is full the least-recently-used entry is evicted.

        Args:
            model: Model tag.
            prompt: Exact prompt string (normalised internally).
            response_text: Generated text to cache.
        """
        async with self._lock:
            key = (model, prompt.lower().strip())
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key].response_text = response_text
                self._store[key].created_at = time.monotonic()
                return

            if len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("cache_lru_eviction", evicted_model=evicted_key[0])

            self._store[key] = _CacheEntry(response_text=response_text)
            logger.debug("cache_put", model=model, cache_size=len(self._store))

    async def invalidate(self, model: str, prompt: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            model: Model tag.
            prompt: Exact prompt string (normalised internally).

        Returns:
            ``True`` if an entry was removed, ``False`` if not found.
        """
        async with self._lock:
            key = (model, prompt.lower().strip())
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        async with self._lock:
            self._store.clear()
        logger.info("cache_cleared")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total cache hits since creation."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation."""
        return self._misses

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)

    @property
    def max_size(self) -> int:
        """Maximum cache capacity."""
        return self._max_size


# Backward-compatible alias
SemanticCache = ExactMatchCache

__all__ = ["ExactMatchCache", "SemanticCache"]
