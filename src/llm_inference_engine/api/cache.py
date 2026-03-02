"""Semantic cache for inference requests.

Provides an exact-match cache with TTL expiry and LRU eviction.  For the
current implementation *semantic* similarity is approximated by exact
prompt/model equality; a vector-similarity extension can be layered on in
Phase 7 or beyond.
"""

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


class SemanticCache:
    """Exact-match response cache with TTL and LRU eviction.

    The cache key is a ``(model, prompt)`` tuple.  Entries are evicted
    when they exceed *ttl_seconds* or when the cache reaches *max_size*
    (LRU eviction).

    Thread-safety: This class is **not** thread-safe by itself; callers
    should use an :class:`asyncio.Lock` if concurrent access is required.

    Usage::

        cache = SemanticCache(max_size=512, ttl_seconds=300)
        hit = cache.get("llama3.1:8b", "Hello!")
        if hit is None:
            result = await ollama_generate(...)
            cache.put("llama3.1:8b", "Hello!", result)
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
        logger.info("semantic_cache_initialized", max_size=max_size, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, model: str, prompt: str) -> str | None:
        """Look up a cached response.

        Args:
            model: Model tag.
            prompt: Exact prompt string.

        Returns:
            Cached response text, or ``None`` on a miss.
        """
        key = (model, prompt)
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

    def put(self, model: str, prompt: str, response_text: str) -> None:
        """Store a response in the cache.

        If the cache is full the least-recently-used entry is evicted.

        Args:
            model: Model tag.
            prompt: Exact prompt string.
            response_text: Generated text to cache.
        """
        key = (model, prompt)
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

    def invalidate(self, model: str, prompt: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            model: Model tag.
            prompt: Exact prompt string.

        Returns:
            ``True`` if an entry was removed, ``False`` if not found.
        """
        key = (model, prompt)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
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


__all__ = ["SemanticCache"]
