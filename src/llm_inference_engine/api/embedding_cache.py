"""Embedding-based semantic response cache.

Uses Ollama's ``/api/embed`` endpoint to compute embeddings for prompts and
finds cache hits via cosine similarity, enabling near-miss matching for
rephrased or slightly different prompts.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class _EmbeddingCacheEntry:
    """Internal entry storing embedding + response."""

    model: str
    prompt: str
    embedding: list[float]
    response_text: str
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0


class EmbeddingCache:
    """Semantic response cache using cosine similarity on prompt embeddings.

    Embeds each incoming prompt using an Ollama embedding model and searches
    existing cache entries for a match above ``similarity_threshold``.  Falls
    back to exact-match when embedding fails.

    Args:
        embed_fn: Async callable ``(text) -> list[float]`` that returns an
            embedding vector.  Typically bound to
            ``ollama_client.embed(model, text)``.
        max_size: Maximum number of cached entries (LRU eviction).
        ttl_seconds: Seconds before an entry is considered stale.
        similarity_threshold: Minimum cosine similarity for a semantic hit
            (0.0–1.0).  Higher values require closer matches.
    """

    def __init__(
        self,
        embed_fn,  # noqa: ANN001  — Callable[[str], Awaitable[list[float]]]
        *,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
        similarity_threshold: float = 0.92,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self._embed_fn = embed_fn
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._entries: list[_EmbeddingCacheEntry] = []
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
        logger.info(
            "embedding_cache_initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            similarity_threshold=similarity_threshold,
        )

    async def get(self, model: str, prompt: str) -> str | None:
        """Look up a semantically similar cached response.

        Returns:
            Cached response text if a similar prompt is found, else ``None``.
        """
        try:
            embedding = await self._embed_fn(prompt)
        except Exception:
            logger.debug("embedding_failed_for_get", model=model)
            self._misses += 1
            return None

        async with self._lock:
            now = time.monotonic()
            best_score = -1.0
            best_entry: _EmbeddingCacheEntry | None = None

            for entry in self._entries:
                if entry.model != model:
                    continue
                if now - entry.created_at > self._ttl:
                    continue
                score = _cosine_similarity(embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= self._threshold:
                best_entry.hits += 1
                self._hits += 1
                logger.debug(
                    "semantic_cache_hit",
                    model=model,
                    similarity=round(best_score, 4),
                )
                return best_entry.response_text

            self._misses += 1
            return None

    async def put(self, model: str, prompt: str, response_text: str) -> None:
        """Store a response with its prompt embedding.

        Silently skips caching if the embedding call fails.
        """
        try:
            embedding = await self._embed_fn(prompt)
        except Exception:
            logger.debug("embedding_failed_for_put", model=model)
            return

        async with self._lock:
            # Evict stale entries
            now = time.monotonic()
            self._entries = [e for e in self._entries if now - e.created_at <= self._ttl]

            # LRU eviction
            while len(self._entries) >= self._max_size:
                self._entries.pop(0)

            self._entries.append(
                _EmbeddingCacheEntry(
                    model=model,
                    prompt=prompt,
                    embedding=embedding,
                    response_text=response_text,
                )
            )
            logger.debug("semantic_cache_put", model=model, cache_size=len(self._entries))

    async def invalidate(self, model: str, prompt: str) -> bool:
        """Remove a specific entry (exact prompt match)."""
        async with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries if not (e.model == model and e.prompt == prompt)
            ]
            return len(self._entries) < before

    async def clear(self) -> None:
        """Remove all entries."""
        async with self._lock:
            self._entries.clear()
        logger.info("semantic_cache_cleared")

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def max_size(self) -> int:
        return self._max_size


__all__ = ["EmbeddingCache"]
