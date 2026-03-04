"""Fallback routing when the primary backend pool is unavailable.

When all backends in the pool have open circuit breakers, the fallback router
tries the following in order:

1. Route to the configured fallback model on any available backend.
2. Return a cached response if one exists (with the given key).
3. Raise a 503 ``HTTPException`` as last resort.

Usage::

    router = FallbackRouter(
        pool=backend_pool,
        cache=redis_cache,
        fallback_model="mistralai/Mistral-7B-Instruct-v0.2",
    )
    result = await router.route(model, prompt, generate_kwargs)
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import HTTPException, status

from llm_inference_engine.integration.backend import BackendResult
from llm_inference_engine.integration.backend_pool import BackendPool

logger = structlog.get_logger(__name__)


class FallbackRouter:
    """Attempts to serve a request when the primary pool is fully down.

    Args:
        pool: The :class:`~llm_inference_engine.integration.backend_pool.BackendPool`
            to use for fallback routing (same pool, different model).
        fallback_model: Model name to use when the primary model's backend is down.
        cache: Optional :class:`~llm_inference_engine.api.cache.RedisCache`
            instance for cache-based fallback.
    """

    def __init__(
        self,
        pool: BackendPool,
        fallback_model: str,
        cache: Any = None,
    ) -> None:
        self._pool = pool
        self._fallback_model = fallback_model
        self._cache = cache

    async def route(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """Attempt fallback strategies in priority order.

        Returns:
            A :class:`~llm_inference_engine.integration.backend.BackendResult`.

        Raises:
            :exc:`fastapi.HTTPException` with status 503 if all fallbacks fail.
        """
        # Strategy 1: route to fallback model on any healthy backend
        if self._fallback_model != model:
            backend = self._pool.get_healthy_backend()
            if backend is not None:
                try:
                    result = await backend.generate(
                        self._fallback_model,
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                    )
                    self._pool.record_success(backend)
                    logger.info(
                        "fallback_model_used",
                        original_model=model,
                        fallback_model=self._fallback_model,
                    )
                    return result
                except Exception as exc:
                    self._pool.record_failure(backend)
                    logger.warning("fallback_model_failed", error=str(exc))

        # Strategy 2: return a stale cached response if available
        if self._cache is not None:
            cached = await self._cache.get(model, prompt)
            if cached is not None:
                logger.info("fallback_cache_hit", model=model)
                return BackendResult(
                    text=cached,
                    metadata={"cache_hit": True, "stale_fallback": True},
                )

        # Strategy 3: 503
        logger.error("fallback_all_strategies_exhausted", model=model)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="All backends unavailable and no fallback could serve the request.",
        )

    async def route_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> BackendResult:
        """Chat-completion variant of :meth:`route`."""
        # Strategy 1: fallback model on any healthy backend
        if self._fallback_model != model:
            backend = self._pool.get_healthy_backend()
            if backend is not None:
                try:
                    result = await backend.chat(
                        self._fallback_model,
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                    )
                    self._pool.record_success(backend)
                    logger.info(
                        "fallback_model_used",
                        original_model=model,
                        fallback_model=self._fallback_model,
                    )
                    return result
                except Exception as exc:
                    self._pool.record_failure(backend)
                    logger.warning("fallback_chat_model_failed", error=str(exc))

        # Strategy 2: 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="All backends unavailable.",
        )


__all__ = ["FallbackRouter"]
