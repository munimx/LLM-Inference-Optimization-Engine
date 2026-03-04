"""FastAPI dependency injection providers for the inference engine."""

from typing import Annotated

import structlog
from fastapi import Depends, Request

from llm_inference_engine.api.cache import RedisCache
from llm_inference_engine.api.coalescer import RequestCoalescer
from llm_inference_engine.api.fallback_router import FallbackRouter
from llm_inference_engine.api.model_router import ModelRouter
from llm_inference_engine.integration.backend_pool import BackendPool
from llm_inference_engine.optimization.throttler import AdaptiveThrottler

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency accessors – pull components from app.state
# ---------------------------------------------------------------------------


def get_pool(request: Request) -> BackendPool:
    """Return the shared :class:`~llm_inference_engine.integration.backend_pool.BackendPool`."""
    pool: BackendPool | None = getattr(request.app.state, "pool", None)
    if pool is None:
        raise RuntimeError("BackendPool not initialized — check server startup logs")
    return pool


def get_cache(request: Request) -> RedisCache:
    """Return the shared :class:`~llm_inference_engine.api.cache.RedisCache`."""
    cache: RedisCache | None = getattr(request.app.state, "cache", None)
    if cache is None:
        raise RuntimeError("RedisCache not initialized — check server startup logs")
    return cache


def get_throttler(request: Request) -> AdaptiveThrottler:
    """Return the shared :class:`~llm_inference_engine.optimization.throttler.AdaptiveThrottler`."""
    throttler: AdaptiveThrottler | None = getattr(request.app.state, "throttler", None)
    if throttler is None:
        raise RuntimeError("AdaptiveThrottler not initialized — check server startup logs")
    return throttler


def get_coalescer(request: Request) -> RequestCoalescer:
    """Return the shared :class:`~llm_inference_engine.api.coalescer.RequestCoalescer`."""
    coalescer: RequestCoalescer | None = getattr(request.app.state, "coalescer", None)
    if coalescer is None:
        raise RuntimeError("RequestCoalescer not initialized — check server startup logs")
    return coalescer


def get_model_router(request: Request) -> ModelRouter:
    """Return the shared :class:`~llm_inference_engine.api.model_router.ModelRouter`."""
    router: ModelRouter | None = getattr(request.app.state, "model_router", None)
    if router is None:
        raise RuntimeError("ModelRouter not initialized — check server startup logs")
    return router


def get_fallback_router(request: Request) -> FallbackRouter:
    """Return the shared :class:`~llm_inference_engine.api.fallback_router.FallbackRouter`."""
    router: FallbackRouter | None = getattr(request.app.state, "fallback_router", None)
    if router is None:
        raise RuntimeError("FallbackRouter not initialized — check server startup logs")
    return router


# ---------------------------------------------------------------------------
# Type aliases for route handler injection
# ---------------------------------------------------------------------------

PoolDep = Annotated[BackendPool, Depends(get_pool)]
CacheDep = Annotated[RedisCache, Depends(get_cache)]
ThrottlerDep = Annotated[AdaptiveThrottler, Depends(get_throttler)]
CoalescerDep = Annotated[RequestCoalescer, Depends(get_coalescer)]
ModelRouterDep = Annotated[ModelRouter, Depends(get_model_router)]
FallbackRouterDep = Annotated[FallbackRouter, Depends(get_fallback_router)]


__all__ = [
    "get_pool",
    "get_cache",
    "get_throttler",
    "get_coalescer",
    "get_model_router",
    "get_fallback_router",
    "PoolDep",
    "CacheDep",
    "ThrottlerDep",
    "CoalescerDep",
    "ModelRouterDep",
    "FallbackRouterDep",
]
