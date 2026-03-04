"""FastAPI dependency injection providers for the inference engine."""

from typing import Annotated

import structlog
from fastapi import Depends, Request

from llm_inference_engine.api.aggregator import RequestAggregator
from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.optimization.memory import MemoryEstimator
from llm_inference_engine.optimization.throttler import AdaptiveThrottler
from llm_inference_engine.scheduling.scheduler import Scheduler

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency accessors – pull components from app.state
# ---------------------------------------------------------------------------


def get_ollama_client(request: Request) -> OllamaClient:
    """Return the shared :class:`~llm_inference_engine.integration.\
ollama_client.OllamaClient` from app state."""
    client: OllamaClient | None = getattr(request.app.state, "ollama_client", None)
    if client is None:
        raise RuntimeError("OllamaClient not initialized — check server startup logs")
    return client


def get_scheduler(request: Request) -> Scheduler:
    """Return the shared :class:`~llm_inference_engine.scheduling.\
scheduler.Scheduler` from app state."""
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Scheduler not initialized — check server startup logs")
    return scheduler


def get_cache(request: Request) -> SemanticCache:
    """Return the shared :class:`~llm_inference_engine.api.cache.\
SemanticCache` from app state."""
    cache: SemanticCache | None = getattr(request.app.state, "cache", None)
    if cache is None:
        raise RuntimeError("Cache not initialized — check server startup logs")
    return cache


def get_aggregator(request: Request) -> RequestAggregator:
    """Return the shared :class:`~llm_inference_engine.api.aggregator.\
RequestAggregator` from app state."""
    aggregator: RequestAggregator | None = getattr(request.app.state, "aggregator", None)
    if aggregator is None:
        raise RuntimeError("Aggregator not initialized — check server startup logs")
    return aggregator


def get_throttler(request: Request) -> AdaptiveThrottler:
    """Return the shared :class:`~llm_inference_engine.optimization.\
throttler.AdaptiveThrottler` from app state."""
    throttler: AdaptiveThrottler | None = getattr(request.app.state, "throttler", None)
    if throttler is None:
        raise RuntimeError("Throttler not initialized — check server startup logs")
    return throttler


def get_memory_estimator(request: Request) -> MemoryEstimator:
    """Return the shared :class:`~llm_inference_engine.optimization.\
memory.MemoryEstimator` from app state."""
    estimator: MemoryEstimator | None = getattr(request.app.state, "memory_estimator", None)
    if estimator is None:
        raise RuntimeError("MemoryEstimator not initialized — check server startup logs")
    return estimator


# ---------------------------------------------------------------------------
# Type aliases for route handler injection
# ---------------------------------------------------------------------------

OllamaClientDep = Annotated[OllamaClient, Depends(get_ollama_client)]
SchedulerDep = Annotated[Scheduler, Depends(get_scheduler)]
CacheDep = Annotated[SemanticCache, Depends(get_cache)]
AggregatorDep = Annotated[RequestAggregator, Depends(get_aggregator)]
ThrottlerDep = Annotated[AdaptiveThrottler, Depends(get_throttler)]
MemoryEstimatorDep = Annotated[MemoryEstimator, Depends(get_memory_estimator)]


__all__ = [
    "get_ollama_client",
    "get_scheduler",
    "get_cache",
    "get_aggregator",
    "get_throttler",
    "get_memory_estimator",
    "OllamaClientDep",
    "SchedulerDep",
    "CacheDep",
    "AggregatorDep",
    "ThrottlerDep",
    "MemoryEstimatorDep",
]
