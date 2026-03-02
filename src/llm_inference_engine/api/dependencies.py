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
    client: OllamaClient = request.app.state.ollama_client
    return client


def get_scheduler(request: Request) -> Scheduler:
    """Return the shared :class:`~llm_inference_engine.scheduling.\
scheduler.Scheduler` from app state."""
    scheduler: Scheduler = request.app.state.scheduler
    return scheduler


def get_cache(request: Request) -> SemanticCache:
    """Return the shared :class:`~llm_inference_engine.api.cache.\
SemanticCache` from app state."""
    cache: SemanticCache = request.app.state.cache
    return cache


def get_aggregator(request: Request) -> RequestAggregator:
    """Return the shared :class:`~llm_inference_engine.api.aggregator.\
RequestAggregator` from app state."""
    aggregator: RequestAggregator = request.app.state.aggregator
    return aggregator


def get_throttler(request: Request) -> AdaptiveThrottler:
    """Return the shared :class:`~llm_inference_engine.optimization.\
throttler.AdaptiveThrottler` from app state."""
    throttler: AdaptiveThrottler = request.app.state.throttler
    return throttler


def get_memory_estimator(request: Request) -> MemoryEstimator:
    """Return the shared :class:`~llm_inference_engine.optimization.\
memory.MemoryEstimator` from app state."""
    estimator: MemoryEstimator = request.app.state.memory_estimator
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
