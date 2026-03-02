"""FastAPI application for the LLM Inference Optimization Engine.

Exposes the following endpoints:

- ``GET  /health``               — Liveness/readiness check
- ``GET  /metrics``              — JSON metrics snapshot
- ``POST /completions``          — OpenAI-compatible text completion
- ``POST /chat/completions``     — OpenAI-compatible chat completion
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from llm_inference_engine.api.aggregator import RequestAggregator, dispatch_batch
from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.api.dependencies import AggregatorDep, ThrottlerDep
from llm_inference_engine.api.models import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    UsageInfo,
)
from llm_inference_engine.config import InferenceConfig
from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.optimization.memory import MemoryEstimator
from llm_inference_engine.optimization.throttler import AdaptiveThrottler
from llm_inference_engine.scheduling.policies import SchedulingPolicy
from llm_inference_engine.scheduling.scheduler import Scheduler

logger = structlog.get_logger(__name__)

VERSION = "0.1.0"


def create_app(config: InferenceConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional :class:`~llm_inference_engine.config.InferenceConfig`.
            If ``None``, a default configuration is used.

    Returns:
        A configured :class:`fastapi.FastAPI` application instance.
    """
    if config is None:
        config = InferenceConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage application startup and shutdown."""
        logger.info("server_starting", version=VERSION)

        # --- Initialise shared components ---
        ollama_client = OllamaClient(
            host=config.ollama.host,
            port=config.ollama.port,
            timeout=config.ollama.timeout_seconds,
            max_retries=config.ollama.retry_count,
            retry_backoff_seconds=config.ollama.retry_backoff_seconds,
        )
        await ollama_client.connect()

        cache = SemanticCache(
            max_size=config.cache.max_size,
            ttl_seconds=config.cache.ttl_seconds,
        ) if config.cache.enabled else None
        memory_estimator = MemoryEstimator(safety_margin=config.memory.safety_margin)
        throttler = AdaptiveThrottler(memory_limit_gb=config.memory.limit_gb)

        scheduler = Scheduler(
            dispatch_fn=lambda batch: dispatch_batch(ollama_client, batch),
            policy=SchedulingPolicy(config.scheduling.policy),
            max_requests_per_batch=config.scheduling.max_requests_per_batch,
            max_tokens_per_batch=config.scheduling.max_tokens_per_batch,
        )

        aggregator = RequestAggregator(
            ollama_client=ollama_client,
            scheduler=scheduler,
            cache=cache,
        )

        # --- Store in app state for DI ---
        app.state.ollama_client = ollama_client
        app.state.scheduler = scheduler
        app.state.cache = cache
        app.state.aggregator = aggregator
        app.state.throttler = throttler
        app.state.memory_estimator = memory_estimator
        app.state.config = config

        logger.info("server_ready")
        yield

        # --- Cleanup ---
        await ollama_client.close()
        logger.info("server_stopped")

    app = FastAPI(
        title="LLM Inference Optimization Engine",
        description=(
            "Production-grade inference orchestration layer on top of Ollama, "
            "optimising throughput and latency on Apple Silicon."
        ),
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health & Metrics
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health(aggregator: AggregatorDep) -> HealthResponse:
        """Return service health status."""
        try:
            ollama_client: OllamaClient = app.state.ollama_client
            available = await ollama_client.is_available()
        except Exception:
            available = False

        return HealthResponse(
            status="ok" if available else "degraded",
            ollama_available=available,
            version=VERSION,
            details={"pending_requests": aggregator.pending_count},
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def metrics(
        aggregator: AggregatorDep,
        throttler: ThrottlerDep,
    ) -> MetricsResponse:
        """Return a JSON snapshot of runtime metrics."""
        stats = throttler.stats
        cache: SemanticCache = app.state.cache
        return MetricsResponse(
            committed_memory_gb=stats.committed_gb,
            available_memory_gb=stats.available_gb,
            memory_limit_gb=stats.memory_limit_gb,
            active_requests=stats.active_requests,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            total_requests=aggregator.total_requests,
        )

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    @app.post(
        "/completions",
        response_model=CompletionResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["Inference"],
    )
    async def completions(
        body: CompletionRequest,
        aggregator: AggregatorDep,
    ) -> CompletionResponse:
        """Generate a text completion for the given prompt."""
        response = await aggregator.complete(
            model=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
            priority=body.priority,
        )
        if response.result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        result = response.result
        return CompletionResponse(
            id=response.request_id,
            model=body.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=result.text,
                    finish_reason=result.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=result.tokens_used,
                total_tokens=result.tokens_used,
            ),
            latency_ms=result.latency_ms,
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    @app.post(
        "/chat/completions",
        response_model=ChatCompletionResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["Inference"],
    )
    async def chat_completions(
        body: ChatCompletionRequest,
        aggregator: AggregatorDep,
    ) -> ChatCompletionResponse:
        """Generate a chat completion from a list of messages."""
        # Flatten messages into a single prompt string (Ollama /generate style).
        prompt = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in body.messages
        )

        response = await aggregator.complete(
            model=body.model,
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
            priority=body.priority,
        )
        if response.result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        result = response.result
        return ChatCompletionResponse(
            id=response.request_id,
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=result.text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=result.tokens_used,
                total_tokens=result.tokens_used,
            ),
            latency_ms=result.latency_ms,
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()

__all__ = ["create_app", "app"]
