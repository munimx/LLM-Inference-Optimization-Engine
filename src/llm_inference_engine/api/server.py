"""FastAPI application for the LLM Inference Optimization Engine.

Exposes the following endpoints:

- ``GET  /health``               — Liveness/readiness check
- ``GET  /metrics``              — JSON metrics snapshot
- ``POST /completions``          — OpenAI-compatible text completion (+ SSE streaming)
- ``POST /chat/completions``     — OpenAI-compatible chat completion (+ SSE streaming)
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llm_inference_engine.api.aggregator import RequestAggregator, dispatch_batch
from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.api.dependencies import AggregatorDep, OllamaClientDep, ThrottlerDep
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
from llm_inference_engine.metrics.prometheus import (
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    CACHE_SIZE,
    COMMITTED_MEMORY_GB,
    PROMPT_TOKENS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TOKENS_GENERATED,
)
from llm_inference_engine.optimization.memory import MemoryEstimator
from llm_inference_engine.optimization.throttler import AdaptiveThrottler
from llm_inference_engine.scheduling.policies import SchedulingPolicy
from llm_inference_engine.scheduling.scheduler import Scheduler
from llm_inference_engine.utils.tokenizer import estimate_prompt_tokens

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
        allow_origins=config.auth.enabled and ["*"] or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API-key authentication middleware ---
    if config.auth.enabled:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse

        _public_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics/prometheus"}
        _valid_keys = frozenset(config.auth.api_keys)

        class _APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: StarletteRequest, call_next):  # type: ignore[override]
                if request.url.path in _public_paths:
                    return await call_next(request)
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    if token in _valid_keys:
                        return await call_next(request)
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )

        app.add_middleware(_APIKeyMiddleware)

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

    @app.get("/metrics/prometheus", tags=["System"])
    async def prometheus_metrics() -> StreamingResponse:
        """Return Prometheus-format metrics for scraping."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        # Update gauges from current state
        try:
            throttler_obj: AdaptiveThrottler = app.state.throttler
            stats = throttler_obj.stats
            COMMITTED_MEMORY_GB.set(stats.committed_gb)
            ACTIVE_REQUESTS.set(stats.active_requests)
        except Exception:
            pass
        try:
            cache_obj = app.state.cache
            if cache_obj is not None:
                CACHE_SIZE.set(cache_obj.size)
        except Exception:
            pass

        return StreamingResponse(
            iter([generate_latest()]),
            media_type=CONTENT_TYPE_LATEST,
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
        ollama_client: OllamaClientDep,
    ) -> CompletionResponse | StreamingResponse:
        """Generate a text completion for the given prompt."""
        if body.stream:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="stream").inc()
            return StreamingResponse(
                _stream_completion(ollama_client, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

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
            REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        result = response.result
        prompt_tokens = int(
            result.metadata.get("prompt_eval_count") or 0
        ) if result.metadata else 0
        if prompt_tokens == 0:
            prompt_tokens = estimate_prompt_tokens(body.prompt)
        resp = CompletionResponse(
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
                prompt_tokens=prompt_tokens,
                completion_tokens=result.tokens_used,
                total_tokens=prompt_tokens + result.tokens_used,
            ),
            latency_ms=result.latency_ms,
        )

        # Instrument Prometheus counters
        REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="ok").inc()
        REQUEST_LATENCY.labels(model=body.model, endpoint="completions").observe(
            result.latency_ms / 1000.0
        )
        TOKENS_GENERATED.labels(model=body.model).inc(result.tokens_used)
        PROMPT_TOKENS.labels(model=body.model).inc(prompt_tokens)
        if result.metadata and result.metadata.get("cache_hit"):
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

        return resp

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
        ollama_client: OllamaClientDep,
    ) -> ChatCompletionResponse | StreamingResponse:
        """Generate a chat completion from a list of messages."""
        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]

        if body.stream:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="stream").inc()
            return StreamingResponse(
                _stream_chat_completion(ollama_client, body, messages),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        response = await aggregator.chat_complete(
            model=body.model,
            messages=messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
            priority=body.priority,
        )
        if response.result is None:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        result = response.result
        prompt_tokens = int(
            result.metadata.get("prompt_eval_count") or 0
        ) if result.metadata else 0
        if prompt_tokens == 0:
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = estimate_prompt_tokens(prompt_text)
        resp = ChatCompletionResponse(
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
                prompt_tokens=prompt_tokens,
                completion_tokens=result.tokens_used,
                total_tokens=prompt_tokens + result.tokens_used,
            ),
            latency_ms=result.latency_ms,
        )

        # Instrument Prometheus counters
        REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="ok").inc()
        REQUEST_LATENCY.labels(model=body.model, endpoint="chat").observe(
            result.latency_ms / 1000.0
        )
        TOKENS_GENERATED.labels(model=body.model).inc(result.tokens_used)
        PROMPT_TOKENS.labels(model=body.model).inc(prompt_tokens)

        return resp

    return app


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


async def _stream_completion(
    client: OllamaClient, body: CompletionRequest
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming text completion."""
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    async for chunk in client.generate_stream(
        model=body.model,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop_sequences=body.stop or None,
    ):
        token = chunk.get("response", "")
        done = chunk.get("done", False)
        finish = "stop" if done else None
        event = {
            "id": request_id,
            "object": "text_completion",
            "model": body.model,
            "choices": [{"index": 0, "text": token, "finish_reason": finish}],
        }
        if done:
            event["usage"] = {
                "prompt_tokens": int(chunk.get("prompt_eval_count", 0)),
                "completion_tokens": int(chunk.get("eval_count", 0)),
                "total_tokens": int(chunk.get("prompt_eval_count", 0))
                + int(chunk.get("eval_count", 0)),
            }
            event["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        yield f"data: {json.dumps(event)}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_chat_completion(
    client: OllamaClient,
    body: ChatCompletionRequest,
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming chat completion."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    async for chunk in client.chat_stream(
        model=body.model,
        messages=messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop_sequences=body.stop or None,
    ):
        msg = chunk.get("message", {})
        token = msg.get("content", "")
        done = chunk.get("done", False)
        finish = "stop" if done else None
        event = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": finish,
                }
            ],
        }
        if done:
            event["usage"] = {
                "prompt_tokens": int(chunk.get("prompt_eval_count", 0)),
                "completion_tokens": int(chunk.get("eval_count", 0)),
                "total_tokens": int(chunk.get("prompt_eval_count", 0))
                + int(chunk.get("eval_count", 0)),
            }
            event["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        yield f"data: {json.dumps(event)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()

__all__ = ["create_app", "app"]
