"""FastAPI application for the LLM Inference Optimization Engine.

Exposes the following endpoints:

- ``GET  /health``               — Liveness/readiness check
- ``GET  /metrics``              — JSON metrics snapshot
- ``POST /completions``          — OpenAI-compatible text completion (+ SSE streaming)
- ``POST /chat/completions``     — OpenAI-compatible chat completion (+ SSE streaming)
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llm_inference_engine.api.aggregator import RequestAggregator, dispatch_batch
from llm_inference_engine.api.cache import SemanticCache
from llm_inference_engine.api.circuit_breaker import CircuitBreaker
from llm_inference_engine.api.coalescer import RequestCoalescer
from llm_inference_engine.api.dependencies import AggregatorDep, OllamaClientDep, ThrottlerDep
from llm_inference_engine.api.embedding_cache import EmbeddingCache
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

        # Build the cache based on configured mode.
        cache: Any = None
        if config.cache.enabled:
            if config.cache.mode == "semantic":
                embed_model = config.cache.embedding_model

                async def _embed(text: str) -> list[float]:
                    return await ollama_client.embed(embed_model, text)

                cache = EmbeddingCache(
                    _embed,
                    max_size=config.cache.max_size,
                    ttl_seconds=config.cache.ttl_seconds,
                    similarity_threshold=config.cache.similarity_threshold,
                )
            else:
                cache = SemanticCache(
                    max_size=config.cache.max_size,
                    ttl_seconds=config.cache.ttl_seconds,
                )

        memory_estimator = MemoryEstimator(safety_margin=config.memory.safety_margin)
        throttler = AdaptiveThrottler(memory_limit_gb=config.memory.limit_gb)

        scheduler = Scheduler(
            dispatch_fn=lambda batch: dispatch_batch(ollama_client, batch),
            policy=SchedulingPolicy(config.scheduling.policy),
            max_requests_per_batch=config.scheduling.max_requests_per_batch,
            max_tokens_per_batch=config.scheduling.max_tokens_per_batch,
        )

        coalescer = RequestCoalescer()

        circuit_breaker = CircuitBreaker(
            failure_threshold=config.scheduling.circuit_breaker_threshold,
            cooldown_seconds=config.scheduling.circuit_breaker_cooldown_seconds,
        )

        aggregator = RequestAggregator(
            ollama_client=ollama_client,
            scheduler=scheduler,
            cache=cache,
            drain_delay_seconds=config.scheduling.drain_delay_seconds,
            coalescer=coalescer,
        )

        # --- Store in app state for DI ---
        app.state.ollama_client = ollama_client
        app.state.scheduler = scheduler
        app.state.cache = cache
        app.state.aggregator = aggregator
        app.state.throttler = throttler
        app.state.memory_estimator = memory_estimator
        app.state.config = config
        app.state.circuit_breaker = circuit_breaker

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
        from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse
        from starlette.responses import Response as StarletteResponse

        _public_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics/prometheus"}
        _valid_keys = frozenset(config.auth.api_keys)

        class _APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: StarletteRequest, call_next: RequestResponseEndpoint) -> StarletteResponse:
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

        cb: CircuitBreaker = app.state.circuit_breaker
        cb_state = cb.state.value
        if not available or not cb.is_available:
            health_status = "degraded"
        else:
            health_status = "ok"

        return HealthResponse(
            status=health_status,
            ollama_available=available,
            version=VERSION,
            details={
                "pending_requests": aggregator.pending_count,
                "circuit_breaker": cb_state,
            },
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
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

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
    # Shared admission guards
    # ------------------------------------------------------------------

    def _check_admission(pending: int) -> None:
        """Reject early if circuit is open or queue is full."""
        cb: CircuitBreaker = app.state.circuit_breaker
        if not cb.is_available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Backend unavailable (circuit breaker open)",
            )
        max_q = config.scheduling.max_queue_depth
        if max_q > 0 and pending >= max_q:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full ({pending}/{max_q}). Retry later.",
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
        _check_admission(aggregator.pending_count)

        if body.stream:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="stream").inc()
            cache_for_stream = app.state.cache
            return StreamingResponse(
                _stream_completion(ollama_client, body, cache=cache_for_stream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        timeout = body.timeout_seconds or 300.0
        try:
            response = await asyncio.wait_for(
                aggregator.complete(
                    model=body.model,
                    prompt=body.prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    stop=body.stop,
                    priority=body.priority,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timed out after {timeout}s",
            ) from None

        cb: CircuitBreaker = app.state.circuit_breaker
        if response.result is None:
            cb.record_failure()
            REQUESTS_TOTAL.labels(model=body.model, endpoint="completions", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        cb.record_success()
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
        _check_admission(aggregator.pending_count)
        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]

        if body.stream:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="stream").inc()
            cache_for_stream = app.state.cache
            return StreamingResponse(
                _stream_chat_completion(ollama_client, body, messages, cache=cache_for_stream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        timeout = body.timeout_seconds or 300.0
        try:
            response = await asyncio.wait_for(
                aggregator.chat_complete(
                    model=body.model,
                    messages=messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    stop=body.stop,
                    priority=body.priority,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timed out after {timeout}s",
            ) from None

        cb: CircuitBreaker = app.state.circuit_breaker
        if response.result is None:
            cb.record_failure()
            REQUESTS_TOTAL.labels(model=body.model, endpoint="chat", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.error or "Inference failed",
            )
        cb.record_success()
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
    client: OllamaClient, body: CompletionRequest, *, cache: Any = None,
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming text completion.

    If *cache* is provided, checks for a cached response first (streams it
    as synthetic events) and stores the completed stream response.
    """
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    # Check cache before streaming
    if cache is not None:
        cached = await cache.get(body.model, body.prompt)
        if cached is not None:
            CACHE_HITS.inc()
            event = {
                "id": request_id,
                "object": "text_completion",
                "model": body.model,
                "choices": [{"index": 0, "text": cached, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": estimate_prompt_tokens(body.prompt),
                    "completion_tokens": len(cached) // 4,
                    "total_tokens": estimate_prompt_tokens(body.prompt) + len(cached) // 4,
                },
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
            }
            yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
            return
        CACHE_MISSES.inc()

    collected_text: list[str] = []
    total_tokens = 0

    async for chunk in client.generate_stream(
        model=body.model,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop_sequences=body.stop or None,
    ):
        token = chunk.get("response", "")
        collected_text.append(token)
        done = chunk.get("done", False)
        finish = "stop" if done else None
        event = {
            "id": request_id,
            "object": "text_completion",
            "model": body.model,
            "choices": [{"index": 0, "text": token, "finish_reason": finish}],
        }
        if done:
            total_tokens = int(chunk.get("eval_count", 0))
            prompt_tok = int(chunk.get("prompt_eval_count", 0))
            event["usage"] = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": total_tokens,
                "total_tokens": prompt_tok + total_tokens,
            }
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            event["latency_ms"] = latency_ms
            # Instrument Prometheus
            REQUEST_LATENCY.labels(model=body.model, endpoint="completions").observe(
                latency_ms / 1000.0
            )
            TOKENS_GENERATED.labels(model=body.model).inc(total_tokens)
            if prompt_tok:
                PROMPT_TOKENS.labels(model=body.model).inc(prompt_tok)
        yield f"data: {json.dumps(event)}\n\n"

    # Cache completed stream response
    if cache is not None and collected_text:
        full_text = "".join(collected_text)
        await cache.put(body.model, body.prompt, full_text)

    yield "data: [DONE]\n\n"


async def _stream_chat_completion(
    client: OllamaClient,
    body: ChatCompletionRequest,
    messages: list[dict[str, str]],
    *,
    cache: Any = None,
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming chat completion.

    Checks *cache* before streaming and stores the completed response.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()
    cache_key = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # Check cache before streaming
    if cache is not None:
        cached = await cache.get(body.model, cache_key)
        if cached is not None:
            CACHE_HITS.inc()
            prompt_text = " ".join(m["content"] for m in messages)
            ptok = estimate_prompt_tokens(prompt_text)
            event = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": cached},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": ptok,
                    "completion_tokens": len(cached) // 4,
                    "total_tokens": ptok + len(cached) // 4,
                },
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
            }
            yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
            return
        CACHE_MISSES.inc()

    collected_text: list[str] = []

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
        collected_text.append(token)
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
            total_tokens = int(chunk.get("eval_count", 0))
            prompt_tok = int(chunk.get("prompt_eval_count", 0))
            event["usage"] = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": total_tokens,
                "total_tokens": prompt_tok + total_tokens,
            }
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            event["latency_ms"] = latency_ms
            # Instrument Prometheus
            REQUEST_LATENCY.labels(model=body.model, endpoint="chat").observe(
                latency_ms / 1000.0
            )
            TOKENS_GENERATED.labels(model=body.model).inc(total_tokens)
            if prompt_tok:
                PROMPT_TOKENS.labels(model=body.model).inc(prompt_tok)
        yield f"data: {json.dumps(event)}\n\n"

    # Cache completed stream response
    if cache is not None and collected_text:
        full_text = "".join(collected_text)
        await cache.put(body.model, cache_key, full_text)

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()

__all__ = ["create_app", "app"]
