"""FastAPI application for the LLM Inference Optimization Engine.

Exposes the following endpoints:

- ``GET  /health``               — Liveness/readiness check
- ``GET  /metrics``              — JSON metrics snapshot
- ``GET  /metrics/prometheus``   — Prometheus-format metrics
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

from llm_inference_engine.api.cache import RedisCache
from llm_inference_engine.api.coalescer import RequestCoalescer
from llm_inference_engine.api.dependencies import PoolDep, ThrottlerDep
from llm_inference_engine.api.fallback_router import FallbackRouter
from llm_inference_engine.api.model_router import ModelRouter
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
from llm_inference_engine.integration.backend_pool import BackendPool
from llm_inference_engine.integration.vllm_backend import VLLMBackend
from llm_inference_engine.metrics.prometheus import (
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    HEALTHY_BACKENDS,
    KV_CACHE_USAGE,
    PROMPT_TOKENS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TOKENS_GENERATED,
)
from llm_inference_engine.optimization.throttler import AdaptiveThrottler, AdmissionDecision
from llm_inference_engine.utils.tokenizer import estimate_prompt_tokens

logger = structlog.get_logger(__name__)

VERSION = "0.2.0"

# In-process request counter (approximate — not cross-worker)
_total_requests: int = 0


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
        global _total_requests
        logger.info("server_starting", version=VERSION)

        # --- Backend pool ---
        vllm_urls = [inst.url for inst in config.vllm.instances]
        pool = BackendPool.from_urls(
            vllm_urls,
            timeout=config.vllm.timeout_seconds,
            max_retries=config.vllm.retry_count,
            retry_backoff_seconds=config.vllm.retry_backoff_seconds,
            failure_threshold=config.circuit_breaker.failure_threshold,
            cooldown_seconds=config.circuit_breaker.cooldown_seconds,
        )

        # Warn if no backends are reachable at startup (non-blocking)
        available = False
        for inst in config.vllm.instances:
            probe = VLLMBackend(inst.url, timeout=config.vllm.timeout_seconds)
            available = await probe.is_available()
            await probe.close()
            if available:
                break
        if not available:
            logger.warning(
                "vllm_unreachable_at_startup",
                instances=[inst.url for inst in config.vllm.instances],
            )

        # --- Redis cache ---
        cache: RedisCache | None = None
        if config.cache.enabled:
            cache = await RedisCache.connect(
                config.redis.url,
                max_size=config.cache.max_size,
                ttl_seconds=config.cache.ttl_seconds,
            )

        # --- Admission throttler ---
        # Use the first vLLM URL for metrics polling
        throttler = AdaptiveThrottler(
            backend_url=vllm_urls[0],
            soft_limit=config.admission_control.soft_limit,
            hard_limit=config.admission_control.hard_limit,
            poll_interval_seconds=config.admission_control.poll_interval_seconds,
        )
        if config.admission_control.enabled:
            await throttler.start()

        # --- Cross-worker coalescer ---
        import redis.asyncio as aioredis
        redis_client = await aioredis.from_url(config.redis.url, decode_responses=True)
        coalescer = RequestCoalescer(redis_client)

        # --- Model router & fallback router ---
        model_router = ModelRouter(config.model_registry)
        fallback_router = FallbackRouter(
            pool=pool,
            fallback_model=config.model_registry.fallback_model,
            cache=cache,
        )

        # --- Store in app state for DI ---
        app.state.pool = pool
        app.state.cache = cache
        app.state.throttler = throttler
        app.state.coalescer = coalescer
        app.state.model_router = model_router
        app.state.fallback_router = fallback_router
        app.state.config = config
        _total_requests = 0

        logger.info("server_ready")
        yield

        # --- Cleanup ---
        await throttler.stop()
        await pool.close()
        if cache is not None:
            await cache.close()
        await redis_client.aclose()
        logger.info("server_stopped")

    app = FastAPI(
        title="LLM Inference Optimization Engine",
        description=(
            "Production-grade inference orchestration layer for vLLM — "
            "Redis-backed caching, health-aware backend pooling, and "
            "KV-cache-pressure admission control."
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

    # --- API-key authentication middleware ---
    if config.auth.enabled:
        from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse
        from starlette.responses import Response as StarletteResponse

        _public_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics/prometheus"}
        _valid_keys = frozenset(config.auth.api_keys)

        class _APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(
                self, request: StarletteRequest, call_next: RequestResponseEndpoint
            ) -> StarletteResponse:
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
    async def health(pool: PoolDep) -> HealthResponse:
        """Return service health status."""
        healthy = pool.healthy_count() > 0
        health_status = "ok" if healthy else "degraded"
        return HealthResponse(
            status=health_status,
            backend_available=healthy,
            version=VERSION,
            details={
                "healthy_backends": pool.healthy_count(),
                "total_backends": len(pool._backends),
            },
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def metrics(
        pool: PoolDep,
        throttler: ThrottlerDep,
    ) -> MetricsResponse:
        """Return a JSON snapshot of runtime metrics."""
        stats = throttler.stats
        cache_obj: RedisCache | None = app.state.cache
        return MetricsResponse(
            kv_cache_usage=stats.kv_cache_usage,
            active_requests=stats.active_requests,
            healthy_backends=pool.healthy_count(),
            cache_hits=cache_obj.hits if cache_obj else 0,
            cache_misses=cache_obj.misses if cache_obj else 0,
            total_requests=_total_requests,
        )

    @app.get("/metrics/prometheus", tags=["System"])
    async def prometheus_metrics() -> StreamingResponse:
        """Return Prometheus-format metrics for scraping."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        try:
            throttler_obj: AdaptiveThrottler = app.state.throttler
            stats = throttler_obj.stats
            KV_CACHE_USAGE.set(stats.kv_cache_usage)
            ACTIVE_REQUESTS.set(stats.active_requests)
        except Exception:
            pass
        try:
            pool_obj: BackendPool = app.state.pool
            HEALTHY_BACKENDS.set(pool_obj.healthy_count())
        except Exception:
            pass

        return StreamingResponse(
            iter([generate_latest()]),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ------------------------------------------------------------------
    # Admission guard
    # ------------------------------------------------------------------

    async def _check_admission() -> None:
        """Reject early if admission control says REJECT."""
        throttler: AdaptiveThrottler = app.state.throttler
        decision = throttler.check()
        if decision == AdmissionDecision.REJECT:
            raise HTTPException(
                status_code=429,
                detail="KV cache pressure is too high. Retry later.",
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
        pool: PoolDep,
        throttler: ThrottlerDep,
    ) -> CompletionResponse | StreamingResponse:
        """Generate a text completion for the given prompt."""
        global _total_requests
        await _check_admission()

        # Resolve target model via router (explicit model in body overrides routing)
        model_router: ModelRouter = app.state.model_router
        model = model_router.route(body.prompt, explicit_model=body.model or None)

        cache_obj: RedisCache | None = app.state.cache
        coalescer: RequestCoalescer = app.state.coalescer

        if body.stream:
            REQUESTS_TOTAL.labels(model=model, endpoint="completions", status="stream").inc()
            throttler.increment_active()
            return StreamingResponse(
                _stream_completion(pool, body, model, cache=cache_obj, throttler=throttler),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        _total_requests += 1
        throttler.increment_active()
        try:
            async def _do_generate() -> Any:
                backend = pool.get_healthy_backend()
                if backend is None:
                    fallback: FallbackRouter = app.state.fallback_router
                    return await fallback.route(
                        model, body.prompt,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        stop=body.stop or None,
                    )
                try:
                    result = await backend.generate(
                        model, body.prompt,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        stop=body.stop or None,
                    )
                    pool.record_success(backend)
                    return result
                except Exception as exc:
                    pool.record_failure(backend)
                    raise exc

            # Check cache first, then coalesce
            if cache_obj is not None:
                cached = await cache_obj.get(model, body.prompt)
                if cached is not None:
                    CACHE_HITS.inc()
                    prompt_tokens = estimate_prompt_tokens(body.prompt)
                    return _build_completion_response(
                        body, model, cached, prompt_tokens, 0, 0.0
                    )
                CACHE_MISSES.inc()

            timeout = body.timeout_seconds or 300.0
            try:
                result = await asyncio.wait_for(
                    coalescer.coalesce(model, body.prompt, _do_generate),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                REQUESTS_TOTAL.labels(model=model, endpoint="completions", status="timeout").inc()
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Request timed out after {timeout}s",
                ) from exc
        finally:
            throttler.decrement_active()

        # Store in cache
        if cache_obj is not None:
            await cache_obj.put(model, body.prompt, result.text)

        prompt_tokens = result.prompt_tokens or estimate_prompt_tokens(body.prompt)
        resp = _build_completion_response(
            body, model, result.text, prompt_tokens, result.tokens_used, result.latency_ms
        )

        REQUESTS_TOTAL.labels(model=model, endpoint="completions", status="ok").inc()
        REQUEST_LATENCY.labels(model=model, endpoint="completions").observe(
            result.latency_ms / 1000.0
        )
        TOKENS_GENERATED.labels(model=model).inc(result.tokens_used)
        PROMPT_TOKENS.labels(model=model).inc(prompt_tokens)
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
        pool: PoolDep,
        throttler: ThrottlerDep,
    ) -> ChatCompletionResponse | StreamingResponse:
        """Generate a chat completion from a list of messages."""
        global _total_requests
        await _check_admission()

        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
        model_router: ModelRouter = app.state.model_router
        model = model_router.route_chat(messages, explicit_model=body.model or None)

        cache_obj: RedisCache | None = app.state.cache
        coalescer: RequestCoalescer = app.state.coalescer

        if body.stream:
            REQUESTS_TOTAL.labels(model=model, endpoint="chat", status="stream").inc()
            throttler.increment_active()
            return StreamingResponse(
                _stream_chat_completion(pool, body, messages, model, cache=cache_obj, throttler=throttler),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        _total_requests += 1
        throttler.increment_active()
        try:
            cache_prompt = " ".join(m["content"] for m in messages)

            async def _do_chat() -> Any:
                backend = pool.get_healthy_backend()
                if backend is None:
                    fallback: FallbackRouter = app.state.fallback_router
                    return await fallback.route_chat(
                        model, messages,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        stop=body.stop or None,
                    )
                try:
                    result = await backend.chat(
                        model, messages,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        stop=body.stop or None,
                    )
                    pool.record_success(backend)
                    return result
                except Exception as exc:
                    pool.record_failure(backend)
                    raise exc

            if cache_obj is not None:
                cached = await cache_obj.get(model, cache_prompt)
                if cached is not None:
                    CACHE_HITS.inc()
                    prompt_tokens = estimate_prompt_tokens(cache_prompt)
                    return _build_chat_response(body, model, cached, prompt_tokens, 0, 0.0)
                CACHE_MISSES.inc()

            timeout = body.timeout_seconds or 300.0
            try:
                result = await asyncio.wait_for(
                    coalescer.coalesce(model, cache_prompt, _do_chat),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                REQUESTS_TOTAL.labels(model=model, endpoint="chat", status="timeout").inc()
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Request timed out after {timeout}s",
                ) from exc
        finally:
            throttler.decrement_active()

        if cache_obj is not None:
            await cache_obj.put(model, cache_prompt, result.text)

        prompt_tokens = result.prompt_tokens or estimate_prompt_tokens(cache_prompt)
        resp = _build_chat_response(
            body, model, result.text, prompt_tokens, result.tokens_used, result.latency_ms
        )

        REQUESTS_TOTAL.labels(model=model, endpoint="chat", status="ok").inc()
        REQUEST_LATENCY.labels(model=model, endpoint="chat").observe(
            result.latency_ms / 1000.0
        )
        TOKENS_GENERATED.labels(model=model).inc(result.tokens_used)
        PROMPT_TOKENS.labels(model=model).inc(prompt_tokens)
        return resp

    return app


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _build_completion_response(
    body: CompletionRequest,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> CompletionResponse:
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        model=model,
        choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        latency_ms=latency_ms,
    )


def _build_chat_response(
    body: ChatCompletionRequest,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


async def _stream_completion(
    pool: BackendPool,
    body: CompletionRequest,
    model: str,
    *,
    cache: RedisCache | None = None,
    throttler: AdaptiveThrottler | None = None,
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming text completion."""
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    if cache is not None:
        cached = await cache.get(model, body.prompt)
        if cached is not None:
            CACHE_HITS.inc()
            if throttler is not None:
                throttler.decrement_active()
            event = {
                "id": request_id,
                "object": "text_completion",
                "model": model,
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

    collected: list[str] = []
    backend = pool.get_healthy_backend()
    if backend is None:
        if throttler is not None:
            throttler.decrement_active()
        yield f"data: {json.dumps({'error': 'No healthy backends available'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        async for chunk in backend.generate_stream(
            model, body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        ):
            text = chunk.get("text", "")
            done = chunk.get("done", False)
            collected.append(text)
            event = {
                "id": request_id,
                "object": "text_completion",
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop" if done else None}],
            }
            if done:
                latency_ms = round((time.monotonic() - start) * 1000, 1)
                event["latency_ms"] = latency_ms
                REQUEST_LATENCY.labels(model=model, endpoint="completions").observe(
                    latency_ms / 1000.0
                )
            yield f"data: {json.dumps(event)}\n\n"

        pool.record_success(backend)
        if cache is not None and collected:
            await cache.put(model, body.prompt, "".join(collected))
    except Exception:
        pool.record_failure(backend)
        raise
    finally:
        if throttler is not None:
            throttler.decrement_active()

    yield "data: [DONE]\n\n"


async def _stream_chat_completion(
    pool: BackendPool,
    body: ChatCompletionRequest,
    messages: list[dict[str, str]],
    model: str,
    *,
    cache: RedisCache | None = None,
    throttler: AdaptiveThrottler | None = None,
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming chat completion."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()
    cache_prompt = " ".join(m["content"] for m in messages)

    if cache is not None:
        cached = await cache.get(model, cache_prompt)
        if cached is not None:
            CACHE_HITS.inc()
            if throttler is not None:
                throttler.decrement_active()
            ptok = estimate_prompt_tokens(cache_prompt)
            event = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model,
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

    collected: list[str] = []
    backend = pool.get_healthy_backend()
    if backend is None:
        if throttler is not None:
            throttler.decrement_active()
        yield f"data: {json.dumps({'error': 'No healthy backends available'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        async for chunk in backend.chat_stream(
            model, messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        ):
            content = chunk.get("content", "")
            done = chunk.get("done", False)
            collected.append(content)
            event = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop" if done else None,
                }],
            }
            if done:
                latency_ms = round((time.monotonic() - start) * 1000, 1)
                event["latency_ms"] = latency_ms
                REQUEST_LATENCY.labels(model=model, endpoint="chat").observe(
                    latency_ms / 1000.0
                )
            yield f"data: {json.dumps(event)}\n\n"

        pool.record_success(backend)
        if cache is not None and collected:
            await cache.put(model, cache_prompt, "".join(collected))
    except Exception:
        pool.record_failure(backend)
        raise
    finally:
        if throttler is not None:
            throttler.decrement_active()

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()

__all__ = ["create_app", "app"]
