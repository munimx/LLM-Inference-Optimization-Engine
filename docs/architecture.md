# Architecture

## Overview

The LLM Inference Optimization Engine is a middleware layer between your application and one or more [vLLM](https://github.com/vllm-project/vllm) instances. It adds caching, cross-worker request coalescing, adaptive admission control, smart model routing, and backend-pool management — without requiring changes to client code that already speaks the OpenAI API.

[Redis](https://redis.io) is the only shared-state store. All workers and replicas share the same cache and coalescer; there is no diverging in-process state.

---

## Component Map

```
Client
  │
  ▼
FastAPI (server.py)
  │
  ├─ GET /health, /metrics, /metrics/prometheus
  │
  └─ POST /completions, /chat/completions
        │
        ├─ 1. Auth check (optional Bearer token)
        │
        ├─ 2. ModelRouter           ← config.model_registry
        │       route(prompt, explicit_model)
        │       fast_model (short) | large_model (long)
        │
        ├─ 3. RedisCache.get(key)   ← Redis
        │       hit  → return cached response immediately
        │       miss → continue
        │
        ├─ 4. AdaptiveThrottler.check()   ← polls vLLM /metrics
        │       ACCEPT → continue
        │       QUEUE  → wait
        │       REJECT → 429
        │
        ├─ 5. RequestCoalescer.coalesce()   ← Redis SET NX + pub/sub
        │       winner → execute inference, publish result
        │       waiter → subscribe, receive result from winner
        │
        ├─ 6. BackendPool.get_healthy_backend()
        │       round-robin, skip open circuits
        │       None → FallbackRouter (fallback model → stale cache → 503)
        │
        ├─ 7. VLLMBackend.generate() / .chat()
        │       POST /v1/completions or /v1/chat/completions
        │       retry on transient network errors
        │
        ├─ 8. BackendPool.record_success/failure()
        │       update CircuitBreaker state
        │
        └─ 9. RedisCache.put(key, response)
```

---

## Components

### VLLMBackend (`integration/vllm_backend.py`)

Wraps a single vLLM instance behind an async `httpx` client. Implements the `InferenceBackend` ABC.

**Endpoints called:**

| Method | vLLM path | Purpose |
|--------|-----------|---------|
| `GET` | `/health/ready` | Liveness check |
| `GET` | `/v1/models` | List served models |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/chat/completions` | Chat completion |

**Error handling:** Retries on `httpx.NetworkError` / `httpx.TimeoutException`. HTTP 4xx/5xx are wrapped in `RuntimeError("vLLM returned {status}: {body}")`.

---

### BackendPool (`integration/backend_pool.py`)

Distributes requests across multiple `VLLMBackend` instances using round-robin. Each backend has a paired `CircuitBreaker`; backends with an open circuit are skipped in `get_healthy_backend()`.

**State:**
- `_backends: list[VLLMBackend]` — backend instances
- `_breakers: list[CircuitBreaker]` — parallel list of circuit breakers
- `_index: int` — round-robin cursor

**Factory:** `BackendPool.from_urls(["http://vllm-1:8080", "http://vllm-2:8080"])` creates backends and breakers from URL list.

---

### CircuitBreaker (`api/circuit_breaker.py`)

Standard closed → open → half-open state machine.

| State | Behaviour |
|-------|-----------|
| **Closed** | Requests pass through normally |
| **Open** | Requests are blocked; backend is skipped in pool |
| **Half-open** | One probe request is allowed; success closes, failure reopens |

Opens after `failure_threshold` consecutive failures. Transitions to half-open after `cooldown_seconds`.

---

### RedisCache (`api/cache.py`)

Redis-backed LRU response cache shared across all workers and replicas.

**Storage layout:**
- String key `cache:{sha256_of_cache_key}` → serialised response JSON
- Sorted set `cache:lru` → member: cache key, score: access timestamp

**Eviction:** On `put()`, if `ZCARD cache:lru > max_size`, the oldest entry (lowest score) is deleted atomically in a pipeline.

**TTL:** Each string key is set with `EX ttl_seconds`. An expired key is a cache miss; the LRU entry is cleaned up lazily.

**Key construction:** `"{model}:{prompt_normalised}:{max_tokens}:{temperature}"` — prompt is lowercased and whitespace-normalised.

---

### AdaptiveThrottler (`optimization/throttler.py`)

Polls `GET /metrics` on the primary vLLM instance and parses `vllm:kv_cache_usage_perc` from the Prometheus text payload. A background `asyncio` task refreshes this value every `poll_interval_seconds`.

**Admission decisions:**

| KV-cache usage | Decision |
|----------------|----------|
| `< soft_limit` | `ACCEPT` |
| `soft_limit` ≤ usage `< hard_limit` | `QUEUE` |
| `≥ hard_limit` | `REJECT` (HTTP 429) |

`check()` is synchronous (reads the cached metric). `increment_active()` / `decrement_active()` track in-flight count for the `/metrics` endpoint.

---

### RequestCoalescer (`api/coalescer.py`)

Prevents redundant backend calls when multiple workers receive identical requests concurrently.

**Protocol:**
1. Compute `key = sha256("{model}:{prompt}")`.
2. Attempt `SET coalesce:{key} "locked" NX PX 30000`.
3. **Winner** (SET succeeded): execute inference, then `PUBLISH coalesce:result:{key} <json>`.
4. **Waiters** (SET failed): `SUBSCRIBE coalesce:result:{key}`, wait up to 25 s, deserialise result.
5. If a waiter times out, it falls back to executing the inference itself.

---

### ModelRouter (`api/model_router.py`)

Selects the target model name before the request reaches the backend.

**Rules (evaluated in order):**
1. If `request.model` is non-empty, use it as-is (explicit override).
2. If `estimate_prompt_tokens(prompt) < fast_model_token_threshold`, use `fast_model`.
3. Otherwise use `large_model`.

Configuration lives in `model_registry` in `configs/default.yaml`.

---

### FallbackRouter (`api/fallback_router.py`)

Called when `BackendPool.get_healthy_backend()` returns `None` (all circuits open).

**Fallback chain:**
1. Try `fallback_model` on any available backend.
2. Return a cached response for the same cache key if one exists (may be stale).
3. Raise `HTTPException(503)`.

---

## Request Lifecycle

### Non-streaming completion

```
POST /completions
 → auth check
 → ModelRouter.route(prompt)           # select model
 → RedisCache.get(key)                 # cache check
   cache hit  → CompletionResponse (cached=True)
   cache miss:
     → AdaptiveThrottler.check()
       REJECT → 429
       QUEUE  → asyncio.sleep + re-check
       ACCEPT:
         → RequestCoalescer.coalesce()
           winner:
             → BackendPool.get_healthy_backend()
               None → FallbackRouter
             → VLLMBackend.generate()
             → BackendPool.record_success()
             → RedisCache.put(key, response)
             → coalescer publishes result
           waiter:
             → receive result from Redis pub/sub
 → CompletionResponse
```

### Streaming completion

The streaming path skips caching (streaming responses are not cached) and coalescing. It goes directly to `VLLMBackend.generate_stream()` / `.chat_stream()` and returns a `StreamingResponse` with `text/event-stream` content type.

---

## Prometheus Metrics

Exported at `GET /metrics/prometheus`:

| Metric | Type | Description |
|--------|------|-------------|
| `llm_engine_requests_total` | Counter | Total requests by status |
| `llm_engine_request_latency_seconds` | Histogram | End-to-end latency |
| `llm_engine_prompt_tokens_total` | Counter | Prompt tokens processed |
| `llm_engine_tokens_generated_total` | Counter | Completion tokens generated |
| `llm_engine_cache_hits_total` | Counter | Cache hits |
| `llm_engine_cache_misses_total` | Counter | Cache misses |
| `llm_engine_active_requests` | Gauge | In-flight requests |
| `llm_engine_kv_cache_usage` | Gauge | vLLM KV-cache pressure (0.0–1.0) |
| `llm_engine_healthy_backends` | Gauge | Backends with closed/half-open circuit |

---

## Configuration Reference

All config is in `configs/default.yaml` and loaded into `InferenceConfig` at startup.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `vllm` | `instances` | `[{url: "http://localhost:8080"}]` | vLLM instance URLs |
| `vllm` | `timeout_seconds` | `120` | HTTP request timeout |
| `vllm` | `retry_count` | `2` | Retries on transient errors |
| `vllm` | `retry_backoff_seconds` | `0.5` | Wait between retries |
| `redis` | `url` | `redis://localhost:6379/0` | Redis connection URL |
| `redis` | `socket_timeout_seconds` | `5.0` | Redis socket timeout |
| `cache` | `enabled` | `true` | Enable response caching |
| `cache` | `max_size` | `256` | Max LRU entries |
| `cache` | `ttl_seconds` | `300` | Entry TTL |
| `admission_control` | `soft_limit` | `0.70` | QUEUE threshold |
| `admission_control` | `hard_limit` | `0.90` | REJECT threshold |
| `admission_control` | `poll_interval_seconds` | `5.0` | Throttler poll interval |
| `circuit_breaker` | `failure_threshold` | `5` | Failures before opening |
| `circuit_breaker` | `cooldown_seconds` | `30` | Open → half-open wait |
| `model_registry` | `fast_model` | `mistralai/Mistral-7B-Instruct-v0.2` | Short-prompt model |
| `model_registry` | `large_model` | `meta-llama/Meta-Llama-3-70B-Instruct` | Long-prompt model |
| `model_registry` | `fast_model_token_threshold` | `512` | Token boundary |
| `model_registry` | `fallback_model` | `mistralai/Mistral-7B-Instruct-v0.2` | Pool-down fallback |
| `auth` | `enabled` | `false` | Require Bearer token |
| `auth` | `api_keys` | `[]` | Accepted API keys |
| `server` | `host` | `0.0.0.0` | Bind address |
| `server` | `port` | `8000` | Bind port |
| `server` | `workers` | `4` | Uvicorn workers |

