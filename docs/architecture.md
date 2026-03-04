# Architecture

## Overview

The LLM Inference Optimization Engine is a middleware layer that sits between client applications and an Ollama inference backend. It provides caching, request scheduling, memory-aware admission control, and reliability features.

```
Client Request
      │
      ▼
┌─────────────┐
│   FastAPI    │  Routes: /completions, /chat/completions, /health, /metrics
│   Server     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Response   │  Case-insensitive key lookup, TTL + LRU eviction
│   Cache      │  Cache hit → return immediately (1.6–2.8 ms)
└──────┬──────┘
       │ cache miss
       ▼
┌─────────────┐
│  Adaptive    │  Estimate memory cost → ACCEPT / QUEUE / REJECT
│  Throttler   │
└──────┬──────┘
       │ accepted
       ▼
┌─────────────┐
│  Scheduler   │  Per-model queue, configurable policy (FCFS/SJF/Priority/TokenBudget)
│              │  Drain loop dispatches batches to backend
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Request     │  Identical in-flight prompts share one backend call
│  Coalescer   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Circuit     │  Closed → Open (after N failures) → Half-Open (probe after cooldown)
│  Breaker     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Ollama      │  httpx async client with retry + exponential backoff
│  Client      │
└─────────────┘
```

## Components

### API Layer

| Component | File | Responsibility |
|-----------|------|---------------|
| `create_app()` | `api/server.py` | FastAPI app factory, lifespan management, route registration |
| `CompletionRequest` | `api/routes.py` | Pydantic model for `/completions` input validation |
| `ChatCompletionRequest` | `api/routes.py` | Pydantic model for `/chat/completions` input validation |

### Caching

| Component | File | Responsibility |
|-----------|------|---------------|
| `ExactMatchCache` | `cache/exact_match.py` | LRU + TTL cache with case-insensitive key normalisation |
| `EmbeddingCache` | `cache/embedding_cache.py` | Semantic similarity cache using embedding vectors |
| `CacheManager` | `cache/manager.py` | Factory that selects cache backend based on config |

**Cache key normalisation**: Keys are lowered and stripped. Generation parameters (max_tokens, temperature) are appended as `\x00mt=N\x00t=T` to prevent cross-parameter cache pollution.

### Scheduling

| Component | File | Responsibility |
|-----------|------|---------------|
| `Scheduler` | `scheduling/scheduler.py` | Per-model queue management, drain loop, dispatch with timeout |
| `FCFSPolicy` | `scheduling/policies.py` | First-Come First-Served ordering |
| `SJFPolicy` | `scheduling/policies.py` | Shortest Job First with 30-second starvation guard |
| `PriorityPolicy` | `scheduling/policies.py` | Priority-based ordering |
| `TokenBudgetPolicy` | `scheduling/policies.py` | Fill batches up to a token budget |
| `RequestQueue` | `scheduling/request_queue.py` | asyncio.Queue wrapper with overflow re-enqueue |

**Overflow handling**: When the queue is full, requests are re-enqueued asynchronously rather than dropped.

**Dispatch errors**: Failed requests are marked `FAILED` and their Future receives the exception.

### Optimisation

| Component | File | Responsibility |
|-----------|------|---------------|
| `AdaptiveThrottler` | `optimization/adaptive_throttler.py` | Sliding-window memory tracking, ACCEPT/QUEUE/REJECT decisions |
| `MemoryEstimator` | `optimization/memory_estimator.py` | Model-size-based memory cost estimates |
| `CircuitBreaker` | `optimization/circuit_breaker.py` | Failure counting, state transitions, probe-based recovery |

### Integration

| Component | File | Responsibility |
|-----------|------|---------------|
| `OllamaClient` | `core/ollama_client.py` | Async HTTP client for Ollama `/api/generate` and `/api/chat` |
| `InferenceBackend` | `core/backend.py` | Abstract base class for backend extensibility |
| `DependencyContainer` | `core/dependencies.py` | Singleton container for runtime component wiring |

## Request Lifecycle

1. **Route handler** validates the request body via Pydantic
2. **Cache lookup** — on hit, returns stored response (streaming replays cached tokens)
3. **Throttler** estimates memory cost and returns ACCEPT, QUEUE, or REJECT
4. **Scheduler** enqueues the request into the model's queue
5. **Drain loop** selects next batch using the configured policy
6. **Coalescer** checks for identical in-flight prompts; shares result if found
7. **Circuit breaker** checks backend health; rejects immediately if circuit is open
8. **OllamaClient** dispatches to Ollama with retry + backoff
9. **Response** is cached and returned (or streamed token-by-token via SSE)

## Design Decisions

**Why per-model queues?** — Different models have different latency profiles. A fast model shouldn't wait behind a slow model's backlog.

**Why in-process cache?** — Eliminates network hops for cache hits (1.6–2.8 ms). The tradeoff is no persistence across restarts and no cross-worker sharing. For single-worker Ollama setups this is the right default.

**Why request coalescing?** — Multiple users asking the same question simultaneously generate one backend call instead of N. Most valuable for popular prompts.

**Why circuit breaker?** — Prevents cascading failures when Ollama is down. Without it, every request would wait for the full timeout before failing.

**Why SJF starvation guard?** — Pure SJF starves long prompts indefinitely. The 30-second guard promotes waiting requests to prevent this.
