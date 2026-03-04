# LLM Inference Optimization Engine — Project Direction

---

## Where We're Going and Why

The original project was built as a middleware optimization layer on top of **Ollama**. Ollama is a great tool for running models locally during development, but it was never designed for anything resembling production workloads. It processes one request at a time, offers no real backpressure signals, no continuous batching, and no GPU-level memory visibility. Building a sophisticated optimization layer on top of it is the wrong foundation for the wrong audience.

We are pivoting the backend to **vLLM**.

vLLM is a proper inference engine. It implements PagedAttention for efficient KV cache management, continuous batching so requests don't queue behind each other, real GPU memory metrics you can actually query, and an OpenAI-compatible API surface. These aren't incremental improvements — they change what kind of problems this middleware can meaningfully solve.

With vLLM underneath, the middleware stops being a band-aid over Ollama's limitations and starts being a genuine orchestration layer. Caching, circuit breaking, request coalescing, and admission control all become more valuable when the backend can actually handle load. The features we're adding — a backend pool, fallback routing, a model registry — are only worth building if the inference engine can back them up.

**The target user has also shifted.** This is no longer aimed at a solo developer running a model on their laptop. The intended user is a small-to-medium engineering team running one or more vLLM instances and needing a reliable, observable, and maintainable proxy in front of them.

One more thing: the codebase needs to be simpler. The current implementation has strong engineering discipline (strict typing, high test coverage, good structure) but some components are over-engineered for what they do. Going forward, every module should be readable by a competent engineer without needing to trace through abstractions. Complexity is only acceptable when it is doing real work.

---

## What We're Keeping

### CircuitBreaker
Fully backend-agnostic. The closed → open → half-open state machine, failure counting, and probe-based recovery need no changes. The only update is wiring it to vLLM's health endpoint instead of Ollama's.

### FastAPI Routes and Pydantic Models
The API surface is already shaped like OpenAI's contract, which vLLM also speaks natively. The `/completions` and `/chat/completions` routes, request validation, SSE streaming logic, and error response shapes are all reusable.

### Prometheus Metrics and Structlog
Operational instrumentation is backend-agnostic. The Prometheus histogram for request latency, the cache hit/miss counters, and the structured logging setup all stay as-is.

### Test Infrastructure
616 unit tests, strict mypy, ruff, and pytest-asyncio. This is the most underappreciated part of the project and it needs to be preserved. vLLM integration will require more test coverage, not less.

### InferenceBackend Abstract Base Class
The decision to define an abstract backend interface was the right call. It becomes the foundation for the backend pool described below.

### Cache Logic (Not the Storage)
The cache key normalisation strategy, TTL handling, LRU eviction, and the concept of a semantic/embedding cache are all worth keeping. What changes is the storage layer underneath them.

---

## What We're Abandoning

### OllamaClient
Replaced entirely with an `httpx` async client targeting vLLM's OpenAI-compatible endpoints. Since vLLM speaks standard OpenAI API, the client becomes dramatically simpler — no bespoke Ollama-specific request formatting needed.

### MemoryEstimator Heuristics
The current estimator makes rough guesses based on model size because Ollama provides no real memory visibility. vLLM exposes actual GPU cache utilisation via its Prometheus metrics endpoint (`vllm:gpu_cache_usage_perc`). The guesswork gets replaced with real observations.

### DraftModelManager and Speculative Decoding
vLLM implements speculative decoding natively at the engine level. Doing it in middleware is redundant and incorrect — you cannot implement this properly without access to the token-level state that lives inside the inference engine.

### Scheduling Policies (SJF, FCFS, Priority, TokenBudget)
vLLM has its own continuous batching scheduler that operates at the GPU level. The per-model queues and drain loop in this project were compensating for Ollama's serial execution model. With vLLM, that problem doesn't exist. The scheduling subsystem — all four policies, the drain loop, the request queue wrapper — gets removed. The complexity was never doing real work against a capable backend.

### In-Process Cache Storage
The in-memory cache with no persistence was acceptable for a single-user Ollama setup. For a proper deployment with multiple potential workers, it's a liability. The cache logic stays; the in-process storage layer is replaced with Redis.

---

## What We're Adding

### Redis-Backed Cache Storage
The existing cache logic (LRU, TTL, key normalisation) is wired to an in-process dict today. That layer gets replaced with Redis as the storage backend. This gives the cache persistence across restarts and makes it accessible across multiple workers. The cache interface doesn't change — only what sits behind it.

### BackendPool with Health-Aware Routing
The `InferenceBackend` ABC gets its first real use. A `BackendPool` class manages a list of vLLM backend instances and routes incoming requests based on health state and load. Initially this means simple round-robin with health exclusion — backends that fail circuit breaker checks are skipped. This is the feature that makes the middleware worth deploying in front of vLLM rather than hitting vLLM directly.

### Real Admission Control Based on vLLM Metrics
The `AdaptiveThrottler` gets rebuilt around vLLM's actual GPU cache pressure metrics rather than estimated memory costs. The throttler polls vLLM's `/metrics` endpoint on a configurable interval and uses `vllm:gpu_cache_usage_perc` as the admission signal. When cache pressure is high, new requests get queued or rejected with a proper 429. When pressure is normal, requests flow through without interference.

### Cross-Worker Request Coalescing via Redis
The `RequestCoalescer` concept (identical in-flight prompts share one backend call) is correct and vLLM doesn't do this natively. The current implementation only works within a single process because it tracks in-flight requests in memory. The rebuild uses Redis for coordination so coalescing works across multiple workers.

### Fallback Routing on Circuit Open
Today when the circuit breaker opens, the middleware returns 503. The new behaviour is smarter: when the primary backend is unavailable, the request gets routed to a fallback — either a different model on the same vLLM instance, a secondary backend, or a cached response under relaxed similarity thresholds. 503 becomes the last resort, not the first response.

### Model Registry and Request Router
A config-driven model registry that maps request characteristics to backend targets. This is a YAML configuration specifying which model handles which category of request. At its simplest, it's a routing table: fast models for short interactive prompts, larger models for complex or long-context requests. This turns the middleware from a proxy into an orchestration layer.

---

## Simplification Mandate

Every module that remains or is added must meet this bar: a competent engineer should be able to read the code, understand what it does, and trace a request through it without consulting separate documentation.

Concretely, this means:

**Flatten unnecessary abstraction.** If a class exists only to wrap another class with no added behaviour, it gets removed. Delegation chains that exist for theoretical extensibility get collapsed.

**Functions do one thing.** Any function longer than 40 lines is examined for decomposition. Any function that requires an inline comment to explain what a block of code is doing gets that block extracted into a named function.

**No clever code.** No single-expression list comprehensions doing three things at once. No chained method calls that require reading inside-out. Clarity over brevity in every tradeoff.

**Config stays in one place.** All tuneable values live in `configs/default.yaml`. No magic numbers in source code. No environment variable checks scattered across modules — one config loader, one place to look.

**Module names describe what the module does.** If you have to open the file to understand what it contains, the name is wrong.

This is not about reducing functionality. It is about making the functionality that exists legible to the people who will maintain it.

---

## New Architecture Overview

```
Client Request
      │
      ▼
┌─────────────┐
│  FastAPI     │  /completions, /chat/completions, /health, /metrics
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Redis       │  Persistent LRU+TTL cache, cross-worker coalescing
│  Cache       │  Cache hit → return immediately
└──────┬──────┘
       │ cache miss
       ▼
┌─────────────┐
│  Admission   │  Polls vLLM /metrics for real GPU cache pressure
│  Controller  │  ACCEPT / QUEUE / REJECT based on live signal
└──────┬──────┘
       │ accepted
       ▼
┌─────────────┐
│  Request     │  Redis-coordinated deduplication of identical
│  Coalescer   │  in-flight prompts across workers
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Model       │  Config-driven routing table: request → model → backend
│  Router      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Backend     │  Health-aware pool of vLLM instances
│  Pool        │  Round-robin with circuit breaker exclusion
└──────┬──────┘
       │ primary unavailable
       ▼
┌─────────────┐
│  Fallback    │  Secondary model, secondary backend, or relaxed cache hit
│  Router      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  vLLM        │  OpenAI-compatible API, continuous batching, PagedAttention
│  Backend(s)  │
└─────────────┘
```

---

## Component Decision Summary

| Component | Decision | Reason |
|---|---|---|
| CircuitBreaker | **Keep** | Backend-agnostic, correct as-is |
| FastAPI routes + Pydantic | **Keep** | Already OpenAI-shaped |
| Prometheus + structlog | **Keep** | Backend-agnostic ops |
| Test infrastructure | **Keep** | Highest-value asset in the project |
| InferenceBackend ABC | **Keep + Expand** | Foundation for BackendPool |
| Cache logic | **Keep, swap storage** | Logic is correct, storage backend changes to Redis |
| OllamaClient | **Remove** | Replaced by vLLM httpx client |
| MemoryEstimator heuristics | **Remove** | Replaced by real vLLM metrics |
| DraftModelManager | **Remove** | vLLM handles this natively |
| Scheduling policies + drain loop | **Remove** | vLLM's continuous batching owns this |
| In-process cache storage | **Remove** | Replaced by Redis |
| BackendPool + load balancer | **Add** | Core value proposition |
| Real admission control | **Add** | GPU cache pressure feedback loop |
| Cross-worker coalescing via Redis | **Add** | Extends correct existing concept |
| Fallback routing | **Add** | Smarter failure handling |
| Model registry + request router | **Add** | Genuine orchestration capability |
