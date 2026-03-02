# Architecture Overview

## System Design

The LLM Inference Optimization Engine is built as an orchestration layer on top of Ollama, focusing on intelligent request management and optimization rather than reimplementing model serving.

### High-Level Architecture

```
┌──────────────────────────────────────────────┐
│         Your REST API (Port 8000)            │
│    /completions, /chat/completions, /health   │
├──────────────────────────────────────────────┤
│      Request Orchestration Layer             │
│  • Batching & Scheduling                     │
│  • Result Aggregation                        │
│  • Caching & Optimization                    │
├──────────────────────────────────────────────┤
│         Ollama Client (HTTP)                 │
│  • Connection Management                     │
│  • Retry Logic & Error Handling              │
│  • Model Information Management              │
├──────────────────────────────────────────────┤
│         Ollama Service (Port 11434)          │
│  • Model Serving                             │
│  • Quantization (Built-in)                   │
│  • Forward Passes                            │
│  • Metal Optimization                        │
├──────────────────────────────────────────────┤
│         Llama.cpp + Metal                    │
│  • Actual Inference Execution                │
└──────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. Ollama Integration Layer (Phase 1)

**Purpose**: Manage communication with Ollama service

**Components**:
- **OllamaClient**: HTTP client for Ollama API
  - Connection pooling
  - Retry logic with exponential backoff
  - Timeout management
  - Health checks

- **OllamaModelManager**: Model information and availability
  - Model discovery
  - Quantization level detection
  - Memory estimation
  - Model verification

**Key Design Decisions**:
- Asynchronous HTTP for scalability
- Separate client and manager responsibilities
- Configuration-driven connection parameters
- Graceful error handling with detailed messages

### 2. Core Types (Phase 1)

**Purpose**: Define data models and type safety

**Components**:
- **Request**: Inference request representation
- **Response**: Inference result wrapper
- **GenerationConfig**: Text generation parameters
- **RequestStatus**: Request lifecycle tracking

**Key Design Decisions**:
- Dataclasses for simplicity and performance
- Validation at construction time
- Immutable request IDs for tracing
- Explicit status tracking

### 3. Configuration System (Phase 1)

**Purpose**: Centralized configuration management

**Components**:
- **OllamaConfig**: Ollama service connection
- **ModelConfig**: Individual model settings
- **ServerConfig**: API server settings
- **InferenceConfig**: Top-level configuration

**Key Design Decisions**:
- YAML-based configuration files
- Environment variable overrides (future)
- Type-safe configuration objects
- Default values for all settings

## Data Flow

### Request Processing Flow

```
1. Client sends HTTP request
   ↓
2. API validates request parameters
   ↓
3. Request object created with ID
   ↓
4. Scheduler evaluates queue
   ↓
5. Batch formation decision
   ├─ Batch ready → Execute
   └─ Not ready → Queue
   ↓
6. Call Ollama API
   ↓
7. Aggregate results
   ↓
8. Return to client
```

### Model Information Flow

```
1. Startup: Query Ollama for models
   ↓
2. Parse model names & metadata
   ↓
3. Cache model information
   ↓
4. Periodic refresh (background)
   ↓
5. Provide to request handler
```

## Design Principles

### 1. Separation of Concerns
- Ollama handles model serving and inference
- Our code handles orchestration and optimization
- Clear boundaries between layers

### 2. Fail-Fast with Helpful Messages
- Validation at entry points
- Detailed error messages with context
- Actionable guidance for users

### 3. Async-First
- All I/O operations are asynchronous
- Supports high concurrency
- Efficient resource utilization

### 4. Type Safety
- Comprehensive type hints
- Mypy strict mode
- Pydantic validation where needed

### 5. Observability
- Structured logging with structlog
- Request tracing via IDs
- Metrics collection at key points

## Technology Choices

### Python 3.11+
- Modern async/await support
- Performance improvements
- Type hinting enhancements

### FastAPI
- Async-native web framework
- Automatic OpenAPI documentation
- Pydantic integration

### httpx
- Async HTTP client
- Connection pooling
- Timeout support

### structlog
- Structured logging
- Context preservation
- JSON output support

## Phase 1 Status

**Completed**:
- ✅ Ollama client implementation
- ✅ Model manager
- ✅ Core type definitions
- ✅ Configuration system
- ✅ Setup verification script
- ✅ Unit tests

**Next Steps** (Phase 2):
- Quantization understanding
- Benchmarking framework
- Quality/speed tradeoff analysis

## Integration Points

### With Ollama
- HTTP API at `localhost:11434`
- Endpoints: `/api/generate`, `/api/tags`
- JSON request/response format
- Streaming support (future)

### With Future Phases
- Phase 3: Scheduler uses model manager for batch sizing
- Phase 4: Memory estimates from model info
- Phase 5: REST API orchestrates all components
- Phase 6: Speculation uses model compatibility checks

## Deployment Considerations

### Requirements
- Ollama installed and running
- Python 3.11+ environment
- 16GB RAM (M2 Air target)
- Models pulled via Ollama

### Configuration
- YAML files in `configs/`
- Override via environment variables
- Sensible defaults provided

### Monitoring
- Health check endpoint
- Structured logs
- Metrics (Phase 5)

## Future Enhancements

1. **Connection Pooling**: Reuse connections to Ollama
2. **Circuit Breaker**: Fail fast if Ollama is down
3. **Model Caching**: Cache model metadata longer
4. **Configuration Validation**: JSON Schema validation
5. **Secret Management**: Secure credential handling

---

## Phases 3–6: Implemented Components

### Phase 3: Scheduling Engine (`src/llm_inference_engine/scheduling/`)

| Component | File | Responsibility |
|---|---|---|
| `RequestQueue` | `queue.py` | Async priority queue with FIFO tie-breaking and cancellation |
| `Batch` | `batch.py` | Groups requests with token budget and memory footprint tracking |
| `SchedulingPolicy` | `policies.py` | FCFS, SJF, Priority, TokenBudget batch formation strategies |
| `Scheduler` | `scheduler.py` | Per-model queues → policy → batch dispatch orchestration |

**Data flow:**
```
submit(Request) → per-model RequestQueue
drain(model)    → collect requests → form Batch via policy → dispatch_fn(Batch)
```

### Phase 4: Memory & Capacity Planning (`src/llm_inference_engine/optimization/`)

| Component | File | Responsibility |
|---|---|---|
| `MemoryEstimator` | `memory.py` | Predicts peak memory = model weights + KV-cache |
| `AdaptiveThrottler` | `throttler.py` | ACCEPT / QUEUE / REJECT admission control |
| `ContextWindowManager` | `context.py` | Model context window lookup and utilisation calculation |

**Memory model:**
- Model weights: `num_params × bytes_per_param(quantization) × safety_margin`
- KV-cache: `num_tokens × num_layers × 512 B/token/layer × safety_margin`
- Throttling thresholds: soft (85 % of limit) and hard (14 GB default for M2 Air)

### Phase 5: API Layer (`src/llm_inference_engine/api/`)

| Component | File | Responsibility |
|---|---|---|
| `CompletionRequest/Response` | `models.py` | OpenAI-compatible Pydantic models |
| `SemanticCache` | `cache.py` | Exact-match LRU cache with TTL eviction |
| `ResultMapper` | `result_mapper.py` | Maps request IDs to `asyncio.Future` objects |
| `RequestAggregator` | `aggregator.py` | Cache check → schedule → dispatch → fan-in |
| FastAPI app | `server.py` | `/completions`, `/chat/completions`, `/health`, `/metrics` |
| DI providers | `dependencies.py` | FastAPI `Depends()` accessors for shared components |

**Request lifecycle:**
```
POST /completions
  → CompletionRequest validation (Pydantic)
  → SemanticCache.get() — hit returns immediately
  → ResultMapper.register() — creates Future
  → Scheduler.submit() + drain()
  → dispatch_batch() — concurrent Ollama calls
  → ResultMapper.resolve() — delivers Response
  → Future awaited, CompletionResponse returned
```

### Phase 6: Draft Model Infrastructure (`src/llm_inference_engine/optimization/`)

| Component | File | Responsibility |
|---|---|---|
| `DraftModelManager` | `draft_manager.py` | Small model lifecycle + candidate generation |

> **Note:** The `SpeculationEngine` (speculative decoding) was removed in
> Round 5 because HTTP-level APIs cannot expose the logit-level access
> required for proper draft-verify loops. The `DraftModelManager` remains
> as infrastructure for future use with backends that support logit access.

---

## Performance Improvements (perf/1–8)

Eight targeted improvements were applied to production-critical code paths after Phase 7.

### perf/1 — O(1) Batch Token Tracking
**File:** `scheduling/batch.py`
- Added `_total_tokens: int` field incremented in `add()`
- Removed O(n) sum on every `can_add()` call
- Impact: scheduling throughput at batch sizes > 16

### perf/2 — Async-Safe SemanticCache
**File:** `api/cache.py`
- Added `asyncio.Lock` guard around all multi-step operations
- Converted `get`/`put`/`invalidate`/`clear` to async methods
- Impact: prevents cache corruption under concurrent requests

### perf/3 — Lock-Free Queue Creation
**File:** `scheduling/scheduler.py`
- Replaced `async with self._lock` in `_get_or_create_queue()` with `dict.setdefault()`
- Atomic under CPython GIL; eliminates coroutine suspension on every submit
- Impact: submit throughput under high concurrency

### perf/4 — Regex Precompile
**Files:** `optimization/draft_manager.py`
- Moved `re.compile(r"(\s+)")` to module-level constant `_TOKEN_SPLIT_RE`
- Impact: draft model candidate generation latency

### perf/5 — Memoized Estimators
**Files:** `optimization/memory.py`, `optimization/context.py`
- Applied `@functools.lru_cache` to weight estimation and context window lookups
- Pre-sorted registry items by key length for O(1) prefix matching
- Impact: per-request lookup latency, especially under high concurrency

### perf/6 — Bounded Cancellation Set
**File:** `scheduling/queue.py`
- Added `_queued_ids: set[str]`; `cancel()` only inserts to `_cancelled` if ID is queued
- Bounds `_cancelled` set memory to at most queue size (previously unbounded)
- Impact: memory in long-running servers with stale cancellation IDs

### perf/7 — Config-Driven Server Startup
**Files:** `config.py`, `api/server.py`, `configs/default.yaml`
- Added `SchedulingConfig`, `CacheConfig`, `MemoryConfig` to `InferenceConfig`
- Server lifespan now reads `config.cache.*`, `config.scheduling.*`, `config.memory.*`
- Impact: correct resource sizing at startup; tunable without code changes

### perf/8 — Jitter in Retry Backoff
**File:** `integration/ollama_client.py`
- Changed `sleep(2 ** attempt)` to `sleep(base * (2 ** attempt) + uniform(0, 1))`
- Base delay sourced from `config.ollama.retry_backoff_seconds`
- Impact: eliminates thundering-herd retry storms; improves tail latency
