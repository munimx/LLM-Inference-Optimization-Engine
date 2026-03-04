# LLM Inference Optimization Engine

Request scheduling, caching, streaming, and inference orchestration middleware for [Ollama](https://ollama.com).

Sits between your application and Ollama. Incoming requests are checked against a case-insensitive response cache, queued by a configurable scheduling policy, dispatched to the backend, and streamed back via SSE or returned as a complete JSON response.

## Quick Start

```bash
# Prerequisites: Python 3.11+, Ollama running on localhost:11434
pip install -e .
uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000
```

```bash
# Text completion
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Explain quicksort", "max_tokens": 128}'

# Streaming
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Explain quicksort", "max_tokens": 128, "stream": true}'
```

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────┐     ┌────────┐
│ Application │────▶│  LLM Inference Optimization Engine          │────▶│ Ollama │
│             │◀────│  Cache → Throttler → Scheduler → Dispatch   │◀────│        │
└─────────────┘     └─────────────────────────────────────────────┘     └────────┘
```

| Layer | Components | Purpose |
|-------|-----------|---------|
| **API** | FastAPI server, Pydantic models | OpenAI-compatible HTTP endpoints |
| **Cache** | ExactMatchCache, EmbeddingCache | Case-insensitive response caching with TTL + LRU |
| **Scheduling** | Scheduler, Policies, RequestQueue | Per-model queuing with FCFS/SJF/Priority/TokenBudget |
| **Optimization** | AdaptiveThrottler, MemoryEstimator | Memory-aware admission control |
| **Reliability** | CircuitBreaker, Coalescer | Failure isolation and request deduplication |
| **Integration** | OllamaClient, InferenceBackend ABC | Async Ollama communication with retry + backoff |

See [docs/architecture.md](docs/architecture.md) for component details and request lifecycle.

## Features

- **OpenAI-compatible API** — `/completions` and `/chat/completions` with the same request/response format
- **SSE streaming** — real-time token-by-token delivery with `stream: true`
- **Response caching** — case-insensitive key normalisation, parameter-aware keys include `max_tokens` and `temperature`
- **Request coalescing** — identical in-flight requests share one backend call
- **4 scheduling policies** — FCFS, SJF (with starvation guard), Priority, Token Budget
- **Circuit breaker** — closed → open → half-open pattern isolates backend failures
- **Memory throttling** — ACCEPT / QUEUE / REJECT admission based on estimated memory pressure
- **API key authentication** — optional Bearer token validation
- **Prometheus metrics** — request latency, token throughput, cache hit rate at `/metrics/prometheus`
- **Docker support** — multi-stage build with docker-compose for Ollama + engine

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — Ollama connectivity, circuit breaker state, queue depth |
| `GET` | `/metrics` | JSON metrics snapshot — cache stats, request counts, memory usage |
| `GET` | `/metrics/prometheus` | Prometheus-format metrics for scraping |
| `POST` | `/completions` | Text completion (streaming or non-streaming) |
| `POST` | `/chat/completions` | Chat completion with message history (streaming or non-streaming) |

See [docs/integration_guide.md](docs/integration_guide.md) for full API reference with request/response schemas.

## Configuration

The engine reads `configs/default.yaml` and supports env var overrides (`OLLAMA_HOST`, `OLLAMA_PORT`).

```yaml
ollama:
  host: localhost
  port: 11434
  timeout_seconds: 300
  retry_count: 3

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: INFO

cache:
  enabled: true
  max_size: 256         # LRU entries
  ttl_seconds: 300      # 5-minute TTL
  mode: exact           # exact | semantic

scheduling:
  policy: fcfs           # fcfs | sjf | priority | token_budget
  max_requests_per_batch: 8
  max_tokens_per_batch: 0  # 0 = unlimited
  drain_delay_seconds: 0.05

memory:
  limit_gb: 14.0        # hard memory cap
  safety_margin: 1.1    # +10% over estimates
```

See [docs/usage_guide.md](docs/usage_guide.md) for tuning recommendations.

## Docker

```bash
docker compose up --build
# Engine on :8000, Ollama on :11434
```

The compose file runs Ollama with a health check and starts the engine once Ollama is ready. Config is mounted read-only from `./configs/`.

## Performance

Benchmarked on Apple M2 Air (16 GB) with 4 local Ollama models:

| Scenario | Result |
|----------|--------|
| Cache hit latency | 1.6–2.8 ms |
| Streaming TTFT | 261–1549 ms (model-dependent) |
| Concurrent burst (4 parallel) | 1.15–1.17× vs sequential |
| Mixed workload (60% cache hit) | 2.15× throughput improvement |
| Sequential cached throughput | 415–426 req/s |

See [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) for full methodology and per-model numbers.

## Caveats

- **Ollama is single-threaded** — concurrent "batching" is fan-out of individual calls, not true GPU batching. Measured speedup is 1.15–1.17× (model loading amortisation), not the 2–4× that true batching provides.
- **Memory estimates are heuristic** — the throttler uses a fixed 0.5 GB per-request estimate. Real memory depends on prompt length, model quantisation, and system state.
- **Cache is in-process** — no persistence across restarts, no shared cache across workers. Use `workers: 1` or accept cold cache on restart.
- **Single backend** — designed for Ollama. The `InferenceBackend` ABC exists for extensibility but only Ollama is implemented.

## Testing

```bash
pip install -e ".[dev]"
python3 -m pytest tests/unit/ --no-cov -q   # 616 tests, ~3 seconds
ruff check src/ tests/                       # linting
mypy src/llm_inference_engine --strict       # type checking
```

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | Component design, request lifecycle, design decisions |
| [Usage Guide](docs/usage_guide.md) | When the engine helps, configuration tuning, troubleshooting |
| [Integration Guide](docs/integration_guide.md) | API reference, streaming, error handling, Docker deployment |
| [Performance Report](docs/PERFORMANCE_REPORT.md) | Benchmark methodology and per-model results |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR workflow.

## License

[Apache 2.0](LICENSE)
