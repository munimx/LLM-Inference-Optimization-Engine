# LLM Inference Optimization Engine

OpenAI-compatible inference middleware for [vLLM](https://github.com/vllm-project/vllm).

Sits between your application and one or more vLLM instances. Each request flows through Redis-backed response caching, cross-worker request coalescing, token-count-based model routing, and KV-cache-pressure-aware admission control before being dispatched to the backend pool.

## Quick Start

```bash
# Prerequisites: Python 3.11+, vLLM on localhost:8080, Redis on localhost:6379
pip install -e .
uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000
```

```bash
# Text completion
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.2", "prompt": "Explain quicksort", "max_tokens": 128}'

# Chat completion (streaming)
curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.2", "messages": [{"role": "user", "content": "Explain quicksort"}], "stream": true}'
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────────────────┐     ┌────────────┐
│ Application │────▶│  LLM Inference Optimization Engine                           │────▶│  vLLM(s)   │
│             │◀────│  ModelRouter → Cache → Coalescer → Throttler → BackendPool   │◀────│            │
└─────────────┘     └──────────────────────────────────────────────────────────────┘     └────────────┘
                                              │   ▲
                                              ▼   │
                                           ┌────────┐
                                           │ Redis  │
                                           └────────┘
```

| Layer | Component | Purpose |
|-------|-----------|---------|
| **API** | FastAPI server, Pydantic models | OpenAI-compatible HTTP endpoints |
| **Routing** | ModelRouter, FallbackRouter | Token-count-based fast/large model selection; stale-cache fallback |
| **Cache** | RedisCache | Redis-backed LRU response cache with TTL |
| **Coalescing** | RequestCoalescer | Cross-worker deduplication via Redis SET NX + pub/sub |
| **Admission** | AdaptiveThrottler | ACCEPT / QUEUE / REJECT based on live `vllm:kv_cache_usage_perc` |
| **Reliability** | BackendPool, CircuitBreaker | Round-robin pool with per-backend circuit breaker |
| **Integration** | VLLMBackend | Async httpx client targeting vLLM's OpenAI-compatible API |

See [docs/architecture.md](docs/architecture.md) for component details and the full request lifecycle.

## Features

- **OpenAI-compatible API** — `/completions` and `/chat/completions` with the same request/response schema used by the OpenAI SDK
- **SSE streaming** — real-time token-by-token delivery with `"stream": true`
- **Redis response cache** — LRU eviction + TTL; shared across all workers and replicas
- **Cross-worker coalescing** — identical concurrent requests share one backend call via Redis pub/sub
- **Automatic model routing** — short prompts go to the fast model, long prompts to the large model; explicit `model` field overrides routing
- **KV-cache admission control** — polls `vllm:kv_cache_usage_perc`; queues or rejects when GPU memory is under pressure
- **Backend pool + circuit breaker** — round-robin across multiple vLLM instances; open circuits are skipped
- **Fallback chain** — on full pool failure: fallback model → stale cache → 503
- **Prometheus metrics** — latency histograms, token counters, KV-cache gauge, healthy backend count at `/metrics/prometheus`
- **API key authentication** — optional Bearer token validation via `auth.enabled`
- **Docker support** — compose file brings up Redis, vLLM, and the engine with health checks

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness/readiness — backend availability, version |
| `GET` | `/metrics` | JSON snapshot — KV-cache usage, healthy backends, cache hit/miss |
| `GET` | `/metrics/prometheus` | Prometheus-format metrics for scraping |
| `POST` | `/completions` | Text completion (streaming or non-streaming) |
| `POST` | `/chat/completions` | Chat completion with message history (streaming or non-streaming) |

See [docs/integration_guide.md](docs/integration_guide.md) for full API reference with request/response schemas.

## Configuration

The engine reads `configs/default.yaml`. Two env vars override the most common deployment settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_URL` | `http://localhost:8080` | vLLM base URL (overrides `vllm.instances[0].url`) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

```yaml
vllm:
  instances:
    - url: "http://localhost:8080"   # add more for multi-instance pools
  timeout_seconds: 120
  retry_count: 2

redis:
  url: "redis://localhost:6379/0"

cache:
  enabled: true
  max_size: 256       # LRU eviction after this many entries
  ttl_seconds: 300.0  # entries older than this are treated as misses

admission_control:
  soft_limit: 0.70    # QUEUE above this KV-cache fraction
  hard_limit: 0.90    # REJECT above this KV-cache fraction

model_registry:
  fast_model: "mistralai/Mistral-7B-Instruct-v0.2"
  large_model: "meta-llama/Meta-Llama-3-70B-Instruct"
  fast_model_token_threshold: 512   # prompts shorter than this → fast model
  fallback_model: "mistralai/Mistral-7B-Instruct-v0.2"
```

See [docs/usage_guide.md](docs/usage_guide.md) for all options and tuning recommendations.

## Docker

```bash
# Set your Hugging Face token and model name, then:
HF_TOKEN=hf_... VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2 docker compose up --build
# Engine on :8000, vLLM on :8080, Redis on :6379
```

The compose file starts Redis and vLLM with health checks, then starts the engine once both are healthy.

## Testing

```bash
pip install -e ".[dev]"
python3 -m pytest tests/unit/ -q          # 234 tests
ruff check src/ tests/                    # linting
mypy src/llm_inference_engine --strict    # type checking
```

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | Component design, request lifecycle, design decisions |
| [Usage Guide](docs/usage_guide.md) | Configuration reference, tuning, troubleshooting |
| [Integration Guide](docs/integration_guide.md) | API reference, streaming, error handling, Docker deployment |
| [Performance Report](docs/PERFORMANCE_REPORT.md) | Benchmark methodology and results |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR workflow.

## License

[Apache 2.0](LICENSE)
