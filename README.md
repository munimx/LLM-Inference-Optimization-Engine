# LLM Inference Optimization Engine

Request scheduling, caching, streaming, and inference orchestration middleware for [Ollama](https://ollama.ai/) (with an extensible multi-backend interface), exposing an OpenAI-compatible HTTP API.

[![CI](https://github.com/munimx/LLM-Inference-Optimization-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/munimx/LLM-Inference-Optimization-Engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

Sits between your application and Ollama (or other inference backends). Incoming requests are checked against an exact-match or embedding-based semantic cache, queued by configurable scheduling policy, dispatched to the backend in concurrent batches, and streamed back via SSE or returned as a complete response. Features include API-key authentication, Prometheus metrics, request coalescing, and adaptive memory throttling.

## Architecture

```
POST /completions or /chat/completions
       │
       ▼
  API-Key Auth (optional)
       │
       ▼
RequestCoalescer ── dedup identical in-flight requests
       │
       ▼
ExactMatchCache / EmbeddingCache ── hit ─────────────▶ response
       │ miss
       ▼
RequestAggregator
       │
       ▼
Scheduler  (per-model RequestQueue + SchedulingPolicy)
       │
       ▼
dispatch_batch()  ── concurrent httpx ──▶  Ollama / InferenceBackend
       │
       ▼
ResultMapper  (asyncio.Future per request)
       │
       ▼
CompletionResponse or SSE stream
```

See [docs/architecture.md](docs/architecture.md) for component details.

## Documentation

| Doc | What it covers |
|-----|----------------|
| [integration_guide.md](docs/integration_guide.md) | How to connect your app to the engine (Python, Node.js, OpenAI SDK, Ollama Cloud FAQ) |
| [usage_guide.md](docs/usage_guide.md) | When the engine helps vs hurts, config tuning, per-model characteristics |
| [PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) | Measured benchmark results across all 4 local models |
| [architecture.md](docs/architecture.md) | Component design and data flow |

## Setup

```bash
# Requires Ollama running locally (https://ollama.ai)
ollama pull llama3.1:8b

pip install -e ".[dev]"

python scripts/start_server.py
```

### Docker

```bash
docker compose up        # starts Ollama + engine
# Engine at http://localhost:8000, Ollama at http://localhost:11434
```

## Usage

```bash
curl -s http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Explain KV-cache in one sentence."}' \
  | python -m json.tool
```

Interactive API docs: <http://localhost:8000/docs>

Send the same prompt twice — the second call returns in **~2ms** (cache hit, Ollama not contacted).

See [docs/integration_guide.md](docs/integration_guide.md) to connect your existing app. See [docs/usage_guide.md](docs/usage_guide.md) for performance tuning.

## Configuration

All settings are in `configs/default.yaml`. The key knobs:

| Key | Default | Notes |
|---|---|---|
| `ollama.host` | `localhost` | Overridable via `OLLAMA_HOST` env var |
| `ollama.port` | `11434` | Overridable via `OLLAMA_PORT` env var |

| Key | Default | Notes |
|---|---|---|
| `scheduling.policy` | `fcfs` | `fcfs` · `sjf` · `priority` · `token_budget` |
| `scheduling.max_requests_per_batch` | `8` | Requests dispatched per drain cycle |
| `scheduling.max_queue_depth` | `0` | Max pending requests (0=unlimited); rejects 429 when full |
| `scheduling.circuit_breaker_threshold` | `5` | Consecutive failures before circuit opens |
| `scheduling.circuit_breaker_cooldown_seconds` | `30` | Seconds before circuit half-opens |
| `cache.max_size` | `256` | LRU capacity (entries) |
| `cache.ttl_seconds` | `300` | Seconds before a cache entry is treated as a miss |
| `cache.mode` | `exact` | `exact` or `semantic` (semantic uses Ollama embeddings) |
| `memory.limit_gb` | `14.0` | Hard admission reject threshold (M2 Air default) |
| `auth.enabled` | `false` | Enable API-key authentication |
| `auth.api_keys` | `[]` | List of valid Bearer tokens |
| `ollama.retry_count` | `3` | Retries on Ollama transport errors |
| `ollama.retry_backoff_seconds` | `1.0` | Base for exponential + jitter backoff |

## Key Features

- **SSE Streaming** — `"stream": true` proxies Ollama's token-by-token output via Server-Sent Events, with cache integration and Prometheus instrumentation
- **Chat completions** — `/chat/completions` uses Ollama's native `/api/chat` with structured messages
- **Exact-match cache** — LRU cache keyed on `(model, prompt, max_tokens, temperature)`; case-insensitive and whitespace-normalised. Streaming responses also cached.
- **Semantic cache** — embedding-based similarity matching via Ollama's `/api/embed` (opt-in via `cache.mode: "semantic"`)
- **Memory-based admission control** — adaptive throttler rejects requests with HTTP 503 when estimated memory pressure exceeds configured limit
- **Request coalescing** — identical in-flight chat requests are deduplicated
- **Circuit breaker** — opens after N consecutive Ollama failures, auto-recovers after cooldown
- **Queue limits** — configurable `max_queue_depth` with HTTP 429 backpressure
- **Per-request timeout** — optional `timeout_seconds` on each request (returns 504 on expiry)
- **API-key auth** — optional Bearer-token authentication middleware
- **Prometheus metrics** — scrapable at `GET /metrics/prometheus`, includes streaming latency and token counts
- **Multi-backend interface** — abstract `InferenceBackend` ABC; Ollama adapter included, extensible to vLLM/TGI/llama.cpp
- **Docker deployment** — `docker compose up` with Ollama + engine in one command; non-root container with healthcheck
- **Env var config overrides** — `OLLAMA_HOST` and `OLLAMA_PORT` env vars override YAML config (useful for Docker/K8s)
- **Prompt token counting** — uses Ollama's `prompt_eval_count` with char/4 fallback

## Development

```bash
# Tests (no Ollama required, ~2 s, 600+ tests)
pytest tests/unit/ --no-cov

# Coverage report
pytest tests/unit/ --cov=src/llm_inference_engine --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/llm_inference_engine --strict

# Quantization benchmarks (requires Ollama)
python scripts/run_benchmarks.py --config configs/benchmarks.yaml
```

## Caveats

- **Streaming bypasses batching and memory admission** — `stream: true` requests go directly to Ollama without passing through the scheduler or throttler. Cache lookup/storage and Prometheus metrics are still applied.
- **Batching is concurrent fanout, not true batching** — Ollama's API processes one request at a time. `dispatch_batch()` issues requests concurrently but Ollama serialises them internally.
- **Memory estimates are heuristic** — the throttler uses a fixed per-request estimate (0.5 GB). There is no feedback from Ollama's actual memory usage.

## Roadmap

- [ ] Swap aggregator's `OllamaClient` for `InferenceBackend` interface to enable vLLM/TGI backends at runtime
- [ ] Route streaming requests through scheduler/throttler for full admission control
- [ ] Event-driven batch formation (scheduler.run_loop as background task)
- [ ] Request preemption — high-priority requests can interrupt running batches
- [ ] Re-benchmark with current architecture; add TTFT (time to first token) metrics
- [ ] Cluster mode — multiple engine instances with shared cache (Redis)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
