# LLM Inference Optimization Engine

Request scheduling, semantic caching, and speculative decoding middleware for [Ollama](https://ollama.ai/), exposing an OpenAI-compatible HTTP API.

[![CI](https://github.com/munimx/LLM-Inference-Optimization-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/munimx/LLM-Inference-Optimization-Engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

Sits between your application and Ollama. Incoming completions requests are checked against a semantic cache, queued by configurable scheduling policy, dispatched to Ollama in concurrent batches, and returned via per-request futures. A draft-model speculation loop and adaptive memory throttler are available as opt-in layers.

## Architecture

```
POST /completions
       │
       ▼
SemanticCache ──── hit ───────────────────────────▶ response
       │ miss
       ▼
RequestAggregator
       │
       ▼
Scheduler  (per-model RequestQueue + SchedulingPolicy)
       │
       ▼
dispatch_batch()  ── concurrent httpx ──▶  Ollama
       │
       ▼
ResultMapper  (asyncio.Future per request)
       │
       ▼
CompletionResponse
```

See [docs/architecture.md](docs/architecture.md) for component details.

## Setup

```bash
# Requires Ollama running locally (https://ollama.ai)
ollama pull llama3:8b

pip install -e ".[dev]"

python scripts/start_server.py
```

## Usage

```bash
curl -s http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3:8b", "prompt": "Explain KV-cache in one sentence."}' \
  | python -m json.tool
```

Interactive API docs: <http://localhost:8000/docs>

## Configuration

All settings are in `configs/default.yaml`. The key knobs:

| Key | Default | Notes |
|---|---|---|
| `scheduling.policy` | `fcfs` | `fcfs` · `sjf` · `priority` · `token_budget` |
| `scheduling.max_requests_per_batch` | `8` | Requests dispatched per drain cycle |
| `cache.max_size` | `256` | LRU capacity (entries) |
| `cache.ttl_seconds` | `300` | Seconds before a cache entry is treated as a miss |
| `memory.limit_gb` | `14.0` | Hard admission reject threshold (M2 Air default) |
| `ollama.retry_count` | `3` | Retries on Ollama transport errors |
| `ollama.retry_backoff_seconds` | `1.0` | Base for exponential + jitter backoff |

## Development

```bash
# Tests (no Ollama required, ~2 s, 539 tests)
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
