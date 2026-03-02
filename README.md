# LLM Inference Optimization Engine

A production-grade LLM inference optimization layer built on top of Ollama for Apple M2 Air.

[![CI](https://github.com/<org>/llm-inference-engine/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## Overview

An intelligent orchestration and optimization layer on top of Ollama:

- **Smart Batching & Scheduling** — FCFS, SJF, Priority, and TokenBudget policies
- **Memory & Capacity Planning** — KV-cache estimation, adaptive admission control
- **OpenAI-compatible REST API** — drop-in replacement for clients using `/completions`
- **Semantic Caching** — LRU + TTL cache eliminates redundant Ollama calls
- **Speculative Decoding** — draft-verify loop for up to 1.35× speedup

## Architecture

```
POST /completions  (FastAPI, port 8000)
       ↓
SemanticCache  ──── hit ────▶  return cached response
       │ miss
       ▓
RequestAggregator
       ↓
Scheduler  (per-model RequestQueue + SchedulingPolicy)
       ↓
dispatch_batch()  ──── concurrent httpx calls ────▶  Ollama
       ↓
ResultMapper  (asyncio.Future per request)
       ↓
CompletionResponse
```

See [docs/architecture.md](docs/architecture.md) for detailed component diagrams.

## Features

| Feature | Status |
|---|---|
| Ollama async client with retry / backoff | ✅ Complete |
| Quantization analysis & benchmarking | ✅ Complete |
| Batching & scheduling (FCFS, SJF, Priority, TokenBudget) | ✅ Complete |
| Memory estimator + adaptive throttler | ✅ Complete |
| OpenAI-compatible REST API (FastAPI) | ✅ Complete |
| Semantic cache (LRU + TTL) | ✅ Complete |
| Speculative decoding engine | ✅ Complete |
| GitHub Actions CI (lint, type-check, tests) | ✅ Complete |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Apple M2 Air recommended (8 GB minimum unified memory)

## Quick Start

```bash
# 1. Pull a model
ollama pull llama3:8b

# 2. Install (editable + dev dependencies)
pip install -e ".[dev]"

# 3. Start the server
python scripts/start_server.py

# 4. Send a request
curl -s http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3:8b", "prompt": "Hello, world!"}' | python -m json.tool
```

The Swagger UI is available at <http://localhost:8000/docs>.

## Configuration

All settings live in `configs/default.yaml`:

```yaml
ollama:
  host: "localhost"
  port: 11434
  timeout_seconds: 30

server:
  host: "0.0.0.0"
  port: 8000

scheduling:
  policy: "token_budget"     # fcfs | sjf | priority | token_budget
  max_batch_size: 8
  token_budget: 512

memory:
  limit_gb: 14.0             # soft threshold = 85 % of this
  safety_margin: 1.10

cache:
  enabled: true
  max_size: 1000
  ttl_seconds: 300
```

Override any value via environment variable using `LLM_ENGINE_<KEY>` notation.

## Performance Tuning

| Goal | Recommendation |
|---|---|
| Maximise throughput | Use `TokenBudget` policy, increase `max_batch_size` |
| Minimise P50 latency | Use `SJF` policy, keep batches small |
| Reduce memory pressure | Lower `memory.limit_gb`, enable throttler |
| Improve cache hit rate | Increase `cache.ttl_seconds`, reuse prompts |
| Faster generation | Enable speculative decoding with a small draft model |

See [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) for benchmark numbers.

## Project Status

| Phase | Branch | Status |
|---|---|---|
| Phase 1 — Ollama Integration | `phase-1-ollama-integration` | ✅ Complete |
| Phase 2 — Quantization Analysis | `phase-2-quantization-analysis` | ✅ Complete |
| Phase 3 — Batching & Scheduling | `phase-3-batching-scheduling` | ✅ Complete |
| Phase 4 — Memory & Capacity | `phase-4-memory-capacity` | ✅ Complete |
| Phase 5 — API Layer | `phase-5-api-layer` | ✅ Complete |
| Phase 6 — Speculative Decoding | `phase-6-speculative-decoding` | ✅ Complete |
| Phase 7 — Production Hardening | `phase-7-production-hardening` | ✅ Complete |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Performance Report](docs/PERFORMANCE_REPORT.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Ollama Integration Guide](docs/ollama_integration.md)
- [Configuration Guide](docs/configuration.md)
- [Quantization Analysis Guide](docs/quantization_analysis.md)
- [Latest Benchmark Results](docs/benchmark_results.md)

## Development

```bash
# Run unit tests (no Ollama required)
pytest tests/unit/ --no-cov

# Run with coverage
pytest tests/unit/ --cov=src/llm_inference_engine --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/llm_inference_engine --strict

# Run quantization benchmarks (requires Ollama)
python scripts/run_benchmarks.py --config configs/benchmarks.yaml
```

## License

Apache 2.0

## Author

Built as a FAANG-ready portfolio project demonstrating systems design and inference optimization expertise.
