# Usage Guide

## When This Engine Helps

The engine provides the most benefit when:

- **Repeated prompts are common** — the response cache eliminates redundant inference. At 60% cache hit rate, measured throughput improves 2.15×.
- **Multiple users hit the same model** — request coalescing deduplicates identical in-flight requests.
- **You need request isolation** — the circuit breaker prevents one failing model from blocking all requests.
- **Monitoring matters** — Prometheus metrics expose cache hit rate, request latency, queue depth, and token throughput.

### When It Won't Help Much

- **Every request is unique** — cache and coalescing provide no benefit. You're adding one network hop of latency (~2 ms).
- **Single-user interactive use** — the scheduling and coalescing features are designed for concurrent load.
- **You need true GPU batching** — Ollama processes one request at a time. The engine dispatches concurrently, but Ollama serialises execution. Measured concurrent speedup is 1.15–1.17× (model loading amortisation), not the 2–4× that true batching provides.

## Running the Engine

### Standalone

```bash
pip install -e .
uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up --build
```

The compose file starts Ollama, waits for its health check, then starts the engine. Config is mounted from `./configs/`.

### Verify

```bash
curl http://localhost:8000/health
# {"status": "healthy", "ollama_connected": true, ...}
```

## Configuration Tuning

Edit `configs/default.yaml`. Key parameters:

### Cache

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cache.enabled` | `true` | Toggle caching entirely |
| `cache.max_size` | `256` | Max LRU entries. Increase for workloads with many distinct prompts. |
| `cache.ttl_seconds` | `300` | Time-to-live. Lower for rapidly changing information. |
| `cache.mode` | `exact` | `exact` (string match) or `semantic` (embedding similarity) |

### Scheduling

| Parameter | Default | Effect |
|-----------|---------|--------|
| `scheduling.policy` | `fcfs` | `fcfs`, `sjf`, `priority`, or `token_budget` |
| `scheduling.max_requests_per_batch` | `8` | Max requests dispatched per drain cycle |
| `scheduling.drain_delay_seconds` | `0.05` | Pause between drain cycles. Lower = more responsive, higher = better batching. |

**Policy recommendations:**
- `fcfs` — fair and predictable, good default
- `sjf` — optimises average latency when prompt lengths vary significantly
- `priority` — when some requests genuinely matter more
- `token_budget` — pack short requests together to maximise throughput

### Memory

| Parameter | Default | Effect |
|-----------|---------|--------|
| `memory.limit_gb` | `14.0` | Hard memory cap. Set to ~80% of available RAM. |
| `memory.safety_margin` | `1.1` | Multiplier on estimates (1.1 = +10%). Increase if OOM occurs. |

### Ollama Connection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ollama.host` | `localhost` | Ollama hostname. Override with `OLLAMA_HOST` env var. |
| `ollama.port` | `11434` | Ollama port. Override with `OLLAMA_PORT` env var. |
| `ollama.timeout_seconds` | `300` | Max wait for a single inference call |
| `ollama.retry_count` | `3` | Retries on transient failures |

## Model Considerations

The engine works with any model Ollama has pulled. Considerations:

- **Small models** (phi3, gemma:2b) — fast TTFT (~260 ms), low memory, good for high-throughput workloads
- **Large models** (llama3.1:8b, deepseek-r1:8b) — slower TTFT (~1.5s), better quality, benefit more from caching
- **Quantised models** — lower memory footprint allows more headroom for the memory throttler

Pull models before starting the engine:

```bash
ollama pull llama3.1:8b
ollama pull phi3
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Returns Ollama connectivity, circuit breaker state, and queue depth.

### JSON Metrics

```bash
curl http://localhost:8000/metrics
```

Returns cache hit/miss counts, total requests, active model sessions, memory usage.

### Prometheus

```bash
curl http://localhost:8000/metrics/prometheus
```

Returns metrics in Prometheus text format. Scrape this endpoint with Prometheus or compatible collectors.

Key metrics:
- `llm_request_duration_seconds` — request latency histogram
- `llm_tokens_generated_total` — total tokens produced
- `llm_cache_hits_total` / `llm_cache_misses_total` — cache effectiveness

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/health` returns `ollama_connected: false` | Ollama not running or wrong host/port | Start Ollama, check `OLLAMA_HOST`/`OLLAMA_PORT` |
| Requests rejected with 429 | Memory throttler is rejecting | Increase `memory.limit_gb` or reduce concurrent load |
| Circuit breaker is open | Backend failed repeatedly | Wait for cooldown period, check Ollama logs |
| High latency despite caching | Low cache hit rate | Check `cache.max_size` and `cache.ttl_seconds` |
| OOM kills | Memory estimates too low | Increase `memory.safety_margin`, decrease `memory.limit_gb` |
