# Usage Guide

## When Does the Engine Help?

The engine adds the most value when:

- **Many duplicate or near-duplicate requests** — the Redis cache and cross-worker coalescer prevent redundant backend calls. This is common in chatbots (repeated greetings, FAQ answers) or batch pipelines that reprocess the same text.
- **Multiple vLLM instances** — the backend pool distributes load via round-robin and isolates failures with per-backend circuit breakers.
- **Mixed prompt sizes** — the model router automatically sends short prompts to a fast model and long prompts to a large model, improving latency for the majority of requests without manual routing logic.
- **GPU memory pressure** — the throttler reads `vllm:kv_cache_usage_perc` directly from vLLM and queues or rejects requests before OOM errors occur, rather than relying on heuristic memory estimates.

---

## Prerequisites

- Python 3.11+
- A running [vLLM](https://github.com/vllm-project/vllm) instance (GPU recommended)
- A running [Redis](https://redis.io) 7+ instance

---

## Installation

```bash
git clone <repo>
cd llm-inference-optimization-engine
pip install -e .
```

Development dependencies (testing, linting, type checking):

```bash
pip install -e ".[dev]"
```

---

## Running the Engine

### Direct

```bash
uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000
```

Override the two most common settings via env vars:

```bash
VLLM_URL=http://my-vllm:8080 REDIS_URL=redis://my-redis:6379/0 \
  uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
# Set your Hugging Face token and model:
export HF_TOKEN=hf_...
export VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

docker compose up --build
```

The compose file starts three services:
- **redis** — Redis 7 Alpine with a `redis-cli ping` health check
- **vllm** — vLLM OpenAI server (requires NVIDIA GPU on the host)
- **engine** — this engine, starts after both dependencies are healthy

The engine is available on `http://localhost:8000`.

---

## Configuration

All configuration lives in `configs/default.yaml`. Edit it directly or mount a custom file via Docker. The two env var overrides (`VLLM_URL`, `REDIS_URL`) take precedence over the file.

### vLLM (`vllm`)

```yaml
vllm:
  instances:
    - url: "http://localhost:8080"
    # Add more instances for a multi-GPU pool:
    # - url: "http://vllm-2:8080"
  timeout_seconds: 120        # increase for very long generations
  retry_count: 2              # transient network retries per request
  retry_backoff_seconds: 0.5
  health_check_interval_seconds: 15
```

**Multi-instance pools:** Add entries to `instances`. Requests are distributed round-robin; open-circuit backends are skipped automatically.

### Redis (`redis`)

```yaml
redis:
  url: "redis://localhost:6379/0"
  socket_timeout_seconds: 5.0
```

### Cache (`cache`)

```yaml
cache:
  enabled: true
  max_size: 256       # LRU eviction threshold; increase for more cache hits
  ttl_seconds: 300    # set to 0 to disable expiry (not recommended)
```

Cache keys encode `model + prompt + max_tokens + temperature`. Two requests for the same prompt with different `temperature` values are treated as distinct.

**Tuning:** For read-heavy workloads (FAQ bots, batch re-runs), set `max_size` higher (e.g. 1024–4096). For creative or unique prompts, a smaller cache reduces Redis memory with little benefit.

### Admission Control (`admission_control`)

```yaml
admission_control:
  enabled: true
  soft_limit: 0.70    # queue requests above 70% KV-cache usage
  hard_limit: 0.90    # reject requests above 90% KV-cache usage
  poll_interval_seconds: 5.0
```

The throttler polls `vllm:kv_cache_usage_perc` from the vLLM `/metrics` endpoint. Between `soft_limit` and `hard_limit`, requests are queued (awaited in an asyncio loop). Above `hard_limit`, requests are rejected with HTTP 429.

**Tuning:**
- Lower `soft_limit` (e.g. 0.60) to queue earlier and smooth burst traffic.
- Raise `hard_limit` (e.g. 0.95) if 429s are too frequent and your workload is bursty.
- Reduce `poll_interval_seconds` for faster reaction to pressure spikes.

### Model Registry (`model_registry`)

```yaml
model_registry:
  fast_model: "mistralai/Mistral-7B-Instruct-v0.2"
  large_model: "meta-llama/Meta-Llama-3-70B-Instruct"
  fast_model_token_threshold: 512
  fallback_model: "mistralai/Mistral-7B-Instruct-v0.2"
  fallback_cache_similarity_threshold: 0.75
```

Routing rules:
1. If the request sets `model`, that model is used as-is (no routing).
2. If `estimate_prompt_tokens(prompt) < fast_model_token_threshold`, use `fast_model`.
3. Otherwise use `large_model`.

**Tuning:** Adjust `fast_model_token_threshold` based on your token distribution. Set both `fast_model` and `large_model` to the same value to disable routing.

### Circuit Breaker (`circuit_breaker`)

```yaml
circuit_breaker:
  failure_threshold: 5    # consecutive failures before opening
  cooldown_seconds: 30    # how long to wait before probing again
```

### Authentication (`auth`)

```yaml
auth:
  enabled: false
  api_keys: []
```

When `enabled: true`, every request must include `Authorization: Bearer <key>`. Keys are matched against `api_keys`. Requests without a valid key receive HTTP 401.

---

## Troubleshooting

### HTTP 429 Too Many Requests

The throttler is rejecting requests because `vllm:kv_cache_usage_perc >= hard_limit`. Options:
- Increase `admission_control.hard_limit` (e.g. 0.95).
- Reduce request concurrency from the client.
- Add more vLLM instances (scale horizontally).

### HTTP 503 Service Unavailable

All backends have open circuit breakers and the fallback chain was exhausted. Check:
- `GET /health` — inspect `details.healthy_backends`.
- `GET /metrics` — check `healthy_backends`.
- vLLM logs for OOM or crash.

### Cache Not Hitting

- Verify `cache.enabled: true`.
- Ensure `temperature` and `max_tokens` match across requests (they are part of the cache key).
- Check `GET /metrics` — `cache_hits` should increase with repeated identical requests.

### High Latency on Cache Hits

Redis round-trip should be well under 5 ms on localhost. If cache hit latency is high:
- Check Redis host is on the same network as the engine.
- Increase `redis.socket_timeout_seconds` if the connection is flaky.

### Coalescer Not Deduplicating

The coalescer only deduplicates requests with the same `model` and `prompt`. Ensure:
- `model` is identical (including routing — let the engine route, don't set it explicitly per request if using auto-routing).
- Redis is reachable from all engine workers.

---

## Multiple Workers

The engine is designed to run with multiple Uvicorn workers (`--workers N`). All workers share Redis for cache and coalescing. The backend pool is created per-process (each worker has its own `BackendPool` instance pointing to the same vLLM URLs).

```bash
uvicorn llm_inference_engine.api.server:app \
  --host 0.0.0.0 --port 8000 --workers 4
```

---

## Testing

```bash
# Unit tests
python3 -m pytest tests/unit/ -q

# With coverage
python3 -m pytest tests/unit/ --cov=llm_inference_engine --cov-report=term-missing

# Single module
python3 -m pytest tests/unit/api/test_cache.py -v
```

