# Integration Guide

## Base URL

All examples assume the engine is running at `http://localhost:8000`.

---

## Authentication

If `auth.enabled: true` in config, every request must include a Bearer token:

```
Authorization: Bearer <api_key>
```

Requests without a valid key receive:

```json
{"detail": "Unauthorized"}
```
HTTP 401.

---

## Endpoints

### GET /health

Liveness and readiness check.

**Response 200:**

```json
{
  "status": "ok",
  "backend_available": true,
  "version": "0.2.0",
  "details": {
    "healthy_backends": 2
  }
}
```

`backend_available` is `true` when at least one backend in the pool has a closed or half-open circuit.

---

### GET /metrics

JSON snapshot of current engine state.

**Response 200:**

```json
{
  "kv_cache_usage": 0.42,
  "active_requests": 3,
  "healthy_backends": 2,
  "cache_hits": 1500,
  "cache_misses": 300,
  "total_requests": 1800
}
```

---

### GET /metrics/prometheus

Prometheus-format metrics for scraping. Returns `text/plain; version=0.0.4`.

```
# HELP llm_engine_requests_total Total requests
# TYPE llm_engine_requests_total counter
llm_engine_requests_total{status="success"} 1800.0
llm_engine_kv_cache_usage 0.42
llm_engine_healthy_backends 2.0
...
```

---

### POST /completions

OpenAI-compatible text completion.

**Request:**

```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "prompt": "Explain quicksort in one paragraph.",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": [],
  "stream": false,
  "priority": 0,
  "timeout_seconds": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | vLLM model ID. Empty string triggers automatic model routing. |
| `prompt` | string | required | Text prompt. Must not be empty. |
| `max_tokens` | int | `256` | Max tokens to generate (1–32768). |
| `temperature` | float | `0.7` | Sampling temperature (0.0–2.0). |
| `top_p` | float | `0.9` | Top-p nucleus sampling (0.0–1.0). |
| `stop` | list[string] | `[]` | Stop sequences. |
| `stream` | bool | `false` | Stream tokens via SSE. |
| `priority` | int | `0` | Request priority 0–10 (informational; not used for scheduling). |
| `timeout_seconds` | float\|null | `null` | Per-request timeout (1–600 s). |

**Response 200:**

```json
{
  "id": "req-a1b2c3",
  "object": "text_completion",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "index": 0,
      "text": "Quicksort is a divide-and-conquer algorithm...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 87,
    "total_tokens": 97
  },
  "latency_ms": 342.1
}
```

---

### POST /chat/completions

OpenAI-compatible chat completion.

**Request:**

```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quicksort."}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": [],
  "stream": false,
  "priority": 0,
  "timeout_seconds": null
}
```

`messages` is a list of `{"role": "system"|"user"|"assistant", "content": "..."}`. Must contain at least one message.

**Response 200:**

```json
{
  "id": "req-d4e5f6",
  "object": "chat.completion",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quicksort is a divide-and-conquer algorithm..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 87,
    "total_tokens": 109
  },
  "latency_ms": 398.5
}
```

---

## Streaming

Set `"stream": true` to receive tokens as Server-Sent Events. The response content type is `text/event-stream`.

Each event is a JSON object followed by `\n\n`:

```
data: {"id":"req-a1b2c3","choices":[{"index":0,"text":"Quick","finish_reason":null}]}

data: {"id":"req-a1b2c3","choices":[{"index":0,"text":"sort","finish_reason":null}]}

data: {"id":"req-a1b2c3","choices":[{"index":0,"text":" is","finish_reason":null}]}

data: [DONE]
```

The final `data: [DONE]` signals end of stream.

**Caching note:** Streaming responses are not cached. The cache and coalescer are only applied to non-streaming requests.

### curl example

```bash
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.2", "prompt": "Tell me a story", "stream": true}' \
  --no-buffer
```

### Python example

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/completions",
    json={"model": "mistralai/Mistral-7B-Instruct-v0.2", "prompt": "Tell me a story", "stream": True},
    timeout=120,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            print(chunk["choices"][0]["text"], end="", flush=True)
```

---

## Model Routing

If you leave `model` empty (or rely on the engine to route), the engine applies the following rules automatically:

1. `estimate_prompt_tokens(prompt) < fast_model_token_threshold` → `fast_model`
2. Otherwise → `large_model`

To opt out of routing and target a specific model, set `model` explicitly.

---

## Error Responses

All errors return a JSON body:

```json
{
  "error": "short description",
  "detail": "longer explanation",
  "request_id": "req-a1b2c3"
}
```

| Status | Cause |
|--------|-------|
| `400` | Invalid request (empty prompt, bad role, etc.) |
| `401` | Missing or invalid Bearer token |
| `422` | Pydantic validation failure (field type/range error) |
| `429` | KV-cache above `hard_limit`; retry after a moment |
| `500` | Unexpected server error |
| `503` | All backends down and fallback chain exhausted |

---

## Docker Deployment

### docker-compose.yml overview

```yaml
services:
  redis:    # Redis 7 Alpine, port 6379
  vllm:     # vLLM OpenAI server, port 8080 (requires NVIDIA GPU)
  engine:   # This engine, port 8000
```

The engine container reads env vars `VLLM_URL` and `REDIS_URL` which are injected by compose.

### Bring up the stack

```bash
export HF_TOKEN=hf_...
export VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

docker compose up --build -d
docker compose logs -f engine
```

### Custom config

Mount a custom config file:

```yaml
# in docker-compose.yml, under engine:
volumes:
  - ./my-configs:/app/configs:ro
```

### Running vLLM separately

Comment out the `vllm:` service and set `VLLM_URL` to your existing vLLM endpoint:

```bash
VLLM_URL=http://192.168.1.50:8080 docker compose up engine redis
```

### Multi-instance pool

To run multiple vLLM instances, add them to `configs/default.yaml`:

```yaml
vllm:
  instances:
    - url: "http://vllm-1:8080"
    - url: "http://vllm-2:8080"
    - url: "http://vllm-3:8080"
```

And add corresponding services in `docker-compose.yml`.

---

## OpenAI SDK Compatibility

The engine implements the same paths and response shapes as the OpenAI API. You can point the OpenAI Python SDK at it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="your-key-if-auth-enabled",
)

response = client.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    prompt="Explain quicksort",
    max_tokens=256,
)
print(response.choices[0].text)
```

