# Integration Guide

## API Reference

Base URL: `http://localhost:8000` (default)

### Authentication

Authentication is optional. When enabled, pass your API key as a Bearer token:

```bash
curl http://localhost:8000/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Hello"}'
```

Unauthenticated requests receive `401 Unauthorized` when auth is enabled.

---

### POST /completions

Generate a text completion.

**Request:**

```json
{
  "model": "llama3.1:8b",
  "prompt": "Explain quicksort in one paragraph",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Ollama model name |
| `prompt` | string | yes | — | Input text |
| `max_tokens` | int | no | 256 | Maximum tokens to generate |
| `temperature` | float | no | 0.7 | Sampling temperature (0.0–2.0) |
| `stream` | bool | no | false | Enable SSE streaming |

**Response (non-streaming):**

```json
{
  "text": "Quicksort is a divide-and-conquer algorithm...",
  "model": "llama3.1:8b",
  "tokens_generated": 87,
  "processing_time": 2.41
}
```

**Response (streaming):**

```
data: {"text": "Quick", "done": false}
data: {"text": "sort", "done": false}
data: {"text": " is", "done": false}
...
data: {"text": "", "done": true, "tokens_generated": 87, "processing_time": 2.41}
```

Each SSE event is a JSON object. The final event has `"done": true` with summary fields.

---

### POST /chat/completions

Generate a chat completion with message history.

**Request:**

```json
{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Ollama model name |
| `messages` | array | yes | — | Array of `{role, content}` objects |
| `max_tokens` | int | no | 256 | Maximum tokens to generate |
| `temperature` | float | no | 0.7 | Sampling temperature |
| `stream` | bool | no | false | Enable SSE streaming |

**Response:** Same format as `/completions`.

---

### GET /health

**Response:**

```json
{
  "status": "healthy",
  "ollama_connected": true,
  "circuit_breaker_state": "closed",
  "queue_depth": 0
}
```

---

### GET /metrics

**Response:**

```json
{
  "cache_hits": 142,
  "cache_misses": 58,
  "total_requests": 200,
  "active_models": ["llama3.1:8b"],
  "memory_usage_gb": 4.2
}
```

---

### GET /metrics/prometheus

Returns metrics in Prometheus exposition format:

```
# HELP llm_request_duration_seconds Request processing duration
# TYPE llm_request_duration_seconds histogram
llm_request_duration_seconds_bucket{le="0.1"} 142
...
```

## Error Handling

| Status | Meaning | When |
|--------|---------|------|
| `200` | Success | Request completed |
| `400` | Bad Request | Invalid model name, missing required fields |
| `401` | Unauthorized | Auth enabled and no/invalid Bearer token |
| `429` | Too Many Requests | Memory throttler rejected the request |
| `503` | Service Unavailable | Circuit breaker is open or Ollama unreachable |
| `500` | Internal Error | Unexpected failure during inference |

Error responses include a JSON body:

```json
{
  "detail": "Circuit breaker is open — backend unavailable"
}
```

## Streaming Integration

### Python (httpx)

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/completions", json={
    "model": "llama3.1:8b",
    "prompt": "Write a haiku about coding",
    "stream": True
}) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "llama3.1:8b",
    prompt: "Write a haiku about coding",
    stream: true,
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.slice(6));
      process.stdout.write(data.text || "");
    }
  }
}
```

### curl

```bash
curl -N http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "Write a haiku", "stream": true}'
```

The `-N` flag disables buffering for real-time output.

## Docker Deployment

### docker-compose (recommended)

```bash
docker compose up --build
```

This starts:
- **Ollama** on port 11434 with a health check
- **Engine** on port 8000, depends on Ollama health

### Standalone Docker

```bash
docker build -t llm-engine .
docker run -p 8000:8000 \
  -e OLLAMA_HOST=host.docker.internal \
  llm-engine
```

Use `host.docker.internal` when Ollama runs on the host machine outside Docker.

### Configuration in Docker

The compose file mounts `./configs/` read-only into the container. Edit `configs/default.yaml` on the host and restart the container to apply changes.

Environment variable overrides:
- `OLLAMA_HOST` — Ollama hostname (default: `localhost`)
- `OLLAMA_PORT` — Ollama port (default: `11434`)

## Ollama Model Management

### Pulling Models

```bash
ollama pull llama3.1:8b
ollama pull phi3
ollama pull gemma:2b
```

### Listing Available Models

```bash
ollama list
```

### Model Memory Requirements

| Model | Parameters | Quantisation | Approximate RAM |
|-------|-----------|-------------|-----------------|
| phi3 | 3.8B | Q4_K_M | ~2.5 GB |
| gemma:2b | 2B | Q4_K_M | ~1.8 GB |
| llama3.1:8b | 8B | Q4_K_M | ~5.0 GB |
| deepseek-r1:8b | 8B | Q4_K_M | ~5.0 GB |

Set `memory.limit_gb` in config to at least the size of your largest model plus 2 GB headroom.

### Quantisation Levels

Lower quantisation = less RAM, slightly lower quality:

| Level | Bits | Quality | Speed |
|-------|------|---------|-------|
| Q4_K_M | 4-bit | Good | Fast |
| Q5_K_M | 5-bit | Better | Moderate |
| Q6_K | 6-bit | High | Slower |
| Q8_0 | 8-bit | Near-original | Slowest |

Most Ollama models default to Q4_K_M, which is a good balance for this engine's use case.
