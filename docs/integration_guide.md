# Application Integration Guide

The engine is an HTTP middleware layer that sits between your application and Ollama. Your application sends requests to the engine instead of Ollama directly; the engine handles caching, scheduling, batching, and memory throttling transparently.

```
Your App  →  POST localhost:8000/completions  →  Engine  →  Ollama :11434
                                                    ↑
                                              cache / scheduler
                                              (Ollama never called on cache hit)
```

---

## Prerequisites

1. **Ollama running locally** with at least one model pulled:
   ```bash
   ollama serve           # starts on localhost:11434
   ollama pull llama3.1:8b
   ```

2. **Engine running:**
   ```bash
   pip install -e ".[dev]"
   python scripts/start_server.py   # starts on localhost:8000
   ```

3. **Verify both are up:**
   ```bash
   curl -s http://localhost:8000/health | python -m json.tool
   # {"status": "ok", "ollama_available": true, "version": "0.1.0", ...}
   ```

---

## Quick Start — One-URL Migration

If your app currently calls Ollama's native API directly:

**Before (Ollama native):**
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"What is 2+2?","stream":false}'
```

**After (through engine):**
```bash
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","prompt":"What is 2+2?"}'
```

Two differences:
- Port `11434` → `8000`
- Path `/api/generate` + Ollama format → `/completions` + OpenAI-compatible format
- Response field `response` → `choices[0].text`

---

## API Reference

### POST /completions

```json
{
  "model":       "llama3.1:8b",
  "prompt":      "Explain gradient descent in one paragraph.",
  "max_tokens":  256,
  "temperature": 0.7,
  "top_p":       0.9,
  "stop":        [],
  "priority":    0
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | string | required | Any model tag from `ollama list` |
| `prompt` | string | required | Non-empty |
| `max_tokens` | int | 256 | 1–32768 |
| `temperature` | float | 0.7 | 0.0–2.0 |
| `top_p` | float | 0.9 | 0.0–1.0 |
| `stop` | string[] | `[]` | Stop sequences |
| `priority` | int | 0 | 0–10 (higher = scheduled first with `priority` policy) |

**Response:**
```json
{
  "id":      "req_abc123",
  "object":  "text_completion",
  "model":   "llama3.1:8b",
  "choices": [{"index": 0, "text": "Gradient descent is...", "finish_reason": "stop"}],
  "usage":   {"prompt_tokens": 0, "completion_tokens": 47, "total_tokens": 47},
  "latency_ms": 1423.7
}
```

---

### POST /chat/completions

```json
{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is gradient descent?"},
    {"role": "assistant", "content": "It's an optimization algorithm."},
    {"role": "user",      "content": "Give me a concrete example."}
  ],
  "max_tokens":  512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id":    "req_def456",
  "object": "chat.completion",
  "model": "llama3.1:8b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Consider training a neural network..."},
    "finish_reason": "stop"
  }],
  "usage":      {"prompt_tokens": 0, "completion_tokens": 89, "total_tokens": 89},
  "latency_ms": 2841.3
}
```

---

## Integration Patterns

### curl (any shell script or CI pipeline)

```bash
ENGINE="http://localhost:8000"

# Text completion
curl -s "$ENGINE/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "prompt": "Summarise the following in one sentence: '"$TEXT"'",
    "max_tokens": 64
  }' | python -m json.tool

# Chat completion
curl -s "$ENGINE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }' | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

### Python — `requests`

```python
import requests

ENGINE = "http://localhost:8000"

def complete(prompt: str, model: str = "llama3.1:8b", max_tokens: int = 256) -> str:
    resp = requests.post(
        f"{ENGINE}/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]

def chat(messages: list[dict], model: str = "llama3.1:8b") -> str:
    resp = requests.post(
        f"{ENGINE}/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": 512},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# Usage
answer = complete("Explain KV-cache in one sentence.")
reply = chat([{"role": "user", "content": "What is attention in transformers?"}])
```

---

### Python — `httpx` (async)

```python
import httpx
import asyncio

ENGINE = "http://localhost:8000"

async def complete(prompt: str, model: str = "llama3.1:8b") -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{ENGINE}/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 256},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

async def batch_complete(prompts: list[str], model: str = "llama3.1:8b") -> list[str]:
    """Send multiple prompts concurrently — engine batches them automatically."""
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [
            client.post(
                f"{ENGINE}/completions",
                json={"model": model, "prompt": p, "max_tokens": 256},
            )
            for p in prompts
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json()["choices"][0]["text"] for r in responses]
```

---

### Python — OpenAI SDK

The engine's `/completions` and `/chat/completions` endpoints use the same JSON shape as OpenAI's API. You can use the `openai` Python package by pointing it at the engine:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",  # engine, not api.openai.com
    api_key="not-needed",              # required by SDK but ignored by engine
)

# Text completion
resp = client.completions.create(
    model="llama3.1:8b",
    prompt="Explain gradient descent.",
    max_tokens=256,
)
print(resp.choices[0].text)

# Chat completion
resp = client.chat.completions.create(
    model="mistral:7b",
    messages=[{"role": "user", "content": "What is backpropagation?"}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

> **Note:** Token usage fields (`prompt_tokens`) are returned as 0 — the engine does not currently tokenise prompts server-side. `completion_tokens` is accurate.

---

### Node.js — `fetch`

```javascript
const ENGINE = "http://localhost:8000";

async function complete(prompt, model = "llama3.1:8b", maxTokens = 256) {
  const res = await fetch(`${ENGINE}/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt, max_tokens: maxTokens }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!res.ok) throw new Error(`Engine error: ${res.status}`);
  const data = await res.json();
  return data.choices[0].text;
}

async function chat(messages, model = "llama3.1:8b") {
  const res = await fetch(`${ENGINE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, max_tokens: 512 }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!res.ok) throw new Error(`Engine error: ${res.status}`);
  const data = await res.json();
  return data.choices[0].message.content;
}

// Usage
const answer = await complete("What is a transformer model?");
const reply  = await chat([{ role: "user", content: "Explain attention." }]);
```

---

### Node.js — OpenAI SDK

```bash
npm install openai
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000",
  apiKey: "not-needed",
});

const resp = await client.chat.completions.create({
  model: "llama3.1:8b",
  messages: [{ role: "user", content: "What is RLHF?" }],
  max_tokens: 256,
});

console.log(resp.choices[0].message.content);
```

---

## Ollama Cloud and Remote Ollama

### Does this engine work with Ollama Cloud?

**Ollama.com is a model library** (browse and `ollama pull` models). It is not a hosted inference API. There is no "Ollama Cloud" service that runs inference — all Ollama inference runs locally on your machine. This engine is designed for exactly that local setup.

If you want cloud-hosted LLM inference (pay-per-token), you would use OpenAI, Anthropic, Groq, Together, or similar providers directly. Those services already expose OpenAI-compatible APIs — they do not benefit from this engine's local caching and scheduling layer.

### Can I point the engine at a remote Ollama instance?

Yes. The engine talks to Ollama over HTTP. If Ollama is running on another machine (a LAN server, a VPS, or another Mac on the same network), set `ollama.host` in `configs/default.yaml`:

```yaml
ollama:
  host: 192.168.1.50     # IP or hostname of the machine running Ollama
  port: 11434
  timeout_seconds: 300
  retry_backoff_seconds: 2.0   # increase for LAN latency
```

Then run the engine on any machine that can reach that host. Your application always talks to the engine (`localhost:8000`).

```
App (machine A)  →  Engine (machine A, :8000)  →  Ollama (machine B, :11434)
```

> **Ollama firewall note:** By default Ollama only binds to `127.0.0.1`. To allow remote connections, start it with `OLLAMA_HOST=0.0.0.0 ollama serve` on the Ollama machine.

---

## Health Checks and Observability

### GET /health

Use this in your load balancer, Docker healthcheck, or Kubernetes liveness probe:

```bash
curl -sf http://localhost:8000/health
```

```json
{
  "status": "ok",          // "ok" | "degraded" (Ollama unreachable)
  "ollama_available": true,
  "version": "0.1.0",
  "details": {"pending_requests": 0}
}
```

Exit code is 0 on 200, non-zero on any error — usable directly in shell scripts.

**Docker healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1
```

### GET /metrics

```bash
curl -s http://localhost:8000/metrics | python -m json.tool
```

```json
{
  "committed_memory_gb": 9.1,
  "available_memory_gb": 4.9,
  "memory_limit_gb": 14.0,
  "active_requests": 2,
  "cache_hits": 847,
  "cache_misses": 312,
  "total_requests": 1159
}
```

Use `cache_hits / total_requests` to monitor cache hit rate. A rate below 30% in your workload means caching is not helping much — review whether your prompts vary too much.

---

## Error Handling

| HTTP Status | Meaning | Action |
|-------------|---------|--------|
| `200` | Success | Read `choices[0].text` or `choices[0].message.content` |
| `422` | Validation error | Fix request body (empty prompt, bad model name format, etc.) |
| `503` | Ollama unreachable or memory limit exceeded | Retry with backoff; check `GET /health` |

```python
import requests
from requests.exceptions import Timeout, ConnectionError

def safe_complete(prompt: str, model: str = "llama3.1:8b") -> str | None:
    try:
        resp = requests.post(
            "http://localhost:8000/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 256},
            timeout=120,
        )
        if resp.status_code == 503:
            print("Engine overloaded or Ollama down — retry later")
            return None
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]
    except Timeout:
        print("Request timed out — consider increasing timeout or reducing max_tokens")
        return None
    except ConnectionError:
        print("Engine not running — start with: python scripts/start_server.py")
        return None
```

---

## Production Deployment

### Running the engine as a persistent service (macOS launchd)

Create `~/Library/LaunchAgents/com.llm-engine.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.llm-engine</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/path/to/engine/scripts/start_server.py</string>
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/llm-engine.log</string>
  <key>StandardErrorPath</key> <string>/tmp/llm-engine.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.llm-engine.plist
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1
CMD ["python", "scripts/start_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

Point `ollama.host` in `configs/default.yaml` at the Docker host IP (or use `host.docker.internal` on macOS).

### Timeout recommendations

| Model | Suggested `timeout` in client | Notes |
|-------|-------------------------------|-------|
| phi3:latest | 30s | Fast, small model |
| mistral:7b | 60s | Standard generation |
| llama3.1:8b | 90s | Longer context windows |
| deepseek-r1:7b | 300s | Reasoning model — emits chain-of-thought |

---

## Differences from Direct Ollama

| | Direct Ollama | Engine |
|--|---------------|--------|
| Port | 11434 | 8000 |
| Endpoint | `/api/generate` · `/api/chat` | `/completions` · `/chat/completions` |
| Request format | Ollama-native | OpenAI-compatible |
| Response field | `response` | `choices[0].text` |
| Streaming | ✅ Supported | ❌ Not yet implemented (`stream: true` is accepted but ignored) |
| Cache | ❌ None | ✅ LRU exact-match (2ms hit latency) |
| Batching | ❌ None | ✅ Up to 8 concurrent requests per drain cycle |
| Scheduling | ❌ None | ✅ FCFS / SJF / Priority / Token-budget |
| Memory guard | ❌ None | ✅ 503 on memory limit breach |
| Model tagging | Any valid Ollama tag | Same — use `ollama list` tags directly |

The engine does not expose Ollama's `/api/tags`, `/api/show`, or `/api/pull` endpoints. Use the Ollama CLI for model management.
