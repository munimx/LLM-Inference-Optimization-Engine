# Usage Guide — LLM Inference Optimization Engine

This guide covers when the engine helps, when it doesn't, and how to configure it for maximum performance on an Apple M2 Air running Ollama locally.

---

## When to Use This Engine

### ✅ Strong fit: repeated or similar prompts

The cache is keyed on `(model, normalized_prompt)` where the prompt includes parameter suffixes (`max_tokens`, `temperature`) so that different generation settings produce separate cache entries. Prompts are normalized via `.lower().strip()` to avoid case/whitespace misses.

Examples:
- FAQ chatbot (20–50 unique questions, many repeat)
- Document Q&A over a fixed corpus (same questions from different users)
- Coding assistant with template prompts (`"Fix this Python: ..."`)
- Testing/evaluation harness re-running the same eval prompts

At 50% cache hit rate the engine delivers ~2x throughput. At 70%+ it's 3x+.

### ✅ Good fit: multiple concurrent users

The scheduler batches concurrent requests and dispatches them together to Ollama. Measured speedup on M2 Air:

| Concurrency | Model | Wall-time speedup |
|---|---|---|
| 4 parallel | `mistral:7b` | 1.21x |
| 4 parallel | `llama3.1:8b` | 1.44x |

Combine with cache hits and speedups stack: a concurrent workload with 50% hit rate on `llama3.1:8b` achieves 2.5–3x effective throughput.

### ✅ Good fit: long-running sessions

Cache TTL defaults to 300 seconds. In a session where the user returns to the same topics, early responses are served from cache for the duration of the session.

---

## When NOT to Use This Engine

### ❌ Single unique cold requests

If every prompt is different and never repeats, the engine adds 0–25% overhead over direct Ollama:

| Model | Engine cold overhead |
|---|---|
| `llama3.1:8b` | ~1% (negligible) |
| `mistral:7b` | ~24% |

For `llama3.1:8b` the overhead is noise. For `mistral:7b` with unique prompts, the engine is measurably slower.

### ❌ Very short generations

With `max_tokens=20`, model generation time is 0.5–3s. Engine scheduling overhead (~150–300ms) becomes a significant fraction. Use Ollama directly for one-shot short-answer queries with no repetition.

### ❌ deepseek-r1:7b without caching

deepseek-r1 generates internal reasoning chains (`<think>...</think>`) before answering. A simple factual question takes 40–120 seconds. The engine's scheduling overhead is irrelevant at this scale; batching doesn't help either because Ollama processes one request at a time. The **only** engine feature that helps deepseek-r1 is the cache — a cache hit returns in 1.7ms regardless of how long the original generation took.

---

## Installed Models — Characteristics

| Model | Size | tok/s | Best for |
|---|---|---|---|
| `phi3:latest` | 2.2 GB | ~30–80 (variable) | Quick experimentation; not latency-sensitive |
| `mistral:7b` | 4.4 GB | ~35–40 | Instruction following, summarisation |
| `llama3.1:8b` | 4.9 GB | ~25–30 | General purpose; lowest cold-path overhead |
| `deepseek-r1:7b` | 4.7 GB | ~9 | Reasoning tasks where quality > speed |

On 16 GB M2 Air you can run one model at a time without memory pressure. The engine's admission control is configured at `memory.limit_gb=14.0`.

---

## Configuration for Maximum Performance

All settings in `configs/default.yaml`.

### Cache

```yaml
cache:
  enabled: true
  max_size: 256      # Increase for larger prompt pools; each entry ~KB
  ttl_seconds: 300   # Increase for long sessions; decrease for dynamic content
```

**For a FAQ chatbot with 500 unique questions**: set `max_size: 600` so all questions fit in cache without eviction.

**For a real-time chat with high variation**: set `ttl_seconds: 60` to avoid stale responses, or disable cache entirely.

### Scheduling

```yaml
scheduling:
  policy: fcfs                    # fcfs | sjf | priority | token_budget
  max_requests_per_batch: 8       # Increase for higher concurrency (>4 users)
```

| Policy | Best for |
|---|---|
| `fcfs` | Default; fair, predictable |
| `sjf` | Minimise P50 latency (short jobs first) |
| `priority` | Multi-tier API (premium users get lower latency) |
| `token_budget` | Maximise throughput when request lengths vary widely |

**For a single-user desktop assistant**: `fcfs` with `max_requests_per_batch: 4` is sufficient.  
**For a multi-user local server** (family/team): `sjf` with `max_requests_per_batch: 8`.

### Memory

```yaml
memory:
  limit_gb: 14.0     # M2 Air 16GB — leaves 2GB headroom for OS
  safety_margin: 1.1
```

If running other apps alongside Ollama, lower to `12.0` to prevent Ollama from being swapped out.

---

## Quick Start

```bash
# 1. Pull your preferred model
ollama pull llama3.1:8b

# 2. Start the engine
python scripts/start_server.py

# 3. Send a request (OpenAI-compatible)
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "What is a CPU cache?"}'

# 4. Send the same request again — served from cache in ~2ms
curl http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "prompt": "What is a CPU cache?"}'
```

---

## Benchmarking Your Own Workload

```bash
# Run built-in 6-scenario benchmark
python scripts/start_server.py &
python scripts/run_integration_benchmarks.py
# Outputs: docs/benchmark_results.json, docs/PERFORMANCE_REPORT.md
```

The benchmark covers:
- **S1**: Short generation (20 tokens) — worst-case overhead
- **S2**: Medium generation (80 tokens) — realistic workload
- **S3**: Cache hit latency — exact repeat (10 runs)
- **S4**: Mixed workload at 60% hit rate — realistic throughput speedup
- **S5**: Concurrent burst (4 parallel) — batching benefit
- **S6**: Sequential throughput (10 requests) — req/sec comparison

---

## Summary

| Scenario | Engine vs Direct Ollama |
|---|---|
| Cache hit (exact repeat) | **1–2ms** vs seconds — 500–10,000x faster |
| Mixed workload, 60% hit rate | **~2x** wall-time speedup |
| 4 concurrent users | **1.2–1.4x** |
| Single unique cold request (`llama3.1:8b`) | **≈ same** (~1% overhead) |
| Single unique cold request (`mistral:7b`) | **−24%** slower |
| deepseek-r1 (no cache) | **same or slower** |

**Bottom line**: Use the engine whenever your workload has any prompt repetition. For pure unique-prompt single-user use, route directly to Ollama.
