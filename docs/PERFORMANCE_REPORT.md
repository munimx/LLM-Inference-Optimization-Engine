# Performance Report

> **Note:** This report reflects the vLLM-backed architecture (v0.2.0). Benchmarks from the previous Ollama-backed version (v0.1.x) are no longer representative and have been removed.

Benchmarks for this version have not yet been collected. This document will be updated once representative numbers are available from a GPU host running vLLM.

---

## Expected Characteristics

Based on the architecture:

### Cache hits

Redis round-trip on localhost is typically 0.2–1 ms. A cache hit returns a full response in under 5 ms regardless of response length. Hit rate depends on workload repetitiveness.

### Coalescing

When N workers receive the same request concurrently, N-1 of them subscribe to Redis pub/sub and wait for the winner. The additional latency for waiters is the winner's inference time plus one Redis publish/subscribe round-trip (~1 ms).

### Model routing overhead

`ModelRouter.route()` calls `estimate_prompt_tokens()` (a character-based heuristic, no tokeniser loaded). Overhead is under 0.1 ms per request.

### Admission control

`AdaptiveThrottler.check()` reads a Python float (the cached KV-cache metric). It is synchronous with no I/O; overhead is negligible. The background poll task runs every `poll_interval_seconds` (default 5 s).

### Backend pool

`BackendPool.get_healthy_backend()` iterates a Python list (O(n) where n = number of instances, typically 1–4). Overhead is under 0.1 ms.

---

## Benchmark Plan

When GPU hardware is available, the following scenarios should be measured:

| Scenario | Metric | Tool |
|----------|--------|------|
| Cache hit rate vs. miss rate | Requests/s, p50/p95 latency | `locust` or `wrk` |
| Cold vs. warm cache | TTFT, end-to-end latency | `httpx` async client |
| Coalescing under concurrent identical requests | Backend call count vs. request count | Custom script |
| Admission control under GPU pressure | 429 rate, latency distribution | Prometheus scrape |
| Multi-instance pool (2 vs. 4 backends) | Throughput, p99 latency | `locust` |
| Streaming TTFT | Time-to-first-token | `httpx` streaming client |

---

## Profiling

To profile the engine locally:

```bash
pip install pyinstrument
python3 -m pyinstrument -m uvicorn llm_inference_engine.api.server:app --port 8000
```

Or add `py-spy` for sampling:

```bash
pip install py-spy
py-spy record -o profile.svg --pid $(pgrep -f uvicorn)
```

