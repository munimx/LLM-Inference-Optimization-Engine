# Performance Report — Integration Benchmarks

**Hardware**: Apple M2 Air, 16 GB unified memory  
**Ollama version**: local  
**Benchmark date**: 2026-03-02  
**Methodology**: 1 warmup run discarded, 3 measured runs averaged. All 4 models installed locally (Q4_K_M quantization).

---

## Summary

| Model | Direct Ollama (ms) | Engine Cold (ms) | Cache Hit (ms) | Concurrent Speedup |
|---|---|---|---|---|
| `phi3:latest` (3.8B) | ~80 warm¹ | ~770 avg | **2** | varies² |
| `mistral:7b` (7B) | 127 | 426 | **3** | 1.11x |
| `llama3.1:8b` (8B) | 262 | 434 | **3** | 1.19x |
| `deepseek-r1:7b` (7B reasoning) | ~50 000 | ~107 000 | **2** | N/A³ |

¹ phi3 first request included model cold-load (~3100ms); subsequent warm requests averaged ~80ms.  
² phi3 concurrent shows high variance — model loads/unloads between model switches on 16GB.  
³ deepseek-r1 generates chain-of-thought reasoning tokens; concurrent test exceeded 5-minute timeout.

---

## Key Findings

### 1. Cache Hit: 2–3 ms for all models

Repeated identical prompts are served entirely from the in-process LRU cache, bypassing Ollama completely. For `mistral:7b` (127ms warm baseline), this is a **42x speedup**. For `deepseek-r1:7b` (50s baseline), the speedup approaches **25,000x** on cached responses.

Cache miss (cold path) adds ~150–600ms scheduling and dispatch overhead on top of model inference time.

### 2. Engine Cold Path Overhead

The engine adds scheduling → queue → dispatch → result-mapping overhead. For short generations (1–5 tokens), this overhead is proportionally high. For longer generations the ratio improves — the overhead is roughly constant at 150–400ms regardless of generation length.

| Model | Baseline (ms) | Engine cold (ms) | Overhead (ms) |
|---|---|---|---|
| `mistral:7b` | 127 | 426 | ~300 |
| `llama3.1:8b` | 262 | 434 | ~170 |

### 3. Concurrent Batching (4 parallel requests)

| Model | Sequential estimate (ms) | Concurrent wall (ms) | Speedup |
|---|---|---|---|
| `mistral:7b` — P1 | 572 | 491 | 1.17x |
| `mistral:7b` — P2 | 476 | 438 | 1.08x |
| `mistral:7b` — P3 | 476 | 438 | 1.09x |
| `llama3.1:8b` — P1 | 1068 | 908 | 1.18x |
| `llama3.1:8b` — P2 | 1032 | 880 | 1.17x |
| `llama3.1:8b` — P3 | 1048 | 864 | 1.21x |

Batching provides a consistent ~1.1–1.2x wall-time speedup for 4 concurrent requests on an M2 Air with a single loaded model. Gains are limited by Ollama's own single-threaded inference — the engine's concurrent `httpx` fan-out saturates Ollama's queue, but Ollama processes requests sequentially. The speedup comes from pipelining queue drain, HTTP connection reuse, and result dispatch.

### 4. deepseek-r1:7b — Reasoning Model

deepseek-r1 generates an internal `<think>...</think>` chain before answering, producing hundreds of tokens even for trivial prompts. Measured baseline was ~50s per request. The engine's cache is the only optimization that provides meaningful speedup (2ms hit vs 50s miss = ~25,000x) — scheduling and batching do not help because the bottleneck is generation time.

---

## Per-Model Detail

### phi3:latest

| Prompt | Baseline (ms) | Cold (ms) | Hit (ms) | tok/s |
|---|---|---|---|---|
| "What is 2+2?" (cold model) | 3146 | 266 | 2 | 38.5 |
| "Capital of France?" | 94 | 184 | 2 | 57.6 |
| "Sky colour?" | 69 | 1860 | 2 | 77.0 |

Note: Prompt 1 baseline includes model cold-load. phi3 was first model benchmarked. Warm baseline (P2, P3) is 69–94ms at 57–77 tok/s.

### mistral:7b

| Prompt | Baseline (ms) | Cold (ms) | Hit (ms) | tok/s |
|---|---|---|---|---|
| "What is 2+2?" | 143 | 840 | 3 | 35.1 |
| "Capital of France?" | 119 | 219 | 3 | 39.5 |
| "Sky colour?" | 119 | 219 | 3 | 39.4 |

### llama3.1:8b

| Prompt | Baseline (ms) | Cold (ms) | Hit (ms) | tok/s |
|---|---|---|---|---|
| "What is 2+2?" | 267 | 464 | 3 | 27.1 |
| "Capital of France?" | 258 | 403 | 2 | 27.6 |
| "Sky colour?" | 262 | 434 | 3 | 27.3 |

### deepseek-r1:7b

| Prompt | Baseline (ms) | Cold (ms) | Hit (ms) | tok/s |
|---|---|---|---|---|
| "What is 2+2?" | 49 783 | 107 266 | 2 | 9.0 |

One prompt only; reasoning chain tokens drove generation time to ~50s. Cache hit still 2ms.

---

## Memory Footprint by Quantization Level

Estimated for 7B parameter models (applies to mistral:7b, deepseek-r1:7b):

| Quantization | Weights | KV-cache (2k tokens) | Total peak |
|---|---|---|---|
| fp16 | 14.0 GB | 0.84 GB | 14.84 GB |
| q8_0 | 7.0 GB | 0.84 GB | 7.84 GB |
| q4_K_M | 3.5 GB | 0.84 GB | 4.34 GB |
| q4_0 | 3.2 GB | 0.84 GB | 4.04 GB |
| q3_K_M | 2.7 GB | 0.84 GB | 3.54 GB |
| q2_K | 2.0 GB | 0.84 GB | 2.84 GB |

All installed models use Q4_K_M. Engine memory estimator uses these values for admission control (configured at `memory.limit_gb=14.0`).

---

## Reproducing

```bash
# Requires Ollama running with models pulled
python scripts/start_server.py &
python scripts/run_integration_benchmarks.py
```

Outputs: `docs/benchmark_results.json`, `docs/PERFORMANCE_REPORT.md`
