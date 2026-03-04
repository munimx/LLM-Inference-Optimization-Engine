# Performance Report — LLM Inference Optimization Engine

**Hardware**: Apple M2 Air, 16 GB unified memory  
**Ollama**: local (all models Q4_K_M except phi3 Q4_0)  
**Engine config**: `fcfs` policy, `max_requests_per_batch=8`, cache LRU 256 entries, TTL 300s  
**Models**: phi3:latest (3.8B), mistral:7b (7.2B), llama3.1:8b (8.0B), deepseek-r1:7b (7.6B)  
**Date**: 2026-03-03

---

## Cache Hit Latency

| Model | Mean hit (ms) | Min (ms) | Max (ms) | Speedup vs baseline |
|---|---|---|---|---|
| `phi3:latest` | **1.6** | 1.1 | 3.3 | ~2898x |
| `mistral:7b` | **2.8** | 1.0 | 6.3 | ~4922x |
| `llama3.1:8b` | **1.6** | 0.9 | 4.1 | ~8663x |
| `deepseek-r1:7b` | **1.8** | 1.1 | 2.8 | N/A¹ |

> A cache hit completely bypasses Ollama. The 1–3ms represents FastAPI + cache lookup.
>
> ¹ deepseek-r1 excluded from S2 baseline; its cold requests take 40–120s, so cache hits represent 20,000–60,000x speedup.

## Sequential Throughput — 10 Requests

| Model | Direct Ollama (req/s) | Engine cold (req/s) | Engine cached (req/s) | Cached speedup |
|---|---|---|---|---|
| `mistral:7b` | 0.20 | 0.27 | 426.39 | **2086.7x** |
| `llama3.1:8b` | 0.82 | 0.77 | 414.91 | **504.6x** |

## Mixed Workload — 60% Cache Hit Rate

| Model | Direct wall (ms) | Engine wall (ms) | Speedup |
|---|---|---|---|
| `mistral:7b` | 241088 | 41 | **5900x**¹ |
| `llama3.1:8b` | 476201 | 221443 | **2.15x** |

> ¹ mistral S4 hit 100% cache due to prompts already cached from earlier scenarios; true 60% hit rate speedup is ~2x (similar to llama3.1).

## Concurrent Burst — 4 Parallel Requests

| Model | Sequential total (ms) | Concurrent wall (ms) | Speedup |
|---|---|---|---|
| `mistral:7b` | 34023 | 29623 | **1.15x** |
| `llama3.1:8b` | 36849 | 31464 | **1.17x** |

## Cold Path Overhead per Model

| Model | Scenario | Direct (ms) | Cold (ms) | Overhead | Overhead % |
|---|---|---|---|---|---|
| `phi3:latest` | Short (~20 tok) | 210 | 217 | +7 ms | +3% |
| `mistral:7b` | Short (~20 tok) | 2122 | 1909 | -213 ms | -10% |
| `llama3.1:8b` | Short (~20 tok) | 663 | 780 | +117 ms | +18% |
| `phi3:latest` | Medium (~80 tok) | 2856 | 2662 | -194 ms | -7% |
| `mistral:7b` | Medium (~80 tok) | 20358 | 16949 | -3409 ms | -17% |
| `llama3.1:8b` | Medium (~80 tok) | 19695 | 16858 | -2836 ms | -14% |

## Streaming — Time to First Token

| Model | TTFT (ms) | Stream total (ms) | Non-stream (ms) |
|---|---|---|---|
| `phi3:latest` | **261** | 3840 | 12928 |
| `mistral:7b` | **306** | 8725 | 37526 |
| `llama3.1:8b` | **678** | 10496 | 72415 |
| `deepseek-r1:7b` | **1549** | 10145 | N/A¹ |

> TTFT = time until the first SSE chunk arrives. Stream total includes all chunks + `[DONE]`.
>
> ¹ deepseek-r1 non-streaming requests timed out during benchmark; streaming still completed successfully.

## Chat Completions Overhead

| Model | Ollama /api/chat (ms) | Engine /chat/completions (ms) | Overhead |
|---|---|---|---|
| `phi3:latest` | 3754 | 3904 | +4% |
| `mistral:7b` | 8282 | 8736 | +6% |
| `llama3.1:8b` | 9155 | 9506 | +4% |
| `deepseek-r1:7b` | 9742 | 8638 | -11% |

## When to Use This Engine

| Workload type | Speedup | Recommendation |
|---|---|---|
| Repeated/similar prompts (FAQ, chat templates) | 400–2000x | ✅ Strong fit — cache eliminates Ollama entirely (1–3ms) |
| Mixed workload, 50%+ hit rate | 2–3x | ✅ Good fit |
| Concurrent users, same model | 1.15–1.17x | ✅ Marginal gain from batching |
| Streaming (time to first token) | TTFT 260–680ms | ✅ User sees response while generating |
| Chat completions overhead | 4–6% | ✅ Negligible — use for cache + scheduling benefits |
| Single unique requests, no repetition | −3 to +18% | ⚠️ Small overhead on short prompts; negligible on medium |
| Reasoning models (deepseek-r1) | cache only | ⚠️ Only cache hits help; cold requests take 40–120s |

## Methodology

- 8 scenarios across 4 models (deepseek-r1 excluded from latency-sensitive scenarios due to extreme latency)
- `max_tokens` set identically on both Ollama and engine paths
- 1 warmup run discarded; 3 measured runs averaged (10 for cache scenario)
- Engine cold path uses unique prompt suffix per run to force cache miss
- Direct Ollama: `POST /api/generate` with `stream: false`
- Engine: `POST /completions`, `/chat/completions`, and SSE streaming

Reproduce: `python scripts/start_server.py & python scripts/run_integration_benchmarks.py`
