"""Prometheus metrics for the inference engine."""

from prometheus_client import Counter, Gauge, Histogram

# Request counters
REQUESTS_TOTAL = Counter(
    "llm_engine_requests_total",
    "Total inference requests",
    ["model", "endpoint", "status"],
)

# Latency histogram
REQUEST_LATENCY = Histogram(
    "llm_engine_request_duration_seconds",
    "Request latency in seconds",
    ["model", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Cache metrics
CACHE_HITS = Counter(
    "llm_engine_cache_hits_total",
    "Total cache hits",
)
CACHE_MISSES = Counter(
    "llm_engine_cache_misses_total",
    "Total cache misses",
)
CACHE_SIZE = Gauge(
    "llm_engine_cache_entries",
    "Current number of cache entries",
)

# vLLM backend metrics
KV_CACHE_USAGE = Gauge(
    "llm_engine_kv_cache_usage",
    "vLLM KV cache usage fraction (0.0 – 1.0)",
)
ACTIVE_REQUESTS = Gauge(
    "llm_engine_active_requests",
    "Number of in-flight requests",
)
HEALTHY_BACKENDS = Gauge(
    "llm_engine_healthy_backends",
    "Number of backends with closed or half-open circuit breakers",
)

# Token metrics
TOKENS_GENERATED = Counter(
    "llm_engine_tokens_generated_total",
    "Total tokens generated",
    ["model"],
)
PROMPT_TOKENS = Counter(
    "llm_engine_prompt_tokens_total",
    "Total prompt tokens processed",
    ["model"],
)


__all__ = [
    "REQUESTS_TOTAL",
    "REQUEST_LATENCY",
    "CACHE_HITS",
    "CACHE_MISSES",
    "CACHE_SIZE",
    "KV_CACHE_USAGE",
    "ACTIVE_REQUESTS",
    "HEALTHY_BACKENDS",
    "TOKENS_GENERATED",
    "PROMPT_TOKENS",
]
