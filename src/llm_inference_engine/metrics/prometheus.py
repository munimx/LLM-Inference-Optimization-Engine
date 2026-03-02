"""Prometheus metrics for the inference engine.

Exposes Prometheus-compatible metrics at ``/metrics/prometheus`` so that
standard monitoring stacks (Prometheus, Grafana, Datadog) can scrape them.
"""

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

# Memory metrics
COMMITTED_MEMORY_GB = Gauge(
    "llm_engine_committed_memory_gb",
    "Currently committed memory in gigabytes",
)
ACTIVE_REQUESTS = Gauge(
    "llm_engine_active_requests",
    "Number of in-flight requests",
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
    "COMMITTED_MEMORY_GB",
    "ACTIVE_REQUESTS",
    "TOKENS_GENERATED",
    "PROMPT_TOKENS",
]
