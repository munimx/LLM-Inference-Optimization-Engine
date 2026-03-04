"""Tests for Prometheus metrics module."""

from llm_inference_engine.metrics.prometheus import (
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    CACHE_SIZE,
    HEALTHY_BACKENDS,
    KV_CACHE_USAGE,
    PROMPT_TOKENS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TOKENS_GENERATED,
)


class TestPrometheusMetrics:

    def test_counters_exist(self):
        assert REQUESTS_TOTAL is not None
        assert CACHE_HITS is not None
        assert CACHE_MISSES is not None
        assert TOKENS_GENERATED is not None
        assert PROMPT_TOKENS is not None

    def test_gauges_exist(self):
        assert CACHE_SIZE is not None
        assert KV_CACHE_USAGE is not None
        assert HEALTHY_BACKENDS is not None
        assert ACTIVE_REQUESTS is not None

    def test_histogram_exists(self):
        assert REQUEST_LATENCY is not None

    def test_counter_can_increment(self):
        CACHE_HITS.inc()
