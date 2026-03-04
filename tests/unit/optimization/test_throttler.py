"""Unit tests for AdaptiveThrottler."""

import pytest

from llm_inference_engine.optimization.throttler import (
    AdaptiveThrottler,
    AdmissionDecision,
    ThrottlerStats,
    _parse_kv_cache_usage,
)


class TestAdaptiveThrottler:
    """Tests for AdaptiveThrottler."""

    def test_invalid_soft_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="soft_limit"):
            AdaptiveThrottler(backend_url="http://vllm:8080", soft_limit=0.0, hard_limit=0.9)

    def test_soft_exceeds_hard_raises(self) -> None:
        with pytest.raises(ValueError, match="soft_limit"):
            AdaptiveThrottler(backend_url="http://vllm:8080", soft_limit=0.95, hard_limit=0.90)

    def test_accept_when_below_soft_limit(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080", soft_limit=0.70, hard_limit=0.90)
        throttler._kv_usage = 0.50
        assert throttler.check() == AdmissionDecision.ACCEPT

    def test_queue_when_between_soft_and_hard(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080", soft_limit=0.70, hard_limit=0.90)
        throttler._kv_usage = 0.80
        assert throttler.check() == AdmissionDecision.QUEUE

    def test_reject_when_at_hard_limit(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080", soft_limit=0.70, hard_limit=0.90)
        throttler._kv_usage = 0.90
        assert throttler.check() == AdmissionDecision.REJECT

    def test_reject_when_above_hard_limit(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080", soft_limit=0.70, hard_limit=0.90)
        throttler._kv_usage = 0.95
        assert throttler.check() == AdmissionDecision.REJECT

    def test_increment_and_decrement_active(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080")
        throttler.increment_active()
        throttler.increment_active()
        assert throttler.stats.active_requests == 2
        throttler.decrement_active()
        assert throttler.stats.active_requests == 1

    def test_decrement_does_not_go_below_zero(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080")
        throttler.decrement_active()
        assert throttler.stats.active_requests == 0

    def test_stats_snapshot(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080", soft_limit=0.60, hard_limit=0.85)
        throttler._kv_usage = 0.42
        throttler.increment_active()
        stats = throttler.stats
        assert isinstance(stats, ThrottlerStats)
        assert stats.kv_cache_usage == pytest.approx(0.42)
        assert stats.soft_limit == 0.60
        assert stats.hard_limit == 0.85
        assert stats.active_requests == 1

    async def test_start_and_stop_lifecycle(self) -> None:
        throttler = AdaptiveThrottler("http://vllm:8080")
        await throttler.start()
        assert throttler._poll_task is not None
        assert not throttler._poll_task.done()
        await throttler.stop()
        assert throttler._poll_task is None


class TestParseKvCacheUsage:
    """Tests for _parse_kv_cache_usage helper."""

    def test_parses_standard_prometheus_format(self) -> None:
        text = 'vllm:kv_cache_usage_perc{model_name="llama3"} 0.72\n'
        assert _parse_kv_cache_usage(text) == pytest.approx(0.72)

    def test_returns_zero_when_metric_absent(self) -> None:
        assert _parse_kv_cache_usage("some_other_metric 1.0\n") == 0.0

    def test_returns_zero_for_empty_string(self) -> None:
        assert _parse_kv_cache_usage("") == 0.0

    def test_parses_scientific_notation(self) -> None:
        text = 'vllm:kv_cache_usage_perc{model_name="llama3"} 5.0e-01\n'
        assert _parse_kv_cache_usage(text) == pytest.approx(0.5)
