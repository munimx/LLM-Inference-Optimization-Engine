"""Unit tests for AdaptiveThrottler."""


import pytest

from llm_inference_engine.optimization.throttler import (
    AdaptiveThrottler,
    AdmissionDecision,
)


class TestAdaptiveThrottler:
    """Tests for AdaptiveThrottler."""

    def test_invalid_memory_limit(self) -> None:
        """Non-positive memory_limit_gb should raise ValueError."""
        with pytest.raises(ValueError, match="memory_limit_gb"):
            AdaptiveThrottler(memory_limit_gb=0.0)

    def test_invalid_soft_limit_ratio(self) -> None:
        """soft_limit_ratio outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="soft_limit_ratio"):
            AdaptiveThrottler(soft_limit_ratio=0.0)
        with pytest.raises(ValueError, match="soft_limit_ratio"):
            AdaptiveThrottler(soft_limit_ratio=1.0)

    async def test_accept_when_well_below_limit(self) -> None:
        """Decision should be ACCEPT when committed + new is below soft limit."""
        throttler = AdaptiveThrottler(memory_limit_gb=16.0, soft_limit_ratio=0.85)
        decision = await throttler.check(1.0)
        assert decision == AdmissionDecision.ACCEPT

    async def test_queue_when_between_soft_and_hard_limit(self) -> None:
        """Decision should be QUEUE when projected memory is in the soft zone."""
        throttler = AdaptiveThrottler(memory_limit_gb=10.0, soft_limit_ratio=0.85)
        # Reserve 8.6 GB → committed = 8.6 GB (above soft limit of 8.5 GB)
        await throttler.reserve("existing", 8.6)
        decision = await throttler.check(0.5)
        assert decision == AdmissionDecision.QUEUE

    async def test_reject_when_at_hard_limit(self) -> None:
        """Decision should be REJECT when projected memory meets/exceeds hard limit."""
        throttler = AdaptiveThrottler(memory_limit_gb=10.0, soft_limit_ratio=0.85)
        await throttler.reserve("big-req", 9.5)
        decision = await throttler.check(0.6)  # 9.5 + 0.6 = 10.1 >= 10.0
        assert decision == AdmissionDecision.REJECT

    async def test_reserve_increases_committed(self) -> None:
        """reserve() should increase committed_gb and active_requests."""
        throttler = AdaptiveThrottler()
        assert throttler.committed_gb == 0.0
        await throttler.reserve("r1", 3.5)
        assert throttler.committed_gb == pytest.approx(3.5)
        assert throttler.stats.active_requests == 1

    async def test_release_decreases_committed(self) -> None:
        """release() should decrease committed_gb and active_requests."""
        throttler = AdaptiveThrottler()
        await throttler.reserve("r1", 3.5)
        await throttler.release("r1", 3.5)
        assert throttler.committed_gb == pytest.approx(0.0)
        assert throttler.stats.active_requests == 0

    async def test_release_does_not_go_negative(self) -> None:
        """Releasing more than committed should floor at 0.0."""
        throttler = AdaptiveThrottler()
        await throttler.release("phantom", 100.0)  # release with no reservation
        assert throttler.committed_gb == 0.0

    async def test_stats_snapshot(self) -> None:
        """stats property should reflect current throttler state."""
        throttler = AdaptiveThrottler(memory_limit_gb=10.0, soft_limit_ratio=0.8)
        await throttler.reserve("r1", 4.0)
        s = throttler.stats
        assert s.committed_gb == pytest.approx(4.0)
        assert s.available_gb == pytest.approx(6.0)
        assert s.memory_limit_gb == pytest.approx(10.0)
        assert s.soft_limit_gb == pytest.approx(8.0)
        assert s.active_requests == 1

    async def test_negative_estimated_memory_raises(self) -> None:
        """check() with negative estimated_memory_gb should raise ValueError."""
        throttler = AdaptiveThrottler()
        with pytest.raises(ValueError, match="negative"):
            await throttler.check(-1.0)

    async def test_zero_estimated_memory_accepted(self) -> None:
        """check() with 0.0 GB (overhead-free) should be ACCEPT when headroom exists."""
        throttler = AdaptiveThrottler(memory_limit_gb=16.0)
        decision = await throttler.check(0.0)
        assert decision == AdmissionDecision.ACCEPT

    async def test_multiple_reserves_accumulate(self) -> None:
        """Multiple reserve() calls should accumulate committed memory."""
        throttler = AdaptiveThrottler(memory_limit_gb=20.0)
        for i in range(5):
            await throttler.reserve(f"r{i}", 1.0)
        assert throttler.committed_gb == pytest.approx(5.0)
        assert throttler.stats.active_requests == 5

    async def test_queue_decision_near_soft_limit(self) -> None:
        """When committed exceeds soft limit, decision should be QUEUE."""
        throttler = AdaptiveThrottler(memory_limit_gb=10.0, soft_limit_ratio=0.5)
        # Soft limit is 5 GB; reserve 5 GB, then check a tiny request
        await throttler.reserve("r1", 5.0)
        decision = await throttler.check(0.1)
        assert decision == AdmissionDecision.QUEUE

    async def test_reject_decision_near_hard_limit(self) -> None:
        """When request would exceed hard limit, decision should be REJECT."""
        throttler = AdaptiveThrottler(memory_limit_gb=4.0, soft_limit_ratio=0.8)
        # Reserve 3.5 GB; only 0.5 GB left; new request of 1.0 GB exceeds hard limit
        await throttler.reserve("r1", 3.5)
        decision = await throttler.check(1.0)
        assert decision == AdmissionDecision.REJECT

    async def test_reserve_and_release_symmetry(self) -> None:
        """reserve then release should return committed_gb to its original value."""
        throttler = AdaptiveThrottler(memory_limit_gb=16.0)
        initial = throttler.committed_gb
        await throttler.reserve("req1", 4.0)
        await throttler.release("req1", 4.0)
        assert throttler.committed_gb == pytest.approx(initial)

    async def test_release_clamps_at_zero(self) -> None:
        """Releasing more than committed should not go negative."""
        throttler = AdaptiveThrottler(memory_limit_gb=16.0)
        await throttler.reserve("req1", 1.0)
        await throttler.release("req1", 10.0)  # release more than reserved
        assert throttler.committed_gb >= 0.0

    async def test_active_requests_decrements_on_release(self) -> None:
        throttler = AdaptiveThrottler(memory_limit_gb=16.0)
        await throttler.reserve("req1", 1.0)
        await throttler.reserve("req2", 1.0)
        await throttler.release("req1", 1.0)
        assert throttler.stats.active_requests == 1

    async def test_available_gb_decreases_after_reserve(self) -> None:
        throttler = AdaptiveThrottler(memory_limit_gb=10.0)
        before = throttler.stats.available_gb
        await throttler.reserve("req1", 3.0)
        after = throttler.stats.available_gb
        assert after < before

    async def test_check_accept_with_headroom(self) -> None:
        """Fresh throttler should ACCEPT a small request."""
        throttler = AdaptiveThrottler(memory_limit_gb=32.0, soft_limit_ratio=0.9)
        decision = await throttler.check(1.0)
        assert decision == AdmissionDecision.ACCEPT

    async def test_concurrent_reserves_accumulate(self) -> None:
        """Sequential reserves should accumulate correctly."""
        throttler = AdaptiveThrottler(memory_limit_gb=20.0)
        await throttler.reserve("a", 2.0)
        await throttler.reserve("b", 3.0)
        assert throttler.committed_gb == pytest.approx(5.0)
        assert throttler.stats.active_requests == 2

    async def test_soft_limit_reflected_in_stats(self) -> None:
        throttler = AdaptiveThrottler(memory_limit_gb=10.0, soft_limit_ratio=0.7)
        assert throttler.stats.soft_limit_gb == pytest.approx(7.0)

    async def test_memory_limit_reflected_in_stats(self) -> None:
        throttler = AdaptiveThrottler(memory_limit_gb=16.0)
        assert throttler.stats.memory_limit_gb == pytest.approx(16.0)
