"""Unit tests for batch formation policies."""

import pytest

from llm_inference_engine.core.types import GenerationConfig, Request
from llm_inference_engine.scheduling.batch import Batch
from llm_inference_engine.scheduling.policies import (
    FCFSPolicy,
    PriorityPolicy,
    SchedulingPolicy,
    SJFPolicy,
    TokenBudgetPolicy,
    get_policy,
)


def _make_request(
    request_id: str = "r1",
    max_tokens: int = 64,
    priority: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt="Hello",
        model="llama3:8b",
        generation_config=GenerationConfig(max_tokens=max_tokens),
        priority=priority,
    )


def _ids(batch: Batch) -> list[str]:
    return [r.request_id for r in batch]


class TestFCFSPolicy:
    """Tests for FCFSPolicy."""

    def test_empty_requests_produces_empty_batch(self) -> None:
        policy = FCFSPolicy()
        batch = policy.form_batch([], "b1", "m", max_requests=8, max_tokens=0)
        assert batch.is_empty

    def test_preserves_arrival_order(self) -> None:
        policy = FCFSPolicy()
        reqs = [_make_request(f"r{i}") for i in range(4)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=0)
        assert _ids(batch) == ["r0", "r1", "r2", "r3"]

    def test_respects_max_requests(self) -> None:
        policy = FCFSPolicy()
        reqs = [_make_request(f"r{i}") for i in range(5)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=3, max_tokens=0)
        assert batch.size == 3
        assert _ids(batch) == ["r0", "r1", "r2"]

    def test_respects_token_budget(self) -> None:
        policy = FCFSPolicy()
        # r0=60, r1=60 — second would overflow 100 budget
        reqs = [_make_request("r0", max_tokens=60), _make_request("r1", max_tokens=60)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=100)
        assert batch.size == 1
        assert _ids(batch) == ["r0"]

    def test_single_request(self) -> None:
        policy = FCFSPolicy()
        batch = policy.form_batch([_make_request()], "b1", "m", max_requests=4, max_tokens=0)
        assert batch.size == 1

    def test_stops_at_first_non_fitting_request(self) -> None:
        """FCFS stops when a request doesn't fit."""
        policy = FCFSPolicy()
        reqs = [
            _make_request("small_first", max_tokens=60),  # fits
            _make_request("big", max_tokens=200),  # won't fit after small_first
        ]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=150)
        assert batch.size == 1
        assert _ids(batch) == ["small_first"]


class TestSJFPolicy:
    """Tests for SJFPolicy."""

    def test_sorts_by_max_tokens_ascending(self) -> None:
        policy = SJFPolicy()
        reqs = [
            _make_request("big", max_tokens=256),
            _make_request("small", max_tokens=16),
            _make_request("medium", max_tokens=64),
        ]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=10, max_tokens=0)
        assert _ids(batch) == ["small", "medium", "big"]

    def test_skips_over_non_fitting_jobs(self) -> None:
        """SJF continues to the next request if one doesn't fit."""
        policy = SJFPolicy()
        reqs = [
            _make_request("huge", max_tokens=200),
            _make_request("tiny", max_tokens=10),
        ]
        # budget=150 → "huge" doesn't fit, but "tiny" does
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=150)
        assert "tiny" in _ids(batch)
        assert batch.size == 1

    def test_empty_requests(self) -> None:
        policy = SJFPolicy()
        batch = policy.form_batch([], "b1", "m", max_requests=4, max_tokens=0)
        assert batch.is_empty

    def test_respects_max_requests(self) -> None:
        policy = SJFPolicy()
        reqs = [_make_request(f"r{i}", max_tokens=10 + i) for i in range(5)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=3, max_tokens=0)
        assert batch.size == 3


class TestPriorityPolicy:
    """Tests for PriorityPolicy."""

    def test_sorts_by_priority_descending(self) -> None:
        policy = PriorityPolicy()
        reqs = [
            _make_request("low", priority=1),
            _make_request("high", priority=10),
            _make_request("medium", priority=5),
        ]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=3, max_tokens=0)
        assert _ids(batch)[0] == "high"

    def test_fifo_within_same_priority(self) -> None:
        policy = PriorityPolicy()
        reqs = [_make_request(f"r{i}", priority=5) for i in range(3)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=3, max_tokens=0)
        assert _ids(batch) == ["r0", "r1", "r2"]

    def test_stops_at_first_non_fitting(self) -> None:
        """PriorityPolicy stops when a request doesn't fit."""
        policy = PriorityPolicy()
        reqs = [
            _make_request("first", priority=10, max_tokens=200),  # won't fit
            _make_request("second", priority=5, max_tokens=50),
        ]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=150)
        # first (priority 10) processed first but doesn't fit — stops
        assert "second" not in _ids(batch)

    def test_empty_requests(self) -> None:
        policy = PriorityPolicy()
        batch = policy.form_batch([], "b1", "m", max_requests=4, max_tokens=0)
        assert batch.is_empty


class TestTokenBudgetPolicy:
    """Tests for TokenBudgetPolicy."""

    def test_greedily_packs_high_priority_first(self) -> None:
        policy = TokenBudgetPolicy()
        reqs = [
            _make_request("low", priority=1, max_tokens=80),
            _make_request("high", priority=10, max_tokens=80),
        ]
        # budget=100 — fits one 80-token request; high priority should win
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=100)
        assert "high" in _ids(batch)
        assert "low" not in _ids(batch)

    def test_skips_non_fitting_unlike_fcfs(self) -> None:
        """TokenBudgetPolicy skips non-fitting requests (greedy bin-packing)."""
        policy = TokenBudgetPolicy()
        reqs = [
            _make_request("big", priority=10, max_tokens=200),
            _make_request("small", priority=5, max_tokens=50),
        ]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=4, max_tokens=150)
        # "big" doesn't fit, but "small" should still be packed
        assert "small" in _ids(batch)

    def test_empty_requests(self) -> None:
        policy = TokenBudgetPolicy()
        batch = policy.form_batch([], "b1", "m", max_requests=4, max_tokens=0)
        assert batch.is_empty

    def test_all_requests_fit_within_budget(self) -> None:
        policy = TokenBudgetPolicy()
        reqs = [_make_request(f"r{i}", max_tokens=20) for i in range(4)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=10, max_tokens=100)
        assert batch.size == 4

    def test_respects_max_requests(self) -> None:
        policy = TokenBudgetPolicy()
        reqs = [_make_request(f"r{i}", max_tokens=10) for i in range(10)]
        batch = policy.form_batch(reqs, "b1", "m", max_requests=3, max_tokens=0)
        assert batch.size == 3


class TestGetPolicy:
    """Tests for the get_policy factory function."""

    @pytest.mark.parametrize(
        "policy_enum,expected_type",
        [
            (SchedulingPolicy.FCFS, FCFSPolicy),
            (SchedulingPolicy.SJF, SJFPolicy),
            (SchedulingPolicy.PRIORITY, PriorityPolicy),
            (SchedulingPolicy.TOKEN_BUDGET, TokenBudgetPolicy),
        ],
    )
    def test_returns_correct_type(
        self, policy_enum: SchedulingPolicy, expected_type: type
    ) -> None:
        policy = get_policy(policy_enum)
        assert isinstance(policy, expected_type)

    def test_policy_enum_values(self) -> None:
        assert SchedulingPolicy.FCFS.value == "fcfs"
        assert SchedulingPolicy.SJF.value == "sjf"
        assert SchedulingPolicy.PRIORITY.value == "priority"
        assert SchedulingPolicy.TOKEN_BUDGET.value == "token_budget"
