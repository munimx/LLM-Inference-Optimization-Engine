"""Unit tests for scheduling policies and the Scheduler."""

import time
from typing import Any

from llm_inference_engine.core.types import (
    GenerationConfig,
    GenerationResult,
    Request,
    RequestStatus,
    Response,
)
from llm_inference_engine.scheduling.batch import Batch
from llm_inference_engine.scheduling.policies import (
    FCFSPolicy,
    PriorityPolicy,
    SchedulingPolicy,
    SJFPolicy,
    TokenBudgetPolicy,
    get_policy,
)
from llm_inference_engine.scheduling.scheduler import Scheduler

MODEL = "llama3.1:8b"


def _make_request(
    request_id: str = "req-1",
    max_tokens: int = 64,
    priority: int = 0,
    timestamp: float = 0.0,
    model: str = MODEL,
) -> Request:
    r = Request(
        request_id=request_id,
        prompt="Hello",
        model=model,
        generation_config=GenerationConfig(max_tokens=max_tokens),
        priority=priority,
    )
    r.timestamp = timestamp
    return r


def _make_response(request_id: str) -> Response:
    result = GenerationResult(
        request_id=request_id,
        text="ok",
        finish_reason="stop",
        tokens_used=10,
        latency_ms=50.0,
        model=MODEL,
    )
    return Response(request_id=request_id, result=result, status=RequestStatus.COMPLETED)


async def _simple_dispatch(batch: Batch) -> list[Response]:
    """Simple dispatch function that returns one Response per request."""
    return [_make_response(r.request_id) for r in batch]


# ------------------------------------------------------------------
# Policy tests
# ------------------------------------------------------------------


class TestFCFSPolicy:
    def test_order_preserved(self) -> None:
        """FCFS should preserve the input order."""
        policy = FCFSPolicy()
        requests = [_make_request(f"r{i}") for i in range(3)]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=0)
        assert [r.request_id for r in batch] == ["r0", "r1", "r2"]

    def test_respects_max_requests(self) -> None:
        """FCFS should stop filling when max_requests is reached."""
        policy = FCFSPolicy()
        requests = [_make_request(f"r{i}") for i in range(5)]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=3, max_tokens=0)
        assert batch.size == 3

    def test_respects_token_budget(self) -> None:
        """FCFS should stop when the token budget is exhausted."""
        policy = FCFSPolicy()
        requests = [_make_request(f"r{i}", max_tokens=50) for i in range(4)]
        # Budget = 120 → fits 2 (100 tokens), not 3 (150 tokens)
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=120)
        assert batch.size == 2


class TestSJFPolicy:
    def test_shortest_jobs_first(self) -> None:
        """SJF should process requests with fewer tokens first."""
        now = time.time()
        policy = SJFPolicy()
        requests = [
            _make_request("big", max_tokens=200, timestamp=now),
            _make_request("small", max_tokens=10, timestamp=now),
            _make_request("medium", max_tokens=100, timestamp=now),
        ]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=0)
        assert [r.request_id for r in batch] == ["small", "medium", "big"]

    def test_skips_over_non_fitting(self) -> None:
        """SJF should skip jobs that don't fit while trying smaller ones."""
        now = time.time()
        policy = SJFPolicy()
        requests = [
            _make_request("big", max_tokens=90, timestamp=now),
            _make_request("small", max_tokens=20, timestamp=now),
        ]
        # Budget=100: big alone fits, but together they don't.
        # SJF visits small first (20 tokens), then tries big (90 > 80 remaining) — skips.
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=100)
        # small fits, big doesn't fit after small → big is skipped
        ids = [r.request_id for r in batch]
        assert "small" in ids

    def test_starvation_guard_promotes_old_requests(self) -> None:
        """Requests waiting longer than max_wait should be promoted."""
        now = time.time()
        policy = SJFPolicy(max_wait_seconds=10.0)
        requests = [
            _make_request("big", max_tokens=200, timestamp=now - 15),  # old
            _make_request("small", max_tokens=10, timestamp=now),  # fresh
        ]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=0)
        # big is promoted (waited >10s), goes before small
        assert [r.request_id for r in batch] == ["big", "small"]


class TestPriorityPolicy:
    def test_highest_priority_first(self) -> None:
        """Priority policy should dequeue highest-priority requests first."""
        policy = PriorityPolicy()
        requests = [
            _make_request("low", priority=1, timestamp=1.0),
            _make_request("high", priority=10, timestamp=2.0),
            _make_request("mid", priority=5, timestamp=3.0),
        ]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=0)
        assert [r.request_id for r in batch] == ["high", "mid", "low"]

    def test_fifo_tiebreak_on_equal_priority(self) -> None:
        """Equal-priority requests should be ordered by timestamp (FIFO)."""
        policy = PriorityPolicy()
        requests = [
            _make_request("r2", priority=5, timestamp=2.0),
            _make_request("r1", priority=5, timestamp=1.0),
            _make_request("r3", priority=5, timestamp=3.0),
        ]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=0)
        assert [r.request_id for r in batch] == ["r1", "r2", "r3"]


class TestTokenBudgetPolicy:
    def test_greedy_packing(self) -> None:
        """TokenBudget should greedily pack as many requests as possible."""
        policy = TokenBudgetPolicy()
        requests = [
            _make_request("r1", max_tokens=60, priority=5, timestamp=1.0),
            _make_request("r2", max_tokens=60, priority=4, timestamp=2.0),
            _make_request("r3", max_tokens=30, priority=3, timestamp=3.0),
        ]
        # Budget=110: r1 (60) + r3 (30) fit; r2 (60) would push r1+r2 to 120 > 110
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=110)
        ids = [r.request_id for r in batch]
        assert "r1" in ids
        assert "r3" in ids
        # r2 might or might not fit depending on order, but total must be <= 110
        assert batch.total_tokens <= 110

    def test_all_fit_within_budget(self) -> None:
        """All requests should be included when budget is sufficient."""
        policy = TokenBudgetPolicy()
        requests = [_make_request(f"r{i}", max_tokens=10) for i in range(5)]
        batch = policy.form_batch(requests, "b1", MODEL, max_requests=10, max_tokens=100)
        assert batch.size == 5


class TestGetPolicy:
    def test_returns_correct_policy_types(self) -> None:
        """get_policy should return the correct implementation."""
        assert isinstance(get_policy(SchedulingPolicy.FCFS), FCFSPolicy)
        assert isinstance(get_policy(SchedulingPolicy.SJF), SJFPolicy)
        assert isinstance(get_policy(SchedulingPolicy.PRIORITY), PriorityPolicy)
        assert isinstance(get_policy(SchedulingPolicy.TOKEN_BUDGET), TokenBudgetPolicy)


# ------------------------------------------------------------------
# Scheduler tests
# ------------------------------------------------------------------


class TestScheduler:
    def _make_dispatch_fn(self) -> Any:
        """Create a mock dispatch function that returns one Response per Request."""

        async def dispatch(batch: Batch) -> list[Response]:
            return [_make_response(r.request_id) for r in batch]

        return dispatch

    async def test_submit_and_drain(self) -> None:
        """Submitting a request and draining should dispatch it."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        req = _make_request("req-1")
        await scheduler.submit(req)
        responses = await scheduler.drain(MODEL)
        assert len(responses) == 1
        assert responses[0].request_id == "req-1"

    async def test_drain_empty_queue(self) -> None:
        """Draining an empty queue should return an empty list."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        responses = await scheduler.drain(MODEL)
        assert responses == []

    async def test_drain_unknown_model(self) -> None:
        """Draining a model that has no queue should return empty list."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        responses = await scheduler.drain("unknown:model")
        assert responses == []

    async def test_cancel_before_drain(self) -> None:
        """Cancelling a request before drain should exclude it from the batch."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        req1 = _make_request("r1")
        req2 = _make_request("r2")
        await scheduler.submit(req1)
        await scheduler.submit(req2)
        scheduler.cancel("r1", MODEL)
        responses = await scheduler.drain(MODEL)
        ids = [r.request_id for r in responses]
        assert "r1" not in ids
        assert "r2" in ids

    async def test_status_set_to_completed_after_drain(self) -> None:
        """Requests should have COMPLETED status after being dispatched."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        req = _make_request("req-status")
        await scheduler.submit(req)
        await scheduler.drain(MODEL)
        assert req.status == RequestStatus.COMPLETED

    async def test_queue_size(self) -> None:
        """queue_size should reflect the number of items in the model queue."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        assert scheduler.queue_size(MODEL) == 0
        await scheduler.submit(_make_request("r1"))
        await scheduler.submit(_make_request("r2"))
        assert scheduler.queue_size(MODEL) == 2

    async def test_multiple_models_isolated_queues(self) -> None:
        """Different model queues should be independent."""
        scheduler = Scheduler(dispatch_fn=self._make_dispatch_fn())
        model_a = "llama3.1:8b"
        model_b = "mistral:7b"
        req_a = Request(
            request_id="a1",
            prompt="hi",
            model=model_a,
            generation_config=GenerationConfig(max_tokens=10),
        )
        req_b = Request(
            request_id="b1",
            prompt="hi",
            model=model_b,
            generation_config=GenerationConfig(max_tokens=10),
        )
        await scheduler.submit(req_a)
        await scheduler.submit(req_b)

        responses_a = await scheduler.drain(model_a)
        responses_b = await scheduler.drain(model_b)

        assert len(responses_a) == 1 and responses_a[0].request_id == "a1"
        assert len(responses_b) == 1 and responses_b[0].request_id == "b1"

    async def test_batch_respects_max_requests(self) -> None:
        """Scheduler should limit batch size to max_requests_per_batch."""
        scheduler = Scheduler(
            dispatch_fn=self._make_dispatch_fn(), max_requests_per_batch=2
        )
        for i in range(5):
            await scheduler.submit(_make_request(f"r{i}"))
        responses = await scheduler.drain(MODEL)
        assert len(responses) <= 2

    async def test_sjf_policy_wired_correctly(self) -> None:
        """Scheduler with SJF policy should process shortest jobs first."""
        now = time.time()
        dispatched_order: list[str] = []

        async def recording_dispatch(batch: Batch) -> list[Response]:
            for r in batch:
                dispatched_order.append(r.request_id)
            return [_make_response(r.request_id) for r in batch]

        scheduler = Scheduler(
            dispatch_fn=recording_dispatch,
            policy=SchedulingPolicy.SJF,
            max_requests_per_batch=10,
        )
        await scheduler.submit(_make_request("big", max_tokens=200, timestamp=now))
        await scheduler.submit(_make_request("small", max_tokens=10, timestamp=now))
        await scheduler.drain(MODEL)
        assert dispatched_order[0] == "small"

    async def test_cancel_before_drain_prevents_dispatch(self) -> None:
        """Cancelling a request before drain should prevent it from being dispatched."""
        dispatched_ids: list[str] = []

        async def recording_dispatch(batch: Batch) -> list[Response]:
            for r in batch:
                dispatched_ids.append(r.request_id)
            return [_make_response(r.request_id) for r in batch]

        scheduler = Scheduler(dispatch_fn=recording_dispatch, policy=SchedulingPolicy.FCFS)
        await scheduler.submit(_make_request("r1"))
        await scheduler.submit(_make_request("r2"))
        scheduler.cancel("r1", MODEL)
        await scheduler.drain(MODEL)
        assert "r1" not in dispatched_ids
        assert "r2" in dispatched_ids

    async def test_drain_empty_queue_returns_empty(self) -> None:
        """Draining an empty queue should return an empty list."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        responses = await scheduler.drain(MODEL)
        assert responses == []

    async def test_queue_size_reflects_submissions(self) -> None:
        """queue_size should increase after each submit."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        assert scheduler.queue_size(MODEL) == 0
        await scheduler.submit(_make_request("r1"))
        assert scheduler.queue_size(MODEL) == 1
        await scheduler.submit(_make_request("r2"))
        assert scheduler.queue_size(MODEL) == 2

    async def test_multi_model_queues_are_independent(self) -> None:
        """Different model queues should be completely independent."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        await scheduler.submit(_make_request("r1", model="llama3:8b"))
        await scheduler.submit(_make_request("r2", model="mistral:7b"))
        assert scheduler.queue_size("llama3:8b") == 1
        assert scheduler.queue_size("mistral:7b") == 1

    async def test_responses_include_request_ids(self) -> None:
        """drain() should return responses with matching request IDs."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        await scheduler.submit(_make_request("my-req-id"))
        responses = await scheduler.drain(MODEL)
        assert any(r.request_id == "my-req-id" for r in responses)

    async def test_cancel_unknown_model_returns_false(self) -> None:
        """Cancelling from a model queue that doesn't exist should return False."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        result = scheduler.cancel("r1", "nonexistent:model")
        assert result is False

    async def test_status_set_to_completed_after_simple_dispatch(self) -> None:
        """After a successful drain, response status should be COMPLETED."""
        scheduler = Scheduler(dispatch_fn=_simple_dispatch, policy=SchedulingPolicy.FCFS)
        await scheduler.submit(_make_request("r1"))
        responses = await scheduler.drain(MODEL)
        assert all(r.status == RequestStatus.COMPLETED for r in responses)

    async def test_max_requests_per_batch_limits_drain(self) -> None:
        """max_requests_per_batch should cap the number dispatched per drain."""
        scheduler = Scheduler(
            dispatch_fn=_simple_dispatch,
            policy=SchedulingPolicy.FCFS,
            max_requests_per_batch=2,
        )
        for i in range(5):
            await scheduler.submit(_make_request(f"r{i}"))
        responses = await scheduler.drain(MODEL)
        assert len(responses) <= 2

    async def test_priority_policy_dispatches_high_priority_first(self) -> None:
        """Priority policy should dispatch higher priority requests first."""
        dispatched: list[str] = []

        async def recording(batch: Batch) -> list[Response]:
            for r in batch:
                dispatched.append(r.request_id)
            return [_make_response(r.request_id) for r in batch]

        scheduler = Scheduler(
            dispatch_fn=recording,
            policy=SchedulingPolicy.PRIORITY,
            max_requests_per_batch=10,
        )
        await scheduler.submit(_make_request("low", priority=1))
        await scheduler.submit(_make_request("high", priority=10))
        await scheduler.drain(MODEL)
        assert dispatched[0] == "high"
