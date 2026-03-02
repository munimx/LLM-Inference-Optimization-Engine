"""Unit tests for RequestQueue."""

import asyncio

from llm_inference_engine.core.types import GenerationConfig, Request, RequestStatus
from llm_inference_engine.scheduling.queue import RequestQueue


def _make_request(
    request_id: str = "req-1",
    model: str = "llama3.1:8b",
    priority: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt="Hello world",
        model=model,
        generation_config=GenerationConfig(max_tokens=64),
        priority=priority,
    )


class TestRequestQueue:
    """Tests for RequestQueue."""

    async def test_enqueue_and_dequeue_single(self) -> None:
        """A single enqueued request should be dequeued successfully."""
        queue = RequestQueue()
        req = _make_request()
        await queue.enqueue(req)
        result = await queue.dequeue()
        assert result is not None
        assert result.request_id == "req-1"

    async def test_dequeue_empty_returns_none(self) -> None:
        """Dequeuing from an empty queue should return None."""
        queue = RequestQueue()
        result = await queue.dequeue()
        assert result is None

    async def test_priority_ordering(self) -> None:
        """Higher-priority requests should be dequeued first."""
        queue = RequestQueue()
        low = _make_request("low", priority=1)
        high = _make_request("high", priority=10)
        mid = _make_request("mid", priority=5)

        await queue.enqueue(low)
        await queue.enqueue(mid)
        await queue.enqueue(high)

        first = await queue.dequeue()
        second = await queue.dequeue()
        third = await queue.dequeue()

        assert first is not None and first.request_id == "high"
        assert second is not None and second.request_id == "mid"
        assert third is not None and third.request_id == "low"

    async def test_fifo_within_same_priority(self) -> None:
        """Requests with equal priority should be dequeued FIFO."""
        queue = RequestQueue()
        for i in range(3):
            await queue.enqueue(_make_request(f"req-{i}", priority=5))

        ids = []
        for _ in range(3):
            r = await queue.dequeue()
            assert r is not None
            ids.append(r.request_id)

        assert ids == ["req-0", "req-1", "req-2"]

    async def test_cancel_skips_request(self) -> None:
        """Cancelled requests should be silently skipped during dequeue."""
        queue = RequestQueue()
        r1 = _make_request("r1")
        r2 = _make_request("r2")
        await queue.enqueue(r1)
        await queue.enqueue(r2)

        assert queue.cancel("r1") is True
        result = await queue.dequeue()
        assert result is not None and result.request_id == "r2"

    async def test_cancel_already_cancelled_returns_false(self) -> None:
        """Cancelling the same request twice should return False the second time."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("r1"))
        assert queue.cancel("r1") is True
        assert queue.cancel("r1") is False

    async def test_is_cancelled(self) -> None:
        """is_cancelled should reflect cancellation state."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("r1"))
        assert queue.is_cancelled("r1") is False
        queue.cancel("r1")
        assert queue.is_cancelled("r1") is True

    async def test_size_property(self) -> None:
        """size should reflect the number of items in the queue."""
        queue = RequestQueue()
        assert queue.size == 0
        await queue.enqueue(_make_request("r1"))
        await queue.enqueue(_make_request("r2"))
        assert queue.size == 2

    async def test_empty_property(self) -> None:
        """empty should be True when the queue has no items."""
        queue = RequestQueue()
        assert queue.empty is True
        await queue.enqueue(_make_request())
        assert queue.empty is False

    async def test_status_set_to_pending_on_enqueue(self) -> None:
        """Enqueued requests should have their status set to PENDING."""
        queue = RequestQueue()
        req = _make_request()
        req.status = RequestStatus.FAILED  # set to something else first
        await queue.enqueue(req)
        assert req.status == RequestStatus.PENDING

    async def test_dequeue_wait_blocks_until_item(self) -> None:
        """dequeue_wait should block and return the first enqueued item."""
        queue = RequestQueue()

        async def producer() -> None:
            await asyncio.sleep(0.05)
            await queue.enqueue(_make_request("async-req"))

        task = asyncio.create_task(producer())
        result = await queue.dequeue_wait()
        await task
        assert result.request_id == "async-req"

    async def test_all_cancelled_leaves_empty(self) -> None:
        """Dequeuing when all items are cancelled should return None."""
        queue = RequestQueue()
        for i in range(3):
            req = _make_request(f"r{i}")
            await queue.enqueue(req)
            queue.cancel(f"r{i}")

        result = await queue.dequeue()
        assert result is None

    async def test_cancel_non_existent_returns_false(self) -> None:
        """Cancelling an ID never enqueued should return False."""
        queue = RequestQueue()
        assert queue.cancel("never-enqueued") is False

    async def test_cancel_reduces_effective_size(self) -> None:
        """After cancelling, dequeue skips cancelled item."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("r1"))
        await queue.enqueue(_make_request("r2"))
        queue.cancel("r1")
        result = await queue.dequeue()
        assert result is not None and result.request_id == "r2"

    async def test_bounded_cancelled_set(self) -> None:
        """_cancelled set should not grow for IDs never enqueued."""
        queue = RequestQueue()
        for i in range(100):
            queue.cancel(f"phantom-{i}")
        assert len(queue._cancelled) == 0  # type: ignore[attr-defined]

    async def test_dequeue_multiple_priority_groups(self) -> None:
        """Dequeue order respects priority across multiple groups."""
        queue = RequestQueue()
        for p in [3, 7, 1, 9, 5]:
            await queue.enqueue(_make_request(f"r{p}", priority=p))
        first = await queue.dequeue()
        assert first is not None and first.priority == 9

    async def test_enqueue_sets_status_to_pending(self) -> None:
        """Enqueue should set request status to PENDING."""
        queue = RequestQueue()
        req = _make_request("r1")
        req.status = RequestStatus.FAILED
        await queue.enqueue(req)
        assert req.status == RequestStatus.PENDING

    async def test_dequeue_wait_returns_priority_order(self) -> None:
        """dequeue_wait should still respect priority ordering."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("low", priority=1))
        await queue.enqueue(_make_request("high", priority=10))
        result = await queue.dequeue_wait()
        assert result.request_id == "high"

    async def test_multiple_cancel_then_dequeue(self) -> None:
        """Dequeuing after mass cancellation returns only uncancelled requests."""
        queue = RequestQueue()
        for i in range(5):
            await queue.enqueue(_make_request(f"r{i}"))
        for i in [0, 1, 2, 4]:
            queue.cancel(f"r{i}")
        result = await queue.dequeue()
        assert result is not None and result.request_id == "r3"

    async def test_is_cancelled_false_for_dequeued(self) -> None:
        """After dequeuing, cancel should return False (not in _queued_ids)."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("r1"))
        await queue.dequeue()
        assert queue.cancel("r1") is False

    async def test_size_includes_cancelled(self) -> None:
        """size() counts all items in the underlying queue including cancelled."""
        queue = RequestQueue()
        await queue.enqueue(_make_request("r1"))
        await queue.enqueue(_make_request("r2"))
        queue.cancel("r1")
        assert queue.size == 2

    async def test_fifo_tie_break_across_enqueues(self) -> None:
        """Within same priority, sequential enqueues come out FIFO."""
        queue = RequestQueue()
        for i in range(4):
            await queue.enqueue(_make_request(f"r{i}", priority=5))
        ids = []
        for _ in range(4):
            r = await queue.dequeue()
            assert r is not None
            ids.append(r.request_id)
        assert ids == ["r0", "r1", "r2", "r3"]
