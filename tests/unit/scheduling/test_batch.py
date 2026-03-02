"""Unit tests for Batch."""

import pytest

from llm_inference_engine.core.types import GenerationConfig, Request
from llm_inference_engine.scheduling.batch import Batch


def _make_request(
    request_id: str = "req-1",
    max_tokens: int = 64,
    priority: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt="Hello",
        model="llama3.1:8b",
        generation_config=GenerationConfig(max_tokens=max_tokens),
        priority=priority,
    )


class TestBatch:
    """Tests for Batch."""

    def test_empty_batch(self) -> None:
        """A freshly created batch should be empty."""
        batch = Batch(batch_id="b1", model="llama3.1:8b")
        assert batch.is_empty is True
        assert batch.size == 0
        assert batch.total_tokens == 0

    def test_add_request(self) -> None:
        """Adding a request should update size and token count."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=4)
        req = _make_request(max_tokens=128)
        batch.add(req)
        assert batch.size == 1
        assert batch.total_tokens == 128
        assert not batch.is_empty

    def test_is_full_when_max_requests_reached(self) -> None:
        """Batch.is_full should be True when max_requests is reached."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=2)
        batch.add(_make_request("r1"))
        assert batch.is_full is False
        batch.add(_make_request("r2"))
        assert batch.is_full is True

    def test_can_add_respects_max_requests(self) -> None:
        """can_add should return False when max_requests is reached."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=1)
        batch.add(_make_request("r1"))
        assert batch.can_add(_make_request("r2")) is False

    def test_can_add_respects_token_budget(self) -> None:
        """can_add should return False when max_tokens would be exceeded."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_tokens=100)
        batch.add(_make_request("r1", max_tokens=60))
        # Adding r2 (60 tokens) would push total to 120 > 100
        assert batch.can_add(_make_request("r2", max_tokens=60)) is False
        # Adding r3 (40 tokens) should be fine (total = 100)
        assert batch.can_add(_make_request("r3", max_tokens=40)) is True

    def test_add_raises_on_limit_exceeded(self) -> None:
        """add() should raise ValueError when limits are exceeded."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=1)
        batch.add(_make_request("r1"))
        with pytest.raises(ValueError, match="limits exceeded"):
            batch.add(_make_request("r2"))

    def test_unlimited_token_budget(self) -> None:
        """max_tokens=0 should impose no token limit."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=10, max_tokens=0)
        for i in range(5):
            assert batch.can_add(_make_request(f"r{i}", max_tokens=10_000))
            batch.add(_make_request(f"r{i}", max_tokens=10_000))
        assert batch.total_tokens == 50_000

    def test_estimated_memory_bytes(self) -> None:
        """estimated_memory_bytes should be 2 * total_tokens."""
        batch = Batch(batch_id="b1", model="llama3.1:8b")
        batch.add(_make_request(max_tokens=100))
        assert batch.estimated_memory_bytes == 200  # 100 tokens * 2 bytes

    def test_iteration(self) -> None:
        """Iterating over a batch should yield all added requests."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=3)
        reqs = [_make_request(f"r{i}") for i in range(3)]
        for r in reqs:
            batch.add(r)
        assert list(batch) == reqs

    def test_len(self) -> None:
        """len(batch) should equal batch.size."""
        batch = Batch(batch_id="b1", model="llama3.1:8b", max_requests=5)
        for i in range(3):
            batch.add(_make_request(f"r{i}"))
        assert len(batch) == 3
