"""Unit tests for ResultMapper."""


import pytest

from llm_inference_engine.api.result_mapper import ResultMapper
from llm_inference_engine.core.types import GenerationResult, RequestStatus, Response


def _make_response(request_id: str) -> Response:
    result = GenerationResult(
        request_id=request_id,
        text="hello",
        finish_reason="stop",
        tokens_used=5,
        latency_ms=20.0,
        model="llama3:8b",
    )
    return Response(request_id=request_id, result=result, status=RequestStatus.COMPLETED)


class TestResultMapper:
    """Tests for ResultMapper."""

    async def test_register_and_resolve(self) -> None:
        """Registered future should be resolved with the correct response."""
        mapper = ResultMapper()
        future = mapper.register("req-1")
        response = _make_response("req-1")
        mapper.resolve("req-1", response)
        result = await future
        assert result.request_id == "req-1"

    async def test_register_duplicate_raises(self) -> None:
        """Registering the same request_id twice should raise ValueError."""
        mapper = ResultMapper()
        mapper.register("req-1")
        with pytest.raises(ValueError, match="already registered"):
            mapper.register("req-1")

    def test_resolve_unknown_returns_false(self) -> None:
        """Resolving an unregistered request should return False."""
        mapper = ResultMapper()
        response = _make_response("unknown")
        assert mapper.resolve("unknown", response) is False

    async def test_reject_sets_exception(self) -> None:
        """reject() should set an exception on the future."""
        mapper = ResultMapper()
        future = mapper.register("req-1")
        mapper.reject("req-1", ValueError("Inference error"))
        with pytest.raises(ValueError, match="Inference error"):
            await future

    def test_reject_unknown_returns_false(self) -> None:
        """Rejecting an unregistered request should return False."""
        mapper = ResultMapper()
        assert mapper.reject("ghost", RuntimeError("oops")) is False

    async def test_cancel_all(self) -> None:
        """cancel_all() should cancel all pending futures."""
        mapper = ResultMapper()
        futures = [mapper.register(f"r{i}") for i in range(3)]
        count = mapper.cancel_all()
        assert count == 3
        assert mapper.pending_count == 0
        for fut in futures:
            assert fut.cancelled()

    def test_pending_count(self) -> None:
        """pending_count should reflect registered unresolved futures."""
        mapper = ResultMapper()
        assert mapper.pending_count == 0
        mapper.register("r1")
        mapper.register("r2")
        assert mapper.pending_count == 2

    def test_is_pending(self) -> None:
        """is_pending should return True for registered, unresolved IDs."""
        mapper = ResultMapper()
        mapper.register("r1")
        assert mapper.is_pending("r1") is True
        assert mapper.is_pending("r2") is False

    async def test_resolve_removes_from_pending(self) -> None:
        """Resolving a future should remove it from pending."""
        mapper = ResultMapper()
        mapper.register("r1")
        assert mapper.pending_count == 1
        mapper.resolve("r1", _make_response("r1"))
        assert mapper.pending_count == 0
        assert mapper.is_pending("r1") is False

    async def test_reject_removes_from_pending(self) -> None:
        """Rejecting a future should remove it from pending count."""
        mapper = ResultMapper()
        future = mapper.register("r1")
        mapper.reject("r1", RuntimeError("boom"))
        assert mapper.pending_count == 0
        with pytest.raises(RuntimeError):
            await future

    async def test_cancel_all_returns_zero_when_empty(self) -> None:
        """cancel_all() on empty mapper should return 0."""
        mapper = ResultMapper()
        assert mapper.cancel_all() == 0

    async def test_multiple_resolutions_independent(self) -> None:
        """Each future resolves independently."""
        mapper = ResultMapper()
        f1 = mapper.register("r1")
        f2 = mapper.register("r2")
        mapper.resolve("r1", _make_response("r1"))
        mapper.resolve("r2", _make_response("r2"))
        resp1 = await f1
        resp2 = await f2
        assert resp1.result is not None
        assert resp2.result is not None
        assert resp1.request_id == "r1"
        assert resp2.request_id == "r2"

    def test_is_pending_after_cancel_all(self) -> None:
        """After cancel_all(), no IDs should remain pending."""
        mapper = ResultMapper()
        mapper.register("r1")
        mapper.register("r2")
        mapper.cancel_all()
        assert not mapper.is_pending("r1")
        assert not mapper.is_pending("r2")

    async def test_resolve_already_done_future_returns_false(self) -> None:
        """Resolving an already-resolved future should return False."""
        mapper = ResultMapper()
        mapper.register("r1")
        mapper.resolve("r1", _make_response("r1"))
        # Future is done now; second resolve should return False
        result = mapper.resolve("r1", _make_response("r1"))
        assert result is False

    def test_pending_count_decrements_on_each_resolve(self) -> None:
        """pending_count should decrease as futures are resolved."""
        mapper = ResultMapper()
        mapper.register("r1")
        mapper.register("r2")
        mapper.register("r3")
        assert mapper.pending_count == 3
        mapper.resolve("r1", _make_response("r1"))
        assert mapper.pending_count == 2
        mapper.resolve("r2", _make_response("r2"))
        assert mapper.pending_count == 1

    def test_reject_unknown_id_returns_false(self) -> None:
        """Rejecting a completely unknown ID should return False."""
        mapper = ResultMapper()
        assert mapper.reject("nonexistent-id", Exception("oops")) is False
