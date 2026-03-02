"""Unit tests for ResultMapper."""

import asyncio

import pytest

from llm_inference_engine.api.result_mapper import ResultMapper
from llm_inference_engine.core.types import GenerationConfig, GenerationResult, Request, RequestStatus, Response


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
