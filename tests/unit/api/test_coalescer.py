"""Tests for request coalescing."""

import asyncio

import pytest

from llm_inference_engine.api.coalescer import RequestCoalescer


class TestRequestCoalescer:

    async def test_single_request(self):
        coalescer = RequestCoalescer()

        async def producer():
            return "world"

        result = await coalescer.coalesce("m", "hello", producer)
        assert result == "world"

    async def test_coalesces_identical_requests(self):
        coalescer = RequestCoalescer()
        call_count = 0

        async def slow_producer():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "result"

        results = await asyncio.gather(
            coalescer.coalesce("m", "same", slow_producer),
            coalescer.coalesce("m", "same", slow_producer),
            coalescer.coalesce("m", "same", slow_producer),
        )
        assert all(r == "result" for r in results)
        assert call_count == 1
        assert coalescer.coalesced_count == 2

    async def test_different_prompts_not_coalesced(self):
        coalescer = RequestCoalescer()
        call_count = 0

        async def producer():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"result-{call_count}"

        await asyncio.gather(
            coalescer.coalesce("m", "prompt_a", producer),
            coalescer.coalesce("m", "prompt_b", producer),
        )
        assert call_count == 2
        assert coalescer.coalesced_count == 0

    async def test_producer_error_propagates(self):
        coalescer = RequestCoalescer()

        async def failing_producer():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await coalescer.coalesce("m", "hello", failing_producer)

    async def test_error_propagates_to_coalesced(self):
        coalescer = RequestCoalescer()

        async def failing_producer():
            await asyncio.sleep(0.05)
            raise RuntimeError("fail")

        results = await asyncio.gather(
            coalescer.coalesce("m", "same", failing_producer),
            coalescer.coalesce("m", "same", failing_producer),
            return_exceptions=True,
        )
        assert all(isinstance(r, RuntimeError) for r in results)

    async def test_in_flight_count(self):
        coalescer = RequestCoalescer()
        event = asyncio.Event()

        async def blocking_producer():
            await event.wait()
            return "done"

        task = asyncio.create_task(coalescer.coalesce("m", "p", blocking_producer))
        await asyncio.sleep(0.01)
        assert coalescer.in_flight_count == 1
        event.set()
        await task
        await asyncio.sleep(0.01)
        assert coalescer.in_flight_count == 0
