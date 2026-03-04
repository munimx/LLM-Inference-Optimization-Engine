"""Unit tests for RequestCoalescer using fakeredis."""

import pytest
import fakeredis.aioredis

from llm_inference_engine.api.coalescer import RequestCoalescer


@pytest.fixture
async def redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
async def coalescer(redis):
    return RequestCoalescer(redis_client=redis)


class TestRequestCoalescer:
    async def test_single_request_calls_producer(self, coalescer: RequestCoalescer) -> None:
        calls = []

        async def producer():
            calls.append(1)
            return "result"

        result = await coalescer.coalesce("llama3", "hello", producer)
        assert result == "result"
        assert len(calls) == 1

    async def test_initial_coalesced_count_zero(self, coalescer: RequestCoalescer) -> None:
        assert coalescer.coalesced_count == 0

    async def test_producer_exception_propagates(self, coalescer: RequestCoalescer) -> None:
        async def failing_producer():
            raise ValueError("inference failed")

        with pytest.raises(ValueError, match="inference failed"):
            await coalescer.coalesce("llama3", "test", failing_producer)

    async def test_different_prompts_call_producer_each_time(
        self, coalescer: RequestCoalescer
    ) -> None:
        calls = []

        async def producer():
            calls.append(1)
            return "result"

        await coalescer.coalesce("llama3", "prompt1", producer)
        await coalescer.coalesce("llama3", "prompt2", producer)
        assert len(calls) == 2
