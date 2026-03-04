"""Unit tests for VLLMBackend using respx to mock httpx."""

import httpx
import pytest
import respx

from llm_inference_engine.integration.backend import BackendResult
from llm_inference_engine.integration.vllm_backend import VLLMBackend


@pytest.fixture
def backend() -> VLLMBackend:
    return VLLMBackend("http://vllm-test:8080", timeout=5.0, max_retries=0)


class TestVLLMBackendIsAvailable:
    @respx.mock
    async def test_returns_true_on_200(self, backend: VLLMBackend) -> None:
        respx.get("http://vllm-test:8080/health/ready").mock(return_value=httpx.Response(200))
        assert await backend.is_available() is True

    @respx.mock
    async def test_returns_false_on_503(self, backend: VLLMBackend) -> None:
        respx.get("http://vllm-test:8080/health/ready").mock(return_value=httpx.Response(503))
        assert await backend.is_available() is False

    @respx.mock
    async def test_returns_false_on_connection_error(self, backend: VLLMBackend) -> None:
        respx.get("http://vllm-test:8080/health/ready").mock(side_effect=httpx.ConnectError("down"))
        assert await backend.is_available() is False


class TestVLLMBackendGenerate:
    @respx.mock
    async def test_successful_generation(self, backend: VLLMBackend) -> None:
        respx.post("http://vllm-test:8080/v1/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"text": "hello world", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 3},
            })
        )
        result = await backend.generate("llama3", "Say hi")
        assert isinstance(result, BackendResult)
        assert result.text == "hello world"
        assert result.tokens_used == 5
        assert result.prompt_tokens == 3
        assert result.finish_reason == "stop"

    @respx.mock
    async def test_raises_on_4xx(self, backend: VLLMBackend) -> None:
        respx.post("http://vllm-test:8080/v1/completions").mock(
            return_value=httpx.Response(400, json={"error": "bad request"})
        )
        with pytest.raises(RuntimeError, match="vLLM returned 400"):
            await backend.generate("llama3", "Hello")


class TestVLLMBackendChat:
    @respx.mock
    async def test_successful_chat(self, backend: VLLMBackend) -> None:
        respx.post("http://vllm-test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "Hi there!"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3, "prompt_tokens": 4},
            })
        )
        result = await backend.chat("llama3", [{"role": "user", "content": "Hello"}])
        assert result.text == "Hi there!"
        assert result.tokens_used == 3


class TestVLLMBackendListModels:
    @respx.mock
    async def test_returns_model_ids(self, backend: VLLMBackend) -> None:
        respx.get("http://vllm-test:8080/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [{"id": "llama3"}, {"id": "mistral"}]
            })
        )
        models = await backend.list_models()
        assert "llama3" in models
        assert "mistral" in models

    @respx.mock
    async def test_returns_empty_list_on_error(self, backend: VLLMBackend) -> None:
        respx.get("http://vllm-test:8080/v1/models").mock(
            side_effect=httpx.ConnectError("down")
        )
        models = await backend.list_models()
        assert models == []


class TestVLLMBackendClose:
    async def test_close_releases_client(self, backend: VLLMBackend) -> None:
        backend._client = httpx.AsyncClient()
        await backend.close()
        assert backend._client is None

    async def test_close_is_idempotent_when_no_client(self, backend: VLLMBackend) -> None:
        assert backend._client is None
        await backend.close()  # Should not raise


class TestVLLMBackendStopTokens:
    @respx.mock
    async def test_generate_passes_stop_tokens(self, backend: VLLMBackend) -> None:
        respx.post("http://vllm-test:8080/v1/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"text": "done", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1, "prompt_tokens": 2},
            })
        )
        result = await backend.generate("llama3", "Hello", stop=["<|end|>"])
        assert result.text == "done"

    @respx.mock
    async def test_chat_passes_stop_tokens(self, backend: VLLMBackend) -> None:
        respx.post("http://vllm-test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "bye"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1, "prompt_tokens": 2},
            })
        )
        result = await backend.chat(
            "llama3", [{"role": "user", "content": "bye"}], stop=["<|end|>"]
        )
        assert result.text == "bye"


class TestVLLMBackendRetry:
    async def test_raises_after_all_retries_exhausted(self) -> None:
        retrying = VLLMBackend("http://vllm-test:8080", timeout=1.0, max_retries=1, retry_backoff_seconds=0.0)
        with respx.mock:
            respx.post("http://vllm-test:8080/v1/completions").mock(
                side_effect=httpx.ConnectError("down")
            )
            with pytest.raises(RuntimeError, match="failed after"):
                await retrying.generate("llama3", "hello")
        await retrying.close()


class TestVLLMBackendStream:
    @respx.mock
    async def test_generate_stream_yields_chunks(self, backend: VLLMBackend) -> None:
        sse_lines = (
            'data: {"choices": [{"text": "hello", "finish_reason": null}]}\n\n'
            'data: {"choices": [{"text": " world", "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("http://vllm-test:8080/v1/completions").mock(
            return_value=httpx.Response(200, text=sse_lines)
        )
        chunks = []
        async for chunk in backend.generate_stream("llama3", "Say hi"):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello"
        assert chunks[1]["done"] is True

    @respx.mock
    async def test_chat_stream_yields_chunks(self, backend: VLLMBackend) -> None:
        sse_lines = (
            'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
            'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("http://vllm-test:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse_lines)
        )
        chunks = []
        async for chunk in backend.chat_stream("llama3", [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0]["content"] == "Hi"
        assert chunks[1]["done"] is True
