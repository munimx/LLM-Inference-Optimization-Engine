"""Unit tests for VLLMBackend using respx to mock httpx."""

import pytest
import respx
import httpx

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
