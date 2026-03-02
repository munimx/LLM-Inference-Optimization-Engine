"""Tests for the abstract InferenceBackend and OllamaBackend adapter."""

from unittest.mock import AsyncMock

import pytest

from llm_inference_engine.integration.backend import BackendResult, InferenceBackend
from llm_inference_engine.integration.ollama_backend import OllamaBackend


class TestBackendResult:

    def test_defaults(self):
        r = BackendResult(text="hello")
        assert r.text == "hello"
        assert r.tokens_used == 0
        assert r.finish_reason == "stop"


class TestInferenceBackendABC:

    async def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InferenceBackend()


class TestOllamaBackend:

    async def test_generate(self):
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "response": "hello world",
            "eval_count": 5,
            "prompt_eval_count": 3,
            "done_reason": "stop",
        }
        backend = OllamaBackend(mock_client)
        result = await backend.generate("llama3", "Say hi")
        assert result.text == "hello world"
        assert result.tokens_used == 5
        assert result.prompt_tokens == 3

    async def test_chat(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "Hi!"},
            "eval_count": 2,
            "prompt_eval_count": 10,
            "done_reason": "stop",
        }
        backend = OllamaBackend(mock_client)
        result = await backend.chat("llama3", [{"role": "user", "content": "hi"}])
        assert result.text == "Hi!"
        assert result.tokens_used == 2

    async def test_is_available(self):
        mock_client = AsyncMock()
        mock_client.is_available.return_value = True
        backend = OllamaBackend(mock_client)
        assert await backend.is_available() is True

    async def test_list_models(self):
        mock_client = AsyncMock()
        mock_client.list_models.return_value = [
            {"name": "llama3"},
            {"name": "phi3"},
        ]
        backend = OllamaBackend(mock_client)
        models = await backend.list_models()
        assert models == ["llama3", "phi3"]

    async def test_close(self):
        mock_client = AsyncMock()
        backend = OllamaBackend(mock_client)
        await backend.close()
        mock_client.close.assert_awaited_once()
