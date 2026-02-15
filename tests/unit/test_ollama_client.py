import pytest
import respx
import httpx
from unittest.mock import patch
from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.exceptions import OllamaConnectionError, OllamaTimeoutError, ModelNotFoundError

@pytest.mark.asyncio
async def test_ollama_client_is_available(respx_mock):
    respx_mock.get("http://localhost:11434/").mock(return_value=httpx.Response(200))
    
    async with OllamaClient() as client:
        assert await client.is_available()

@pytest.mark.asyncio
async def test_ollama_client_is_unavailable(respx_mock):
    respx_mock.get("http://localhost:11434/").mock(side_effect=httpx.ConnectError("Connection refused"))
    
    async with OllamaClient() as client:
        assert not await client.is_available()

@pytest.mark.asyncio
async def test_list_models_success(respx_mock):
    models_data = {"models": [{"name": "llama2", "size": 1000}]}
    respx_mock.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json=models_data))
    
    async with OllamaClient() as client:
        models = await client.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "llama2"

@pytest.mark.asyncio
async def test_list_models_failure(respx_mock):
    respx_mock.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(500))
    
    async with OllamaClient() as client:
        with pytest.raises(OllamaConnectionError):
            await client.list_models()

@pytest.mark.asyncio
async def test_generate_success(respx_mock):
    respx_mock.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "Hello", "eval_count": 5})
    )
    
    async with OllamaClient() as client:
        result = await client.generate("llama2", "Hi")
        assert result["response"] == "Hello"

@pytest.mark.asyncio
async def test_generate_retry_on_timeout(respx_mock):
    # First call times out, second succeeds
    route = respx_mock.post("http://localhost:11434/api/generate")
    route.side_effect = [
        httpx.ReadTimeout("Timeout"),
        httpx.Response(200, json={"response": "Success"})
    ]
    
    client = OllamaClient(max_retries=2, timeout=0.1) # low timeout for test speed
    # We mock sleep to avoid waiting
    with patch("asyncio.sleep", return_value=None):
        async with client:
            result = await client.generate("llama2", "Hi")
            assert result["response"] == "Success"
            assert route.call_count == 2

@pytest.mark.asyncio
async def test_generate_retry_on_5xx(respx_mock):
    # First call 503, second succeeds
    route = respx_mock.post("http://localhost:11434/api/generate")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json={"response": "Success"})
    ]
    
    client = OllamaClient(max_retries=2)
    with patch("asyncio.sleep", return_value=None):
        async with client:
            result = await client.generate("llama2", "Hi")
            assert result["response"] == "Success"
            assert route.call_count == 2

@pytest.mark.asyncio
async def test_generate_fail_after_retries(respx_mock):
    route = respx_mock.post("http://localhost:11434/api/generate")
    route.side_effect = httpx.ConnectError("Failed")
    
    client = OllamaClient(max_retries=2)
    with patch("asyncio.sleep", return_value=None):
        async with client:
            with pytest.raises(OllamaConnectionError):
                await client.generate("llama2", "Hi")
    assert route.call_count == 2

@pytest.mark.asyncio
async def test_generate_timeout_raises_timeout_error(respx_mock):
    respx_mock.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.ReadTimeout("Timeout")
    )

    client = OllamaClient(max_retries=2, timeout=0.1)
    with patch("asyncio.sleep", return_value=None):
        async with client:
            with pytest.raises(OllamaTimeoutError):
                await client.generate("llama2", "Hi")

@pytest.mark.asyncio
async def test_generate_does_not_retry_on_4xx(respx_mock):
    route = respx_mock.post("http://localhost:11434/api/generate")
    route.side_effect = [httpx.Response(400, json={"error": "bad request"})]

    client = OllamaClient(max_retries=3)
    async with client:
        with pytest.raises(OllamaConnectionError):
            await client.generate("llama2", "Hi")
    assert route.call_count == 1
