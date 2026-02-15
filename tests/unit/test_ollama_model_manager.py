import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock
from llm_inference_engine.integration.ollama_models import OllamaModelManager, ModelInfo
from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.exceptions import ModelNotFoundError

@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=OllamaClient)
    client.list_models.return_value = [
        {"name": "llama2:7b", "size": 3826028885, "details": {"family": "llama", "parameter_count": 7000000000}},
        {"name": "mistral:7b", "size": 4126028885}
    ]
    return client

@pytest.fixture
def manager(mock_client: AsyncMock) -> OllamaModelManager:
    return OllamaModelManager(mock_client)

@pytest.mark.asyncio
async def test_refresh_models(manager: OllamaModelManager) -> None:
    await manager.refresh_models()
    assert len(manager._model_cache) == 2
    assert "llama2:7b" in manager._model_cache
    assert "mistral:7b" in manager._model_cache

@pytest.mark.asyncio
async def test_get_available_models(manager: OllamaModelManager) -> None:
    models = await manager.get_available_models()
    assert len(models) == 2
    assert isinstance(models[0], ModelInfo)

@pytest.mark.asyncio
async def test_get_model_info_found(manager: OllamaModelManager) -> None:
    info = await manager.get_model_info("llama2:7b")
    assert info.name == "llama2:7b"
    assert info.family == "llama"
    # Check memory estimation logic
    # 3826028885 bytes ≈ 3.56 GiB; 3.56 * 1.2 ≈ 4.28 GiB
    assert info.memory_estimate_gb > 4.0

@pytest.mark.asyncio
async def test_get_model_info_not_found(manager: OllamaModelManager) -> None:
    with pytest.raises(ModelNotFoundError):
        await manager.get_model_info("nonexistent")

@pytest.mark.asyncio
async def test_verify_model_available(manager: OllamaModelManager) -> None:
    assert await manager.verify_model_available("llama2:7b")
    assert not await manager.verify_model_available("gpt4")
