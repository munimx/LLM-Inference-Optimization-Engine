from unittest.mock import AsyncMock

import pytest

from llm_inference_engine.integration.ollama_models import ModelInfo, OllamaModelManager
from llm_inference_engine.quantization.collector import QuantizationInfoCollector
from llm_inference_engine.quantization.types import QualityTier, QuantizationLevel


@pytest.fixture
def model_manager() -> AsyncMock:
    manager = AsyncMock(spec=OllamaModelManager)
    manager.get_available_models.return_value = [
        ModelInfo(
            name="llama3.1:8b-instruct-q4_K_M",
            size_bytes=4_000_000_000,
            quantization="4-bit",
            family="llama3.1",
            format="gguf",
        ),
        ModelInfo(
            name="llama3.1:8b-instruct-q8_0",
            size_bytes=8_000_000_000,
            quantization="8-bit",
            family="llama3.1",
            format="gguf",
        ),
        ModelInfo(
            name="mistral:7b-fp16",
            size_bytes=14_000_000_000,
            quantization="fp16",
            family="mistral",
            format="gguf",
        ),
    ]
    return manager


@pytest.mark.asyncio
async def test_collect_all_quantizations_uses_cache(model_manager: AsyncMock) -> None:
    collector = QuantizationInfoCollector(client=AsyncMock(), model_manager=model_manager)
    first = await collector.collect_all_quantizations()
    second = await collector.collect_all_quantizations()
    assert len(first) == 3
    assert first == second
    assert model_manager.get_available_models.call_count == 1


@pytest.mark.asyncio
async def test_collect_model_family(model_manager: AsyncMock) -> None:
    collector = QuantizationInfoCollector(client=AsyncMock(), model_manager=model_manager)
    models = await collector.collect_model_family("llama3.1")
    assert len(models) == 2
    assert all(model.family == "llama3.1" for model in models)


@pytest.mark.asyncio
async def test_get_quantization_variants(model_manager: AsyncMock) -> None:
    collector = QuantizationInfoCollector(client=AsyncMock(), model_manager=model_manager)
    variants = await collector.get_quantization_variants("llama3.1:8b-instruct")
    assert len(variants) == 2
    assert {item.quantization for item in variants} == {
        QuantizationLevel.Q4_K_M,
        QuantizationLevel.Q8_0,
    }


def test_parse_quantization_level_fallback(model_manager: AsyncMock) -> None:
    collector = QuantizationInfoCollector(client=AsyncMock(), model_manager=model_manager)
    assert collector._parse_quantization_level("custom-model-no-quant") == QuantizationLevel.UNKNOWN


def test_quality_tier_estimation(model_manager: AsyncMock) -> None:
    collector = QuantizationInfoCollector(client=AsyncMock(), model_manager=model_manager)
    assert collector._estimate_quality_tier(QuantizationLevel.FP16) == QualityTier.EXCELLENT
    assert collector._estimate_quality_tier(QuantizationLevel.Q5_K_S) == QualityTier.GOOD
    assert collector._estimate_quality_tier(QuantizationLevel.Q4_0) == QualityTier.ACCEPTABLE
    assert collector._estimate_quality_tier(QuantizationLevel.Q2_K) == QualityTier.DEGRADED
