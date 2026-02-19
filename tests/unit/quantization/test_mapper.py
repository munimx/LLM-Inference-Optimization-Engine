import pytest

from llm_inference_engine.quantization.mapper import QuantizationMapper, UserPreference
from llm_inference_engine.quantization.types import (
    PreferencePriority,
    QualityTier,
    QuantizationLevel,
    QuantizedModelInfo,
)


@pytest.fixture
def models() -> list[QuantizedModelInfo]:
    return [
        QuantizedModelInfo(
            name="llama3.1:8b-instruct-q4_K_M",
            family="llama3.1",
            quantization=QuantizationLevel.Q4_K_M,
            size_bytes=4_000_000_000,
            memory_estimate_gb=4.5,
            parameters=8_000_000_000,
            context_length=8192,
            quality_tier=QualityTier.ACCEPTABLE,
            tokens_per_second=52.0,
            quality_score=0.95,
        ),
        QuantizedModelInfo(
            name="llama3.1:8b-instruct-q8_0",
            family="llama3.1",
            quantization=QuantizationLevel.Q8_0,
            size_bytes=8_000_000_000,
            memory_estimate_gb=8.8,
            parameters=8_000_000_000,
            context_length=8192,
            quality_tier=QualityTier.EXCELLENT,
            tokens_per_second=28.0,
            quality_score=1.0,
        ),
    ]


def test_rank_models_speed_priority(models: list[QuantizedModelInfo]) -> None:
    mapper = QuantizationMapper(models=models)
    ranked = mapper.rank_models(
        UserPreference(priority=PreferencePriority.SPEED, model_family="llama3.1")
    )
    assert ranked[0][0].name.endswith("q4_K_M")


def test_select_model_quality_priority(models: list[QuantizedModelInfo]) -> None:
    mapper = QuantizationMapper(models=models)
    selected = mapper.select_model(
        UserPreference(priority=PreferencePriority.QUALITY, model_family="llama3.1")
    )
    assert selected.name.endswith("q8_0")


def test_constraints_filtering(models: list[QuantizedModelInfo]) -> None:
    mapper = QuantizationMapper(models=models)
    ranked = mapper.rank_models(
        UserPreference(
            priority=PreferencePriority.BALANCED, model_family="llama3.1", max_memory_gb=5.0
        )
    )
    assert len(ranked) == 1
    assert ranked[0][0].name.endswith("q4_K_M")


def test_select_model_no_candidates(models: list[QuantizedModelInfo]) -> None:
    mapper = QuantizationMapper(models=models)
    with pytest.raises(ValueError):
        mapper.select_model(
            UserPreference(
                priority=PreferencePriority.SPEED,
                model_family="llama3.1",
                min_tokens_per_second=1000.0,
            )
        )
