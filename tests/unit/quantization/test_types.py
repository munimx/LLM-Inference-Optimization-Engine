from datetime import datetime

import pytest

from llm_inference_engine.quantization.types import (
    BenchmarkConfig,
    BenchmarkResult,
    PreferencePriority,
    QualityScore,
    QualityTestCase,
    QualityTier,
    QuantizationLevel,
    QuantizedModelInfo,
)


def test_quantization_enum_values() -> None:
    assert QuantizationLevel.Q4_K_M.value == "q4_K_M"
    assert PreferencePriority.BALANCED.value == "balanced"
    assert QualityTier.EXCELLENT.value == "excellent"


def test_quantized_model_info_validation() -> None:
    with pytest.raises(ValueError):
        QuantizedModelInfo(
            name="",
            family="llama",
            quantization=QuantizationLevel.Q4_0,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
        )


def test_quantized_model_info_serialization() -> None:
    info = QuantizedModelInfo(
        name="llama3.1:8b-instruct-q4_K_M",
        family="llama3.1",
        quantization=QuantizationLevel.Q4_K_M,
        size_bytes=4,
        memory_estimate_gb=4.8,
        parameters=8_000_000_000,
        context_length=8192,
        quality_tier=QualityTier.ACCEPTABLE,
    )
    deserialized = QuantizedModelInfo.from_dict(info.to_dict())
    assert deserialized.name == info.name
    assert deserialized.quantization == QuantizationLevel.Q4_K_M


def test_quality_test_case_validation() -> None:
    with pytest.raises(ValueError):
        QualityTestCase(prompt="", reference_outputs=["test"], task_type="factual")


def test_benchmark_config_validation() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(prompt="x", max_tokens=0)


def test_benchmark_result_round_trip() -> None:
    result = BenchmarkResult(
        model_name="mistral:7b-q4_0",
        quantization="q4_0",
        tokens_per_second=52.3,
        time_to_first_token_ms=101.1,
        total_latency_ms=900.0,
        memory_peak_mb=4500.0,
        memory_baseline_mb=1200.0,
        generated_text="Hello",
        prompt_tokens=16,
        completion_tokens=24,
        timestamp=datetime.utcnow(),
    )
    reloaded = BenchmarkResult.from_dict(result.to_dict())
    assert reloaded.model_name == result.model_name
    assert reloaded.tokens_per_second == result.tokens_per_second


def test_quality_score_to_dict() -> None:
    score = QualityScore(
        model_name="mistral:7b-q4_0",
        bleu=0.7,
        rouge={"rouge1": 0.8},
        semantic_similarity=0.9,
        perplexity=None,
        overall_score=0.8,
    )
    assert score.to_dict()["overall_score"] == 0.8
