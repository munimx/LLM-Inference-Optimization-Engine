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


def test_all_quantization_level_values_are_strings() -> None:
    """All QuantizationLevel enum values should be non-empty strings."""
    for level in QuantizationLevel:
        assert isinstance(level.value, str)
        assert len(level.value) > 0


def test_all_preference_priority_values() -> None:
    priorities = {p.value for p in PreferencePriority}
    assert "speed" in priorities
    assert "quality" in priorities
    assert "balanced" in priorities
    assert "memory" in priorities  # actual enum value is "memory" not "memory_efficient"


def test_all_quality_tier_values() -> None:
    tiers = {t.value for t in QualityTier}
    assert "excellent" in tiers
    assert "good" in tiers
    assert "acceptable" in tiers
    assert "degraded" in tiers


def test_quantized_model_info_to_dict_roundtrip() -> None:
    model = QuantizedModelInfo(
        name="llama3:8b-q4_K_M",
        family="llama3",
        quantization=QuantizationLevel.Q4_K_M,
        size_bytes=4_000_000_000,
        memory_estimate_gb=4.5,
        parameters=8_000_000_000,
        context_length=8192,
        quality_tier=QualityTier.ACCEPTABLE,
        tokens_per_second=52.0,
        quality_score=0.95,
    )
    assert model.name == "llama3:8b-q4_K_M"
    assert model.family == "llama3"


def test_quantized_model_info_invalid_name_raises() -> None:
    with pytest.raises(ValueError):
        QuantizedModelInfo(
            name="",
            family="llama3",
            quantization=QuantizationLevel.Q4_K_M,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
            tokens_per_second=1.0,
            quality_score=0.5,
        )


def test_quantized_model_info_negative_memory_raises() -> None:
    with pytest.raises(ValueError):
        QuantizedModelInfo(
            name="llama3:8b",
            family="llama3",
            quantization=QuantizationLevel.Q4_K_M,
            size_bytes=1,
            memory_estimate_gb=-1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
            tokens_per_second=1.0,
            quality_score=0.5,
        )


def test_benchmark_config_defaults() -> None:
    config = BenchmarkConfig(prompt="test prompt", max_tokens=256)
    assert config.warmup_runs >= 0
    assert config.num_runs > 0


def test_benchmark_config_invalid_num_runs_raises() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(prompt="test", max_tokens=256, num_runs=0)


def test_benchmark_result_tokens_per_second_positive() -> None:
    result = BenchmarkResult(
        model_name="test:model",
        quantization="q4_K_M",
        tokens_per_second=45.0,
        time_to_first_token_ms=85.0,
        total_latency_ms=500.0,
        memory_peak_mb=3000.0,
        memory_baseline_mb=1000.0,
        generated_text="Hello world",
        prompt_tokens=10,
        completion_tokens=20,
        timestamp=datetime.utcnow(),
    )
    assert result.tokens_per_second > 0


def test_benchmark_result_to_dict_has_model_name() -> None:
    result = BenchmarkResult(
        model_name="test:model",
        quantization="q4_K_M",
        tokens_per_second=45.0,
        time_to_first_token_ms=85.0,
        total_latency_ms=500.0,
        memory_peak_mb=3000.0,
        memory_baseline_mb=1000.0,
        generated_text="Hello world",
        prompt_tokens=10,
        completion_tokens=20,
        timestamp=datetime.utcnow(),
    )
    d = result.to_dict()
    assert d["model_name"] == "test:model"
    assert "tokens_per_second" in d


def test_benchmark_result_from_dict_roundtrip() -> None:
    ts = datetime.utcnow()
    result = BenchmarkResult(
        model_name="m:v",
        quantization="q8_0",
        tokens_per_second=30.0,
        time_to_first_token_ms=100.0,
        total_latency_ms=600.0,
        memory_peak_mb=5000.0,
        memory_baseline_mb=2000.0,
        generated_text="text",
        prompt_tokens=5,
        completion_tokens=10,
        timestamp=ts,
    )
    recovered = BenchmarkResult.from_dict(result.to_dict())
    assert recovered.quantization == "q8_0"
    assert recovered.tokens_per_second == 30.0


def test_quality_score_to_dict_includes_bleu() -> None:
    score = QualityScore(
        model_name="m:v",
        bleu=0.65,
        rouge={"rouge1": 0.7, "rouge2": 0.5},
        semantic_similarity=0.88,
        perplexity=25.0,
        overall_score=0.75,
    )
    d = score.to_dict()
    assert "bleu" in d
    assert d["bleu"] == pytest.approx(0.65)


def test_quality_score_none_perplexity_in_dict() -> None:
    score = QualityScore(
        model_name="m:v",
        bleu=0.5,
        rouge={},
        semantic_similarity=0.8,
        perplexity=None,
        overall_score=0.6,
    )
    d = score.to_dict()
    assert d.get("perplexity") is None


def test_quality_test_case_max_tokens_default() -> None:
    tc = QualityTestCase(
        prompt="What is 2+2?",
        reference_outputs=["4"],
        task_type="math",
    )
    assert tc.max_tokens > 0
