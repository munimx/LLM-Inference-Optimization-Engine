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


def test_quality_priority_not_overridden_by_tiny_memory() -> None:
    mapper = QuantizationMapper(
        models=[
            QuantizedModelInfo(
                name="model-a-q4_0",
                family="model-a",
                quantization=QuantizationLevel.Q4_0,
                size_bytes=500_000_000,
                memory_estimate_gb=0.2,
                parameters=None,
                context_length=4096,
                quality_tier=QualityTier.ACCEPTABLE,
                tokens_per_second=10.0,
                quality_score=0.2,
            ),
            QuantizedModelInfo(
                name="model-a-q8_0",
                family="model-a",
                quantization=QuantizationLevel.Q8_0,
                size_bytes=4_000_000_000,
                memory_estimate_gb=4.0,
                parameters=None,
                context_length=4096,
                quality_tier=QualityTier.EXCELLENT,
                tokens_per_second=10.0,
                quality_score=0.95,
            ),
        ]
    )
    selected = mapper.select_model(
        UserPreference(priority=PreferencePriority.QUALITY, model_family="model-a")
    )
    assert selected.name == "model-a-q8_0"


def _make_model(
    name: str,
    family: str,
    quant: QuantizationLevel,
    quality: float,
    tps: float,
    mem_gb: float,
) -> QuantizedModelInfo:
    return QuantizedModelInfo(
        name=name,
        family=family,
        quantization=quant,
        size_bytes=int(mem_gb * 1_000_000_000),
        memory_estimate_gb=mem_gb,
        parameters=7_000_000_000,
        context_length=4096,
        quality_tier=QualityTier.GOOD,
        tokens_per_second=tps,
        quality_score=quality,
    )


class TestQuantizationMapper:
    def test_select_speed_prefers_fastest(self) -> None:
        slow = _make_model("m:q8_0", "m", QuantizationLevel.Q8_0, 0.95, 20.0, 7.0)
        fast = _make_model("m:q4_K_M", "m", QuantizationLevel.Q4_K_M, 0.85, 60.0, 4.0)
        mapper = QuantizationMapper([slow, fast])
        selected = mapper.select_model(UserPreference(priority=PreferencePriority.SPEED, model_family="m"))
        assert selected.name == "m:q4_K_M"

    def test_select_quality_prefers_best_score(self) -> None:
        low = _make_model("m:q4_0", "m", QuantizationLevel.Q4_0, 0.7, 60.0, 3.0)
        high = _make_model("m:q8_0", "m", QuantizationLevel.Q8_0, 0.98, 25.0, 7.0)
        mapper = QuantizationMapper([low, high])
        selected = mapper.select_model(UserPreference(priority=PreferencePriority.QUALITY, model_family="m"))
        assert selected.name == "m:q8_0"

    def test_select_memory_efficient_prefers_smallest_memory(self) -> None:
        big = _make_model("m:q8_0", "m", QuantizationLevel.Q8_0, 0.95, 25.0, 8.0)
        small = _make_model("m:q4_K_M", "m", QuantizationLevel.Q4_K_M, 0.85, 50.0, 4.0)
        mapper = QuantizationMapper([big, small])
        selected = mapper.select_model(
            UserPreference(priority=PreferencePriority.MEMORY_EFFICIENT, model_family="m")
        )
        assert selected.name == "m:q4_K_M"

    def test_select_no_match_raises_value_error(self) -> None:
        model = _make_model("m:q4_K_M", "m", QuantizationLevel.Q4_K_M, 0.85, 10.0, 4.0)
        mapper = QuantizationMapper([model])
        with pytest.raises(ValueError, match="No models"):
            mapper.select_model(
                UserPreference(priority=PreferencePriority.SPEED, min_tokens_per_second=500.0)
            )

    def test_filter_by_max_memory(self) -> None:
        models = [
            _make_model("m:q4", "m", QuantizationLevel.Q4_K_M, 0.85, 50.0, 4.0),
            _make_model("m:q8", "m", QuantizationLevel.Q8_0, 0.95, 25.0, 8.0),
        ]
        mapper = QuantizationMapper(models)
        ranked = mapper.rank_models(UserPreference(
            priority=PreferencePriority.BALANCED, max_memory_gb=5.0
        ))
        assert all(m.memory_estimate_gb <= 5.0 for m, _ in ranked)

    def test_filter_by_min_tps(self) -> None:
        models = [
            _make_model("fast", "m", QuantizationLevel.Q4_K_M, 0.85, 60.0, 4.0),
            _make_model("slow", "m", QuantizationLevel.Q8_0, 0.95, 10.0, 7.0),
        ]
        mapper = QuantizationMapper(models)
        ranked = mapper.rank_models(UserPreference(
            priority=PreferencePriority.SPEED, min_tokens_per_second=30.0
        ))
        assert all(m.tokens_per_second >= 30.0 for m, _ in ranked)  # type: ignore[operator]

    def test_filter_by_family(self) -> None:
        models = [
            _make_model("llama:q4", "llama", QuantizationLevel.Q4_K_M, 0.85, 50.0, 4.0),
            _make_model("mistral:q4", "mistral", QuantizationLevel.Q4_K_M, 0.88, 55.0, 4.5),
        ]
        mapper = QuantizationMapper(models)
        ranked = mapper.rank_models(UserPreference(
            priority=PreferencePriority.BALANCED, model_family="llama"
        ))
        assert all(m.family == "llama" for m, _ in ranked)

    def test_get_recommendations_returns_four_keys(self) -> None:
        models = [
            _make_model("m:q4", "m", QuantizationLevel.Q4_K_M, 0.85, 50.0, 4.0),
            _make_model("m:q8", "m", QuantizationLevel.Q8_0, 0.95, 25.0, 7.0),
        ]
        mapper = QuantizationMapper(models)
        recs = mapper.get_recommendations("m")
        assert set(recs.keys()) == {"speed", "balanced", "quality", "memory"}

    def test_get_recommendations_no_family_match(self) -> None:
        models = [_make_model("m:q4", "m", QuantizationLevel.Q4_K_M, 0.85, 50.0, 4.0)]
        mapper = QuantizationMapper(models)
        recs = mapper.get_recommendations("nonexistent-family")
        assert recs == {}

    def test_rank_models_returns_sorted_descending(self) -> None:
        models = [
            _make_model("m:q4", "m", QuantizationLevel.Q4_K_M, 0.85, 60.0, 4.0),
            _make_model("m:q8", "m", QuantizationLevel.Q8_0, 0.98, 25.0, 7.0),
        ]
        mapper = QuantizationMapper(models)
        ranked = mapper.rank_models(UserPreference(priority=PreferencePriority.QUALITY, model_family="m"))
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_select_balanced_returns_model(self) -> None:
        models = [
            _make_model("m:q4", "m", QuantizationLevel.Q4_K_M, 0.85, 60.0, 4.0),
        ]
        mapper = QuantizationMapper(models)
        result = mapper.select_model(UserPreference(priority=PreferencePriority.BALANCED))
        assert result.name == "m:q4"
