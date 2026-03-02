from unittest.mock import AsyncMock

import pytest

from llm_inference_engine.quantization.metrics import QualityBenchmark, QualityMetricsCalculator
from llm_inference_engine.quantization.types import QualityTestCase


def test_bleu_and_rouge_calculation() -> None:
    calculator = QualityMetricsCalculator()
    candidate = "the cat is on the mat"
    references = ["the cat sat on the mat"]
    bleu = calculator.calculate_bleu(candidate, references)
    rouge = calculator.calculate_rouge(candidate, references)
    assert bleu > 0.0
    assert rouge["rouge1"] > 0.0


def test_semantic_similarity() -> None:
    calculator = QualityMetricsCalculator()
    similarity = calculator.calculate_semantic_similarity("hello world", "hello there world")
    assert 0.0 < similarity <= 1.0


@pytest.mark.asyncio
async def test_quality_benchmark() -> None:
    client = AsyncMock()
    client.generate.return_value = {"response": "Tokyo is the capital of Japan."}
    benchmark = QualityBenchmark(
        client=client,
        test_dataset=[
            QualityTestCase(
                prompt="What is the capital of Japan?",
                reference_outputs=["Tokyo is the capital of Japan."],
                task_type="factual",
                max_tokens=32,
            )
        ],
    )
    scores = await benchmark.run_quality_comparison(["model-a-q4_0"])
    assert "model-a-q4_0" in scores
    assert scores["model-a-q4_0"].overall_score > 0


@pytest.mark.asyncio
async def test_compare_to_baseline() -> None:
    client = AsyncMock()
    client.generate.return_value = {"response": "answer"}
    benchmark = QualityBenchmark(
        client=client,
        test_dataset=[
            QualityTestCase(
                prompt="Question?",
                reference_outputs=["answer"],
                task_type="factual",
                max_tokens=8,
            )
        ],
    )
    ratio = await benchmark.compare_to_baseline("model-a", "model-b")
    assert ratio >= 0.0


def test_quality_benchmark_requires_dataset() -> None:
    client = AsyncMock()
    with pytest.raises(ValueError, match="test_dataset"):
        QualityBenchmark(client=client, test_dataset=[])


class TestQualityMetricsCalculator:
    def test_bleu_empty_references_returns_zero(self) -> None:
        calc = QualityMetricsCalculator()
        score = calc.calculate_bleu("candidate text", [])
        assert score == 0.0

    def test_bleu_identical_candidate_and_reference_high_score(self) -> None:
        calc = QualityMetricsCalculator()
        text = "the quick brown fox jumps"
        score = calc.calculate_bleu(text, [text])
        assert score > 0.5

    def test_rouge_empty_references_returns_zeros(self) -> None:
        calc = QualityMetricsCalculator()
        scores = calc.calculate_rouge("candidate", [])
        assert all(v == 0.0 for v in scores.values())

    def test_rouge_identical_texts_all_ones(self) -> None:
        calc = QualityMetricsCalculator()
        text = "the cat sat on the mat"
        scores = calc.calculate_rouge(text, [text])
        assert scores["rouge1"] == pytest.approx(1.0, abs=1e-3)

    def test_rouge_returns_rouge1_and_rouge2(self) -> None:
        calc = QualityMetricsCalculator()
        scores = calc.calculate_rouge("hello world", ["hello world"])
        assert "rouge1" in scores
        assert "rouge2" in scores

    def test_semantic_similarity_identical_texts_is_one(self) -> None:
        calc = QualityMetricsCalculator()
        sim = calc.calculate_semantic_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_semantic_similarity_disjoint_texts_less_than_one(self) -> None:
        calc = QualityMetricsCalculator()
        sim = calc.calculate_semantic_similarity("hello world", "foo bar baz qux")
        assert sim < 1.0

    def test_semantic_similarity_bounded_between_zero_and_one(self) -> None:
        calc = QualityMetricsCalculator()
        sim = calc.calculate_semantic_similarity("abc def", "xyz uvw")
        assert 0.0 <= sim <= 1.0

    def test_perplexity_all_perfect_probs_is_one(self) -> None:
        calc = QualityMetricsCalculator()
        perp = calc.calculate_perplexity("hello world", [1.0, 1.0])
        assert perp == pytest.approx(1.0, abs=1e-6)

    def test_perplexity_low_probs_gives_high_perplexity(self) -> None:
        calc = QualityMetricsCalculator()
        perp = calc.calculate_perplexity("test text", [0.01, 0.01])
        assert perp > 1.0

    def test_perplexity_empty_probs_raises_or_returns_special(self) -> None:
        calc = QualityMetricsCalculator()
        # Should either raise or return inf/0 — must not crash silently
        try:
            result = calc.calculate_perplexity("text", [])
            assert result >= 0.0
        except (ValueError, ZeroDivisionError):
            pass  # acceptable


class TestQualityBenchmarkAdditional:
    async def test_benchmark_run_multiple_models(self) -> None:
        client = AsyncMock()
        client.generate.return_value = {"response": "Paris"}
        benchmark = QualityBenchmark(
            client=client,
            test_dataset=[
                QualityTestCase(
                    prompt="Capital of France?",
                    reference_outputs=["Paris"],
                    task_type="factual",
                    max_tokens=4,
                )
            ],
        )
        scores = await benchmark.run_quality_comparison(["m:q4", "m:q8"])
        assert "m:q4" in scores
        assert "m:q8" in scores

    async def test_benchmark_overall_score_between_zero_and_one(self) -> None:
        client = AsyncMock()
        client.generate.return_value = {"response": "some answer"}
        benchmark = QualityBenchmark(
            client=client,
            test_dataset=[
                QualityTestCase(
                    prompt="Test",
                    reference_outputs=["answer"],
                    task_type="qa",
                    max_tokens=8,
                )
            ],
        )
        scores = await benchmark.run_quality_comparison(["m:q4"])
        score = scores["m:q4"].overall_score
        assert 0.0 <= score <= 1.0

    async def test_compare_to_baseline_returns_float(self) -> None:
        client = AsyncMock()
        client.generate.return_value = {"response": "answer"}
        benchmark = QualityBenchmark(
            client=client,
            test_dataset=[
                QualityTestCase(
                    prompt="Question?",
                    reference_outputs=["answer"],
                    task_type="factual",
                    max_tokens=8,
                )
            ],
        )
        ratio = await benchmark.compare_to_baseline("model-a", "model-b")
        assert isinstance(ratio, float)
        assert ratio >= 0.0
