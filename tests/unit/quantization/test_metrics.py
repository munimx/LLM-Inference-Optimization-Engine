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
