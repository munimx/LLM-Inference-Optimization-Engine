"""Quality metrics and quality benchmarking for quantized models."""

from __future__ import annotations

import math
from statistics import mean

from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
    SmoothingFunction,
    sentence_bleu,
)
from rouge_score import rouge_scorer  # type: ignore[import-untyped]

from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.quantization.types import QualityScore, QualityTestCase


class QualityMetricsCalculator:
    """Calculator for text quality metrics."""

    def calculate_perplexity(self, text: str, reference_probs: list[float]) -> float:
        """Calculate perplexity from token probabilities."""
        if not text or not reference_probs:
            return float("inf")
        safe_probs = [max(prob, 1e-12) for prob in reference_probs]
        return math.exp(-sum(math.log(prob) for prob in safe_probs) / len(safe_probs))

    def calculate_bleu(self, candidate: str, references: list[str]) -> float:
        """Calculate sentence BLEU."""
        if not candidate or not references:
            return 0.0
        candidate_tokens = candidate.split()
        reference_tokens = [reference.split() for reference in references]
        smoothing = SmoothingFunction().method1
        return float(
            sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
        )

    def calculate_rouge(self, candidate: str, references: list[str]) -> dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2 and ROUGE-L F1 scores."""
        if not candidate or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(reference, candidate) for reference in references]
        return {
            "rouge1": float(mean(score["rouge1"].fmeasure for score in scores)),
            "rouge2": float(mean(score["rouge2"].fmeasure for score in scores)),
            "rougeL": float(mean(score["rougeL"].fmeasure for score in scores)),
        }

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple token-overlap semantic similarity."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union


class QualityBenchmark:
    """Run quality benchmark comparisons across models."""

    def __init__(self, client: OllamaClient, test_dataset: list[QualityTestCase]) -> None:
        """Initialize quality benchmark runner."""
        self._client = client
        self._test_dataset = test_dataset
        self._calculator = QualityMetricsCalculator()

    async def run_quality_comparison(self, models: list[str]) -> dict[str, QualityScore]:
        """Run quality benchmark for each model and return aggregated scores."""
        results: dict[str, QualityScore] = {}
        for model in models:
            bleu_scores: list[float] = []
            rouge1_scores: list[float] = []
            semantic_scores: list[float] = []
            for test_case in self._test_dataset:
                response = await self._client.generate(
                    model=model,
                    prompt=test_case.prompt,
                    max_tokens=test_case.max_tokens,
                )
                generated = str(response.get("response", ""))
                bleu = self._calculator.calculate_bleu(generated, test_case.reference_outputs)
                rouge = self._calculator.calculate_rouge(generated, test_case.reference_outputs)
                semantic = mean(
                    self._calculator.calculate_semantic_similarity(generated, reference)
                    for reference in test_case.reference_outputs
                )
                bleu_scores.append(bleu)
                rouge1_scores.append(rouge["rouge1"])
                semantic_scores.append(semantic)

            overall = (
                (mean(bleu_scores) * 0.4)
                + (mean(rouge1_scores) * 0.3)
                + (mean(semantic_scores) * 0.3)
            )
            results[model] = QualityScore(
                model_name=model,
                bleu=mean(bleu_scores),
                rouge={"rouge1": mean(rouge1_scores)},
                semantic_similarity=mean(semantic_scores),
                perplexity=None,
                overall_score=overall,
            )
        return results

    async def compare_to_baseline(self, model: str, baseline_model: str) -> float:
        """Compare model quality to a baseline model and return retention ratio."""
        scores = await self.run_quality_comparison([model, baseline_model])
        baseline_score = scores[baseline_model].overall_score
        if baseline_score <= 0:
            return 0.0
        return scores[model].overall_score / baseline_score
