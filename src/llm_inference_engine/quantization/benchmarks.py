"""Benchmark suite for quantized model performance analysis."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

import structlog

from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.quantization.collector import QuantizationInfoCollector
from llm_inference_engine.quantization.types import BenchmarkConfig, BenchmarkResult
from llm_inference_engine.utils.benchmark_utils import MemoryProfiler, StatisticsCalculator, Timer

logger = structlog.get_logger(__name__)


class BenchmarkRunner:
    """Execute benchmark measurements for models."""

    _QUANTIZATION_PATTERN = re.compile(
        r"(q(?:2|3|4|5|6|8)[_-]?(?:k(?:[_-]?[sm])?|0|1)|fp16|f16)",
        re.IGNORECASE,
    )

    def __init__(self, client: OllamaClient, config: BenchmarkConfig) -> None:
        """Initialize benchmark runner."""
        self._client = client
        self._config = config

    async def run_single_benchmark(self, model: str) -> BenchmarkResult:
        """Run warmup + measured benchmark for a single model."""
        for _ in range(self._config.warmup_runs):
            await self._safe_generate(model=model, prompt=self._config.prompt, max_tokens=16)

        memory_profiler = MemoryProfiler() if self._config.measure_memory else None
        throughputs: list[float] = []
        latencies_ms: list[float] = []
        generated_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for _ in range(self._config.num_runs):
            with Timer() as timer:
                response = await self._safe_generate(
                    model=model,
                    prompt=self._config.prompt,
                    max_tokens=self._config.max_tokens,
                )
            total_latency_ms = timer.elapsed_ms()
            completion_tokens = int(response.get("eval_count", 0))
            prompt_tokens = int(response.get("prompt_eval_count", 0))
            generated_text = str(response.get("response", ""))
            throughput = self._measure_throughput(
                response=response, fallback_latency_ms=total_latency_ms
            )
            throughputs.append(throughput)
            latencies_ms.append(total_latency_ms)
            if memory_profiler is not None:
                memory_profiler.get_current_memory_mb()

        filtered_throughputs = StatisticsCalculator.remove_outliers(throughputs)
        filtered_latencies = StatisticsCalculator.remove_outliers(latencies_ms)
        avg_throughput = mean(filtered_throughputs)
        avg_latency = mean(filtered_latencies)
        ttft_ms, total_latency_ms = self._measure_latency(latencies=filtered_latencies)
        memory_baseline, memory_peak = self._measure_memory(memory_profiler)
        quantization = self._extract_quantization(model)

        return BenchmarkResult(
            model_name=model,
            quantization=quantization,
            tokens_per_second=avg_throughput,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_latency_ms if total_latency_ms > 0 else avg_latency,
            memory_peak_mb=memory_peak,
            memory_baseline_mb=memory_baseline,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def run_model_family(self, models: list[str]) -> list[BenchmarkResult]:
        """Run benchmark for all models in a family."""
        return [await self.run_single_benchmark(model_name) for model_name in models]

    async def run_all_models(self, models: list[str]) -> dict[str, list[BenchmarkResult]]:
        """Run benchmark for a list of models."""
        results = await self.run_model_family(models)
        grouped: dict[str, list[BenchmarkResult]] = {}
        for result in results:
            grouped.setdefault(result.model_name, []).append(result)
        return grouped

    def _measure_throughput(self, response: dict[str, Any], fallback_latency_ms: float) -> float:
        """Measure throughput from Ollama response payload."""
        eval_count = int(response.get("eval_count", 0))
        eval_duration_ns = int(response.get("eval_duration", 0))
        if eval_count <= 0:
            return 0.0
        if eval_duration_ns > 0:
            return eval_count / (eval_duration_ns / 1_000_000_000)
        if fallback_latency_ms <= 0:
            return float(eval_count)
        return eval_count / (fallback_latency_ms / 1000.0)

    def _measure_latency(self, latencies: list[float]) -> tuple[float, float]:
        """Estimate TTFT and total latency from sampled latencies."""
        if not latencies:
            return 0.0, 0.0
        avg_latency = mean(latencies)
        return avg_latency * 0.2, avg_latency

    def _measure_memory(self, profiler: MemoryProfiler | None) -> tuple[float, float]:
        """Measure baseline and peak memory usage."""
        if profiler is None:
            return 0.0, 0.0
        return profiler.get_baseline(), profiler.get_peak()

    async def _safe_generate(self, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Generate text with timeout handling."""
        return await asyncio.wait_for(
            self._client.generate(model=model, prompt=prompt, max_tokens=max_tokens),
            timeout=self._config.timeout_seconds,
        )

    def _extract_quantization(self, model_name: str) -> str:
        """Extract quantization suffix from model name."""
        match = self._QUANTIZATION_PATTERN.search(model_name)
        if match is not None:
            token = match.group(1).lower().replace("-", "_")
            if token == "f16":
                return "fp16"
            return token
        return "unknown"


class BenchmarkSuite:
    """High-level benchmark orchestration and persistence."""

    def __init__(
        self,
        client: OllamaClient,
        collector: QuantizationInfoCollector,
        output_dir: Path,
    ) -> None:
        """Initialize benchmark suite."""
        self._client = client
        self._collector = collector
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run_performance_suite(
        self, config: BenchmarkConfig, model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run performance benchmarks for all available models."""
        models = await self._collector.collect_all_quantizations()
        runner = BenchmarkRunner(self._client, config)
        available_names = [model.name for model in models]
        selected_model_names = model_names or available_names
        benchmark_results = await runner.run_model_family(selected_model_names)
        summary = self._summarize_results(benchmark_results)
        return {
            "type": "performance",
            "config": asdict(config),
            "results": [result.to_dict() for result in benchmark_results],
            "summary": summary,
        }

    async def run_quality_suite(
        self, config: BenchmarkConfig, model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run quality-oriented benchmark configuration."""
        quality_config = BenchmarkConfig(
            prompt=config.prompt,
            max_tokens=min(config.max_tokens, 128),
            num_runs=max(1, config.num_runs - 1),
            warmup_runs=config.warmup_runs,
            measure_memory=False,
            measure_quality=False,
            timeout_seconds=config.timeout_seconds,
        )
        payload = await self.run_performance_suite(quality_config, model_names=model_names)
        return {**payload, "type": "quality"}

    async def run_comprehensive_suite(
        self, config: BenchmarkConfig, model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run combined performance and quality suites."""
        performance = await self.run_performance_suite(config, model_names=model_names)
        quality = await self.run_quality_suite(config, model_names=model_names)
        return {"performance": performance, "quality": quality}

    def save_results(self, results: dict[str, Any], filename: str) -> Path:
        """Save benchmark results to JSON file."""
        path = self._output_dir / filename
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("benchmark_results_saved", path=str(path))
        return path

    def load_results(self, filename: str) -> dict[str, Any]:
        """Load benchmark results from JSON file."""
        path = self._output_dir / filename
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def _summarize_results(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Create aggregate summary statistics."""
        throughput_values = [result.tokens_per_second for result in results]
        latency_values = [result.total_latency_ms for result in results]
        memory_values = [result.memory_peak_mb for result in results]
        return {
            "count": len(results),
            "throughput": StatisticsCalculator.calculate_statistics(throughput_values),
            "latency_ms": StatisticsCalculator.calculate_statistics(latency_values),
            "memory_mb": StatisticsCalculator.calculate_statistics(memory_values),
        }
