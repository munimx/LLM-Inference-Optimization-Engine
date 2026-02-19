from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm_inference_engine.quantization.benchmarks import BenchmarkRunner, BenchmarkSuite
from llm_inference_engine.quantization.types import (
    BenchmarkConfig,
    QualityTier,
    QuantizationLevel,
    QuantizedModelInfo,
)


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(prompt="Explain quantization", max_tokens=32, num_runs=2, warmup_runs=1)


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.generate.return_value = {
        "response": "Quantization reduces model size.",
        "eval_count": 20,
        "prompt_eval_count": 5,
        "eval_duration": 1_000_000_000,
    }
    return client


@pytest.mark.asyncio
async def test_run_single_benchmark(
    mock_client: AsyncMock, benchmark_config: BenchmarkConfig
) -> None:
    runner = BenchmarkRunner(client=mock_client, config=benchmark_config)
    result = await runner.run_single_benchmark("llama3.1:8b-instruct-q4_K_M")
    assert result.model_name == "llama3.1:8b-instruct-q4_K_M"
    assert result.tokens_per_second > 0
    assert mock_client.generate.call_count == 3


@pytest.mark.asyncio
async def test_run_model_family(mock_client: AsyncMock, benchmark_config: BenchmarkConfig) -> None:
    runner = BenchmarkRunner(client=mock_client, config=benchmark_config)
    results = await runner.run_model_family(["model-a-q4_0", "model-b-q5_K_M"])
    assert len(results) == 2


@pytest.mark.asyncio
async def test_benchmark_suite_save_load(
    mock_client: AsyncMock, benchmark_config: BenchmarkConfig, tmp_path: Path
) -> None:
    collector = AsyncMock()
    collector.collect_all_quantizations.return_value = [
        QuantizedModelInfo(
            name="model-a-q4_0",
            family="model-a",
            quantization=QuantizationLevel.Q4_0,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
        )
    ]
    suite = BenchmarkSuite(client=mock_client, collector=collector, output_dir=tmp_path)
    payload = await suite.run_performance_suite(benchmark_config)
    path = suite.save_results(payload, "results.json")
    loaded = suite.load_results("results.json")
    assert path.exists()
    assert loaded["type"] == "performance"


@pytest.mark.asyncio
async def test_quality_suite_returns_quality_type(
    mock_client: AsyncMock, benchmark_config: BenchmarkConfig, tmp_path: Path
) -> None:
    collector = AsyncMock()
    collector.collect_all_quantizations.return_value = [
        QuantizedModelInfo(
            name="model-a-q4_0",
            family="model-a",
            quantization=QuantizationLevel.Q4_0,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
        )
    ]
    suite = BenchmarkSuite(client=mock_client, collector=collector, output_dir=tmp_path)
    payload = await suite.run_quality_suite(benchmark_config)
    assert payload["type"] == "quality"


@pytest.mark.asyncio
async def test_performance_suite_respects_model_filter(
    mock_client: AsyncMock, benchmark_config: BenchmarkConfig, tmp_path: Path
) -> None:
    collector = AsyncMock()
    collector.collect_all_quantizations.return_value = [
        QuantizedModelInfo(
            name="model-a-q4_0",
            family="model-a",
            quantization=QuantizationLevel.Q4_0,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
        ),
        QuantizedModelInfo(
            name="model-b-q4_0",
            family="model-b",
            quantization=QuantizationLevel.Q4_0,
            size_bytes=1,
            memory_estimate_gb=1.0,
            parameters=None,
            context_length=4096,
            quality_tier=QualityTier.ACCEPTABLE,
        ),
    ]
    suite = BenchmarkSuite(client=mock_client, collector=collector, output_dir=tmp_path)
    payload = await suite.run_performance_suite(benchmark_config, model_names=["model-b-q4_0"])
    result_rows = payload["results"]
    assert len(result_rows) == 1
    assert result_rows[0]["model_name"] == "model-b-q4_0"


def test_extract_quantization_handles_fp16_variant(mock_client: AsyncMock) -> None:
    config = BenchmarkConfig(prompt="test", max_tokens=8, num_runs=1, warmup_runs=0)
    runner = BenchmarkRunner(client=mock_client, config=config)
    assert runner._extract_quantization("mistral:7b-fp16") == "fp16"
