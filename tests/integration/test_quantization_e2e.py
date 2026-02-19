from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm_inference_engine.integration.ollama_models import ModelInfo, OllamaModelManager
from llm_inference_engine.quantization import (
    BenchmarkConfig,
    BenchmarkSuite,
    QuantizationInfoCollector,
)
from llm_inference_engine.utils.reporting import BenchmarkReporter


@pytest.mark.asyncio
async def test_quantization_pipeline_e2e(tmp_path: Path) -> None:
    client = AsyncMock()
    client.generate.return_value = {
        "response": "Quantization improves efficiency.",
        "eval_count": 24,
        "prompt_eval_count": 12,
        "eval_duration": 1_200_000_000,
    }
    manager = AsyncMock(spec=OllamaModelManager)
    manager.get_available_models.return_value = [
        ModelInfo(
            name="llama3.1:8b-instruct-q4_K_M",
            size_bytes=4_000_000_000,
            quantization="4-bit",
            family="llama3.1",
            format="gguf",
        )
    ]
    collector = QuantizationInfoCollector(client=client, model_manager=manager)
    suite = BenchmarkSuite(client=client, collector=collector, output_dir=tmp_path)
    config = BenchmarkConfig(
        prompt="Explain quantization.", max_tokens=32, num_runs=2, warmup_runs=1
    )

    payload = await suite.run_performance_suite(config)
    saved_path = suite.save_results(payload, "e2e_results.json")
    loaded = suite.load_results("e2e_results.json")
    reporter = BenchmarkReporter(loaded)
    report_path = tmp_path / "e2e_report.md"
    reporter.save_report(report_path)

    assert saved_path.exists()
    assert report_path.exists()
    assert loaded["type"] == "performance"
