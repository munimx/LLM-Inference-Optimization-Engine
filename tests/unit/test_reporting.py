from pathlib import Path

from llm_inference_engine.utils.reporting import BenchmarkReporter


def test_generate_markdown_report(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "model_name": "llama3.1:8b-instruct-q4_K_M",
                "quantization": "q4_K_M",
                "tokens_per_second": 50.0,
                "time_to_first_token_ms": 100.0,
                "total_latency_ms": 800.0,
                "memory_peak_mb": 4500.0,
                "memory_baseline_mb": 1200.0,
            }
        ],
        "quality_scores": {
            "llama3.1:8b-instruct-q4_K_M": {
                "bleu": 0.8,
                "rouge": {"rouge1": 0.82},
                "semantic_similarity": 0.9,
                "overall_score": 0.85,
            }
        },
    }
    reporter = BenchmarkReporter(payload)
    markdown = reporter.generate_markdown_report()
    assert "Quantization Benchmark Results" in markdown
    output = tmp_path / "report.md"
    reporter.save_report(output)
    assert output.exists()
