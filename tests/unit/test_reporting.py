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


def test_generate_comparison_table() -> None:
    payload = {
        "results": [
            {
                "model_name": "llama3",
                "quantization": "q4",
                "tokens_per_second": 40.0,
                "time_to_first_token_ms": 90.0,
                "total_latency_ms": 700.0,
                "memory_peak_mb": 3000.0,
                "memory_baseline_mb": 1000.0,
            }
        ],
    }
    reporter = BenchmarkReporter(payload)
    table = reporter.generate_comparison_table()
    assert "llama3" in table


def test_generate_markdown_report_no_quality_scores() -> None:
    payload = {
        "results": [
            {
                "model_name": "llama3",
                "quantization": "q4",
                "tokens_per_second": 40.0,
                "time_to_first_token_ms": 90.0,
                "total_latency_ms": 700.0,
                "memory_peak_mb": 3000.0,
                "memory_baseline_mb": 1000.0,
            }
        ],
    }
    reporter = BenchmarkReporter(payload)
    markdown = reporter.generate_markdown_report()
    assert "No quality scores" in markdown


def test_generate_markdown_report_empty_results() -> None:
    """No benchmark rows → recommendations section uses placeholder."""
    reporter = BenchmarkReporter({"results": []})
    markdown = reporter.generate_markdown_report()
    assert "No benchmark rows available" in markdown


def test_generate_markdown_report_performance_key_format() -> None:
    """Supports alternative payload format with 'performance' key."""
    payload = {
        "performance": {
            "results": [
                {
                    "model_name": "mistral",
                    "quantization": "q8",
                    "tokens_per_second": 35.0,
                    "time_to_first_token_ms": 120.0,
                    "total_latency_ms": 900.0,
                    "memory_peak_mb": 5000.0,
                    "memory_baseline_mb": 1500.0,
                }
            ]
        }
    }
    reporter = BenchmarkReporter(payload)
    markdown = reporter.generate_markdown_report()
    assert "mistral" in markdown
