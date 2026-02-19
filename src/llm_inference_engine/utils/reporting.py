"""Benchmark reporting utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


class BenchmarkReporter:
    """Generate benchmark reports from machine-readable results."""

    def __init__(self, results: dict[str, Any]) -> None:
        """Initialize reporter with benchmark results payload."""
        self._results = results

    def generate_markdown_report(self) -> str:
        """Generate a complete markdown benchmark report."""
        generated_at = datetime.utcnow().isoformat(timespec="seconds")
        lines = [
            "# Quantization Benchmark Results",
            "",
            f"**Generated**: {generated_at}",
            "",
            "## Performance Comparison",
            "",
            self._create_performance_table(),
            "",
            "## Client Memory Usage (Benchmark Runner RSS)",
            "",
            self._create_memory_table(),
            "",
            "## Quality Metrics",
            "",
            self._create_quality_table(),
            "",
            "## Recommendations",
            "",
            self._create_recommendations_section(),
            "",
        ]
        return "\n".join(lines)

    def generate_comparison_table(self) -> str:
        """Generate compact comparison table only."""
        return self._create_performance_table()

    def save_report(self, output_path: Path) -> None:
        """Save markdown report to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_markdown_report(), encoding="utf-8")

    def _create_performance_table(self) -> str:
        """Create performance section markdown table."""
        rows = [
            "| Model | Quant | T/s | TTFT (ms) | Total Latency (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in self._get_rows():
            rows.append(
                f"| {item['model_name']} | {item['quantization']} | "
                f"{item['tokens_per_second']:.2f} | {item['time_to_first_token_ms']:.2f} | "
                f"{item['total_latency_ms']:.2f} |"
            )
        return "\n".join(rows)

    def _create_memory_table(self) -> str:
        """Create memory section markdown table."""
        rows = ["| Model | Baseline MB | Peak MB | Delta MB |", "|---|---:|---:|---:|"]
        for item in self._get_rows():
            baseline_mb = item.get("client_memory_baseline_mb", item["memory_baseline_mb"])
            peak_mb = item.get("client_memory_peak_mb", item["memory_peak_mb"])
            delta = peak_mb - baseline_mb
            rows.append(
                f"| {item['model_name']} | {baseline_mb:.2f} | {peak_mb:.2f} | {delta:.2f} |"
            )
        return "\n".join(rows)

    def _create_quality_table(self) -> str:
        """Create quality section markdown table."""
        quality_rows = self._results.get("quality_scores", {})
        if not quality_rows:
            return "_No quality scores collected in this run._"
        rows = ["| Model | BLEU | ROUGE-1 | Semantic | Overall |", "|---|---:|---:|---:|---:|"]
        for model, score in quality_rows.items():
            rouge = score.get("rouge", {}).get("rouge1", 0.0)
            rows.append(
                f"| {model} | {score.get('bleu', 0.0):.3f} | {rouge:.3f} | "
                f"{score.get('semantic_similarity', 0.0):.3f} | {score.get('overall_score', 0.0):.3f} |"
            )
        return "\n".join(rows)

    def _create_recommendations_section(self) -> str:
        """Create recommendations section based on throughput and quality."""
        rows = self._get_rows()
        if not rows:
            return "_No benchmark rows available._"
        fastest = max(rows, key=lambda item: item["tokens_per_second"])
        leanest = min(rows, key=lambda item: item["memory_peak_mb"])
        quality_scores = self._results.get("quality_scores", {})
        if quality_scores:
            best_quality_model = max(
                quality_scores.items(),
                key=lambda item: float(item[1].get("overall_score", 0.0)),
            )[0]
        else:
            best_quality_model = fastest["model_name"]
        return (
            f"- **Speed priority**: `{fastest['model_name']}` ({fastest['tokens_per_second']:.2f} t/s)\n"
            f"- **Memory priority**: `{leanest['model_name']}` ({leanest['memory_peak_mb']:.2f} MB peak)\n"
            f"- **Quality priority**: `{best_quality_model}`"
        )

    def _get_rows(self) -> list[dict[str, Any]]:
        """Extract benchmark rows from both suite and flat payload formats."""
        if "results" in self._results and isinstance(self._results["results"], list):
            return [row for row in self._results["results"] if isinstance(row, dict)]
        performance = self._results.get("performance", {})
        rows = performance.get("results", [])
        return [row for row in rows if isinstance(row, dict)]
