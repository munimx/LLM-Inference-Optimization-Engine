#!/usr/bin/env python3
"""Run quantization benchmarks and generate reports."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_inference_engine.config import load_config
from llm_inference_engine.integration import OllamaClient, OllamaModelManager
from llm_inference_engine.quantization import (
    BenchmarkConfig,
    BenchmarkSuite,
    PreferencePriority,
    QualityBenchmark,
    QualityTestCase,
    QuantizationInfoCollector,
    QuantizationMapper,
    UserPreference,
)
from llm_inference_engine.utils import BenchmarkReporter


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML format at {path}")
    return payload


def _load_quality_cases(path: Path) -> list[QualityTestCase]:
    payload = _load_yaml(path)
    raw_cases = payload.get("quality_test_cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("quality_test_cases must be a list")
    cases: list[QualityTestCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        cases.append(
            QualityTestCase(
                prompt=str(item["prompt"]),
                reference_outputs=[str(reference) for reference in item["reference_outputs"]],
                task_type=str(item["task_type"]),
                max_tokens=int(item.get("max_tokens", 128)),
            )
        )
    return cases


def _select_scenario(perf_cfg: dict[str, Any], scenario_name: str | None) -> dict[str, Any]:
    raw_scenarios = perf_cfg.get("scenarios", [{}])
    scenarios = [item for item in raw_scenarios if isinstance(item, dict)]
    if not scenarios:
        return {}
    if scenario_name is None:
        return scenarios[0]
    for scenario in scenarios:
        if str(scenario.get("name", "")) == scenario_name:
            return scenario
    raise ValueError(f"Unknown scenario: {scenario_name}")


def _select_models(
    models: list[Any], model_name: str | None, family_name: str | None
) -> list[Any]:
    if model_name is not None:
        selected = [model for model in models if getattr(model, "name", "") == model_name]
    elif family_name is not None:
        normalized_family = family_name.lower()
        selected = [
            model for model in models if str(getattr(model, "family", "")).lower() == normalized_family
        ]
    else:
        selected = models
    if not selected:
        raise ValueError("No models matched the provided CLI filters")
    return selected


async def run(args: argparse.Namespace) -> Path:
    """Run benchmark command and return output JSON path."""
    config_payload = _load_yaml(args.config)
    benchmark_cfg = config_payload.get("benchmark", {})
    perf_cfg = benchmark_cfg.get("performance", {})
    scenario = _select_scenario(perf_cfg, getattr(args, "scenario", None))

    config = load_config(Path("configs/default.yaml"))
    client = OllamaClient(
        host=config.ollama.host,
        port=config.ollama.port,
        timeout=config.ollama.timeout_seconds,
        max_retries=config.ollama.retry_count,
    )
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    async with client:
        manager = OllamaModelManager(client)
        collector = QuantizationInfoCollector(client, manager)
        suite = BenchmarkSuite(client=client, collector=collector, output_dir=output_dir)
        discovered_models = await collector.collect_all_quantizations()
        selected_models = _select_models(discovered_models, args.model, args.family)
        selected_model_names = [model.name for model in selected_models]
        model_metadata = {model.name: model for model in selected_models}
        benchmark_config = BenchmarkConfig(
            prompt=str(scenario.get("prompt", "Explain quantization in one paragraph.")),
            max_tokens=int(scenario.get("max_tokens", 96)),
            num_runs=int(perf_cfg.get("measurement_runs", 3)),
            warmup_runs=int(perf_cfg.get("warmup_runs", 1)),
            measure_memory=bool(perf_cfg.get("measure_memory", True)),
            timeout_seconds=float(perf_cfg.get("timeout_seconds", 120)),
        )
        if args.quality_only:
            results: dict[str, Any] = await suite.run_quality_suite(
                benchmark_config, model_names=selected_model_names
            )
        elif args.performance_only:
            results = await suite.run_performance_suite(
                benchmark_config, model_names=selected_model_names
            )
        else:
            results = await suite.run_comprehensive_suite(
                benchmark_config, model_names=selected_model_names
            )

        quality_scores: dict[str, Any] = {}
        quality_cfg = benchmark_cfg.get("quality", {})
        if bool(quality_cfg.get("enabled", False)) and not args.performance_only:
            quality_cases_file = Path(
                str(quality_cfg.get("test_cases_file", "configs/quality_test_cases.yaml"))
            )
            quality_cases = _load_quality_cases(quality_cases_file)
            quality_runner = QualityBenchmark(client=client, test_dataset=quality_cases)
            model_rows = results.get("results", results.get("performance", {}).get("results", []))
            model_names = [row["model_name"] for row in model_rows if isinstance(row, dict)]
            if model_names:
                quality_scores = {
                    name: score.to_dict()
                    for name, score in (
                        await quality_runner.run_quality_comparison(model_names)
                    ).items()
                }

        models_payload = []
        model_rows = results.get("results", results.get("performance", {}).get("results", []))
        for row in model_rows:
            if not isinstance(row, dict):
                continue
            metadata = model_metadata.get(row["model_name"])
            models_payload.append(
                {
                    "name": row["model_name"],
                    "family": (
                        metadata.family if metadata is not None else row["model_name"].split(":")[0]
                    ),
                    "quantization": (
                        metadata.quantization.value
                        if metadata is not None
                        else row.get("quantization", "unknown")
                    ),
                    "size_bytes": metadata.size_bytes if metadata is not None else 0,
                    "memory_estimate_gb": (
                        metadata.memory_estimate_gb if metadata is not None else 0.0
                    ),
                    "parameters": metadata.parameters if metadata is not None else None,
                    "context_length": metadata.context_length if metadata is not None else 4096,
                    "quality_tier": (
                        metadata.quality_tier.value if metadata is not None else "acceptable"
                    ),
                    "format": metadata.format if metadata is not None else "gguf",
                    "tokens_per_second": float(row.get("tokens_per_second", 0.0)),
                    "time_to_first_token_ms": float(row.get("time_to_first_token_ms", 0.0)),
                    "memory_peak_mb": float(row.get("memory_peak_mb", 0.0)),
                    "quality_score": float(
                        quality_scores.get(row["model_name"], {}).get("overall_score", 0.0)
                    ),
                }
            )

        mapper = QuantizationMapper.from_benchmark_payload({"models": models_payload})
        recommendations: dict[str, str] = {}
        for priority in [
            PreferencePriority.SPEED,
            PreferencePriority.BALANCED,
            PreferencePriority.QUALITY,
            PreferencePriority.MEMORY_EFFICIENT,
        ]:
            selection = mapper.select_model(UserPreference(priority=priority))
            recommendations[priority.value] = selection.name

        final_payload = {
            **results,
            "quality_scores": quality_scores,
            "models": models_payload,
            "recommendations": recommendations,
        }
        json_path = suite.save_results(final_payload, "benchmark_results.json")
        report_path = output_dir / "benchmark_results.md"
        BenchmarkReporter(final_payload).save_report(report_path)
        report_summary = {
            "json_results": str(json_path),
            "markdown_report": str(report_path),
            "recommendations": recommendations,
        }
        print(json.dumps(report_summary, indent=2))
        return json_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run quantization benchmarks")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--all", action="store_true", help="Run benchmark over all discovered models"
    )
    target_group.add_argument("--family", type=str, help="Benchmark a specific model family")
    target_group.add_argument("--model", type=str, help="Benchmark a specific model")
    parser.add_argument("--scenario", type=str, help="Select a named benchmark scenario")
    parser.add_argument(
        "--performance-only", action="store_true", help="Run only performance suite"
    )
    parser.add_argument("--quality-only", action="store_true", help="Run only quality suite")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.yaml"))
    parser.add_argument("--output", type=Path, default=Path("tests/benchmarks/benchmark_results"))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
