# Phase 2: Quantization Analysis

This document describes the Phase 2 quantization analysis stack and how to run it.

## Components

- `QuantizationInfoCollector`: discovers Ollama models and normalizes quantization metadata.
- `BenchmarkRunner` / `BenchmarkSuite`: runs warmup + measured inference benchmarks, captures throughput, latency, and memory.
- `QualityMetricsCalculator` / `QualityBenchmark`: computes BLEU, ROUGE, and semantic-overlap quality scores.
- `QuantizationMapper`: maps user preference priorities (`speed`, `balanced`, `quality`, `memory`) to model recommendations.
- `BenchmarkReporter`: generates markdown reports from benchmark payloads.

## Configuration

- Main benchmark config: `configs/benchmarks.yaml`
- Quality prompts dataset: `configs/quality_test_cases.yaml`

The benchmark config controls scenarios, run counts, quality toggles, model family targets, and output paths.

## Running Benchmarks

```bash
python scripts/run_benchmarks.py --all
python scripts/run_benchmarks.py --performance-only
python scripts/run_benchmarks.py --quality-only
```

Outputs are written to `tests/benchmarks/benchmark_results/`:

- `benchmark_results.json`
- `benchmark_results.md`

## Recommendation Flow

1. Run benchmark suite and collect metrics.
2. Build model payload with throughput, memory, and quality.
3. Use `QuantizationMapper` with `UserPreference`:
   - `speed` prefers higher tokens/sec.
   - `balanced` weights speed and quality similarly.
   - `quality` favors minimal quality degradation.
   - `memory` prioritizes low memory footprint.
