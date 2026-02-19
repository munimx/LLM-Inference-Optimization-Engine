# Quantization Benchmark Results

**Generated**: 2026-02-19T11:17:15

## Performance Comparison

| Model | Quant | T/s | TTFT (ms) | Total Latency (ms) |
|---|---:|---:|---:|---:|
| llama3.1:8b | unknown | 17.57 | 5725.28 | 28626.42 |
| mistral:7b | unknown | 18.02 | 3221.18 | 16105.89 |
| phi3:latest | unknown | 37.18 | 897.73 | 4488.63 |
| deepseek-r1:7b | unknown | 10.41 | 15158.12 | 75790.58 |

## Memory Usage

| Model | Baseline MB | Peak MB | Delta MB |
|---|---:|---:|---:|
| llama3.1:8b | 25.95 | 43.44 | 17.48 |
| mistral:7b | 22.69 | 26.73 | 4.05 |
| phi3:latest | 22.19 | 26.25 | 4.06 |
| deepseek-r1:7b | 22.55 | 26.17 | 3.62 |

## Quality Metrics

| Model | BLEU | ROUGE-1 | Semantic | Overall |
|---|---:|---:|---:|---:|
| llama3.1:8b | 0.364 | 0.431 | 0.297 | 0.364 |
| mistral:7b | 0.072 | 0.157 | 0.112 | 0.109 |
| phi3:latest | 0.026 | 0.135 | 0.065 | 0.070 |
| deepseek-r1:7b | 0.341 | 0.365 | 0.270 | 0.327 |

## Recommendations

- **Speed priority**: `phi3:latest` (37.18 t/s)
- **Memory priority**: `deepseek-r1:7b` (26.17 MB peak)
- **Quality priority**: `llama3.1:8b`
