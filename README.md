# LLM Inference Optimization Engine

A production-grade LLM inference optimization layer built on top of Ollama for MacBook M2 Air.

## Overview

This project implements an intelligent orchestration and optimization layer on top of Ollama that focuses on:
- **Smart Batching**: Group requests for maximum throughput
- **Intelligent Scheduling**: Different strategies for different workloads  
- **Request Optimization**: Cache common queries, optimize prompts
- **Advanced Techniques**: Speculative decoding for 2-4x speedup

## Architecture

```
Your REST API (Port 8000)
    ↓
Orchestration Layer (Batching, Scheduling, Optimization)
    ↓
Ollama Client (HTTP Integration)
    ↓
Ollama Service (Model Serving, Quantization, Inference)
```

## Features

- ✅ **Ollama Integration**: Seamless connection to Ollama service
- ✅ **Multiple Quantization Levels**: Support for 2-bit to FP16
- 🚧 **Intelligent Batching**: Smart request grouping (In Progress)
- 🚧 **Scheduling Policies**: FCFS, SJF, Priority, Token-based
- 🚧 **KV-Cache Management**: Efficient memory handling
- 🚧 **REST API**: FastAPI-based inference server
- 🚧 **Speculative Decoding**: 2-3x generation speedup

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- MacBook M2 Air (16GB RAM recommended)

## Quick Start

### 1. Install Ollama

```bash
# Download from https://ollama.ai/
# Then pull required models
ollama pull mistral
ollama pull llama2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
python scripts/setup_ollama.py
```

### 4. Run Server

```bash
python scripts/run_server.py
```

## Project Status

**Current Phase**: Phase 2 - Quantization Analysis  
**Progress**: 🟢 In Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Ollama Integration & Setup |
| Phase 2 | 🟢 In Progress | Quantization Analysis & Benchmarking |
| Phase 3 | ⚪ Planned | Batching & Scheduling |
| Phase 4 | ⚪ Planned | KV-Cache Management |
| Phase 5 | ⚪ Planned | REST API Orchestration |
| Phase 6 | ⚪ Planned | Speculative Decoding (Optional) |
| Phase 7 | ⚪ Planned | Production Hardening |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Ollama Integration Guide](docs/ollama_integration.md)
- [Configuration Guide](docs/configuration.md)
- [Quantization Analysis Guide](docs/quantization_analysis.md)
- [Latest Benchmark Results](docs/benchmark_results.md)

## Quantization Benchmarking

```bash
# Run comprehensive quantization benchmark suite
python scripts/run_benchmarks.py --all
```

## Performance Targets

- **Baseline (Ollama alone)**: ~50 tokens/sec
- **With batching**: ~82 tokens/sec (1.6x improvement)
- **With speculation**: ~200+ tokens/sec (4x improvement)

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
ruff check src/

# Type checking
mypy src/
```

## License

Apache 2.0

## Author

Built as a FAANG-ready portfolio project demonstrating systems thinking and optimization expertise.
