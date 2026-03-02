# Ollama Integration Guide

## Overview

This project uses Ollama as the model serving layer. Ollama handles model loading, quantization, and inference, while our code focuses on orchestration and optimization on top of it.

## Prerequisites

### 1. Install Ollama

**macOS**:
```bash
# Download from https://ollama.ai/
# Or use Homebrew
brew install ollama
```

**Verify Installation**:
```bash
ollama --version
```

### 2. Start Ollama Service

Ollama usually starts automatically. To manually start:

```bash
ollama serve
```

This starts the Ollama service on `localhost:11434`.

### 3. Pull Required Models

Pull the models you want to use:

```bash
# Mistral 7B (recommended, 4.2GB)
ollama pull mistral

# Llama 2 13B (larger, 8.5GB)
ollama pull llama2

# Phi-3 Mini (smaller, 2.8GB)
ollama pull phi3
```

**Check Available Models**:
```bash
ollama list
```

## Verifying Setup

Run our setup verification script:

```bash
python scripts/setup_ollama.py
```

This will check:
- ✅ Ollama service is running
- ✅ Models are available
- ✅ Configuration is valid
- ✅ Client can connect

## Ollama API Basics

### API Endpoints

**Base URL**: `http://localhost:11434`

**Key Endpoints**:
- `GET /api/tags` - List available models
- `POST /api/generate` - Generate text
- `POST /api/chat` - Chat completion

### Generate Text Example

**Request**:
```json
{
  "model": "mistral",
  "prompt": "Why is the sky blue?",
  "temperature": 0.7,
  "stream": false
}
```

**Response**:
```json
{
  "model": "mistral:7b-instruct",
  "response": "The sky appears blue because...",
  "total_duration": 1234567890,
  "eval_count": 100
}
```

## Model Naming Convention

Ollama uses the following naming pattern:

```
<model>:<size>-<variant>-<quantization>
```

Examples:
- `mistral:7b-instruct-q4_K_M` - Mistral 7B, instruct-tuned, 4-bit quantized
- `llama2:13b-chat-q4_K_M` - Llama 2 13B, chat variant, 4-bit quantized
- `phi3:3.8b-mini-instruct-4k-q4_K_M` - Phi-3 mini, 4K context, 4-bit

### Quantization Levels

| Level | Description | Memory | Quality | Speed |
|-------|-------------|--------|---------|-------|
| q2_K | 2-bit | 16% | ~97% | Fastest |
| q3_K | 3-bit | 22% | ~98% | Very fast |
| q4_K_M | 4-bit medium | 29% | ~99% | Fast (recommended) |
| q4_K_S | 4-bit small | 27% | ~99% | Fast |
| q5_K_M | 5-bit medium | 32% | ~99.5% | Moderate |
| q6_K | 6-bit | 38% | ~99.8% | Slower |
| q8_0 | 8-bit | 52% | ~99.9% | Slowest |

## Configuration

Our engine connects to Ollama using the configuration in `configs/default.yaml`:

```yaml
ollama:
  host: localhost
  port: 11434
  timeout_seconds: 300
  retry_count: 3
  retry_backoff_seconds: 1.0
```

### Retry Backoff with Jitter

As of `perf/8`, the client uses **full-jitter exponential backoff**:

```
sleep = retry_backoff_seconds × (2 ^ attempt) + uniform(0, 1)
```

This spreads retry storms when multiple requests fail simultaneously
(thundering-herd prevention). Tuning guide:

| Scenario | Recommended `retry_backoff_seconds` |
|---|---|
| Local Ollama, stable | `0.5` (fast recovery) |
| Local Ollama, shared M2 memory | `1.0` (default) |
| Remote Ollama over LAN | `2.0` (tolerate network jitter) |

Set `retry_count: 0` to disable retries entirely.

### Environment Variables (Future)

Override configuration via environment variables:

```bash
export OLLAMA_HOST=192.168.1.100
export OLLAMA_PORT=11434
export OLLAMA_TIMEOUT=600
```

## Using the Integration

### Basic Usage

```python
from llm_inference_engine.integration import OllamaClient

# Create client
client = OllamaClient(host="localhost", port=11434)

# Connect
await client.connect()

# List models
models = await client.list_models()
print(f"Available: {len(models)} models")

# Generate text
result = await client.generate(
    model="mistral",
    prompt="Explain quantum computing",
    temperature=0.7
)

print(result["response"])

# Close connection
await client.close()
```

### Using Context Manager

```python
async with OllamaClient() as client:
    # Health check
    health = await client.health_check()
    print(f"Status: {health['status']}")
    
    # Generate
    result = await client.generate(
        model="mistral",
        prompt="What is machine learning?"
    )
```

### Model Manager

```python
from llm_inference_engine.integration import OllamaModelManager

manager = OllamaModelManager(client)

# Refresh models
await manager.refresh_models()

# Get available models
models = await manager.get_available_models()

# Get model info
info = await manager.get_model_info("mistral:7b-instruct-q4_K_M")
print(f"Memory estimate: {info.memory_estimate_gb:.2f} GB")

# Check if model available
is_available = await manager.verify_model_available("mistral")
```

## Troubleshooting

### Ollama Not Running

**Error**: `OllamaConnectionError: Failed to connect`

**Solution**:
```bash
# Start Ollama
ollama serve

# Or check if running
ps aux | grep ollama
```

### Model Not Found

**Error**: `ModelNotFoundError: Model 'mistral' not found`

**Solution**:
```bash
# Pull the model
ollama pull mistral

# List available models
ollama list
```

### Timeout Errors

**Error**: `OllamaTimeoutError: Request timed out`

**Solutions**:
1. Increase timeout in configuration:
```yaml
ollama:
  timeout_seconds: 600  # 10 minutes
```

2. Reduce generation length:
```python
result = await client.generate(
    model="mistral",
    prompt="...",
    max_tokens=256  # Reduce from default
)
```

### Memory Issues

**Error**: Ollama crashes or becomes unresponsive

**Solutions**:
1. Use smaller quantization:
```bash
ollama pull mistral:7b-instruct-q4_K_M  # Instead of q8
```

2. Close other applications to free memory

3. Use smaller models:
```bash
ollama pull phi3  # 2.8GB instead of mistral 4.2GB
```

## Performance Tuning

### Connection Pooling

Reuse connections for better performance:
```python
# Keep client alive
async with OllamaClient() as client:
    # Multiple requests reuse connection
    for prompt in prompts:
        result = await client.generate(model="mistral", prompt=prompt)
```

### Parallel Requests

Make multiple requests concurrently:
```python
import asyncio

tasks = [
    client.generate(model="mistral", prompt=p)
    for p in prompts
]

results = await asyncio.gather(*tasks)
```

### Quantization Selection

Choose appropriate quantization level:
- **Latency-critical**: q4_K_M (fastest while maintaining quality)
- **Memory-constrained**: q2_K or q3_K
- **Quality-critical**: q5_K_M or q6_K

## Advanced Topics

### Streaming Responses (Future)

```python
async for chunk in client.stream_generate(model="mistral", prompt="..."):
    print(chunk["response"], end="", flush=True)
```

### Custom Model Parameters

```python
result = await client.generate(
    model="mistral",
    prompt="...",
    temperature=0.9,      # More creative
    top_p=0.95,           # Nucleus sampling
    max_tokens=512,       # Longer output
    stop_sequences=["END"]  # Stop tokens
)
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Available Models](https://ollama.ai/library)
