# Contributing

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (for integration tests)
- Git

## Setup

```bash
git clone https://github.com/your-org/llm-inference-optimization-engine.git
cd llm-inference-optimization-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

### Run Tests

```bash
python3 -m pytest tests/unit/ --no-cov -q   # 616 unit tests, ~3 seconds
python3 -m pytest tests/ -q                  # all tests including integration
```

### Lint and Type Check

```bash
ruff check src/ tests/              # linting
ruff check src/ tests/ --fix        # auto-fix
mypy src/llm_inference_engine --strict   # type checking
```

All three must pass before merging.

### Run Locally

```bash
uvicorn llm_inference_engine.api.server:app --host 0.0.0.0 --port 8000 --reload
curl http://localhost:8000/health
```

## Code Style

- **Formatting**: Ruff (configured in `pyproject.toml`)
- **Type hints**: All public functions. `mypy --strict` enforced.
- **Logging**: Use `structlog` — no `print()` statements
- **Tests**: Use `pytest` with `unittest.mock`. Async tests use `pytest-asyncio`.

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes with clear, atomic commits
3. Ensure `pytest`, `ruff check`, and `mypy --strict` all pass
4. Open a PR with a description of what changed and why
5. Address review feedback

## Project Structure

```
src/llm_inference_engine/
├── api/           # FastAPI routes and Pydantic models
├── cache/         # ExactMatchCache, EmbeddingCache
├── core/          # Config, dependency injection
├── models/        # Request/response data classes
├── optimization/  # Throttler, memory estimator, circuit breaker
└── scheduling/    # Scheduler, policies, request queue
```
