# Contributing to LLM Inference Optimization Engine

Thank you for your interest! This document covers setup, conventions, and the
branching workflow used in this project.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Running Tests](#running-tests)
3. [Linting & Type Checking](#linting--type-checking)
4. [Commit Conventions](#commit-conventions)
5. [Branching Workflow](#branching-workflow)
6. [Code Style](#code-style)
7. [Adding a New Module](#adding-a-new-module)

---

## Development Setup

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and
running locally.

```bash
# Clone the repository
git clone <repo-url>
cd "LLM Inference Optimization Engine"

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install project in editable mode with all dev dependencies
pip install -e ".[dev]"

# Verify setup
python -c "from llm_inference_engine import OllamaClient; print('OK')"
```

### Configuration

Copy and edit the default config:

```bash
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml with your Ollama host/port
```

Override the config path:

```bash
export LLM_ENGINE_CONFIG=configs/local.yaml
```

---

## Running Tests

```bash
# All unit tests (fast, no Ollama required)
pytest tests/unit/

# All tests including integration (requires Ollama)
pytest

# With coverage report
pytest --cov=src/llm_inference_engine --cov-report=term-missing

# Single module
pytest tests/unit/scheduling/ -v

# Skip coverage for speed
pytest tests/unit/ --no-cov
```

---

## Linting & Type Checking

```bash
# Style and import linting (zero errors expected)
ruff check src/ tests/

# Strict type checking (zero errors expected)
mypy src/llm_inference_engine --strict

# Auto-fix safe ruff issues
ruff check --fix src/ tests/
```

CI enforces both checks on every push and pull request.

---

## Commit Conventions

This project uses **Conventional Commits** (`type(scope): subject`).

| Type | When to use |
|---|---|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `refactor` | Refactoring with no behaviour change |
| `chore` | Build, CI, or tooling changes |
| `perf` | Performance improvement |

**Examples:**

```
feat(scheduling): add TokenBudget scheduling policy
fix(api): use timeout_seconds instead of timeout in OllamaConfig
test(optimization): add throttler edge cases for soft threshold
docs(api): document OpenAI-compatible endpoints
```

**Rules:**
- Subject in imperative mood, no trailing period
- 72-character limit on subject line
- Body explains *why*, not *what*
- Always include the Co-authored-by trailer when using Copilot:
  ```
  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```

---

## Branching Workflow

All development happens on `main`. Create a feature branch if a change is
large or experimental; otherwise commit directly to `main` with meaningful
atomic commits.

```
main ← all changes land here
  └── feature/my-improvement  (optional, for larger work)
```

- Keep commits atomic and well-described.
- Open a PR for large changes or when you want review.
- Squash is allowed for WIP commits.

---

## Code Style

- **Docstrings:** Google style on all public classes, methods, and module
  `__init__.py` files.
- **Type hints:** All function signatures must be fully annotated. Use
  `X | None` instead of `Optional[X]` (Python 3.10+ union syntax).
- **Naming:** `snake_case` for modules/functions/variables, `PascalCase` for
  classes, `UPPER_SNAKE_CASE` for module-level constants.
- **Imports:** standard library → third-party → local, separated by blank
  lines. `ruff` enforces this automatically.

---

## Adding a New Module

1. Create the file under `src/llm_inference_engine/<package>/<module>.py`.
2. Export the public API in `src/llm_inference_engine/<package>/__init__.py`.
3. Add a corresponding test file under `tests/unit/<package>/test_<module>.py`.
4. Run `ruff check` and `mypy --strict` before committing.
5. Ensure coverage stays above 80 % for the new module.
