"""Pytest configuration and fixtures."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from llm_inference_engine.config import InferenceConfig, OllamaConfig
from llm_inference_engine.integration import OllamaClient


@pytest.fixture
def config() -> InferenceConfig:
    """Provide test configuration."""
    return InferenceConfig()


@pytest.fixture
def ollama_config() -> OllamaConfig:
    """Provide Ollama configuration for tests."""
    return OllamaConfig(
        host="localhost",
        port=11434,
        timeout_seconds=30.0,
        retry_count=2,
    )


@pytest.fixture
async def ollama_client(ollama_config: OllamaConfig) -> AsyncGenerator[OllamaClient, None]:
    """Provide an Ollama client for testing."""
    client = OllamaClient(
        host=ollama_config.host,
        port=ollama_config.port,
        timeout=ollama_config.timeout_seconds,
        max_retries=ollama_config.retry_count,
    )

    await client.connect()
    yield client
    await client.close()


@pytest.fixture
def config_path() -> Path:
    """Provide path to test configuration file."""
    return Path(__file__).parent.parent / "configs" / "default.yaml"
