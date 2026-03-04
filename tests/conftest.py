"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest

from llm_inference_engine.config import InferenceConfig


@pytest.fixture
def config() -> InferenceConfig:
    """Provide test configuration."""
    return InferenceConfig()


@pytest.fixture
def config_path() -> Path:
    """Provide path to test configuration file."""
    return Path(__file__).parent.parent / "configs" / "default.yaml"
