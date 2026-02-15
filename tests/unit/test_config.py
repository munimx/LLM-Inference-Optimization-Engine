"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from llm_inference_engine.config import (
    OllamaConfig,
    ModelConfig,
    ServerConfig,
    InferenceConfig,
    load_config,
)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self) -> None:
        """Test default Ollama configuration."""
        config = OllamaConfig()
        assert config.host == "localhost"
        assert config.port == 11434
        assert config.timeout_seconds == 300.0
        assert config.retry_count == 3

    def test_custom_values(self) -> None:
        """Test custom Ollama configuration."""
        config = OllamaConfig(
            host="192.168.1.1",
            port=8080,
            timeout_seconds=60.0,
            retry_count=5,
        )
        assert config.host == "192.168.1.1"
        assert config.port == 8080
        assert config.timeout_seconds == 60.0
        assert config.retry_count == 5


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self) -> None:
        """Test creating a model configuration."""
        config = ModelConfig(
            name="mistral:7b-instruct",
            context_length=8192,
            memory_gb=4.2,
            quantization="4-bit",
        )

        assert config.name == "mistral:7b-instruct"
        assert config.context_length == 8192
        assert config.memory_gb == 4.2
        assert config.quantization == "4-bit"


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self) -> None:
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.reload is False


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self) -> None:
        """Test default inference configuration."""
        config = InferenceConfig()
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.models, dict)

    def test_from_yaml(self, config_path: Path) -> None:
        """Test loading configuration from YAML."""
        config = InferenceConfig.from_yaml(config_path)

        assert config.ollama.host == "localhost"
        assert config.ollama.port == 11434
        assert config.server.port == 8000
        assert len(config.models) > 0

    def test_from_yaml_file_not_found(self) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            InferenceConfig.from_yaml(Path("nonexistent.yaml"))

    def test_get_model_config(self, config_path: Path) -> None:
        """Test retrieving model configuration."""
        config = InferenceConfig.from_yaml(config_path)
        model_config = config.get_model_config("mistral")

        assert model_config is not None
        assert "mistral" in model_config.name.lower()

    def test_get_nonexistent_model(self) -> None:
        """Test retrieving non-existent model configuration."""
        config = InferenceConfig()
        model_config = config.get_model_config("nonexistent")
        assert model_config is None

    def test_to_dict(self) -> None:
        """Test converting configuration to dictionary."""
        config = InferenceConfig()
        config_dict = config.to_dict()

        assert "ollama" in config_dict
        assert "server" in config_dict
        assert "models" in config_dict
        assert isinstance(config_dict, dict)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_path(self, config_path: Path) -> None:
        """Test loading configuration with explicit path."""
        config = load_config(config_path)
        assert isinstance(config, InferenceConfig)

    def test_load_config_default(self) -> None:
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, InferenceConfig)
        assert config.ollama.host == "localhost"
