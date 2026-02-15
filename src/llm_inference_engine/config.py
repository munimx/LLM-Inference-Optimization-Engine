"""Configuration management for the inference engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import structlog

from llm_inference_engine.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama service connection."""

    host: str = "localhost"
    port: int = 11434
    timeout_seconds: float = 300.0
    retry_count: int = 3
    retry_backoff_seconds: float = 1.0
    health_check_interval_seconds: int = 60


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    context_length: int = 4096
    memory_gb: float = 4.0
    quantization: str = "4-bit"
    default_temperature: float = 0.7
    default_top_p: float = 0.9


@dataclass
class ServerConfig:
    """Configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "INFO"


@dataclass
class InferenceConfig:
    """Main configuration for the inference engine."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "InferenceConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            InferenceConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigurationError: If config is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("config_yaml_error", error=str(e))
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}") from e

        if not isinstance(data, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        # Parse Ollama config
        ollama_data = data.get("ollama", {})
        try:
            ollama_config = OllamaConfig(**ollama_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid ollama configuration: {e}") from e

        # Parse server config
        server_data = data.get("server", {})
        try:
            server_config = ServerConfig(**server_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid server configuration: {e}") from e

        # Parse model configs
        models_data = data.get("models", {})
        if not isinstance(models_data, dict):
            raise ConfigurationError("Models configuration must be a dictionary")

        models = {}
        for model_name, model_data in models_data.items():
            if not isinstance(model_data, dict):
                raise ConfigurationError(
                    f"Model configuration for '{model_name}' must be a dictionary"
                )
            # Work on a copy to avoid mutating the original YAML-loaded data
            model_data = model_data.copy()
            if "name" not in model_data:
                model_data["name"] = model_name
            try:
                models[model_name] = ModelConfig(**model_data)
            except TypeError as e:
                logger.error("config_type_error", error=str(e))
                raise ConfigurationError(
                    f"Invalid model configuration for '{model_name}': {e}"
                ) from e

        logger.info("configuration_loaded", config_path=str(config_path))

        return cls(
            ollama=ollama_config,
            server=server_config,
            models=models,
        )

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        return self.models.get(model_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "ollama": {
                "host": self.ollama.host,
                "port": self.ollama.port,
                "timeout_seconds": self.ollama.timeout_seconds,
                "retry_count": self.ollama.retry_count,
                "retry_backoff_seconds": self.ollama.retry_backoff_seconds,
                "health_check_interval_seconds": self.ollama.health_check_interval_seconds,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "reload": self.server.reload,
                "log_level": self.server.log_level,
            },
            "models": {
                name: {
                    "name": config.name,
                    "context_length": config.context_length,
                    "memory_gb": config.memory_gb,
                    "quantization": config.quantization,
                    "default_temperature": config.default_temperature,
                    "default_top_p": config.default_top_p,
                }
                for name, config in self.models.items()
            },
        }


def load_config(config_path: Optional[Path] = None) -> InferenceConfig:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file

    Returns:
        InferenceConfig instance
    """
    if config_path is None:
        logger.info("using_default_configuration")
        return InferenceConfig()

    return InferenceConfig.from_yaml(config_path)


__all__ = [
    "OllamaConfig",
    "ModelConfig",
    "ServerConfig",
    "InferenceConfig",
    "load_config",
]
