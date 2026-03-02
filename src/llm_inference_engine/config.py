"""Configuration management for the inference engine."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

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

    def __post_init__(self) -> None:
        # Allow env vars to override YAML values (useful for Docker)
        env_host = os.environ.get("OLLAMA_HOST")
        if env_host:
            self.host = env_host
        env_port = os.environ.get("OLLAMA_PORT")
        if env_port:
            try:
                self.port = int(env_port)
            except ValueError as exc:
                raise ConfigurationError(f"OLLAMA_PORT must be an integer, got {env_port!r}") from exc
        if self.port <= 0 or self.port > 65535:
            raise ConfigurationError("ollama.port must be between 1 and 65535")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("ollama.timeout_seconds must be positive")
        if self.retry_count < 0:
            raise ConfigurationError("ollama.retry_count must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ConfigurationError("ollama.retry_backoff_seconds must be non-negative")


@dataclass
class CacheConfig:
    """Configuration for the response cache."""

    enabled: bool = True
    max_size: int = 256
    ttl_seconds: float = 300.0
    mode: str = "exact"  # "exact" or "semantic"
    embedding_model: str = "nomic-embed-text"
    similarity_threshold: float = 0.92

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ConfigurationError("cache.max_size must be positive")
        if self.ttl_seconds <= 0:
            raise ConfigurationError("cache.ttl_seconds must be positive")
        if self.mode not in ("exact", "semantic"):
            raise ConfigurationError(
                f"cache.mode must be 'exact' or 'semantic', got {self.mode!r}"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ConfigurationError(
                "cache.similarity_threshold must be between 0.0 and 1.0"
            )


@dataclass
class SchedulingConfig:
    """Configuration for the request scheduler."""

    policy: str = "fcfs"
    max_requests_per_batch: int = 8
    max_tokens_per_batch: int = 0
    drain_delay_seconds: float = 0.05
    max_queue_depth: int = 0  # 0 = unlimited
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        valid_policies = {"fcfs", "sjf", "priority", "token_budget"}
        if self.policy not in valid_policies:
            raise ConfigurationError(
                f"scheduling.policy must be one of {valid_policies}, got {self.policy!r}"
            )
        if self.max_requests_per_batch <= 0:
            raise ConfigurationError("scheduling.max_requests_per_batch must be positive")
        if self.max_tokens_per_batch < 0:
            raise ConfigurationError("scheduling.max_tokens_per_batch must be non-negative")
        if self.drain_delay_seconds < 0:
            raise ConfigurationError("scheduling.drain_delay_seconds must be non-negative")
        if self.max_queue_depth < 0:
            raise ConfigurationError("scheduling.max_queue_depth must be non-negative")
        if self.circuit_breaker_threshold <= 0:
            raise ConfigurationError("scheduling.circuit_breaker_threshold must be positive")
        if self.circuit_breaker_cooldown_seconds < 0:
            raise ConfigurationError("scheduling.circuit_breaker_cooldown_seconds must be non-negative")


@dataclass
class MemoryConfig:
    """Configuration for memory-based admission control."""

    limit_gb: float = 14.0
    safety_margin: float = 1.1

    def __post_init__(self) -> None:
        if self.limit_gb <= 0:
            raise ConfigurationError("memory.limit_gb must be positive")
        if self.safety_margin < 1.0:
            raise ConfigurationError("memory.safety_margin must be >= 1.0")


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
class AuthConfig:
    """Configuration for API key authentication.

    When ``api_keys`` is empty (default), authentication is disabled and
    all requests are allowed through.
    """

    enabled: bool = False
    api_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled and not self.api_keys:
            raise ConfigurationError(
                "auth.enabled is true but no api_keys are configured"
            )


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
    auth: AuthConfig = field(default_factory=AuthConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)

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

        # Parse auth config
        auth_data = data.get("auth", {})
        try:
            auth_config = AuthConfig(**auth_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid auth configuration: {e}") from e

        # Parse cache config
        cache_data = data.get("cache", {})
        try:
            cache_config = CacheConfig(**cache_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid cache configuration: {e}") from e

        # Parse scheduling config
        scheduling_data = data.get("scheduling", {})
        try:
            scheduling_config = SchedulingConfig(**scheduling_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid scheduling configuration: {e}") from e

        # Parse memory config
        memory_data = data.get("memory", {})
        try:
            memory_config = MemoryConfig(**memory_data)
        except TypeError as e:
            logger.error("config_type_error", error=str(e))
            raise ConfigurationError(f"Invalid memory configuration: {e}") from e

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
            auth=auth_config,
            cache=cache_config,
            scheduling=scheduling_config,
            memory=memory_config,
            models=models,
        )

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        return self.models.get(model_name)

    def to_dict(self) -> dict[str, Any]:
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
            "cache": {
                "enabled": self.cache.enabled,
                "max_size": self.cache.max_size,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "scheduling": {
                "policy": self.scheduling.policy,
                "max_requests_per_batch": self.scheduling.max_requests_per_batch,
                "max_tokens_per_batch": self.scheduling.max_tokens_per_batch,
            },
            "memory": {
                "limit_gb": self.memory.limit_gb,
                "safety_margin": self.memory.safety_margin,
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


def load_config(config_path: Path | None = None) -> InferenceConfig:
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
    "CacheConfig",
    "SchedulingConfig",
    "MemoryConfig",
    "ModelConfig",
    "ServerConfig",
    "InferenceConfig",
    "load_config",
]
