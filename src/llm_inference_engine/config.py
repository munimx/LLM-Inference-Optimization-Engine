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
class VLLMInstanceConfig:
    """Configuration for a single vLLM instance."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"vllm instance url must start with http:// or https://, got {self.url!r}"
            )


@dataclass
class VLLMConfig:
    """Configuration for vLLM backend connections."""

    instances: list[VLLMInstanceConfig] = field(
        default_factory=lambda: [VLLMInstanceConfig(url="http://localhost:8080")]
    )
    timeout_seconds: float = 120.0
    retry_count: int = 2
    retry_backoff_seconds: float = 0.5
    health_check_interval_seconds: int = 15

    def __post_init__(self) -> None:
        # Allow env vars to override the first instance URL (useful for Docker)
        env_url = os.environ.get("VLLM_URL")
        if env_url:
            self.instances = [VLLMInstanceConfig(url=env_url)]
        if not self.instances:
            raise ConfigurationError("vllm.instances must contain at least one entry")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("vllm.timeout_seconds must be positive")
        if self.retry_count < 0:
            raise ConfigurationError("vllm.retry_count must be non-negative")


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""

    url: str = "redis://localhost:6379/0"
    socket_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        env_url = os.environ.get("REDIS_URL")
        if env_url:
            self.url = env_url
        if self.socket_timeout_seconds <= 0:
            raise ConfigurationError("redis.socket_timeout_seconds must be positive")


@dataclass
class CacheConfig:
    """Configuration for the response cache."""

    enabled: bool = True
    max_size: int = 256
    ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ConfigurationError("cache.max_size must be positive")
        if self.ttl_seconds <= 0:
            raise ConfigurationError("cache.ttl_seconds must be positive")


@dataclass
class AdmissionControlConfig:
    """Configuration for vLLM-metric-based admission control."""

    enabled: bool = True
    soft_limit: float = 0.70
    hard_limit: float = 0.90
    poll_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 < self.soft_limit < 1.0):
            raise ConfigurationError(
                "admission_control.soft_limit must be between 0.0 and 1.0 (exclusive)"
            )
        if not (0.0 < self.hard_limit <= 1.0):
            raise ConfigurationError(
                "admission_control.hard_limit must be between 0.0 and 1.0"
            )
        if self.soft_limit >= self.hard_limit:
            raise ConfigurationError(
                "admission_control.soft_limit must be less than hard_limit"
            )
        if self.poll_interval_seconds <= 0:
            raise ConfigurationError(
                "admission_control.poll_interval_seconds must be positive"
            )


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ConfigurationError(
                "circuit_breaker.failure_threshold must be positive"
            )
        if self.cooldown_seconds < 0:
            raise ConfigurationError(
                "circuit_breaker.cooldown_seconds must be non-negative"
            )


@dataclass
class ModelRegistryConfig:
    """Configuration for the model registry and request router."""

    fast_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    large_model: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    fast_model_token_threshold: int = 512
    fallback_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    fallback_cache_similarity_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.fast_model_token_threshold <= 0:
            raise ConfigurationError(
                "model_registry.fast_model_token_threshold must be positive"
            )
        if not (0.0 <= self.fallback_cache_similarity_threshold <= 1.0):
            raise ConfigurationError(
                "model_registry.fallback_cache_similarity_threshold must be between 0.0 and 1.0"
            )


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

    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    admission_control: AdmissionControlConfig = field(
        default_factory=AdmissionControlConfig
    )
    circuit_breaker: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    model_registry: ModelRegistryConfig = field(
        default_factory=ModelRegistryConfig
    )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "InferenceConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            InferenceConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ConfigurationError: If config is invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("config_yaml_error", error=str(e))
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        def _load_section(
            section_name: str, cls_: type, default: dict[str, Any] | None = None
        ) -> Any:
            raw = data.get(section_name, default or {})
            try:
                return cls_(**raw)
            except TypeError as e:
                raise ConfigurationError(
                    f"Invalid {section_name} configuration: {e}"
                ) from e

        # Parse vLLM config — instances is a list of dicts
        vllm_raw = data.get("vllm", {})
        instances_raw = vllm_raw.get("instances", [{"url": "http://localhost:8080"}])
        vllm_instances = [
            VLLMInstanceConfig(**inst) if isinstance(inst, dict) else VLLMInstanceConfig(url=inst)
            for inst in instances_raw
        ]
        vllm_kwargs = {k: v for k, v in vllm_raw.items() if k != "instances"}
        try:
            vllm_config = VLLMConfig(instances=vllm_instances, **vllm_kwargs)
        except TypeError as e:
            raise ConfigurationError(f"Invalid vllm configuration: {e}") from e

        redis_config: RedisConfig = _load_section("redis", RedisConfig)
        server_config: ServerConfig = _load_section("server", ServerConfig)
        auth_config: AuthConfig = _load_section("auth", AuthConfig)
        cache_config: CacheConfig = _load_section("cache", CacheConfig)
        admission_config: AdmissionControlConfig = _load_section(
            "admission_control", AdmissionControlConfig
        )
        cb_config: CircuitBreakerConfig = _load_section(
            "circuit_breaker", CircuitBreakerConfig
        )
        model_registry_config: ModelRegistryConfig = _load_section(
            "model_registry", ModelRegistryConfig
        )

        logger.info("configuration_loaded", config_path=str(config_path))

        return cls(
            vllm=vllm_config,
            redis=redis_config,
            server=server_config,
            auth=auth_config,
            cache=cache_config,
            admission_control=admission_config,
            circuit_breaker=cb_config,
            model_registry=model_registry_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vllm": {
                "instances": [
                    {"url": inst.url} for inst in self.vllm.instances
                ],
                "timeout_seconds": self.vllm.timeout_seconds,
                "retry_count": self.vllm.retry_count,
                "retry_backoff_seconds": self.vllm.retry_backoff_seconds,
            },
            "redis": {
                "url": self.redis.url,
                "socket_timeout_seconds": self.redis.socket_timeout_seconds,
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
            "admission_control": {
                "enabled": self.admission_control.enabled,
                "soft_limit": self.admission_control.soft_limit,
                "hard_limit": self.admission_control.hard_limit,
                "poll_interval_seconds": self.admission_control.poll_interval_seconds,
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "cooldown_seconds": self.circuit_breaker.cooldown_seconds,
            },
            "model_registry": {
                "fast_model": self.model_registry.fast_model,
                "large_model": self.model_registry.large_model,
                "fast_model_token_threshold": self.model_registry.fast_model_token_threshold,
                "fallback_model": self.model_registry.fallback_model,
            },
        }


def load_config(config_path: Path | None = None) -> InferenceConfig:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file.

    Returns:
        InferenceConfig instance.
    """
    if config_path is None:
        logger.info("using_default_configuration")
        return InferenceConfig()

    return InferenceConfig.from_yaml(config_path)


__all__ = [
    "VLLMInstanceConfig",
    "VLLMConfig",
    "RedisConfig",
    "CacheConfig",
    "AdmissionControlConfig",
    "CircuitBreakerConfig",
    "ModelRegistryConfig",
    "ServerConfig",
    "AuthConfig",
    "InferenceConfig",
    "load_config",
]
