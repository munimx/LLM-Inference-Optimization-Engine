"""Unit tests for configuration management."""

from pathlib import Path

import pytest

from llm_inference_engine.config import (
    AdmissionControlConfig,
    CacheConfig,
    CircuitBreakerConfig,
    InferenceConfig,
    ModelRegistryConfig,
    RedisConfig,
    ServerConfig,
    VLLMConfig,
    VLLMInstanceConfig,
    load_config,
)
from llm_inference_engine.exceptions import ConfigurationError


class TestVLLMConfig:
    """Tests for VLLMConfig."""

    def test_default_values(self) -> None:
        config = VLLMConfig()
        assert config.timeout_seconds == 120.0
        assert config.retry_count == 2
        assert len(config.instances) >= 1

    def test_custom_instance_url(self) -> None:
        config = VLLMConfig(instances=[VLLMInstanceConfig(url="http://gpu-host:8080")])
        assert config.instances[0].url == "http://gpu-host:8080"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="http"):
            VLLMInstanceConfig(url="not-a-url")

    def test_empty_instances_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            VLLMConfig(instances=[])


class TestRedisConfig:
    """Tests for RedisConfig."""

    def test_default_values(self) -> None:
        config = RedisConfig()
        assert config.url == "redis://localhost:6379/0"
        assert config.socket_timeout_seconds == 5.0

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="socket_timeout_seconds"):
            RedisConfig(socket_timeout_seconds=-1.0)


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self) -> None:
        config = CacheConfig()
        assert config.enabled is True
        assert config.max_size == 256
        assert config.ttl_seconds == 300.0

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_size"):
            CacheConfig(max_size=0)

    def test_invalid_ttl_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="ttl_seconds"):
            CacheConfig(ttl_seconds=0.0)


class TestAdmissionControlConfig:
    """Tests for AdmissionControlConfig."""

    def test_default_values(self) -> None:
        config = AdmissionControlConfig()
        assert config.soft_limit == 0.70
        assert config.hard_limit == 0.90

    def test_soft_exceeds_hard_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="soft_limit"):
            AdmissionControlConfig(soft_limit=0.95, hard_limit=0.90)

    def test_zero_soft_limit_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            AdmissionControlConfig(soft_limit=0.0, hard_limit=0.90)


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self) -> None:
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.reload is False


class TestModelRegistryConfig:
    """Tests for ModelRegistryConfig."""

    def test_default_values(self) -> None:
        config = ModelRegistryConfig()
        assert config.fast_model_token_threshold == 512
        assert config.fast_model != ""
        assert config.large_model != ""

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="fast_model_token_threshold"):
            ModelRegistryConfig(fast_model_token_threshold=0)


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self) -> None:
        config = InferenceConfig()
        assert isinstance(config.vllm, VLLMConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.cache, CacheConfig)

    def test_from_yaml(self, config_path: Path) -> None:
        config = InferenceConfig.from_yaml(config_path)
        assert len(config.vllm.instances) >= 1
        assert config.server.port == 8000
        assert config.cache.enabled is True

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            InferenceConfig.from_yaml(Path("nonexistent.yaml"))

    def test_from_yaml_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("invalid: [ yaml: {")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            InferenceConfig.from_yaml(p)

    def test_from_yaml_not_a_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2")
        with pytest.raises(ConfigurationError, match="Configuration must be a dictionary"):
            InferenceConfig.from_yaml(p)

    def test_to_dict(self) -> None:
        config = InferenceConfig()
        d = config.to_dict()
        assert "vllm" in d
        assert "redis" in d
        assert "cache" in d
        assert "admission_control" in d

    def test_load_config_no_path_returns_defaults(self) -> None:
        config = load_config(None)
        assert isinstance(config, InferenceConfig)
