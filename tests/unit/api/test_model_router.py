"""Unit tests for ModelRouter."""

import pytest

from llm_inference_engine.api.model_router import ModelRouter
from llm_inference_engine.config import ModelRegistryConfig


@pytest.fixture
def config() -> ModelRegistryConfig:
    return ModelRegistryConfig(
        fast_model="fast-llm",
        large_model="large-llm",
        fallback_model="fallback-llm",
        fast_model_token_threshold=512,
    )


@pytest.fixture
def router(config: ModelRegistryConfig) -> ModelRouter:
    return ModelRouter(config)


class TestModelRouter:
    def test_short_prompt_routes_to_fast_model(self, router: ModelRouter) -> None:
        short_prompt = "Hi" * 10  # well under 512 tokens
        assert router.route(short_prompt) == "fast-llm"

    def test_long_prompt_routes_to_large_model(self, router: ModelRouter) -> None:
        # Approx 1 token per 4 chars; 512*4 = 2048 chars
        long_prompt = "x " * 1100  # ~1100 tokens
        assert router.route(long_prompt) == "large-llm"

    def test_explicit_model_overrides_routing(self, router: ModelRouter) -> None:
        short_prompt = "Hi"
        assert router.route(short_prompt, explicit_model="custom-model") == "custom-model"

    def test_route_chat_uses_concatenated_content(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        result = router.route_chat(messages)
        assert result in ("fast-llm", "large-llm")

    def test_route_chat_explicit_model_override(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        assert router.route_chat(messages, explicit_model="override-model") == "override-model"
