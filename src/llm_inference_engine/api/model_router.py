"""Config-driven model router.

Maps request characteristics to target model names using a simple routing
table defined in :class:`~llm_inference_engine.config.ModelRegistryConfig`.

Routing rules (evaluated in order):

1. If the request explicitly names a model, use it as-is.
2. If the prompt token count is below ``fast_model_token_threshold``, route
   to ``fast_model``.
3. Otherwise route to ``large_model``.

Usage::

    router = ModelRouter(config.model_registry)
    model_name = router.route(prompt="Hello!", explicit_model=None)
"""

from __future__ import annotations

import structlog

from llm_inference_engine.config import ModelRegistryConfig
from llm_inference_engine.utils.tokenizer import estimate_prompt_tokens

logger = structlog.get_logger(__name__)


class ModelRouter:
    """Routes requests to the appropriate vLLM model.

    Args:
        config: :class:`~llm_inference_engine.config.ModelRegistryConfig`
            from the loaded :class:`~llm_inference_engine.config.InferenceConfig`.
    """

    def __init__(self, config: ModelRegistryConfig) -> None:
        self._config = config

    def route(
        self,
        prompt: str,
        explicit_model: str | None = None,
    ) -> str:
        """Return the model name to use for this request.

        Args:
            prompt: The prompt or concatenated message text.
            explicit_model: If the caller specified a model name, pass it here
                to bypass automatic routing.

        Returns:
            The resolved model identifier string.
        """
        if explicit_model:
            logger.debug("model_router_explicit", model=explicit_model)
            return explicit_model

        token_count = estimate_prompt_tokens(prompt)
        if token_count < self._config.fast_model_token_threshold:
            logger.debug(
                "model_router_fast",
                tokens=token_count,
                threshold=self._config.fast_model_token_threshold,
            )
            return self._config.fast_model

        logger.debug(
            "model_router_large",
            tokens=token_count,
            threshold=self._config.fast_model_token_threshold,
        )
        return self._config.large_model

    def route_chat(
        self,
        messages: list[dict[str, str]],
        explicit_model: str | None = None,
    ) -> str:
        """Route a chat request by flattening message content for token counting."""
        if explicit_model:
            return explicit_model
        combined = " ".join(m.get("content", "") for m in messages)
        return self.route(combined)


__all__ = ["ModelRouter"]
