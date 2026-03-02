"""Dynamic context-window management for LLM inference requests.

Calculates the effective context window available for a given model and
request configuration, accounting for the model's maximum supported
context length and the number of tokens already used by the prompt.
"""

import functools
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Known context windows for common Ollama model families.
# Values are conservative lower bounds to avoid edge-case truncation.
# ---------------------------------------------------------------------------
_DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    "llama3": 8_192,
    "llama3.1": 128_000,
    "llama3.2": 128_000,
    "llama2": 4_096,
    "mistral": 32_768,
    "mixtral": 32_768,
    "phi3": 128_000,
    "phi4": 16_384,
    "gemma": 8_192,
    "gemma2": 8_192,
    "deepseek": 32_768,
    "qwen": 32_768,
    "codellama": 16_384,
    "falcon": 2_048,
    "vicuna": 4_096,
    "orca": 4_096,
}

# Fallback context window when the model family is not recognised.
_FALLBACK_CONTEXT_WINDOW: int = 4_096


@dataclass(frozen=True)
class ContextWindowInfo:
    """Result of a context-window calculation.

    Attributes:
        model: The Ollama model tag queried.
        max_context_tokens: Maximum context length supported by the model.
        prompt_tokens: Estimated token count of the prompt.
        available_tokens: Tokens remaining after the prompt is consumed.
        utilisation_ratio: Fraction of the context window used by the prompt.
    """

    model: str
    max_context_tokens: int
    prompt_tokens: int
    available_tokens: int
    utilisation_ratio: float


class ContextWindowManager:
    """Calculates and enforces context-window limits.

    Usage::

        manager = ContextWindowManager()
        info = manager.calculate("llama3.1:8b", prompt_tokens=512)
        print(info.available_tokens)  # tokens left for generation

    A custom context registry can be supplied to override or extend the
    built-in defaults::

        manager = ContextWindowManager(
            context_registry={"my-model": 65_536}
        )
    """

    def __init__(
        self,
        context_registry: dict[str, int] | None = None,
        fallback_context_window: int = _FALLBACK_CONTEXT_WINDOW,
    ) -> None:
        """Initialise the manager.

        Args:
            context_registry: Optional mapping of model-family prefixes
                to context window sizes.  Merged with built-in defaults
                (custom entries take precedence).
            fallback_context_window: Context window to use when no match
                is found in the registry (default 4096).
        """
        if fallback_context_window <= 0:
            raise ValueError("fallback_context_window must be positive")
        self._registry: dict[str, int] = {**_DEFAULT_CONTEXT_WINDOWS}
        if context_registry:
            self._registry.update(context_registry)
        self._fallback = fallback_context_window
        # Pre-sort keys by descending length so the first prefix match is
        # always the longest one — O(1) cached lookup after the first call.
        self._sorted_families: list[tuple[str, int]] = sorted(
            self._registry.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        # Wrap the lookup in a per-instance LRU cache so repeated model
        # lookups are O(1) after the first call.
        self.get_max_context_tokens = functools.lru_cache(maxsize=256)(
            self._get_max_context_tokens_uncached
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_max_context_tokens_uncached(self, model: str) -> int:
        """Return the maximum context window for *model* (uncached).

        Called via the per-instance LRU-cached wrapper
        :meth:`get_max_context_tokens`.  Uses a pre-sorted list of
        ``(family, tokens)`` pairs (longest prefix first) so the first
        match found is always the most specific one, avoiding O(n) full
        scans on repeated lookups.

        Args:
            model: Ollama model tag (e.g. ``"llama3.1:8b"``).

        Returns:
            Maximum context window in tokens.
        """
        model_lower = model.lower()
        for family, tokens in self._sorted_families:
            if model_lower.startswith(family.lower()):
                return tokens
        logger.warning("context_window_unknown", model=model, fallback=self._fallback)
        return self._fallback

    def calculate(
        self,
        model: str,
        prompt_tokens: int,
        override_max_context: int | None = None,
    ) -> ContextWindowInfo:
        """Calculate the available context window for a request.

        Args:
            model: Ollama model tag.
            prompt_tokens: Estimated number of tokens in the prompt.
            override_max_context: If provided, use this as the model's
                maximum context window instead of the registry value.

        Returns:
            A :class:`ContextWindowInfo` dataclass with the breakdown.

        Raises:
            ValueError: If ``prompt_tokens`` is negative or exceeds the
                model's maximum context window.
        """
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")

        max_ctx = override_max_context if override_max_context is not None else self.get_max_context_tokens(model)
        if max_ctx <= 0:
            raise ValueError("max_context must be positive")
        if prompt_tokens > max_ctx:
            raise ValueError(
                f"prompt_tokens ({prompt_tokens}) exceeds model context window "
                f"({max_ctx}) for model {model!r}"
            )

        available = max_ctx - prompt_tokens
        utilisation = prompt_tokens / max_ctx

        info = ContextWindowInfo(
            model=model,
            max_context_tokens=max_ctx,
            prompt_tokens=prompt_tokens,
            available_tokens=available,
            utilisation_ratio=utilisation,
        )
        logger.debug(
            "context_window_calculated",
            model=model,
            max_context=max_ctx,
            prompt_tokens=prompt_tokens,
            available_tokens=available,
            utilisation_pct=round(utilisation * 100, 1),
        )
        return info

    def estimate_prompt_tokens(self, prompt: str, chars_per_token: float = 4.0) -> int:
        """Roughly estimate the token count of *prompt*.

        This is a fast heuristic (characters / ``chars_per_token``).  For
        accurate counts, use a proper tokenizer from the model's library.

        Args:
            prompt: The prompt string to estimate.
            chars_per_token: Average characters per token (default 4.0).

        Returns:
            Estimated token count (minimum 1).
        """
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        return max(1, round(len(prompt) / chars_per_token))

    def register_model(self, model_prefix: str, context_tokens: int) -> None:
        """Add or update a model family in the context registry.

        Args:
            model_prefix: Model family prefix (e.g. ``"my-model"``).
            context_tokens: Maximum context window in tokens.
        """
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        self._registry[model_prefix] = context_tokens
        # Rebuild sorted list and clear cached lookups so the new entry
        # is visible immediately.
        self._sorted_families = sorted(
            self._registry.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        self.get_max_context_tokens.cache_clear()
        logger.info(
            "model_registered",
            model_prefix=model_prefix,
            context_tokens=context_tokens,
        )


__all__ = ["ContextWindowManager", "ContextWindowInfo"]
