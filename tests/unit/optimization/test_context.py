"""Unit tests for ContextWindowManager."""

import pytest

from llm_inference_engine.optimization.context import ContextWindowInfo, ContextWindowManager


class TestContextWindowManager:
    """Tests for ContextWindowManager."""

    def test_known_model_family(self) -> None:
        """Known model families should return their configured context window."""
        manager = ContextWindowManager()
        assert manager.get_max_context_tokens("llama3.1:8b") == 128_000
        assert manager.get_max_context_tokens("mistral:7b") == 32_768
        assert manager.get_max_context_tokens("phi3:latest") == 128_000

    def test_unknown_model_falls_back(self) -> None:
        """Unknown model families should return the fallback window."""
        manager = ContextWindowManager(fallback_context_window=2_048)
        result = manager.get_max_context_tokens("totally-unknown-model:7b")
        assert result == 2_048

    def test_custom_registry_overrides_default(self) -> None:
        """Custom registry entries should take precedence over defaults."""
        manager = ContextWindowManager(context_registry={"llama3": 4_096})
        assert manager.get_max_context_tokens("llama3:8b") == 4_096

    def test_longest_prefix_wins(self) -> None:
        """The longest matching prefix in the registry should be used."""
        manager = ContextWindowManager(
            context_registry={"llama3": 8_192, "llama3.1": 64_000}
        )
        assert manager.get_max_context_tokens("llama3.1:8b") == 64_000
        assert manager.get_max_context_tokens("llama3:8b") == 8_192

    def test_calculate_basic(self) -> None:
        """calculate() should return correct ContextWindowInfo."""
        manager = ContextWindowManager()
        info = manager.calculate("llama3:8b", prompt_tokens=512)
        assert isinstance(info, ContextWindowInfo)
        assert info.max_context_tokens == 8_192
        assert info.prompt_tokens == 512
        assert info.available_tokens == 8_192 - 512
        assert info.utilisation_ratio == pytest.approx(512 / 8_192, rel=0.001)

    def test_calculate_zero_prompt_tokens(self) -> None:
        """Zero prompt tokens should leave all context available."""
        manager = ContextWindowManager()
        info = manager.calculate("mistral:7b", prompt_tokens=0)
        assert info.available_tokens == info.max_context_tokens
        assert info.utilisation_ratio == 0.0

    def test_calculate_prompt_exceeds_context_raises(self) -> None:
        """Prompt exceeding model context should raise ValueError."""
        manager = ContextWindowManager()
        with pytest.raises(ValueError, match="exceeds model context window"):
            manager.calculate("llama2:7b", prompt_tokens=99_999)

    def test_calculate_negative_prompt_tokens_raises(self) -> None:
        """Negative prompt_tokens should raise ValueError."""
        manager = ContextWindowManager()
        with pytest.raises(ValueError, match="negative"):
            manager.calculate("llama3:8b", prompt_tokens=-1)

    def test_calculate_override_max_context(self) -> None:
        """override_max_context should replace the registry value."""
        manager = ContextWindowManager()
        info = manager.calculate("llama3:8b", prompt_tokens=100, override_max_context=1_000)
        assert info.max_context_tokens == 1_000
        assert info.available_tokens == 900

    def test_estimate_prompt_tokens(self) -> None:
        """estimate_prompt_tokens should return chars/chars_per_token."""
        manager = ContextWindowManager()
        # "Hello world" = 11 chars / 4.0 ≈ 2.75 → rounds to 3
        assert manager.estimate_prompt_tokens("Hello world") == 3
        # Empty string → min of 1
        assert manager.estimate_prompt_tokens("") == 1

    def test_estimate_prompt_tokens_custom_ratio(self) -> None:
        """estimate_prompt_tokens should respect custom chars_per_token."""
        manager = ContextWindowManager()
        assert manager.estimate_prompt_tokens("ABCDEFGH", chars_per_token=2.0) == 4

    def test_estimate_prompt_tokens_invalid_ratio(self) -> None:
        """Non-positive chars_per_token should raise ValueError."""
        manager = ContextWindowManager()
        with pytest.raises(ValueError, match="chars_per_token"):
            manager.estimate_prompt_tokens("Hello", chars_per_token=0.0)

    def test_register_model(self) -> None:
        """register_model should add/update an entry in the registry."""
        manager = ContextWindowManager()
        manager.register_model("my-model", 65_536)
        assert manager.get_max_context_tokens("my-model:13b") == 65_536

    def test_register_model_invalid_tokens(self) -> None:
        """register_model with non-positive context_tokens should raise ValueError."""
        manager = ContextWindowManager()
        with pytest.raises(ValueError, match="context_tokens"):
            manager.register_model("bad-model", 0)

    def test_invalid_fallback_context_window(self) -> None:
        """Non-positive fallback_context_window should raise ValueError."""
        with pytest.raises(ValueError, match="fallback_context_window"):
            ContextWindowManager(fallback_context_window=0)

    def test_case_insensitive_prefix_matching(self) -> None:
        """Prefix matching should be case-insensitive."""
        manager = ContextWindowManager()
        assert manager.get_max_context_tokens("Mistral:7b") == 32_768
        assert manager.get_max_context_tokens("LLAMA3:8b") == 8_192
