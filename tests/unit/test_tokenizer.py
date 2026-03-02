"""Tests for prompt token estimation utility."""

from llm_inference_engine.utils.tokenizer import estimate_prompt_tokens


class TestEstimatePromptTokens:

    def test_empty_string(self):
        assert estimate_prompt_tokens("") == 0

    def test_short_string(self):
        # "hello" = 5 chars → 5/4 = 1
        assert estimate_prompt_tokens("hello") >= 1

    def test_longer_string(self):
        text = "a" * 400
        tokens = estimate_prompt_tokens(text)
        assert tokens == 100  # 400 / 4

    def test_returns_int(self):
        assert isinstance(estimate_prompt_tokens("test"), int)
