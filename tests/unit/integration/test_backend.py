"""Tests for the abstract InferenceBackend and BackendResult."""

import pytest

from llm_inference_engine.integration.backend import BackendResult, InferenceBackend


class TestBackendResult:

    def test_defaults(self):
        r = BackendResult(text="hello")
        assert r.text == "hello"
        assert r.tokens_used == 0
        assert r.finish_reason == "stop"
        assert r.latency_ms == 0.0

    def test_custom_values(self):
        r = BackendResult(text="hi", tokens_used=10, prompt_tokens=5, finish_reason="length")
        assert r.tokens_used == 10
        assert r.prompt_tokens == 5
        assert r.finish_reason == "length"


class TestInferenceBackendABC:

    async def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InferenceBackend()  # type: ignore[abstract]
