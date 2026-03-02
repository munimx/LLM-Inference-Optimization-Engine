"""Ollama adapter — wraps :class:`OllamaClient` as an :class:`InferenceBackend`.

This is the default backend.  It delegates to the existing
:class:`~llm_inference_engine.integration.ollama_client.OllamaClient` so all
existing functionality (retries, error mapping, streaming) is preserved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llm_inference_engine.integration.backend import BackendResult, InferenceBackend
from llm_inference_engine.integration.ollama_client import OllamaClient


class OllamaBackend(InferenceBackend):
    """Ollama :class:`InferenceBackend` adapter.

    Args:
        client: An existing :class:`OllamaClient` instance.
    """

    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> BackendResult:
        result = await self._client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
        )
        return BackendResult(
            text=result.get("response", ""),
            tokens_used=result.get("eval_count", 0),
            prompt_tokens=result.get("prompt_eval_count", 0),
            finish_reason=result.get("done_reason", "stop"),
            metadata=result,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> BackendResult:
        result = await self._client.chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
        )
        msg = result.get("message", {})
        return BackendResult(
            text=msg.get("content", ""),
            tokens_used=result.get("eval_count", 0),
            prompt_tokens=result.get("prompt_eval_count", 0),
            finish_reason=result.get("done_reason", "stop"),
            metadata=result,
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._client.generate_stream(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
        ):
            yield chunk

    async def is_available(self) -> bool:
        return await self._client.is_available()

    async def list_models(self) -> list[str]:
        models = await self._client.list_models()
        return [m.get("name", "") for m in models]

    async def close(self) -> None:
        await self._client.close()


__all__ = ["OllamaBackend"]
