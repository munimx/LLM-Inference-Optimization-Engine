"""Abstract inference backend interface.

Defines the contract that all inference backends (Ollama, vLLM, llama.cpp,
TGI, remote APIs) must implement, enabling the engine to work with
multiple providers.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendResult:
    """Standardised result from any inference backend."""

    text: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    metadata: dict[str, Any] | None = None


class InferenceBackend(abc.ABC):
    """Abstract base class for inference backends.

    All backends must implement at least :meth:`generate` and
    :meth:`is_available`.  Streaming and chat are optional — the default
    implementations raise ``NotImplementedError``.

    To add a new backend:

    1.  Subclass :class:`InferenceBackend`.
    2.  Implement the required abstract methods.
    3.  Register it via the backend factory or config.
    """

    @abc.abstractmethod
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
        """Generate a text completion."""

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
        """Generate a chat completion from structured messages.

        Default implementation flattens messages into a prompt string and
        delegates to :meth:`generate`.  Override for backends with native
        chat support.
        """
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return await self.generate(
            model, prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, stop=stop,
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream tokens for a text completion.

        Yields dicts with at least ``{"text": str, "done": bool}``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming"
        )
        # Make this an async generator so the type signature is satisfied
        yield  # type: ignore[misc]  # pragma: no cover

    @abc.abstractmethod
    async def is_available(self) -> bool:
        """Check whether the backend is reachable."""

    async def list_models(self) -> list[str]:
        """Return available model names (if the backend supports listing)."""
        return []

    async def close(self) -> None:
        """Release resources held by the backend."""


__all__ = ["BackendResult", "InferenceBackend"]
