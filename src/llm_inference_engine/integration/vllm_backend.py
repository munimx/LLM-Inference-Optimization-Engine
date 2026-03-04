"""vLLM inference backend.

Implements :class:`~llm_inference_engine.integration.backend.InferenceBackend`
using vLLM's OpenAI-compatible HTTP API via an async ``httpx`` client.

Endpoints used:

- ``GET  /health/ready``         — liveness check
- ``GET  /v1/models``            — list available models
- ``POST /v1/completions``       — text completion
- ``POST /v1/chat/completions``  — chat completion (streaming and non-streaming)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from llm_inference_engine.integration.backend import BackendResult, InferenceBackend

logger = structlog.get_logger(__name__)


class VLLMBackend(InferenceBackend):
    """Async httpx client targeting a single vLLM instance.

    Args:
        base_url: Base URL of the vLLM server, e.g. ``"http://localhost:8080"``.
        timeout: HTTP request timeout in seconds.
        max_retries: Number of times to retry on transient errors.
        retry_backoff_seconds: Wait time between retries.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Release the underlying httpx client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------------
    # InferenceBackend implementation
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Return True if vLLM's /health/ready endpoint returns 200."""
        try:
            client = await self._get_client()
            response = await client.get("/health/ready")
            return response.status_code == 200
        except Exception as exc:
            logger.debug("vllm_health_check_failed", url=self._base_url, error=str(exc))
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs from /v1/models."""
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as exc:
            logger.warning("vllm_list_models_failed", url=self._base_url, error=str(exc))
            return []

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
        """Send a text completion request to /v1/completions."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        start = time.monotonic()
        data = await self._post_with_retry("/v1/completions", payload)
        latency_ms = (time.monotonic() - start) * 1000.0

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return BackendResult(
            text=choice["text"],
            tokens_used=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
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
        """Send a chat completion request to /v1/chat/completions."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        start = time.monotonic()
        data = await self._post_with_retry("/v1/chat/completions", payload)
        latency_ms = (time.monotonic() - start) * 1000.0

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return BackendResult(
            text=choice["message"]["content"],
            tokens_used=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream text completion tokens from /v1/completions (SSE)."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async for chunk in self._stream("/v1/completions", payload):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            text = choice.get("text", "")
            done = choice.get("finish_reason") is not None
            yield {"text": text, "done": done, "finish_reason": choice.get("finish_reason")}

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion tokens from /v1/chat/completions (SSE)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async for chunk in self._stream("/v1/chat/completions", payload):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            done = choice.get("finish_reason") is not None
            yield {"content": content, "done": done, "finish_reason": choice.get("finish_reason")}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST *payload* to *path*, retrying on transient network errors."""
        client = await self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    import asyncio
                    await asyncio.sleep(self._retry_backoff * (attempt + 1))
                    logger.debug(
                        "vllm_retry",
                        attempt=attempt + 1,
                        path=path,
                        error=str(exc),
                    )
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"vLLM returned {exc.response.status_code}: {exc.response.text}"
                ) from exc
        raise RuntimeError(
            f"vLLM request to {path} failed after {self._max_retries + 1} attempts"
        ) from last_exc

    async def _stream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed SSE JSON chunks from a streaming vLLM response."""
        import json

        client = await self._get_client()
        async with client.stream("POST", path, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    raw = line[6:]
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug("vllm_stream_parse_error", raw=raw)


__all__ = ["VLLMBackend"]
