"""Ollama client for HTTP communication with Ollama service."""

import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from llm_inference_engine.exceptions import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaTimeoutError,
)

logger = structlog.get_logger(__name__)


class OllamaClient:
    """HTTP client for communicating with Ollama service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        """Initialize Ollama client.

        Args:
            host: Ollama service hostname
            port: Ollama service port
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_backoff_seconds: Base delay (seconds) for exponential
                backoff.  Actual sleep = ``base × 2^attempt + jitter``
                where jitter is a uniform random value in [0, 1).
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._client: httpx.AsyncClient | None = None
        logger.info("ollama_client_initialized", base_url=self.base_url)

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Establish connection to Ollama service."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
            logger.info("ollama_connection_established")

    async def close(self) -> None:
        """Close connection to Ollama service."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("ollama_connection_closed")

    async def is_available(self) -> bool:
        """Check if Ollama service is available.

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            if self._client is None:
                await self.connect()

            client = self._client
            if client is None:
                logger.warning("client_initialization_failed")
                return False

            response = await client.get("/")
            return response.status_code == 200
        except Exception as e:
            logger.warning("ollama_unavailable", error=str(e))
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaConnectionError: If unable to connect to Ollama
        """
        try:
            if self._client is None:
                await self.connect()

            client = self._client
            if client is None:
                raise OllamaConnectionError("Failed to initialize client")

            response = await client.get("/api/tags")
            response.raise_for_status()

            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError) as e:
                logger.error("failed_to_list_models", error=str(e))
                raise OllamaConnectionError("Failed to list models: invalid JSON response") from e

            models: list[dict[str, Any]] = data.get("models", [])
            logger.info("models_listed", count=len(models))
            return models

        except httpx.HTTPError as e:
            logger.error("failed_to_list_models", error=str(e))
            raise OllamaConnectionError(f"Failed to list models: {e}") from e

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary

        Raises:
            ModelNotFoundError: If model doesn't exist
            OllamaConnectionError: If unable to connect
        """
        try:
            models = await self.list_models()
            for model in models:
                if model.get("name") == model_name:
                    logger.info("model_info_retrieved", model=model_name)
                    return model

            raise ModelNotFoundError(f"Model '{model_name}' not found")

        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error("failed_to_get_model_info", model=model_name, error=str(e))
            raise OllamaConnectionError(f"Failed to get model info: {e}") from e

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text using Ollama.

        Args:
            model: Model name to use
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences

        Returns:
            Generation result dictionary

        Raises:
            OllamaTimeoutError: If request times out
            OllamaConnectionError: If unable to connect
        """
        if self._client is None:
            await self.connect()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if max_tokens is not None:
            payload["num_predict"] = max_tokens

        if stop_sequences:
            payload["stop"] = stop_sequences

        for attempt in range(self.max_retries):
            try:
                # Safe type narrowing
                client = self._client
                if client is None:
                    raise OllamaConnectionError("Failed to initialize client")

                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()

                try:
                    result: dict[str, Any] = response.json()
                except (ValueError, json.JSONDecodeError) as e:
                    logger.error("invalid_json_response_generate", model=model, error=str(e))
                    raise OllamaConnectionError(f"Invalid JSON response from Ollama generation: {e}") from e

                logger.debug(
                    "generation_completed",
                    model=model,
                    tokens=result.get("eval_count", 0),
                    attempt=attempt + 1,
                )
                return result

            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    logger.error("generation_timeout", model=model, attempt=attempt + 1, error=str(e))
                    raise OllamaTimeoutError(
                        f"Request timed out after {self.max_retries} attempts: {e}"
                    ) from e

                logger.warning(
                    "generation_timeout_retry",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                    wait_time=2 ** attempt,
                )
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code < 500:
                    logger.error("generation_client_error", model=model, status=status_code)
                    raise OllamaConnectionError(f"Generation failed: {e}") from e

                if attempt == self.max_retries - 1:
                    logger.error("generation_server_error", model=model, status=status_code)
                    raise OllamaConnectionError(
                        f"Server error after {self.max_retries} attempts: {e}"
                    ) from e

                logger.warning(
                    "generation_server_error_retry",
                    model=model,
                    attempt=attempt + 1,
                    status=status_code,
                    wait_time=2 ** attempt,
                )
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

            except httpx.RequestError as e:
                if attempt == self.max_retries - 1:
                    logger.error("generation_connection_error", model=model, attempt=attempt + 1, error=str(e))
                    raise OllamaConnectionError(
                        f"Connection error after {self.max_retries} attempts: {e}"
                    ) from e

                logger.warning(
                    "generation_retry",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                    wait_time=2 ** attempt,
                )
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

        raise OllamaConnectionError("Unexpected error: max retries exhausted")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a chat completion using Ollama's /api/chat endpoint.

        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences

        Returns:
            Chat completion result dictionary

        Raises:
            OllamaTimeoutError: If request times out
            OllamaConnectionError: If unable to connect
        """
        if self._client is None:
            await self.connect()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        for attempt in range(self.max_retries):
            try:
                client = self._client
                if client is None:
                    raise OllamaConnectionError("Failed to initialize client")

                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()

                try:
                    result: dict[str, Any] = response.json()
                except (ValueError, json.JSONDecodeError) as e:
                    raise OllamaConnectionError(
                        f"Invalid JSON response from Ollama chat: {e}"
                    ) from e

                logger.debug(
                    "chat_completed",
                    model=model,
                    tokens=result.get("eval_count", 0),
                    attempt=attempt + 1,
                )
                return result

            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    raise OllamaTimeoutError(
                        f"Chat request timed out after {self.max_retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise OllamaConnectionError(f"Chat failed: {e}") from e
                if attempt == self.max_retries - 1:
                    raise OllamaConnectionError(
                        f"Server error after {self.max_retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

            except httpx.RequestError as e:
                if attempt == self.max_retries - 1:
                    raise OllamaConnectionError(
                        f"Connection error after {self.max_retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(self._retry_backoff * (2**attempt) + random.uniform(0, 1))

        raise OllamaConnectionError("Unexpected error: max retries exhausted")

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream text generation from Ollama, yielding each chunk.

        Yields:
            Dicts with at least a ``response`` key containing the token chunk.
            The final chunk has ``done: true``.
        """
        if self._client is None:
            await self.connect()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        client = self._client
        if client is None:
            raise OllamaConnectionError("Failed to initialize client")

        async with client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    yield chunk
                    if chunk.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion from Ollama, yielding each chunk.

        Yields:
            Dicts with a ``message`` key containing ``{"role": ..., "content": ...}``.
            The final chunk has ``done: true``.
        """
        if self._client is None:
            await self.connect()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        client = self._client
        if client is None:
            raise OllamaConnectionError("Failed to initialize client")

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    yield chunk
                    if chunk.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue

    async def embed(self, model: str, text: str) -> list[float]:
        """Get an embedding vector for *text* using the Ollama ``/api/embed`` endpoint.

        Args:
            model: Embedding model name (e.g. ``"nomic-embed-text"``).
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        if self._client is None:
            await self.connect()
        assert self._client is not None
        payload: dict[str, Any] = {"model": model, "input": text}
        response = await self._client.post(
            "/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns {"embeddings": [[...]]} — take first vector
        embeddings: list[Any] = data.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return list(embeddings[0])
        return list(embeddings)

    async def health_check(self) -> dict[str, Any]:
        try:
            is_available = await self.is_available()
            if not is_available:
                return {
                    "status": "unhealthy",
                    "message": "Ollama service is not reachable",
                }

            models = await self.list_models()
            return {
                "status": "healthy",
                "models_available": len(models),
                "base_url": self.base_url,
            }

        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "message": str(e),
            }


__all__ = ["OllamaClient"]
