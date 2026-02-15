"""Ollama client for HTTP communication with Ollama service."""

import asyncio
from typing import Any, Dict, List, Optional
import httpx
import structlog

from llm_inference_engine.exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    ModelNotFoundError,
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
    ) -> None:
        """Initialize Ollama client.

        Args:
            host: Ollama service hostname
            port: Ollama service port
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
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

    async def list_models(self) -> List[Dict[str, Any]]:
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

            data = response.json()
            models: List[Dict[str, Any]] = data.get("models", [])
            logger.info("models_listed", count=len(models))
            return models

        except httpx.HTTPError as e:
            logger.error("failed_to_list_models", error=str(e))
            raise OllamaConnectionError(f"Failed to list models: {e}")

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
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
            raise OllamaConnectionError(f"Failed to get model info: {e}")

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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

        payload: Dict[str, Any] = {
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
                if self._client is None:
                    await self.connect()
                
                # Safe type narrowing
                client = self._client
                if client is None:
                    raise OllamaConnectionError("Failed to initialize client")

                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()

                result: Dict[str, Any] = response.json()
                logger.info(
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
                await asyncio.sleep(2 ** attempt)

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
                await asyncio.sleep(2 ** attempt)

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
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPError as e:
                logger.error("generation_failed_fatal", model=model, error=str(e))
                raise OllamaConnectionError(f"Generation failed: {e}") from e

        raise OllamaConnectionError("Unexpected error: max retries exhausted")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Ollama service.

        Returns:
            Health status dictionary
        """
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
