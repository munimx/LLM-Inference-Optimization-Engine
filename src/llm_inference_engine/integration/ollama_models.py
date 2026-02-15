"""Ollama model management."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import structlog

from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.exceptions import ModelNotFoundError

logger = structlog.get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about an Ollama model."""

    name: str
    size_bytes: int
    quantization: str
    family: str
    format: str
    parameters: Optional[int] = None
    context_length: int = 4096
    memory_estimate_gb: Optional[float] = None

    @classmethod
    def from_ollama_response(cls, data: Dict) -> "ModelInfo":
        """Create ModelInfo from Ollama API response.

        Args:
            data: Model data from Ollama API

        Returns:
            ModelInfo instance
        """
        size_bytes = data.get("size", 0)
        memory_estimate_gb = size_bytes / (1024**3) * 1.2  # Add 20% overhead

        # Extract quantization level from model name
        name = data.get("name", "")
        quantization = "unknown"
        if "q2" in name.lower():
            quantization = "2-bit"
        elif "q3" in name.lower():
            quantization = "3-bit"
        elif "q4" in name.lower():
            quantization = "4-bit"
        elif "q5" in name.lower():
            quantization = "5-bit"
        elif "q6" in name.lower():
            quantization = "6-bit"
        elif "q8" in name.lower():
            quantization = "8-bit"
        elif "fp16" in name.lower():
            quantization = "fp16"

        return cls(
            name=name,
            size_bytes=size_bytes,
            quantization=quantization,
            family=data.get("details", {}).get("family", "unknown"),
            format=data.get("details", {}).get("format", "unknown"),
            parameters=data.get("details", {}).get("parameter_count"),
            memory_estimate_gb=memory_estimate_gb,
        )


class OllamaModelManager:
    """Manages available Ollama models and their properties."""

    def __init__(self, client: OllamaClient) -> None:
        """Initialize model manager.

        Args:
            client: Ollama client instance
        """
        self.client = client
        self._model_cache: Dict[str, ModelInfo] = {}
        logger.info("model_manager_initialized")

    async def refresh_models(self) -> None:
        """Refresh the list of available models from Ollama."""
        try:
            models_data = await self.client.list_models()
            self._model_cache.clear()

            for model_data in models_data:
                model_info = ModelInfo.from_ollama_response(model_data)
                self._model_cache[model_info.name] = model_info

            logger.info("models_refreshed", count=len(self._model_cache))

        except Exception as e:
            logger.error("failed_to_refresh_models", error=str(e))
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models.

        Returns:
            List of ModelInfo objects

        Raises:
            OllamaConnectionError: If unable to connect to Ollama
        """
        if not self._model_cache:
            await self.refresh_models()

        return list(self._model_cache.values())

    async def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo for the requested model

        Raises:
            ModelNotFoundError: If model is not available
        """
        if not self._model_cache:
            await self.refresh_models()

        if model_name not in self._model_cache:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self._model_cache.keys())}"
            )

        return self._model_cache[model_name]

    async def verify_model_available(self, model_name: str) -> bool:
        """Verify if a model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available, False otherwise
        """
        try:
            await self.get_model_info(model_name)
            return True
        except ModelNotFoundError:
            return False

    async def get_model_memory_estimate(self, model_name: str) -> float:
        """Get estimated memory usage for a model.

        Args:
            model_name: Name of the model

        Returns:
            Estimated memory in GB

        Raises:
            ModelNotFoundError: If model is not available
        """
        model_info = await self.get_model_info(model_name)
        return model_info.memory_estimate_gb or 0.0


__all__ = ["ModelInfo", "OllamaModelManager"]
