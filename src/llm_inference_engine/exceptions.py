"""Custom exceptions for the LLM inference engine."""

from typing import Optional


class InferenceEngineError(Exception):
    """Base exception for all inference engine errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OllamaConnectionError(InferenceEngineError):
    """Raised when unable to connect to Ollama service."""

    pass


class OllamaTimeoutError(InferenceEngineError):
    """Raised when Ollama request times out."""

    pass


class ModelNotFoundError(InferenceEngineError):
    """Raised when requested model is not available."""

    pass


class ConfigurationError(InferenceEngineError):
    """Raised when configuration is invalid."""

    pass


class ValidationError(InferenceEngineError):
    """Raised when request validation fails."""

    pass


class OutOfMemoryError(InferenceEngineError):
    """Raised when system runs out of memory."""

    pass


class SchedulingError(InferenceEngineError):
    """Raised when scheduling operation fails."""

    pass


class BatchFormationError(InferenceEngineError):
    """Raised when batch formation fails."""

    pass


class CacheError(InferenceEngineError):
    """Raised when cache operation fails."""

    pass
