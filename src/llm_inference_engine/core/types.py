"""Core types and data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class RequestStatus(str, Enum):
    """Status of an inference request."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")


@dataclass
class Request:
    """Represents a single inference request."""

    request_id: str
    prompt: str
    model: str
    generation_config: GenerationConfig
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    priority: int = 0
    status: RequestStatus = RequestStatus.PENDING
    max_wait_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate request."""
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if not self.request_id:
            raise ValueError("request_id cannot be empty")


@dataclass
class GenerationResult:
    """Result of text generation."""

    request_id: str
    text: str
    finish_reason: str
    tokens_used: int
    latency_ms: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Response to an inference request."""

    request_id: str
    result: Optional[GenerationResult] = None
    error: Optional[str] = None
    status: RequestStatus = RequestStatus.COMPLETED

    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return self.status == RequestStatus.COMPLETED and self.error is None


__all__ = [
    "RequestStatus",
    "GenerationConfig",
    "Request",
    "GenerationResult",
    "Response",
]
