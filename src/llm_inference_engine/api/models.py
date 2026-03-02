"""OpenAI-compatible Pydantic request/response models for the inference API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Request body for ``POST /completions``."""

    model: str = Field(..., description="Ollama model tag (e.g. 'llama3.1:8b')")
    prompt: str = Field(..., description="The prompt to complete")
    max_tokens: int = Field(default=256, ge=1, le=32_768, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream tokens (not yet implemented)")
    priority: int = Field(default=0, ge=0, le=10, description="Request priority (0=lowest)")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        """Ensure prompt is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("prompt cannot be empty or whitespace-only")
        return v


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        """Validate chat message role."""
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got {v!r}")
        return v


class ChatCompletionRequest(BaseModel):
    """Request body for ``POST /chat/completions``."""

    model: str = Field(..., description="Ollama model tag")
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)
    stream: bool = Field(default=False)
    priority: int = Field(default=0, ge=0, le=10)

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure at least one message is present."""
        if not v:
            raise ValueError("messages must contain at least one item")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    text: str
    finish_reason: str


class UsageInfo(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response body for ``POST /completions``."""

    id: str = Field(..., description="Request ID")
    object: str = Field(default="text_completion")
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo
    latency_ms: float = Field(..., description="Total request latency in milliseconds")


class ChatCompletionMessage(BaseModel):
    """Assistant message in a chat completion response."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Response body for ``POST /chat/completions``."""

    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    latency_ms: float


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    ollama_available: bool
    version: str = "0.1.0"
    details: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Response body for ``GET /metrics`` (simplified JSON metrics)."""

    committed_memory_gb: float
    available_memory_gb: float
    memory_limit_gb: float
    active_requests: int
    cache_hits: int
    cache_misses: int
    total_requests: int


class ErrorResponse(BaseModel):
    """Standardised error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None


__all__ = [
    "CompletionRequest",
    "ChatMessage",
    "ChatCompletionRequest",
    "CompletionChoice",
    "UsageInfo",
    "CompletionResponse",
    "ChatCompletionMessage",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]
