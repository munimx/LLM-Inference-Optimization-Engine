"""Types and data structures for quantization analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class QuantizationLevel(StrEnum):
    """Standard quantization levels."""

    Q2_K = "q2_K"
    Q3_K = "q3_K"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q4_K_S = "q4_K_S"
    Q4_K_M = "q4_K_M"
    Q5_K_S = "q5_K_S"
    Q5_K_M = "q5_K_M"
    Q6_K = "q6_K"
    Q8_0 = "q8_0"
    FP16 = "fp16"
    UNKNOWN = "unknown"


class PreferencePriority(StrEnum):
    """User preference priorities for model recommendation."""

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    MEMORY_EFFICIENT = "memory"


class QualityTier(StrEnum):
    """Quality tier classification for quantized models."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"


@dataclass
class QuantizedModelInfo:
    """Extended model info with quantization-focused metadata."""

    name: str
    family: str
    quantization: QuantizationLevel
    size_bytes: int
    memory_estimate_gb: float
    parameters: int | None
    context_length: int
    quality_tier: QualityTier
    format: str = "unknown"
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None
    memory_peak_mb: float | None = None
    quality_score: float | None = None

    def __post_init__(self) -> None:
        """Validate dataclass values."""
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        if self.memory_estimate_gb < 0:
            raise ValueError("memory_estimate_gb cannot be negative")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize object to dictionary."""
        return {
            "name": self.name,
            "family": self.family,
            "quantization": self.quantization.value,
            "size_bytes": self.size_bytes,
            "memory_estimate_gb": self.memory_estimate_gb,
            "parameters": self.parameters,
            "context_length": self.context_length,
            "quality_tier": self.quality_tier.value,
            "format": self.format,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizedModelInfo":
        """Deserialize object from dictionary."""
        return cls(
            name=str(data["name"]),
            family=str(data["family"]),
            quantization=QuantizationLevel(str(data["quantization"])),
            size_bytes=int(data["size_bytes"]),
            memory_estimate_gb=float(data["memory_estimate_gb"]),
            parameters=int(data["parameters"]) if data.get("parameters") is not None else None,
            context_length=int(data["context_length"]),
            quality_tier=QualityTier(str(data["quality_tier"])),
            format=str(data.get("format", "unknown")),
            tokens_per_second=(
                float(data["tokens_per_second"])
                if data.get("tokens_per_second") is not None
                else None
            ),
            time_to_first_token_ms=(
                float(data["time_to_first_token_ms"])
                if data.get("time_to_first_token_ms") is not None
                else None
            ),
            memory_peak_mb=(
                float(data["memory_peak_mb"]) if data.get("memory_peak_mb") is not None else None
            ),
            quality_score=(
                float(data["quality_score"]) if data.get("quality_score") is not None else None
            ),
        )


@dataclass
class QualityTestCase:
    """A test case for quality benchmarking."""

    prompt: str
    reference_outputs: list[str]
    task_type: str
    max_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate quality test case."""
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if not self.reference_outputs:
            raise ValueError("reference_outputs cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class QualityScore:
    """Quality score result for a model."""

    model_name: str
    bleu: float | None
    rouge: dict[str, float] | None
    semantic_similarity: float | None
    perplexity: float | None
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize quality score to dictionary."""
        return {
            "model_name": self.model_name,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "semantic_similarity": self.semantic_similarity,
            "perplexity": self.perplexity,
            "overall_score": self.overall_score,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    prompt: str
    max_tokens: int
    num_runs: int = 5
    warmup_runs: int = 2
    measure_memory: bool = True
    measure_quality: bool = False
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        """Validate benchmark configuration."""
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.num_runs <= 0:
            raise ValueError("num_runs must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    quantization: str
    tokens_per_second: float
    time_to_first_token_ms: float
    total_latency_ms: float
    memory_peak_mb: float
    memory_baseline_mb: float
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark result to dictionary."""
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_latency_ms": self.total_latency_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_baseline_mb": self.memory_baseline_mb,
            "generated_text": self.generated_text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        """Deserialize benchmark result from dictionary."""
        return cls(
            model_name=str(data["model_name"]),
            quantization=str(data["quantization"]),
            tokens_per_second=float(data["tokens_per_second"]),
            time_to_first_token_ms=float(data["time_to_first_token_ms"]),
            total_latency_ms=float(data["total_latency_ms"]),
            memory_peak_mb=float(data["memory_peak_mb"]),
            memory_baseline_mb=float(data["memory_baseline_mb"]),
            generated_text=str(data["generated_text"]),
            prompt_tokens=int(data["prompt_tokens"]),
            completion_tokens=int(data["completion_tokens"]),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
        )
