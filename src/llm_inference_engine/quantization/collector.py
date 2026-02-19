"""Quantization metadata collection from Ollama models."""

import re

import structlog

from llm_inference_engine.integration.ollama_client import OllamaClient
from llm_inference_engine.integration.ollama_models import ModelInfo, OllamaModelManager
from llm_inference_engine.quantization.types import (
    QualityTier,
    QuantizationLevel,
    QuantizedModelInfo,
)

logger = structlog.get_logger(__name__)

_QUANT_PATTERNS: list[tuple[re.Pattern[str], QuantizationLevel]] = [
    (re.compile(r"q2[_-]?k", re.IGNORECASE), QuantizationLevel.Q2_K),
    (re.compile(r"q3[_-]?k", re.IGNORECASE), QuantizationLevel.Q3_K),
    (re.compile(r"q4[_-]?0", re.IGNORECASE), QuantizationLevel.Q4_0),
    (re.compile(r"q4[_-]?1", re.IGNORECASE), QuantizationLevel.Q4_1),
    (re.compile(r"q4[_-]?k[_-]?s", re.IGNORECASE), QuantizationLevel.Q4_K_S),
    (re.compile(r"q4[_-]?k[_-]?m", re.IGNORECASE), QuantizationLevel.Q4_K_M),
    (re.compile(r"q5[_-]?k[_-]?s", re.IGNORECASE), QuantizationLevel.Q5_K_S),
    (re.compile(r"q5[_-]?k[_-]?m", re.IGNORECASE), QuantizationLevel.Q5_K_M),
    (re.compile(r"q6[_-]?k", re.IGNORECASE), QuantizationLevel.Q6_K),
    (re.compile(r"q8[_-]?0", re.IGNORECASE), QuantizationLevel.Q8_0),
    (re.compile(r"fp16|f16", re.IGNORECASE), QuantizationLevel.FP16),
]


class QuantizationInfoCollector:
    """Collect and normalize quantization information from available models."""

    def __init__(self, client: OllamaClient, model_manager: OllamaModelManager) -> None:
        """Initialize collector with Ollama dependencies."""
        self._client = client
        self._model_manager = model_manager
        self._cache: list[QuantizedModelInfo] | None = None

    async def collect_all_quantizations(self) -> list[QuantizedModelInfo]:
        """Collect quantization metadata for all available models."""
        if self._cache is not None:
            return self._cache

        model_infos = await self._model_manager.get_available_models()
        collected = [self._to_quantized_model_info(model_info) for model_info in model_infos]
        collected.sort(key=lambda model: (model.family, model.name))
        self._cache = collected
        logger.info("quantizations_collected", count=len(collected))
        return collected

    async def collect_model_family(self, family: str) -> list[QuantizedModelInfo]:
        """Collect quantization metadata for one model family."""
        normalized = family.strip().lower()
        all_models = await self.collect_all_quantizations()
        return [model for model in all_models if model.family.lower() == normalized]

    async def get_quantization_variants(self, base_model: str) -> list[QuantizedModelInfo]:
        """Get all quantization variants for a model prefix."""
        normalized = base_model.strip().lower()
        all_models = await self.collect_all_quantizations()
        variants = [model for model in all_models if model.name.lower().startswith(normalized)]
        variants.sort(key=lambda model: model.quantization.value)
        return variants

    def _to_quantized_model_info(self, model_info: ModelInfo) -> QuantizedModelInfo:
        """Convert `ModelInfo` to `QuantizedModelInfo`."""
        quant_level = self._parse_quantization_level(model_info.name)
        quality_tier = self._estimate_quality_tier(quant_level)
        memory_estimate_gb = (
            model_info.memory_estimate_gb or model_info.size_bytes / (1024**3) * 1.2
        )
        return QuantizedModelInfo(
            name=model_info.name,
            family=model_info.family,
            quantization=quant_level,
            size_bytes=model_info.size_bytes,
            memory_estimate_gb=memory_estimate_gb,
            parameters=model_info.parameters,
            context_length=model_info.context_length,
            quality_tier=quality_tier,
            format=model_info.format,
        )

    def _parse_quantization_level(self, model_name: str) -> QuantizationLevel:
        """Parse quantization level from model name."""
        for pattern, quant_level in _QUANT_PATTERNS:
            if pattern.search(model_name):
                return quant_level
        return QuantizationLevel.UNKNOWN

    def _estimate_quality_tier(self, quant_level: QuantizationLevel) -> QualityTier:
        """Estimate quality tier from quantization level."""
        if quant_level in {QuantizationLevel.FP16, QuantizationLevel.Q8_0}:
            return QualityTier.EXCELLENT
        if quant_level in {
            QuantizationLevel.Q6_K,
            QuantizationLevel.Q5_K_M,
            QuantizationLevel.Q5_K_S,
        }:
            return QualityTier.GOOD
        if quant_level in {
            QuantizationLevel.Q4_0,
            QuantizationLevel.Q4_1,
            QuantizationLevel.Q4_K_M,
            QuantizationLevel.Q4_K_S,
        }:
            return QualityTier.ACCEPTABLE
        return QualityTier.DEGRADED
