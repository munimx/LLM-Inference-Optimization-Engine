"""Preference-to-model mapping for quantization variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_inference_engine.quantization.types import PreferencePriority, QuantizedModelInfo


@dataclass
class UserPreference:
    """User preference definition for model selection."""

    priority: PreferencePriority
    model_family: str | None = None
    max_memory_gb: float | None = None
    min_tokens_per_second: float | None = None
    acceptable_quality_drop: float | None = None


class QuantizationMapper:
    """Map user preferences to ranked model recommendations."""

    def __init__(self, models: list[QuantizedModelInfo]) -> None:
        """Initialize mapper with benchmark-enriched models."""
        self._models = models

    @classmethod
    def from_benchmark_payload(cls, payload: dict[str, Any]) -> QuantizationMapper:
        """Create mapper from benchmark JSON payload."""
        models = [QuantizedModelInfo.from_dict(model) for model in payload.get("models", [])]
        return cls(models=models)

    @classmethod
    def from_file(cls, path: Path) -> QuantizationMapper:
        """Create mapper from benchmark JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_benchmark_payload(payload)

    def select_model(self, preference: UserPreference) -> QuantizedModelInfo:
        """Select highest-ranked model for the user preference."""
        ranked = self.rank_models(preference)
        if not ranked:
            raise ValueError("No models meet the provided constraints")
        return ranked[0][0]

    def rank_models(self, preference: UserPreference) -> list[tuple[QuantizedModelInfo, float]]:
        """Rank candidate models and return sorted model/score pairs."""
        candidates = self._filter_by_constraints(preference)
        scored = [
            (model, self._calculate_preference_score(model, preference)) for model in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def get_recommendations(self, model_family: str) -> dict[str, QuantizedModelInfo]:
        """Get recommendations for common priorities in one family."""
        priorities = [
            PreferencePriority.SPEED,
            PreferencePriority.BALANCED,
            PreferencePriority.QUALITY,
            PreferencePriority.MEMORY_EFFICIENT,
        ]
        recommendations: dict[str, QuantizedModelInfo] = {}
        for priority in priorities:
            preference = UserPreference(priority=priority, model_family=model_family)
            ranked = self.rank_models(preference)
            if ranked:
                recommendations[priority.value] = ranked[0][0]
        return recommendations

    def _filter_by_constraints(self, preference: UserPreference) -> list[QuantizedModelInfo]:
        """Filter models by hard constraints."""
        filtered = self._models
        if preference.model_family:
            normalized_family = preference.model_family.lower()
            filtered = [model for model in filtered if model.family.lower() == normalized_family]
        if preference.max_memory_gb is not None:
            filtered = [
                model for model in filtered if model.memory_estimate_gb <= preference.max_memory_gb
            ]
        if preference.min_tokens_per_second is not None:
            filtered = [
                model
                for model in filtered
                if model.tokens_per_second is not None
                and model.tokens_per_second >= preference.min_tokens_per_second
            ]
        if preference.acceptable_quality_drop is not None:
            min_quality = 1.0 - preference.acceptable_quality_drop
            filtered = [
                model
                for model in filtered
                if model.quality_score is not None and model.quality_score >= min_quality
            ]
        return filtered

    def _calculate_preference_score(
        self, model: QuantizedModelInfo, preference: UserPreference
    ) -> float:
        """Calculate model score based on weighted preference objective."""
        speed = min((model.tokens_per_second or 0.0) / 100.0, 1.0)
        quality = model.quality_score or 0.0
        memory_efficiency = 1.0 / (1.0 + max(model.memory_estimate_gb, 0.0))
        if preference.priority == PreferencePriority.SPEED:
            return (0.65 * speed) + (0.2 * quality) + (0.15 * memory_efficiency)
        if preference.priority == PreferencePriority.BALANCED:
            return (0.4 * speed) + (0.4 * quality) + (0.2 * memory_efficiency)
        if preference.priority == PreferencePriority.QUALITY:
            return (0.05 * speed) + (0.9 * quality) + (0.05 * memory_efficiency)
        return (0.2 * speed) + (0.2 * quality) + (0.6 * memory_efficiency)
