"""Memory estimation, adaptive throttling, context window management, and speculative decoding."""

from llm_inference_engine.optimization.context import ContextWindowInfo, ContextWindowManager
from llm_inference_engine.optimization.draft_manager import DraftCandidate, DraftModelManager
from llm_inference_engine.optimization.memory import MemoryEstimator
from llm_inference_engine.optimization.speculation import SpeculationEngine, SpeculationResult
from llm_inference_engine.optimization.throttler import (
    AdaptiveThrottler,
    AdmissionDecision,
    ThrottlerStats,
)

__all__ = [
    "MemoryEstimator",
    "AdaptiveThrottler",
    "AdmissionDecision",
    "ThrottlerStats",
    "ContextWindowManager",
    "ContextWindowInfo",
    "DraftModelManager",
    "DraftCandidate",
    "SpeculationEngine",
    "SpeculationResult",
]
