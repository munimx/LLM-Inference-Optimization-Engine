"""Draft model manager for speculative decoding.

Manages the lifecycle of small "draft" models used in the draft-verify
speculation pipeline.  The draft model produces candidate token sequences
that are then verified by the larger target model, reducing end-to-end
latency when the acceptance rate is high.
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_inference_engine.integration.ollama_client import OllamaClient

logger = structlog.get_logger(__name__)


@dataclass
class DraftCandidate:
    """A candidate sequence produced by the draft model.

    Attributes:
        tokens: List of generated token strings.
        text: Concatenated text of all tokens.
        draft_model: Model tag of the draft model that produced this candidate.
        metadata: Raw response metadata from Ollama.
    """

    tokens: list[str]
    text: str
    draft_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of candidate tokens."""
        return len(self.tokens)


class DraftModelManager:
    """Manages the draft model used in speculative decoding.

    The draft model is typically a small, fast model (e.g. Phi-3-mini or
    Gemma-2B) that generates candidate token sequences ahead of the target
    model.  This class abstracts the generation call and provides token
    splitting utilities.

    Usage::

        manager = DraftModelManager(
            ollama_client=client,
            draft_model="phi3:mini",
            max_draft_tokens=8,
        )
        candidate = await manager.generate_draft(prompt="The quick brown")
        print(candidate.tokens)  # ['fox', ' jumps', ' over', ...]
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        draft_model: str = "phi3:mini",
        max_draft_tokens: int = 8,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the draft model manager.

        Args:
            ollama_client: Connected Ollama client.
            draft_model: Tag of the draft model to use.
            max_draft_tokens: Maximum number of tokens to generate per draft.
            temperature: Sampling temperature for the draft model (0 = greedy).
        """
        if max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        self._client = ollama_client
        self._draft_model = draft_model
        self._max_draft_tokens = max_draft_tokens
        self._temperature = temperature
        logger.info(
            "draft_model_manager_initialized",
            draft_model=draft_model,
            max_draft_tokens=max_draft_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_draft(self, prompt: str) -> DraftCandidate:
        """Generate a draft token sequence for *prompt*.

        Args:
            prompt: The current prompt (input tokens so far).

        Returns:
            A :class:`DraftCandidate` with the generated tokens.
        """
        raw: dict[str, Any] = await self._client.generate(
            model=self._draft_model,
            prompt=prompt,
            max_tokens=self._max_draft_tokens,
            temperature=self._temperature,
        )
        text = str(raw.get("response", ""))
        tokens = self._split_tokens(text)
        candidate = DraftCandidate(
            tokens=tokens,
            text=text,
            draft_model=self._draft_model,
            metadata={
                "eval_count": raw.get("eval_count"),
                "eval_duration_ns": raw.get("eval_duration"),
            },
        )
        logger.debug(
            "draft_generated",
            draft_model=self._draft_model,
            num_tokens=candidate.length,
        )
        return candidate

    @property
    def draft_model(self) -> str:
        """Draft model tag."""
        return self._draft_model

    @property
    def max_draft_tokens(self) -> int:
        """Maximum candidate tokens per draft call."""
        return self._max_draft_tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_tokens(text: str) -> list[str]:
        """Split *text* into a list of word-like tokens.

        This is a whitespace-preserving splitter that is sufficient for
        the acceptance scoring heuristic used by
        :class:`~llm_inference_engine.optimization.speculation.SpeculationEngine`.
        A proper sub-word tokenizer can replace this in a production setting.

        Args:
            text: The raw text to split.

        Returns:
            List of non-empty token strings.
        """
        if not text:
            return []
        # Preserve leading space per token for natural reconstruction.
        import re

        return [t for t in re.split(r"(\s+)", text) if t]


__all__ = ["DraftModelManager", "DraftCandidate"]
