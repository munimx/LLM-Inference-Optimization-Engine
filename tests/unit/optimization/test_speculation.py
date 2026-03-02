"""Unit tests for DraftModelManager and SpeculationEngine."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_inference_engine.optimization.draft_manager import DraftCandidate, DraftModelManager
from llm_inference_engine.optimization.speculation import SpeculationEngine, SpeculationResult


def _make_ollama_response(text: str, eval_count: int = 5) -> dict[str, Any]:
    return {"response": text, "done": True, "eval_count": eval_count, "eval_duration": 100_000}


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Build a mock OllamaClient that returns successive responses."""
    client = MagicMock()
    client.generate = AsyncMock(
        side_effect=[_make_ollama_response(r) for r in responses]
    )
    return client


# ------------------------------------------------------------------
# DraftModelManager tests
# ------------------------------------------------------------------


class TestDraftModelManager:
    """Tests for DraftModelManager."""

    def test_invalid_max_draft_tokens(self) -> None:
        """Non-positive max_draft_tokens should raise ValueError."""
        client = _make_mock_client([])
        with pytest.raises(ValueError, match="max_draft_tokens"):
            DraftModelManager(client, max_draft_tokens=0)

    async def test_generate_draft_returns_candidate(self) -> None:
        """generate_draft should return a DraftCandidate with non-empty tokens."""
        client = _make_mock_client(["fox jumps over"])
        manager = DraftModelManager(client, draft_model="phi3:mini", max_draft_tokens=8)
        candidate = await manager.generate_draft("The quick brown ")
        assert isinstance(candidate, DraftCandidate)
        assert candidate.draft_model == "phi3:mini"
        assert candidate.text == "fox jumps over"
        assert len(candidate.tokens) > 0

    async def test_generate_draft_empty_response(self) -> None:
        """Empty draft response should produce a candidate with no tokens."""
        client = _make_mock_client([""])
        manager = DraftModelManager(client, max_draft_tokens=4)
        candidate = await manager.generate_draft("Hello")
        assert candidate.tokens == []
        assert candidate.length == 0

    def test_split_tokens_basic(self) -> None:
        """_split_tokens should split text preserving whitespace tokens."""
        tokens = DraftModelManager._split_tokens("fox jumps")
        assert "fox" in tokens
        assert "jumps" in tokens

    def test_split_tokens_empty(self) -> None:
        """_split_tokens on empty string should return empty list."""
        assert DraftModelManager._split_tokens("") == []

    def test_properties(self) -> None:
        """draft_model and max_draft_tokens properties should reflect constructor args."""
        client = _make_mock_client([])
        manager = DraftModelManager(client, draft_model="gemma2:2b", max_draft_tokens=6)
        assert manager.draft_model == "gemma2:2b"
        assert manager.max_draft_tokens == 6


# ------------------------------------------------------------------
# SpeculationEngine tests
# ------------------------------------------------------------------


class TestSpeculationResult:
    """Tests for SpeculationResult dataclass."""

    def test_acceptance_rate(self) -> None:
        """acceptance_rate should be accepted / drafted."""
        result = SpeculationResult(
            text="hello world",
            total_tokens=2,
            drafted_tokens=4,
            accepted_tokens=3,
            rejected_tokens=1,
            num_drafts=1,
        )
        assert result.acceptance_rate == pytest.approx(0.75)

    def test_acceptance_rate_zero_drafted(self) -> None:
        """acceptance_rate should be 0.0 when drafted_tokens == 0."""
        result = SpeculationResult(
            text="",
            total_tokens=0,
            drafted_tokens=0,
            accepted_tokens=0,
            rejected_tokens=0,
            num_drafts=0,
        )
        assert result.acceptance_rate == 0.0

    def test_speedup_estimate_high_acceptance(self) -> None:
        """High acceptance rate should produce speedup > 1."""
        result = SpeculationResult(
            text="hello world how are you",
            total_tokens=5,
            drafted_tokens=8,
            accepted_tokens=6,
            rejected_tokens=2,
            num_drafts=2,
        )
        assert result.speedup_estimate > 1.0

    def test_speedup_estimate_zero_drafts(self) -> None:
        """Zero drafts should give speedup of 1.0."""
        result = SpeculationResult(
            text="",
            total_tokens=0,
            drafted_tokens=0,
            accepted_tokens=0,
            rejected_tokens=0,
            num_drafts=0,
        )
        assert result.speedup_estimate == 1.0


class TestSpeculationEngine:
    """Tests for SpeculationEngine."""

    def test_invalid_max_output_tokens(self) -> None:
        """Non-positive max_output_tokens should raise ValueError."""
        draft_manager = MagicMock()
        draft_manager.draft_model = "phi3:mini"
        target_client = MagicMock()
        with pytest.raises(ValueError, match="max_output_tokens"):
            SpeculationEngine(target_client, draft_manager, "llama3.1:8b", max_output_tokens=0)

    def test_invalid_max_rounds(self) -> None:
        """Non-positive max_rounds should raise ValueError."""
        draft_manager = MagicMock()
        draft_manager.draft_model = "phi3:mini"
        target_client = MagicMock()
        with pytest.raises(ValueError, match="max_rounds"):
            SpeculationEngine(
                target_client, draft_manager, "llama3.1:8b", max_output_tokens=64, max_rounds=0
            )

    async def test_generate_all_accepted(self) -> None:
        """When target agrees with all draft tokens, acceptance rate should be 1.0."""
        draft_text = "fox jumps"
        target_text = "fox jumps"

        draft_client = _make_mock_client([draft_text] * 10)
        target_client = _make_mock_client([target_text] * 10)

        draft_manager = DraftModelManager(draft_client, draft_model="phi3:mini", max_draft_tokens=4)
        engine = SpeculationEngine(
            target_client,
            draft_manager,
            target_model="llama3.1:8b",
            max_output_tokens=4,
            max_rounds=5,
        )
        result = await engine.generate("The quick brown ")
        assert result.total_tokens > 0
        assert result.acceptance_rate > 0.0

    async def test_generate_empty_draft_exits_early(self) -> None:
        """An empty draft response should terminate generation cleanly."""
        draft_client = _make_mock_client([""])
        target_client = _make_mock_client([])

        draft_manager = DraftModelManager(draft_client, max_draft_tokens=4)
        engine = SpeculationEngine(
            target_client,
            draft_manager,
            target_model="llama3.1:8b",
            max_output_tokens=16,
        )
        result = await engine.generate("Hello")
        assert result.total_tokens == 0
        assert result.num_drafts == 1  # one draft was attempted

    async def test_generate_rejected_tokens_counted(self) -> None:
        """Mismatched tokens should be counted as rejected."""
        draft_text = "cat"
        target_text = "dog"

        draft_client = _make_mock_client([draft_text] * 10)
        target_client = _make_mock_client([target_text] * 10)

        draft_manager = DraftModelManager(draft_client, max_draft_tokens=2)
        engine = SpeculationEngine(
            target_client,
            draft_manager,
            target_model="llama3.1:8b",
            max_output_tokens=4,
            max_rounds=2,
        )
        result = await engine.generate("The ")
        assert result.rejected_tokens > 0


class TestSpeculationResultDataclass:
    def test_acceptance_rate_zero_when_no_drafts(self) -> None:
        result = SpeculationResult(
            text="", total_tokens=0, drafted_tokens=0,
            accepted_tokens=0, rejected_tokens=0, num_drafts=0
        )
        assert result.acceptance_rate == 0.0

    def test_acceptance_rate_partial(self) -> None:
        result = SpeculationResult(
            text="hello", total_tokens=2, drafted_tokens=4,
            accepted_tokens=2, rejected_tokens=2, num_drafts=1
        )
        assert result.acceptance_rate == pytest.approx(0.5)

    def test_acceptance_rate_full(self) -> None:
        result = SpeculationResult(
            text="hi there", total_tokens=2, drafted_tokens=2,
            accepted_tokens=2, rejected_tokens=0, num_drafts=1
        )
        assert result.acceptance_rate == pytest.approx(1.0)

    def test_speedup_estimate_never_below_one(self) -> None:
        result = SpeculationResult(
            text="", total_tokens=0, drafted_tokens=4,
            accepted_tokens=0, rejected_tokens=4, num_drafts=1
        )
        assert result.speedup_estimate >= 1.0

    def test_speedup_estimate_high_acceptance(self) -> None:
        result = SpeculationResult(
            text="hello world", total_tokens=4, drafted_tokens=4,
            accepted_tokens=4, rejected_tokens=0, num_drafts=1
        )
        assert result.speedup_estimate >= 1.0

    def test_speedup_estimate_no_drafts(self) -> None:
        result = SpeculationResult(
            text="", total_tokens=0, drafted_tokens=0,
            accepted_tokens=0, rejected_tokens=0, num_drafts=0
        )
        assert result.speedup_estimate == 1.0


class TestSpeculationEngineEdges:
    def test_max_output_tokens_zero_raises(self) -> None:
        import pytest
        from unittest.mock import AsyncMock
        from llm_inference_engine.optimization.draft_manager import DraftModelManager
        mock = AsyncMock()
        dm = DraftModelManager(mock, max_draft_tokens=4)
        with pytest.raises(ValueError, match="max_output_tokens"):
            SpeculationEngine(mock, dm, target_model="llama3.1:8b", max_output_tokens=0)

    def test_max_rounds_zero_raises(self) -> None:
        from unittest.mock import AsyncMock
        from llm_inference_engine.optimization.draft_manager import DraftModelManager
        mock = AsyncMock()
        dm = DraftModelManager(mock, max_draft_tokens=4)
        with pytest.raises(ValueError, match="max_rounds"):
            SpeculationEngine(mock, dm, target_model="llama3.1:8b", max_rounds=0)

    async def test_max_rounds_limits_iterations(self) -> None:
        """Engine must stop after max_rounds even if output not exhausted."""
        draft_client = _make_mock_client(["word "] * 20)
        target_client = _make_mock_client(["word "] * 20)
        draft_manager = DraftModelManager(draft_client, max_draft_tokens=2)
        engine = SpeculationEngine(
            target_client, draft_manager,
            target_model="llama3.1:8b",
            max_output_tokens=100,
            max_rounds=2,
        )
        result = await engine.generate("prompt")
        assert result.num_drafts <= 2

    async def test_generate_returns_speculation_result(self) -> None:
        draft_client = _make_mock_client(["hello "] * 5)
        target_client = _make_mock_client(["hello "] * 5)
        draft_manager = DraftModelManager(draft_client, max_draft_tokens=3)
        engine = SpeculationEngine(
            target_client, draft_manager,
            target_model="llama3.1:8b",
            max_output_tokens=10,
        )
        result = await engine.generate("test prompt")
        assert isinstance(result, SpeculationResult)
        assert result.total_tokens >= 0
        assert result.num_drafts >= 1
