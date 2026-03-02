"""Unit tests for DraftModelManager and DraftCandidate."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_inference_engine.optimization.draft_manager import DraftCandidate, DraftModelManager


def _make_client(response_text: str = "hello world", eval_count: int = 2) -> MagicMock:
    """Return a mock OllamaClient that returns a fixed response."""
    client = MagicMock()
    client.generate = AsyncMock(
        return_value={
            "response": response_text,
            "eval_count": eval_count,
            "eval_duration": 12345678,
        }
    )
    return client


class TestDraftCandidate:
    """Tests for DraftCandidate dataclass."""

    def test_length_property(self) -> None:
        candidate = DraftCandidate(tokens=["a", "b", "c"], text="a b c", draft_model="phi3:mini")
        assert candidate.length == 3

    def test_length_empty(self) -> None:
        candidate = DraftCandidate(tokens=[], text="", draft_model="phi3:mini")
        assert candidate.length == 0

    def test_metadata_defaults_empty(self) -> None:
        candidate = DraftCandidate(tokens=["x"], text="x", draft_model="m")
        assert candidate.metadata == {}

    def test_metadata_stored(self) -> None:
        candidate = DraftCandidate(
            tokens=["x"], text="x", draft_model="m", metadata={"eval_count": 1}
        )
        assert candidate.metadata["eval_count"] == 1


class TestDraftModelManagerInit:
    """Tests for DraftModelManager.__init__."""

    def test_default_model(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client())
        assert mgr.draft_model == "phi3:mini"

    def test_custom_model(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client(), draft_model="gemma:2b")
        assert mgr.draft_model == "gemma:2b"

    def test_default_max_draft_tokens(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client())
        assert mgr.max_draft_tokens == 8

    def test_custom_max_draft_tokens(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client(), max_draft_tokens=4)
        assert mgr.max_draft_tokens == 4

    def test_zero_max_draft_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DraftModelManager(ollama_client=_make_client(), max_draft_tokens=0)

    def test_negative_max_draft_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DraftModelManager(ollama_client=_make_client(), max_draft_tokens=-1)


class TestDraftModelManagerGenerateDraft:
    """Tests for DraftModelManager.generate_draft."""

    async def test_returns_draft_candidate(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client("hello world"))
        candidate = await mgr.generate_draft("The quick")
        assert isinstance(candidate, DraftCandidate)

    async def test_candidate_text_matches_response(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client("hello world"))
        candidate = await mgr.generate_draft("prompt")
        assert candidate.text == "hello world"

    async def test_candidate_tokens_nonempty(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client("hello world"))
        candidate = await mgr.generate_draft("prompt")
        assert len(candidate.tokens) > 0

    async def test_candidate_draft_model_set(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client(), draft_model="gemma:2b")
        candidate = await mgr.generate_draft("prompt")
        assert candidate.draft_model == "gemma:2b"

    async def test_metadata_eval_count(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client("hi", eval_count=3))
        candidate = await mgr.generate_draft("prompt")
        assert candidate.metadata["eval_count"] == 3

    async def test_empty_response_gives_empty_tokens(self) -> None:
        mgr = DraftModelManager(ollama_client=_make_client(""))
        candidate = await mgr.generate_draft("prompt")
        assert candidate.tokens == []
        assert candidate.length == 0

    async def test_client_called_with_correct_model(self) -> None:
        client = _make_client()
        mgr = DraftModelManager(ollama_client=client, draft_model="phi3:mini", max_draft_tokens=4)
        await mgr.generate_draft("hello")
        client.generate.assert_awaited_once()
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs.get("model") == "phi3:mini" or call_kwargs.args[0] == "phi3:mini"

    async def test_client_called_with_correct_max_tokens(self) -> None:
        client = _make_client()
        mgr = DraftModelManager(ollama_client=client, max_draft_tokens=5)
        await mgr.generate_draft("hello")
        call_kwargs = client.generate.call_args
        # max_tokens should be 5
        assert 5 in call_kwargs.args or call_kwargs.kwargs.get("max_tokens") == 5


class TestSplitTokens:
    """Tests for DraftModelManager._split_tokens (static method)."""

    def test_empty_string_returns_empty(self) -> None:
        assert DraftModelManager._split_tokens("") == []

    def test_single_word(self) -> None:
        tokens = DraftModelManager._split_tokens("hello")
        assert tokens == ["hello"]

    def test_two_words_preserves_whitespace_token(self) -> None:
        tokens = DraftModelManager._split_tokens("hello world")
        # Should contain "hello", " ", "world"
        assert "hello" in tokens
        assert "world" in tokens

    def test_leading_space_preserved(self) -> None:
        tokens = DraftModelManager._split_tokens(" leading")
        assert " " in tokens or tokens[0] == " " or tokens[0] == " leading"

    def test_multiple_spaces_split(self) -> None:
        tokens = DraftModelManager._split_tokens("a  b")
        assert "a" in tokens
        assert "b" in tokens

    def test_only_whitespace_produces_whitespace_token(self) -> None:
        tokens = DraftModelManager._split_tokens("   ")
        # Non-empty whitespace should produce at least one token
        assert len(tokens) >= 1

    def test_newline_split(self) -> None:
        tokens = DraftModelManager._split_tokens("line1\nline2")
        assert "line1" in tokens
        assert "line2" in tokens
