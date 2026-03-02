"""Lightweight prompt token estimation.

Provides a character-based heuristic for estimating prompt token counts
without requiring a model-specific tokenizer.  The ratio of ~4 characters
per token is a widely-used approximation for English text with BPE
tokenizers (GPT-style).
"""


def estimate_prompt_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the ~4 characters per token heuristic.  This is intentionally
    conservative: actual BPE tokenizers produce slightly fewer tokens for
    English text but more for code or non-Latin scripts.

    Args:
        text: The prompt or message text to estimate.

    Returns:
        Estimated token count (always >= 1 for non-empty text).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


__all__ = ["estimate_prompt_tokens"]
