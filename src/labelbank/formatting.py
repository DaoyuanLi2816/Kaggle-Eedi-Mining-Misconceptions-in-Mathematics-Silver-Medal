"""Prompt formatting for both stages, faithful to the competition templates.

Pure-Python (no torch) so it is importable from the core install and easy to
pin byte-for-byte in tests.
"""

from __future__ import annotations

import string
from typing import Optional, Sequence

__all__ = [
    "DEFAULT_QUERY_PREFIX",
    "RERANK_SYSTEM_PROMPT",
    "CHATML_RERANK_TEMPLATE",
    "format_query",
    "format_label",
    "build_rerank_text",
    "render_chatml",
]

# The stage-1 instruction prefix used in the competition (Eedi wording).
DEFAULT_QUERY_PREFIX = (
    "<instruct>Given a math question and its incorrect answer, identify the "
    "underlying misconception that led to the mistake.\n<query>"
)

# The stage-2 system prompt used in the competition.
RERANK_SYSTEM_PROMPT = (
    "Given a math question and its incorrect answer, identify the underlying "
    "misconception that led to the mistake."
)

# The stage-2 ChatML template used in the competition (Qwen-style).
CHATML_RERANK_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
"""


def format_query(text: str, prefix: str = "") -> str:
    """Format a retrieval query: ``prefix + text``, stripped, plus a
    ``<response>`` marker so last-token pooling lands on a stable position."""
    text = f"{prefix}{text}"
    text = text.strip()
    return f"{text}\n<response>"


def format_label(text: str, prefix: str = "") -> str:
    """Format a bank entry (passage side): ``prefix + text``, stripped."""
    text = f"{prefix}{text}"
    return text.strip()


def _letter_list(n: int) -> str:
    """Human list of option letters: ``"A, B, C, D or E"`` for ``n=5``."""
    letters = list(string.ascii_uppercase[:n])
    if n == 1:
        return letters[0]
    return ", ".join(letters[:-1]) + " or " + letters[-1]


def build_rerank_text(
    query_text: str,
    candidate_texts: Sequence[str],
    candidate_noun: str = "misconception",
    question: Optional[str] = None,
) -> str:
    """Build the listwise rerank user turn: query + lettered candidates + ask.

    With ``len(candidate_texts) == 5`` and the default ``candidate_noun``,
    reproduces the competition text byte-for-byte.
    """
    n = len(candidate_texts)
    text = query_text
    text = text + f"\n\nHere are {n} possible candidates for {candidate_noun}:\n"
    text = text + "\n".join(
        f"{chr(65 + i)}. {candidate}" for i, candidate in enumerate(candidate_texts)
    )
    if question is None:
        question = (
            f"\nWhich {candidate_noun} candidate best explains what led to the "
            f"wrong answer? (Please directly answer {_letter_list(n)})\nAnswer:"
        )
    return text + question


def render_chatml(user_text: str, answer_letter: str = "", system: str = RERANK_SYSTEM_PROMPT) -> str:
    """Render the full ChatML training example (or, with ``answer_letter=""``,
    a generation prompt ending right after the assistant tag)."""
    rendered = CHATML_RERANK_TEMPLATE.format(
        system=system, user=user_text, answer=answer_letter
    )
    if answer_letter == "":
        # Cut after the assistant tag for generation.
        marker = "<|im_start|>assistant\n"
        rendered = rendered[: rendered.index(marker) + len(marker)]
    return rendered
