"""Prompt formatting equivalence with the competition templates."""

import random
import string

import reference_impl as ref
from labelbank import (
    DEFAULT_QUERY_PREFIX,
    build_rerank_text,
    format_label,
    format_query,
    render_chatml,
)


def _random_text(rng, n_words=8):
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 9))) for _ in range(n_words)]
    return " ".join(words)


class TestQueryFormatting:
    def test_matches_add_suffix_query(self):
        rng = random.Random(0)
        for _ in range(200):
            text = _random_text(rng)
            prefix = rng.choice(["", DEFAULT_QUERY_PREFIX, "  padded prefix "])
            assert format_query(text, prefix) == ref.add_suffix(text, prefix, is_query=True)

    def test_matches_add_suffix_label(self):
        rng = random.Random(1)
        for _ in range(200):
            text = _random_text(rng)
            prefix = rng.choice(["", "<label>", "  x "])
            assert format_label(text, prefix) == ref.add_suffix(text, prefix, is_query=False)


class TestRerankPrompt:
    def test_user_turn_byte_identical(self):
        rng = random.Random(2)
        for _ in range(100):
            all_text = _random_text(rng, n_words=30)
            candidates = [_random_text(rng, n_words=6) for _ in range(5)]
            assert build_rerank_text(all_text, candidates) == ref.build_rerank_alltext(
                all_text, candidates
            )

    def test_chatml_byte_identical(self):
        rng = random.Random(3)
        for _ in range(50):
            all_text = _random_text(rng, n_words=20)
            candidates = [_random_text(rng) for _ in range(5)]
            letter = rng.choice("ABCDE")
            user = build_rerank_text(all_text, candidates)
            assert render_chatml(user, answer_letter=letter) == ref.apply_template(
                ref.build_rerank_alltext(all_text, candidates), letter
            )

    def test_generation_prompt_ends_at_assistant_tag(self):
        prompt = render_chatml("question", answer_letter="")
        assert prompt.endswith("<|im_start|>assistant\n")
        assert "<|im_end|>\n<|im_start|>assistant\n" in prompt

    def test_letter_list_generalizes(self):
        three = build_rerank_text("q", ["a", "b", "c"])
        assert "Here are 3 possible candidates" in three
        assert "(Please directly answer A, B or C)" in three
