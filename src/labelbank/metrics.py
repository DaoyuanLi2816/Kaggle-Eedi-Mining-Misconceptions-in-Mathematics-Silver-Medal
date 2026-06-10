"""Ranking metrics for closed-bank retrieval: MAP@K and Recall@K.

The MAP@K implementation keeps the exact semantics used to score the Eedi
competition (and Kaggle's reference metric): per-query average precision over
the top-K predictions, averaged across queries.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["apk", "mapk", "recall_at_k"]


def apk(actual: Sequence, predicted: Sequence, k: int = 25) -> float:
    """Average precision at k for a single query.

    Args:
        actual: Relevant items for this query (order does not matter).
        predicted: Ranked predictions (order matters).
        k: Cutoff.

    Returns:
        AP@k. Predictions repeated earlier in the list are not credited twice.
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 25) -> float:
    """Mean average precision at k over a batch of queries."""
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def recall_at_k(
    gold_ids: Sequence,
    ranked_ids: Sequence[Sequence],
    ks: Sequence[int] = (1, 10, 25, 50, 100),
) -> dict:
    """Fraction of queries whose single gold id appears in the top-k ranking.

    Args:
        gold_ids: One gold label id per query.
        ranked_ids: Full (or sufficiently deep) ranking per query.
        ks: Cutoffs to report.

    Returns:
        ``{"recall@k": value}`` for each requested ``k``.
    """
    scores = {}
    n = len(gold_ids)
    for k in ks:
        num_correct = sum(
            1 if gt in preds[:k] else 0 for gt, preds in zip(gold_ids, ranked_ids)
        )
        scores[f"recall@{k}"] = num_correct / n
    return scores
