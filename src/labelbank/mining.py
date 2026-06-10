"""Self-bootstrapping hard-negative mining over a closed label bank.

The core loop of the competition recipe: train a retriever, rank the whole
bank for every training query, then reuse each query's own top-k ranking as
the candidate pool for the next round — with the gold label forced to the
front so position 0 is always the positive and positions 1..k-1 are the
hardest negatives the current model can produce.

In a fine-grained bank, random negatives are trivially easy and in-batch
negatives are often *false* negatives (sibling labels). Mined pools are what
make the contrastive signal informative; iterating rounds makes them harder.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

__all__ = ["gold_first_pool", "build_pools", "ensure_gold_in_top_k"]


def gold_first_pool(predicted: Sequence, gold, top_k: int = 25) -> list:
    """Build a retriever training pool from a ranked prediction list.

    Moves ``gold`` to the front if present; otherwise inserts it at the front
    and drops the last element (so mined pools keep a constant length before
    truncation). Truncates to ``top_k``.

    Position 0 of the result is the positive; the rest are hard negatives.
    """
    pool = list(predicted)

    if gold in pool:
        pool.remove(gold)
        pool.insert(0, gold)
    else:
        pool.insert(0, gold)
        pool.pop()

    return pool[:top_k]


def build_pools(
    all_predictions: Sequence[Sequence],
    golds: Sequence,
    top_k: int = 25,
) -> list:
    """Vectorized :func:`gold_first_pool` over a batch of queries."""
    return [gold_first_pool(p, g, top_k) for p, g in zip(all_predictions, golds)]


def ensure_gold_in_top_k(
    predicted: Sequence,
    gold,
    k: int = 5,
    rng: Optional[random.Random] = None,
) -> tuple:
    """Build a reranker training candidate set from a ranked prediction list.

    Takes the top-k predictions; if the gold label is missing, replaces the
    last candidate with it. The candidate order is then shuffled so the gold
    position is uniform — the listwise reranker must not learn a position
    prior (the retriever puts likelier candidates earlier).

    Returns:
        ``(candidates, gold_index)`` where ``candidates`` is the shuffled
        list of length ``k`` and ``gold_index`` is the gold's position in it.
    """
    top = list(predicted)[:k]
    if gold not in top:
        top = top[:-1] + [gold]

    rng = rng or random
    shuffled = list(top)
    rng.shuffle(shuffled)

    return shuffled, shuffled.index(gold)
