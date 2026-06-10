"""Contrastive loss over explicit candidate pools — no in-batch negatives.

Each query arrives with its own group of ``group_size`` passages where index
0 is the positive and the rest are (mined) hard negatives. The loss is
cross-entropy over the within-group similarity scores with target 0.

Deliberately *not* using in-batch negatives: in a closed, fine-grained label
bank, another query's positive is frequently a near-duplicate of this
query's gold (a sibling label), so in-batch terms inject false negatives.
The informative signal comes from the mined pool, not from batch noise.
"""

from __future__ import annotations

import torch

__all__ = [
    "get_local_score",
    "compute_score",
    "compute_local_score",
    "compute_loss",
    "no_in_batch_neg_loss",
    "compute_no_in_batch_neg_loss",
]


def get_local_score(q_reps: torch.Tensor, p_reps: torch.Tensor, all_scores: torch.Tensor) -> torch.Tensor:
    """Select each query's own group of scores from the full score matrix.

    Args:
        q_reps: Query representations, shape ``(batch_size, dim)``.
        p_reps: Passage representations, shape ``(batch_size * group_size, dim)``.
        all_scores: Full query-passage score matrix.

    Returns:
        Scores of shape ``(batch_size, group_size)`` where column 0 is each
        query's positive.
    """
    group_size = p_reps.size(0) // q_reps.size(0)
    indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
    specific_scores = []
    for i in range(group_size):
        specific_scores.append(
            all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
        )
    return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)


def _compute_similarity(q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))


def compute_score(q_reps: torch.Tensor, p_reps: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled similarity between all queries and all passages."""
    scores = _compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores


def compute_local_score(q_reps: torch.Tensor, p_reps: torch.Tensor, temperature: float) -> torch.Tensor:
    """Within-group scores of shape ``(batch_size, group_size)``."""
    all_scores = compute_score(q_reps, p_reps, temperature)
    return get_local_score(q_reps, p_reps, all_scores)


def compute_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over group scores."""
    cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
    return cross_entropy(scores, target)


def no_in_batch_neg_loss(
    q_reps: torch.Tensor,
    p_reps: torch.Tensor,
    temperature: float,
) -> tuple:
    """Group-wise contrastive loss with the positive at index 0 of each group.

    Args:
        q_reps: Query representations, shape ``(batch_size, dim)``. Expected
            L2-normalized.
        p_reps: Passage representations, shape ``(batch_size * group_size,
            dim)``, grouped per query with the positive first.
        temperature: Similarity scale (the competition used ``0.01``).

    Returns:
        ``(local_scores, loss)`` — the ``(batch_size, group_size)`` score
        matrix and the scalar loss.
    """
    local_scores = compute_local_score(q_reps, p_reps, temperature)
    local_targets = torch.zeros(
        local_scores.size(0), device=local_scores.device, dtype=torch.long
    )
    loss = compute_loss(local_scores, local_targets)
    return local_scores, loss


# Name used by the original competition code.
compute_no_in_batch_neg_loss = no_in_batch_neg_loss
