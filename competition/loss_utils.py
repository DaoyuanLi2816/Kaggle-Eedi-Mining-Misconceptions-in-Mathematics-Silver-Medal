import torch


def get_local_score(q_reps, p_reps, all_scores):
    """Get the local scores between queries and passages.

    Args:
        q_reps (torch.Tensor): Representations of the queries.
        p_reps (torch.Tensor): Representations of the passages.
        all_scores (torch.Tensor): All computed query-passage scores.

    Returns:
        torch.Tensor: Local scores used for loss computation.
    """
    group_size = p_reps.size(0) // q_reps.size(0) # Number of passages per query
    indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size # Index of each query within all_scores
    specific_scores = []
    for i in range(group_size):
        # Extract the score of the i-th passage for each query from all_scores
        specific_scores.append(
            all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i] # (batch_size, group_size)
        )
    # Stack all the selected scores together and reshape to (batch_size, group_size)
    return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)





def _compute_similarity(q_reps, p_reps):
    """Compute the similarity between the query and passage representations using the inner product.

    Args:
        q_reps (torch.Tensor): Representations of the queries.
        p_reps (torch.Tensor): Representations of the passages.

    Returns:
        torch.Tensor: The computed similarity matrix.
    """
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))


def compute_score(q_reps, p_reps, temperature):
    """Compute the scores between queries and passages.

    Args:
        q_reps (torch.Tensor): Representations of the queries.
        p_reps (torch.Tensor): Representations of the passages.
        temperature (float): Temperature parameter used to scale the scores.

    Returns:
        torch.Tensor: The scaled scores.
    """
    scores = _compute_similarity(q_reps, p_reps) / temperature  # (batch_size, group_size)
    scores = scores.view(q_reps.size(0), -1)  # (batch_size, group_size)
    return scores


def compute_local_score(q_reps, p_reps, temperature):
    """Compute the local scores between queries and passages.

    Args:
        q_reps (torch.Tensor): Representations of the queries.
        p_reps (torch.Tensor): Representations of the passages.
        temperature (float): Temperature parameter used to scale the scores.

    Returns:
        torch.Tensor: Local scores used for loss computation.
    """
    all_scores = compute_score(q_reps, p_reps, temperature)

    loacl_scores = get_local_score(q_reps, p_reps, all_scores)
    return loacl_scores



def compute_loss(scores, target):
    """Compute the loss using cross-entropy.

    Args:
        scores (torch.Tensor): The computed scores.
        target (torch.Tensor): The target values.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy(scores, target)


def compute_no_in_batch_neg_loss(q_reps, p_reps, temperature):
    """
    Compute the loss without using in-batch negatives or cross-device negatives.
    
    Args:
        q_reps (torch.Tensor): Representations of the queries, with shape (batch_size, dim).
        p_reps (torch.Tensor): Representations of the passages, with shape (batch_size * group_size, dim).
        temperature (float): Temperature parameter used to scale the scores.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Returns the local scores and the computed loss.
    """
    local_scores = compute_local_score(q_reps, p_reps, temperature)   # (batch_size, group_size)
    local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)  # (batch_size)
    
    loss = compute_loss(local_scores, local_targets)

    return local_scores, loss