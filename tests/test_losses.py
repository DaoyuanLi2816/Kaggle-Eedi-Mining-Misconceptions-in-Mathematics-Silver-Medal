"""Loss equivalence: labelbank.losses == competition loss, tensor-for-tensor."""

import pytest

torch = pytest.importorskip("torch")

import reference_impl as ref
from labelbank.losses import no_in_batch_neg_loss


class TestCompetitionEquivalence:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("group_size", [2, 5, 15, 25])
    @pytest.mark.parametrize("dim", [8, 64])
    def test_scores_and_loss_identical(self, batch_size, group_size, dim):
        g = torch.Generator().manual_seed(batch_size * 1000 + group_size * 10 + dim)
        q = torch.nn.functional.normalize(
            torch.randn(batch_size, dim, generator=g), p=2, dim=1
        )
        p = torch.nn.functional.normalize(
            torch.randn(batch_size * group_size, dim, generator=g), p=2, dim=1
        )

        ours_scores, ours_loss = no_in_batch_neg_loss(q, p, temperature=0.01)
        ref_scores, ref_loss = ref.compute_no_in_batch_neg_loss(q, p, temperature=0.01)

        assert torch.equal(ours_scores, ref_scores)
        assert torch.equal(ours_loss, ref_loss)

    def test_gradients_flow(self):
        q = torch.randn(2, 16, requires_grad=True)
        p = torch.randn(2 * 5, 16, requires_grad=True)
        _, loss = no_in_batch_neg_loss(q, p, temperature=0.05)
        loss.backward()
        assert q.grad is not None and p.grad is not None

    def test_perfect_retrieval_low_loss(self):
        # Positive identical to query, orthogonal negatives -> near-zero loss.
        q = torch.eye(4, 8)
        groups = []
        for i in range(4):
            groups.append(q[i])
            for j in range(2):
                neg = torch.zeros(8)
                neg[4 + ((i + j) % 4)] = 1.0
                groups.append(neg)
        p = torch.stack(groups)
        _, loss = no_in_batch_neg_loss(q, p, temperature=0.01)
        assert loss.item() < 1e-4
