"""last_token_pool equivalence with the competition implementation."""

import pytest

torch = pytest.importorskip("torch")

import reference_impl as ref
from labelbank.retriever import last_token_pool


class TestCompetitionEquivalence:
    def test_left_padded(self):
        g = torch.Generator().manual_seed(0)
        hidden = torch.randn(3, 7, 16, generator=g)
        # Left padding: every sequence's final position is real.
        mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        assert torch.equal(last_token_pool(hidden, mask), ref.last_token_pool(hidden, mask))
        assert torch.equal(last_token_pool(hidden, mask), hidden[:, -1])

    def test_right_padded(self):
        g = torch.Generator().manual_seed(1)
        hidden = torch.randn(3, 7, 16, generator=g)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        ours = last_token_pool(hidden, mask)
        assert torch.equal(ours, ref.last_token_pool(hidden, mask))
        assert torch.equal(ours[0], hidden[0, 4])
        assert torch.equal(ours[1], hidden[1, 2])
        assert torch.equal(ours[2], hidden[2, 6])

    def test_fuzz(self):
        g = torch.Generator().manual_seed(2)
        for trial in range(50):
            b, t, d = 2 + trial % 5, 4 + trial % 9, 8
            hidden = torch.randn(b, t, d, generator=g)
            lengths = torch.randint(1, t + 1, (b,), generator=g)
            left = trial % 2 == 0
            mask = torch.zeros(b, t, dtype=torch.long)
            for i, L in enumerate(lengths.tolist()):
                if left:
                    mask[i, t - L :] = 1
                else:
                    mask[i, :L] = 1
            assert torch.equal(
                last_token_pool(hidden, mask), ref.last_token_pool(hidden, mask)
            )
