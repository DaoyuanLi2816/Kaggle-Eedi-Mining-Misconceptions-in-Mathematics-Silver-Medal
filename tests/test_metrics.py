"""MAP@K equivalence with the competition metric, plus recall sanity."""

import random

import pytest

import reference_impl as ref
from labelbank import apk, mapk, recall_at_k


@pytest.fixture
def rng():
    return random.Random(0)


class TestCompetitionEquivalence:
    def test_apk_fuzz(self, rng):
        for _ in range(500):
            n_bank = rng.randint(1, 50)
            actual = [rng.randrange(n_bank)]
            depth = rng.randint(0, 40)
            predicted = [rng.randrange(n_bank) for _ in range(depth)]
            k = rng.choice([1, 5, 25])
            assert apk(actual, predicted, k) == ref.apk(actual, predicted, k)

    def test_apk_with_duplicates_and_empty(self):
        assert apk([3], [3, 3, 3], k=25) == ref.apk([3], [3, 3, 3], k=25)
        assert apk([], [1, 2], k=25) == ref.apk([], [1, 2], k=25) == 0.0

    def test_mapk_fuzz(self, rng):
        for _ in range(100):
            n = rng.randint(1, 20)
            actual = [[rng.randrange(100)] for _ in range(n)]
            predicted = [
                [rng.randrange(100) for _ in range(rng.randint(1, 30))] for _ in range(n)
            ]
            assert mapk(actual, predicted, 25) == pytest.approx(
                float(ref.mapk(actual, predicted, 25))
            )


class TestRecallAtK:
    def test_manual(self):
        gold = [1, 2, 3]
        ranked = [[1, 9, 9], [9, 2, 9], [9, 9, 9]]
        scores = recall_at_k(gold, ranked, ks=(1, 2, 3))
        assert scores["recall@1"] == pytest.approx(1 / 3)
        assert scores["recall@2"] == pytest.approx(2 / 3)
        assert scores["recall@3"] == pytest.approx(2 / 3)

    def test_matches_competition_loop(self):
        # The competition computed recall inline; mirror that computation.
        gold = [5, 7, 7, 1]
        ranked = [[5, 1], [1, 7], [7, 5], [2, 3]]
        for k in (1, 2):
            expected = sum(1 if g in r[:k] else 0 for g, r in zip(gold, ranked)) / len(gold)
            assert recall_at_k(gold, ranked, ks=(k,))[f"recall@{k}"] == pytest.approx(expected)
