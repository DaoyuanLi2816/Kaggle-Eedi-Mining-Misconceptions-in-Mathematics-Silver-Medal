"""Hard-negative pool construction equivalence with the competition code."""

import random

import reference_impl as ref
from labelbank import build_pools, ensure_gold_in_top_k, gold_first_pool


class TestCompetitionEquivalence:
    def test_gold_present_moves_to_front(self):
        preds = [4, 8, 15, 16, 23, 42]
        assert gold_first_pool(preds, 16, top_k=4) == ref.adjust_passage_ids(preds, 16, 4)
        assert gold_first_pool(preds, 16, top_k=4)[0] == 16

    def test_gold_absent_inserted_and_last_dropped(self):
        preds = [4, 8, 15, 16, 23, 42]
        out = gold_first_pool(preds, 99, top_k=10)
        assert out == ref.adjust_passage_ids(preds, 99, 10)
        assert out[0] == 99 and 42 not in out and len(out) == 6

    def test_fuzz(self):
        rng = random.Random(7)
        for _ in range(500):
            depth = rng.randint(1, 60)
            preds = rng.sample(range(1000), depth)
            gold = rng.randrange(1000)
            topk = rng.choice([5, 25, 50])
            assert gold_first_pool(preds, gold, topk) == ref.adjust_passage_ids(
                preds, gold, topk
            )

    def test_input_not_mutated(self):
        preds = [1, 2, 3]
        gold_first_pool(preds, 9, top_k=3)
        assert preds == [1, 2, 3]

    def test_build_pools(self):
        all_preds = [[1, 2, 3], [4, 5, 6]]
        golds = [2, 9]
        assert build_pools(all_preds, golds, top_k=3) == [
            ref.adjust_passage_ids([1, 2, 3], 2, 3),
            ref.adjust_passage_ids([4, 5, 6], 9, 3),
        ]


class TestEnsureGoldInTopK:
    def test_gold_kept_when_present(self):
        cands, idx = ensure_gold_in_top_k([1, 2, 3, 4, 5, 6], gold=3, k=5, rng=random.Random(0))
        assert sorted(cands) == [1, 2, 3, 4, 5]
        assert cands[idx] == 3

    def test_gold_replaces_last_when_absent(self):
        # Mirrors the competition rule: top_mm_ids[:-1] + [MisconceptionId].
        cands, idx = ensure_gold_in_top_k([1, 2, 3, 4, 5, 6], gold=99, k=5, rng=random.Random(0))
        assert sorted(cands) == [1, 2, 3, 4, 99]
        assert 5 not in cands
        assert cands[idx] == 99

    def test_shuffle_is_seeded(self):
        a, _ = ensure_gold_in_top_k(list(range(10)), gold=0, k=5, rng=random.Random(123))
        b, _ = ensure_gold_in_top_k(list(range(10)), gold=0, k=5, rng=random.Random(123))
        assert a == b

    def test_gold_position_varies(self):
        rng = random.Random(0)
        positions = {
            ensure_gold_in_top_k(list(range(5)), gold=0, k=5, rng=rng)[1]
            for _ in range(100)
        }
        assert len(positions) == 5  # no position prior
