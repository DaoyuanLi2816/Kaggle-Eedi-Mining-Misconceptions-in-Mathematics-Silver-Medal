"""Do self-mined hard negatives actually beat random negatives?

The library's central claim, measured end to end through its public API on a
public dataset: banking77 (77 customer-intent labels — a real closed label
bank). Four arms, identical per-arm training budgets:

1. **zero-shot** — the raw backbone as a bi-encoder, no training
2. **random negatives** — one epoch, pools = [gold + random labels]; this is
   also the *bootstrap* round that produces a competent miner
3. **mined, round 1** — fresh model, pools mined from arm 2's own top-k
4. **mined, round 2** — fresh model, pools mined from the round-1 model

This mirrors the competition protocol: each round is a fresh adapter trained
on pools ranked by the *previous* round's model. Mining only works once the
miner is competent — ``--cold-start`` instead mines round 1 from the
zero-shot model, which measurably *hurts* (the mined pools collapse to the
untrained embedder's biases). Numbers for both protocols are in the README.

Defaults fit a single consumer GPU (RTX 4080 16 GB, ~1 h end to end):

    python examples/mined_negatives_experiment.py

Requires: pip install -e .[retrieve] datasets
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random

from labelbank import BiEncoderRetriever, LabelBank, build_pools

QUERY_PREFIX = (
    "<instruct>Match the customer message to the banking intent it expresses."
    "\n<query>"
)


def load_banking77(train_size: int, test_size: int, seed: int):
    from datasets import load_dataset

    # The canonical repo is script-based (unsupported by datasets>=4); the
    # auto-converted parquet branch carries the same data + label names.
    ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
    label_names = ds["train"].features["label"].names
    bank = LabelBank(
        ids=list(range(len(label_names))),
        texts=[name.replace("_", " ") for name in label_names],
    )

    rng = random.Random(seed)

    def sample(split, n):
        idx = rng.sample(range(len(split)), n)
        return [split[i]["text"] for i in idx], [split[i]["label"] for i in idx]

    train_q, train_gold = sample(ds["train"], train_size)
    test_q, test_gold = sample(ds["test"], test_size)
    return bank, train_q, train_gold, test_q, test_gold


def fresh_retriever(model_name: str, trainable: bool) -> BiEncoderRetriever:
    return BiEncoderRetriever.from_pretrained(
        model_name,
        trainable=trainable,
        lora={"r": 16, "alpha": 32},
        query_prefix=QUERY_PREFIX,
        query_max_length=128,
        label_max_length=16,
    )


def free(retriever) -> None:
    import torch

    del retriever.model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(retriever, queries, golds, bank, batch_size):
    result = retriever.evaluate(
        queries, golds, bank, ks=(1, 3, 5, 10), map_k=25, batch_size=batch_size
    )
    result.pop("ranked")
    return {k: round(v, 4) for k, v in result.items()}


def trained_arm(args, bank, train_q, train_gold, test_q, test_gold, pools):
    """Train a fresh LoRA retriever on the given pools, return test metrics."""
    from labelbank import RetrieverTrainConfig, train_retriever

    retriever = fresh_retriever(args.model, trainable=True)
    train_retriever(
        retriever,
        train_q,
        [bank.texts_of(p) for p in pools],
        RetrieverTrainConfig(
            epochs=1,
            batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            lr=1e-4,
            temperature=0.01,
            seed=args.seed,
        ),
    )
    metrics = evaluate(retriever, test_q, test_gold, bank, args.eval_batch_size)
    return retriever, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/mined_neg")
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Mine round-1 pools from the zero-shot model instead of the "
        "random-negatives bootstrap model (ablation; measurably worse)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    bank, train_q, train_gold, test_q, test_gold = load_banking77(
        args.train_size, args.test_size, args.seed
    )
    print(f"bank: {len(bank)} labels | train: {len(train_q)} | test: {len(test_q)}")
    metrics = {}

    # --- zero-shot baseline
    print("[1/4] zero-shot baseline ...")
    base = fresh_retriever(args.model, trainable=False)
    metrics["zero_shot"] = evaluate(base, test_q, test_gold, bank, args.eval_batch_size)
    base_rankings = None
    if args.cold_start:
        base_rankings = base.retrieve(train_q, bank, batch_size=args.eval_batch_size)
    free(base)

    # --- random negatives: the bootstrap round, and the miner for round 1
    print("[2/4] random negatives (bootstrap) ...")
    rng = random.Random(args.seed)
    random_pools = []
    for gold in train_gold:
        others = [i for i in bank.ids if i != gold]
        random_pools.append([gold] + rng.sample(others, args.pool_size - 1))
    retriever, metrics["random_negatives"] = trained_arm(
        args, bank, train_q, train_gold, test_q, test_gold, random_pools
    )
    miner_rankings = retriever.retrieve(train_q, bank, batch_size=args.eval_batch_size)
    free(retriever)

    # --- mined round 1: fresh model, pools from the previous round's model
    # (the bootstrap model, or the zero-shot model under --cold-start)
    print("[3/4] mined hard negatives, round 1 ...")
    source = base_rankings if args.cold_start else miner_rankings
    mined_pools = build_pools(source, train_gold, top_k=args.pool_size)
    retriever, metrics["mined_round_1"] = trained_arm(
        args, bank, train_q, train_gold, test_q, test_gold, mined_pools
    )
    r1_rankings = retriever.retrieve(train_q, bank, batch_size=args.eval_batch_size)
    free(retriever)

    # --- mined round 2: fresh model, pools from the round-1 model
    print("[4/4] mined hard negatives, round 2 ...")
    mined_pools_2 = build_pools(r1_rankings, train_gold, top_k=args.pool_size)
    retriever, metrics["mined_round_2"] = trained_arm(
        args, bank, train_q, train_gold, test_q, test_gold, mined_pools_2
    )
    free(retriever)

    meta = {
        "model": args.model,
        "bank_size": len(bank),
        "train_size": len(train_q),
        "test_size": len(test_q),
        "pool_size": args.pool_size,
        "mining_protocol": "cold_start" if args.cold_start else "bootstrap",
    }
    out = {"meta": meta, "metrics": metrics}
    out_path = os.path.join(args.output, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    cols = ["map@25", "recall@1", "recall@3", "recall@5", "recall@10"]
    print("\n| arm | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for arm, m in metrics.items():
        print(f"| {arm} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
    print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()
