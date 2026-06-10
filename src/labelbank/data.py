"""The label bank, generic pair loading, and the Eedi reference loader.

``load_eedi`` reproduces the competition's wide-to-long transformation (and
its ``AllText`` query template) byte-for-byte — pinned by a golden test
against a verbatim copy of the original polars pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

__all__ = ["LabelBank", "from_pairs", "load_eedi"]


@dataclass
class LabelBank:
    """A closed catalog of labels: parallel ``ids`` and ``texts``.

    The bank is the retrieval target. Keep it small enough to re-embed at
    evaluation time (hundreds to tens of thousands of entries).
    """

    ids: list
    texts: list

    def __post_init__(self):
        if len(self.ids) != len(self.texts):
            raise ValueError(
                f"ids ({len(self.ids)}) and texts ({len(self.texts)}) must align"
            )
        self._id_to_text = dict(zip(self.ids, self.texts))

    def __len__(self) -> int:
        return len(self.ids)

    def text_of(self, label_id) -> str:
        return self._id_to_text[label_id]

    def texts_of(self, label_ids: Sequence) -> list:
        return [self._id_to_text[i] for i in label_ids]

    @classmethod
    def from_csv(cls, path, id_col: str, text_col: str) -> "LabelBank":
        df = pd.read_csv(path)
        return cls(ids=df[id_col].tolist(), texts=df[text_col].tolist())


def from_pairs(queries: Sequence[str], gold_ids: Sequence) -> pd.DataFrame:
    """Build a canonical training frame from plain Python lists."""
    if len(queries) != len(gold_ids):
        raise ValueError("queries and gold_ids must align")
    return pd.DataFrame({"query": list(queries), "gold_id": list(gold_ids)})


_EEDI_QUERY_TEMPLATE = (
    "### Construct\n{construct}"
    "\n### Subject\n{subject}"
    "\n### Question\n{question}"
    "\n### Correct Answer\n{correct}"
    "\n### Wrong Answer\n{wrong}"
)


def load_eedi(
    train_csv,
    misconception_csv,
    keep_fold: bool = True,
) -> tuple:
    """Load the Eedi competition data into the canonical long format.

    Args:
        train_csv: Path to ``train.csv`` (optionally with a ``fold`` column).
        misconception_csv: Path to ``misconception_mapping.csv``.
        keep_fold: Carry a ``fold`` column through if present.

    Returns:
        ``(df, bank)`` where ``df`` has one row per (question, wrong answer)
        with columns ``QuestionId_Answer`` / ``AllText`` (the query text) /
        ``MisconceptionId`` (gold), sorted by ``QuestionId_Answer``, and
        ``bank`` is the :class:`LabelBank` of all misconceptions.
    """
    train_df = pd.read_csv(train_csv)
    mapping_df = pd.read_csv(misconception_csv)
    bank = LabelBank(
        ids=mapping_df["MisconceptionId"].tolist(),
        texts=mapping_df["MisconceptionName"].tolist(),
    )

    has_fold = keep_fold and "fold" in train_df.columns

    rows = []
    for _, q in train_df.iterrows():
        correct_text = q[f"Answer{q['CorrectAnswer']}Text"]
        for alpha in ["A", "B", "C", "D"]:
            mis_id = q.get(f"Misconception{alpha}Id")
            if pd.isna(mis_id):
                continue
            all_text = _EEDI_QUERY_TEMPLATE.format(
                construct=q["ConstructName"],
                subject=q["SubjectName"],
                question=q["QuestionText"],
                correct=correct_text,
                wrong=q[f"Answer{alpha}Text"],
            )
            row = {
                "QuestionId_Answer": f"{q['QuestionId']}_{alpha}",
                "AllText": all_text,
                "MisconceptionId": int(mis_id),
            }
            if has_fold:
                row["fold"] = q["fold"]
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("QuestionId_Answer").reset_index(drop=True)
    return df, bank
