"""Eedi loader equivalence with the competition's polars pipeline.

Runs both the library loader (pandas) and a verbatim copy of the original
polars chain on the same synthetic CSVs and asserts identical output —
row order, query text bytes, and gold ids.
"""

import pandas as pd
import pytest

import reference_impl as ref
from labelbank import LabelBank, from_pairs, load_eedi


@pytest.fixture
def synthetic_eedi(tmp_path):
    rows = []
    # 4 questions; correct answers carry NaN misconceptions (dropped in long
    # format), one question missing one distractor's misconception too.
    specs = [
        (0, "B", {"A": 101, "C": 102, "D": 103}),
        (1, "A", {"B": 201, "C": 202, "D": 203}),
        (2, "D", {"A": 301, "B": 302, "C": None}),
        (3, "C", {"A": 401, "B": 402, "D": 403}),
    ]
    for qid, correct, mis in specs:
        row = {
            "QuestionId": qid,
            "ConstructName": f"Construct {qid} with, punctuation & symbols",
            "SubjectName": f"Subject {qid}",
            "QuestionText": f"What is {qid} + {qid}?\nChoose one.",
            "CorrectAnswer": correct,
            "fold": qid % 2,
        }
        for alpha in ["A", "B", "C", "D"]:
            row[f"Answer{alpha}Text"] = f"answer {alpha.lower()} of q{qid}"
            row[f"Misconception{alpha}Id"] = mis.get(alpha)
        rows.append(row)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    mapping = pd.DataFrame(
        {
            "MisconceptionId": [101, 102, 103, 201, 202, 203, 301, 302, 401, 402, 403],
            "MisconceptionName": [f"misconception #{i}" for i in range(11)],
        }
    )
    mapping_csv = tmp_path / "misconception_mapping.csv"
    mapping.to_csv(mapping_csv, index=False)
    return train_csv, mapping_csv


class TestCompetitionEquivalence:
    def test_long_format_identical(self, synthetic_eedi):
        train_csv, mapping_csv = synthetic_eedi
        ours, _ = load_eedi(train_csv, mapping_csv)
        theirs = ref.eedi_long_df(str(train_csv), str(mapping_csv))

        assert len(ours) == len(theirs) == 11
        assert ours["QuestionId_Answer"].tolist() == theirs["QuestionId_Answer"].tolist()
        assert ours["AllText"].tolist() == theirs["AllText"].tolist()  # byte-identical
        assert ours["MisconceptionId"].tolist() == theirs["MisconceptionId"].tolist()
        assert ours["fold"].tolist() == theirs["fold"].tolist()

    def test_template_structure(self, synthetic_eedi):
        train_csv, mapping_csv = synthetic_eedi
        ours, _ = load_eedi(train_csv, mapping_csv)
        text = ours["AllText"].iloc[0]
        for section in ("### Construct", "### Subject", "### Question", "### Correct Answer", "### Wrong Answer"):
            assert section in text


class TestLabelBank:
    def test_roundtrip(self, synthetic_eedi):
        _, mapping_csv = synthetic_eedi
        bank = LabelBank.from_csv(mapping_csv, "MisconceptionId", "MisconceptionName")
        assert len(bank) == 11
        assert bank.text_of(201) == "misconception #3"
        assert bank.texts_of([101, 403]) == ["misconception #0", "misconception #10"]

    def test_misaligned_raises(self):
        with pytest.raises(ValueError):
            LabelBank(ids=[1, 2], texts=["only one"])


class TestFromPairs:
    def test_basic(self):
        df = from_pairs(["q1", "q2"], [10, 20])
        assert df["query"].tolist() == ["q1", "q2"]
        assert df["gold_id"].tolist() == [10, 20]

    def test_misaligned_raises(self):
        with pytest.raises(ValueError):
            from_pairs(["q1"], [1, 2])
