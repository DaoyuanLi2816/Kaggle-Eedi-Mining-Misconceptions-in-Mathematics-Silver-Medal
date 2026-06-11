"""Config loading and data plumbing for the CLI pipelines (no models)."""

import pandas as pd
import pytest

from labelbank.run import RunConfig, load_config, load_data


class TestConfig:
    def test_defaults_roundtrip(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("model_name: foo/bar\npool_size: 8\n")
        cfg = load_config(str(cfg_file))
        assert cfg.model_name == "foo/bar"
        assert cfg.pool_size == 8
        assert cfg.stage == "retriever"
        assert cfg.temperature == 0.01  # competition default preserved

    def test_unknown_keys_rejected(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("pool_sized: 8\n")
        with pytest.raises(ValueError, match="pool_sized"):
            load_config(str(cfg_file))

    def test_empty_file_gives_defaults(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        assert load_config(str(cfg_file)) == RunConfig()

    def test_invalid_stage_rejected(self):
        with pytest.raises(ValueError, match="stage"):
            RunConfig(stage="train")

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError, match="data_format"):
            RunConfig(data_format="csv")

    def test_shipped_example_configs_load(self):
        import os

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("quickstart.yaml", "reproduce_competition.yaml"):
            cfg = load_config(os.path.join(here, "examples", "configs", name))
            assert cfg.stage == "retriever"


@pytest.fixture
def generic_data(tmp_path):
    train = pd.DataFrame(
        {
            "utterance": [f"query number {i}" for i in range(20)],
            "intent": [i % 4 for i in range(20)],
            "fold": [i % 5 for i in range(20)],
        }
    )
    bank = pd.DataFrame({"intent_id": [0, 1, 2, 3], "name": ["a", "b", "c", "d"]})
    train_csv = tmp_path / "train.csv"
    bank_csv = tmp_path / "bank.csv"
    train.to_csv(train_csv, index=False)
    bank.to_csv(bank_csv, index=False)
    return str(train_csv), str(bank_csv)


class TestLoadData:
    def _cfg(self, train_csv, bank_csv, **kw):
        return RunConfig(
            train_path=train_csv,
            bank_path=bank_csv,
            query_col="utterance",
            gold_col="intent",
            id_col="intent_id",
            text_col="name",
            **kw,
        )

    def test_generic_columns_renamed(self, generic_data):
        train_df, eval_df, bank = load_data(self._cfg(*generic_data))
        assert {"query", "gold_id"} <= set(train_df.columns)
        assert len(bank) == 4
        assert bank.text_of(2) == "c"

    def test_random_holdout(self, generic_data):
        train_df, eval_df, _ = load_data(self._cfg(*generic_data, eval_holdout=0.25))
        assert len(eval_df) == 5
        assert len(train_df) == 15

    def test_fold_split(self, generic_data):
        train_df, eval_df, _ = load_data(self._cfg(*generic_data, eval_fold=0))
        assert len(eval_df) == 4  # fold 0 of 5 folds over 20 rows
        assert len(train_df) == 16
        assert (eval_df["fold"] == 0).all()

    def test_holdout_is_seeded(self, generic_data):
        a = load_data(self._cfg(*generic_data))[1]
        b = load_data(self._cfg(*generic_data))[1]
        pd.testing.assert_frame_equal(a, b)
