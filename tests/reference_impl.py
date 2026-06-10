"""Verbatim copies of the competition implementations, used as golden
references. Do not refactor this file — its value is being byte-for-byte
the code that earned the medal (see ``competition/``)."""

import numpy as np
import torch
from torch import Tensor


# --- competition/utils.py: apk / mapk (credit: kaggle.com/code/abdullahmeda/eedi-map-k-metric)
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# --- competition/utils.py: last_token_pool
def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# --- competition/loss_utils.py (verbatim, including the `loacl_scores` typo)
def get_local_score(q_reps, p_reps, all_scores):
    group_size = p_reps.size(0) // q_reps.size(0)
    indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
    specific_scores = []
    for i in range(group_size):
        specific_scores.append(
            all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
        )
    return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)


def _compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))


def compute_score(q_reps, p_reps, temperature):
    scores = _compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores


def compute_local_score(q_reps, p_reps, temperature):
    all_scores = compute_score(q_reps, p_reps, temperature)
    loacl_scores = get_local_score(q_reps, p_reps, all_scores)
    return loacl_scores


def compute_loss(scores, target):
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy(scores, target)


def compute_no_in_batch_neg_loss(q_reps, p_reps, temperature):
    local_scores = compute_local_score(q_reps, p_reps, temperature)
    local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
    loss = compute_loss(local_scores, local_targets)
    return local_scores, loss


# --- competition/stage1_train_retriever.py: adjust_passage_ids (row -> args)
def adjust_passage_ids(predict_list, misconception_id, topk=25):
    predict_list = list(predict_list)

    # If MisconceptionId is in preds_all_mm_ids, move it to the front
    if misconception_id in predict_list:
        predict_list.remove(misconception_id)
        predict_list.insert(0, misconception_id)
    else:
        # If it is not, insert it at the front and drop the last element
        predict_list.insert(0, misconception_id)
        predict_list.pop()

    predict_list = predict_list[:topk]

    return predict_list


# --- competition/stage1_train_retriever.py: add_suffix
def add_suffix(text, suffix_text, is_query):
    text = f"{suffix_text}{text}"
    text = text.strip()
    if is_query:
        text = f"{text}\n<response>"
    return text


# --- competition/stage2_train_reranker.py: candidate-block construction + PROMPT
def build_rerank_alltext(all_text, top_mm_texts):
    all_text = all_text + "\n\nHere are 5 possible candidates for misconception:\n"
    all_text = all_text + "\n".join(
        [f"{chr(65 + i)}. {candidate}" for i, candidate in enumerate(top_mm_texts)]
    )
    all_text = all_text + "\nWhich misconception candidate best explains what led to the wrong answer? (Please directly answer A, B, C, D or E)\nAnswer:"
    return all_text


PROMPT = """<|im_start|>system
Given a math question and its incorrect answer, identify the underlying misconception that led to the mistake.<|im_end|>
<|im_start|>user
{AllText}<|im_end|>
<|im_start|>assistant
{AnswerLetter}<|im_end|>
"""


def apply_template(all_text, answer_letter):
    return PROMPT.format(AllText=all_text, AnswerLetter=answer_letter)


# --- competition/stage1_train_retriever.py: wide -> long AllText pipeline (polars, verbatim chain)
def eedi_long_df(train_csv_path, mapping_csv_path):
    import polars as pl

    train_df = pl.read_csv(train_csv_path)

    common_col = [
        "QuestionId",
        "ConstructName",
        "SubjectName",
        "QuestionText",
        "CorrectAnswer",
        "fold",
    ]

    long_df = (
        train_df
        .select(
            pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]])
        )
        .with_columns(
            pl.when(pl.col("CorrectAnswer") == "A").then(pl.col("AnswerAText"))
            .when(pl.col("CorrectAnswer") == "B").then(pl.col("AnswerBText"))
            .when(pl.col("CorrectAnswer") == "C").then(pl.col("AnswerCText"))
            .when(pl.col("CorrectAnswer") == "D").then(pl.col("AnswerDText"))
            .otherwise(None)
            .alias("CorrectAnswerText")
        )
        .unpivot(
            index=common_col + ["CorrectAnswerText"],
            variable_name="AnswerType",
            value_name="AnswerText",
        )
        .with_columns(
            pl.concat_str(
                [
                    '### Construct\n' + pl.col("ConstructName"),
                    '\n### Subject\n' + pl.col("SubjectName"),
                    '\n### Question\n' + pl.col("QuestionText"),
                    '\n### Correct Answer\n' + pl.col("CorrectAnswerText"),
                    '\n### Wrong Answer\n' + pl.col("AnswerText"),
                ],
                separator="",
            ).alias("AllText"),
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str(
                [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
            ).alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
    )

    misconception_mapping_df_long = (
        train_df.select(
            pl.col(
                common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]
            )
        )
        .unpivot(
            index=common_col,
            variable_name="MisconceptionType",
            value_name="MisconceptionId",
        )
        .with_columns(
            pl.col("MisconceptionType")
            .str.extract(r"Misconception([A-D])Id$")
            .alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str(
                [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
            ).alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
        .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )

    long_df = long_df.join(misconception_mapping_df_long, on="QuestionId_Answer")

    long_df = long_df.to_pandas()
    import pandas as pd

    long_df = long_df[~pd.isna(long_df["MisconceptionId"])].reset_index(drop=True)
    long_df["MisconceptionId"] = long_df["MisconceptionId"].astype(int)
    long_df = long_df[["QuestionId_Answer", "AllText", "MisconceptionId", "fold"]]
    return long_df
