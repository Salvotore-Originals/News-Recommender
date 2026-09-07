import numpy as np
import pandas as pd
import pytest

from src.evaluation.a2_baseline import evaluate_validation, predict_validation
from src.ranking.a2_baseline import A2BaselineRanker, FEATURE_COLUMNS, SCORE_COLUMN


def make_frame():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "impression_id": np.repeat(["I1", "I2", "I3"], 4),
        "user_id": ["U1"] * 4 + ["U2"] * 4 + ["U3"] * 4,
        "timestamp": pd.date_range("2019-11-12", periods=12, freq="s"),
        "article_id": [f"N{i}" for i in range(12)],
        "clicked": np.tile([1, 0, 0, 0], 3),
    })
    for column in FEATURE_COLUMNS:
        df[column] = rng.random(len(df)).astype(np.float32)
    return df


@pytest.fixture
def fitted_ranker():
    return A2BaselineRanker(max_iter=20).fit(make_frame())


def test_validation_prediction_preserves_rows(fitted_ranker):
    df = make_frame()
    scored = predict_validation(fitted_ranker, df)
    assert len(scored) == len(df)
    assert scored["impression_id"].equals(df["impression_id"])
    assert scored["article_id"].equals(df["article_id"])
    assert SCORE_COLUMN in scored.columns
    assert np.isfinite(scored[SCORE_COLUMN]).all()


def test_validation_scores_are_probabilities(fitted_ranker):
    scores = predict_validation(fitted_ranker, make_frame())[SCORE_COLUMN].to_numpy()
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_validation_evaluation_returns_all_metrics(fitted_ranker):
    metrics = evaluate_validation(predict_validation(fitted_ranker, make_frame()))
    assert set(metrics) == {"mrr", "hit@5", "hit@10", "ndcg@5", "ndcg@10"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())


def test_evaluation_rejects_missing_score():
    with pytest.raises(ValueError, match="Score column"):
        evaluate_validation(make_frame())
