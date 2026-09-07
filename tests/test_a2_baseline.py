import numpy as np
import pandas as pd
import pytest

from src.ranking.a2_baseline import (
    A2BaselineRanker,
    FEATURE_COLUMNS,
)


def make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_groups = 6
    rows_per_group = 5
    n = n_groups * rows_per_group

    df = pd.DataFrame(
        {
            "impression_id": np.repeat([f"I{i}" for i in range(n_groups)], rows_per_group),
            "article_id": [f"N{i}" for i in range(n)],
            "clicked": np.tile([1, 0, 0, 0, 0], n_groups),
        }
    )

    for i, column in enumerate(FEATURE_COLUMNS):
        df[column] = rng.random(n).astype(np.float32)

    return df


def test_feature_schema_is_fixed():
    assert len(FEATURE_COLUMNS) == 8
    assert "clicked" not in FEATURE_COLUMNS
    assert "impression_id" not in FEATURE_COLUMNS
    assert "article_id" not in FEATURE_COLUMNS


def test_fit_and_predict_scores():
    df = make_frame()
    ranker = A2BaselineRanker(max_iter=20)
    ranker.fit(df)

    scores = ranker.predict_scores(df)

    assert scores.shape == (len(df),)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_missing_feature_is_rejected():
    df = make_frame().drop(columns=["behavioral_score"])

    with pytest.raises(ValueError, match="Missing required"):
        A2BaselineRanker.validate_training_frame(df)


def test_invalid_label_is_rejected():
    df = make_frame()
    df.loc[0, "clicked"] = 2

    with pytest.raises(ValueError, match="binary"):
        A2BaselineRanker.validate_training_frame(df)


def test_rank_predictions_groups_by_impression():
    df = make_frame()
    ranker = A2BaselineRanker(max_iter=20).fit(df)
    scored = ranker.add_scores(df)
    ranked = ranker.rank_predictions(scored)

    assert len(ranked) == len(df)
    assert ranked["impression_id"].nunique() == df["impression_id"].nunique()

    for _, group in ranked.groupby("impression_id", sort=False):
        scores = group["baseline_predicted_score"].to_numpy()
        assert np.all(scores[:-1] >= scores[1:])


def test_save_and_load_round_trip(tmp_path):
    df = make_frame()
    ranker = A2BaselineRanker(max_iter=20).fit(df)

    path = tmp_path / "ranker.joblib"
    ranker.save(path)
    loaded = A2BaselineRanker.load(path)

    np.testing.assert_allclose(
        ranker.predict_scores(df),
        loaded.predict_scores(df),
        rtol=0,
        atol=1e-7,
    )
