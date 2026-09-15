import math

import pandas as pd
import pytest

from src.evaluation.phase5_reranker_evaluation import (
    evaluate_ranking,
)


def test_perfect_ranking():
    predictions = pd.DataFrame(
        {
            "impression_id": [1, 1, 1, 2, 2, 2],
            "clicked": [1, 0, 0, 1, 0, 0],
            "score": [0.9, 0.2, 0.1, 0.8, 0.3, 0.1],
        }
    )

    metrics = evaluate_ranking(predictions)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@5"] == pytest.approx(1.0)
    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["hit@5"] == pytest.approx(1.0)
    assert metrics["hit@10"] == pytest.approx(1.0)


def test_click_at_rank_three():
    predictions = pd.DataFrame(
        {
            "impression_id": [1, 1, 1, 1, 1],
            "clicked": [0, 0, 1, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    metrics = evaluate_ranking(predictions)

    assert metrics["mrr"] == pytest.approx(1.0 / 3.0)
    assert metrics["hit@5"] == pytest.approx(1.0)
    assert metrics["hit@10"] == pytest.approx(1.0)


def test_click_outside_top_five():
    predictions = pd.DataFrame(
        {
            "impression_id": [1] * 6,
            "clicked": [0, 0, 0, 0, 0, 1],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )

    metrics = evaluate_ranking(predictions)

    assert metrics["mrr"] == pytest.approx(1.0 / 6.0)
    assert metrics["hit@5"] == pytest.approx(0.0)
    assert metrics["hit@10"] == pytest.approx(1.0)


def test_multiple_clicked_articles():
    predictions = pd.DataFrame(
        {
            "impression_id": [1, 1, 1, 1],
            "clicked": [0, 1, 0, 1],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )

    metrics = evaluate_ranking(predictions)

    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["hit@5"] == pytest.approx(1.0)


def test_missing_columns():
    predictions = pd.DataFrame(
        {
            "impression_id": [1],
            "clicked": [1],
        }
    )

    with pytest.raises(ValueError):
        evaluate_ranking(predictions)


def test_empty_predictions():
    predictions = pd.DataFrame(
        columns=[
            "impression_id",
            "clicked",
            "score",
        ]
    )

    with pytest.raises(ValueError):
        evaluate_ranking(predictions)


def test_non_finite_scores():
    predictions = pd.DataFrame(
        {
            "impression_id": [1, 1],
            "clicked": [1, 0],
            "score": [math.nan, 0.1],
        }
    )

    with pytest.raises(ValueError):
        evaluate_ranking(predictions)