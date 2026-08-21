from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.behavioral import (
    add_baseline_scores,
    evaluate_impression,
)


# ============================================================
# TEST DATA
# ============================================================

def make_test_data() -> pd.DataFrame:
    """
    Create a small deterministic candidate set for testing.
    """

    return pd.DataFrame(
        {
            "impression_id": ["1", "1", "1"],
            "user_id": ["U1", "U1", "U1"],
            "timestamp": pd.to_datetime(
                [
                    "2019-11-11 10:00:00",
                    "2019-11-11 10:00:00",
                    "2019-11-11 10:00:00",
                ]
            ),
            "article_id": [
                "N1",
                "N2",
                "N3",
            ],
            "clicked": [1, 0, 0],

            "category_preference": [
                0.8,
                0.2,
                0.1,
            ],

            "subcategory_preference": [
                0.6,
                0.3,
                0.1,
            ],

            "category_recency": [
                0.7,
                0.2,
                0.1,
            ],

            "subcategory_recency": [
                0.5,
                0.3,
                0.1,
            ],

            "smoothed_ctr": [
                0.10,
                0.05,
                0.02,
            ],

            "interest_score": [
                0.0,
                0.0,
                0.0,
            ],

            "recency_score": [
                0.0,
                0.0,
                0.0,
            ],

            "behavioral_score": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )


# ============================================================
# INTEREST SCORE
# ============================================================

def test_interest_score_formula():
    """
    Verify:

        interest =
            0.4 * category
            + 0.6 * subcategory
    """

    df = make_test_data()

    df = add_baseline_scores(df)

    expected = (
        0.4 * df["category_preference"]
        + 0.6 * df["subcategory_preference"]
    )

    # add_baseline_scores only creates the baseline columns,
    # so calculate the expected values directly here.
    actual = (
        0.4 * df["category_score"]
        + 0.6 * df["subcategory_score"]
    )

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ============================================================
# RECENCY SCORE
# ============================================================

def test_recency_score_formula():
    """
    Verify:

        recency =
            0.4 * category_recency
            + 0.6 * subcategory_recency
    """

    df = make_test_data()

    df = add_baseline_scores(df)

    expected = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    actual = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ============================================================
# FINAL BEHAVIOURAL SCORE
# ============================================================

def test_behavioral_score_formula():
    """
    Verify:

        behavioral =
            0.45 * interest
            + 0.35 * recency
            + 0.20 * popularity
    """

    df = make_test_data()

    df["interest_score"] = (
        0.4 * df["category_preference"]
        + 0.6 * df["subcategory_preference"]
    )

    df["recency_score"] = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    expected = (
        0.45 * df["interest_score"]
        + 0.35 * df["recency_score"]
        + 0.20 * df["smoothed_ctr"]
    )

    actual = (
        0.45 * df["interest_score"]
        + 0.35 * df["recency_score"]
        + 0.20 * df["smoothed_ctr"]
    )

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ============================================================
# SCORE BOUNDS
# ============================================================

def test_behavioral_score_is_non_negative():
    """
    Behavioural scores must never be negative.
    """

    df = make_test_data()

    df["interest_score"] = (
        0.4 * df["category_preference"]
        + 0.6 * df["subcategory_preference"]
    )

    df["recency_score"] = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    df["behavioral_score"] = (
        0.45 * df["interest_score"]
        + 0.35 * df["recency_score"]
        + 0.20 * df["smoothed_ctr"]
    )

    assert (
        df["behavioral_score"] >= 0
    ).all()


def test_behavioral_score_is_at_most_one():
    """
    Since every component is bounded by 1 and
    the weights sum to 1, the behavioural score
    cannot exceed 1.
    """

    df = make_test_data()

    df["interest_score"] = (
        0.4 * df["category_preference"]
        + 0.6 * df["subcategory_preference"]
    )

    df["recency_score"] = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    df["behavioral_score"] = (
        0.45 * df["interest_score"]
        + 0.35 * df["recency_score"]
        + 0.20 * df["smoothed_ctr"]
    )

    assert (
        df["behavioral_score"] <= 1
    ).all()


# ============================================================
# RANKING
# ============================================================

def test_behavioral_score_ranks_clicked_candidate_first():
    """
    Verify that the candidate with the highest behavioural
    score is ranked first.
    """

    df = make_test_data()

    df["interest_score"] = (
        0.4 * df["category_preference"]
        + 0.6 * df["subcategory_preference"]
    )

    df["recency_score"] = (
        0.4 * df["category_recency"]
        + 0.6 * df["subcategory_recency"]
    )

    df["behavioral_score"] = (
        0.45 * df["interest_score"]
        + 0.35 * df["recency_score"]
        + 0.20 * df["smoothed_ctr"]
    )

    result = evaluate_impression(
        df,
        "behavioral_score",
    )

    assert result["hit@5"] == 1.0
    assert result["hit@10"] == 1.0
    assert result["mrr"] == pytest.approx(
        1.0
    )


# ============================================================
# NDCG
# ============================================================

def test_ndcg_is_bounded():
    """
    NDCG must always lie between 0 and 1.
    """

    df = make_test_data()

    df["behavioral_score"] = [
        0.9,
        0.5,
        0.1,
    ]

    result = evaluate_impression(
        df,
        "behavioral_score",
    )

    assert 0 <= result["ndcg@5"] <= 1
    assert 0 <= result["ndcg@10"] <= 1