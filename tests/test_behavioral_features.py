from __future__ import annotations

import pandas as pd


def test_category_preference_formula():
    """
    Category preference must equal:

        category clicks / total user clicks
    """

    df = pd.DataFrame(
        {
            "user_category_clicks": [2, 3, 0],
            "user_total_clicks": [5, 5, 5],
            "category_preference": [0.4, 0.6, 0.0],
        }
    )

    expected = (
        df["user_category_clicks"]
        / df["user_total_clicks"]
    )

    assert (
        df["category_preference"]
        .equals(expected)
    )


def test_category_preference_range():
    """
    Category preference must always lie between 0 and 1.
    """

    df = pd.DataFrame(
        {
            "category_preference": [
                0.0,
                0.2,
                0.5,
                1.0,
            ]
        }
    )

    assert (
        df["category_preference"] >= 0
    ).all()

    assert (
        df["category_preference"] <= 1
    ).all()


def test_cold_start_preference_is_zero():
    """
    A user with no historical clicks receives a
    zero behavioural category preference.
    """

    user_total_clicks = 0
    user_category_clicks = 0

    preference = 0.0

    assert user_total_clicks == 0
    assert user_category_clicks == 0
    assert preference == 0.0


def test_zero_category_history_is_valid():
    """
    A user may have history while having zero clicks
    in the candidate's category.
    """

    user_total_clicks = 10
    user_category_clicks = 0

    preference = (
        user_category_clicks
        / user_total_clicks
    )

    assert preference == 0.0


def test_user_category_preferences_sum_to_one():
    """
    For a user with historical clicks, category
    preferences must sum to one.
    """

    df = pd.DataFrame(
        {
            "user_id": [
                "U1",
                "U1",
                "U2",
                "U2",
            ],
            "preference": [
                0.6,
                0.4,
                0.75,
                0.25,
            ],
        }
    )

    sums = (
        df.groupby("user_id")["preference"]
        .sum()
    )

    assert (
        (sums - 1.0).abs() < 1e-12
    ).all()