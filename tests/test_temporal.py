import pandas as pd
import pytest

from src.evaluation.temporal import (
    TEMPORAL_PERIODS,
    assign_temporal_periods,
)


def make_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": [
                "I1", "I1",
                "I2", "I2",
                "I3", "I3",
                "I4", "I4",
                "I5", "I5",
                "I6", "I6",
            ],
            "timestamp": [
                "2019-11-12 00:00:00",
                "2019-11-12 00:00:00",
                "2019-11-12 04:00:00",
                "2019-11-12 04:00:00",
                "2019-11-12 08:00:00",
                "2019-11-12 08:00:00",
                "2019-11-12 12:00:00",
                "2019-11-12 12:00:00",
                "2019-11-12 16:00:00",
                "2019-11-12 16:00:00",
                "2019-11-12 20:00:00",
                "2019-11-12 20:00:00",
            ],
            "article_id": [
                "A1", "A2",
                "A3", "A4",
                "A5", "A6",
                "A7", "A8",
                "A9", "A10",
                "A11", "A12",
            ],
            "clicked": [
                1, 0,
                0, 1,
                1, 0,
                0, 1,
                1, 0,
                0, 1,
            ],
        }
    )


def test_temporal_periods_exist():

    df = make_test_data()

    result = assign_temporal_periods(
        df,
        n_periods=3,
    )

    assert "temporal_period" in result.columns

    assert set(
        result["temporal_period"]
    ) == set(TEMPORAL_PERIODS)


def test_candidate_count_is_preserved():

    df = make_test_data()

    result = assign_temporal_periods(
        df
    )

    assert len(result) == len(df)


def test_each_impression_has_one_period():

    df = make_test_data()

    result = assign_temporal_periods(
        df
    )

    counts = (
        result
        .groupby("impression_id")[
            "temporal_period"
        ]
        .nunique()
    )

    assert (
        counts == 1
    ).all()


def test_temporal_order_is_preserved():

    df = make_test_data()

    result = assign_temporal_periods(
        df
    )

    periods = (
        result[
            [
                "impression_id",
                "timestamp",
                "temporal_period",
            ]
        ]
        .drop_duplicates(
            subset=["impression_id"]
        )
        .sort_values(
            "timestamp"
        )
    )

    ordered = (
        periods["temporal_period"]
        .tolist()
    )

    period_rank = {
        "early": 0,
        "middle": 1,
        "late": 2,
    }

    numeric = [
        period_rank[value]
        for value in ordered
    ]

    assert numeric == sorted(numeric)


def test_missing_columns_raise():

    df = pd.DataFrame(
        {
            "impression_id": ["I1"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        assign_temporal_periods(df)


def test_invalid_period_count_raise():

    df = make_test_data()

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        assign_temporal_periods(
            df,
            n_periods=1,
        )


def test_too_many_periods_raise():

    df = make_test_data()

    with pytest.raises(
        ValueError,
        match="at least",
    ):
        assign_temporal_periods(
            df,
            n_periods=10,
        )