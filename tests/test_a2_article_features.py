import pandas as pd
import pytest

from src.data.ebnerd_article_features import (
    attach_article_metadata,
    build_article_features,
)


def make_articles():
    return pd.DataFrame(
        {
            "article_id": ["A", "B", "C"],
            "category": [1, 2, 1],
            "category_str": ["Sports", "Politics", "Sports"],
            "published_time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                ]
            ),
        }
    )


def make_interactions():
    return pd.DataFrame(
        {
            "impression_id": [
                1, 1,
                2, 2,
                3, 3,
            ],
            "user_id": [
                10, 10,
                10, 10,
                10, 10,
            ],
            "article_id": [
                "A", "B",
                "A", "C",
                "B", "A",
            ],
            "impression_time": pd.to_datetime(
                [
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 03:00:00",
                    "2026-01-01 03:00:00",
                ]
            ),
            "clicked": [
                1, 0,
                0, 1,
                0, 1,
            ],
            "session_id": [
                100, 100,
                100, 100,
                100, 100,
            ],
        }
    )


def prepare():
    interactions = make_interactions()
    articles = make_articles()

    return attach_article_metadata(
        interactions,
        articles,
    )


def test_first_impression_has_no_article_history():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    first = result[
        result["impression_id"] == 1
    ]

    assert (
        first["article_impression_count_before"]
        == 0
    ).all()

    assert (
        first["article_click_count_before"]
        == 0
    ).all()


def test_previous_impression_is_visible():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    second = result[
        result["impression_id"] == 2
    ]

    row_a = second[
        second["article_id"] == "A"
    ].iloc[0]

    assert row_a[
        "article_impression_count_before"
    ] == 1

    assert row_a[
        "article_click_count_before"
    ] == 1


def test_current_impression_does_not_leak():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    first = result[
        result["impression_id"] == 1
    ]

    # A is clicked in the current impression.
    # Its own feature must still see zero previous clicks.
    row_a = first[
        first["article_id"] == "A"
    ].iloc[0]

    assert row_a[
        "article_click_count_before"
    ] == 0

    assert row_a[
        "article_impression_count_before"
    ] == 0


def test_candidates_in_same_impression_do_not_leak_to_each_other():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    first = result[
        result["impression_id"] == 1
    ]

    assert (
        first["article_impression_count_before"]
        == 0
    ).all()

    assert (
        first["article_click_count_before"]
        == 0
    ).all()


def test_ctr_is_smoothed():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    second = result[
        result["impression_id"] == 2
    ]

    row_a = second[
        second["article_id"] == "A"
    ].iloc[0]

    # One previous impression and one previous click:
    #
    # (1 + 1) / (1 + 1 + 1) = 2/3
    #
    assert row_a[
        "article_ctr_before"
    ] == pytest.approx(
        2 / 3
    )


def test_article_age_is_correct():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    row = result.iloc[0]

    assert row[
        "article_age_hours"
    ] == pytest.approx(
        1.0
    )

    assert row[
        "article_age_days"
    ] == pytest.approx(
        1 / 24
    )

    assert row[
        "article_published_after_impression"
    ] == 0


def test_article_metadata_is_preserved():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    expected = {
        "A": "Sports",
        "B": "Politics",
        "C": "Sports",
    }

    actual = (
        result[
            [
                "article_id",
                "category_str",
            ]
        ]
        .drop_duplicates("article_id")
        .set_index("article_id")[
            "category_str"
        ]
        .to_dict()
    )

    assert actual == expected

def test_article_statistics_are_non_negative():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    assert (
        result[
            "article_impression_count_before"
        ] >= 0
    ).all()

    assert (
        result[
            "article_click_count_before"
        ] >= 0
    ).all()

    assert (
        result[
            "article_ctr_before"
        ] >= 0
    ).all()

    assert (
        result[
            "article_ctr_before"
        ] <= 1
    ).all()


def test_same_timestamp_candidates_do_not_update_state():
    interactions = prepare()

    result = build_article_features(
        interactions
    )

    first = result[
        result["impression_id"] == 1
    ]

    # Even though A is clicked and B is exposed in impression 1,
    # neither candidate can see the current impression as history.
    assert (
        first["article_impression_count_before"]
        == 0
    ).all()

    assert (
        first["article_click_count_before"]
        == 0
    ).all()

def test_future_publication_timestamp_is_flagged_and_age_is_clipped():
    interactions = prepare()

    interactions.loc[
        interactions["article_id"] == "A",
        "published_time",
    ] = pd.Timestamp(
        "2026-01-01 02:00:00"
    )

    result = build_article_features(
        interactions
    )

    row = result[
        (result["impression_id"] == 1)
        & (result["article_id"] == "A")
    ].iloc[0]

    assert row[
        "article_age_hours"
    ] == pytest.approx(0.0)

    assert row[
        "article_age_days"
    ] == pytest.approx(0.0)

    assert row[
        "article_published_after_impression"
    ] == 1