import pandas as pd
import pytest

from src.data.ebnerd_candidate_features import (
    build_candidate_features,
)


def make_temporal():
    return pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "user_id": [10, 10, 10, 10],
            "article_id": ["A", "B", "A", "C"],
            "impression_time": pd.to_datetime(
                [
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 02:00:00",
                ]
            ),
            "clicked": [1, 0, 0, 1],
            "session_id": [100, 100, 100, 100],
            "category": [1, 2, 1, 1],
            "category_str": [
                "Sports",
                "Politics",
                "Sports",
                "Sports",
            ],
            "category_count_before": [0, 0, 1, 1],
            "history_count_before": [0, 0, 2, 2],
            "category_preference": [
                0.0,
                0.0,
                0.5,
                0.5,
            ],
            "mean_read_time_before": [
                0.0,
                0.0,
                10.0,
                10.0,
            ],
            "mean_scroll_percentage_before": [
                0.0,
                0.0,
                50.0,
                50.0,
            ],
            "category_mean_read_time_before": [
                0.0,
                0.0,
                10.0,
                10.0,
            ],
            "category_mean_scroll_percentage_before": [
                0.0,
                0.0,
                50.0,
                50.0,
            ],
            "candidate_seen_before": [0, 0, 1, 0],
            "candidate_recency_hours": [
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        }
    )


def make_session():
    return pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "user_id": [10, 10, 10, 10],
            "article_id": ["A", "B", "A", "C"],
            "impression_time": pd.to_datetime(
                [
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 02:00:00",
                ]
            ),
            "clicked": [1, 0, 0, 1],
            "session_id": [100, 100, 100, 100],
            "category": [1, 2, 1, 1],
            "category_str": [
                "Sports",
                "Politics",
                "Sports",
                "Sports",
            ],
            "session_event_count_before": [0, 0, 1, 1],
            "session_click_count_before": [0, 0, 1, 1],
            "session_category_count_before": [
                0,
                0,
                1,
                1,
            ],
            "session_category_preference": [
                0.0,
                0.0,
                0.5,
                0.5,
            ],
            "session_candidate_seen_before": [
                0,
                0,
                1,
                0,
            ],
            "seconds_since_previous_session_event": [
                0.0,
                0.0,
                3600.0,
                3600.0,
            ],
            "session_duration_seconds_before": [
                0.0,
                0.0,
                3600.0,
                3600.0,
            ],
        }
    )


def make_article():
    return pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "user_id": [10, 10, 10, 10],
            "article_id": ["A", "B", "A", "C"],
            "impression_time": pd.to_datetime(
                [
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 02:00:00",
                ]
            ),
            "clicked": [1, 0, 0, 1],
            "session_id": [100, 100, 100, 100],
            "category": [1, 2, 1, 1],
            "category_str": [
                "Sports",
                "Politics",
                "Sports",
                "Sports",
            ],
            "published_time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:30:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:30:00",
                ]
            ),
            "article_age_hours": [
                1.0,
                0.5,
                2.0,
                0.5,
            ],
            "article_age_days": [
                1 / 24,
                0.5 / 24,
                2 / 24,
                0.5 / 24,
            ],
            "article_published_after_impression": [
                0,
                0,
                0,
                0,
            ],
            "article_impression_count_before": [
                0,
                0,
                1,
                0,
            ],
            "article_click_count_before": [
                0,
                0,
                1,
                0,
            ],
            "article_ctr_before": [
                0.5,
                0.5,
                2 / 3,
                0.5,
            ],
        }
    )


def test_candidate_features_merge_without_row_multiplication():
    result = build_candidate_features(
        make_temporal(),
        make_session(),
        make_article(),
    )

    assert len(result) == 4


def test_candidate_identity_is_preserved():
    result = build_candidate_features(
        make_temporal(),
        make_session(),
        make_article(),
    )

    expected = {
        (1, "A"),
        (1, "B"),
        (2, "A"),
        (2, "C"),
    }

    actual = set(
        zip(
            result["impression_id"],
            result["article_id"],
        )
    )

    assert actual == expected


def test_all_feature_families_are_present():
    result = build_candidate_features(
        make_temporal(),
        make_session(),
        make_article(),
    )

    expected_columns = {
        "category_preference",
        "candidate_recency_hours",
        "session_event_count_before",
        "session_click_count_before",
        "session_category_preference",
        "session_candidate_seen_before",
        "article_age_hours",
        "article_age_days",
        "article_ctr_before",
        "article_impression_count_before",
        "article_click_count_before",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_clicked_is_preserved_as_target():
    result = build_candidate_features(
        make_temporal(),
        make_session(),
        make_article(),
    )

    assert result["clicked"].tolist() == [
        1,
        0,
        0,
        1,
    ]


def test_missing_session_candidate_is_rejected():
    session = make_session().iloc[:-1].copy()

    with pytest.raises(ValueError):
        build_candidate_features(
            make_temporal(),
            session,
            make_article(),
        )


def test_duplicate_candidate_is_rejected():
    temporal = pd.concat(
        [
            make_temporal(),
            make_temporal().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        build_candidate_features(
            temporal,
            make_session(),
            make_article(),
        )


def test_identity_mismatch_is_rejected():
    session = make_session().copy()

    session.loc[
        session["article_id"] == "A",
        "user_id",
    ] = 999

    with pytest.raises(ValueError):
        build_candidate_features(
            make_temporal(),
            session,
            make_article(),
        )


def test_missing_feature_values_are_rejected():
    article = make_article().copy()

    article.loc[
        article["article_id"] == "A",
        "article_ctr_before",
    ] = float("nan")

    with pytest.raises(ValueError):
        build_candidate_features(
            make_temporal(),
            make_session(),
            article,
        )