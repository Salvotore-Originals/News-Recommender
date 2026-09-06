import pandas as pd

from src.data.ebnerd_session_features import (
    build_session_features,
)


def make_interactions(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "user_id",
            "article_id",
            "impression_time",
            "clicked",
            "session_id",
            "category",
            "category_str",
        ],
    )


def test_previous_session_impression_is_included():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                200,
                "2023-01-01 10:05:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert current[
        "session_event_count_before"
    ] == 1

    assert current[
        "session_click_count_before"
    ] == 1


def test_current_impression_candidates_do_not_leak():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                200,
                "2023-01-01 10:05:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                300,
                "2023-01-01 10:05:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ]

    assert (
        current["session_event_count_before"]
        == 1
    ).all()

    assert (
        current["session_click_count_before"]
        == 1
    ).all()


def test_same_timestamp_does_not_leak():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                200,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert current[
        "session_event_count_before"
    ] == 0

    assert current[
        "session_click_count_before"
    ] == 0


def test_session_category_preference():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                101,
                "2023-01-01 10:01:00",
                0,
                50,
                1,
                "Sports",
            ),
            (
                3,
                10,
                102,
                "2023-01-01 10:02:00",
                0,
                50,
                2,
                "Politics",
            ),
            (
                4,
                10,
                103,
                "2023-01-01 10:03:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 4
    ].iloc[0]

    assert current[
        "session_event_count_before"
    ] == 3

    assert current[
        "session_category_count_before"
    ] == 2

    assert abs(
        current[
            "session_category_preference"
        ]
        - (2 / 3)
    ) < 1e-9


def test_candidate_seen_before():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                100,
                "2023-01-01 10:05:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert current[
        "session_candidate_seen_before"
    ] == 1


def test_session_duration():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                101,
                "2023-01-01 10:05:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert (
        current[
            "session_duration_seconds_before"
        ]
        == 300
    )


def test_previous_event_gap():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                101,
                "2023-01-01 10:05:00",
                0,
                50,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert (
        current[
            "seconds_since_previous_session_event"
        ]
        == 300
    )


def test_different_session_does_not_share_state():

    interactions = make_interactions(
        [
            (
                1,
                10,
                100,
                "2023-01-01 10:00:00",
                1,
                50,
                1,
                "Sports",
            ),
            (
                2,
                10,
                200,
                "2023-01-01 10:05:00",
                0,
                51,
                1,
                "Sports",
            ),
        ]
    )

    result = build_session_features(
        interactions
    )

    current = result[
        result["impression_id"] == 2
    ].iloc[0]

    assert current[
        "session_event_count_before"
    ] == 0

    assert current[
        "session_click_count_before"
    ] == 0

    assert current[
        "session_candidate_seen_before"
    ] == 0