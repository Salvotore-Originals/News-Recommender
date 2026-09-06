"""
A2 temporal leakage regression tests.

These tests enforce the point-in-time behavioural feature contract:

    history_event_time < impression_time

Future and same-timestamp history events must never contribute
to behavioural features for the current impression.

MIND and EB-NeRD use different history representations:

- MIND:
    History is stored as an impression-specific snapshot.

- EB-NeRD:
    History contains actual event timestamps, so the strict
    timestamp boundary can be tested directly.
"""

import pandas as pd

from src.data.ebnerd_temporal_features import (
    build_temporal_features,
)


# =====================================================================
# EB-NeRD TEMPORAL LEAKAGE TESTS
# =====================================================================


def test_ebnerd_history_must_be_strictly_before_impression():
    """
    History events at or after the impression timestamp must not
    contribute to the behavioural features.

    History:

        09:00 -> A -> sports
        10:00 -> B -> technology
        11:00 -> C -> politics

    Impression:

        10:00

    Expected:

        A is included.
        B is excluded because it has the same timestamp.
        C is excluded because it is in the future.
    """

    interactions = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "X",
                "impression_time": "2026-01-01 10:00:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S1",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "A",
                "timestamp": "2026-01-01 09:00:00",
                "category_str": "sports",
            },
            {
                "user_id": "U1",
                "article_id": "B",
                "timestamp": "2026-01-01 10:00:00",
                "category_str": "technology",
            },
            {
                "user_id": "U1",
                "article_id": "C",
                "timestamp": "2026-01-01 11:00:00",
                "category_str": "politics",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    assert len(result) == 1

    row = result.iloc[0]

    # Only A occurred strictly before the impression.
    assert row["history_count_before"] == 1

    # A belongs to the candidate's category.
    assert row["category_count_before"] == 1

    assert row["category_preference"] == 1.0


def test_ebnerd_future_events_do_not_change_features():
    """
    Adding future history events must not change features for an
    earlier impression.
    """

    interactions = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "X",
                "impression_time": "2026-01-01 10:00:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S1",
            }
        ]
    )

    history_without_future = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "A",
                "timestamp": "2026-01-01 09:00:00",
                "category_str": "sports",
            }
        ]
    )

    history_with_future = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "A",
                "timestamp": "2026-01-01 09:00:00",
                "category_str": "sports",
            },
            {
                "user_id": "U1",
                "article_id": "B",
                "timestamp": "2026-01-01 12:00:00",
                "category_str": "sports",
            },
        ]
    )

    result_without_future = build_temporal_features(
        interactions,
        history_without_future,
    )

    result_with_future = build_temporal_features(
        interactions,
        history_with_future,
    )

    first = result_without_future.iloc[0]
    second = result_with_future.iloc[0]

    assert first["history_count_before"] == 1
    assert second["history_count_before"] == 1

    assert first["category_count_before"] == 1
    assert second["category_count_before"] == 1

    assert first["category_preference"] == 1.0
    assert second["category_preference"] == 1.0


def test_ebnerd_history_grows_only_as_time_advances():
    """
    Verify that the available history grows correctly as time
    advances.

    History:

        09:00 -> A -> sports
        10:00 -> B -> technology

    Impressions:

        09:30
        10:30

    Expected:

        09:30 -> A only
        10:30 -> A + B
    """

    interactions = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "X",
                "impression_time": "2026-01-01 09:30:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S1",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "Y",
                "impression_time": "2026-01-01 10:30:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S2",
            },
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "A",
                "timestamp": "2026-01-01 09:00:00",
                "category_str": "sports",
            },
            {
                "user_id": "U1",
                "article_id": "B",
                "timestamp": "2026-01-01 10:00:00",
                "category_str": "technology",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    result = result.sort_values(
        "impression_time"
    ).reset_index(drop=True)

    first = result.iloc[0]
    second = result.iloc[1]

    # At 09:30, only A is available.
    assert first["history_count_before"] == 1

    # At 10:30, A and B are available.
    assert second["history_count_before"] == 2


def test_ebnerd_same_timestamp_event_is_not_available():
    """
    Explicitly verify the strict '<' boundary.

    A history event occurring at exactly the same timestamp as
    the impression must not be included.
    """

    interactions = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "X",
                "impression_time": "2026-01-01 10:00:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S1",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "A",
                "timestamp": "2026-01-01 10:00:00",
                "category_str": "sports",
            }
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["history_count_before"] == 0
    assert row["category_count_before"] == 0
    assert row["category_preference"] == 0.0


# =====================================================================
# EB-NeRD CANDIDATE SEEN / RECENCY LEAKAGE TEST
# =====================================================================


def test_ebnerd_future_candidate_occurrence_is_not_seen():
    """
    A candidate article appearing only in the future must not be
    marked as previously seen.

    Candidate X:

        impression -> 10:00

    History:

        X -> 11:00

    Expected:

        candidate_seen_before = 0
        candidate_recency_hours = inf
    """

    interactions = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "X",
                "impression_time": "2026-01-01 10:00:00",
                "category": "Sports",
                "category_str": "sports",
                "clicked": 0,
                "session_id": "S1",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "article_id": "X",
                "timestamp": "2026-01-01 11:00:00",
                "category_str": "sports",
            }
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["candidate_seen_before"] == 0
    assert row["candidate_recency_hours"] == float("inf")


# =====================================================================
# MIND SNAPSHOT ISOLATION TESTS
# =====================================================================


def test_mind_impression_history_isolation():
    """
    MIND history is stored as an impression-specific snapshot.

    Earlier impression:

        I1 -> A, B

    Later impression:

        I2 -> A, B, C

    C must not appear in I1's history.
    """

    history = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "A",
            },
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "B",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "A",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "B",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "C",
            },
        ]
    )

    earlier_history = history.loc[
        history["impression_id"] == "I1"
    ]

    later_history = history.loc[
        history["impression_id"] == "I2"
    ]

    assert set(
        earlier_history["article_id"]
    ) == {"A", "B"}

    assert set(
        later_history["article_id"]
    ) == {"A", "B", "C"}

    assert "C" not in set(
        earlier_history["article_id"]
    )


def test_mind_history_is_keyed_by_impression():
    """
    Verify that the same user can have different history snapshots
    for different impressions.
    """

    history = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "article_id": "A",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "A",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "article_id": "B",
            },
        ]
    )

    history_sizes = (
        history
        .groupby(
            ["impression_id", "user_id"]
        )
        .size()
    )

    assert history_sizes.loc[
        ("I1", "U1")
    ] == 1

    assert history_sizes.loc[
        ("I2", "U1")
    ] == 2

import pandas as pd

from src.data.ebnerd_temporal_features import (
    build_temporal_features,
)


def test_ebnerd_future_read_time_does_not_leak():
    """Future read_time must not affect an earlier impression."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 09:00:00",
                "read_time": 20.0,
                "scroll_percentage": 50.0,
                "category_str": "sports",
            },
            {
                "user_id": 100,
                "article_id": 30,
                "timestamp": "2023-01-01 11:00:00",
                "read_time": 1000.0,
                "scroll_percentage": 100.0,
                "category_str": "sports",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_read_time_before"] == 20.0
    assert row["category_mean_read_time_before"] == 20.0


def test_ebnerd_future_scroll_does_not_leak():
    """Future scroll percentage must not affect an earlier impression."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 09:00:00",
                "read_time": 20.0,
                "scroll_percentage": 40.0,
                "category_str": "sports",
            },
            {
                "user_id": 100,
                "article_id": 30,
                "timestamp": "2023-01-01 11:00:00",
                "read_time": 1000.0,
                "scroll_percentage": 100.0,
                "category_str": "sports",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_scroll_percentage_before"] == 40.0
    assert row["category_mean_scroll_percentage_before"] == 40.0


def test_ebnerd_same_timestamp_read_time_is_excluded():
    """A history event at exactly impression time must be excluded."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 10:00:00",
                "read_time": 500.0,
                "scroll_percentage": 90.0,
                "category_str": "sports",
            }
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_read_time_before"] == 0.0
    assert row["category_mean_read_time_before"] == 0.0


def test_ebnerd_same_timestamp_scroll_is_excluded():
    """A same-timestamp scroll event must not leak."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 10:00:00",
                "read_time": 500.0,
                "scroll_percentage": 90.0,
                "category_str": "sports",
            }
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_scroll_percentage_before"] == 0.0
    assert row["category_mean_scroll_percentage_before"] == 0.0


def test_ebnerd_earlier_engagement_is_included():
    """Historical engagement before the impression must be included."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 08:00:00",
                "read_time": 20.0,
                "scroll_percentage": 40.0,
                "category_str": "sports",
            },
            {
                "user_id": 100,
                "article_id": 30,
                "timestamp": "2023-01-01 09:00:00",
                "read_time": 40.0,
                "scroll_percentage": 60.0,
                "category_str": "sports",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_read_time_before"] == 30.0
    assert row["mean_scroll_percentage_before"] == 50.0
    assert row["category_mean_read_time_before"] == 30.0
    assert row["category_mean_scroll_percentage_before"] == 50.0


def test_ebnerd_category_engagement_respects_time_boundary():
    """Category engagement must use only prior events in that category."""

    interactions = pd.DataFrame(
        [
            {
                "impression_id": 1,
                "user_id": 100,
                "article_id": 10,
                "impression_time": "2023-01-01 10:00:00",
                "clicked": 0,
                "session_id": 1,
                "category": 1,
                "category_str": "sports",
            }
        ]
    )

    history = pd.DataFrame(
        [
            {
                "user_id": 100,
                "article_id": 20,
                "timestamp": "2023-01-01 08:00:00",
                "read_time": 20.0,
                "scroll_percentage": 40.0,
                "category_str": "sports",
            },
            {
                "user_id": 100,
                "article_id": 30,
                "timestamp": "2023-01-01 09:00:00",
                "read_time": 40.0,
                "scroll_percentage": 60.0,
                "category_str": "news",
            },
            {
                "user_id": 100,
                "article_id": 40,
                "timestamp": "2023-01-01 11:00:00",
                "read_time": 1000.0,
                "scroll_percentage": 100.0,
                "category_str": "sports",
            },
        ]
    )

    result = build_temporal_features(
        interactions,
        history,
    )

    row = result.iloc[0]

    assert row["mean_read_time_before"] == 30.0
    assert row["mean_scroll_percentage_before"] == 50.0

    assert row["category_mean_read_time_before"] == 20.0
    assert row["category_mean_scroll_percentage_before"] == 40.0