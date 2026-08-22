import numpy as np
import pandas as pd

from src.retrieval.ebnerd_behavioral import (
    build_article_category_lookup,
    build_temporal_user_category_lookup,
    get_behavioral_preference,
    score_candidates,
)


def test_article_category_lookup():

    articles = pd.DataFrame(
        {
            "article_id": [1, 2, 3],
            "category_str": [
                "sports",
                "technology",
                "politics",
            ],
        }
    )

    lookup = (
        build_article_category_lookup(
            articles
        )
    )

    assert lookup == {
        1: "sports",
        2: "technology",
        3: "politics",
    }


def test_temporal_category_preference():

    articles = pd.DataFrame(
        {
            "article_id": [1, 2],
            "category_str": [
                "sports",
                "technology",
            ],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [10, 10],
            "article_id": [1, 2],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 11:00:00"
                ),
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    score = get_behavioral_preference(
        10,
        "sports",
        pd.Timestamp(
            "2023-05-20 12:00:00"
        ),
        lookup,
    )

    assert abs(
        score - 0.5
    ) < 1e-9


def test_future_behavior_is_excluded():

    articles = pd.DataFrame(
        {
            "article_id": [1, 2],
            "category_str": [
                "sports",
                "technology",
            ],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [10, 10],
            "article_id": [1, 2],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 12:00:00"
                ),
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    score = get_behavioral_preference(
        10,
        "technology",
        pd.Timestamp(
            "2023-05-20 11:00:00"
        ),
        lookup,
    )

    assert score == 0.0


def test_same_timestamp_is_excluded():

    articles = pd.DataFrame(
        {
            "article_id": [1],
            "category_str": ["sports"],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [10],
            "article_id": [1],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                )
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    score = get_behavioral_preference(
        10,
        "sports",
        pd.Timestamp(
            "2023-05-20 10:00:00"
        ),
        lookup,
    )

    assert score == 0.0


def test_unknown_user_returns_zero():

    lookup = {}

    score = get_behavioral_preference(
        999,
        "sports",
        pd.Timestamp(
            "2023-05-20"
        ),
        lookup,
    )

    assert score == 0.0


def test_unknown_category_returns_zero():

    lookup = {}

    score = get_behavioral_preference(
        10,
        None,
        pd.Timestamp(
            "2023-05-20"
        ),
        lookup,
    )

    assert score == 0.0


def test_candidate_ranking():

    articles = pd.DataFrame(
        {
            "article_id": [
                1,
                2,
                3,
            ],
            "category_str": [
                "sports",
                "technology",
                "politics",
            ],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [
                10,
                10,
                10,
                10,
            ],
            "article_id": [
                1,
                1,
                2,
                3,
            ],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 11:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 12:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 13:00:00"
                ),
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    ranked = score_candidates(
        10,
        pd.Timestamp(
            "2023-05-20 14:00:00"
        ),
        [1, 2, 3],
        categories,
        lookup,
    )

    assert ranked[0][0] == 1
    assert ranked[0][1] == 0.5


def test_history_without_known_article_category():

    articles = pd.DataFrame(
        {
            "article_id": [1],
            "category_str": ["sports"],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [10],
            "article_id": [999],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20"
                )
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    score = get_behavioral_preference(
        10,
        "sports",
        pd.Timestamp(
            "2023-05-21"
        ),
        lookup,
    )

    assert score == 0.0

def test_multiple_events_at_same_timestamp_are_excluded():

    articles = pd.DataFrame(
        {
            "article_id": [1, 2, 3],
            "category_str": [
                "sports",
                "technology",
                "politics",
            ],
        }
    )

    categories = (
        build_article_category_lookup(
            articles
        )
    )

    history = pd.DataFrame(
        {
            "user_id": [
                10,
                10,
                10,
            ],
            "article_id": [
                1,
                2,
                3,
            ],
            "timestamp": [
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 10:00:00"
                ),
                pd.Timestamp(
                    "2023-05-20 11:00:00"
                ),
            ],
        }
    )

    lookup = (
        build_temporal_user_category_lookup(
            history,
            categories,
        )
    )

    # At exactly 10:00, none of the 10:00
    # history events should be visible.
    score_at_same_time = (
        get_behavioral_preference(
            10,
            "sports",
            pd.Timestamp(
                "2023-05-20 10:00:00"
            ),
            lookup,
        )
    )

    assert score_at_same_time == 0.0

    # At 11:00, the two 10:00 events are
    # visible, but the 11:00 event is not.
    score_before_11 = (
        get_behavioral_preference(
            10,
            "sports",
            pd.Timestamp(
                "2023-05-20 11:00:00"
            ),
            lookup,
        )
    )

    assert abs(
        score_before_11 - 0.5
    ) < 1e-9