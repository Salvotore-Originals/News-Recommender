"""
Tests for Phase 4D: Point-in-Time Fresh + Category Candidate Retrieval.

These tests target the actual public API implemented in
src/retrieval/ebnerd_fresh_candidates.py.

They intentionally do not modify or exercise the existing BM25, semantic,
or RRF implementations.
"""

import pandas as pd
import pytest

from src.retrieval.ebnerd_fresh_candidates import (
    FreshCategoryCandidateGenerator,
    compute_point_in_time_category_preferences,
    freshness_score,
    prepare_articles,
    prepare_history,
)


IMPRESSION_TIME = pd.Timestamp("2023-05-28 04:21:24")


@pytest.fixture
def articles():
    """Article corpus plus articles referenced by test user history."""
    return pd.DataFrame(
        {
            "article_id": [
                "h1", "h2", "h3", "h4",
                "a1", "a2", "a3", "a4", "a5", "a6",
            ],
            "category_str": [
                "sport", "sport", "musik", "nyheder",
                "sport", "sport", "musik", "sport", "nyheder", "sport",
            ],
            "published_time": [
                pd.Timestamp("2023-05-20 09:00:00"),
                pd.Timestamp("2023-05-21 09:00:00"),
                pd.Timestamp("2023-05-22 09:00:00"),
                pd.Timestamp("2023-05-23 09:00:00"),
                IMPRESSION_TIME - pd.Timedelta(hours=2),   # fresh
                IMPRESSION_TIME - pd.Timedelta(hours=20),  # fresh
                IMPRESSION_TIME - pd.Timedelta(hours=30),  # fresh
                IMPRESSION_TIME - pd.Timedelta(hours=60),  # outside 48h
                IMPRESSION_TIME,                            # exact time
                IMPRESSION_TIME + pd.Timedelta(minutes=1),  # future
            ],
        }
    )


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 1],
            "article_id": ["h1", "h2", "h3", "h4", "h1", "h2"],
            "timestamp": [
                pd.Timestamp("2023-05-20 10:00:00"),
                pd.Timestamp("2023-05-21 10:00:00"),
                pd.Timestamp("2023-05-22 10:00:00"),
                pd.Timestamp("2023-05-23 10:00:00"),
                IMPRESSION_TIME,                            # same timestamp
                pd.Timestamp("2023-05-29 10:00:00"),        # future
            ],
        }
    )


def test_prepare_articles_requires_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_articles(
            pd.DataFrame(
                {
                    "article_id": ["a1"],
                    "category_str": ["sport"],
                }
            )
        )


def test_prepare_history_requires_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_history(
            pd.DataFrame(
                {
                    "user_id": [1],
                    "article_id": ["h1"],
                }
            )
        )


def test_point_in_time_preferences_use_strict_timestamp_cutoff():
    events = [
        (pd.Timestamp("2023-05-20 10:00:00"), "h1", "sport"),
        (pd.Timestamp("2023-05-21 10:00:00"), "h2", "sport"),
        (pd.Timestamp("2023-05-22 10:00:00"), "h3", "musik"),
        (IMPRESSION_TIME, "h4", "nyheder"),
        (pd.Timestamp("2023-05-29 10:00:00"), "h5", "sport"),
    ]

    preferences = compute_point_in_time_category_preferences(
        events,
        IMPRESSION_TIME,
    )

    assert preferences["sport"] == pytest.approx(2 / 3)
    assert preferences["musik"] == pytest.approx(1 / 3)
    assert "nyheder" not in preferences


def test_point_in_time_preferences_return_empty_when_no_eligible_history():
    events = [
        (IMPRESSION_TIME, "h1", "sport"),
        (IMPRESSION_TIME + pd.Timedelta(hours=1), "h2", "sport"),
    ]

    assert (
        compute_point_in_time_category_preferences(
            events,
            IMPRESSION_TIME,
        )
        == {}
    )


def test_freshness_score():
    assert freshness_score(0.0, decay_hours=24.0) == pytest.approx(1.0)
    assert freshness_score(24.0, decay_hours=24.0) == pytest.approx(
        0.36787944117
    )


def test_freshness_score_rejects_invalid_values():
    with pytest.raises(ValueError):
        freshness_score(-1.0)

    with pytest.raises(ValueError):
        freshness_score(1.0, decay_hours=0.0)


def test_generator_excludes_future_and_stale_articles(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
        decay_hours=24.0,
        top_categories=3,
        per_category=100,
    )

    result = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=100,
    )

    ids = set(result["article_id"])

    assert "a6" not in ids
    assert "a4" not in ids


def test_exact_publication_time_is_eligible(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
    )

    result = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=100,
    )

    assert "a5" in set(result["article_id"])


def test_generator_uses_point_in_time_preferences(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
    )

    preferences = generator.get_point_in_time_preferences(
        user_id=1,
        impression_time=IMPRESSION_TIME,
    )

    # h1/h2 = sport, h3 = musik, h4 = nyheder.
    assert preferences["sport"] == pytest.approx(0.5)
    assert preferences["musik"] == pytest.approx(0.25)
    assert preferences["nyheder"] == pytest.approx(0.25)


def test_same_timestamp_and_future_history_are_excluded(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
    )

    preferences = generator.get_point_in_time_preferences(
        user_id=1,
        impression_time=IMPRESSION_TIME,
    )

    # The same-time h1 and future h2 must not affect the profile.
    assert preferences["sport"] == pytest.approx(0.5)


def test_empty_history_returns_no_candidates(articles):
    empty_history = pd.DataFrame(
        columns=["user_id", "article_id", "timestamp"]
    )

    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=empty_history,
    )

    result = generator.generate_candidates(
        user_id=999,
        impression_time=IMPRESSION_TIME,
        top_k=10,
    )

    assert result.empty


def test_unknown_user_returns_no_candidates(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
    )

    result = generator.generate_candidates(
        user_id=999,
        impression_time=IMPRESSION_TIME,
        top_k=10,
    )

    assert result.empty


def test_top_k_validation(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
    )

    with pytest.raises(ValueError):
        generator.generate_candidates(
            user_id=1,
            impression_time=IMPRESSION_TIME,
            top_k=0,
        )

    with pytest.raises(ValueError):
        generator.generate_candidates(
            user_id=1,
            impression_time=IMPRESSION_TIME,
            top_k=-1,
        )


def test_category_and_freshness_score_are_present(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
        decay_hours=24.0,
    )

    result = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=100,
    )

    expected_columns = {
        "article_id",
        "category_str",
        "published_time",
        "category_preference",
        "freshness",
        "score",
    }

    assert expected_columns.issubset(result.columns)
    assert (result["category_preference"] >= 0).all()
    assert (result["freshness"] > 0).all()
    assert (result["freshness"] <= 1).all()
    assert (result["score"] >= 0).all()


def test_results_are_deterministic(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
    )

    first = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=100,
    )
    second = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=100,
    )

    pd.testing.assert_frame_equal(first, second)


def test_retrieve_alias_matches_generate_candidates(articles, history):
    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
    )

    generated = generator.generate_candidates(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=10,
    )
    retrieved = generator.retrieve(
        user_id=1,
        impression_time=IMPRESSION_TIME,
        top_k=10,
    )

    pd.testing.assert_frame_equal(generated, retrieved)


def test_96791_sanity_case():
    """
    Minimal reproduction of the forensic impression 96791.

    The clicked article 9784696 is a newly published sport article.
    The user's pre-impression history contains sport interactions, so the
    fresh/category branch should be able to discover it from the corpus.
    """
    articles = pd.DataFrame(
        {
            "article_id": [
                "h1", "h2", "h3", "h4",
                "9784696", "other-sport", "music",
            ],
            "category_str": [
                "sport", "sport", "sport", "nyheder",
                "sport", "sport", "musik",
            ],
            "published_time": [
                pd.Timestamp("2023-05-20 12:25:34"),
                pd.Timestamp("2023-05-21 09:48:00"),
                pd.Timestamp("2023-05-21 15:11:36"),
                pd.Timestamp("2023-05-25 04:45:13"),
                pd.Timestamp("2023-05-27 20:11:36"),
                pd.Timestamp("2023-05-27 18:00:00"),
                pd.Timestamp("2023-05-27 19:00:00"),
            ],
        }
    )

    history = pd.DataFrame(
        {
            "user_id": [22548] * 4,
            "article_id": ["h1", "h2", "h3", "h4"],
            "timestamp": [
                pd.Timestamp("2023-05-20 12:25:34"),
                pd.Timestamp("2023-05-21 09:48:00"),
                pd.Timestamp("2023-05-21 15:11:36"),
                pd.Timestamp("2023-05-25 04:45:13"),
            ],
        }
    )

    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
        decay_hours=24.0,
        top_categories=3,
        per_category=100,
    )

    result = generator.generate_candidates(
        user_id=22548,
        impression_time=IMPRESSION_TIME,
        top_k=50,
    )

    assert "9784696" in set(result["article_id"])
    assert (result["published_time"] <= IMPRESSION_TIME).all()
    assert (
        result["published_time"]
        >= IMPRESSION_TIME - pd.Timedelta(hours=48)
    ).all()
