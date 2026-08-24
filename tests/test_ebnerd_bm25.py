from pathlib import Path

import pandas as pd

from src.retrieval.ebnerd_bm25 import (
    tokenize,
    build_query_from_history,
    load_articles,
    build_bm25,
    score_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)


def test_tokenizer_is_deterministic():

    text = "Premier League: Arsenal WON 3-1!"

    tokens = tokenize(text)

    assert tokens == [
        "premier",
        "league",
        "arsenal",
        "won",
        "3",
        "1",
    ]


def test_bm25_article_count():

    articles = load_articles()

    assert len(articles) == 20_738


def test_bm25_article_ids_are_unique():

    articles = load_articles()

    assert articles["article_id"].is_unique


def test_history_query_respects_time():

    history = [
        (
            pd.Timestamp("2023-05-20 10:00:00"),
            1,
            "Sports News",
        ),
        (
            pd.Timestamp("2023-05-21 10:00:00"),
            2,
            "Technology News",
        ),
        (
            pd.Timestamp("2023-05-23 10:00:00"),
            3,
            "Future Article",
        ),
    ]

    query = build_query_from_history(
        history,
        pd.Timestamp(
            "2023-05-22 00:00:00"
        ),
    )

    assert "sports" in query
    assert "technology" in query
    assert "future" not in query


def test_history_query_deduplicates_articles():

    history = [
        (
            pd.Timestamp("2023-05-20"),
            1,
            "Sports News",
        ),
        (
            pd.Timestamp("2023-05-21"),
            1,
            "Sports News",
        ),
    ]

    query = build_query_from_history(
        history,
        pd.Timestamp("2023-05-22"),
    )

    assert query.count("sports") == 1


def test_score_candidates_returns_candidates():

    articles = load_articles()

    bm25, article_to_index = build_bm25(
        articles
    )

    candidates = articles[
        "article_id"
    ].head(5).tolist()

    results = score_candidates(
        bm25,
        article_to_index,
        ["news"],
        candidates,
    )

    result_ids = {
        article_id
        for article_id, score
        in results
    }

    assert result_ids == set(
        candidates
    )


def test_scores_are_numeric():

    articles = load_articles()

    bm25, article_to_index = build_bm25(
        articles
    )

    candidates = articles[
        "article_id"
    ].head(5).tolist()

    results = score_candidates(
        bm25,
        article_to_index,
        ["news"],
        candidates,
    )

    assert all(
        isinstance(score, float)
        for article_id, score
        in results
    )

def test_restricted_bm25_matches_full_corpus_scores():

    from src.retrieval.ebnerd_bm25 import (
        score_candidates,
        score_candidates_restricted,
        build_candidate_bm25_statistics,
    )

    articles = load_articles()

    bm25, article_to_index = build_bm25(
        articles
    )

    statistics = (
        build_candidate_bm25_statistics(
            articles
        )
    )

    query = [
        "news",
        "sports",
        "technology",
    ]

    candidates = articles[
        "article_id"
    ].head(10).tolist()

    old_scores = dict(
        score_candidates(
            bm25,
            article_to_index,
            query,
            candidates,
        )
    )

    new_scores = dict(
        score_candidates_restricted(
            query,
            candidates,
            article_to_index,
            statistics,
        )
    )

    assert set(old_scores) == set(
        new_scores
    )

    for article_id in old_scores:

        assert abs(
            old_scores[article_id]
            - new_scores[article_id]
        ) < 1e-6

def test_restricted_bm25_matches_full_bm25_on_real_impressions():

    from src.retrieval.ebnerd_bm25 import (
        build_bm25,
        build_candidate_bm25_statistics,
        build_history_lookup,
        build_query_from_history,
        load_articles,
        load_history,
        load_interactions,
        score_candidates,
        score_candidates_restricted,
    )

    articles = load_articles()

    bm25, article_to_index = build_bm25(
        articles
    )

    statistics = (
        build_candidate_bm25_statistics(
            articles
        )
    )

    interactions = load_interactions(
        "train"
    )

    history = load_history(
        "train"
    )

    history_lookup = build_history_lookup(
        history,
        articles,
    )

    # Use the first 100 real impressions.
    groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    checked = 0

    for impression_id, group in groups:

        if checked >= 100:
            break

        group = group.sort_values(
            "article_id"
        )

        user_id = group.iloc[0]["user_id"]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidates = (
            group["article_id"].tolist()
        )

        user_history = history_lookup.get(
            user_id,
            [],
        )

        query_tokens = (
            build_query_from_history(
                user_history,
                impression_time,
            )
        )

        full_results = score_candidates(
            bm25,
            article_to_index,
            query_tokens,
            candidates,
        )

        restricted_results = (
            score_candidates_restricted(
                query_tokens,
                candidates,
                article_to_index,
                statistics,
            )
        )

        assert len(full_results) == len(
            restricted_results
        )

        for (
            full,
            restricted,
        ) in zip(
            full_results,
            restricted_results,
        ):

            assert full[0] == restricted[0]

            assert abs(
                full[1] - restricted[1]
            ) < 1e-6

        checked += 1

    assert checked == 100