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