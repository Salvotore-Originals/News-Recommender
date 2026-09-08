"""
PHASE 4G — Optimized BM25 Integration Tests

Verifies that the Phase 4F optimized BM25 retrieval can replace
the Phase 4E temporal BM25 retrieval without changing candidate
IDs or scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retrieval.ebnerd_bm25 import (
    build_bm25,
    build_history_lookup,
    build_query_from_history,
    load_articles,
    load_history,
)
from src.retrieval.ebnerd_bm25_optimized import (
    build_optimized_bm25_resource,
    retrieve_bm25_temporal_optimized,
)


def reference_retrieve_bm25_temporal(
    bm25,
    bm25_published_times,
    bm25_index_to_article_id,
    query_tokens,
    impression_time,
    top_k,
):
    """
    Exact Phase 4E BM25 temporal retrieval behavior.
    """

    if not query_tokens:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float64,
    )

    eligible_mask = (
        ~np.isnat(bm25_published_times)
        & (
            bm25_published_times
            <= np.datetime64(impression_time)
        )
    )

    eligible_indices = np.flatnonzero(
        eligible_mask
    )

    if len(eligible_indices) == 0:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    eligible_scores = scores[
        eligible_indices
    ]

    k = min(
        top_k,
        len(eligible_scores),
    )

    if k < len(eligible_scores):
        selected_local = np.argpartition(
            -eligible_scores,
            k - 1,
        )[:k]
    else:
        selected_local = np.arange(
            len(eligible_scores)
        )

    selected_local = sorted(
        selected_local.tolist(),
        key=lambda j: (
            -float(eligible_scores[j]),
            bm25_index_to_article_id[
                int(eligible_indices[j])
            ],
        ),
    )[:top_k]

    selected_indices = eligible_indices[
        selected_local
    ]

    return pd.DataFrame(
        {
            "article_id": [
                bm25_index_to_article_id[
                    int(index)
                ]
                for index in selected_indices
            ],
            "score": [
                float(scores[index])
                for index in selected_indices
            ],
        }
    )


@pytest.fixture(scope="module")
def resources():
    articles = load_articles()
    history = load_history("validation")

    bm25, article_id_to_index = build_bm25(
        articles
    )

    optimized_resource = (
        build_optimized_bm25_resource(
            bm25,
            article_id_to_index,
            articles,
        )
    )

    publication_times = pd.to_datetime(
        articles["published_time"],
        errors="coerce",
    )

    bm25_published_times = np.full(
        len(article_id_to_index),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    bm25_index_to_article_id = {}

    for article_id, corpus_index in (
        article_id_to_index.items()
    ):
        article_id = str(article_id)
        corpus_index = int(corpus_index)

        bm25_index_to_article_id[
            corpus_index
        ] = article_id

        article_rows = (
            articles["article_id"].astype(str)
            == article_id
        )

        if article_rows.any():
            timestamp = publication_times[
                article_rows
            ].iloc[0]

            if pd.notna(timestamp):
                bm25_published_times[
                    corpus_index
                ] = np.datetime64(timestamp)

    history_lookup = build_history_lookup(
        history,
        articles,
    )

    return {
        "articles": articles,
        "bm25": bm25,
        "optimized": optimized_resource,
        "bm25_published_times": bm25_published_times,
        "bm25_index_to_article_id": (
            bm25_index_to_article_id
        ),
        "history_lookup": history_lookup,
    }


@pytest.mark.parametrize(
    "user_id,impression_time,top_k",
    [
        (
            1361500,
            "2023-05-25 07:00:02",
            50,
        ),
        (
            1361500,
            "2023-05-25 07:00:02",
            100,
        ),
        (
            1361500,
            "2023-05-25 07:00:02",
            500,
        ),
        (
            22548,
            "2023-05-28 04:21:24",
            50,
        ),
        (
            22548,
            "2023-05-28 04:21:24",
            100,
        ),
        (
            22548,
            "2023-05-28 04:21:24",
            500,
        ),
    ],
)
def test_phase4e_bm25_and_optimized_are_identical(
    resources,
    user_id,
    impression_time,
    top_k,
):
    history_events = resources[
        "history_lookup"
    ].get(user_id, [])

    impression_time = pd.Timestamp(
        impression_time
    )

    query_tokens = build_query_from_history(
        history_events,
        impression_time,
    )

    reference = (
        reference_retrieve_bm25_temporal(
            resources["bm25"],
            resources[
                "bm25_published_times"
            ],
            resources[
                "bm25_index_to_article_id"
            ],
            query_tokens,
            impression_time,
            top_k,
        )
    )

    optimized = (
        retrieve_bm25_temporal_optimized(
            resources["optimized"],
            query_tokens,
            impression_time,
            top_k,
        )
    )

    assert (
        reference["article_id"].tolist()
        == optimized["article_id"].tolist()
    )

    np.testing.assert_allclose(
        reference["score"].to_numpy(),
        optimized["score"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_optimized_bm25_respects_publication_cutoff(
    resources,
):
    impression_time = pd.Timestamp(
        "2023-05-28 04:21:24"
    )

    query_tokens = ["sports"]

    result = (
        retrieve_bm25_temporal_optimized(
            resources["optimized"],
            query_tokens,
            impression_time,
            top_k=500,
        )
    )

    articles = resources["articles"].copy()

    articles["article_id"] = (
        articles["article_id"].astype(str)
    )

    metadata = result.merge(
        articles[
            [
                "article_id",
                "published_time",
            ]
        ],
        on="article_id",
        how="left",
    )

    published = pd.to_datetime(
        metadata["published_time"],
        errors="coerce",
    )

    assert (
        published <= impression_time
    ).all()


def test_optimized_bm25_returns_string_ids(
    resources,
):
    result = (
        retrieve_bm25_temporal_optimized(
            resources["optimized"],
            ["sports"],
            pd.Timestamp(
                "2023-05-28 04:21:24"
            ),
            top_k=50,
        )
    )

    assert all(
        isinstance(article_id, str)
        for article_id
        in result["article_id"]
    )