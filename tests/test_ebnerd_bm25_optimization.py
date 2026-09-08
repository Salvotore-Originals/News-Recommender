"""
Phase 4F: BM25 optimization equivalence tests.

The optimized BM25 temporal retrieval must produce the same ranked article
IDs as the existing Phase 4E implementation.

These tests verify:
    - identical candidate IDs
    - identical candidate ordering
    - identical BM25 scores
    - temporal eligibility
    - empty-query behavior
    - multiple top-K values

The existing BM25 implementation is treated as the reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retrieval.ebnerd_bm25 import (
    build_bm25,
    build_history_lookup,
    build_query_from_history,
)

from src.retrieval.ebnerd_bm25_optimized import (
    build_optimized_bm25_resource,
    retrieve_bm25_temporal_optimized,
)


ARTICLES_PATH = (
    "data/features/ebnerd/small/articles.parquet"
)

HISTORY_PATH = (
    "data/processed/ebnerd/small/"
    "validation_user_history.parquet"
)


def reference_retrieve_bm25_temporal(
    bm25,
    article_id_to_index: dict,
    articles: pd.DataFrame,
    query_tokens: list[str],
    impression_time: pd.Timestamp,
    top_k: int,
) -> pd.DataFrame:
    """
    Reference implementation copied from the validated Phase 4E evaluator.

    This function is deliberately kept local to the test so that the
    optimized implementation can be compared against a stable reference.
    """

    if not query_tokens:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float64,
    )

    publication_times = pd.to_datetime(
        articles["published_time"],
        errors="coerce",
    )

    eligible_indices = [
        corpus_index
        for article_id, corpus_index
        in article_id_to_index.items()
        if corpus_index < len(scores)
        and pd.notna(
            publication_times[
                articles["article_id"]
                == article_id
            ]
        ).any()
        and publication_times[
            articles["article_id"]
            == article_id
        ].iloc[0] <= impression_time
    ]

    if not eligible_indices:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    eligible_indices = np.asarray(
        eligible_indices,
        dtype=np.int64,
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

    index_to_article_id = {
        int(corpus_index): str(article_id)
        for article_id, corpus_index
        in article_id_to_index.items()
    }

    selected_local = sorted(
        selected_local.tolist(),
        key=lambda j: (
            -float(eligible_scores[j]),
            index_to_article_id[
                int(eligible_indices[j])
            ],
        ),
    )[:top_k]

    selected_indices = (
        eligible_indices[selected_local]
    )

    return pd.DataFrame(
        {
            "article_id": [
                index_to_article_id[
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
    """Load the small EB-NeRD resources once for the test module."""

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    history = pd.read_parquet(
        HISTORY_PATH
    )

    bm25, article_id_to_index = build_bm25(
        articles
    )

    optimized_resource = (
        build_optimized_bm25_resource(
            bm25=bm25,
            article_id_to_index=article_id_to_index,
            articles=articles,
        )
    )

    history_lookup = build_history_lookup(
        history,
        articles,
    )

    return {
        "articles": articles,
        "history": history,
        "bm25": bm25,
        "article_id_to_index": article_id_to_index,
        "optimized_resource": optimized_resource,
        "history_lookup": history_lookup,
    }


def get_query_for_impression(
    resources,
    user_id,
    impression_time,
):
    """Build the same history query used by Phase 4E."""

    events = resources[
        "history_lookup"
    ].get(
        user_id,
        [],
    )

    return build_query_from_history(
        events,
        pd.Timestamp(impression_time),
    )


@pytest.mark.parametrize(
    "impression_time, user_id, top_k",
    [
        (
            "2023-05-25 07:00:02",
            1361500,
            50,
        ),
        (
            "2023-05-25 07:00:02",
            1361500,
            100,
        ),
        (
            "2023-05-25 07:00:02",
            1361500,
            500,
        ),
        (
            "2023-05-28 04:21:24",
            22548,
            50,
        ),
        (
            "2023-05-28 04:21:24",
            22548,
            100,
        ),
        (
            "2023-05-28 04:21:24",
            22548,
            500,
        ),
    ],
)
def test_optimized_bm25_matches_reference(
    resources,
    impression_time,
    user_id,
    top_k,
):
    """
    Optimized retrieval must exactly match the reference retrieval.
    """

    impression_time = pd.Timestamp(
        impression_time
    )

    query_tokens = get_query_for_impression(
        resources,
        user_id,
        impression_time,
    )

    reference = (
        reference_retrieve_bm25_temporal(
            bm25=resources["bm25"],
            article_id_to_index=(
                resources[
                    "article_id_to_index"
                ]
            ),
            articles=resources["articles"],
            query_tokens=query_tokens,
            impression_time=impression_time,
            top_k=top_k,
        )
    )

    optimized = (
        retrieve_bm25_temporal_optimized(
            resource=resources[
                "optimized_resource"
            ],
            query_tokens=query_tokens,
            impression_time=impression_time,
            top_k=top_k,
        )
    )

    assert (
        reference["article_id"].tolist()
        == optimized["article_id"].tolist()
    ), (
        "Optimized BM25 returned a different "
        "candidate ranking."
    )

    np.testing.assert_allclose(
        reference["score"].to_numpy(),
        optimized["score"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_empty_query_returns_empty(
    resources,
):
    """Empty queries must produce empty candidate sets."""

    result = (
        retrieve_bm25_temporal_optimized(
            resource=resources[
                "optimized_resource"
            ],
            query_tokens=[],
            impression_time=pd.Timestamp(
                "2023-05-28 04:21:24"
            ),
            top_k=500,
        )
    )

    assert result.empty
    assert list(result.columns) == [
        "article_id",
        "score",
    ]


def test_future_articles_are_excluded(
    resources,
):
    """Articles published after the impression must not be returned."""

    impression_time = pd.Timestamp(
        "2023-05-28 04:21:24"
    )

    query_tokens = [
        "sport",
        "haaland",
    ]

    result = (
        retrieve_bm25_temporal_optimized(
            resource=resources[
                "optimized_resource"
            ],
            query_tokens=query_tokens,
            impression_time=impression_time,
            top_k=500,
        )
    )

    if result.empty:
        pytest.skip(
            "No eligible BM25 candidates "
            "returned for this query."
        )

    article_ids = result[
        "article_id"
    ].astype(str)

    articles = resources[
        "articles"
    ].copy()

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
    )

    articles["published_time"] = (
        pd.to_datetime(
            articles["published_time"],
            errors="coerce",
        )
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

    assert (
        metadata["published_time"]
        <= impression_time
    ).all()


def test_top_k_never_exceeds_requested_value(
    resources,
):
    """The optimized retriever must respect top_k."""

    query_tokens = [
        "nyheder",
        "sport",
    ]

    for top_k in (
        1,
        5,
        10,
        50,
        100,
        500,
    ):
        result = (
            retrieve_bm25_temporal_optimized(
                resource=resources[
                    "optimized_resource"
                ],
                query_tokens=query_tokens,
                impression_time=pd.Timestamp(
                    "2023-05-28 04:21:24"
                ),
                top_k=top_k,
            )
        )

        assert len(result) <= top_k


def test_article_ids_are_strings(
    resources,
):
    """Optimized candidate IDs should have a stable string representation."""

    result = (
        retrieve_bm25_temporal_optimized(
            resource=resources[
                "optimized_resource"
            ],
            query_tokens=[
                "sport",
                "nyheder",
            ],
            impression_time=pd.Timestamp(
                "2023-05-28 04:21:24"
            ),
            top_k=50,
        )
    )

    if result.empty:
        pytest.skip(
            "No candidates returned."
        )

    assert all(
        isinstance(article_id, str)
        for article_id
        in result["article_id"]
    )

