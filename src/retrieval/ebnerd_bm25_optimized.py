"""
Phase 4F: Optimized EB-NeRD BM25 candidate retrieval.

This module preserves the existing static BM25 corpus and BM25 scores while
optimizing temporal eligibility lookup.

Protocol:
    - BM25 index is built once over the complete article corpus.
    - BM25 IDF values therefore remain identical to the existing implementation.
    - Article publication timestamps are sorted once.
    - Point-in-time eligibility is obtained using binary search.
    - Only eligible corpus indices are considered for top-K selection.

This module is intended to be behaviorally equivalent to the Phase 4E
BM25 retrieval implementation.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptimizedBM25Resource:
    """
    Reusable BM25 resource with precomputed publication-time ordering.
    """

    bm25: object
    index_to_article_id: dict[int, str]
    article_id_to_index: dict[str, int]
    sorted_published_times: list[pd.Timestamp]
    sorted_corpus_indices: np.ndarray


def build_optimized_bm25_resource(
    bm25,
    article_id_to_index: dict,
    articles: pd.DataFrame,
) -> OptimizedBM25Resource:
    """
    Build reusable temporal metadata around an existing BM25 index.

    The BM25 object itself is NOT modified.
    """

    article_metadata = articles[
        ["article_id", "published_time"]
    ].copy()

    article_metadata["article_id"] = (
        article_metadata["article_id"]
        .astype(str)
    )

    article_metadata["published_time"] = pd.to_datetime(
        article_metadata["published_time"],
        errors="coerce",
    )

    index_to_article_id = {
        int(corpus_index): str(article_id)
        for article_id, corpus_index
        in article_id_to_index.items()
    }

    rows = []

    for article_id, corpus_index in article_id_to_index.items():
        article_id = str(article_id)
        corpus_index = int(corpus_index)

        if corpus_index not in index_to_article_id:
            continue

        published_time = article_metadata.loc[
            article_metadata["article_id"] == article_id,
            "published_time",
        ]

        if published_time.empty:
            continue

        timestamp = published_time.iloc[0]

        if pd.isna(timestamp):
            continue

        rows.append(
            (
                pd.Timestamp(timestamp),
                corpus_index,
            )
        )

    rows.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    sorted_published_times = [
        timestamp
        for timestamp, _ in rows
    ]

    sorted_corpus_indices = np.asarray(
        [
            corpus_index
            for _, corpus_index in rows
        ],
        dtype=np.int64,
    )

    return OptimizedBM25Resource(
        bm25=bm25,
        index_to_article_id=index_to_article_id,
        article_id_to_index={
            str(article_id): int(corpus_index)
            for article_id, corpus_index
            in article_id_to_index.items()
        },
        sorted_published_times=sorted_published_times,
        sorted_corpus_indices=sorted_corpus_indices,
    )


def retrieve_bm25_temporal_optimized(
    resource: OptimizedBM25Resource,
    query_tokens: list[str],
    impression_time: pd.Timestamp,
    top_k: int = 500,
) -> pd.DataFrame:
    """
    Retrieve temporally eligible BM25 candidates.

    The BM25 scoring semantics remain unchanged. The optimization concerns
    temporal eligibility lookup and top-K selection only.
    """

    if not query_tokens:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    scores = np.asarray(
        resource.bm25.get_scores(query_tokens),
        dtype=np.float64,
    )

    cutoff = bisect_right(
        resource.sorted_published_times,
        pd.Timestamp(impression_time),
    )

    eligible_indices = (
        resource.sorted_corpus_indices[:cutoff]
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
        len(eligible_indices),
    )

    if k < len(eligible_indices):
        selected_local = np.argpartition(
            -eligible_scores,
            k - 1,
        )[:k]
    else:
        selected_local = np.arange(
            len(eligible_indices)
        )

    selected_local = sorted(
        selected_local.tolist(),
        key=lambda j: (
            -float(eligible_scores[j]),
            resource.index_to_article_id[
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
                resource.index_to_article_id[
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

