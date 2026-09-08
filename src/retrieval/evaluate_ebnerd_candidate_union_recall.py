"""
Phase 4E: EB-NeRD Candidate-Source Union Evaluation.

Evaluates candidate discovery on the same validation impressions for:
    1. BM25
    2. Semantic
    3. Fresh + Category
    4. BM25 UNION Fresh
    5. Semantic UNION Fresh
    6. BM25 UNION Semantic
    7. BM25 UNION Semantic UNION Fresh

This is a candidate-recall experiment. It does not modify or call the
existing RRF implementation and does not use official impression candidate
lists for generation.

Temporal protocol:
    - user history timestamp < impression_time
    - article published_time <= impression_time
    - Fresh branch additionally requires published_time >= T - 48h

The BM25 and Semantic branches apply temporal article eligibility BEFORE
selecting their top-K candidates in this evaluator, so the comparison is
corpus-level and temporally consistent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd

from src.retrieval.ebnerd_bm25 import (
    build_bm25,
    build_history_lookup as build_bm25_history_lookup,
    build_query_from_history,
)
from src.retrieval.ebnerd_fresh_candidates import (
    FreshCategoryCandidateGenerator,
)
from src.retrieval.ebnerd_semantic import (
    build_history_lookup as build_semantic_history_lookup,
    build_query_embedding,
)

from src.retrieval.ebnerd_bm25_optimized import (
    build_optimized_bm25_resource,
    retrieve_bm25_temporal_optimized,
)


PROCESSED_ROOT = Path("data/processed/ebnerd/small")
FEATURES_ROOT = Path("data/features/ebnerd/small")
EMBEDDINGS_ROOT = Path("data/embeddings/ebnerd/small")
RESULTS_ROOT = Path("data/results/ebnerd/small")

ARTICLES_PATH = FEATURES_ROOT / "articles.parquet"
HISTORY_PATH = PROCESSED_ROOT / "validation_user_history.parquet"
INTERACTIONS_PATH = PROCESSED_ROOT / "splits" / "validation.parquet"
EMBEDDINGS_PATH = EMBEDDINGS_ROOT / "article_embeddings.npy"
EMBEDDING_IDS_PATH = EMBEDDINGS_ROOT / "article_ids.parquet"

OUTPUT_PATH = RESULTS_ROOT / "candidate_union_recall_1000.csv"

N_IMPRESSIONS = 1000
RETRIEVAL_TOP_K = 500

FRESHNESS_WINDOW_HOURS = 48.0
DECAY_HOURS = 24.0
TOP_CATEGORIES = 3
PER_CATEGORY = 200

RECALL_KS = (50, 100, 200, 500)


def validate_paths() -> None:
    """Validate that all required Phase 4E artifacts exist."""
    paths = (
        ARTICLES_PATH,
        HISTORY_PATH,
        INTERACTIONS_PATH,
        EMBEDDINGS_PATH,
        EMBEDDING_IDS_PATH,
    )

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )


def load_data():
    """Load articles, history, validation interactions, and embeddings."""
    validate_paths()

    articles = pd.read_parquet(ARTICLES_PATH)
    history = pd.read_parquet(HISTORY_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)

    # Evaluation click IDs are normalized to strings so that they match
    # candidate-generator output IDs.
    interactions["article_id"] = (
        interactions["article_id"].astype(str)
    )

    embeddings = np.load(EMBEDDINGS_PATH)
    embedding_ids = pd.read_parquet(EMBEDDING_IDS_PATH)

    if "article_id" not in embedding_ids.columns:
        raise ValueError(
            "Embedding ID artifact must contain article_id."
        )

    embedding_ids = (
        embedding_ids["article_id"]
        .astype(str)
        .tolist()
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Article embeddings must be a 2-D array."
        )

    if embeddings.shape[0] != len(embedding_ids):
        raise ValueError(
            "Embedding row count does not match embedding ID count."
        )

    # Stored vectors are expected to be normalized, but normalize
    # defensively before retrieval.
    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError(
            "Embedding artifact contains zero vectors."
        )

    embeddings = (
        embeddings / norms[:, None]
    ).astype(np.float32)

    article_ids = (
        articles["article_id"]
        .astype(str)
        .tolist()
    )

    if set(embedding_ids) != set(article_ids):
        raise ValueError(
            "Embedding IDs do not exactly match the article corpus IDs."
        )

    embedding_index = {
        article_id: index
        for index, article_id in enumerate(embedding_ids)
    }

    return (
        articles,
        history,
        interactions,
        embeddings,
        embedding_ids,
        embedding_index,
    )


def group_clicked_articles(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Group all clicked article IDs by impression."""
    clicked = interactions.loc[
        interactions["clicked"].astype(int) == 1,
        ["impression_id", "article_id"],
    ].copy()

    return (
        clicked.groupby("impression_id")["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .reset_index(name="clicked_articles")
    )


def retrieve_semantic_temporal(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: Dict[str, int],
    query_embedding: np.ndarray | None,
    impression_time: pd.Timestamp,
    top_k: int,
) -> pd.DataFrame:
    """
    Retrieve semantic top-K after publication-time eligibility filtering.
    """

    if query_embedding is None:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    norm = np.linalg.norm(query)

    if norm == 0:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    query = query / norm

    article_ids = (
        articles["article_id"]
        .astype(str)
        .tolist()
    )

    published = pd.to_datetime(
        articles["published_time"],
        errors="coerce",
    )

    eligible = (
        published.notna()
        & (published <= impression_time)
    )

    eligible_positions = np.flatnonzero(
        eligible.to_numpy()
    )

    if len(eligible_positions) == 0:
        return pd.DataFrame(
            columns=["article_id", "score"]
        )

    eligible_embedding_indices = np.array(
        [
            embedding_index[
                article_ids[i]
            ]
            for i in eligible_positions
        ],
        dtype=np.int64,
    )

    scores = (
        embeddings[
            eligible_embedding_indices
        ] @ query
    )

    k = min(
        top_k,
        len(scores),
    )

    if k < len(scores):
        selected_local = np.argpartition(
            -scores,
            k - 1,
        )[:k]
    else:
        selected_local = np.arange(
            len(scores)
        )

    selected_local = sorted(
        selected_local.tolist(),
        key=lambda j: (
            -float(scores[j]),
            article_ids[
                eligible_positions[j]
            ],
        ),
    )[:top_k]

    return pd.DataFrame(
        {
            "article_id": [
                article_ids[
                    eligible_positions[j]
                ]
                for j in selected_local
            ],
            "score": [
                float(scores[j])
                for j in selected_local
            ],
        }
    )


def union_ranked_ids(
    source_frames: Dict[str, pd.DataFrame],
    sources: Iterable[str],
    k: int,
) -> list[str]:
    """
    Form a deterministic union of the top-k candidates from each source.

    Candidate generation is evaluated by membership, not by a ranking fusion.

    Ordering:
        source order -> source rank -> article_id

    The complete union is returned. It is intentionally NOT truncated to k.
    """

    seen: Set[str] = set()
    result: list[str] = []

    for source in sources:
        frame = source_frames[source].head(k)

        for article_id in frame["article_id"].astype(str):
            if article_id not in seen:
                seen.add(article_id)
                result.append(article_id)

    return result


def recall_at_k(
    candidate_ids: list[str],
    clicked_articles: Set[str],
) -> int:
    """
    Check whether any clicked article exists in the supplied candidate pool.

    The caller is responsible for constructing the appropriate top-k source
    pool before invoking this function.
    """

    if not clicked_articles:
        return 0

    return int(
        bool(
            set(candidate_ids).intersection(
                clicked_articles
            )
        )
    )


def evaluate(
    n_impressions: int = N_IMPRESSIONS,
) -> pd.DataFrame:
    """Run the Phase 4E candidate-source union experiment."""

    (
        articles,
        history,
        interactions,
        embeddings,
        embedding_ids,
        embedding_index,
    ) = load_data()

    impressions = (
        interactions[
            [
                "impression_id",
                "user_id",
                "impression_time",
            ]
        ]
        .drop_duplicates("impression_id")
        .sort_values(
            [
                "impression_time",
                "impression_id",
            ]
        )
        .head(n_impressions)
        .reset_index(drop=True)
    )

    clicks = group_clicked_articles(
        interactions
    )

    evaluation = impressions.merge(
        clicks,
        on="impression_id",
        how="left",
    )

    evaluation["clicked_articles"] = (
        evaluation["clicked_articles"]
        .apply(
            lambda value:
                value
                if isinstance(value, set)
                else set()
        )
    )

    print("=" * 80)
    print(
        "PHASE 4E — CANDIDATE-SOURCE UNION RECALL"
    )
    print("=" * 80)
    print(
        f"Articles: {len(articles):,}"
    )
    print(
        "Validation impressions evaluated: "
        f"{len(evaluation):,}"
    )
    print(
        f"Source top-K: {RETRIEVAL_TOP_K}"
    )
    print(
        "Fresh configuration: "
        f"window={FRESHNESS_WINDOW_HOURS:.1f}h, "
        f"decay={DECAY_HOURS:.1f}h, "
        f"top_categories={TOP_CATEGORIES}, "
        f"per_category={PER_CATEGORY}"
    )
    print()

    # ------------------------------------------------------------------
    # Build reusable resources once.
    # ------------------------------------------------------------------

    print("Building BM25 corpus...")

    bm25_start = time.perf_counter()

    bm25, article_id_to_index = build_bm25(
        articles
    )

    optimized_bm25_resource = build_optimized_bm25_resource(
        bm25=bm25,
        article_id_to_index=article_id_to_index,
        articles=articles,
    )

    article_id_to_published = (
        articles.assign(
            article_id=articles[
                "article_id"
            ].astype(str),
            published_time=pd.to_datetime(
                articles[
                    "published_time"
                ],
                errors="coerce",
            ),
        )
        .set_index("article_id")[
            "published_time"
        ]
        .to_dict()
    )
    print(
        "BM25 build time: "
        f"{time.perf_counter() - bm25_start:.2f}s"
    )

    print("Building history lookups...")

    bm25_history_lookup = (
        build_bm25_history_lookup(
            history,
            articles,
        )
    )

    # The semantic history builder returns integer article IDs, while
    # embedding_index uses string article IDs. Normalize only at this
    # evaluator boundary; do not modify the tested semantic module.
    semantic_history_lookup_raw = (
        build_semantic_history_lookup(
            history
        )
    )

    semantic_history_lookup = {
        user_id: [
            (timestamp, str(article_id))
            for timestamp, article_id in events
        ]
        for user_id, events
        in semantic_history_lookup_raw.items()
    }

    fresh_generator = (
        FreshCategoryCandidateGenerator(
            articles=articles,
            history=history,
            freshness_window_hours=(
                FRESHNESS_WINDOW_HOURS
            ),
            decay_hours=DECAY_HOURS,
            top_categories=TOP_CATEGORIES,
            per_category=PER_CATEGORY,
        )
    )

    source_names = (
        "bm25",
        "semantic",
        "fresh",
    )

    union_names = {
        "bm25_fresh": (
            "bm25",
            "fresh",
        ),
        "semantic_fresh": (
            "semantic",
            "fresh",
        ),
        "bm25_semantic": (
            "bm25",
            "semantic",
        ),
        "all_three": (
            "bm25",
            "semantic",
            "fresh",
        ),
    }

    all_names = (
        "bm25",
        "semantic",
        "fresh",
        "bm25_fresh",
        "semantic_fresh",
        "bm25_semantic",
        "all_three",
    )

    counts = {
        name: {
            k: 0
            for k in RECALL_KS
        }
        for name in all_names
    }

    source_pool_sizes = {
        name: []
        for name in all_names
    }

    source_times = {
        name: []
        for name in source_names
    }

    future_violations = {
        name: 0
        for name in source_names
    }

    total_start = time.perf_counter()

    for row in evaluation.itertuples(
        index=False
    ):
        impression_time = pd.Timestamp(
            row.impression_time
        )

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------

        start = time.perf_counter()

        history_events = (
            bm25_history_lookup.get(
                row.user_id,
                [],
            )
        )

        query = build_query_from_history(
            history_events,
            impression_time,
        )

        bm25_candidates = retrieve_bm25_temporal_optimized(
            resource=optimized_bm25_resource,
            query_tokens=query,
            impression_time=impression_time,
            top_k=RETRIEVAL_TOP_K,
        )

        source_times["bm25"].append(
            time.perf_counter() - start
        )

        # --------------------------------------------------------------
        # Semantic
        # --------------------------------------------------------------

        start = time.perf_counter()

        semantic_events = (
            semantic_history_lookup.get(
                row.user_id,
                [],
            )
        )

        query_embedding = (
            build_query_embedding(
                history_events=semantic_events,
                impression_time=(
                    impression_time
                ),
                article_to_index=(
                    embedding_index
                ),
                article_embeddings=embeddings,
            )
        )

        semantic_candidates = (
            retrieve_semantic_temporal(
                articles=articles,
                embeddings=embeddings,
                embedding_index=(
                    embedding_index
                ),
                query_embedding=(
                    query_embedding
                ),
                impression_time=(
                    impression_time
                ),
                top_k=RETRIEVAL_TOP_K,
            )
        )

        source_times["semantic"].append(
            time.perf_counter() - start
        )

        # --------------------------------------------------------------
        # Fresh + Category
        # --------------------------------------------------------------

        start = time.perf_counter()

        fresh_candidates = (
            fresh_generator.generate_candidates(
                user_id=row.user_id,
                impression_time=(
                    impression_time
                ),
                top_k=RETRIEVAL_TOP_K,
            )
        )

        source_times["fresh"].append(
            time.perf_counter() - start
        )

        source_frames = {
            "bm25": bm25_candidates,
            "semantic": semantic_candidates,
            "fresh": fresh_candidates,
        }

        clicked = row.clicked_articles

        # --------------------------------------------------------------
        # Individual source recall.
        #
        # At each K, evaluate the source's actual top-K.
        # --------------------------------------------------------------

        for source in source_names:
            source_frame = source_frames[source]

            source_pool_sizes[source].append(
                len(source_frame)
            )

            for k in RECALL_KS:
                top_k_ids = (
                    source_frame
                    .head(k)["article_id"]
                    .astype(str)
                    .tolist()
                )

                counts[source][k] += (
                    recall_at_k(
                        top_k_ids,
                        clicked,
                    )
                )

        # --------------------------------------------------------------
        # Union recall.
        #
        # At each K:
        #
        #   BM25 ∪ Fresh
        #
        # means:
        #
        #   top-K(BM25) ∪ top-K(Fresh)
        #
        # We do NOT truncate the resulting union back to K.
        # --------------------------------------------------------------

        for name, sources in union_names.items():
            full_union = union_ranked_ids(
                source_frames,
                sources,
                RETRIEVAL_TOP_K,
            )

            source_pool_sizes[name].append(
                len(full_union)
            )

            for k in RECALL_KS:
                union_at_k = union_ranked_ids(
                    source_frames,
                    sources,
                    k,
                )

                counts[name][k] += (
                    recall_at_k(
                        union_at_k,
                        clicked,
                    )
                )

        # --------------------------------------------------------------
        # Temporal validation.
        # --------------------------------------------------------------

        article_metadata = articles[
            [
                "article_id",
                "published_time",
            ]
        ].copy()

        article_metadata["article_id"] = (
            article_metadata["article_id"]
            .astype(str)
        )

        for source_name, candidate_frame in (
            source_frames.items()
        ):
            if candidate_frame.empty:
                continue

            candidates_for_check = (
                candidate_frame.copy()
            )

            candidates_for_check[
                "article_id"
            ] = (
                candidates_for_check[
                    "article_id"
                ].astype(str)
            )

            if (
                "published_time"
                in candidates_for_check.columns
            ):
                published = pd.to_datetime(
                    candidates_for_check[
                        "published_time"
                    ],
                    errors="coerce",
                )
            else:
                metadata = (
                    candidates_for_check.merge(
                        article_metadata,
                        on="article_id",
                        how="left",
                    )
                )

                published = pd.to_datetime(
                    metadata[
                        "published_time"
                    ],
                    errors="coerce",
                )

            future_violations[
                source_name
            ] += int(
                (
                    published
                    > impression_time
                ).sum()
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print("Results:")
    print()

    print(
        f"{'Candidate source':<28}"
        f"{'@50':>10}"
        f"{'@100':>10}"
        f"{'@200':>10}"
        f"{'@500':>10}"
    )

    print("-" * 68)

    display_order = (
        "bm25",
        "semantic",
        "fresh",
        "bm25_fresh",
        "semantic_fresh",
        "bm25_semantic",
        "all_three",
    )

    for name in display_order:
        values = [
            counts[name][k]
            / len(evaluation)
            for k in RECALL_KS
        ]

        print(
            f"{name:<28}"
            f"{values[0]:>10.4f}"
            f"{values[1]:>10.4f}"
            f"{values[2]:>10.4f}"
            f"{values[3]:>10.4f}"
        )

    print()
    print("Mean candidate-pool sizes:")

    for name in display_order:
        print(
            f"  {name:<25}: "
            f"{np.mean(source_pool_sizes[name]):.2f}"
        )

    print()
    print("Mean source generation times:")

    for name in source_names:
        print(
            f"  {name:<25}: "
            f"{np.mean(source_times[name]):.4f}s"
        )

    print()
    print(
        f"Total evaluation time: "
        f"{total_time:.2f}s"
    )

    print()
    print(
        "Future publication violations:"
    )

    for name in source_names:
        print(
            f"  {name:<25}: "
            f"{future_violations[name]}"
        )

    # ------------------------------------------------------------------
    # Save compact summary.
    # ------------------------------------------------------------------

    summary_rows = []

    for name in display_order:
        summary_rows.append(
            {
                "candidate_source": name,
                "impressions": len(
                    evaluation
                ),
                **{
                    f"recall@{k}": (
                        counts[name][k]
                        / len(evaluation)
                    )
                    for k in RECALL_KS
                },
                "mean_candidates": np.mean(
                    source_pool_sizes[name]
                ),
                "future_publication_violations": (
                    future_violations.get(
                        name,
                        0,
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved summary: {OUTPUT_PATH}"
    )

    return summary


if __name__ == "__main__":
    evaluate()

