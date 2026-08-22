from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.retrieval.ebnerd_bm25 import (
    build_bm25,
    build_candidate_bm25_statistics,
    build_history_lookup as build_bm25_history_lookup,
    build_query_from_history,
    score_candidates_restricted,
)

from src.retrieval.ebnerd_semantic import (
    build_history_lookup as build_semantic_history_lookup,
    build_query_embedding,
    score_candidates_vectorized,
)

from src.retrieval.ebnerd_behavioral import (
    build_article_category_lookup,
    build_temporal_user_category_lookup,
    get_behavioral_preference,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "ebnerd"
    / "small"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)


# ============================================================================
# DEFAULT HYBRID WEIGHTS
# ============================================================================

LEXICAL_WEIGHT = 0.0
SEMANTIC_WEIGHT = 0.02
BEHAVIORAL_WEIGHT = 0.98

PAIRWISE_CONFIGS = {
    "lexical_semantic": {
        "lexical_weight": 0.5,
        "semantic_weight": 0.5,
        "behavioral_weight": 0.0,
    },

    "lexical_behavioral": {
        "lexical_weight": 0.5,
        "semantic_weight": 0.0,
        "behavioral_weight": 0.5,
    },

    "semantic_behavioral": {
        "lexical_weight": 0.0,
        "semantic_weight": 0.5,
        "behavioral_weight": 0.5,
    },
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_articles() -> pd.DataFrame:

    return pd.read_parquet(
        FEATURE_ROOT / "articles.parquet"
    )


def load_history(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / f"{split}_user_history.parquet"
    )


def load_interactions(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / "splits"
        / f"{split}.parquet"
    )


def load_embeddings():

    embeddings = np.load(
        EMBEDDING_ROOT
        / "article_embeddings.npy"
    )

    article_ids = pd.read_parquet(
        EMBEDDING_ROOT
        / "article_ids.parquet"
    )

    article_to_index = {
        article_id: index
        for index, article_id
        in enumerate(
            article_ids["article_id"]
        )
    }

    return (
        embeddings,
        article_to_index,
    )


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def normalize_scores(
    scores,
) -> np.ndarray:
    """
    Min-max normalize scores within one
    impression candidate set.

    x_hat = (x - min(x)) / (max(x) - min(x))

    If all scores are identical, return zeros.
    """

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(values) == 0:

        return values

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.zeros_like(
            values,
            dtype=np.float64,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================================
# HYBRID SCORE
# ============================================================================

def combine_scores(
    lexical_scores,
    semantic_scores,
    behavioral_scores,
    lexical_weight=1.0/3.0,
    semantic_weight=1.0/3.0,
    behavioral_weight=1.0/3.0,
):
    """
    Combine normalized lexical, semantic,
    and behavioral scores.

    H = w_L L_hat + w_S S_hat + w_B B_hat
    """

    weight_sum = (
        lexical_weight
        + semantic_weight
        + behavioral_weight
    )

    if not np.isclose(
        weight_sum,
        1.0,
    ):

        raise ValueError(
            "Hybrid weights must sum to 1.0. "
            f"Received {weight_sum:.8f}"
        )

    lexical_normalized = (
        normalize_scores(
            lexical_scores
        )
    )

    semantic_normalized = (
        normalize_scores(
            semantic_scores
        )
    )

    behavioral_normalized = (
        normalize_scores(
            behavioral_scores
        )
    )

    return (
        lexical_weight
        * lexical_normalized
        +
        semantic_weight
        * semantic_normalized
        +
        behavioral_weight
        * behavioral_normalized
    )


# ============================================================================
# RANKING
# ============================================================================

def rank_candidates(
    candidate_ids,
    scores,
):
    """
    Rank candidates by descending hybrid score.

    The article ID provides deterministic
    tie-breaking.
    """

    ranked = sorted(
        zip(
            candidate_ids,
            scores,
        ),
        key=lambda item: (
            -float(item[1]),
            str(item[0]),
        ),
    )

    return ranked


# ============================================================================
# METRICS
# ============================================================================

def evaluate_ranking(
    ranked_ids,
    clicked,
):
    """
    Compute Hit@5, Hit@10 and reciprocal rank.
    """

    hit_at_5 = int(
        any(
            article_id in clicked
            for article_id
            in ranked_ids[:5]
        )
    )

    hit_at_10 = int(
        any(
            article_id in clicked
            for article_id
            in ranked_ids[:10]
        )
    )

    reciprocal_rank = 0.0

    for rank, article_id in enumerate(
        ranked_ids,
        start=1,
    ):

        if article_id in clicked:

            reciprocal_rank = (
                1.0 / rank
            )

            break

    return (
        hit_at_5,
        hit_at_10,
        reciprocal_rank,
    )


# ============================================================================
# BEHAVIORAL SCORING
# ============================================================================

def build_behavioral_scores(
    user_id,
    impression_time,
    candidate_ids,
    article_categories,
    temporal_lookup,
):
    """
    Generate behavioral preference for every
    candidate article.
    """

    scores = []

    for article_id in candidate_ids:

        category = (
            article_categories.get(
                article_id
            )
        )

        score = (
            get_behavioral_preference(
                user_id,
                category,
                impression_time,
                temporal_lookup,
            )
        )

        scores.append(
            score
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


# ============================================================================
# RESOURCE BUILDING
# ============================================================================

def build_resources(
    split,
    articles,
):
    """
    Build the three retrieval resources
    required by the hybrid evaluator.
    """

    print(
        f"\nLoading {split} history..."
    )

    history = load_history(
        split
    )

    print(
        f"{split} history events: "
        f"{len(history):,}"
    )

    # ------------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------------

    print(
        f"\nBuilding {split} BM25 resources..."
    )

    _, bm25_article_to_index = build_bm25(
        articles
    )

    bm25_statistics = (
        build_candidate_bm25_statistics(
            articles
        )
    )

    bm25_history_lookup = (
        build_bm25_history_lookup(
            history,
            articles,
        )
    )

    semantic_history_lookup = (
    build_semantic_history_lookup(
        history,
    )
)

    # ------------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------------

    print(
        "\nLoading semantic embeddings..."
    )

    (
        semantic_embeddings,
        semantic_article_to_index,
    ) = load_embeddings()

    # ------------------------------------------------------------------------
    # Behavioral
    # ------------------------------------------------------------------------

    print(
        f"\nBuilding {split} behavioral resources..."
    )

    article_categories = (
        build_article_category_lookup(
            articles
        )
    )

    temporal_lookup = (
        build_temporal_user_category_lookup(
            history,
            article_categories,
        )
    )

    return {
    "bm25_article_to_index":
        bm25_article_to_index,

    "bm25_statistics":
        bm25_statistics,

    "bm25_history_lookup":
        bm25_history_lookup,

    "semantic_history_lookup":
        semantic_history_lookup,

    "semantic_embeddings":
        semantic_embeddings,

    "semantic_article_to_index":
        semantic_article_to_index,

    "article_categories":
        article_categories,

    "temporal_lookup":
        temporal_lookup,
}


# ============================================================================
# SPLIT EVALUATION
# ============================================================================

def evaluate_split(
    split,
    resources,
    lexical_weight=LEXICAL_WEIGHT,
    semantic_weight=SEMANTIC_WEIGHT,
    behavioral_weight=BEHAVIORAL_WEIGHT,
    max_impressions=None,
):
    """
    Evaluate the complete L + S + B hybrid.
    """

    interactions = load_interactions(
        split
    )

    grouped = interactions.groupby(
        "impression_id",
        sort=False,
    )

    hits_5 = 0
    hits_10 = 0
    reciprocal_rank_sum = 0.0

    total_candidates = 0
    evaluated = 0

    start = time.perf_counter()

    for (
        impression_id,
        group,
    ) in grouped:

        if (
            max_impressions is not None
            and evaluated >= max_impressions
        ):
            break

        user_id = group.iloc[0][
            "user_id"
        ]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidate_ids = (
            group["article_id"]
            .tolist()
        )

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        # ====================================================================
        # LEXICAL
        # ====================================================================

        bm25_history_events = (
            resources[
                "bm25_history_lookup"
            ].get(
                user_id,
            [],
        )
    )

        semantic_history_events = (
            resources[
                "semantic_history_lookup"
            ].get(
                user_id,
            [],
        )
    )

        query_tokens = (
            build_query_from_history(
                bm25_history_events,
                impression_time,
            )
        )

        lexical_ranked = (
            score_candidates_restricted(
                query_tokens,
                candidate_ids,
                resources[
                    "bm25_article_to_index"
                ],
                resources[
                    "bm25_statistics"
                ],
            )
        )

        lexical_map = dict(
            lexical_ranked
        )

        lexical_scores = np.asarray(
            [
                lexical_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ====================================================================
        # SEMANTIC
        # ====================================================================

        query_embedding = (
            build_query_embedding(
                semantic_history_events,
                impression_time,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_ranked = (
            score_candidates_vectorized(
                query_embedding,
                candidate_ids,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_map = dict(
            semantic_ranked
        )

        semantic_scores = np.asarray(
            [
                semantic_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ====================================================================
        # BEHAVIORAL
        # ====================================================================

        behavioral_scores = (
            build_behavioral_scores(
                user_id,
                impression_time,
                candidate_ids,
                resources[
                    "article_categories"
                ],
                resources[
                    "temporal_lookup"
                ],
            )
        )

        # ====================================================================
        # HYBRID
        # ====================================================================

        final_scores = combine_scores(
            lexical_scores,
            semantic_scores,
            behavioral_scores,
            lexical_weight,
            semantic_weight,
            behavioral_weight,
        )

        ranked = rank_candidates(
            candidate_ids,
            final_scores,
        )

        ranked_ids = [
            article_id
            for article_id, _score
            in ranked
        ]

        (
            hit5,
            hit10,
            reciprocal_rank,
        ) = evaluate_ranking(
            ranked_ids,
            clicked,
        )

        hits_5 += hit5
        hits_10 += hit10

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        total_candidates += len(
            candidate_ids
        )

        evaluated += 1

        if evaluated % 10_000 == 0:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{split}: "
                f"{evaluated:,} impressions | "
                f"{elapsed:.1f}s"
            )

    if evaluated == 0:

        raise ValueError(
            f"No impressions evaluated for "
            f"{split}."
        )

    runtime = (
        time.perf_counter()
        - start
    )

    return {
        "split": split,
        "lexical_weight":
            lexical_weight,
        "semantic_weight":
            semantic_weight,
        "behavioral_weight":
            behavioral_weight,
        "impressions":
            evaluated,
        "hit_at_5":
            hits_5 / evaluated,
        "hit_at_10":
            hits_10 / evaluated,
        "mrr":
            reciprocal_rank_sum
            / evaluated,
        "mean_candidates":
            total_candidates
            / evaluated,
        "runtime_seconds":
            runtime,
    }


def evaluate_pairwise_split(
    split,
    resources,
    configs,
    max_impressions=None,
):
    """
    Evaluate multiple pairwise hybrid configurations
    using the same L/S/B scores.

    This avoids recomputing retrieval scores for
    every configuration.
    """

    interactions = load_interactions(
        split
    )

    grouped = interactions.groupby(
        "impression_id",
        sort=False,
    )

    accumulators = {}

    for name in configs:

        accumulators[name] = {
            "hits_5": 0,
            "hits_10": 0,
            "rr_sum": 0.0,
        }

    evaluated = 0
    total_candidates = 0

    start = time.perf_counter()

    for (
        impression_id,
        group,
    ) in grouped:

        if (
            max_impressions is not None
            and evaluated >= max_impressions
        ):
            break

        user_id = group.iloc[0][
            "user_id"
        ]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidate_ids = (
            group["article_id"]
            .tolist()
        )

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        # ================================================================
        # BM25
        # ================================================================

        bm25_history_events = (
            resources[
                "bm25_history_lookup"
            ].get(
                user_id,
                [],
            )
        )

        query_tokens = (
            build_query_from_history(
                bm25_history_events,
                impression_time,
            )
        )

        lexical_ranked = (
            score_candidates_restricted(
                query_tokens,
                candidate_ids,
                resources[
                    "bm25_article_to_index"
                ],
                resources[
                    "bm25_statistics"
                ],
            )
        )

        lexical_map = dict(
            lexical_ranked
        )

        lexical_scores = np.asarray(
            [
                lexical_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ================================================================
        # SEMANTIC
        # ================================================================

        semantic_history_events = (
            resources[
                "semantic_history_lookup"
            ].get(
                user_id,
                [],
            )
        )

        query_embedding = (
            build_query_embedding(
                semantic_history_events,
                impression_time,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_ranked = (
            score_candidates_vectorized(
                query_embedding,
                candidate_ids,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_map = dict(
            semantic_ranked
        )

        semantic_scores = np.asarray(
            [
                semantic_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ================================================================
        # BEHAVIORAL
        # ================================================================

        behavioral_scores = (
            build_behavioral_scores(
                user_id,
                impression_time,
                candidate_ids,
                resources[
                    "article_categories"
                ],
                resources[
                    "temporal_lookup"
                ],
            )
        )

        # ================================================================
        # EVALUATE EACH PAIR
        # ================================================================

        for (
            name,
            weights,
        ) in configs.items():

            hybrid_scores = combine_scores(
                lexical_scores,
                semantic_scores,
                behavioral_scores,
                weights[
                    "lexical_weight"
                ],
                weights[
                    "semantic_weight"
                ],
                weights[
                    "behavioral_weight"
                ],
            )

            ranked = rank_candidates(
                candidate_ids,
                hybrid_scores,
            )

            ranked_ids = [
                article_id
                for article_id, _score
                in ranked
            ]

            (
                hit5,
                hit10,
                reciprocal_rank,
            ) = evaluate_ranking(
                ranked_ids,
                clicked,
            )

            accumulators[name][
                "hits_5"
            ] += hit5

            accumulators[name][
                "hits_10"
            ] += hit10

            accumulators[name][
                "rr_sum"
            ] += reciprocal_rank

        evaluated += 1

        total_candidates += len(
            candidate_ids
        )

        if evaluated % 10_000 == 0:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{split}: "
                f"{evaluated:,} impressions | "
                f"{elapsed:.1f}s"
            )

    if evaluated == 0:

        raise ValueError(
            f"No impressions evaluated "
            f"for {split}."
        )

    runtime = (
        time.perf_counter()
        - start
    )

    results = []

    for name, metrics in (
        accumulators.items()
    ):

        weights = configs[name]

        results.append(
            {
                "split": split,
                "model": name,
                "lexical_weight":
                    weights[
                        "lexical_weight"
                    ],
                "semantic_weight":
                    weights[
                        "semantic_weight"
                    ],
                "behavioral_weight":
                    weights[
                        "behavioral_weight"
                    ],
                "impressions":
                    evaluated,
                "hit_at_5":
                    metrics["hits_5"]
                    / evaluated,
                "hit_at_10":
                    metrics["hits_10"]
                    / evaluated,
                "mrr":
                    metrics["rr_sum"]
                    / evaluated,
                "mean_candidates":
                    total_candidates
                    / evaluated,
                "runtime_seconds":
                    runtime,
            }
        )

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("PHASE 7.12 — FINAL TUNED EB-NeRD EVALUATION")
    print("=" * 80)

    print(
        "\nHybrid configuration:"
    )

    print(
        f"Lexical weight:    "
        f"{LEXICAL_WEIGHT:.6f}"
    )

    print(
        f"Semantic weight:   "
        f"{SEMANTIC_WEIGHT:.6f}"
    )

    print(
        f"Behavioral weight: "
        f"{BEHAVIORAL_WEIGHT:.6f}"
    )

    articles = load_articles()

    print(
        f"\nArticles: "
        f"{len(articles):,}"
    )

    # ========================================================================
    # TRAIN
    # ========================================================================

    train_resources = build_resources(
        "train",
        articles,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAIN HYBRID SMOKE TEST"
    )

    print(
        "=" * 80
    )

    train_results = evaluate_split(
        "train",
        train_resources,
        lexical_weight=LEXICAL_WEIGHT,
        semantic_weight=SEMANTIC_WEIGHT,
        behavioral_weight=BEHAVIORAL_WEIGHT,
        max_impressions=None,
    )

    print(
        "\nTRAIN RESULTS"
    )

    print(
        f"Impressions: "
        f"{train_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{train_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{train_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{train_results['mrr']:.6f}"
    )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    validation_resources = (
        build_resources(
            "validation",
            articles,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION HYBRID SMOKE TEST"
    )

    print(
        "=" * 80
    )

    validation_results = evaluate_split(
        "validation",
        validation_resources,
        lexical_weight=LEXICAL_WEIGHT,
        semantic_weight=SEMANTIC_WEIGHT,
        behavioral_weight=BEHAVIORAL_WEIGHT,
        max_impressions=None,
    )

    print(
        "\nVALIDATION RESULTS"
    )

    print(
        f"Impressions: "
        f"{validation_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{validation_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{validation_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{validation_results['mrr']:.6f}"
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            train_results,
            validation_results,
        ]
    )

    output_path = (
        RESULT_ROOT
        / "hybrid_final.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL TUNED EB-NeRD EVALUATION COMPLETE"
    )

    print("=" * 80)

    print(
        f"\nFrozen weights: "
        f"L={LEXICAL_WEIGHT:.2f}, "
        f"S={SEMANTIC_WEIGHT:.2f}, "
        f"B={BEHAVIORAL_WEIGHT:.2f}"
    )

    print(
        f"Final results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
