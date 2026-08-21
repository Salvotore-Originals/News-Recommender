from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.hybrid import build_hybrid_scores


# ============================================================
# RANKING METRICS
# ============================================================

def reciprocal_rank(
    labels: np.ndarray,
) -> float:
    """
    Reciprocal rank of the first relevant candidate.
    """

    positions = np.flatnonzero(
        labels > 0
    )

    if len(positions) == 0:
        return 0.0

    return 1.0 / (
        positions[0] + 1
    )


def hit_at_k(
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Hit@K for binary relevance labels.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return float(
        np.any(
            labels[:k] > 0
        )
    )


def ndcg_at_k(
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Binary-label NDCG@K.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    labels = labels[:k]

    if len(labels) == 0:
        return 0.0

    positions = np.arange(
        1,
        len(labels) + 1,
    )

    gains = (
        2 ** labels
    ) - 1

    discounts = np.log2(
        positions + 1
    )

    dcg = np.sum(
        gains / discounts
    )

    relevant_count = int(
        np.sum(labels > 0)
    )

    if relevant_count == 0:
        return 0.0

    ideal_length = min(
        relevant_count,
        k,
    )

    ideal_positions = np.arange(
        1,
        ideal_length + 1,
    )

    ideal_dcg = np.sum(
        1.0
        /
        np.log2(
            ideal_positions + 1
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg / ideal_dcg
    )


# ============================================================
# SINGLE SCORE EVALUATION
# ============================================================

def evaluate_score(
    df: pd.DataFrame,
    score_column: str,
    k_values: tuple[int, ...] = (
        5,
        10,
    ),
) -> dict[str, float]:
    """
    Evaluate one candidate-level score.

    Ranking is performed independently within
    each impression.
    """

    required = {
        "impression_id",
        "clicked",
        score_column,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    metrics = []

    for _, group in df.groupby(
        "impression_id",
        sort=False,
    ):

        ranked = group.sort_values(
            score_column,
            ascending=False,
            kind="mergesort",
        )

        labels = ranked[
            "clicked"
        ].to_numpy()

        result = {
            "MRR": reciprocal_rank(
                labels
            ),
        }

        for k in k_values:

            result[
                f"Hit@{k}"
            ] = hit_at_k(
                labels,
                k,
            )

            result[
                f"NDCG@{k}"
            ] = ndcg_at_k(
                labels,
                k,
            )

        metrics.append(
            result
        )

    if not metrics:
        raise ValueError(
            "No impressions found."
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    return {
        column: float(
            metrics_df[column].mean()
        )
        for column
        in metrics_df.columns
    }


# ============================================================
# EVALUATE HYBRID + BASELINES
# ============================================================

def evaluate_hybrid(
    candidate_scores: pd.DataFrame,
    lexical_weight: float = 0.40,
    semantic_weight: float = 0.30,
    behavioral_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Evaluate BM25, semantic, behavioural and hybrid rankings
    on exactly the same candidate impressions.
    """

    required = {
        "impression_id",
        "clicked",
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    }

    missing = (
        required
        - set(candidate_scores.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Candidate rows: "
        f"{len(candidate_scores):,}"
    )

    print(
        f"Impressions: "
        f"{candidate_scores['impression_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Build hybrid score
    # --------------------------------------------------------

    scored = build_hybrid_scores(
        candidate_scores,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        behavioral_weight=behavioral_weight,
    )

    score_columns = {
        "BM25": "bm25_score",
        "Semantic": "semantic_score",
        "Behavioural": "behavioral_score",
        "Hybrid": "hybrid_score",
    }

    results = []

    for name, column in score_columns.items():

        print(
            f"\nEvaluating {name}..."
        )

        metrics = evaluate_score(
            scored,
            column,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

        print(
            f"       Hit@5:   "
            f"{metrics['Hit@5']:.6f}"
        )

        print(
            f"       Hit@10:  "
            f"{metrics['Hit@10']:.6f}"
        )

        print(
            f"       MRR:     "
            f"{metrics['MRR']:.6f}"
        )

        print(
            f"       NDCG@5:  "
            f"{metrics['NDCG@5']:.6f}"
        )

        print(
            f"       NDCG@10: "
            f"{metrics['NDCG@10']:.6f}"
        )

    return pd.DataFrame(
        results
    )