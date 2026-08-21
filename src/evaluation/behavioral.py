from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

def get_project_root() -> Path:
    """
    Return the News-Recommender project root.
    """

    return Path(__file__).resolve().parents[2]


FEATURE_MIND = (
    get_project_root()
    / "data"
    / "features"
    / "mind"
)


# ============================================================
# LOAD BEHAVIOURAL SCORES
# ============================================================

def load_behavioral_features(
    split: str,
) -> pd.DataFrame:
    """
    Load candidate-level behavioural features.

    Expected file:

        data/features/mind/
        {split}_behavioral_score.parquet
    """

    path = (
        FEATURE_MIND
        / f"{split}_behavioral_score.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Behavioural score file not found:\n{path}"
        )

    df = pd.read_parquet(path)

    required_columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "clicked",
        "category_preference",
        "subcategory_preference",
        "category_recency",
        "subcategory_recency",
        "smoothed_ctr",
        "interest_score",
        "recency_score",
        "behavioral_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# RANKING METRICS
# ============================================================

def reciprocal_rank(
    labels: np.ndarray,
) -> float:
    """
    Calculate reciprocal rank.

    If the first clicked article is at position k:

        RR = 1 / k

    If there is no clicked article:

        RR = 0
    """

    clicked_positions = np.flatnonzero(
        labels > 0
    )

    if len(clicked_positions) == 0:
        return 0.0

    rank = clicked_positions[0] + 1

    return 1.0 / rank


def hit_at_k(
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Return 1 if at least one clicked article
    appears in the top-k positions.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return float(
        np.any(labels[:k] > 0)
    )


def ndcg_at_k(
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Calculate binary-label NDCG@k.

    MIND impressions may contain multiple clicked
    candidates, so all clicked articles contribute
    to DCG.
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
        (2 ** labels) - 1
    )

    discounts = np.log2(
        positions + 1
    )

    dcg = np.sum(
        gains / discounts
    )

    total_relevant = int(
        np.sum(labels > 0)
    )

    if total_relevant == 0:
        return 0.0

    ideal_length = min(
        total_relevant,
        k,
    )

    ideal_positions = np.arange(
        1,
        ideal_length + 1,
    )

    ideal_dcg = np.sum(
        1.0
        / np.log2(
            ideal_positions + 1
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg / ideal_dcg
    )


# ============================================================
# SINGLE IMPRESSION EVALUATION
# ============================================================

def evaluate_impression(
    group: pd.DataFrame,
    score_column: str,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """
    Evaluate one impression.

    Candidates are sorted from highest score
    to lowest score.
    """

    if score_column not in group.columns:
        raise ValueError(
            f"Score column '{score_column}' "
            "not found."
        )

    ranked = group.sort_values(
        score_column,
        ascending=False,
        kind="mergesort",
    )

    labels = ranked["clicked"].to_numpy()

    result = {
        "mrr": reciprocal_rank(labels),
    }

    for k in k_values:
        result[f"hit@{k}"] = hit_at_k(
            labels,
            k,
        )

        result[f"ndcg@{k}"] = ndcg_at_k(
            labels,
            k,
        )

    return result


# ============================================================
# FULL DATASET EVALUATION
# ============================================================

def evaluate_score(
    df: pd.DataFrame,
    score_column: str,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """
    Evaluate a ranking score across all impressions.

    Metrics are averaged across impressions, not
    candidate rows.
    """

    if score_column not in df.columns:
        raise ValueError(
            f"Score column '{score_column}' "
            "not found."
        )

    required_columns = [
        "impression_id",
        "clicked",
        score_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    metrics = []

    for _, group in df.groupby(
        "impression_id",
        sort=False,
    ):
        metrics.append(
            evaluate_impression(
                group,
                score_column,
                k_values,
            )
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
        for column in metrics_df.columns
    }


# ============================================================
# CREATE BASELINE SCORES
# ============================================================

def add_baseline_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add individual behavioural baseline scores.

    Baselines:

        category
        subcategory
        recency
        popularity
        behavioural
    """

    df = df.copy()

    df["category_score"] = (
        df["category_preference"]
    )

    df["subcategory_score"] = (
        df["subcategory_preference"]
    )

    df["recency_baseline_score"] = (
        0.4 * df["category_recency"]
        +
        0.6 * df["subcategory_recency"]
    )

    df["popularity_score"] = (
        df["smoothed_ctr"]
    )

    return df


# ============================================================
# EVALUATE ALL BEHAVIOURAL BASELINES
# ============================================================

def evaluate_behavioral_baselines(
    split: str = "validation",
) -> pd.DataFrame:
    """
    Evaluate all behavioural ranking baselines.
    """

    print("=" * 70)
    print(
        f"MIND BEHAVIOURAL EVALUATION — "
        f"{split.upper()}"
    )
    print("=" * 70)

    df = load_behavioral_features(
        split
    )

    print(
        f"\nCandidate rows: "
        f"{len(df):,}"
    )

    print(
        f"Impressions: "
        f"{df['impression_id'].nunique():,}"
    )

    df = add_baseline_scores(
        df
    )

    score_columns = {
        "Category": "category_score",
        "Subcategory": "subcategory_score",
        "Recency": "recency_baseline_score",
        "Popularity": "popularity_score",
        "Behavioural": "behavioral_score",
    }

    results = []

    for name, score_column in score_columns.items():

        print(
            f"\nEvaluating {name}..."
        )

        metrics = evaluate_score(
            df,
            score_column,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

        print(
            f"       Hit@5:  "
            f"{metrics['hit@5']:.6f}"
        )

        print(
            f"       Hit@10: "
            f"{metrics['hit@10']:.6f}"
        )

        print(
            f"       MRR:    "
            f"{metrics['mrr']:.6f}"
        )

        print(
            f"       NDCG@5: "
            f"{metrics['ndcg@5']:.6f}"
        )

        print(
            f"       NDCG@10:"
            f" {metrics['ndcg@10']:.6f}"
        )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("BEHAVIOURAL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\n"
        + results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_behavioral_baselines(
        "validation"
    )