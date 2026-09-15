from __future__ import annotations

from math import log2

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def hit_at_k(
    group: pd.DataFrame,
    k: int,
) -> float:
    """Return 1 if a clicked article appears in the top-k."""
    ranked = group.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).head(k)

    return float(ranked["clicked"].sum() > 0)


def reciprocal_rank(
    group: pd.DataFrame,
) -> float:
    """Return reciprocal rank of the first clicked article."""
    ranked = group.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    clicked_positions = np.flatnonzero(
        ranked["clicked"].to_numpy() > 0
    )

    if len(clicked_positions) == 0:
        return 0.0

    return 1.0 / float(clicked_positions[0] + 1)


def ndcg_at_k(
    group: pd.DataFrame,
    k: int,
) -> float:
    """Compute binary-label nDCG@k for one impression."""
    ranked = group.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).head(k)

    relevance = ranked["clicked"].to_numpy(
        dtype=np.float64
    )

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(relevance) + 2)
    )

    dcg = float(
        np.sum(
            (2.0 ** relevance - 1.0)
            / discounts
        )
    )

    ideal_relevance = np.sort(
        group["clicked"].to_numpy(
            dtype=np.float64
        )
    )[::-1][:k]

    ideal_discounts = np.log2(
        np.arange(2, len(ideal_relevance) + 2)
    )

    idcg = float(
        np.sum(
            (2.0 ** ideal_relevance - 1.0)
            / ideal_discounts
        )
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_ranking(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate impression-level news recommendation ranking.

    Required columns:
        impression_id
        clicked
        score
    """

    required = {
        "impression_id",
        "clicked",
        "score",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "Missing required prediction columns: "
            f"{sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "Prediction table cannot be empty."
        )

    if not np.isfinite(
        predictions["score"].to_numpy()
    ).all():
        raise ValueError(
            "Prediction scores contain NaN or infinite values."
        )

    grouped = predictions.groupby(
        "impression_id",
        sort=True,
    )

    mrr_values = []
    hit5_values = []
    hit10_values = []
    ndcg5_values = []
    ndcg10_values = []

    for _, group in grouped:
        mrr_values.append(
            reciprocal_rank(group)
        )

        hit5_values.append(
            hit_at_k(group, 5)
        )

        hit10_values.append(
            hit_at_k(group, 10)
        )

        ndcg5_values.append(
            ndcg_at_k(group, 5)
        )

        ndcg10_values.append(
            ndcg_at_k(group, 10)
        )

    labels = predictions["clicked"].to_numpy()

    if len(np.unique(labels)) >= 2:
        auc = float(
            roc_auc_score(
                labels,
                predictions["score"].to_numpy(),
            )
        )
    else:
        auc = float("nan")

    return {
        "auc": auc,
        "mrr": float(np.mean(mrr_values)),
        "ndcg@5": float(np.mean(ndcg5_values)),
        "ndcg@10": float(np.mean(ndcg10_values)),
        "hit@5": float(np.mean(hit5_values)),
        "hit@10": float(np.mean(hit10_values)),
        "impressions": float(len(mrr_values)),
        "candidates": float(len(predictions)),
    }