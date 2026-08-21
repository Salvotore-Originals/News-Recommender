from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def min_max_normalize(
    scores: pd.Series,
) -> pd.Series:
    """
    Normalize scores to [0, 1].

    Constant scores are mapped to zero.

    This preserves ranking order while making
    heterogeneous retrieval signals comparable.
    """

    values = scores.to_numpy(
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "Cannot normalize an empty score series."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "scores contain non-finite values."
        )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(
                len(values),
                dtype=np.float64,
            ),
            index=scores.index,
        )

    normalized = (
        (values - minimum)
        /
        (maximum - minimum)
    )

    return pd.Series(
        normalized,
        index=scores.index,
    )


# ============================================================
# WEIGHT VALIDATION
# ============================================================

def validate_hybrid_weights(
    lexical_weight: float,
    semantic_weight: float,
    behavioral_weight: float,
) -> None:
    """
    Validate hybrid fusion weights.

    All weights must be non-negative and sum to one.
    """

    weights = [
        lexical_weight,
        semantic_weight,
        behavioral_weight,
    ]

    if any(
        weight < 0
        for weight in weights
    ):
        raise ValueError(
            "Hybrid weights must be non-negative."
        )

    if not all(
        np.isfinite(weight)
        for weight in weights
    ):
        raise ValueError(
            "Hybrid weights must be finite."
        )

    total = sum(weights)

    if not np.isclose(
        total,
        1.0,
    ):
        raise ValueError(
            "Hybrid weights must sum to 1."
        )


# ============================================================
# SINGLE-IMPRESSION HYBRID SCORE
# ============================================================

def score_impression(
    group: pd.DataFrame,
    lexical_weight: float = 0.40,
    semantic_weight: float = 0.30,
    behavioral_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Normalize and combine retrieval signals for one impression.

    Input columns:

        bm25_score
        semantic_score
        behavioral_score

    Output columns:

        bm25_normalized
        semantic_normalized
        behavioral_normalized
        hybrid_score
    """

    required_columns = {
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    }

    missing = (
        required_columns
        - set(group.columns)
    )

    if missing:
        raise ValueError(
            "Missing required score columns: "
            f"{sorted(missing)}"
        )

    validate_hybrid_weights(
        lexical_weight,
        semantic_weight,
        behavioral_weight,
    )

    result = group.copy()

    result["bm25_normalized"] = (
        min_max_normalize(
            result["bm25_score"]
        )
    )

    result["semantic_normalized"] = (
        min_max_normalize(
            result["semantic_score"]
        )
    )

    result["behavioral_normalized"] = (
        min_max_normalize(
            result["behavioral_score"]
        )
    )

    result["hybrid_score"] = (
        lexical_weight
        * result["bm25_normalized"]
        +
        semantic_weight
        * result["semantic_normalized"]
        +
        behavioral_weight
        * result["behavioral_normalized"]
    )

    if result["hybrid_score"].isna().any():
        raise ValueError(
            "hybrid_score contains missing values."
        )

    if not np.all(
        np.isfinite(
            result["hybrid_score"].to_numpy()
        )
    ):
        raise ValueError(
            "hybrid_score contains non-finite values."
        )

    return result


# ============================================================
# FULL DATASET HYBRID SCORING
# ============================================================

def build_hybrid_scores(
    df: pd.DataFrame,
    lexical_weight: float = 0.40,
    semantic_weight: float = 0.30,
    behavioral_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Build hybrid scores across all impressions.

    Normalization is performed independently within each
    impression.

    This is important because ranking is performed within
    each MIND impression.
    """

    required_columns = {
        "impression_id",
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    validate_hybrid_weights(
        lexical_weight,
        semantic_weight,
        behavioral_weight,
    )

    scored_groups = []

    for _, group in df.groupby(
        "impression_id",
        sort=False,
    ):
        scored_groups.append(
            score_impression(
                group,
                lexical_weight=lexical_weight,
                semantic_weight=semantic_weight,
                behavioral_weight=behavioral_weight,
            )
        )

    if not scored_groups:
        raise ValueError(
            "No impressions found."
        )

    result = pd.concat(
        scored_groups,
        ignore_index=True,
    )

    return result