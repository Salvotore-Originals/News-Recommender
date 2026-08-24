from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.candidates import (
    build_hybrid_candidate_scores,
)
from src.ranking.hybrid import (
    score_impression,
)


def predict_mind_candidates(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    bm25_retriever,
    article_embeddings: np.ndarray,
    article_ids: np.ndarray,
    behavioral_scores: pd.DataFrame,
    max_history: int = 10,
    lexical_weight: float = 0.40,
    semantic_weight: float = 0.30,
    behavioral_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Generate label-free MIND candidate predictions.

    The input candidate table must NOT require a `clicked`
    column. The existing candidate scorer expects `clicked`
    because it was originally designed for offline evaluation,
    so this adapter supplies a temporary dummy value.

    The dummy value is never used as a prediction feature and
    is removed from the returned result.

    Returns one row per candidate with:

        impression_id
        user_id
        timestamp
        article_id
        bm25_score
        semantic_score
        behavioral_score
        bm25_normalized
        semantic_normalized
        behavioral_normalized
        hybrid_score
        rank
    """

    required_candidate_columns = {
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
    }

    missing = (
        required_candidate_columns
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            "Candidates are missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Copy candidates.
    #
    # IMPORTANT:
    # The official test set has no clicked label.
    # The existing candidate scorer expects one because it
    # was originally built for evaluation.
    #
    # We therefore provide a temporary dummy value.
    # --------------------------------------------------------

    scoring_candidates = candidates.copy()

    if "clicked" not in scoring_candidates.columns:
        scoring_candidates["clicked"] = 0

    # --------------------------------------------------------
    # Generate candidate-level retrieval/behavioural scores
    # using the existing Phase 2-5 machinery.
    # --------------------------------------------------------

    scored = build_hybrid_candidate_scores(
        candidates=scoring_candidates,
        history=history,
        articles=articles,
        bm25_retriever=bm25_retriever,
        article_embeddings=article_embeddings,
        article_ids=article_ids,
        behavioral_scores=behavioral_scores,
        max_history=max_history,
    )

    # --------------------------------------------------------
    # Apply the existing hybrid fusion independently within
    # each impression.
    # --------------------------------------------------------

    scored_groups = []

    for _, group in scored.groupby(
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
            "No candidate impressions were scored."
        )

    result = pd.concat(
        scored_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove the evaluation-only label.
    # --------------------------------------------------------

    result = result.drop(
        columns=["clicked"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Rank candidates within each impression.
    #
    # Higher hybrid score = better rank.
    # Stable sorting makes ties deterministic.
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "impression_id",
            "hybrid_score",
            "article_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    result["rank"] = (
        result.groupby(
            "impression_id"
        ).cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    score_columns = [
        "bm25_score",
        "semantic_score",
        "behavioral_score",
        "bm25_normalized",
        "semantic_normalized",
        "behavioral_normalized",
        "hybrid_score",
    ]

    for column in score_columns:

        if result[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not np.all(
            np.isfinite(
                result[column].to_numpy(
                    dtype=np.float64
                )
            )
        ):
            raise ValueError(
                f"{column} contains non-finite values."
            )

    if result["article_id"].isna().any():
        raise ValueError(
            "article_id contains missing values."
        )

    if result["rank"].isna().any():
        raise ValueError(
            "rank contains missing values."
        )

    return result