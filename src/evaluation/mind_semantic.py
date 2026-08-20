from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.retrieval.mind_semantic import (
    build_user_embedding,
)


def load_semantic_feature_store(
    embeddings_path: str,
    article_ids_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load article embeddings and their corresponding article IDs.
    """

    embeddings = np.load(
        embeddings_path
    )

    article_ids = np.load(
        article_ids_path,
        allow_pickle=True,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Article embeddings must be a 2D matrix."
        )

    if len(article_ids) != embeddings.shape[0]:
        raise ValueError(
            "Article IDs and embeddings must have "
            "the same number of rows."
        )

    return (
        embeddings.astype(
            np.float32,
            copy=False,
        ),
        article_ids.astype(str),
    )


def build_history_lookup(
    history: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Build:

        impression_id -> ordered history article IDs

    The existing MIND history table already represents
    the history available for each impression.
    """

    required_columns = {
        "impression_id",
        "article_id",
        "history_position",
    }

    missing = (
        required_columns
        - set(history.columns)
    )

    if missing:
        raise ValueError(
            f"Missing history columns: {sorted(missing)}"
        )

    ordered = history.sort_values(
        [
            "impression_id",
            "history_position",
        ]
    )

    return (
        ordered
        .groupby(
            "impression_id",
        )["article_id"]
        .apply(
            lambda values:
            values.astype(str).tolist()
        )
        .to_dict()
    )


def evaluate_semantic_retrieval(
    candidates: pd.DataFrame,
    history_lookup: dict[str, list[str]],
    article_embeddings: np.ndarray,
    article_ids: np.ndarray,
    max_history: int = 10,
    ks: Iterable[int] = (5, 10),
) -> dict[str, float]:
    """
    Evaluate semantic retrieval on MIND candidate impressions.

    This implementation is optimized for repeated evaluation
    while preserving the same retrieval semantics as the
    original implementation.

    Skip reasons are explicitly tracked.

    Parameters
    ----------
    candidates:
        Candidate interaction dataframe containing:
        impression_id, user_id, timestamp, article_id, clicked.

    history_lookup:
        impression_id -> historical article IDs.

    article_embeddings:
        Article embedding matrix.

    article_ids:
        Article IDs corresponding to embedding rows.

    max_history:
        Number of most recent historical articles used.

    ks:
        Hit@K values to calculate.

    Returns
    -------
    dict[str, float]
        Retrieval metrics, coverage and skip statistics.
    """

    # ==================================================
    # Validate candidate dataframe
    # ==================================================

    required_columns = {
        "impression_id",
        "article_id",
        "clicked",
    }

    missing = (
        required_columns
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    # ==================================================
    # Validate K values
    # ==================================================

    ks = sorted(
        set(int(k) for k in ks)
    )

    if not ks:
        raise ValueError(
            "At least one k value must be provided."
        )

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All k values must be greater than zero."
        )

    # ==================================================
    # Validate embedding dimensions
    # ==================================================

    if article_embeddings.ndim != 2:
        raise ValueError(
            "Article embeddings must be a 2D matrix."
        )

    if len(article_ids) != article_embeddings.shape[0]:
        raise ValueError(
            "Article IDs and embeddings must have "
            "the same number of rows."
        )

    # ==================================================
    # Article ID -> embedding index
    #
    # Build once.
    # ==================================================

    article_to_index = {
        str(article_id): index
        for index, article_id
        in enumerate(article_ids)
    }

    # ==================================================
    # Normalize article embeddings once.
    #
    # Cosine similarity:
    #
    # cosine(a,b) = normalized(a) dot normalized(b)
    # ==================================================

    embedding_norms = np.linalg.norm(
        article_embeddings,
        axis=1,
        keepdims=True,
    )

    normalized_embeddings = np.divide(
        article_embeddings,
        embedding_norms,
        out=np.zeros_like(
            article_embeddings,
            dtype=np.float32,
        ),
        where=embedding_norms != 0,
    )

    # ==================================================
    # IMPORTANT OPTIMIZATION
    #
    # Convert candidate article IDs to embedding indices
    # ONCE before entering the impression loop.
    #
    # The previous implementation repeated the dictionary
    # lookup for every candidate during every impression.
    # ==================================================

    candidate_work = candidates[
        [
            "impression_id",
            "article_id",
            "clicked",
        ]
    ].copy()

    candidate_work["article_id"] = (
        candidate_work["article_id"]
        .astype(str)
    )

    candidate_work["_embedding_index"] = (
        candidate_work["article_id"]
        .map(article_to_index)
    )

    # Convert clicked to compact integer representation.
    candidate_work["_clicked"] = (
        candidate_work["clicked"]
        .astype(np.int8)
    )

    # ==================================================
    # Metrics
    # ==================================================

    hit_counts = {
        k: 0
        for k in ks
    }

    reciprocal_rank_sum = 0.0

    evaluated_impressions = 0

    # ==================================================
    # Skip counters
    # ==================================================

    skip_counts = {
        "no_history": 0,
        "unknown_history_articles": 0,
        "zero_user_embedding": 0,
        "no_valid_candidates": 0,
    }

    # ==================================================
    # Total impressions
    # ==================================================

    total_impressions = (
        candidate_work["impression_id"]
        .nunique()
    )

    # ==================================================
    # Group candidates by impression
    #
    # groupby is performed once and reused.
    # ==================================================

    grouped_candidates = candidate_work.groupby(
        "impression_id",
        sort=False,
    )

    # ==================================================
    # Evaluate impressions
    # ==================================================

    for impression_id, group in grouped_candidates:

        impression_id = str(
            impression_id
        )

        # ----------------------------------------------
        # Retrieve history
        # ----------------------------------------------

        history = history_lookup.get(
            impression_id,
            [],
        )

        if not history:

            skip_counts[
                "no_history"
            ] += 1

            continue

        # ----------------------------------------------
        # Build semantic user embedding
        # ----------------------------------------------

        try:

            user_embedding = build_user_embedding(
                history=history,
                article_embeddings=article_embeddings,
                article_ids=article_ids,
                max_history=max_history,
            )

        except ValueError as exc:

            message = str(exc).lower()

            if (
                "unknown" in message
                or "article" in message
                or "history" in message
            ):

                skip_counts[
                    "unknown_history_articles"
                ] += 1

            else:

                skip_counts[
                    "zero_user_embedding"
                ] += 1

            continue

        # ----------------------------------------------
        # Normalize user embedding
        # ----------------------------------------------

        user_norm = np.linalg.norm(
            user_embedding
        )

        if user_norm == 0:

            skip_counts[
                "zero_user_embedding"
            ] += 1

            continue

        normalized_user = (
            user_embedding
            /
            user_norm
        )

        # ----------------------------------------------
        # Extract valid candidate rows
        #
        # This replaces the previous Python loop over
        # group.itertuples().
        # ----------------------------------------------

        candidate_indices = (
            group["_embedding_index"]
            .to_numpy()
        )

        valid_mask = (
            pd.notna(candidate_indices)
        )

        if not valid_mask.any():

            skip_counts[
                "no_valid_candidates"
            ] += 1

            continue

        candidate_rows = (
            candidate_indices[
                valid_mask
            ]
            .astype(
                np.int64,
                copy=False,
            )
        )

        clicked_flags = (
            group.loc[
                valid_mask,
                "_clicked",
            ]
            .to_numpy(
                dtype=np.int8,
            )
        )

        # ----------------------------------------------
        # Score candidates
        # ----------------------------------------------

        candidate_vectors = (
            normalized_embeddings[
                candidate_rows
            ]
        )

        scores = (
            candidate_vectors
            @
            normalized_user
        )

        # ----------------------------------------------
        # Rank candidates
        # ----------------------------------------------

        ranking = np.argsort(
            -scores
        )

        ranked_clicked = (
            clicked_flags[
                ranking
            ]
        )

        evaluated_impressions += 1

        # ----------------------------------------------
        # Hit@K
        # ----------------------------------------------

        for k in ks:

            top_k = ranked_clicked[
                :k
            ]

            if np.any(
                top_k == 1
            ):

                hit_counts[k] += 1

        # ----------------------------------------------
        # MRR
        #
        # Find first relevant rank using NumPy.
        # ----------------------------------------------

        relevant_positions = np.flatnonzero(
            ranked_clicked == 1
        )

        if len(relevant_positions) > 0:

            first_relevant_rank = (
                int(
                    relevant_positions[0]
                )
                + 1
            )

            reciprocal_rank_sum += (
                1.0
                /
                first_relevant_rank
            )

    # ==================================================
    # Final validation
    # ==================================================

    skipped_impressions = (
        total_impressions
        - evaluated_impressions
    )

    counted_skips = sum(
        skip_counts.values()
    )

    if counted_skips != skipped_impressions:

        raise RuntimeError(
            "Skip counters do not match the total "
            "number of skipped impressions. "
            f"Expected {skipped_impressions}, "
            f"but counted {counted_skips}."
        )

    if evaluated_impressions == 0:

        raise ValueError(
            "No impressions could be evaluated."
        )

    # ==================================================
    # Final metrics
    # ==================================================

    metrics = {
        "impressions": float(
            evaluated_impressions
        ),

        "total_impressions": float(
            total_impressions
        ),

        "skipped_impressions": float(
            skipped_impressions
        ),

        "coverage": (
            evaluated_impressions
            /
            total_impressions
        ),

        "skip_rate": (
            skipped_impressions
            /
            total_impressions
        ),

        "MRR": (
            reciprocal_rank_sum
            /
            evaluated_impressions
        ),
    }

    for k in ks:

        metrics[
            f"Hit@{k}"
        ] = (
            hit_counts[k]
            /
            evaluated_impressions
        )

    # ==================================================
    # Skip reason counts
    # ==================================================

    for reason, count in skip_counts.items():

        metrics[
            f"skip_{reason}"
        ] = float(count)

    return metrics