from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.mind_bm25 import build_mind_bm25
from src.retrieval.mind_query import MINDQueryBuilder
from src.retrieval.mind_semantic import build_user_embedding


def build_hybrid_candidate_scores(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    bm25_retriever,
    article_embeddings: np.ndarray,
    article_ids: np.ndarray,
    behavioral_scores: pd.DataFrame,
    max_history: int = 10,
) -> pd.DataFrame:
    """
    Build an aligned candidate-level score table for hybrid ranking.

    Every row corresponds to one MIND candidate.

    Output columns:

        impression_id
        user_id
        timestamp
        article_id
        clicked
        bm25_score
        semantic_score
        behavioral_score

    The MIND candidate set is the authoritative candidate universe.
    """

    # ==================================================
    # Validate candidate data
    # ==================================================

    required_candidate_columns = {
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "clicked",
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

    # ==================================================
    # Validate history
    # ==================================================

    required_history_columns = {
        "impression_id",
        "article_id",
        "history_position",
    }

    missing = (
        required_history_columns
        - set(history.columns)
    )

    if missing:
        raise ValueError(
            "History is missing required columns: "
            f"{sorted(missing)}"
        )

    # ==================================================
    # Validate article metadata
    # ==================================================

    required_article_columns = {
        "article_id",
        "title",
    }

    missing = (
        required_article_columns
        - set(articles.columns)
    )

    if missing:
        raise ValueError(
            "Articles are missing required columns: "
            f"{sorted(missing)}"
        )

    # ==================================================
    # Validate behavioural scores
    # ==================================================

    required_behavioral_columns = {
        "impression_id",
        "article_id",
        "behavioral_score",
    }

    missing = (
        required_behavioral_columns
        - set(behavioral_scores.columns)
    )

    if missing:
        raise ValueError(
            "Behavioural scores are missing required "
            f"columns: {sorted(missing)}"
        )

    # ==================================================
    # Normalize identifiers
    # ==================================================

    candidates = candidates.copy()

    candidates["impression_id"] = (
        candidates["impression_id"].astype(str)
    )

    candidates["article_id"] = (
        candidates["article_id"].astype(str)
    )

    history = history.copy()

    history["impression_id"] = (
        history["impression_id"].astype(str)
    )

    history["article_id"] = (
        history["article_id"].astype(str)
    )

    articles = articles.copy()

    articles["article_id"] = (
        articles["article_id"].astype(str)
    )

    behavioral_scores = behavioral_scores.copy()

    behavioral_scores["impression_id"] = (
        behavioral_scores["impression_id"].astype(str)
    )

    behavioral_scores["article_id"] = (
        behavioral_scores["article_id"].astype(str)
    )

    # ==================================================
    # Build ordered history lookup
    # ==================================================

    history = history.sort_values(
        [
            "impression_id",
            "history_position",
        ]
    )

    history_lookup = (
        history
        .groupby("impression_id")["article_id"]
        .apply(list)
        .to_dict()
    )

    # ==================================================
    # Build lexical query builder
    # ==================================================

    query_builder = MINDQueryBuilder(
        articles=articles,
        text_field="title",
        max_history=max_history,
        max_query_terms=80,
    )

    # ==================================================
    # Prepare semantic embedding lookup
    # ==================================================

    article_embeddings = np.asarray(
        article_embeddings,
        dtype=np.float32,
    )

    article_ids = np.asarray(
        article_ids,
        dtype=str,
    )

    if article_embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings must be a 2D array."
        )

    if len(article_ids) != article_embeddings.shape[0]:
        raise ValueError(
            "article_ids and article_embeddings must "
            "have the same number of rows."
        )

    embedding_norms = np.linalg.norm(
        article_embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(embedding_norms == 0):
        raise ValueError(
            "article_embeddings contain zero vectors."
        )

    normalized_embeddings = (
        article_embeddings
        / embedding_norms
    )

    article_to_embedding_index = {
        str(article_id): index
        for index, article_id
        in enumerate(article_ids)
    }

    # ==================================================
    # Prepare behavioural lookup
    # ==================================================

    behavioral_key = [
        "impression_id",
        "article_id",
    ]

    if behavioral_scores.duplicated(
        behavioral_key
    ).any():
        raise ValueError(
            "Behavioural scores contain duplicate "
            "(impression_id, article_id) pairs."
        )

    behavioral_lookup = (
        behavioral_scores
        .set_index(behavioral_key)["behavioral_score"]
        .to_dict()
    )

    # ==================================================
    # Score impressions
    # ==================================================

    result_frames = []

    for impression_id, group in candidates.groupby(
        "impression_id",
        sort=False,
    ):

        history_ids = history_lookup.get(
            impression_id,
            [],
        )

        # --------------------------------------------------
        # Build lexical query
        # --------------------------------------------------

        query = query_builder.build_query(
            history_ids
        )

        candidate_ids = (
            group["article_id"]
            .astype(str)
            .tolist()
        )

        bm25_scores = dict(
            bm25_retriever.score_candidates(
                query=query,
                article_ids=candidate_ids,
            )
        )

        # --------------------------------------------------
        # Build semantic user embedding
        # --------------------------------------------------

        semantic_scores = {
            article_id: 0.0
            for article_id in candidate_ids
        }

        if history_ids:

            try:

                user_embedding = build_user_embedding(
                    history=history_ids,
                    article_embeddings=article_embeddings,
                    article_ids=article_ids,
                    max_history=max_history,
                )

                user_norm = np.linalg.norm(
                    user_embedding
                )

                if user_norm > 0:

                    user_embedding = (
                        user_embedding
                        / user_norm
                    )

                    valid_candidates = [
                        article_id
                        for article_id in candidate_ids
                        if article_id
                        in article_to_embedding_index
                    ]

                    if valid_candidates:

                        candidate_indices = [
                            article_to_embedding_index[
                                article_id
                            ]
                            for article_id
                            in valid_candidates
                        ]

                        candidate_vectors = (
                            normalized_embeddings[
                                candidate_indices
                            ]
                        )

                        scores = (
                            candidate_vectors
                            @
                            user_embedding
                        )

                        semantic_scores.update(
                            {
                                article_id: float(score)
                                for article_id, score
                                in zip(
                                    valid_candidates,
                                    scores,
                                )
                            }
                        )

            except ValueError:
                # Preserve deterministic zero fallback for
                # impressions whose history cannot produce
                # a valid semantic representation.
                pass

        # --------------------------------------------------
        # Build aligned impression frame
        # --------------------------------------------------

        frame = group[
            [
                "impression_id",
                "user_id",
                "timestamp",
                "article_id",
                "clicked",
            ]
        ].copy()

        frame["article_id"] = (
            frame["article_id"].astype(str)
        )

        frame["bm25_score"] = (
            frame["article_id"]
            .map(bm25_scores)
            .fillna(0.0)
            .astype(float)
        )

        frame["semantic_score"] = (
            frame["article_id"]
            .map(semantic_scores)
            .fillna(0.0)
            .astype(float)
        )

        frame["behavioral_score"] = [
            float(
                behavioral_lookup.get(
                    (
                        impression_id,
                        article_id,
                    ),
                    0.0,
                )
            )
            for article_id
            in frame["article_id"]
        ]

        result_frames.append(
            frame
        )

    if not result_frames:
        raise ValueError(
            "No candidate impressions were found."
        )

    result = pd.concat(
        result_frames,
        ignore_index=True,
    )

    # ==================================================
    # Final validation
    # ==================================================

    score_columns = [
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    ]

    for column in score_columns:

        if result[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not np.all(
            np.isfinite(
                result[column].to_numpy()
            )
        ):
            raise ValueError(
                f"{column} contains non-finite values."
            )

    return result