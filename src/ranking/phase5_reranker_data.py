"""
Phase 5.1 — Retrieval-aware feature construction for the neural re-ranker.

This module augments the existing EB-NeRD A2 candidate feature table with:

    - BM25 score
    - BM25 rank
    - Semantic score
    - Semantic rank
    - Reciprocal Rank Fusion (RRF) score

Important design constraint
---------------------------
This module does NOT generate a new candidate universe.

It scores the official EB-NeRD impression candidates so that Phase 5
can be compared fairly against the existing A2 baseline models.

Point-in-time rule
------------------
For an impression at time T, only history events with:

    history.timestamp < T

may be used.

Existing BM25 and Semantic implementations are reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.retrieval.ebnerd_semantic import (
    build_query_embedding,
    score_candidates_vectorized,
)

from src.retrieval.ebnerd_bm25 import (
    build_bm25,
    build_candidate_bm25_statistics,
    build_history_lookup as build_bm25_history_lookup,
    build_query_from_history,
    score_candidates_restricted,
)

# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
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

DEFAULT_CANDIDATE_PATH = (
    FEATURE_ROOT / "train_candidates.parquet"
)

DEFAULT_HISTORY_PATH = (
    PROCESSED_ROOT / "train_user_history.parquet"
)

DEFAULT_ARTICLE_PATH = (
    FEATURE_ROOT / "articles.parquet"
)

DEFAULT_EMBEDDING_PATH = (
    EMBEDDING_ROOT / "article_embeddings.npy"
)

DEFAULT_EMBEDDING_IDS_PATH = (
    EMBEDDING_ROOT / "article_ids.parquet"
)

DEFAULT_OUTPUT_PATH = (
    FEATURE_ROOT / "train_phase5_reranker.parquet"
)


# ============================================================================
# Configuration
# ============================================================================

RRF_K = 60
MAX_HISTORY = 10

RETRIEVAL_FEATURE_COLUMNS = [
    "bm25_score",
    "bm25_rank",
    "semantic_score",
    "semantic_rank",
    "rrf_score",
]


@dataclass(frozen=True)
class Phase5Config:
    """Configuration for Phase 5.1."""

    candidate_path: Path = DEFAULT_CANDIDATE_PATH
    history_path: Path = DEFAULT_HISTORY_PATH
    article_path: Path = DEFAULT_ARTICLE_PATH

    embedding_path: Path = DEFAULT_EMBEDDING_PATH
    embedding_ids_path: Path = DEFAULT_EMBEDDING_IDS_PATH

    output_path: Path = DEFAULT_OUTPUT_PATH

    rrf_k: int = RRF_K
    max_history: int = MAX_HISTORY

    # None means all impressions.
    max_impressions: Optional[int] = None


# ============================================================================
# Validation
# ============================================================================


def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    name: str,
) -> None:
    """Raise a clear error when required columns are absent."""

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def _validate_candidate_table(
    candidates: pd.DataFrame,
) -> None:
    """Validate the existing A2 candidate feature table."""

    required = [
        "impression_id",
        "article_id",
        "user_id",
        "impression_time",
        "clicked",
    ]

    _require_columns(
        candidates,
        required,
        "Candidate feature table",
    )

    if candidates["impression_id"].isna().any():
        raise ValueError(
            "Candidate table contains missing impression_id."
        )

    if candidates["article_id"].isna().any():
        raise ValueError(
            "Candidate table contains missing article_id."
        )

    if candidates["user_id"].isna().any():
        raise ValueError(
            "Candidate table contains missing user_id."
        )

    if candidates["impression_time"].isna().any():
        raise ValueError(
            "Candidate table contains missing impression_time."
        )

    if candidates["clicked"].isna().any():
        raise ValueError(
            "Candidate table contains missing click labels."
        )

    invalid_labels = set(
        candidates["clicked"].dropna().unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            "Invalid clicked labels found: "
            f"{sorted(invalid_labels)}"
        )

    duplicate_pairs = candidates.duplicated(
        subset=["impression_id", "article_id"]
    )

    if duplicate_pairs.any():
        raise ValueError(
            "Candidate table contains duplicate "
            "(impression_id, article_id) pairs."
        )


def _validate_history_table(
    history: pd.DataFrame,
) -> None:
    """Validate the user-history table."""

    required = [
        "user_id",
        "article_id",
        "timestamp",
    ]

    _require_columns(
        history,
        required,
        "History table",
    )

    if history["user_id"].isna().any():
        raise ValueError(
            "History table contains missing user_id."
        )

    if history["article_id"].isna().any():
        raise ValueError(
            "History table contains missing article_id."
        )

    if history["timestamp"].isna().any():
        raise ValueError(
            "History table contains missing timestamp."
        )


def _validate_articles(
    articles: pd.DataFrame,
) -> None:
    """Validate the article corpus."""

    required = [
        "article_id",
        "text",
    ]

    _require_columns(
        articles,
        required,
        "Article table",
    )

    if articles["article_id"].duplicated().any():
        raise ValueError(
            "Article table contains duplicate article_id values."
        )


# ============================================================================
# Deterministic ranking
# ============================================================================


def _rank_scores(
    scores: Sequence[Tuple[int, float]],
) -> Dict[int, int]:
    """Convert already-ranked retrieval results into 1-based ranks."""

    return {
        int(article_id): rank
        for rank, (article_id, _) in enumerate(
            scores,
            start=1,
        )
    }


def _scores_to_dict(
    scores: Sequence[Tuple[int, float]],
) -> Dict[int, float]:
    """Convert retrieval output into article_id -> score."""

    return {
        int(article_id): float(score)
        for article_id, score in scores
    }


def _calculate_rrf(
    bm25_rank: int,
    semantic_rank: int,
    k: int,
) -> float:
    """
    Calculate Reciprocal Rank Fusion.

        RRF = 1 / (k + BM25_rank)
            + 1 / (k + semantic_rank)

    A rank of zero means that the channel did not return the candidate.
    """

    score = 0.0

    if bm25_rank > 0:
        score += 1.0 / (
            k + bm25_rank
        )

    if semantic_rank > 0:
        score += 1.0 / (
            k + semantic_rank
        )

    return score


# ============================================================================
# History preparation
# ============================================================================


def _prepare_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and deterministically sort historical events."""

    history = history.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="raise",
    )

    history = history.sort_values(
        [
            "user_id",
            "timestamp",
            "article_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return history
# ============================================================================
# Embeddings
# ============================================================================


def _load_embeddings(
    embedding_path: Path,
    embedding_ids_path: Path,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load article embeddings and construct article_id -> row mapping.
    """

    embeddings = np.load(
        embedding_path
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected a 2-D embedding matrix, got "
            f"shape {embeddings.shape}."
        )

    embedding_ids = pd.read_parquet(
        embedding_ids_path
    )

    _require_columns(
        embedding_ids,
        ["article_id"],
        "Embedding ID table",
    )

    ids = embedding_ids[
        "article_id"
    ].to_numpy()

    if len(ids) != len(embeddings):
        raise ValueError(
            "Embedding count does not match embedding ID count: "
            f"{len(embeddings)} vs {len(ids)}."
        )

    if pd.Series(ids).duplicated().any():
        raise ValueError(
            "Embedding ID table contains duplicate article IDs."
        )

    article_to_index = {
        int(article_id): index
        for index, article_id in enumerate(ids)
    }

    return embeddings, article_to_index


# ============================================================================
# Per-impression retrieval
# ============================================================================


def _build_impression_retrieval_features(
    impression_candidates: pd.DataFrame,
    bm25_history_events: list,
    semantic_history_events: list,
    bm25_article_to_index: Dict[int, int],
    bm25_statistics: dict,
    semantic_article_to_index: Dict[int, int],
    semantic_embeddings: np.ndarray,
    rrf_k: int,
    max_history: int,
) -> pd.DataFrame:
    """
    Generate retrieval features for one official impression.

    Candidate universe is preserved exactly.
    """

    impression_id = int(
        impression_candidates[
            "impression_id"
        ].iloc[0]
    )

    candidate_ids = sorted(
        {
            int(article_id)
            for article_id in impression_candidates[
                "article_id"
            ]
        }
    )

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    bm25_query = build_query_from_history(
        bm25_history_events,
        impression_candidates[
            "impression_time"
        ].iloc[0],
        max_articles=max_history,
    )

    bm25_results = score_candidates_restricted(
        query_tokens=bm25_query,
        candidate_ids=candidate_ids,
        article_to_index=bm25_article_to_index,
        bm25_statistics=bm25_statistics,
    )

    bm25_scores = _scores_to_dict(
        bm25_results
    )

    bm25_ranks = _rank_scores(
        bm25_results
    )

    # ------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------

    impression_time = impression_candidates[
        "impression_time"
    ].iloc[0]

    semantic_query = build_query_embedding(
        history_events=semantic_history_events,
        impression_time=impression_time,
        article_to_index=semantic_article_to_index,
        article_embeddings=semantic_embeddings,
        max_articles=max_history,
    )

    semantic_results = score_candidates_vectorized(
        query_embedding=semantic_query,
        candidate_ids=candidate_ids,
        article_to_index=semantic_article_to_index,
        article_embeddings=semantic_embeddings,
    )

    semantic_scores = _scores_to_dict(
        semantic_results
    )

    semantic_ranks = _rank_scores(
        semantic_results
    )

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    rows = []

    for article_id in candidate_ids:

        bm25_score = bm25_scores.get(
            article_id,
            0.0,
        )

        semantic_score = semantic_scores.get(
            article_id,
            0.0,
        )

        bm25_rank = bm25_ranks.get(
            article_id,
            0,
        )

        semantic_rank = semantic_ranks.get(
            article_id,
            0,
        )

        rows.append(
            {
                "impression_id": impression_id,
                "article_id": article_id,
                "bm25_score": bm25_score,
                "bm25_rank": bm25_rank,
                "semantic_score": semantic_score,
                "semantic_rank": semantic_rank,
                "rrf_score": _calculate_rrf(
                    bm25_rank,
                    semantic_rank,
                    rrf_k,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Main builder
# ============================================================================


def build_phase5_reranker_features(
    config: Phase5Config,
) -> pd.DataFrame:
    """
    Build the complete Phase 5.1 retrieval-aware candidate table.
    """

    # ------------------------------------------------------------------
    # 1. Load existing A2 candidate features.
    # ------------------------------------------------------------------

    print("[1/7] Loading candidate features...")

    candidates = pd.read_parquet(
        config.candidate_path
    )

    _validate_candidate_table(
        candidates
    )

    candidates["impression_time"] = pd.to_datetime(
        candidates["impression_time"],
        errors="raise",
    )

    print(
        f"       Rows: {len(candidates):,}"
    )

    print(
        "       Impressions: "
        f"{candidates['impression_id'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # 2. Load history.
    # ------------------------------------------------------------------

    print("[2/7] Loading user history...")

    history = pd.read_parquet(
        config.history_path
    )

    _validate_history_table(
        history
    )

    history = _prepare_history(
        history
    )

    print(
        f"       History rows: {len(history):,}"
    )

    # ------------------------------------------------------------------
    # 3. Load article corpus.
    # ------------------------------------------------------------------

    print("[3/7] Loading article corpus...")

    articles = pd.read_parquet(
        config.article_path
    )

    _validate_articles(
        articles
    )

    print(
        f"       Articles: {len(articles):,}"
    )

    # ------------------------------------------------------------------
    # 4. Build BM25 statistics.
    # ------------------------------------------------------------------

    print("[4/7] Building BM25 statistics...")

    bm25, bm25_article_to_index = build_bm25(
    articles
    )

    bm25_statistics = build_candidate_bm25_statistics(
        articles,
        bm25=bm25,
    )

    print("       BM25 statistics ready.")

    # ------------------------------------------------------------------
    # 5. Load semantic embeddings.
    # ------------------------------------------------------------------

    print("[5/7] Loading semantic embeddings...")

    embeddings, semantic_article_to_index = (
        _load_embeddings(
            config.embedding_path,
            config.embedding_ids_path,
        )
    )

    print(
        f"       Embedding shape: {embeddings.shape}"
    )

    # ------------------------------------------------------------------
    # Build history lookups.
    #
    # These calls intentionally reuse the existing Phase 2/3 APIs.
    # The lookups are retained here for validation / compatibility.
    # ------------------------------------------------------------------

    history_lookup = build_bm25_history_lookup(
        history,
        articles,
    )

    semantic_history_lookup = {
        user_id: [(timestamp, article_id) for timestamp, article_id, _ in events]
        for user_id, events in history_lookup.items()
    }
    # ------------------------------------------------------------------
    # 6. Optional smoke-test restriction.
    #
    # Select complete impressions, never arbitrary candidate rows.
    # ------------------------------------------------------------------

    selected_impressions = (
        candidates[
            "impression_id"
        ]
        .drop_duplicates()
        .sort_values()
    )

    if config.max_impressions is not None:

        if config.max_impressions <= 0:
            raise ValueError(
                "max_impressions must be positive."
            )

        selected_impressions = (
            selected_impressions.iloc[
                : config.max_impressions
            ]
        )

        selected_set = set(
            selected_impressions.tolist()
        )

        candidates = candidates[
            candidates[
                "impression_id"
            ].isin(selected_set)
        ].copy()

    print(
        f"       Impressions to process: "
        f"{candidates['impression_id'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # Process impressions.
    # ------------------------------------------------------------------

    print("[6/7] Generating retrieval features...")

    retrieval_parts: List[pd.DataFrame] = []

    grouped = candidates.groupby(
        "impression_id",
        sort=True,
    )

    total_impressions = (
        candidates[
            "impression_id"
        ].nunique()
    )

    for position, (
        impression_id,
        impression_candidates,
    ) in enumerate(
        grouped,
        start=1,
    ):

        user_id = int(
            impression_candidates[
                "user_id"
            ].iloc[0]
        )

        impression_time = (
            impression_candidates[
                "impression_time"
            ].iloc[0]
        )

        bm25_history_events = history_lookup.get(
            user_id,
            [],
        )

        semantic_history_events = semantic_history_lookup.get(
            user_id,
            [],
        )

        retrieval_features = _build_impression_retrieval_features(
            impression_candidates=impression_candidates,
            bm25_history_events=bm25_history_events,
            semantic_history_events=semantic_history_events,
            bm25_article_to_index=bm25_article_to_index,
            bm25_statistics=bm25_statistics,
            semantic_article_to_index=semantic_article_to_index,
            semantic_embeddings=embeddings,
            rrf_k=config.rrf_k,
            max_history=config.max_history,
        )

        retrieval_parts.append(
            retrieval_features
        )

        if (
            position == 1
            or position % 1000 == 0
            or position == total_impressions
        ):
            print(
                f"       Processed "
                f"{position:,}/"
                f"{total_impressions:,} impressions"
            )

    if not retrieval_parts:
        raise ValueError(
            "No retrieval features were generated."
        )

    retrieval_features = pd.concat(
        retrieval_parts,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 7. Merge with existing A2 features.
    # ------------------------------------------------------------------

    print("[7/7] Merging retrieval features...")

    result = candidates.merge(
        retrieval_features,
        on=[
            "impression_id",
            "article_id",
        ],
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    for column in RETRIEVAL_FEATURE_COLUMNS:

        if column not in result.columns:
            raise ValueError(
                f"Missing retrieval feature: {column}"
            )

        missing = int(
            result[column].isna().sum()
        )

        if missing:
            raise ValueError(
                f"Feature '{column}' contains "
                f"{missing:,} missing values."
            )

    # Candidate row count must not change.
    if len(result) != len(candidates):
        raise ValueError(
            "Candidate row count changed during Phase 5.1."
        )

    # Candidate identity must not change.
    original_keys = (
        candidates[
            [
                "impression_id",
                "article_id",
            ]
        ]
        .sort_values(
            [
                "impression_id",
                "article_id",
            ]
        )
        .reset_index(drop=True)
    )

    result_keys = (
        result[
            [
                "impression_id",
                "article_id",
            ]
        ]
        .sort_values(
            [
                "impression_id",
                "article_id",
            ]
        )
        .reset_index(drop=True)
    )

    if not original_keys.equals(
        result_keys
    ):
        raise ValueError(
            "Candidate identity changed during Phase 5.1."
        )

    result = result.sort_values(
        [
            "impression_id",
            "article_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return result


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """Build and save the Phase 5.1 feature table."""

    config = Phase5Config()

    print("=" * 70)
    print(
        "PHASE 5.1 — RETRIEVAL-AWARE "
        "RE-RANKER DATA"
    )
    print("=" * 70)

    print(
        f"Candidate file : "
        f"{config.candidate_path}"
    )

    print(
        f"History file   : "
        f"{config.history_path}"
    )

    print(
        f"Article file   : "
        f"{config.article_path}"
    )

    print(
        f"Embedding file : "
        f"{config.embedding_path}"
    )

    print(
        f"Output file    : "
        f"{config.output_path}"
    )

    print()

    result = (
        build_phase5_reranker_features(
            config
        )
    )

    config.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        config.output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 5.1 COMPLETE")
    print("=" * 70)

    print(
        f"Rows        : "
        f"{len(result):,}"
    )

    print(
        f"Impressions : "
        f"{result['impression_id'].nunique():,}"
    )

    print(
        f"Clicks      : "
        f"{int(result['clicked'].sum()):,}"
    )

    print(
        f"Output      : "
        f"{config.output_path}"
    )

    print()
    print("Retrieval features:")

    for column in RETRIEVAL_FEATURE_COLUMNS:
        print(f"  - {column}")

    print()
    print(
        result[
            RETRIEVAL_FEATURE_COLUMNS
        ].describe()
    )


if __name__ == "__main__":
    main()