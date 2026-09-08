from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi

from src.retrieval.ebnerd_bm25 import tokenize

REQUIRED_ARTICLE_COLUMNS = {
    "article_id",
    "published_time",
}


def validate_article_corpus(
    articles: pd.DataFrame,
) -> None:
    """
    Validate the minimum article columns required for
    corpus-level candidate generation.
    """

    missing = REQUIRED_ARTICLE_COLUMNS - set(
        articles.columns
    )

    if missing:
        raise ValueError(
            "Article corpus is missing required columns: "
            f"{sorted(missing)}"
        )


def filter_temporally_eligible_articles(
    articles: pd.DataFrame,
    impression_time,
) -> pd.DataFrame:
    """
    Return only articles that were available by the
    recommendation/impression time.

    Eligibility rule:

        published_time <= impression_time

    Articles with missing publication timestamps are excluded
    because their temporal availability cannot be established.
    """

    validate_article_corpus(articles)

    result = articles.copy()

    result["published_time"] = pd.to_datetime(
        result["published_time"],
        errors="coerce",
    )

    impression_time = pd.Timestamp(
        impression_time
    )

    eligible = (
        result["published_time"].notna()
        & (
            result["published_time"]
            <= impression_time
        )
    )

    result = result.loc[eligible].copy()

    return result.reset_index(drop=True)

def build_bm25_corpus(
    articles: pd.DataFrame,
) -> tuple[BM25Okapi, list[str]]:
    """
    Build a BM25 index over the supplied EB-NeRD article corpus.

    EB-NeRD uses the canonical `text` field as the article
    representation.

    Returns:
        bm25:
            Corpus-level BM25 index.

        article_ids:
            Article IDs in exactly the same order as the
            BM25 corpus documents.
    """

    required = {
        "article_id",
        "text",
    }

    missing = required - set(
        articles.columns
    )

    if missing:
        raise ValueError(
            "Article corpus is missing required columns: "
            f"{sorted(missing)}"
        )

    article_ids = (
        articles["article_id"]
        .astype(str)
        .tolist()
    )

    documents = (
        articles["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=1.5,
        b=0.75,
    )

    return bm25, article_ids

def retrieve_bm25_candidates(
    bm25: BM25Okapi,
    article_ids: list[str],
    query: str,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Retrieve top-K articles from the entire BM25 corpus.

    Returns:
        List of (article_id, bm25_score), ordered by descending
        score with deterministic article-ID tie-breaking.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if not query or not query.strip():
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    scored = [
        (
            article_ids[index],
            float(score),
        )
        for index, score in enumerate(scores)
    ]

    scored.sort(
        key=lambda x: (
            -x[1],
            str(x[0]),
        )
    )

    return scored[:top_k]

def build_semantic_corpus(
    article_ids: list[str],
    embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    Prepare the article embedding corpus for corpus-level
    semantic retrieval.

    Article IDs and embeddings must have identical row ordering.

    The returned embeddings are L2-normalized so that a dot
    product is equivalent to cosine similarity.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2-dimensional array."
        )

    if len(article_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of article IDs must match the number "
            "of embedding rows."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "embeddings contain non-finite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "embeddings contain zero vectors."
        )

    normalized_embeddings = (
        embeddings / norms
    ).astype(
        np.float32,
        copy=False,
    )

    return article_ids, normalized_embeddings

def retrieve_semantic_candidates(
    article_ids: list[str],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Retrieve top-K articles from the entire semantic corpus.

    Returns:
        (article_id, cosine_similarity), ordered by descending
        similarity with deterministic article-ID tie-breaking.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.ndim != 1:
        raise ValueError(
            "query_embedding must be a 1-dimensional vector."
        )

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            "query_embedding dimension does not match "
            "article embedding dimension."
        )

    if not np.isfinite(query_embedding).all():
        raise ValueError(
            "query_embedding contains non-finite values."
        )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        raise ValueError(
            "query_embedding must not be a zero vector."
        )

    query_embedding = (
        query_embedding / query_norm
    )

    similarities = (
        embeddings @ query_embedding
    )

    top_k = min(
        top_k,
        len(article_ids),
    )

    ranked_indices = np.argsort(
        -similarities,
        kind="stable",
    )[:top_k]

    results = [
        (
            article_ids[index],
            float(similarities[index]),
        )
        for index in ranked_indices
    ]

    return results

def merge_retrieval_candidates(
    bm25_candidates: list[tuple[str, float]],
    semantic_candidates: list[tuple[str, float]],
) -> pd.DataFrame:
    """
    Merge BM25 and semantic retrieval results.

    Each candidate appears exactly once.

    The resulting DataFrame preserves:
        - article_id
        - bm25_rank
        - bm25_score
        - semantic_rank
        - semantic_score
        - source

    Source values:
        "bm25"
        "semantic"
        "both"
    """

    records = {}

    # --------------------------------------------------------------
    # BM25 candidates
    # --------------------------------------------------------------

    for rank, (article_id, score) in enumerate(
        bm25_candidates,
        start=1,
    ):
        article_id = str(article_id)

        records.setdefault(
            article_id,
            {
                "article_id": article_id,
                "bm25_rank": None,
                "bm25_score": None,
                "semantic_rank": None,
                "semantic_score": None,
            },
        )

        records[article_id]["bm25_rank"] = rank
        records[article_id]["bm25_score"] = float(score)

    # --------------------------------------------------------------
    # Semantic candidates
    # --------------------------------------------------------------

    for rank, (article_id, score) in enumerate(
        semantic_candidates,
        start=1,
    ):
        article_id = str(article_id)

        records.setdefault(
            article_id,
            {
                "article_id": article_id,
                "bm25_rank": None,
                "bm25_score": None,
                "semantic_rank": None,
                "semantic_score": None,
            },
        )

        records[article_id]["semantic_rank"] = rank
        records[article_id]["semantic_score"] = float(score)

    # --------------------------------------------------------------
    # Construct result
    # --------------------------------------------------------------

    result = pd.DataFrame(
        list(records.values())
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "bm25_rank",
                "bm25_score",
                "semantic_rank",
                "semantic_score",
                "source",
            ]
        )

    result["source"] = "both"

    result.loc[
        result["bm25_rank"].isna()
        & result["semantic_rank"].notna(),
        "source",
    ] = "semantic"

    result.loc[
        result["bm25_rank"].notna()
        & result["semantic_rank"].isna(),
        "source",
    ] = "bm25"

    return result.reset_index(drop=True)

def apply_rrf(
    candidates: pd.DataFrame,
    k: int = 60,
) -> pd.DataFrame:
    """
    Apply Reciprocal Rank Fusion to BM25 and semantic rankings.

    Expected columns:
        article_id
        bm25_rank
        semantic_rank

    Returns a copy containing:
        article_id
        bm25_rank
        bm25_score
        semantic_rank
        semantic_score
        source
        rrf_score

    RRF formula:

        RRF(d) =
            1 / (k + bm25_rank(d))
            +
            1 / (k + semantic_rank(d))

    A missing rank contributes zero.
    """

    required = {
        "article_id",
        "bm25_rank",
        "semantic_rank",
    }

    missing = required - set(
        candidates.columns
    )

    if missing:
        raise ValueError(
            "Candidate table is missing required columns: "
            f"{sorted(missing)}"
        )

    if k <= 0:
        raise ValueError(
            "RRF k must be greater than zero."
        )

    result = candidates.copy()

    # Convert ranks to numeric so missing ranks are represented
    # consistently as NaN.
    bm25_ranks = pd.to_numeric(
        result["bm25_rank"],
        errors="coerce",
    )

    semantic_ranks = pd.to_numeric(
        result["semantic_rank"],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Compute each retrieval channel's RRF contribution.
    #
    # Missing ranks contribute zero.
    # --------------------------------------------------------------

    bm25_contribution = (
        1.0 / (k + bm25_ranks)
    ).fillna(0.0)

    semantic_contribution = (
        1.0 / (k + semantic_ranks)
    ).fillna(0.0)

    result["rrf_score"] = (
        bm25_contribution
        + semantic_contribution
    )

    # --------------------------------------------------------------
    # Deterministic ranking:
    #
    # 1. Highest RRF score first
    # 2. Article ID ascending for exact ties
    # --------------------------------------------------------------

    result = result.sort_values(
        [
            "rrf_score",
            "article_id",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    return result.reset_index(
        drop=True
    )

def validate_embedding_alignment(
    articles: pd.DataFrame,
    article_embeddings: np.ndarray,
) -> None:
    """
    Validate that article embeddings are aligned with the
    article feature store.

    Invariant:

        article_embeddings[i]
        corresponds to
        articles.iloc[i]["article_id"]
    """

    validate_article_corpus(articles)

    if article_embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings must be a 2-dimensional array."
        )

    if len(articles) != article_embeddings.shape[0]:
        raise ValueError(
            "Number of article rows must match the number "
            "of embedding rows."
        )

    if not np.isfinite(article_embeddings).all():
        raise ValueError(
            "article_embeddings contain non-finite values."
        )

    norms = np.linalg.norm(
        article_embeddings,
        axis=1,
    )

    if np.any(norms == 0):
        raise ValueError(
            "article_embeddings contain zero vectors."
        )

    article_ids = (
        articles["article_id"]
        .astype(str)
        .tolist()
    )

    if len(article_ids) != len(
        set(article_ids)
    ):
        raise ValueError(
            "Article IDs must be unique."
        )

def prepare_semantic_corpus(
    articles: pd.DataFrame,
    article_embeddings: np.ndarray,
    impression_time=None,
) -> tuple[list[str], np.ndarray]:
    """
    Prepare the semantic corpus while preserving exact
    article-ID ↔ embedding alignment.

    If impression_time is supplied, only articles satisfying:

        published_time <= impression_time

    are retained.
    """

    validate_embedding_alignment(
        articles,
        article_embeddings,
    )

    if impression_time is None:
        eligible_articles = articles.copy()
        eligible_embeddings = article_embeddings.copy()

    else:
        published_time = pd.to_datetime(
            articles["published_time"],
            errors="coerce",
        )

        impression_time = pd.Timestamp(
            impression_time
        )

        mask = (
            published_time.notna()
            & (
                published_time
                <= impression_time
            )
        )

        eligible_articles = (
            articles.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

        eligible_embeddings = (
            article_embeddings[mask.to_numpy()]
        )

    article_ids = (
        eligible_articles["article_id"]
        .astype(str)
        .tolist()
    )

    _, normalized_embeddings = (
        build_semantic_corpus(
            article_ids,
            eligible_embeddings,
        )
    )

    return (
        article_ids,
        normalized_embeddings,
    )

class EBNeRDCandidateGenerator:
    """
    Reusable corpus-level EB-NeRD candidate generator.

    Expensive resources are built once:

        - temporally eligible article corpus
        - BM25 index
        - normalized semantic embeddings

    Each impression can then be queried efficiently.
    """

    def __init__(
        self,
        articles: pd.DataFrame,
        article_embeddings: np.ndarray,
        bm25_top_k: int = 100,
        semantic_top_k: int = 100,
        rrf_k: int = 60,
    ):
        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k must be greater than zero."
            )

        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than zero."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        validate_embedding_alignment(
            articles,
            article_embeddings,
        )

        self.articles = articles.copy()

        self.article_embeddings = (
            article_embeddings.copy()
        )

        self.bm25_top_k = bm25_top_k
        self.semantic_top_k = semantic_top_k
        self.rrf_k = rrf_k

        # ----------------------------------------------------------
        # Validate article IDs once.
        # ----------------------------------------------------------

        self.article_ids = (
            self.articles["article_id"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------------
        # Build BM25 once.
        # ----------------------------------------------------------

        (
            self.bm25,
            self.bm25_article_ids,
        ) = build_bm25_corpus(
            self.articles
        )

        # ----------------------------------------------------------
        # Normalize semantic embeddings once.
        # ----------------------------------------------------------

        (
            self.semantic_article_ids,
            self.semantic_embeddings,
        ) = build_semantic_corpus(
            self.article_ids,
            self.article_embeddings,
        )

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        impression_time=None,
        final_top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Retrieve candidates for one impression.

        If impression_time is supplied, articles published after
        that timestamp are excluded before retrieval.
        """

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        # ----------------------------------------------------------
        # Determine temporally eligible article rows.
        # ----------------------------------------------------------

        if impression_time is None:

            eligible_mask = np.ones(
                len(self.articles),
                dtype=bool,
            )

        else:

            published_time = pd.to_datetime(
                self.articles["published_time"],
                errors="coerce",
            )

            impression_timestamp = pd.Timestamp(
                impression_time
            )

            eligible_mask = (
                published_time.notna()
                & (
                    published_time
                    <= impression_timestamp
                )
            ).to_numpy()

        if not eligible_mask.any():

            return pd.DataFrame(
                columns=[
                    "article_id",
                    "bm25_rank",
                    "bm25_score",
                    "semantic_rank",
                    "semantic_score",
                    "source",
                    "rrf_score",
                ]
            )

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # We currently retrieve against the full pre-built corpus
        # and then apply the temporal eligibility mask.
        #
        # That is NOT safe because a future article could occupy
        # the top-K retrieval slots.
        #
        # Therefore we need a temporally restricted retrieval
        # implementation.
        # ----------------------------------------------------------

        eligible_article_ids = [
            article_id
            for article_id, eligible
            in zip(
                self.article_ids,
                eligible_mask,
            )
            if eligible
        ]

        eligible_indices = np.flatnonzero(
            eligible_mask
        )

        # ----------------------------------------------------------
        # BM25
        # ----------------------------------------------------------

        all_bm25_scores = self.bm25.get_scores(
            tokenize(query)
        )

        bm25_candidates = [
            (
                self.article_ids[index],
                float(all_bm25_scores[index]),
            )
            for index in eligible_indices
        ]

        bm25_candidates.sort(
            key=lambda x: (
                -x[1],
                str(x[0]),
            )
        )

        bm25_candidates = bm25_candidates[
            :self.bm25_top_k
        ]

        # ----------------------------------------------------------
        # Semantic
        # ----------------------------------------------------------

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm == 0:
            raise ValueError(
                "query_embedding must not be a zero vector."
            )

        query_embedding = (
            query_embedding / query_norm
        )

        eligible_embeddings = (
            self.semantic_embeddings[
                eligible_indices
            ]
        )

        semantic_scores = (
            eligible_embeddings
            @ query_embedding
        )

        semantic_candidates = [
            (
                self.article_ids[index],
                float(score),
            )
            for index, score
            in zip(
                eligible_indices,
                semantic_scores,
            )
        ]

        semantic_candidates.sort(
            key=lambda x: (
                -x[1],
                str(x[0]),
            )
        )

        semantic_candidates = (
            semantic_candidates[
                :self.semantic_top_k
            ]
        )

        # ----------------------------------------------------------
        # Merge
        # ----------------------------------------------------------

        merged = merge_retrieval_candidates(
            bm25_candidates,
            semantic_candidates,
        )

        if merged.empty:

            return merged.assign(
                rrf_score=pd.Series(
                    dtype="float64"
                )
            )

        # ----------------------------------------------------------
        # RRF
        # ----------------------------------------------------------

        fused = apply_rrf(
            merged,
            k=self.rrf_k,
        )

        return fused.head(
            min(
                final_top_k,
                len(fused),
            )
        ).reset_index(
            drop=True
        )

def generate_candidates(
    articles: pd.DataFrame,
    article_embeddings: np.ndarray,
    query: str,
    query_embedding: np.ndarray,
    bm25_top_k: int = 100,
    semantic_top_k: int = 100,
    final_top_k: int = 100,
    impression_time=None,
    rrf_k: int = 60,
) -> pd.DataFrame:
    """
    Generate corpus-level EB-NeRD candidates using BM25
    and semantic retrieval followed by RRF fusion.

    Pipeline:

        article corpus
            ↓
        temporal eligibility
            ↓
        BM25 corpus retrieval
            +
        semantic corpus retrieval
            ↓
        union + deduplication
            ↓
        RRF
            ↓
        final top-K
    """

    if bm25_top_k <= 0:
        raise ValueError(
            "bm25_top_k must be greater than zero."
        )

    if semantic_top_k <= 0:
        raise ValueError(
            "semantic_top_k must be greater than zero."
        )

    if final_top_k <= 0:
        raise ValueError(
            "final_top_k must be greater than zero."
        )

    # --------------------------------------------------------------
    # Validate embedding alignment before doing any retrieval.
    # --------------------------------------------------------------

    validate_embedding_alignment(
        articles,
        article_embeddings,
    )

    # --------------------------------------------------------------
    # Temporal eligibility
    # --------------------------------------------------------------

    if impression_time is not None:

        eligible_articles = (
            filter_temporally_eligible_articles(
                articles,
                impression_time,
            )
        )

        # Build the same row mask for embeddings.
        published_time = pd.to_datetime(
            articles["published_time"],
            errors="coerce",
        )

        impression_timestamp = pd.Timestamp(
            impression_time
        )

        embedding_mask = (
            published_time.notna()
            & (
                published_time
                <= impression_timestamp
            )
        )

        eligible_embeddings = (
            article_embeddings[
                embedding_mask.to_numpy()
            ]
        )

    else:

        eligible_articles = articles.copy()

        eligible_embeddings = (
            article_embeddings.copy()
        )

    if eligible_articles.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "bm25_rank",
                "bm25_score",
                "semantic_rank",
                "semantic_score",
                "source",
                "rrf_score",
            ]
        )

    # --------------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------------

    bm25, bm25_article_ids = (
        build_bm25_corpus(
            eligible_articles
        )
    )

    bm25_candidates = (
        retrieve_bm25_candidates(
            bm25,
            bm25_article_ids,
            query,
            top_k=bm25_top_k,
        )
    )

    # --------------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------------

    semantic_article_ids, semantic_embeddings = (
        build_semantic_corpus(
            eligible_articles[
                "article_id"
            ]
            .astype(str)
            .tolist(),
            eligible_embeddings,
        )
    )

    semantic_candidates = (
        retrieve_semantic_candidates(
            semantic_article_ids,
            semantic_embeddings,
            query_embedding,
            top_k=semantic_top_k,
        )
    )

    # --------------------------------------------------------------
    # Union + provenance
    # --------------------------------------------------------------

    merged = merge_retrieval_candidates(
        bm25_candidates,
        semantic_candidates,
    )

    if merged.empty:
        return merged.assign(
            rrf_score=pd.Series(
                dtype="float64"
            )
        )

    # --------------------------------------------------------------
    # RRF
    # --------------------------------------------------------------

    fused = apply_rrf(
        merged,
        k=rrf_k,
    )

    # --------------------------------------------------------------
    # Final candidate cutoff
    # --------------------------------------------------------------

    return fused.head(
        min(
            final_top_k,
            len(fused),
        )
    ).reset_index(
        drop=True
    )
