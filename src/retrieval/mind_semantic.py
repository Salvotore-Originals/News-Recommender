from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.semantic import SemanticRetriever


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_mind_articles(
    articles_path: str | Path,
) -> pd.DataFrame:
    """
    Load the MIND article feature store.

    The semantic retriever uses the existing `text` column
    created during Phase 1.
    """

    articles = pd.read_parquet(
        articles_path
    )

    required_columns = {
        "article_id",
        "text",
    }

    missing = (
        required_columns
        - set(articles.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required article columns: "
            f"{sorted(missing)}"
        )

    articles = articles.copy()

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
    )

    articles["text"] = (
        articles["text"]
        .fillna("")
        .astype(str)
    )

    if (
        articles["text"]
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Article text contains empty documents."
        )

    return articles


def build_mind_semantic(
    articles_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> SemanticRetriever:
    """
    Build a semantic retriever for MIND articles.

    Parameters
    ----------
    articles_path:
        Path to the MIND article feature-store Parquet file.

    model_name:
        Sentence Transformer model used to generate embeddings.

    batch_size:
        Number of articles encoded per batch.

    show_progress_bar:
        Whether to display embedding progress.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    articles = load_mind_articles(
        articles_path
    )

    model = SentenceTransformer(
        model_name
    )

    embeddings = model.encode(
        articles["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    return SemanticRetriever(
        article_ids=articles["article_id"].tolist(),
        embeddings=embeddings,
    )


def encode_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Encode arbitrary texts using the selected Sentence Transformer.

    This helper will later be used for semantic user queries.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    model = SentenceTransformer(
        model_name
    )

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )
def build_user_embedding(
    history: Sequence[str],
    article_embeddings: np.ndarray,
    article_ids: Sequence[str],
    max_history: int = 10,
) -> np.ndarray:
    """
    Build a semantic user representation from clicked articles.

    The most recent `max_history` articles are used.
    Their embeddings are averaged to create one user vector.

    Parameters
    ----------
    history:
        Ordered article IDs from the user's click history.

    article_embeddings:
        Matrix of article embeddings with shape:
        (number_of_articles, embedding_dimension).

    article_ids:
        Article IDs corresponding to the rows of
        article_embeddings.

    max_history:
        Maximum number of recent history articles to use.

    Returns
    -------
    np.ndarray
        One-dimensional semantic user embedding.
    """

    if max_history <= 0:
        raise ValueError(
            "max_history must be greater than zero."
        )

    if not history:
        raise ValueError(
            "Cannot build user embedding from empty history."
        )

    embeddings = np.asarray(
        article_embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings must be a 2-dimensional array."
        )

    if len(article_ids) != embeddings.shape[0]:
        raise ValueError(
            "article_ids and article_embeddings must have "
            "the same number of rows."
        )

    # --------------------------------------------------
    # Build article ID -> embedding row mapping
    # --------------------------------------------------

    article_to_index = {
        str(article_id): index
        for index, article_id
        in enumerate(article_ids)
    }

    # --------------------------------------------------
    # Keep most recent history
    # --------------------------------------------------

    recent_history = list(history)[-max_history:]

    history_vectors = []

    for article_id in recent_history:

        index = article_to_index.get(
            str(article_id)
        )

        if index is None:
            continue

        history_vectors.append(
            embeddings[index]
        )

    if not history_vectors:
        raise ValueError(
            "None of the history article IDs were found "
            "in the semantic article index."
        )

    # --------------------------------------------------
    # Mean pooling
    # --------------------------------------------------

    user_embedding = np.mean(
        np.stack(history_vectors),
        axis=0,
    )

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    norm = np.linalg.norm(
        user_embedding
    )

    if norm == 0:
        raise ValueError(
            "Generated user embedding is a zero vector."
        )

    return (
        user_embedding / norm
    )


class SemanticRetriever:
    """
    Semantic article retriever using dense embeddings.

    Articles are represented by dense vectors and ranked
    using cosine similarity.
    """

    def __init__(
        self,
        article_ids: Sequence[str],
        article_embeddings: np.ndarray,
    ) -> None:

        embeddings = np.asarray(
            article_embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "article_embeddings must be a 2D array."
            )

        if len(article_ids) != embeddings.shape[0]:
            raise ValueError(
                "article_ids and article_embeddings must "
                "have the same number of rows."
            )

        if len(article_ids) == 0:
            raise ValueError(
                "Cannot build semantic index from empty corpus."
            )

        if not np.all(
            np.isfinite(embeddings)
        ):
            raise ValueError(
                "article_embeddings contain non-finite values."
            )

        # --------------------------------------------------
        # Normalize article embeddings once.
        #
        # This allows cosine similarity to become a
        # simple dot product during search.
        # --------------------------------------------------

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Article embeddings contain zero vectors."
            )

        self.embeddings = (
            embeddings / norms
        )

        self.article_ids = [
            str(article_id)
            for article_id in article_ids
        ]

        self.embedding_dimension = (
            embeddings.shape[1]
        )

    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "query_embedding must be a 1D vector."
            )

        if query.shape[0] != self.embedding_dimension:
            raise ValueError(
                "query embedding dimension does not match "
                "article embedding dimension."
            )

        query_norm = np.linalg.norm(
            query
        )

        if query_norm == 0:
            raise ValueError(
                "query_embedding must not be a zero vector."
            )

        # --------------------------------------------------
        # Normalize query
        # --------------------------------------------------

        query = (
            query / query_norm
        )

        # --------------------------------------------------
        # Cosine similarity
        #
        # Since both query and article vectors are
        # normalized:
        #
        # cosine_similarity = dot_product
        # --------------------------------------------------

        scores = (
            self.embeddings @ query
        )

        # --------------------------------------------------
        # Retrieve top-k efficiently
        # --------------------------------------------------

        k = min(
            top_k,
            len(scores),
        )

        if k < len(scores):

            candidate_indices = np.argpartition(
                -scores,
                k - 1,
            )[:k]

        else:

            candidate_indices = np.arange(
                len(scores)
            )

        # Sort selected candidates by score
        candidate_indices = candidate_indices[
            np.argsort(
                -scores[candidate_indices]
            )
        ]

        return [
            (
                self.article_ids[index],
                float(scores[index]),
            )
            for index in candidate_indices
        ]