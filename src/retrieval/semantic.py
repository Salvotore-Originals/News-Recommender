from __future__ import annotations

from typing import Sequence

import numpy as np


class SemanticRetriever:
    """
    Dense semantic retriever using cosine similarity.

    Parameters
    ----------
    article_ids:
        Article IDs corresponding to each embedding row.

    embeddings:
        2D NumPy array of shape:
        (number_of_articles, embedding_dimension)

    Notes
    -----
    Embeddings are normalized internally so that cosine similarity
    can be calculated efficiently using a matrix-vector product.
    """

    def __init__(
        self,
        article_ids: Sequence[str],
        embeddings: np.ndarray,
    ) -> None:

        if len(article_ids) == 0:
            raise ValueError(
                "Cannot build semantic index from an empty corpus."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a 2-dimensional array."
            )

        if len(article_ids) != embeddings.shape[0]:
            raise ValueError(
                "article_ids and embeddings must have the same "
                "number of rows."
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "embeddings must contain only finite values."
            )

        self.article_ids = [
            str(article_id)
            for article_id in article_ids
        ]

        self.embeddings = self._normalize(
            embeddings
        )

        self.num_documents = embeddings.shape[0]
        self.embedding_dimension = embeddings.shape[1]

    # ======================================================
    # NORMALIZATION
    # ======================================================

    @staticmethod
    def _normalize(
        vectors: np.ndarray,
    ) -> np.ndarray:

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Embeddings must not contain zero vectors."
            )

        return vectors / norms

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
                "query_embedding must be a 1-dimensional array."
            )

        if query.shape[0] != self.embedding_dimension:
            raise ValueError(
                "query_embedding dimension does not match "
                "the article embedding dimension."
            )

        if not np.all(np.isfinite(query)):
            raise ValueError(
                "query_embedding must contain only finite values."
            )

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError(
                "query_embedding must not be a zero vector."
            )

        # Normalize query for cosine similarity.
        query = query / query_norm

        # Since both article embeddings and the query are
        # normalized, their dot product equals cosine similarity.
        scores = self.embeddings @ query

        # Do not request more results than exist.
        k = min(
            top_k,
            self.num_documents,
        )

        # Find the highest-scoring documents.
        top_indices = np.argpartition(
            -scores,
            k - 1,
        )[:k]

        # Sort those candidates by descending score.
        top_indices = top_indices[
            np.argsort(
                -scores[top_indices]
            )
        ]

        return [
            (
                self.article_ids[index],
                float(scores[index]),
            )
            for index in top_indices
        ]