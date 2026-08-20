import numpy as np
import pytest

from src.retrieval.semantic import SemanticRetriever


def test_semantic_retriever_returns_most_similar_article():
    article_ids = [
        "A",
        "B",
        "C",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        embeddings=embeddings,
    )

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    results = retriever.search(
        query_embedding=query,
        top_k=2,
    )

    assert results[0][0] == "A"
    assert results[1][0] == "C"


def test_semantic_scores_are_descending():
    article_ids = [
        "A",
        "B",
        "C",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        embeddings=embeddings,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retriever.search(
        query_embedding=query,
        top_k=3,
    )

    scores = [
        score
        for _, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_top_k_cannot_be_zero():
    article_ids = [
        "A",
        "B",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        embeddings=embeddings,
    )

    with pytest.raises(ValueError):
        retriever.search(
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=0,
        )


def test_dimension_mismatch_is_rejected():
    article_ids = [
        "A",
        "B",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        embeddings=embeddings,
    )

    with pytest.raises(ValueError):
        retriever.search(
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )


def test_zero_query_vector_is_rejected():
    article_ids = [
        "A",
        "B",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        embeddings=embeddings,
    )

    with pytest.raises(ValueError):
        retriever.search(
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )