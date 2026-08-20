import numpy as np
import pandas as pd
import pytest

from src.retrieval.mind_semantic import (
    load_mind_articles,
)


def test_load_mind_articles():
    articles = pd.DataFrame(
        {
            "article_id": [
                "N1",
                "N2",
            ],
            "text": [
                "Football match tonight.",
                "Technology company launches product.",
            ],
        }
    )

    path = "tests/test_articles.parquet"

    articles.to_parquet(
        path,
        index=False,
    )

    loaded = load_mind_articles(
        path
    )

    assert loaded.shape == (2, 2)
    assert loaded["article_id"].tolist() == [
        "N1",
        "N2",
    ]


def test_load_mind_articles_rejects_missing_text():
    articles = pd.DataFrame(
        {
            "article_id": ["N1"],
        }
    )

    path = "tests/test_articles_missing_text.parquet"

    articles.to_parquet(
        path,
        index=False,
    )

    try:
        load_mind_articles(path)
    except ValueError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing text column."
        )


def test_load_mind_articles_rejects_empty_text():
    articles = pd.DataFrame(
        {
            "article_id": ["N1"],
            "text": [""],
        }
    )

    path = "tests/test_articles_empty_text.parquet"

    articles.to_parquet(
        path,
        index=False,
    )

    try:
        load_mind_articles(path)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for empty article text."
        )

def test_build_user_embedding_uses_recent_history():
    from src.retrieval.mind_semantic import (
        build_user_embedding,
    )

    article_ids = [
        "N1",
        "N2",
        "N3",
        "N4",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    history = [
        "N1",
        "N2",
        "N3",
        "N4",
    ]

    user_embedding = build_user_embedding(
        history=history,
        article_embeddings=embeddings,
        article_ids=article_ids,
        max_history=2,
    )

    # N3 + N4 = [1,1] after averaging.
    expected = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    expected = expected / np.linalg.norm(
        expected
    )

    np.testing.assert_allclose(
        user_embedding,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_build_user_embedding_ignores_unknown_articles():
    from src.retrieval.mind_semantic import (
        build_user_embedding,
    )

    article_ids = [
        "N1",
        "N2",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    history = [
        "UNKNOWN",
        "N1",
    ]

    user_embedding = build_user_embedding(
        history=history,
        article_embeddings=embeddings,
        article_ids=article_ids,
        max_history=10,
    )

    expected = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        user_embedding,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_build_user_embedding_rejects_empty_history():
    from src.retrieval.mind_semantic import (
        build_user_embedding,
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        build_user_embedding(
            history=[],
            article_embeddings=embeddings,
            article_ids=["N1"],
        )


def test_build_user_embedding_rejects_unknown_history():
    from src.retrieval.mind_semantic import (
        build_user_embedding,
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        build_user_embedding(
            history=["UNKNOWN"],
            article_embeddings=embeddings,
            article_ids=["N1"],
        )

def test_semantic_retriever_returns_most_similar_article():

    from src.retrieval.mind_semantic import (
        SemanticRetriever,
    )

    article_ids = [
        "N1",
        "N2",
        "N3",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=article_ids,
        article_embeddings=embeddings,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retriever.search(
        query,
        top_k=3,
    )

    assert results[0][0] == "N1"

    assert results[0][1] > results[1][1]


def test_semantic_retriever_scores_are_descending():

    from src.retrieval.mind_semantic import (
        SemanticRetriever,
    )

    article_ids = [
        "N1",
        "N2",
        "N3",
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
        article_embeddings=embeddings,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retriever.search(
        query,
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


def test_semantic_retriever_rejects_zero_query():

    from src.retrieval.mind_semantic import (
        SemanticRetriever,
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=["N1"],
        article_embeddings=embeddings,
    )

    with pytest.raises(ValueError):

        retriever.search(
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            )
        )


def test_semantic_retriever_rejects_dimension_mismatch():

    from src.retrieval.mind_semantic import (
        SemanticRetriever,
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        article_ids=["N1"],
        article_embeddings=embeddings,
    )

    with pytest.raises(ValueError):

        retriever.search(
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )