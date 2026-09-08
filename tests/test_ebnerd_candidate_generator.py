import numpy as np
import pandas as pd
import pytest

from src.retrieval.ebnerd_candidate_generation import (
    EBNeRDCandidateGenerator,
)


def make_articles():
    return pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
                "D",
            ],
            "text": [
                "football match latest football news",
                "football team football championship",
                "stock market market update",
                "future football future football event",
            ],
            "published_time": pd.to_datetime(
                [
                    "2023-05-20 10:00:00",
                    "2023-05-21 10:00:00",
                    "2023-05-22 10:00:00",
                    "2023-05-30 10:00:00",
                ]
            ),
        }
    )


def make_embeddings():
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_generator_builds_once_and_retrieves():
    articles = make_articles()
    embeddings = make_embeddings()

    generator = EBNeRDCandidateGenerator(
        articles,
        embeddings,
        bm25_top_k=3,
        semantic_top_k=3,
        rrf_k=60,
    )

    result = generator.retrieve(
        query="football",
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        impression_time=pd.Timestamp(
            "2023-05-25 00:00:00"
        ),
        final_top_k=3,
    )

    assert not result.empty

    assert set(result["article_id"]).issubset(
        {"A", "B", "C"}
    )

    assert "D" not in set(
        result["article_id"]
    )


def test_future_articles_are_excluded():
    articles = make_articles()
    embeddings = make_embeddings()

    generator = EBNeRDCandidateGenerator(
        articles,
        embeddings,
    )

    result = generator.retrieve(
        query="football",
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        impression_time=pd.Timestamp(
            "2023-05-25 00:00:00"
        ),
        final_top_k=100,
    )

    assert "D" not in set(
        result["article_id"]
    )


def test_exact_publication_time_is_eligible():
    articles = make_articles()

    embeddings = make_embeddings()

    generator = EBNeRDCandidateGenerator(
        articles,
        embeddings,
    )

    result = generator.retrieve(
        query="stock market",
        query_embedding=np.array(
            [0.0, 1.0],
            dtype=np.float32,
        ),
        impression_time=pd.Timestamp(
            "2023-05-22 10:00:00"
        ),
        final_top_k=100,
    )

    assert "C" in set(
        result["article_id"]
    )


def test_invalid_final_top_k():
    articles = make_articles()
    embeddings = make_embeddings()

    generator = EBNeRDCandidateGenerator(
        articles,
        embeddings,
    )

    with pytest.raises(ValueError):
        generator.retrieve(
            query="football",
            query_embedding=np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            final_top_k=0,
        )


def test_future_only_corpus_returns_empty():
    articles = make_articles()
    embeddings = make_embeddings()

    generator = EBNeRDCandidateGenerator(
        articles,
        embeddings,
    )

    result = generator.retrieve(
        query="football",
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        impression_time=pd.Timestamp(
            "2023-05-01 00:00:00"
        ),
        final_top_k=100,
    )

    assert result.empty