import numpy as np
import pandas as pd
import pytest

from src.ranking.candidates import (
    build_hybrid_candidate_scores,
)
from src.retrieval.bm25 import BM25Retriever


def test_hybrid_candidate_scores_align_all_signals():
    candidates = pd.DataFrame(
        {
            "impression_id": ["I1", "I1", "I1"],
            "user_id": ["U1", "U1", "U1"],
            "timestamp": [
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
            ],
            "article_id": ["A", "B", "C"],
            "clicked": [0, 1, 0],
        }
    )

    history = pd.DataFrame(
        {
            "impression_id": ["I1", "I1"],
            "article_id": ["A", "B"],
            "history_position": [0, 1],
        }
    )

    articles = pd.DataFrame(
        {
            "article_id": ["A", "B", "C"],
            "title": [
                "Apple news",
                "Banana news",
                "Orange news",
            ],
        }
    )

    bm25 = BM25Retriever(
        article_ids=["A", "B", "C"],
        documents=[
            "apple news",
            "banana news",
            "orange news",
        ],
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_ids = np.array(
        ["A", "B", "C"]
    )

    behavioral = pd.DataFrame(
        {
            "impression_id": ["I1", "I1", "I1"],
            "article_id": ["A", "B", "C"],
            "behavioral_score": [
                0.20,
                0.80,
                0.40,
            ],
        }
    )

    result = build_hybrid_candidate_scores(
        candidates=candidates,
        history=history,
        articles=articles,
        bm25_retriever=bm25,
        article_embeddings=embeddings,
        article_ids=article_ids,
        behavioral_scores=behavioral,
        max_history=2,
    )

    assert len(result) == 3

    assert result["article_id"].tolist() == [
        "A",
        "B",
        "C",
    ]

    assert result["clicked"].tolist() == [
        0,
        1,
        0,
    ]

    assert result["behavioral_score"].tolist() == [
        0.20,
        0.80,
        0.40,
    ]

    assert result["bm25_score"].notna().all()
    assert result["semantic_score"].notna().all()


def test_missing_behavioral_score_uses_zero_fallback():
    candidates = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "user_id": ["U1"],
            "timestamp": ["2019-11-12 10:00:00"],
            "article_id": ["A"],
            "clicked": [0],
        }
    )

    history = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "article_id": ["A"],
            "history_position": [0],
        }
    )

    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "title": ["Apple news"],
        }
    )

    bm25 = BM25Retriever(
        article_ids=["A"],
        documents=["apple news"],
    )

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    behavioral = pd.DataFrame(
        {
            "impression_id": [],
            "article_id": [],
            "behavioral_score": [],
        }
    )

    result = build_hybrid_candidate_scores(
        candidates=candidates,
        history=history,
        articles=articles,
        bm25_retriever=bm25,
        article_embeddings=embeddings,
        article_ids=np.array(["A"]),
        behavioral_scores=behavioral,
    )

    assert result.loc[
        0,
        "behavioral_score",
    ] == 0.0


def test_duplicate_behavioral_keys_raise_error():
    candidates = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "user_id": ["U1"],
            "timestamp": ["2019-11-12 10:00:00"],
            "article_id": ["A"],
            "clicked": [0],
        }
    )

    history = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "article_id": ["A"],
            "history_position": [0],
        }
    )

    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "title": ["Apple news"],
        }
    )

    bm25 = BM25Retriever(
        article_ids=["A"],
        documents=["apple news"],
    )

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    behavioral = pd.DataFrame(
        {
            "impression_id": ["I1", "I1"],
            "article_id": ["A", "A"],
            "behavioral_score": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError):
        build_hybrid_candidate_scores(
            candidates=candidates,
            history=history,
            articles=articles,
            bm25_retriever=bm25,
            article_embeddings=embeddings,
            article_ids=np.array(["A"]),
            behavioral_scores=behavioral,
        )


def test_non_finite_scores_are_rejected():
    candidates = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "user_id": ["U1"],
            "timestamp": ["2019-11-12 10:00:00"],
            "article_id": ["A"],
            "clicked": [0],
        }
    )

    history = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "article_id": ["A"],
            "history_position": [0],
        }
    )

    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "title": ["Apple news"],
        }
    )

    bm25 = BM25Retriever(
        article_ids=["A"],
        documents=["apple news"],
    )

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    behavioral = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "article_id": ["A"],
            "behavioral_score": [np.nan],
        }
    )

    with pytest.raises(ValueError):
        build_hybrid_candidate_scores(
            candidates=candidates,
            history=history,
            articles=articles,
            bm25_retriever=bm25,
            article_embeddings=embeddings,
            article_ids=np.array(["A"]),
            behavioral_scores=behavioral,
        )