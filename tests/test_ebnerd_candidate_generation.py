import pandas as pd
import pytest

from src.retrieval.ebnerd_candidate_generation import (
    validate_article_corpus,
    filter_temporally_eligible_articles,
)

from src.retrieval.ebnerd_candidate_generation import (
    build_bm25_corpus,
    retrieve_bm25_candidates,
)

import numpy as np

from src.retrieval.ebnerd_candidate_generation import (
    build_semantic_corpus,
    retrieve_semantic_candidates,
)

from src.retrieval.ebnerd_candidate_generation import (
    build_semantic_corpus,
    retrieve_semantic_candidates,
    merge_retrieval_candidates,
)

from src.retrieval.ebnerd_candidate_generation import (
    apply_rrf,
)

from src.retrieval.ebnerd_candidate_generation import (
    validate_embedding_alignment,
    prepare_semantic_corpus,
    generate_candidates,
)


def test_validate_article_corpus_accepts_required_columns():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B"],
            "published_time": [
                "2026-01-01",
                "2026-01-02",
            ],
        }
    )

    validate_article_corpus(articles)


def test_validate_article_corpus_rejects_missing_columns():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B"],
        }
    )

    with pytest.raises(ValueError, match="published_time"):
        validate_article_corpus(articles)


def test_temporal_filter_excludes_future_articles():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B", "C"],
            "published_time": [
                "2026-01-10 09:00:00",
                "2026-01-10 10:00:00",
                "2026-01-10 11:00:00",
            ],
        }
    )

    result = filter_temporally_eligible_articles(
        articles,
        "2026-01-10 10:00:00",
    )

    assert result["article_id"].tolist() == [
        "A",
        "B",
    ]


def test_temporal_filter_includes_exact_timestamp():
    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "published_time": [
                "2026-01-10 10:00:00",
            ],
        }
    )

    result = filter_temporally_eligible_articles(
        articles,
        "2026-01-10 10:00:00",
    )

    assert result["article_id"].tolist() == ["A"]


def test_temporal_filter_excludes_missing_publication_time():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B"],
            "published_time": [
                None,
                "2026-01-10 09:00:00",
            ],
        }
    )

    result = filter_temporally_eligible_articles(
        articles,
        "2026-01-10 10:00:00",
    )

    assert result["article_id"].tolist() == ["B"]


def test_temporal_filter_does_not_modify_original():
    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "published_time": [
                "2026-01-10 09:00:00",
            ],
        }
    )

    original = articles.copy(deep=True)

    filter_temporally_eligible_articles(
        articles,
        "2026-01-10 10:00:00",
    )

    pd.testing.assert_frame_equal(
        articles,
        original,
    )

def test_build_bm25_corpus_preserves_article_order():
    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "text": [
                "football match latest football news",
                "football team football championship",
                "stock market market update",
            ],
            "published_time": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
        }
    )

    bm25, article_ids = build_bm25_corpus(
        articles
    )

    assert article_ids == [
        "A",
        "B",
        "C",
    ]

    assert len(bm25.doc_len) == 3

def test_retrieve_bm25_candidates_returns_relevant_article():
    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "text": [
                "football championship final",
                "weather forecast today",
                "stock market report",
            ],
            "abstract": [
                "",
                "",
                "",
            ],
            "published_time": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
        }
    )

    bm25, article_ids = build_bm25_corpus(
        articles
    )

    results = retrieve_bm25_candidates(
        bm25,
        article_ids,
        "football championship",
        top_k=2,
    )

    result_ids = [
        article_id
        for article_id, score in results
    ]

    assert "A" in result_ids

def test_retrieve_bm25_candidates_respects_top_k():
    articles = pd.DataFrame(
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
            "published_time": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
        }
    )

    bm25, article_ids = build_bm25_corpus(
        articles
    )

    results = retrieve_bm25_candidates(
        bm25,
        article_ids,
        "football",
        top_k=2,
    )

    assert len(results) == 2

def test_retrieve_bm25_candidates_rejects_invalid_top_k():
    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "text": ["football"],
            "published_time": [
                "2026-01-01"
            ],
        }
    )

    bm25, article_ids = build_bm25_corpus(
        articles
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retrieve_bm25_candidates(
            bm25,
            article_ids,
            "football",
            top_k=0,
        )

def test_retrieve_bm25_candidates_rejects_invalid_top_k():
    articles = pd.DataFrame(
        {
            "article_id": ["A"],
            "text": ["football"],
            "published_time": [
                "2026-01-01"
            ],
        }
    )

    bm25, article_ids = build_bm25_corpus(
        articles
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retrieve_bm25_candidates(
            bm25,
            article_ids,
            "football",
            top_k=0,
        )    

def test_build_semantic_corpus_normalizes_embeddings():
    article_ids = [
        "A",
        "B",
    ]

    embeddings = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=np.float32,
    )

    result_ids, result_embeddings = (
        build_semantic_corpus(
            article_ids,
            embeddings,
        )
    )

    assert result_ids == article_ids

    norms = np.linalg.norm(
        result_embeddings,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones(2),
        atol=1e-6,
    )

def test_build_semantic_corpus_rejects_dimension_mismatch():
    article_ids = [
        "A",
        "B",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="article IDs",
    ):
        build_semantic_corpus(
            article_ids,
            embeddings,
        )

def test_build_semantic_corpus_rejects_zero_vectors():
    article_ids = ["A"]

    embeddings = np.array(
        [
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        build_semantic_corpus(
            article_ids,
            embeddings,
        )

def test_retrieve_semantic_candidates_ranks_by_similarity():
    article_ids = [
        "A",
        "B",
        "C",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    article_ids, embeddings = (
        build_semantic_corpus(
            article_ids,
            embeddings,
        )
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retrieve_semantic_candidates(
        article_ids,
        embeddings,
        query,
        top_k=3,
    )

    result_ids = [
        article_id
        for article_id, score in results
    ]

    assert result_ids == [
        "A",
        "B",
        "C",
    ]

    assert results[0][1] == pytest.approx(
        1.0
    )

def test_retrieve_semantic_candidates_respects_top_k():
    article_ids = [
        "A",
        "B",
        "C",
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_ids, embeddings = (
        build_semantic_corpus(
            article_ids,
            embeddings,
        )
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retrieve_semantic_candidates(
        article_ids,
        embeddings,
        query,
        top_k=2,
    )

    assert len(results) == 2

def test_retrieve_semantic_candidates_rejects_zero_query():
    article_ids = ["A"]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    article_ids, embeddings = (
        build_semantic_corpus(
            article_ids,
            embeddings,
        )
    )

    query = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="zero vector",
    ):
        retrieve_semantic_candidates(
            article_ids,
            embeddings,
            query,
        )

def test_merge_retrieval_candidates_unions_results():
    bm25_candidates = [
        ("A", 5.0),
        ("B", 4.0),
        ("C", 3.0),
    ]

    semantic_candidates = [
        ("C", 0.95),
        ("D", 0.90),
        ("E", 0.85),
    ]

    result = merge_retrieval_candidates(
        bm25_candidates,
        semantic_candidates,
    )

    assert set(result["article_id"]) == {
        "A",
        "B",
        "C",
        "D",
        "E",
    }

    assert len(result) == 5

def test_merge_retrieval_candidates_deduplicates_overlap():
    bm25_candidates = [
        ("A", 5.0),
        ("B", 4.0),
    ]

    semantic_candidates = [
        ("B", 0.95),
        ("C", 0.90),
    ]

    result = merge_retrieval_candidates(
        bm25_candidates,
        semantic_candidates,
    )

    assert result["article_id"].tolist().count("B") == 1
    assert len(result) == 3

def test_merge_retrieval_candidates_marks_both():
    bm25_candidates = [
        ("A", 5.0),
    ]

    semantic_candidates = [
        ("A", 0.95),
    ]

    result = merge_retrieval_candidates(
        bm25_candidates,
        semantic_candidates,
    )

    row = result.iloc[0]

    assert row["source"] == "both"
    assert row["bm25_rank"] == 1
    assert row["semantic_rank"] == 1
    assert row["bm25_score"] == pytest.approx(5.0)
    assert row["semantic_score"] == pytest.approx(0.95)

def test_merge_retrieval_candidates_marks_bm25_only():
    result = merge_retrieval_candidates(
        [
            ("A", 5.0),
        ],
        [],
    )

    row = result.iloc[0]

    assert row["source"] == "bm25"
    assert row["bm25_rank"] == 1
    assert row["semantic_rank"] is None

def test_merge_retrieval_candidates_marks_semantic_only():
    result = merge_retrieval_candidates(
        [],
        [
            ("A", 0.95),
        ],
    )

    row = result.iloc[0]

    assert row["source"] == "semantic"
    assert row["bm25_rank"] is None
    assert row["semantic_rank"] == 1

def test_merge_retrieval_candidates_handles_empty_inputs():
    result = merge_retrieval_candidates(
        [],
        [],
    )

    assert result.empty

    assert result.columns.tolist() == [
        "article_id",
        "bm25_rank",
        "bm25_score",
        "semantic_rank",
        "semantic_score",
        "source",
    ]

def test_apply_rrf_rewards_articles_found_by_both():
    candidates = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "bm25_rank": [
                1,
                2,
                None,
            ],
            "bm25_score": [
                5.0,
                4.0,
                None,
            ],
            "semantic_rank": [
                1,
                None,
                2,
            ],
            "semantic_score": [
                0.95,
                None,
                0.90,
            ],
            "source": [
                "both",
                "bm25",
                "semantic",
            ],
        }
    )

    result = apply_rrf(
        candidates,
        k=60,
    )

    assert result.iloc[0]["article_id"] == "A"

def test_apply_rrf_calculates_expected_score():
    candidates = pd.DataFrame(
        {
            "article_id": ["A"],
            "bm25_rank": [1],
            "bm25_score": [5.0],
            "semantic_rank": [2],
            "semantic_score": [0.9],
            "source": ["both"],
        }
    )

    result = apply_rrf(
        candidates,
        k=60,
    )

    expected = (
        1.0 / 61
        + 1.0 / 62
    )

    assert result.iloc[0]["rrf_score"] == pytest.approx(
        expected
    )

def test_apply_rrf_handles_bm25_only_candidate():
    candidates = pd.DataFrame(
        {
            "article_id": ["A"],
            "bm25_rank": [3],
            "bm25_score": [4.0],
            "semantic_rank": [None],
            "semantic_score": [None],
            "source": ["bm25"],
        }
    )

    result = apply_rrf(
        candidates,
        k=60,
    )

    assert result.iloc[0]["rrf_score"] == pytest.approx(
        1.0 / 63
    )

def test_apply_rrf_handles_semantic_only_candidate():
    candidates = pd.DataFrame(
        {
            "article_id": ["A"],
            "bm25_rank": [None],
            "bm25_score": [None],
            "semantic_rank": [4],
            "semantic_score": [0.8],
            "source": ["semantic"],
        }
    )

    result = apply_rrf(
        candidates,
        k=60,
    )

    assert result.iloc[0]["rrf_score"] == pytest.approx(
        1.0 / 64
    )

def test_apply_rrf_rejects_invalid_k():
    candidates = pd.DataFrame(
        {
            "article_id": ["A"],
            "bm25_rank": [1],
            "semantic_rank": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="RRF k",
    ):
        apply_rrf(
            candidates,
            k=0,
        )

def test_apply_rrf_rejects_missing_columns():
    candidates = pd.DataFrame(
        {
            "article_id": ["A"],
            "bm25_rank": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="semantic_rank",
    ):
        apply_rrf(candidates)

def test_validate_embedding_alignment_accepts_matching_corpus():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B", "C"],
            "published_time": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    validate_embedding_alignment(
        articles,
        embeddings,
    )

def test_validate_embedding_alignment_rejects_row_mismatch():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "B"],
            "published_time": [
                "2026-01-01",
                "2026-01-02",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="embedding rows",
    ):
        validate_embedding_alignment(
            articles,
            embeddings,
        )

def test_validate_embedding_alignment_rejects_duplicate_article_ids():
    articles = pd.DataFrame(
        {
            "article_id": ["A", "A"],
            "published_time": [
                "2026-01-01",
                "2026-01-02",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        validate_embedding_alignment(
            articles,
            embeddings,
        )

def test_prepare_semantic_corpus_preserves_alignment_after_temporal_filter():
    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "published_time": [
                "2026-01-01 09:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 10:00:00",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],  # A
            [0.0, 1.0],  # B
            [1.0, 1.0],  # C
        ],
        dtype=np.float32,
    )

    article_ids, filtered_embeddings = (
        prepare_semantic_corpus(
            articles,
            embeddings,
            impression_time="2026-01-01 10:00:00",
        )
    )

    assert article_ids == [
        "A",
        "C",
    ]

    np.testing.assert_allclose(
        filtered_embeddings[0],
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ) / 1.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        filtered_embeddings[1],
        np.array(
            [1.0, 1.0],
            dtype=np.float32,
        ) / np.sqrt(2),
        atol=1e-6,
    )

def test_generate_candidates_excludes_future_articles():
    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "text": [
                "football match latest football news",
                "football team football championship",
                "football player football club"
            ],
            "published_time": [
                "2026-01-01 09:00:00",
                "2026-01-01 09:30:00",
                "2026-01-01 11:00:00",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    query_embedding = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    result = generate_candidates(
        articles=articles,
        article_embeddings=embeddings,
        query="football championship",
        query_embedding=query_embedding,
        bm25_top_k=10,
        semantic_top_k=10,
        final_top_k=10,
        impression_time="2026-01-01 10:00:00",
    )

    assert "C" not in set(
        result["article_id"]
    )

def test_generate_candidates_returns_fused_candidates():
    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "text": [
            "Football news Latest football match",
            "Stock market Latest stock prices",
            "Weather report Today's weather",
            ],
            "published_time": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
        }
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    query_embedding = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    result = generate_candidates(
        articles=articles,
        article_embeddings=embeddings,
        query="football championship",
        query_embedding=query_embedding,
        bm25_top_k=2,
        semantic_top_k=2,
        final_top_k=3,
    )

    assert not result.empty

    assert set(
        [
            "article_id",
            "bm25_rank",
            "bm25_score",
            "semantic_rank",
            "semantic_score",
            "source",
            "rrf_score",
        ]
    ).issubset(
        result.columns
    )

    assert result["article_id"].is_unique

    assert result["rrf_score"].notna().all()

