from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.hybrid import evaluate_hybrid
from src.ranking.candidates import (
    build_hybrid_candidate_scores,
)
from src.ranking.hybrid import build_hybrid_scores
from src.retrieval.bm25 import BM25Retriever


def test_end_to_end_hybrid_pipeline():
    """
    Verify the complete Phase 5 ranking pipeline:

        MIND candidates
            ↓
        BM25
            ↓
        Semantic
            ↓
        Behavioural
            ↓
        Hybrid normalization
            ↓
        Ranking metrics
    """

    # ========================================================
    # 1. Candidate impressions
    # ========================================================

    candidates = pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I1",
                "I1",
                "I2",
                "I2",
                "I2",
            ],
            "user_id": [
                "U1",
                "U1",
                "U1",
                "U2",
                "U2",
                "U2",
            ],
            "timestamp": [
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
                "2019-11-12 11:00:00",
                "2019-11-12 11:00:00",
                "2019-11-12 11:00:00",
            ],
            "article_id": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
            ],
            "clicked": [
                0,
                1,
                0,
                0,
                0,
                1,
            ],
        }
    )

    # ========================================================
    # 2. Historical clicks
    # ========================================================

    history = pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I1",
                "I2",
            ],
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "history_position": [
                0,
                1,
                0,
            ],
        }
    )

    # ========================================================
    # 3. Article metadata
    # ========================================================

    articles = pd.DataFrame(
        {
            "article_id": [
                "A",
                "B",
                "C",
            ],
            "title": [
                "Apple technology",
                "Banana sports",
                "Orange politics",
            ],
        }
    )

    # ========================================================
    # 4. BM25
    # ========================================================

    bm25 = BM25Retriever(
        article_ids=[
            "A",
            "B",
            "C",
        ],
        documents=[
            "apple technology",
            "banana sports",
            "orange politics",
        ],
    )

    # ========================================================
    # 5. Semantic embeddings
    # ========================================================

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_ids = np.array(
        [
            "A",
            "B",
            "C",
        ]
    )

    # ========================================================
    # 6. Behavioural scores
    # ========================================================

    behavioral = pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I1",
                "I1",
                "I2",
                "I2",
                "I2",
            ],
            "article_id": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
            ],
            "behavioral_score": [
                0.20,
                0.90,
                0.10,
                0.20,
                0.30,
                0.95,
            ],
        }
    )

    # ========================================================
    # 7. Build candidate-level signals
    # ========================================================

    candidate_scores = (
        build_hybrid_candidate_scores(
            candidates=candidates,
            history=history,
            articles=articles,
            bm25_retriever=bm25,
            article_embeddings=embeddings,
            article_ids=article_ids,
            behavioral_scores=behavioral,
            max_history=2,
        )
    )

    # ========================================================
    # 8. Validate candidate-level output
    # ========================================================

    assert len(candidate_scores) == 6

    assert set(
        candidate_scores.columns
    ) >= {
        "impression_id",
        "article_id",
        "clicked",
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    }

    assert (
        candidate_scores["impression_id"]
        .nunique()
        == 2
    )

    assert (
        candidate_scores["article_id"]
        .nunique()
        == 3
    )

    # ========================================================
    # 9. Build hybrid scores
    # ========================================================

    scored = build_hybrid_scores(
        candidate_scores,
        lexical_weight=0.40,
        semantic_weight=0.30,
        behavioral_weight=0.30,
    )

    # ========================================================
    # 10. Verify hybrid score exists
    # ========================================================

    assert "hybrid_score" in scored.columns

    assert (
        scored["hybrid_score"]
        .notna()
        .all()
    )

    assert np.isfinite(
        scored["hybrid_score"]
        .to_numpy()
    ).all()

    # ========================================================
    # 11. Verify normalized scores are bounded
    # ========================================================

    for column in [
        "bm25_normalized",
        "semantic_normalized",
        "behavioral_normalized",
        "hybrid_score",
    ]:

        values = scored[column]

        assert (
            values >= 0
        ).all()

        assert (
            values <= 1
        ).all()

    # ========================================================
    # 12. Run final evaluation
    # ========================================================

    results = evaluate_hybrid(
        candidate_scores,
        lexical_weight=0.40,
        semantic_weight=0.30,
        behavioral_weight=0.30,
    )

    # ========================================================
    # 13. Verify all four models were evaluated
    # ========================================================

    assert set(
        results["model"]
    ) == {
        "BM25",
        "Semantic",
        "Behavioural",
        "Hybrid",
    }

    assert len(results) == 4

    # ========================================================
    # 14. Verify metrics are valid
    # ========================================================

    metric_columns = [
        "MRR",
        "Hit@5",
        "Hit@10",
        "NDCG@5",
        "NDCG@10",
    ]

    for column in metric_columns:

        assert (
            results[column]
            .notna()
            .all()
        )

        assert np.isfinite(
            results[column]
            .to_numpy()
        ).all()

        assert (
            results[column] >= 0
        ).all()

        assert (
            results[column] <= 1
        ).all()