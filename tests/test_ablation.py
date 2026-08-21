import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import (
    ABLATION_CONFIGS,
    evaluate_ablation,
    evaluate_ablation_config,
    validate_ablation_configs,
)


# ============================================================
# TEST DATA
# ============================================================

@pytest.fixture
def candidate_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": [
                "I1", "I1", "I1",
                "I2", "I2", "I2",
            ],
            "clicked": [
                1, 0, 0,
                0, 1, 0,
            ],
            "bm25_score": [
                3.0, 2.0, 1.0,
                1.0, 3.0, 2.0,
            ],
            "semantic_score": [
                1.0, 2.0, 3.0,
                3.0, 1.0, 2.0,
            ],
            "behavioral_score": [
                2.0, 3.0, 1.0,
                2.0, 1.0, 3.0,
            ],
        }
    )


# ============================================================
# CONFIGURATION TESTS
# ============================================================

def test_ablation_configs_are_valid():
    validate_ablation_configs()

    assert len(ABLATION_CONFIGS) == 7

    for config in ABLATION_CONFIGS:

        total = (
            config.lexical_weight
            + config.semantic_weight
            + config.behavioral_weight
        )

        assert np.isclose(
            total,
            1.0,
        )


# ============================================================
# REQUIRED MODELS
# ============================================================

def test_expected_ablation_models_exist():

    names = {
        config.name
        for config in ABLATION_CONFIGS
    }

    expected = {
        "BM25",
        "Semantic",
        "Behavioural",
        "BM25 + Semantic",
        "BM25 + Behavioural",
        "Semantic + Behavioural",
        "Full Hybrid",
    }

    assert names == expected


# ============================================================
# SINGLE CONFIGURATION
# ============================================================

def test_single_ablation_config(
    candidate_scores,
):

    config = ABLATION_CONFIGS[0]

    result = evaluate_ablation_config(
        candidate_scores,
        config,
    )

    assert result["model"] == "BM25"

    assert "Hit@5" in result
    assert "Hit@10" in result
    assert "MRR" in result
    assert "NDCG@5" in result
    assert "NDCG@10" in result


# ============================================================
# FULL ABLATION
# ============================================================

def test_full_ablation_returns_seven_models(
    candidate_scores,
):

    results = evaluate_ablation(
        candidate_scores
    )

    assert len(results) == 7

    assert set(
        results["model"]
    ) == {
        "BM25",
        "Semantic",
        "Behavioural",
        "BM25 + Semantic",
        "BM25 + Behavioural",
        "Semantic + Behavioural",
        "Full Hybrid",
    }


# ============================================================
# METRIC VALIDITY
# ============================================================

def test_ablation_metrics_are_valid(
    candidate_scores,
):

    results = evaluate_ablation(
        candidate_scores
    )

    for column in [
        "Hit@5",
        "Hit@10",
        "MRR",
        "NDCG@5",
        "NDCG@10",
    ]:

        assert results[column].notna().all()

        assert (
            results[column]
            >= 0
        ).all()

        assert (
            results[column]
            <= 1
        ).all()


# ============================================================
# EMPTY INPUT
# ============================================================

def test_empty_candidate_scores_raise():

    empty = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        evaluate_ablation(
            empty
        )


# ============================================================
# MISSING COLUMN
# ============================================================

def test_missing_score_column_raises(
    candidate_scores,
):

    broken = candidate_scores.drop(
        columns=["semantic_score"]
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        evaluate_ablation(
            broken
        )