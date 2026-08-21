import numpy as np
import pandas as pd
import pytest

from src.ranking.hybrid import (
    build_hybrid_scores,
    min_max_normalize,
    score_impression,
)


def test_min_max_normalization():
    scores = pd.Series(
        [10.0, 20.0, 30.0]
    )

    normalized = min_max_normalize(
        scores
    )

    assert np.allclose(
        normalized.to_numpy(),
        [0.0, 0.5, 1.0],
    )


def test_constant_scores_return_zero():
    scores = pd.Series(
        [5.0, 5.0, 5.0]
    )

    normalized = min_max_normalize(
        scores
    )

    assert np.allclose(
        normalized.to_numpy(),
        [0.0, 0.0, 0.0],
    )


def test_normalization_preserves_order():
    scores = pd.Series(
        [4.0, 1.0, 10.0]
    )

    normalized = min_max_normalize(
        scores
    )

    original_order = (
        scores
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    normalized_order = (
        normalized
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    assert original_order == normalized_order


def test_hybrid_score_uses_weights():
    df = pd.DataFrame(
        {
            "impression_id": ["I1", "I1", "I1"],
            "bm25_score": [0.0, 5.0, 10.0],
            "semantic_score": [0.0, 0.5, 1.0],
            "behavioral_score": [0.0, 0.5, 1.0],
        }
    )

    result = score_impression(
        df,
        lexical_weight=0.4,
        semantic_weight=0.3,
        behavioral_weight=0.3,
    )

    assert np.isclose(
        result.loc[
            1,
            "hybrid_score",
        ],
        0.5,
    )

    assert np.isclose(
        result.loc[
            2,
            "hybrid_score",
        ],
        1.0,
    )


def test_weights_must_sum_to_one():
    df = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "bm25_score": [1.0],
            "semantic_score": [0.5],
            "behavioral_score": [0.5],
        }
    )

    with pytest.raises(ValueError):
        score_impression(
            df,
            lexical_weight=0.5,
            semantic_weight=0.3,
            behavioral_weight=0.3,
        )


def test_non_negative_weights_required():
    df = pd.DataFrame(
        {
            "impression_id": ["I1"],
            "bm25_score": [1.0],
            "semantic_score": [0.5],
            "behavioral_score": [0.5],
        }
    )

    with pytest.raises(ValueError):
        score_impression(
            df,
            lexical_weight=-0.1,
            semantic_weight=0.6,
            behavioral_weight=0.5,
        )


def test_hybrid_scores_are_built_per_impression():
    df = pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I1",
                "I2",
                "I2",
            ],
            "bm25_score": [
                1.0,
                3.0,
                100.0,
                200.0,
            ],
            "semantic_score": [
                0.2,
                0.8,
                0.1,
                0.9,
            ],
            "behavioral_score": [
                0.3,
                0.7,
                0.2,
                0.8,
            ],
        }
    )

    result = build_hybrid_scores(
        df
    )

    i1 = result[
        result["impression_id"] == "I1"
    ]

    i2 = result[
        result["impression_id"] == "I2"
    ]

    assert i1[
        "bm25_normalized"
    ].tolist() == [0.0, 1.0]

    assert i2[
        "bm25_normalized"
    ].tolist() == [0.0, 1.0]