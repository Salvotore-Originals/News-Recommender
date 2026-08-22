import numpy as np
import pytest

from src.retrieval.ebnerd_hybrid import (
    normalize_scores,
    combine_scores,
    rank_candidates,
    evaluate_ranking,
)


def test_normalize_scores():

    scores = np.array(
        [10.0, 20.0, 30.0]
    )

    result = normalize_scores(
        scores
    )

    assert np.allclose(
        result,
        [0.0, 0.5, 1.0],
    )


def test_normalize_constant_scores():

    scores = np.array(
        [5.0, 5.0, 5.0]
    )

    result = normalize_scores(
        scores
    )

    assert np.allclose(
        result,
        [0.0, 0.0, 0.0],
    )


def test_normalize_empty_scores():

    scores = np.array([])

    result = normalize_scores(
        scores
    )

    assert len(result) == 0


def test_hybrid_equal_weights():

    lexical = np.array(
        [1.0, 0.0, 0.0]
    )

    semantic = np.array(
        [0.0, 1.0, 0.0]
    )

    behavioral = np.array(
        [0.0, 0.0, 1.0]
    )

    result = combine_scores(
        lexical,
        semantic,
        behavioral,
    )

    assert np.allclose(
        result,
        [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    )


def test_hybrid_weights_must_sum_to_one():

    with pytest.raises(
        ValueError
    ):

        combine_scores(
            [1, 2],
            [1, 2],
            [1, 2],
            lexical_weight=0.5,
            semantic_weight=0.5,
            behavioral_weight=0.5,
        )


def test_rank_candidates():

    candidate_ids = [
        1,
        2,
        3,
    ]

    scores = [
        0.2,
        0.9,
        0.5,
    ]

    ranked = rank_candidates(
        candidate_ids,
        scores,
    )

    assert [
        article_id
        for article_id, _score
        in ranked
    ] == [
        2,
        3,
        1,
    ]


def test_rank_candidates_tie_breaking():

    candidate_ids = [
        3,
        1,
        2,
    ]

    scores = [
        0.5,
        0.5,
        0.5,
    ]

    ranked = rank_candidates(
        candidate_ids,
        scores,
    )

    assert [
        article_id
        for article_id, _score
        in ranked
    ] == [
        1,
        2,
        3,
    ]


def test_evaluate_ranking():

    ranked_ids = [
        5,
        2,
        9,
        7,
        3,
    ]

    clicked = {
        9,
    }

    hit5, hit10, mrr = (
        evaluate_ranking(
            ranked_ids,
            clicked,
        )
    )

    assert hit5 == 1
    assert hit10 == 1
    assert mrr == 1.0 / 3.0


def test_evaluate_ranking_miss():

    ranked_ids = [
        1,
        2,
        3,
        4,
        5,
    ]

    clicked = {
        99,
    }

    hit5, hit10, mrr = (
        evaluate_ranking(
            ranked_ids,
            clicked,
        )
    )

    assert hit5 == 0
    assert hit10 == 0
    assert mrr == 0.0

def test_behavioral_only_weight():

    lexical = np.array(
        [0.1, 0.9, 0.2]
    )

    semantic = np.array(
        [0.8, 0.1, 0.3]
    )

    behavioral = np.array(
        [0.2, 0.3, 1.0]
    )

    result = combine_scores(
        lexical,
        semantic,
        behavioral,
        lexical_weight=0.0,
        semantic_weight=0.0,
        behavioral_weight=1.0,
    )

    assert np.allclose(
        result,
        normalize_scores(
            behavioral
        ),
    )

def test_three_signal_hybrid():

    lexical = np.array(
        [1.0, 0.0]
    )

    semantic = np.array(
        [0.0, 1.0]
    )

    behavioral = np.array(
        [0.5, 0.5]
    )

    result = combine_scores(
        lexical,
        semantic,
        behavioral,
        lexical_weight=1.0 / 3.0,
        semantic_weight=1.0 / 3.0,
        behavioral_weight=1.0 / 3.0,
    )

    assert np.allclose(
        result,
        [
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    )

