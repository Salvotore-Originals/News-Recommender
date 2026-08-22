import pytest

from src.evaluation.competition_metrics import (
    auc,
    dcg_at_k,
    evaluate_competition_ranking,
    mrr,
    ndcg_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first_relevant():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
    ]

    assert reciprocal_rank(
        results,
        ["N20"],
    ) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
    ]

    assert reciprocal_rank(
        results,
        ["N99"],
    ) == 0.0


def test_mrr():

    results = [
        [
            ("N10", 10.0),
            ("N20", 9.0),
        ],
        [
            ("N30", 10.0),
            ("N40", 9.0),
        ],
    ]

    relevant = [
        ["N20"],
        ["N30"],
    ]

    # RR values:
    # 1/2 and 1/1
    #
    # MRR = (0.5 + 1.0) / 2 = 0.75

    assert mrr(
        results,
        relevant,
    ) == pytest.approx(0.75)


def test_dcg_at_k():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
    ]

    assert dcg_at_k(
        results,
        ["N10"],
        3,
    ) == pytest.approx(1.0)


def test_ndcg_perfect_ranking():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
    ]

    assert ndcg_at_k(
        results,
        ["N10"],
        3,
    ) == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_two():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
    ]

    score = ndcg_at_k(
        results,
        ["N20"],
        3,
    )

    assert score == pytest.approx(
        1.0 / 1.584962500721156
    )
    # For one relevant article:
    #
    # DCG = 1/log2(3)
    # IDCG = 1/log2(2) = 1
    #
    # Therefore nDCG = 1/log2(3)


def test_auc_perfect_ranking():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 1.0),
        ("N40", 0.0),
    ]

    assert auc(
        results,
        ["N10", "N20"],
    ) == pytest.approx(1.0)


def test_auc_reverse_ranking():

    results = [
        ("N10", 1.0),
        ("N20", 0.0),
        ("N30", 10.0),
        ("N40", 9.0),
    ]

    assert auc(
        results,
        ["N10", "N20"],
    ) == pytest.approx(0.0)


def test_auc_tie():

    results = [
        ("N10", 5.0),
        ("N20", 5.0),
    ]

    assert auc(
        results,
        ["N10"],
    ) == pytest.approx(0.5)


def test_complete_competition_metrics():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 1.0),
    ]

    metrics = evaluate_competition_ranking(
        results,
        ["N10"],
    )

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg_at_5"] == pytest.approx(1.0)
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)