from src.evaluation.retrieval import hit_at_k


def test_hit_at_k():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
        ("N40", 7.0),
        ("N50", 6.0),
    ]

    relevant = [
        "N30",
    ]

    assert hit_at_k(
        results,
        relevant,
        3,
    ) == 1

    assert hit_at_k(
        results,
        relevant,
        2,
    ) == 0


def test_multiple_relevant_articles():

    results = [
        ("N10", 10.0),
        ("N20", 9.0),
        ("N30", 8.0),
    ]

    relevant = [
        "N99",
        "N20",
    ]

    assert hit_at_k(
        results,
        relevant,
        2,
    ) == 1