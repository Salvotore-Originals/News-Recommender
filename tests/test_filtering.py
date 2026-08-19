from src.retrieval.filtering import (
    filter_seen_articles,
)


def test_filter_seen_articles():

    candidates = [
        ("N1", 10.0),
        ("N2", 9.0),
        ("N3", 8.0),
        ("N4", 7.0),
    ]

    seen = [
        "N1",
        "N3",
    ]

    filtered = filter_seen_articles(
        candidates,
        seen,
    )

    assert filtered == [
        ("N2", 9.0),
        ("N4", 7.0),
    ]