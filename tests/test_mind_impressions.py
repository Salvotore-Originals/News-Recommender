from src.data.mind_impressions import (
    parse_impressions,
    get_positive_articles,
    get_negative_articles,
)


def test_parse_impressions():

    impressions = (
        "N55689-1 "
        "N35729-0 "
        "N12345-0 "
        "N99999-1"
    )

    parsed = parse_impressions(
        impressions
    )

    assert parsed == [
        ("N55689", 1),
        ("N35729", 0),
        ("N12345", 0),
        ("N99999", 1),
    ]


def test_positive_articles():

    impressions = (
        "N55689-1 "
        "N35729-0 "
        "N99999-1"
    )

    positives = get_positive_articles(
        impressions
    )

    assert positives == [
        "N55689",
        "N99999",
    ]


def test_negative_articles():

    impressions = (
        "N55689-1 "
        "N35729-0 "
        "N99999-1"
    )

    negatives = get_negative_articles(
        impressions
    )

    assert negatives == [
        "N35729",
    ]