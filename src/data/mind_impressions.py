from __future__ import annotations

from typing import Iterable


def parse_impressions(
    impressions: str,
) -> list[tuple[str, int]]:
    """
    Parse MIND impressions.

    Example
    -------
    'N55689-1 N35729-0'

    becomes

    [
        ('N55689', 1),
        ('N35729', 0),
    ]
    """

    if not isinstance(impressions, str):
        return []

    impressions = impressions.strip()

    if not impressions:
        return []

    parsed = []

    for item in impressions.split():

        try:
            article_id, label = item.rsplit(
                "-",
                1,
            )

            parsed.append(
                (
                    article_id,
                    int(label),
                )
            )

        except ValueError:
            continue

    return parsed


def get_positive_articles(
    impressions: str,
) -> list[str]:
    """
    Return article IDs with click label 1.
    """

    return [
        article_id
        for article_id, label
        in parse_impressions(impressions)
        if label == 1
    ]


def get_negative_articles(
    impressions: str,
) -> list[str]:
    """
    Return article IDs with click label 0.
    """

    return [
        article_id
        for article_id, label
        in parse_impressions(impressions)
        if label == 0
    ]