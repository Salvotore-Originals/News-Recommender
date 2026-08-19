from __future__ import annotations

from typing import Iterable


def filter_seen_articles(
    candidates: list[tuple[str, float]],
    seen_articles: Iterable[str],
) -> list[tuple[str, float]]:
    """
    Remove articles already seen/clicked by the user.

    Parameters
    ----------
    candidates:
        Ranked list of (article_id, score).

    seen_articles:
        Article IDs already present in the user's history.

    Returns
    -------
    list[tuple[str, float]]
        Candidates excluding seen articles.
    """

    seen = {
        str(article_id)
        for article_id in seen_articles
    }

    return [
        (article_id, score)
        for article_id, score in candidates
        if article_id not in seen
    ]