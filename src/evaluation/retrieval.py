def hit_at_k(
    results: list[tuple[str, float]],
    relevant_articles: list[str],
    k: int,
) -> int:

    relevant = set(relevant_articles)

    retrieved = {
        article_id
        for article_id, _ in results[:k]
    }

    return int(
        bool(retrieved & relevant)
    )


def rank_of_article(
    results: list[tuple[str, float]],
    article_id: str,
) -> int | None:

    for rank, (candidate_id, _) in enumerate(
        results,
        start=1,
    ):
        if candidate_id == article_id:
            return rank

    return None