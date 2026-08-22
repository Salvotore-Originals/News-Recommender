from __future__ import annotations

import math
from typing import Sequence


def reciprocal_rank(
    results: Sequence[tuple[str, float]],
    relevant_articles: Sequence[str],
) -> float:
    """
    Compute reciprocal rank for one impression.

    RR = 1 / rank of the first relevant article.

    Returns 0.0 when no relevant article is present.
    """

    relevant = set(relevant_articles)

    if not relevant:
        return 0.0

    for rank, (article_id, _) in enumerate(
        results,
        start=1,
    ):
        if article_id in relevant:
            return 1.0 / rank

    return 0.0


def mrr(
    ranked_results: Sequence[
        Sequence[tuple[str, float]]
    ],
    relevant_articles: Sequence[Sequence[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank across impressions.

    MRR = (1 / N) * sum(RR_i)
    """

    if len(ranked_results) != len(
        relevant_articles
    ):
        raise ValueError(
            "ranked_results and relevant_articles "
            "must have the same length."
        )

    if not ranked_results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(
            results,
            relevant,
        )
        for results, relevant in zip(
            ranked_results,
            relevant_articles,
        )
    ]

    return sum(
        reciprocal_ranks
    ) / len(
        reciprocal_ranks
    )


def dcg_at_k(
    results: Sequence[tuple[str, float]],
    relevant_articles: Sequence[str],
    k: int,
) -> float:
    """
    Compute DCG@K using binary relevance.

    DCG@K = sum(rel_i / log2(i + 1))

    where i is the 1-based rank.
    """

    if k <= 0:
        return 0.0

    relevant = set(relevant_articles)

    dcg = 0.0

    for rank, (article_id, _) in enumerate(
        results[:k],
        start=1,
    ):
        if article_id in relevant:
            dcg += 1.0 / math.log2(
                rank + 1
            )

    return dcg


def ndcg_at_k(
    results: Sequence[tuple[str, float]],
    relevant_articles: Sequence[str],
    k: int,
) -> float:
    """
    Compute normalized discounted cumulative gain at K.

    Binary relevance is used.

    nDCG@K = DCG@K / IDCG@K
    """

    if k <= 0:
        return 0.0

    relevant_count = len(
        set(relevant_articles)
    )

    if relevant_count == 0:
        return 0.0

    dcg = dcg_at_k(
        results,
        relevant_articles,
        k,
    )

    ideal_relevant = min(
        relevant_count,
        k,
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant + 1,
        )
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def auc(
    results: Sequence[tuple[str, float]],
    relevant_articles: Sequence[str],
) -> float:
    """
    Compute pairwise AUC for one impression.

    AUC is the probability that a randomly selected
    relevant article receives a higher score than a
    randomly selected non-relevant article.

    Ties receive 0.5 credit.

    Returns 0.0 when positives or negatives are absent.
    """

    relevant = set(relevant_articles)

    positive_scores = [
        float(score)
        for article_id, score in results
        if article_id in relevant
    ]

    negative_scores = [
        float(score)
        for article_id, score in results
        if article_id not in relevant
    ]

    if not positive_scores or not negative_scores:
        return 0.0

    correct = 0.0
    total = (
        len(positive_scores)
        * len(negative_scores)
    )

    for positive in positive_scores:
        for negative in negative_scores:

            if positive > negative:
                correct += 1.0

            elif positive == negative:
                correct += 0.5

    return correct / total


def evaluate_competition_ranking(
    results: Sequence[tuple[str, float]],
    relevant_articles: Sequence[str],
) -> dict[str, float]:
    """
    Compute all competition-style ranking metrics
    for one impression.
    """

    return {
        "auc": auc(
            results,
            relevant_articles,
        ),
        "mrr": reciprocal_rank(
            results,
            relevant_articles,
        ),
        "ndcg_at_5": ndcg_at_k(
            results,
            relevant_articles,
            5,
        ),
        "ndcg_at_10": ndcg_at_k(
            results,
            relevant_articles,
            10,
        ),
    }