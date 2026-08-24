from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from math import log
import re
import time

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)


K1 = 1.5
B = 0.75

HISTORY_QUERY_LENGTH = 10


def tokenize(text: str) -> list[str]:
    """
    Simple deterministic tokenizer shared by article documents
    and user queries.
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    return re.findall(
        r"\b[a-z0-9]+\b",
        text,
    )


def load_articles() -> pd.DataFrame:
    return pd.read_parquet(
        FEATURE_ROOT / "articles.parquet"
    )


def load_history(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / f"{split}_user_history.parquet"
    )


def load_interactions(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / "splits"
        / f"{split}.parquet"
    )


def build_bm25(
    articles: pd.DataFrame,
):
    """
    Build one BM25 index over the complete EB-NeRD
    article corpus.
    """

    documents = (
        articles["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=K1,
        b=B,
    )

    article_to_index = {
        article_id: index
        for index, article_id
        in enumerate(
            articles["article_id"]
        )
    }

    return bm25, article_to_index

def build_candidate_bm25_statistics(
    articles: pd.DataFrame,
    bm25=None,
):
    """
    Precompute the statistics required for candidate-restricted
    BM25 scoring.

    The IDF values and average document length are identical to
    the BM25Okapi corpus.

    We retain the corpus-level statistics while storing only the
    token frequencies required to score individual candidates.
    """

    tokenized_documents = [
        tokenize(text)
        for text in articles["text"]
        .fillna("")
        .astype(str)
    ]

    document_count = len(
        tokenized_documents
    )

    document_lengths = np.array(
        [
            len(tokens)
            for tokens in tokenized_documents
        ],
        dtype=np.float64,
    )

    average_document_length = (
        document_lengths.mean()
    )

    # Use the exact BM25Okapi IDF implementation.  The existing
    # test calls this function with only `articles`, so construct a
    # reference BM25 object when one was not supplied.
    if bm25 is None:
        bm25 = BM25Okapi(
            tokenized_documents,
            k1=K1,
            b=B,
        )

    idf = {
        token: float(value)
        for token, value in bm25.idf.items()
    }

    token_frequencies = []

    for tokens in tokenized_documents:

        frequencies = defaultdict(int)

        for token in tokens:
            frequencies[token] += 1

        token_frequencies.append(
            frequencies
        )

    return {
        "idf": idf,
        "document_lengths": document_lengths,
        "average_document_length": average_document_length,
        "token_frequencies": token_frequencies,
    }

def score_candidates_restricted(
    query_tokens: list[str],
    candidate_ids,
    article_to_index: dict,
    bm25_statistics: dict,
) -> list[tuple]:
    """
    Score ONLY the candidates in an EB-NeRD impression.

    Uses the same corpus-level BM25 statistics as the
    full BM25 scorer.

    Candidate/article IDs are preserved in their original
    type because EB-NeRD article IDs are integer identifiers.
    """

    if not candidate_ids:
        return []

    if not query_tokens:
        return [
            (
                article_id,
                0.0,
            )
            for article_id in candidate_ids
            if article_id in article_to_index
        ]

    idf = bm25_statistics["idf"]

    document_lengths = (
        bm25_statistics[
            "document_lengths"
        ]
    )

    average_document_length = (
        bm25_statistics[
            "average_document_length"
        ]
    )

    token_frequencies = (
        bm25_statistics[
            "token_frequencies"
        ]
    )

    # BM25Okapi uses query-term frequency.
    query_counts = Counter(
        query_tokens
    )

    scored = []

    for article_id in candidate_ids:

        index = article_to_index.get(
            article_id
        )

        if index is None:
            continue

        document_length = (
            document_lengths[index]
        )

        frequencies = (
            token_frequencies[index]
        )

        score = 0.0

        for token, query_frequency in (
            query_counts.items()
        ):

            token_idf = idf.get(
                token,
                0.0,
            )

            frequency = frequencies.get(
                token,
                0,
            )

            if frequency == 0:
                continue

            denominator = (
                frequency
                + K1
                * (
                    1.0
                    - B
                    + B
                    * (
                        document_length
                        / average_document_length
                    )
                )
            )

            score += (
                token_idf
                * query_frequency
                * frequency
                * (K1 + 1.0)
                / denominator
            )

        scored.append(
            (
                article_id,
                float(score),
            )
        )

    scored.sort(
        key=lambda x: (
            -x[1],
            x[0],
        )
    )

    return scored

def build_history_lookup(
    history: pd.DataFrame,
    articles: pd.DataFrame,
) -> dict:
    """
    Map each historical article ID to its title.

    This is used for constructing lexical user queries.
    """

    title_lookup = dict(
        zip(
            articles["article_id"],
            articles["title"].fillna("").astype(str),
        )
    )

    history = history.sort_values(
        [
            "user_id",
            "timestamp",
        ]
    )

    lookup = defaultdict(list)

    for row in history.itertuples(
        index=False
    ):
        title = title_lookup.get(
            row.article_id,
            "",
        )

        if title:
            lookup[row.user_id].append(
                (
                    row.timestamp,
                    row.article_id,
                    title,
                )
            )

    return lookup


def build_query_from_history(
    history_events: list,
    impression_time,
    max_articles: int = HISTORY_QUERY_LENGTH,
) -> list[str]:
    """
    Construct a lexical query using only history events
    strictly before the impression.

    Most recent historical articles are used first.
    Duplicate article IDs are removed.
    """

    eligible = [
        event
        for event in history_events
        if event[0] < impression_time
    ]

    eligible = eligible[
        -max_articles:
    ]

    tokens = []

    seen_articles = set()

    for (
        timestamp,
        article_id,
        title,
    ) in reversed(eligible):

        if article_id in seen_articles:
            continue

        seen_articles.add(article_id)

        tokens.extend(
            tokenize(title)
        )

    return tokens


def score_candidates(
    bm25,
    article_to_index: dict,
    query_tokens: list[str],
    candidate_ids,
) -> list[tuple]:

    if not candidate_ids:
        return []

    if not query_tokens:
        return [
            (
                article_id,
                0.0,
            )
            for article_id in candidate_ids
        ]

    candidate_indices = []

    valid_candidates = []

    for article_id in candidate_ids:

        index = article_to_index.get(
            article_id
        )

        if index is None:
            continue

        candidate_indices.append(index)
        valid_candidates.append(
            article_id
        )

    if not candidate_indices:
        return []

    # rank_bm25's get_scores() returns scores for the
    # entire corpus. We therefore compute the corpus
    # scores once and select only the impression's
    # candidates.
    #
    # This is correct but not the most efficient approach.
    # We will optimize this after validating correctness.

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        zip(
            valid_candidates,
            [
                float(scores[index])
                for index in candidate_indices
            ],
        ),
        key=lambda x: (
            -x[1],
            str(x[0]),
        ),
    )

    return ranked


def evaluate_split(
    split: str,
    bm25,
    article_to_index,
    articles,
    bm25_statistics,
    max_impressions: int | None = None,
):
    """
    Evaluate BM25 ranking against EB-NeRD impression
    candidate sets.
    """

    interactions = load_interactions(
        split
    )

    history = load_history(
        split
    )

    history_lookup = build_history_lookup(
        history,
        articles,
    )

    impression_groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    hits_5 = 0
    hits_10 = 0
    reciprocal_rank_sum = 0

    evaluated = 0

    total_candidates = 0

    start = time.perf_counter()

    for impression_id, group in impression_groups:

        if (
            max_impressions is not None
            and evaluated >= max_impressions
        ):
            break

        group = group.sort_values(
            "article_id"
        )

        user_id = group.iloc[0]["user_id"]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidates = (
            group["article_id"]
            .tolist()
        )

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        user_history = history_lookup.get(
            user_id,
            [],
        )

        query_tokens = (
            build_query_from_history(
                user_history,
                impression_time,
            )
        )

        ranked = score_candidates_restricted(
            query_tokens,
            candidates,
            article_to_index,
            bm25_statistics,
        )

        ranked_ids = [
            article_id
            for article_id, score
            in ranked
        ]

        total_candidates += len(
            ranked_ids
        )

        # ----------------------------------------------------------
        # Hit@5
        # ----------------------------------------------------------

        if clicked and any(
            article_id in clicked
            for article_id
            in ranked_ids[:5]
        ):
            hits_5 += 1

        # ----------------------------------------------------------
        # Hit@10
        # ----------------------------------------------------------

        if clicked and any(
            article_id in clicked
            for article_id
            in ranked_ids[:10]
        ):
            hits_10 += 1

        # ----------------------------------------------------------
        # MRR
        # ----------------------------------------------------------

        reciprocal_rank = 0.0

        for rank, article_id in enumerate(
            ranked_ids,
            start=1,
        ):

            if article_id in clicked:
                reciprocal_rank = (
                    1.0 / rank
                )
                break

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        evaluated += 1

        if evaluated % 10_000 == 0:
            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{split}: "
                f"{evaluated:,} impressions | "
                f"{elapsed:.1f}s"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    if evaluated == 0:
        raise ValueError(
            "No impressions evaluated."
        )

    return {
        "split": split,
        "impressions": evaluated,
        "hit_at_5": hits_5 / evaluated,
        "hit_at_10": hits_10 / evaluated,
        "mrr": reciprocal_rank_sum / evaluated,
        "mean_candidates": (
            total_candidates / evaluated
        ),
        "runtime_seconds": elapsed,
    }


def main():

    print("=" * 80)
    print("EB-NeRD SMALL — BM25 RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load article corpus
    # --------------------------------------------------------------

    print("\nLoading article corpus...")

    articles = load_articles()

    print(
        f"Articles: {len(articles):,}"
    )

    # --------------------------------------------------------------
    # Build standard BM25 index
    # --------------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    start = time.perf_counter()

    bm25, article_to_index = build_bm25(
        articles
    )

    bm25_build_time = (
        time.perf_counter()
        - start
    )

    print(
        f"BM25 index built in "
        f"{bm25_build_time:.2f}s"
    )

    # --------------------------------------------------------------
    # Build candidate-restricted BM25 statistics
    # --------------------------------------------------------------

    print(
        "\nBuilding candidate-restricted "
        "BM25 statistics..."
    )

    start = time.perf_counter()

    bm25_statistics = (
        build_candidate_bm25_statistics(
            articles,
            bm25,
        )
    )

    statistics_build_time = (
        time.perf_counter()
        - start
    )

    print(
        "Candidate statistics built in "
        f"{statistics_build_time:.2f}s"
    )

    # --------------------------------------------------------------
    # FULL TRAIN EVALUATION
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )
    print(
        "TRAIN FULL EVALUATION"
    )
    print(
        "=" * 80
    )

    train_results = evaluate_split(
        "train",
        bm25,
        article_to_index,
        articles,
        bm25_statistics,
        max_impressions=None,
    )

    print(
        "\nTRAIN RESULTS"
    )

    print(
        f"Impressions: "
        f"{train_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{train_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{train_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{train_results['mrr']:.6f}"
    )

    print(
        f"Mean candidates: "
        f"{train_results['mean_candidates']:.2f}"
    )

    print(
        f"Runtime:     "
        f"{train_results['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------------
    # FULL VALIDATION EVALUATION
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )
    print(
        "VALIDATION FULL EVALUATION"
    )
    print(
        "=" * 80
    )

    validation_results = evaluate_split(
        "validation",
        bm25,
        article_to_index,
        articles,
        bm25_statistics,
        max_impressions=None,
    )

    print(
        "\nVALIDATION RESULTS"
    )

    print(
        f"Impressions: "
        f"{validation_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{validation_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{validation_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{validation_results['mrr']:.6f}"
    )

    print(
        f"Mean candidates: "
        f"{validation_results['mean_candidates']:.2f}"
    )

    print(
        f"Runtime:     "
        f"{validation_results['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------------
    # Validate expected impression counts
    # --------------------------------------------------------------

    expected_train_impressions = 232_887

    expected_validation_impressions = 244_647

    if (
        train_results["impressions"]
        != expected_train_impressions
    ):
        raise AssertionError(
            "Unexpected train impression count: "
            f"{train_results['impressions']:,}; "
            f"expected "
            f"{expected_train_impressions:,}"
        )

    if (
        validation_results["impressions"]
        != expected_validation_impressions
    ):
        raise AssertionError(
            "Unexpected validation impression count: "
            f"{validation_results['impressions']:,}; "
            f"expected "
            f"{expected_validation_impressions:,}"
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            train_results,
            validation_results,
        ]
    )

    output_path = (
        RESULT_ROOT
        / "bm25_full.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"Full BM25 results saved to:\n"
        f"{output_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()