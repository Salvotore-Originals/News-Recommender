from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import bisect
import time

import pandas as pd


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


def build_article_category_lookup(
    articles: pd.DataFrame,
) -> dict:

    required = {
        "article_id",
        "category_str",
    }

    missing = required - set(
        articles.columns
    )

    if missing:
        raise ValueError(
            "Article feature store is missing "
            f"columns: {sorted(missing)}"
        )

    return dict(
        zip(
            articles["article_id"],
            articles["category_str"],
        )
    )


def build_temporal_user_category_lookup(
    history: pd.DataFrame,
    article_categories: dict,
):
    """
    Build chronological cumulative category counts.

    Each snapshot represents the user's behavioral state
    AFTER all events at that timestamp.

    During scoring we use only snapshots with:

        timestamp < impression_time

    Therefore events occurring at the same timestamp as an
    impression cannot leak into the recommendation.
    """

    required = {
        "user_id",
        "article_id",
        "timestamp",
    }

    missing = required - set(
        history.columns
    )

    if missing:
        raise ValueError(
            "History is missing columns: "
            f"{sorted(missing)}"
        )

    history = history.sort_values(
        [
            "user_id",
            "timestamp",
        ],
        kind="mergesort",
    )

    lookup = defaultdict(list)

    user_category_counts = defaultdict(
        lambda: defaultdict(int)
    )

    user_total_counts = defaultdict(int)

    # --------------------------------------------------------------
    # Process one user at a time.
    # --------------------------------------------------------------

    for user_id, user_history in history.groupby(
        "user_id",
        sort=False,
    ):

        user_category_counts[user_id] = (
            defaultdict(int)
        )

        user_total_counts[user_id] = 0

        # ----------------------------------------------------------
        # Process events timestamp by timestamp.
        # ----------------------------------------------------------

        for timestamp, timestamp_group in (
            user_history.groupby(
                "timestamp",
                sort=True,
            )
        ):

            # First apply ALL events at this timestamp.
            for row in timestamp_group.itertuples(
                index=False
            ):

                article_id = row.article_id

                category = article_categories.get(
                    article_id
                )

                if (
                    category is not None
                    and not pd.isna(category)
                ):

                    user_category_counts[
                        user_id
                    ][category] += 1

                    user_total_counts[
                        user_id
                    ] += 1

            # ------------------------------------------------------
            # Store state AFTER this timestamp.
            # ------------------------------------------------------

            lookup[user_id].append(
                (
                    timestamp,
                    dict(
                        user_category_counts[
                            user_id
                        ]
                    ),
                    user_total_counts[
                        user_id
                    ],
                )
            )

    return dict(lookup)

def get_behavioral_preference(
    user_id,
    category,
    impression_time,
    temporal_lookup,
) -> float:
    """
    Return:

        count(user, category before t)
        --------------------------------
        count(all categories for user before t)

    """

    if (
        category is None
        or pd.isna(category)
    ):
        return 0.0

    snapshots = temporal_lookup.get(
        user_id,
        [],
    )

    if not snapshots:
        return 0.0

    timestamps = [
        snapshot[0]
        for snapshot in snapshots
    ]

    # Strictly before impression_time.
    position = bisect.bisect_left(
        timestamps,
        impression_time,
    )

    if position == 0:
        return 0.0

    (
        category_counts,
        total_count,
    ) = snapshots[position - 1][1:]

    if total_count == 0:
        return 0.0

    return float(
        category_counts.get(
            category,
            0,
        )
        / total_count
    )


def score_candidates(
    user_id,
    impression_time,
    candidate_ids,
    article_categories,
    temporal_lookup,
) -> list[tuple]:

    scored = []

    for article_id in candidate_ids:

        category = article_categories.get(
            article_id
        )

        score = get_behavioral_preference(
            user_id,
            category,
            impression_time,
            temporal_lookup,
        )

        scored.append(
            (
                article_id,
                score,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[1],
            str(x[0]),
        )
    )

    return scored


def evaluate_split(
    split: str,
    article_categories,
    temporal_lookup,
    max_impressions: int | None = None,
):

    interactions = load_interactions(
        split
    )

    impression_groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    hits_5 = 0
    hits_10 = 0
    reciprocal_rank_sum = 0.0

    evaluated = 0
    total_candidates = 0

    start = time.perf_counter()

    for (
        impression_id,
        group,
    ) in impression_groups:

        if (
            max_impressions is not None
            and evaluated >= max_impressions
        ):
            break

        group = group.sort_values(
            "article_id"
        )

        user_id = group.iloc[0][
            "user_id"
        ]

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

        ranked = score_candidates(
            user_id,
            impression_time,
            candidates,
            article_categories,
            temporal_lookup,
        )

        ranked_ids = [
            article_id
            for article_id, score
            in ranked
        ]

        total_candidates += len(
            ranked_ids
        )

        if any(
            article_id in clicked
            for article_id in ranked_ids[:5]
        ):
            hits_5 += 1

        if any(
            article_id in clicked
            for article_id in ranked_ids[:10]
        ):
            hits_10 += 1

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
    print("EB-NeRD SMALL — BEHAVIORAL RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load article features
    # --------------------------------------------------------------

    print(
        "\nLoading article features..."
    )

    articles = load_articles()

    print(
        f"Articles: {len(articles):,}"
    )

    article_categories = (
        build_article_category_lookup(
            articles
        )
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    print(
        "\nLoading train history..."
    )

    train_history = load_history(
        "train"
    )

    print(
        f"Train history events: "
        f"{len(train_history):,}"
    )

    print(
        "\nBuilding temporal train "
        "behavioral statistics..."
    )

    train_temporal_lookup = (
        build_temporal_user_category_lookup(
            train_history,
            article_categories,
        )
    )

    print(
        "\nRunning full train evaluation..."
    )

    train_results = evaluate_split(
        "train",
        article_categories,
        train_temporal_lookup,
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
    # VALIDATION
    # --------------------------------------------------------------

    print(
        "\nLoading validation history..."
    )

    validation_history = load_history(
        "validation"
    )

    print(
        f"Validation history events: "
        f"{len(validation_history):,}"
    )

    print(
        "\nBuilding temporal validation "
        "behavioral statistics..."
    )

    validation_temporal_lookup = (
        build_temporal_user_category_lookup(
            validation_history,
            article_categories,
        )
    )

    print(
        "\nRunning full validation evaluation..."
    )

    validation_results = evaluate_split(
        "validation",
        article_categories,
        validation_temporal_lookup,
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
    # Verify expected impression counts
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
    # Save full results
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
        / "behavioral_full.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Full behavioral results saved to:"
    )

    print(
        output_path
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()