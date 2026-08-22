from __future__ import annotations

from pathlib import Path

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


def load_data():
    """Load standardized EB-NeRD tables."""

    articles = pd.read_parquet(
        PROCESSED_ROOT / "articles.parquet"
    )

    train = pd.read_parquet(
        PROCESSED_ROOT / "splits" / "train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_ROOT / "splits" / "validation.parquet"
    )

    train_history = pd.read_parquet(
        PROCESSED_ROOT / "train_user_history.parquet"
    )

    validation_history = pd.read_parquet(
        PROCESSED_ROOT
        / "validation_user_history.parquet"
    )

    return (
        articles,
        train,
        validation,
        train_history,
        validation_history,
    )


def attach_article_categories(
    interactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach article category information to candidate interactions.
    """

    category_lookup = articles[
        [
            "article_id",
            "category",
            "category_str",
        ]
    ].copy()

    result = interactions.merge(
        category_lookup,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing = result["category_str"].isna().sum()

    if missing:
        raise ValueError(
            f"{missing:,} interaction rows have no "
            "article category."
        )

    return result


def prepare_history(
    history: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach article category to historical events.
    """

    category_lookup = articles[
        [
            "article_id",
            "category",
            "category_str",
        ]
    ].copy()

    result = history.merge(
        category_lookup,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing = result["category_str"].isna().sum()

    if missing:
        raise ValueError(
            f"{missing:,} history rows have no "
            "article category."
        )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result = result.sort_values(
        [
            "user_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    return result


def build_history_summary(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cumulative temporal history statistics.

    Every row represents one historical event and contains
    the number of historical events available immediately
    BEFORE that event.
    """

    history = history.copy()

    # Number of previous interactions for each user.
    history["history_count_before"] = (
        history.groupby("user_id")
        .cumcount()
    )

    # Number of previous interactions with the same category.
    history["category_count_before"] = (
        history.groupby(
            ["user_id", "category_str"]
        ).cumcount()
    )

    return history


def build_temporal_features(
    interactions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build point-in-time behavioral features.

    For every impression at time t, only history events with
    timestamp < t are allowed to contribute to the features.
    """

    interactions = interactions.copy()
    history = history.copy()

    interactions["impression_time"] = pd.to_datetime(
        interactions["impression_time"],
        errors="coerce",
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------------

    interactions = interactions.sort_values(
        ["user_id", "impression_time", "impression_id", "article_id"]
    ).reset_index(drop=True)

    history = history.sort_values(
        ["user_id", "timestamp", "article_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # We process each user independently.
    #
    # State maintained while moving forward in time:
    #
    #   total_history
    #   category_counts
    #   article_last_seen
    #
    # The state contains ONLY events strictly before the current
    # impression.
    # --------------------------------------------------------------

    output_chunks = []

    history_groups = {
        user_id: group
        for user_id, group in history.groupby(
            "user_id",
            sort=False,
        )
    }

    interaction_groups = interactions.groupby(
        "user_id",
        sort=False,
    )

    for user_id, user_interactions in interaction_groups:

        user_interactions = user_interactions.sort_values(
            [
                "impression_time",
                "impression_id",
                "article_id",
            ]
        ).copy()

        user_history = history_groups.get(
            user_id
        )

        if user_history is None:
            user_history = history.iloc[0:0].copy()

        user_history = user_history.sort_values(
            ["timestamp", "article_id"]
        )

        # ----------------------------------------------------------
        # Convert history into records for efficient traversal.
        # ----------------------------------------------------------

        history_records = list(
            user_history[
                [
                    "timestamp",
                    "article_id",
                    "category_str",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )

        history_pointer = 0

        total_history = 0

        category_counts = {}

        article_last_seen = {}

        feature_rows = []

        # ----------------------------------------------------------
        # Process impressions chronologically.
        # ----------------------------------------------------------

        for interaction in user_interactions.itertuples(
            index=False
        ):

            impression_time = interaction.impression_time

            # ------------------------------------------------------
            # Add history events STRICTLY BEFORE impression time.
            # ------------------------------------------------------

            while (
                history_pointer < len(history_records)
                and history_records[
                    history_pointer
                ][0] < impression_time
            ):
                (
                    timestamp,
                    historical_article,
                    historical_category,
                ) = history_records[history_pointer]

                total_history += 1

                category_counts[
                    historical_category
                ] = (
                    category_counts.get(
                        historical_category,
                        0,
                    )
                    + 1
                )

                article_last_seen[
                    historical_article
                ] = timestamp

                history_pointer += 1

            # ------------------------------------------------------
            # Category preference.
            # ------------------------------------------------------

            category = interaction.category_str

            category_count = category_counts.get(
                category,
                0,
            )

            category_preference = (
                category_count
                / max(total_history, 1)
            )

            # ------------------------------------------------------
            # Candidate previously seen?
            # ------------------------------------------------------

            article_id = interaction.article_id

            last_seen = article_last_seen.get(
                article_id
            )

            if last_seen is None:
                candidate_seen_before = 0
                candidate_recency_hours = float("inf")

            else:
                candidate_seen_before = 1

                candidate_recency_hours = (
                    (
                        impression_time
                        - last_seen
                    ).total_seconds()
                    / 3600.0
                )

            feature_rows.append(
                {
                    "impression_id": interaction.impression_id,
                    "user_id": interaction.user_id,
                    "article_id": interaction.article_id,
                    "impression_time": impression_time,
                    "clicked": interaction.clicked,
                    "session_id": interaction.session_id,
                    "category": interaction.category,
                    "category_str": category,
                    "category_count_before": category_count,
                    "history_count_before": total_history,
                    "category_preference": category_preference,
                    "candidate_seen_before": candidate_seen_before,
                    "candidate_recency_hours": candidate_recency_hours,
                }
            )

        if feature_rows:
            output_chunks.append(
                pd.DataFrame(feature_rows)
            )

    if not output_chunks:
        raise ValueError(
            "Temporal feature construction produced zero rows."
        )

    result = pd.concat(
        output_chunks,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Final deterministic ordering.
    # --------------------------------------------------------------

    result = result.sort_values(
        [
            "impression_time",
            "impression_id",
            "article_id",
        ]
    ).reset_index(drop=True)

    return result[
        [
            "impression_id",
            "user_id",
            "article_id",
            "impression_time",
            "clicked",
            "session_id",
            "category",
            "category_str",
            "category_count_before",
            "history_count_before",
            "category_preference",
            "candidate_seen_before",
            "candidate_recency_hours",
        ]
    ]

def process_split(
    interactions: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    split_name: str,
) -> None:
    """Build and save temporal features for one split."""

    print(
        f"\nBuilding temporal features: {split_name}"
    )

    interactions = attach_article_categories(
        interactions,
        articles,
    )

    history = prepare_history(
        history,
        articles,
    )

    features = build_temporal_features(
        interactions,
        history,
    )

    output_path = (
        FEATURE_ROOT
        / f"{split_name}_temporal.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Saved: {output_path}"
    )


def main() -> None:

    print("=" * 80)
    print("EB-NeRD SMALL — TEMPORAL FEATURE CONSTRUCTION")
    print("=" * 80)

    (
        articles,
        train,
        validation,
        train_history,
        validation_history,
    ) = load_data()

    process_split(
        train,
        train_history,
        articles,
        "train",
    )

    process_split(
        validation,
        validation_history,
        articles,
        "validation",
    )

    print("\nTemporal feature construction complete.")


if __name__ == "__main__":
    main()