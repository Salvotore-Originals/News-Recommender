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

    if result["timestamp"].isna().any():
        raise ValueError(
            "History contains invalid timestamps."
        )

    result = result.sort_values(
        [
            "user_id",
            "timestamp",
            "article_id",
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
            [
                "user_id",
                "category_str",
            ]
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

    Features include:

    - historical interaction count
    - category interaction count
    - category preference
    - mean historical read time
    - mean historical scroll percentage
    - category-specific mean read time
    - category-specific mean scroll percentage
    - candidate seen-before indicator
    - candidate recency in hours

    All historical features obey the strict temporal boundary:

        history_timestamp < impression_time
    """

    interactions = interactions.copy()
    history = history.copy()

    # --------------------------------------------------------------
    # Validate and convert timestamps.
    # --------------------------------------------------------------

    interactions["impression_time"] = pd.to_datetime(
        interactions["impression_time"],
        errors="coerce",
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    if interactions["impression_time"].isna().any():
        raise ValueError(
            "Interactions contain invalid impression timestamps."
        )

    if history["timestamp"].isna().any():
        raise ValueError(
            "History contains invalid timestamps."
        )

    # --------------------------------------------------------------
    # Optional engagement columns.
    #
    # Real EB-NeRD history contains these columns.
    #
    # The temporal leakage unit tests intentionally use minimal
    # synthetic history tables, so the columns may be absent there.
    #
    # Missing columns are represented as NaN and therefore do not
    # contribute to engagement aggregates.
    # --------------------------------------------------------------

    if "read_time" not in history.columns:
        history["read_time"] = float("nan")

    if "scroll_percentage" not in history.columns:
        history["scroll_percentage"] = float("nan")

    # --------------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------------

    interactions = interactions.sort_values(
        [
            "user_id",
            "impression_time",
            "impression_id",
            "article_id",
        ]
    ).reset_index(drop=True)

    history = history.sort_values(
        [
            "user_id",
            "timestamp",
            "article_id",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # Process each user independently.
    #
    # The state below contains ONLY historical events that have
    # timestamp < the current impression timestamp.
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
            [
                "timestamp",
                "article_id",
            ]
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
                    "read_time",
                    "scroll_percentage",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )

        history_pointer = 0

        # ----------------------------------------------------------
        # Basic history state.
        # ----------------------------------------------------------

        total_history = 0

        category_counts = {}

        article_last_seen = {}

        # ----------------------------------------------------------
        # Engagement state.
        # ----------------------------------------------------------

        total_read_time = 0.0
        total_scroll_percentage = 0.0

        read_time_event_count = 0
        scroll_event_count = 0

        category_read_time = {}
        category_scroll_percentage = {}

        category_read_time_count = {}
        category_scroll_count = {}

        feature_rows = []

        # ----------------------------------------------------------
        # Process impressions chronologically.
        # ----------------------------------------------------------

        for interaction in user_interactions.itertuples(
            index=False
        ):

            impression_time = (
                interaction.impression_time
            )

            # ------------------------------------------------------
            # Add history events STRICTLY BEFORE impression time.
            #
            # IMPORTANT:
            #
            #     timestamp < impression_time
            #
            # Same-timestamp and future events are excluded.
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
                    historical_read_time,
                    historical_scroll_percentage,
                ) = history_records[
                    history_pointer
                ]

                # --------------------------------------------------
                # Basic history statistics.
                # --------------------------------------------------

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

                # --------------------------------------------------
                # Candidate last-seen state.
                # --------------------------------------------------

                article_last_seen[
                    historical_article
                ] = timestamp

                # --------------------------------------------------
                # Read-time engagement.
                # --------------------------------------------------

                if pd.notna(
                    historical_read_time
                ):

                    read_time = float(
                        historical_read_time
                    )

                    total_read_time += read_time

                    read_time_event_count += 1

                    category_read_time[
                        historical_category
                    ] = (
                        category_read_time.get(
                            historical_category,
                            0.0,
                        )
                        + read_time
                    )

                    category_read_time_count[
                        historical_category
                    ] = (
                        category_read_time_count.get(
                            historical_category,
                            0,
                        )
                        + 1
                    )

                # --------------------------------------------------
                # Scroll engagement.
                # --------------------------------------------------

                if pd.notna(
                    historical_scroll_percentage
                ):

                    scroll_percentage = float(
                        historical_scroll_percentage
                    )

                    total_scroll_percentage += (
                        scroll_percentage
                    )

                    scroll_event_count += 1

                    category_scroll_percentage[
                        historical_category
                    ] = (
                        category_scroll_percentage.get(
                            historical_category,
                            0.0,
                        )
                        + scroll_percentage
                    )

                    category_scroll_count[
                        historical_category
                    ] = (
                        category_scroll_count.get(
                            historical_category,
                            0,
                        )
                        + 1
                    )

                # --------------------------------------------------
                # Advance history pointer.
                # --------------------------------------------------

                history_pointer += 1

            # ------------------------------------------------------
            # Candidate category.
            # ------------------------------------------------------

            category = interaction.category_str

            # ------------------------------------------------------
            # Category preference.
            # ------------------------------------------------------

            category_count = category_counts.get(
                category,
                0,
            )

            category_preference = (
                category_count
                / max(total_history, 1)
            )

            # ------------------------------------------------------
            # Point-in-time user engagement.
            # ------------------------------------------------------

            if read_time_event_count:

                mean_read_time_before = (
                    total_read_time
                    / read_time_event_count
                )

            else:

                mean_read_time_before = 0.0

            if scroll_event_count:

                mean_scroll_percentage_before = (
                    total_scroll_percentage
                    / scroll_event_count
                )

            else:

                mean_scroll_percentage_before = 0.0

            # ------------------------------------------------------
            # Point-in-time category engagement.
            # ------------------------------------------------------

            category_read_count = (
                category_read_time_count.get(
                    category,
                    0,
                )
            )

            if category_read_count:

                category_mean_read_time_before = (
                    category_read_time.get(
                        category,
                        0.0,
                    )
                    / category_read_count
                )

            else:

                category_mean_read_time_before = 0.0

            category_scroll_count_value = (
                category_scroll_count.get(
                    category,
                    0,
                )
            )

            if category_scroll_count_value:

                category_mean_scroll_percentage_before = (
                    category_scroll_percentage.get(
                        category,
                        0.0,
                    )
                    / category_scroll_count_value
                )

            else:

                category_mean_scroll_percentage_before = 0.0

            # ------------------------------------------------------
            # Candidate previously seen?
            # ------------------------------------------------------

            article_id = interaction.article_id

            last_seen = article_last_seen.get(
                article_id
            )

            if last_seen is None:

                candidate_seen_before = 0

                # A candidate that has never appeared in the
                # user's history has no meaningful elapsed time.
                # Use 0.0 and rely on candidate_seen_before to
                # distinguish "unseen" from a real recency value.
                candidate_recency_hours = 0.0

            else:

                candidate_seen_before = 1

                candidate_recency_hours = (
                    (
                        impression_time
                        - last_seen
                    ).total_seconds()
                    / 3600.0
                )

            # ------------------------------------------------------
            # Store candidate-level feature row.
            # ------------------------------------------------------

            feature_rows.append(
                {
                    "impression_id": (
                        interaction.impression_id
                    ),
                    "user_id": (
                        interaction.user_id
                    ),
                    "article_id": (
                        interaction.article_id
                    ),
                    "impression_time": (
                        impression_time
                    ),
                    "clicked": (
                        interaction.clicked
                    ),
                    "session_id": (
                        interaction.session_id
                    ),
                    "category": (
                        interaction.category
                    ),
                    "category_str": (
                        category
                    ),
                    "category_count_before": (
                        category_count
                    ),
                    "history_count_before": (
                        total_history
                    ),
                    "category_preference": (
                        category_preference
                    ),
                    "mean_read_time_before": (
                        mean_read_time_before
                    ),
                    "mean_scroll_percentage_before": (
                        mean_scroll_percentage_before
                    ),
                    "category_mean_read_time_before": (
                        category_mean_read_time_before
                    ),
                    "category_mean_scroll_percentage_before": (
                        category_mean_scroll_percentage_before
                    ),
                    "candidate_seen_before": (
                        candidate_seen_before
                    ),
                    "candidate_recency_hours": (
                        candidate_recency_hours
                    ),
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
            "mean_read_time_before",
            "mean_scroll_percentage_before",
            "category_mean_read_time_before",
            "category_mean_scroll_percentage_before",
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

    print(
        "\nTemporal feature construction complete."
    )


if __name__ == "__main__":
    main()