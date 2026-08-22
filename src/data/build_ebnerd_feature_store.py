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


def load_processed_data():
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


def build_articles(
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical article feature table.
    """

    result = articles[
        [
            "article_id",
            "title",
            "subtitle",
            "body",
            "text",
            "category",
            "category_str",
            "published_time",
        ]
    ].copy()

    result["title_length"] = (
        result["title"]
        .str.len()
        .astype("int32")
    )

    result["body_length"] = (
        result["body"]
        .str.len()
        .astype("int32")
    )

    result["text_length"] = (
        result["text"]
        .str.len()
        .astype("int32")
    )

    return result


def build_users(
    train_history: pd.DataFrame,
    validation_history: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive user-level features.

    Article metadata is joined here because the normalized
    history table intentionally does not contain category fields.

    These are descriptive aggregate statistics. They are NOT
    used directly as point-in-time validation ranking features.
    """

    history = pd.concat(
        [
            train_history.assign(
                source_split="train"
            ),
            validation_history.assign(
                source_split="validation"
            ),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Attach article category.
    # --------------------------------------------------------------

    category_lookup = articles[
        [
            "article_id",
            "category_str",
        ]
    ].drop_duplicates(
        "article_id"
    )

    history = history.merge(
        category_lookup,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_categories = (
        history["category_str"].isna().sum()
    )

    if missing_categories:
        raise ValueError(
            f"{missing_categories:,} history rows "
            "could not be mapped to an article category."
        )

    # --------------------------------------------------------------
    # Aggregate user-level statistics.
    # --------------------------------------------------------------

    users = (
        history
        .groupby(
            "user_id",
            as_index=False,
        )
        .agg(
            history_count=(
                "article_id",
                "count",
            ),
            unique_articles=(
                "article_id",
                "nunique",
            ),
            unique_categories=(
                "category_str",
                "nunique",
            ),
            total_read_time=(
                "read_time",
                "sum",
            ),
            mean_read_time=(
                "read_time",
                "mean",
            ),
            mean_scroll_percentage=(
                "scroll_percentage",
                "mean",
            ),
        )
    )

    return users

def build_article_stats(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive article interaction statistics.

    These statistics are NOT treated as point-in-time features
    in the core temporal ranking experiment.
    """

    interactions = pd.concat(
        [
            train.assign(source_split="train"),
            validation.assign(source_split="validation"),
        ],
        ignore_index=True,
    )

    stats = (
        interactions
        .groupby("article_id")
        .agg(
            interaction_count=(
                "article_id",
                "size",
            ),
            unique_users=(
                "user_id",
                "nunique",
            ),
            click_count=(
                "clicked",
                "sum",
            ),
        )
        .reset_index()
    )

    stats["click_rate"] = (
        stats["click_count"]
        / stats["interaction_count"]
    )

    return stats


def build_user_category_preferences(
    train_history: pd.DataFrame,
    validation_history: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive user/category preference statistics.

    Article categories are joined from the canonical article table
    because normalized history intentionally contains only event-level
    fields.

    IMPORTANT:
    These are descriptive aggregate statistics over the supplied
    train + validation histories. They are NOT used as point-in-time
    ranking features. For ranking, use train_temporal.parquet and
    validation_temporal.parquet.
    """

    history = pd.concat(
        [
            train_history,
            validation_history,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Attach article category.
    # --------------------------------------------------------------

    category_lookup = articles[
        [
            "article_id",
            "category_str",
        ]
    ].drop_duplicates(
        "article_id"
    )

    history = history.merge(
        category_lookup,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_categories = (
        history["category_str"].isna().sum()
    )

    if missing_categories:
        raise ValueError(
            f"{missing_categories:,} history rows "
            "could not be mapped to an article category."
        )

    # --------------------------------------------------------------
    # Count user/category interactions.
    # --------------------------------------------------------------

    counts = (
        history
        .groupby(
            [
                "user_id",
                "category_str",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "category_interaction_count"
            }
        )
    )

    # --------------------------------------------------------------
    # Total interactions for each user.
    # --------------------------------------------------------------

    totals = (
        counts
        .groupby("user_id")[
            "category_interaction_count"
        ]
        .sum()
        .rename(
            "total_interactions"
        )
        .reset_index()
    )

    result = counts.merge(
        totals,
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------------
    # Preference:
    #
    # P(category | user)
    #
    # = category interactions / total interactions
    # --------------------------------------------------------------

    result["preference"] = (
        result["category_interaction_count"]
        / result["total_interactions"]
    )

    return result[
        [
            "user_id",
            "category_str",
            "category_interaction_count",
            "total_interactions",
            "preference",
        ]
    ]


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FEATURE_ROOT / filename

    dataframe.to_parquet(
        path,
        index=False,
    )

    print(
        f"{filename}: "
        f"{len(dataframe):,} rows, "
        f"{len(dataframe.columns)} columns"
    )

    print(f"Saved: {path}")


def main():

    print("=" * 80)
    print("EB-NeRD SMALL — FEATURE STORE")
    print("=" * 80)

    (
        articles,
        train,
        validation,
        train_history,
        validation_history,
    ) = load_processed_data()

    # --------------------------------------------------------------
    # Articles
    # --------------------------------------------------------------

    print("\n[1/4] Building article features...")

    article_features = build_articles(
        articles
    )

    save_table(
        article_features,
        "articles.parquet",
    )

    # --------------------------------------------------------------
    # Users
    # --------------------------------------------------------------

    print("\n[2/4] Building user features...")

    user_features = build_users(
        train_history,
        validation_history,
        articles,
    )

    save_table(
        user_features,
        "users.parquet",
    )

    # --------------------------------------------------------------
    # Article statistics
    # --------------------------------------------------------------

    print("\n[3/4] Building article statistics...")

    article_stats = build_article_stats(
        train,
        validation,
    )

    save_table(
        article_stats,
        "article_stats.parquet",
    )

    # --------------------------------------------------------------
    # User/category preferences
    # --------------------------------------------------------------

    print(
        "\n[4/4] Building user/category preferences..."
    )

    preferences = build_user_category_preferences(
        train_history,
        validation_history,
        articles,
    )

    save_table(
        preferences,
        "user_category_preferences.parquet",
    )

    print("\nFeature store construction complete.")


if __name__ == "__main__":
    main()