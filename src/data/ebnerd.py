from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EBNERD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_small"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)


def as_list(value) -> list:
    """Convert EB-NeRD list-like values into a Python list."""
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    return []


def build_articles(
    source_path: Path = EBNERD_ROOT / "articles.parquet",
) -> pd.DataFrame:
    """
    Convert native EB-NeRD article data into the common article schema.
    """
    articles = pd.read_parquet(source_path)

    required_columns = [
        "article_id",
        "title",
        "subtitle",
        "body",
        "category",
        "category_str",
        "published_time",
    ]

    missing = [
        column
        for column in required_columns
        if column not in articles.columns
    ]

    if missing:
        raise ValueError(
            f"Missing article columns: {missing}"
        )

    result = articles[required_columns].copy()

    # Preserve original content while safely handling nulls.
    for column in ["title", "subtitle", "body"]:
        result[column] = result[column].fillna("").astype(str)

    result["text"] = (
        result["title"].str.strip()
        + " "
        + result["subtitle"].str.strip()
        + " "
        + result["body"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    result["published_time"] = pd.to_datetime(
        result["published_time"],
        errors="coerce",
    )

    if result["article_id"].duplicated().any():
        raise ValueError(
            "Duplicate article IDs found after transformation."
        )

    return result[
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
    ]


def build_interactions(
    behaviors_path: Path,
) -> pd.DataFrame:
    """
    Expand EB-NeRD impressions into one row per candidate article.
    """
    behaviors = pd.read_parquet(behaviors_path)

    required_columns = [
        "impression_id",
        "user_id",
        "impression_time",
        "article_ids_inview",
        "article_ids_clicked",
        "session_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in behaviors.columns
    ]

    if missing:
        raise ValueError(
            f"Missing behavior columns: {missing}"
        )

    rows = []

    for row in behaviors.itertuples(index=False):
        candidates = as_list(row.article_ids_inview)
        clicked = set(as_list(row.article_ids_clicked))

        for article_id in candidates:
            rows.append(
                {
                    "impression_id": row.impression_id,
                    "user_id": row.user_id,
                    "article_id": article_id,
                    "impression_time": row.impression_time,
                    "clicked": int(article_id in clicked),
                    "session_id": row.session_id,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "Interaction transformation produced zero rows."
        )

    result["impression_time"] = pd.to_datetime(
        result["impression_time"],
        errors="coerce",
    )

    result["clicked"] = result["clicked"].astype("int8")

    return result[
        [
            "impression_id",
            "user_id",
            "article_id",
            "impression_time",
            "clicked",
            "session_id",
        ]
    ]


def build_user_history(
    history_path: Path,
) -> pd.DataFrame:
    """
    Expand EB-NeRD's parallel history arrays into one row per event.
    """
    history = pd.read_parquet(history_path)

    required_columns = [
        "user_id",
        "impression_time_fixed",
        "article_id_fixed",
        "read_time_fixed",
        "scroll_percentage_fixed",
    ]

    missing = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing history columns: {missing}"
        )

    rows = []

    for row in history.itertuples(index=False):
        timestamps = as_list(row.impression_time_fixed)
        articles = as_list(row.article_id_fixed)
        read_times = as_list(row.read_time_fixed)
        scrolls = as_list(row.scroll_percentage_fixed)

        lengths = {
            len(timestamps),
            len(articles),
            len(read_times),
            len(scrolls),
        }

        if len(lengths) != 1:
            lengths_list = [
                len(timestamps),
                len(articles),
                len(read_times),
                len(scrolls),
            ]   

            raise ValueError(
                f"Inconsistent history array lengths for user "
                f"{row.user_id}: {lengths_list}"
            )

        for timestamp, article_id, read_time, scroll in zip(
            timestamps,
            articles,
            read_times,
            scrolls,
        ):
            rows.append(
                {
                    "user_id": row.user_id,
                    "article_id": article_id,
                    "timestamp": timestamp,
                    "read_time": read_time,
                    "scroll_percentage": scroll,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "History transformation produced zero rows."
        )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    return result[
        [
            "user_id",
            "article_id",
            "timestamp",
            "read_time",
            "scroll_percentage",
        ]
    ]


def build_split(
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build interactions and history for train or validation.
    """
    behaviors_path = (
        EBNERD_ROOT
        / split_name
        / "behaviors.parquet"
    )

    history_path = (
        EBNERD_ROOT
        / split_name
        / "history.parquet"
    )

    interactions = build_interactions(
        behaviors_path
    )

    history = build_user_history(
        history_path
    )

    return interactions, history


def save_outputs() -> None:
    """
    Build and save the standardized EB-NeRD dataset.
    """
    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits_root = PROCESSED_ROOT / "splits"

    splits_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Articles
    # --------------------------------------------------------------

    print("\nBuilding articles...")
    articles = build_articles()

    articles_path = PROCESSED_ROOT / "articles.parquet"

    articles.to_parquet(
        articles_path,
        index=False,
    )

    print(
        f"Articles: {len(articles):,}"
    )
    print(
        f"Saved: {articles_path}"
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print("\nBuilding train split...")

    train_interactions, train_history = build_split(
        "train"
    )

    train_interactions_path = (
        splits_root / "train.parquet"
    )

    train_history_path = (
        PROCESSED_ROOT / "train_user_history.parquet"
    )

    train_interactions.to_parquet(
        train_interactions_path,
        index=False,
    )

    train_history.to_parquet(
        train_history_path,
        index=False,
    )

    print(
        f"Train interactions: "
        f"{len(train_interactions):,}"
    )

    print(
        f"Train history events: "
        f"{len(train_history):,}"
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    print("\nBuilding validation split...")

    validation_interactions, validation_history = (
        build_split("validation")
    )

    validation_interactions_path = (
        splits_root / "validation.parquet"
    )

    validation_history_path = (
        PROCESSED_ROOT
        / "validation_user_history.parquet"
    )

    validation_interactions.to_parquet(
        validation_interactions_path,
        index=False,
    )

    validation_history.to_parquet(
        validation_history_path,
        index=False,
    )

    print(
        f"Validation interactions: "
        f"{len(validation_interactions):,}"
    )

    print(
        f"Validation history events: "
        f"{len(validation_history):,}"
    )

    print("\nTransformation complete.")


if __name__ == "__main__":
    save_outputs()