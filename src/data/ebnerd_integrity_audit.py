from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EBNERD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_small"
)

ARTICLES_PATH = EBNERD_ROOT / "articles.parquet"
TRAIN_BEHAVIORS_PATH = EBNERD_ROOT / "train" / "behaviors.parquet"
TRAIN_HISTORY_PATH = EBNERD_ROOT / "train" / "history.parquet"
VALIDATION_BEHAVIORS_PATH = (
    EBNERD_ROOT / "validation" / "behaviors.parquet"
)
VALIDATION_HISTORY_PATH = (
    EBNERD_ROOT / "validation" / "history.parquet"
)

def as_list(value) -> list:
    """Convert list/tuple/NumPy-array values into a Python list."""
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    return []

def flatten(values: Iterable) -> list:
    """Flatten a column containing list-like values."""
    result = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            result.extend(value)

    return result


def audit_articles(articles: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("1. ARTICLE CORPUS AUDIT")
    print("=" * 80)

    article_ids = articles["article_id"]

    print(f"Article rows: {len(articles):,}")
    print(f"Unique article IDs: {article_ids.nunique():,}")
    print(f"Duplicate article IDs: {article_ids.duplicated().sum():,}")

    print("\nPublication time:")
    print(f"  Min: {articles['published_time'].min()}")
    print(f"  Max: {articles['published_time'].max()}")

    print("\nMissing values in important fields:")

    important = [
        "article_id",
        "title",
        "subtitle",
        "body",
        "category",
        "category_str",
        "published_time",
    ]

    for column in important:
        missing = articles[column].isna().sum()
        print(f"  {column:20s}: {missing:,}")


def audit_behaviors(
    behaviors: pd.DataFrame,
    articles: pd.DataFrame,
    history: pd.DataFrame,
    split_name: str,
) -> None:
    print("\n" + "=" * 80)
    print(f"2. {split_name.upper()} BEHAVIOR AUDIT")
    print("=" * 80)

    print(f"Behavior rows: {len(behaviors):,}")
    print(
        f"Unique impressions: "
        f"{behaviors['impression_id'].nunique():,}"
    )
    print(
        f"Duplicate impression IDs: "
        f"{behaviors['impression_id'].duplicated().sum():,}"
    )

    # ------------------------------------------------------------------
    # Candidate sets
    # ------------------------------------------------------------------

    candidate_lengths = behaviors["article_ids_inview"].apply(
        lambda x: len(as_list(x))
    )

    click_lengths = behaviors["article_ids_clicked"].apply(
        lambda x: len(as_list(x))
    )

    print("\nCandidate-set statistics:")
    print(f"  Empty candidate sets: {(candidate_lengths == 0).sum():,}")
    print(f"  Mean candidates: {candidate_lengths.mean():.2f}")
    print(f"  Median candidates: {candidate_lengths.median():.2f}")
    print(f"  Maximum candidates: {candidate_lengths.max():,}")

    print("\nClick statistics:")
    print(f"  Impressions with zero clicks: {(click_lengths == 0).sum():,}")
    print(
        f"  Impressions with one click: "
        f"{(click_lengths == 1).sum():,}"
    )
    print(
        f"  Impressions with multiple clicks: "
        f"{(click_lengths > 1).sum():,}"
    )
    print(f"  Maximum clicks in an impression: {click_lengths.max():,}")

    # ------------------------------------------------------------------
    # Click ⊆ candidate validation
    # ------------------------------------------------------------------

    invalid_click_sets = 0

    for candidates, clicks in zip(
        behaviors["article_ids_inview"],
        behaviors["article_ids_clicked"],
    ):
        candidates = set(as_list(candidates))
        clicks = set(as_list(clicks))

    if not clicks.issubset(candidates):
        invalid_click_sets += 1

    print("\nClick/candidate consistency:")
    print(
        f"  Impressions where clicked IDs are NOT "
        f"a subset of in-view IDs: {invalid_click_sets:,}"
    )

    # ------------------------------------------------------------------
    # Article coverage
    # ------------------------------------------------------------------

    article_corpus = set(articles["article_id"].astype("int64"))

    candidate_ids = set()

    for candidates in behaviors["article_ids_inview"]:
        candidate_ids.update(as_list(candidates))

    clicked_ids = set()

    for clicks in behaviors["article_ids_clicked"]:
        clicked_ids.update(as_list(clicks))

    missing_candidates = candidate_ids - article_corpus
    missing_clicks = clicked_ids - article_corpus

    print("\nArticle-ID coverage:")
    print(f"  Unique candidate article IDs: {len(candidate_ids):,}")
    print(f"  Candidate IDs missing from corpus: {len(missing_candidates):,}")
    print(f"  Unique clicked article IDs: {len(clicked_ids):,}")
    print(f"  Clicked IDs missing from corpus: {len(missing_clicks):,}")

    if candidate_ids:
        coverage = (
            1
            - len(missing_candidates) / len(candidate_ids)
        ) * 100

        print(f"  Candidate article coverage: {coverage:.4f}%")

    # ------------------------------------------------------------------
    # User coverage
    # ------------------------------------------------------------------

    behavior_users = set(behaviors["user_id"].astype("int64"))
    history_users = set(history["user_id"].astype("int64"))

    missing_history_users = behavior_users - history_users

    print("\nUser/history coverage:")
    print(f"  Unique behavior users: {len(behavior_users):,}")
    print(f"  Unique history users: {len(history_users):,}")
    print(
        f"  Behavior users missing from history: "
        f"{len(missing_history_users):,}"
    )

    if behavior_users:
        user_coverage = (
            1
            - len(missing_history_users) / len(behavior_users)
        ) * 100

        print(f"  User history coverage: {user_coverage:.4f}%")

    # ------------------------------------------------------------------
    # Temporal range
    # ------------------------------------------------------------------

    print("\nBehavior timestamp range:")
    print(f"  Min: {behaviors['impression_time'].min()}")
    print(f"  Max: {behaviors['impression_time'].max()}")

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    print("\nBehavior missing values:")

    for column in behaviors.columns:
        missing = behaviors[column].isna().sum()

        if missing:
            percentage = missing / len(behaviors) * 100
            print(
                f"  {column:25s}: "
                f"{missing:,} ({percentage:.2f}%)"
            )


def audit_history(history: pd.DataFrame, split_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"3. {split_name.upper()} HISTORY AUDIT")
    print("=" * 80)

    print(f"History rows/users: {len(history):,}")
    print(f"Unique users: {history['user_id'].nunique():,}")

    # Each row contains arrays of historical interactions.
    history_lengths = history["article_id_fixed"].apply(
        lambda x: len(as_list(x))
)

    print("\nHistory length:")
    print(f"  Mean: {history_lengths.mean():.2f}")
    print(f"  Median: {history_lengths.median():.2f}")
    print(f"  Minimum: {history_lengths.min():,}")
    print(f"  Maximum: {history_lengths.max():,}")
    print(f"  Empty histories: {(history_lengths == 0).sum():,}")

    # Check parallel-array lengths.
    parallel_columns = [
        "impression_time_fixed",
        "scroll_percentage_fixed",
        "article_id_fixed",
        "read_time_fixed",
    ]

    inconsistent_rows = 0

    for _, row in history.iterrows():
        lengths = []

    for column in parallel_columns:
        lengths.append(len(as_list(row[column])))

    if len(set(lengths)) != 1:
        inconsistent_rows += 1

    print("\nParallel-array consistency:")
    print(
        f"  Rows with inconsistent array lengths: "
        f"{inconsistent_rows:,}"
    )


def compare_temporal_order(
    train_behaviors: pd.DataFrame,
    validation_behaviors: pd.DataFrame,
) -> None:
    print("\n" + "=" * 80)
    print("4. TRAIN → VALIDATION TEMPORAL AUDIT")
    print("=" * 80)

    train_min = train_behaviors["impression_time"].min()
    train_max = train_behaviors["impression_time"].max()

    validation_min = validation_behaviors["impression_time"].min()
    validation_max = validation_behaviors["impression_time"].max()

    print(f"Train minimum:      {train_min}")
    print(f"Train maximum:      {train_max}")
    print(f"Validation minimum: {validation_min}")
    print(f"Validation maximum: {validation_max}")

    print("\nTemporal ordering:")

    if train_max < validation_min:
        print("  PASS: Train ends before validation begins.")
    else:
        print(
            "  WARNING: Train and validation time ranges overlap."
        )


def main() -> None:
    print("=" * 80)
    print("EB-NeRD SMALL — INTEGRITY AUDIT")
    print("=" * 80)

    print(f"\nDataset root:\n{EBNERD_ROOT}")

    # Load
    articles = pd.read_parquet(ARTICLES_PATH)

    train_behaviors = pd.read_parquet(
        TRAIN_BEHAVIORS_PATH
    )

    train_history = pd.read_parquet(
        TRAIN_HISTORY_PATH
    )

    validation_behaviors = pd.read_parquet(
        VALIDATION_BEHAVIORS_PATH
    )

    validation_history = pd.read_parquet(
        VALIDATION_HISTORY_PATH
    )

    # Audits
    audit_articles(articles)

    audit_behaviors(
        train_behaviors,
        articles,
        train_history,
        "train",
    )

    audit_history(
        train_history,
        "train",
    )

    audit_behaviors(
        validation_behaviors,
        articles,
        validation_history,
        "validation",
    )

    audit_history(
        validation_history,
        "validation",
    )

    compare_temporal_order(
        train_behaviors,
        validation_behaviors,
    )

    print("\n" + "=" * 80)
    print("INTEGRITY AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()