from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_testset"
    / "ebnerd_testset"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "test"
)

ARTICLES_PATH = TEST_ROOT / "articles.parquet"
OUTPUT = FEATURE_ROOT / "articles.parquet"


def build_articles(
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical article feature table for
    official EB-NeRD test inference.

    The official test article table does not necessarily
    contain the Small-dataset `text` feature, so construct
    it deterministically from title + subtitle + body.
    """

    required = [
        "article_id",
        "title",
        "subtitle",
        "body",
        "category",
        "subcategory",
        "published_time",
    ]

    missing = [
        column
        for column in required
        if column not in articles.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required article columns: {missing}"
        )

    result = articles[
        required
    ].copy()

    for column in [
        "title",
        "subtitle",
        "body",
    ]:
        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
        )

    # ----------------------------------------------------------
    # Canonical retrieval text
    # ----------------------------------------------------------

    result["text"] = (
        result["title"]
        + " "
        + result["subtitle"]
        + " "
        + result["body"]
    ).str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if result["article_id"].isna().any():
        raise ValueError(
            "Missing article IDs detected."
        )

    if result["article_id"].duplicated().any():
        raise ValueError(
            "Duplicate article IDs detected."
        )

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


def main() -> None:

    print("=" * 80)
    print("EB-NeRD OFFICIAL TEST — ARTICLE FEATURE STORE")
    print("=" * 80)

    print("\n[1/4] Loading official test articles...")

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    print("\n[2/4] Building canonical article features...")

    result = build_articles(
        articles
    )

    print(
        f"       Output columns: "
        f"{len(result.columns)}"
    )

    print("\n[3/4] Validating...")

    print(
        f"       Unique articles: "
        f"{result['article_id'].nunique():,}"
    )

    print(
        f"       Non-empty text: "
        f"{(result['text'].str.len() > 0).sum():,}"
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT,
        index=False,
    )

    print("\n[4/4] Saved:")

    print(
        f"       {OUTPUT}"
    )

    print("\n" + "=" * 80)
    print("EB-NeRD TEST ARTICLE FEATURE STORE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()