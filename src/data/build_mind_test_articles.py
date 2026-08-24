from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_NEWS = (
    PROJECT_ROOT
    / "data/raw/mind/test/news.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/mind_test"
)

OUTPUT = OUTPUT_DIR / "articles.parquet"


def main() -> None:

    print("=" * 70)
    print("MIND LARGE TEST ARTICLE STORE")
    print("=" * 70)

    columns = [
        "article_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    print("\n[1/3] Loading official news.tsv...")

    articles = pd.read_csv(
        RAW_NEWS,
        sep="\t",
        header=None,
        names=columns,
        dtype={
            "article_id": "string",
            "category": "string",
            "subcategory": "string",
            "title": "string",
            "abstract": "string",
        },
        keep_default_na=False,
    )

    print(
        f"       Articles: {len(articles):,}"
    )

    # --------------------------------------------------------
    # Keep only fields required by our retrieval pipeline.
    # --------------------------------------------------------

    articles = articles[
        [
            "article_id",
            "category",
            "subcategory",
            "title",
            "abstract",
        ]
    ].copy()

    articles["title"] = (
        articles["title"]
        .fillna("")
        .astype(str)
    )

    articles["abstract"] = (
        articles["abstract"]
        .fillna("")
        .astype(str)
    )

    articles["text"] = (
    articles["title"]
    + " "
    + articles["abstract"]
)

    if articles["article_id"].duplicated().any():
        raise ValueError(
            "Duplicate article IDs detected."
        )

    if articles["article_id"].isna().any():
        raise ValueError(
            "Missing article IDs detected."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    articles.to_parquet(
        OUTPUT,
        index=False,
    )

    print("\n[2/3] Validation...")
    print(
        f"       Unique articles: "
        f"{articles['article_id'].nunique():,}"
    )

    print("\n[3/3] Saved:")
    print(f"       {OUTPUT}")

    print("\n" + "=" * 70)
    print("ARTICLE STORE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()