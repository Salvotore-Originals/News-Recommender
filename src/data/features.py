from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_MIND = ROOT / "data" / "raw" / "mind"

PROCESSED_MIND = (
    ROOT
    / "data"
    / "processed"
    / "mind"
)

FEATURE_MIND = (
    ROOT
    / "data"
    / "features"
    / "mind"
)

HISTORY_PATH = (
    PROCESSED_MIND
    / "user_history.parquet"
)

TRAIN_PATH = (
    PROCESSED_MIND
    / "splits"
    / "train.parquet"
)


# ============================================================
# RAW MIND NEWS LOADER
# ============================================================

def load_mind_news(split: str) -> pd.DataFrame:

    news_path = (
        RAW_MIND
        / split
        / "news.tsv"
    )

    if not news_path.exists():
        raise FileNotFoundError(
            f"MIND news file not found:\n{news_path}"
        )

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

    print(
        f"\nLoading {split} news..."
    )

    articles = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=columns,
        quoting=3,
        keep_default_na=False,
        dtype=str,
    )

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
        .str.strip()
    )

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

    articles["category"] = (
        articles["category"]
        .fillna("")
        .astype(str)
    )

    articles["subcategory"] = (
        articles["subcategory"]
        .fillna("")
        .astype(str)
    )

    articles["text"] = (
        articles["title"]
        + " "
        + articles["abstract"]
    ).str.strip()

    # Remove accidental duplicate article IDs.
    articles = (
        articles
        .drop_duplicates(
            subset=["article_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"       Raw articles: {len(articles):,}"
    )

    return articles


# ============================================================
# CREATE SPLIT-SPECIFIC ARTICLE FEATURES
# ============================================================

def build_article_features(
    split: str,
) -> pd.DataFrame:

    print(
        f"\n[{split.upper()}] Building article features..."
    )

    articles = load_mind_news(
        split
    )

    columns = [
        "article_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "text",
        "title_entities",
        "abstract_entities",
    ]

    articles = articles[
        columns
    ]

    output = (
        FEATURE_MIND
        / f"{split}_articles.parquet"
    )

    articles.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Articles: {len(articles):,}"
    )

    print(
        f"       Saved: {output}"
    )

    return articles


# ============================================================
# CREATE USER FEATURES
# ============================================================

def build_user_features():

    print(
        "\n[3/4] Building user features..."
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "user_id",
            "timestamp",
            "article_id",
        ],
    )

    user_features = (
        history
        .groupby("user_id")
        .agg(
            history_length=(
                "article_id",
                "size",
            ),
            unique_articles=(
                "article_id",
                "nunique",
            ),
            last_history_time=(
                "timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    output = (
        FEATURE_MIND
        / "users.parquet"
    )

    user_features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Users with history: "
        f"{len(user_features):,}"
    )

    print(
        f"       Saved: {output}"
    )

    return user_features


# ============================================================
# CREATE ARTICLE STATISTICS
# ============================================================

def build_article_stats():

    print(
        "\n[4/4] Building article statistics "
        "from TRAIN only..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "article_id",
            "clicked",
        ],
    )

    impressions = (
        train
        .groupby("article_id")
        .size()
        .rename("impression_count")
    )

    clicks = (
        train
        .groupby("article_id")["clicked"]
        .sum()
        .rename("click_count")
    )

    stats = pd.concat(
        [
            impressions,
            clicks,
        ],
        axis=1,
    ).fillna(0)

    stats["impression_count"] = (
        stats["impression_count"]
        .astype("int64")
    )

    stats["click_count"] = (
        stats["click_count"]
        .astype("int64")
    )

    stats["ctr"] = (
        stats["click_count"]
        / stats["impression_count"]
    )

    stats = (
        stats
        .reset_index()
        .rename(
            columns={
                "index": "article_id"
            }
        )
    )

    output = (
        FEATURE_MIND
        / "article_stats.parquet"
    )

    stats.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Articles with training statistics: "
        f"{len(stats):,}"
    )

    print(
        f"       Saved: {output}"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MIND FEATURE STORE")
    print("=" * 70)

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build separate article stores
    # --------------------------------------------------------

    train_articles = build_article_features(
        "train"
    )

    dev_articles = build_article_features(
        "dev"
    )

    # --------------------------------------------------------
    # User features
    # --------------------------------------------------------

    users = build_user_features()

    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    article_stats = build_article_stats()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE STORE COMPLETE")
    print("=" * 70)

    print(
        f"\nTrain articles: {len(train_articles):,}"
    )

    print(
        f"Dev articles:   {len(dev_articles):,}"
    )

    print(
        f"Users:          {len(users):,}"
    )

    print(
        f"Train statistics: "
        f"{len(article_stats):,}"
    )


if __name__ == "__main__":
    main()