from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

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

ARTICLES_PATH = (
    PROCESSED_MIND
    / "articles.parquet"
)

INTERACTIONS_PATH = (
    PROCESSED_MIND
    / "interactions.parquet"
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
# CREATE ARTICLE FEATURES
# ============================================================

def build_article_features():
    print("\n[1/3] Building article features...")

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    # Replace missing text fields.
    articles["abstract"] = (
        articles["abstract"]
        .fillna("")
        .astype(str)
    )

    articles["title"] = (
        articles["title"]
        .fillna("")
        .astype(str)
    )

    # Combined text for lexical retrieval.
    articles["text"] = (
        articles["title"]
        + " "
        + articles["abstract"]
    ).str.strip()

    # Keep only reusable modelling fields.
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

    articles = articles[columns]

    output = FEATURE_MIND / "articles.parquet"

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
    print("\n[2/3] Building user features...")

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

    output = FEATURE_MIND / "users.parquet"

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
    print("\n[3/3] Building article statistics from TRAIN only...")

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "article_id",
            "clicked",
        ],
    )

    # Number of times each article was shown.
    impressions = (
        train
        .groupby("article_id")
        .size()
        .rename("impression_count")
    )

    # Number of clicks for each article.
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

    # CTR calculated only from training data.
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

    articles = build_article_features()

    users = build_user_features()

    article_stats = build_article_stats()

    print("\n" + "=" * 70)
    print("FEATURE STORE COMPLETE")
    print("=" * 70)

    print(
        f"\nArticles: {len(articles):,}"
    )

    print(
        f"Users with history: {len(users):,}"
    )

    print(
        f"Articles with train statistics: "
        f"{len(article_stats):,}"
    )


if __name__ == "__main__":
    main()