from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    ROOT
    / "data"
    / "features"
    / "mind"
)


def test_article_features_exist():

    path = FEATURE_DIR / "articles.parquet"

    assert path.exists()

    articles = pd.read_parquet(path)

    required = {
        "article_id",
        "title",
        "abstract",
        "text",
        "category",
        "subcategory",
    }

    assert required.issubset(
        articles.columns
    )

    assert (
        articles["article_id"]
        .notna()
        .all()
    )

    assert (
        articles["article_id"]
        .duplicated()
        .sum()
        == 0
    )


def test_user_features_exist():

    path = FEATURE_DIR / "users.parquet"

    assert path.exists()

    users = pd.read_parquet(path)

    required = {
        "user_id",
        "history_length",
        "unique_articles",
        "last_history_time",
    }

    assert required.issubset(
        users.columns
    )

    assert (
        users["user_id"]
        .duplicated()
        .sum()
        == 0
    )

    assert (
        users["history_length"] > 0
    ).all()


def test_article_statistics_exist():

    path = (
        FEATURE_DIR
        / "article_stats.parquet"
    )

    assert path.exists()

    stats = pd.read_parquet(path)

    required = {
        "article_id",
        "impression_count",
        "click_count",
        "ctr",
    }

    assert required.issubset(
        stats.columns
    )

    assert (
        stats["impression_count"] > 0
    ).all()

    assert (
        stats["click_count"] >= 0
    ).all()

    assert (
        (stats["ctr"] >= 0)
        &
        (stats["ctr"] <= 1)
    ).all()


def test_article_statistics_use_valid_clicks():

    stats = pd.read_parquet(
        FEATURE_DIR
        / "article_stats.parquet"
    )

    assert (
        stats["click_count"]
        <= stats["impression_count"]
    ).all()