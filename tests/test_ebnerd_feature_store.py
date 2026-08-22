from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

ARTICLE_COUNT = 20_738


def test_article_feature_store():

    articles = pd.read_parquet(
        FEATURE_ROOT / "articles.parquet"
    )

    assert len(articles) == ARTICLE_COUNT

    assert articles["article_id"].is_unique

    required = {
        "article_id",
        "title",
        "subtitle",
        "body",
        "text",
        "category",
        "category_str",
        "published_time",
    }

    assert required.issubset(
        articles.columns
    )


def test_user_feature_store():

    users = pd.read_parquet(
        FEATURE_ROOT / "users.parquet"
    )

    assert users["user_id"].is_unique

    assert (
        users["history_count"] > 0
    ).all()

    assert (
        users["unique_articles"] > 0
    ).all()

    assert (
        users["unique_categories"] > 0
    ).all()


def test_article_statistics():

    stats = pd.read_parquet(
        FEATURE_ROOT / "article_stats.parquet"
    )

    assert stats["article_id"].is_unique

    assert (
        stats["interaction_count"] > 0
    ).all()

    assert (
        stats["unique_users"] > 0
    ).all()

    assert (
        stats["click_count"] >= 0
    ).all()

    assert (
        stats["click_rate"] >= 0
    ).all()

    assert (
        stats["click_rate"] <= 1
    ).all()


def test_user_category_preferences():

    preferences = pd.read_parquet(
        FEATURE_ROOT
        / "user_category_preferences.parquet"
    )

    assert not preferences.empty

    assert (
        preferences["category_interaction_count"]
        > 0
    ).all()

    assert (
        preferences["preference"] >= 0
    ).all()

    assert (
        preferences["preference"] <= 1
    ).all()


def test_preferences_sum_to_one():

    preferences = pd.read_parquet(
        FEATURE_ROOT
        / "user_category_preferences.parquet"
    )

    totals = (
        preferences
        .groupby("user_id")["preference"]
        .sum()
    )

    assert (
        (totals - 1.0).abs() < 1e-6
    ).all()


def test_temporal_files_still_exist():

    train = pd.read_parquet(
        FEATURE_ROOT
        / "train_temporal.parquet"
    )

    validation = pd.read_parquet(
        FEATURE_ROOT
        / "validation_temporal.parquet"
    )

    assert len(train) == 2_585_747
    assert len(validation) == 2_928_942


def test_temporal_features_not_replaced():

    train = pd.read_parquet(
        FEATURE_ROOT
        / "train_temporal.parquet"
    )

    validation = pd.read_parquet(
        FEATURE_ROOT
        / "validation_temporal.parquet"
    )

    required = {
        "impression_id",
        "user_id",
        "article_id",
        "impression_time",
        "clicked",
        "category",
        "category_str",
        "category_count_before",
        "history_count_before",
        "category_preference",
        "candidate_seen_before",
        "candidate_recency_hours",
    }

    assert required.issubset(
        train.columns
    )

    assert required.issubset(
        validation.columns
    )