from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
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


@pytest.fixture(scope="module")
def raw_articles():
    return pd.read_parquet(
        RAW_ROOT / "articles.parquet"
    )


@pytest.fixture(scope="module")
def processed_articles():
    return pd.read_parquet(
        PROCESSED_ROOT / "articles.parquet"
    )


@pytest.fixture(scope="module")
def raw_train_behaviors():
    return pd.read_parquet(
        RAW_ROOT / "train" / "behaviors.parquet"
    )


@pytest.fixture(scope="module")
def processed_train():
    return pd.read_parquet(
        PROCESSED_ROOT / "splits" / "train.parquet"
    )


@pytest.fixture(scope="module")
def raw_validation_behaviors():
    return pd.read_parquet(
        RAW_ROOT / "validation" / "behaviors.parquet"
    )


@pytest.fixture(scope="module")
def processed_validation():
    return pd.read_parquet(
        PROCESSED_ROOT
        / "splits"
        / "validation.parquet"
    )


@pytest.fixture(scope="module")
def raw_train_history():
    return pd.read_parquet(
        RAW_ROOT / "train" / "history.parquet"
    )


@pytest.fixture(scope="module")
def processed_train_history():
    return pd.read_parquet(
        PROCESSED_ROOT
        / "train_user_history.parquet"
    )


@pytest.fixture(scope="module")
def raw_validation_history():
    return pd.read_parquet(
        RAW_ROOT
        / "validation"
        / "history.parquet"
    )


@pytest.fixture(scope="module")
def processed_validation_history():
    return pd.read_parquet(
        PROCESSED_ROOT
        / "validation_user_history.parquet"
    )


def test_article_count_preserved(
    raw_articles,
    processed_articles,
):
    assert len(raw_articles) == len(processed_articles)


def test_article_ids_preserved(
    raw_articles,
    processed_articles,
):
    assert set(raw_articles["article_id"]) == set(
        processed_articles["article_id"]
    )


def test_train_candidate_count_preserved(
    raw_train_behaviors,
    processed_train,
):
    expected = sum(
        len(x)
        for x in raw_train_behaviors["article_ids_inview"]
    )

    assert len(processed_train) == expected


def test_validation_candidate_count_preserved(
    raw_validation_behaviors,
    processed_validation,
):
    expected = sum(
        len(x)
        for x in raw_validation_behaviors[
            "article_ids_inview"
        ]
    )

    assert len(processed_validation) == expected


def test_train_unique_click_count_preserved(
    raw_train_behaviors,
    processed_train,
):
    expected = sum(
        len(set(x))
        for x in raw_train_behaviors[
            "article_ids_clicked"
        ]
    )

    actual = int(
        processed_train["clicked"].sum()
    )

    assert actual == expected

def test_validation_unique_click_count_preserved(
    raw_validation_behaviors,
    processed_validation,
):
    expected = sum(
        len(set(x))
        for x in raw_validation_behaviors[
            "article_ids_clicked"
        ]
    )

    actual = int(
        processed_validation["clicked"].sum()
    )

    assert actual == expected

def test_train_candidate_articles_exist(
    processed_train,
    processed_articles,
):
    article_ids = set(
        processed_articles["article_id"]
    )

    candidates = set(
        processed_train["article_id"]
    )

    assert candidates.issubset(article_ids)


def test_validation_candidate_articles_exist(
    processed_validation,
    processed_articles,
):
    article_ids = set(
        processed_articles["article_id"]
    )

    candidates = set(
        processed_validation["article_id"]
    )

    assert candidates.issubset(article_ids)


def test_train_click_labels_are_binary(
    processed_train,
):
    assert set(
        processed_train["clicked"].unique()
    ).issubset({0, 1})


def test_validation_click_labels_are_binary(
    processed_validation,
):
    assert set(
        processed_validation["clicked"].unique()
    ).issubset({0, 1})


def test_train_history_event_count_preserved(
    raw_train_history,
    processed_train_history,
):
    expected = sum(
        len(x)
        for x in raw_train_history[
            "article_id_fixed"
        ]
    )

    assert len(processed_train_history) == expected


def test_validation_history_event_count_preserved(
    raw_validation_history,
    processed_validation_history,
):
    expected = sum(
        len(x)
        for x in raw_validation_history[
            "article_id_fixed"
        ]
    )

    assert len(processed_validation_history) == expected


def test_train_history_users_preserved(
    raw_train_history,
    processed_train_history,
):
    expected = set(
        raw_train_history["user_id"]
    )

    actual = set(
        processed_train_history["user_id"]
    )

    assert actual == expected


def test_validation_history_users_preserved(
    raw_validation_history,
    processed_validation_history,
):
    expected = set(
        raw_validation_history["user_id"]
    )

    actual = set(
        processed_validation_history["user_id"]
    )

    assert actual == expected