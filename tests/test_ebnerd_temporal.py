from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_small"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)


def test_train_feature_row_count():
    features = pd.read_parquet(
        FEATURE_ROOT / "train_temporal.parquet"
    )

    interactions = pd.read_parquet(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ebnerd"
        / "small"
        / "splits"
        / "train.parquet"
    )

    assert len(features) == len(interactions)


def test_validation_feature_row_count():
    features = pd.read_parquet(
        FEATURE_ROOT / "validation_temporal.parquet"
    )

    interactions = pd.read_parquet(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ebnerd"
        / "small"
        / "splits"
        / "validation.parquet"
    )

    assert len(features) == len(interactions)


def test_train_temporal_features_are_non_negative():
    features = pd.read_parquet(
        FEATURE_ROOT / "train_temporal.parquet"
    )

    assert (
        features["history_count_before"] >= 0
    ).all()

    assert (
        features["category_count_before"] >= 0
    ).all()

    assert (
        features["category_preference"] >= 0
    ).all()

    assert (
        features["category_preference"] <= 1
    ).all()


def test_validation_temporal_features_are_non_negative():
    features = pd.read_parquet(
        FEATURE_ROOT / "validation_temporal.parquet"
    )

    assert (
        features["history_count_before"] >= 0
    ).all()

    assert (
        features["category_count_before"] >= 0
    ).all()

    assert (
        features["category_preference"] >= 0
    ).all()

    assert (
        features["category_preference"] <= 1
    ).all()


def test_category_count_never_exceeds_history_count():
    for split in ["train", "validation"]:
        features = pd.read_parquet(
            FEATURE_ROOT
            / f"{split}_temporal.parquet"
        )

        assert (
            features["category_count_before"]
            <= features["history_count_before"]
        ).all()


def test_first_history_impressions_have_zero_history():
    """
    Verify that users with no history before an impression
    receive zero historical counts.
    """

    for split in ["train", "validation"]:
        features = pd.read_parquet(
            FEATURE_ROOT
            / f"{split}_temporal.parquet"
        )

        zero_history = features[
            features["history_count_before"] == 0
        ]

        if len(zero_history) == 0:
            continue

        assert (
            zero_history["category_count_before"] == 0
        ).all()

        assert (
            zero_history["category_preference"] == 0
        ).all()

        assert (
            zero_history["candidate_seen_before"] == 0
        ).all()