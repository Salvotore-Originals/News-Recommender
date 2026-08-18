from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the root directory of the News-Recommender project.

    This file is located at:
        project_root/src/data/split.py
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------

def split_by_date(
    interactions: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> dict[str, pd.DataFrame]:
    """
    Split interactions chronologically.

    Parameters
    ----------
    interactions:
        Candidate-level interaction DataFrame.

    train_end:
        Exclusive upper boundary for training data.

    validation_end:
        Exclusive upper boundary for validation data.

    Returns
    -------
    dict
        train, validation, and test DataFrames.

    Important
    ---------
    The split is performed using impression timestamps.

    Since all candidate rows belonging to an impression share
    the same timestamp, an entire impression remains in one split.
    """

    if interactions.empty:
        raise ValueError("Interactions DataFrame is empty.")

    required_columns = {
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "clicked",
    }

    missing_columns = (
        required_columns - set(interactions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Ensure timestamp is datetime.
    interactions = interactions.copy()

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        errors="raise",
    )

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    if train_end >= validation_end:
        raise ValueError(
            "train_end must occur before validation_end."
        )

    min_time = interactions["timestamp"].min()
    max_time = interactions["timestamp"].max()

    if train_end <= min_time:
        raise ValueError(
            "train_end occurs before or at the beginning of the dataset."
        )

    if validation_end <= train_end:
        raise ValueError(
            "validation_end must occur after train_end."
        )

    if validation_end > max_time:
        raise ValueError(
            "validation_end occurs after the dataset ends."
        )

    # ---------------------------------------------------------------
    # Chronological masks
    # ---------------------------------------------------------------

    train_mask = (
        interactions["timestamp"] < train_end
    )

    validation_mask = (
        (interactions["timestamp"] >= train_end)
        & (interactions["timestamp"] < validation_end)
    )

    test_mask = (
        interactions["timestamp"] >= validation_end
    )

    train = interactions.loc[
        train_mask
    ].copy()

    validation = interactions.loc[
        validation_mask
    ].copy()

    test = interactions.loc[
        test_mask
    ].copy()

    # ---------------------------------------------------------------
    # Make sure nothing was lost.
    # ---------------------------------------------------------------

    total_rows = len(interactions)

    split_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    if total_rows != split_rows:
        raise AssertionError(
            "Temporal split lost or duplicated interaction rows."
        )

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


# ---------------------------------------------------------------------
# Impression-level integrity
# ---------------------------------------------------------------------

def validate_impression_integrity(
    splits: dict[str, pd.DataFrame],
) -> None:
    """
    Ensure that an impression never appears in more than one split.
    """

    train_ids = set(
        splits["train"]["impression_id"]
    )

    validation_ids = set(
        splits["validation"]["impression_id"]
    )

    test_ids = set(
        splits["test"]["impression_id"]
    )

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    if train_validation_overlap:
        raise AssertionError(
            "Some impressions occur in both train and validation."
        )

    if train_test_overlap:
        raise AssertionError(
            "Some impressions occur in both train and test."
        )

    if validation_test_overlap:
        raise AssertionError(
            "Some impressions occur in both validation and test."
        )


# ---------------------------------------------------------------------
# Temporal ordering validation
# ---------------------------------------------------------------------

def validate_temporal_order(
    splits: dict[str, pd.DataFrame],
) -> None:
    """
    Verify chronological ordering between train, validation,
    and test.
    """

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    if not train.empty and not validation.empty:

        train_max = train["timestamp"].max()
        validation_min = validation["timestamp"].min()

        if train_max >= validation_min:
            raise AssertionError(
                "Training data overlaps or extends into validation."
            )

    if not validation.empty and not test.empty:

        validation_max = validation["timestamp"].max()
        test_min = test["timestamp"].min()

        if validation_max >= test_min:
            raise AssertionError(
                "Validation data overlaps or extends into test."
            )


# ---------------------------------------------------------------------
# Split statistics
# ---------------------------------------------------------------------

def print_split_statistics(
    splits: dict[str, pd.DataFrame],
) -> None:
    """
    Print useful statistics for each split.
    """

    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT STATISTICS")
    print("=" * 70)

    for name in ["train", "validation", "test"]:

        df = splits[name]

        impressions = (
            df["impression_id"].nunique()
        )

        users = (
            df["user_id"].nunique()
        )

        candidates = len(df)

        clicks = int(
            df["clicked"].sum()
        )

        print(f"\n{name.upper()}")

        print(
            f"  Rows:         {candidates:,}"
        )

        print(
            f"  Impressions:  {impressions:,}"
        )

        print(
            f"  Users:        {users:,}"
        )

        print(
            f"  Clicks:       {clicks:,}"
        )

        print(
            f"  Start:        {df['timestamp'].min()}"
        )

        print(
            f"  End:          {df['timestamp'].max()}"
        )


# ---------------------------------------------------------------------
# Main MIND splitting pipeline
# ---------------------------------------------------------------------

def split_mind(
    project_root: Path | None = None,
) -> dict[str, Path]:
    """
    Create chronological train/validation/test splits for MIND.

    MIND-small training data covers:

        Nov 9 2019
        Nov 10 2019
        Nov 11 2019
        Nov 12 2019
        Nov 13 2019
        Nov 14 2019

    We use:

        Train      = Nov 9 - Nov 11
        Validation = Nov 12
        Test       = Nov 13 - Nov 14
    """

    if project_root is None:
        project_root = get_project_root()

    processed_dir = (
        project_root
        / "data"
        / "processed"
        / "mind"
    )

    interactions_path = (
        processed_dir
        / "interactions.parquet"
    )

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Interactions file not found: "
            f"{interactions_path}"
        )

    print("=" * 70)
    print("MIND TEMPORAL SPLIT")
    print("=" * 70)

    print("\nLoading interactions...")

    interactions = pd.read_parquet(
        interactions_path
    )

    print(
        f"Loaded {len(interactions):,} candidate rows."
    )

    # ---------------------------------------------------------------
    # Split boundaries
    #
    # 2019-11-12 00:00:00 means:
    # train contains everything BEFORE Nov 12.
    #
    # 2019-11-13 00:00:00 means:
    # validation contains Nov 12.
    # ---------------------------------------------------------------

    splits = split_by_date(
        interactions=interactions,
        train_end="2019-11-12 00:00:00",
        validation_end="2019-11-13 00:00:00",
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print("\nValidating impression integrity...")

    validate_impression_integrity(
        splits
    )

    print(
        "  Impression integrity: PASS"
    )

    print("\nValidating temporal ordering...")

    validate_temporal_order(
        splits
    )

    print(
        "  Temporal ordering: PASS"
    )

    # ---------------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------------

    print_split_statistics(
        splits
    )

    # ---------------------------------------------------------------
    # Write files
    # ---------------------------------------------------------------

    split_dir = (
        processed_dir / "splits"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {}

    for name, df in splits.items():

        output_path = (
            split_dir / f"{name}.parquet"
        )

        df.to_parquet(
            output_path,
            index=False,
        )

        output_paths[name] = output_path

        print(
            f"\nSaved {name}:"
            f"\n  {output_path}"
        )

    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT COMPLETE")
    print("=" * 70)

    return output_paths


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    split_mind()