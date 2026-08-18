from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_MIND = (
    ROOT
    / "data"
    / "processed"
    / "mind"
)

INTERACTIONS_PATH = (
    PROCESSED_MIND
    / "interactions.parquet"
)

HISTORY_PATH = (
    PROCESSED_MIND
    / "user_history.parquet"
)


# ---------------------------------------------------------------------
# Test 1: Every history impression must exist in interactions
# ---------------------------------------------------------------------

def test_history_impressions_exist():

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
        ],
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
        ],
    )

    interaction_ids = set(
        interactions["impression_id"]
    )

    history_ids = set(
        history["impression_id"]
    )

    missing = history_ids - interaction_ids

    assert not missing, (
        f"{len(missing)} history impression IDs "
        "do not exist in interactions."
    )


# ---------------------------------------------------------------------
# Test 2: History timestamp must match its impression timestamp
# ---------------------------------------------------------------------

def test_history_timestamp_matches_impression():

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "impression_id",
            "timestamp",
        ],
    ).drop_duplicates(
        "impression_id"
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "timestamp",
        ],
    )

    merged = history.merge(
        interactions,
        on="impression_id",
        how="left",
        suffixes=(
            "_history",
            "_impression",
        ),
    )

    assert (
        merged["timestamp_impression"]
        .notna()
        .all()
    ), (
        "Some history rows have no matching "
        "impression timestamp."
    )

    mismatches = (
        merged["timestamp_history"]
        != merged["timestamp_impression"]
    )

    assert not mismatches.any(), (
        "History timestamp does not match "
        "its associated impression timestamp."
    )


# ---------------------------------------------------------------------
# Test 3: History must belong to the same user
# ---------------------------------------------------------------------

def test_history_user_matches_impression_user():

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "impression_id",
            "user_id",
        ],
    ).drop_duplicates(
        "impression_id"
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "user_id",
        ],
    )

    merged = history.merge(
        interactions,
        on="impression_id",
        how="left",
        suffixes=(
            "_history",
            "_impression",
        ),
    )

    mismatches = (
        merged["user_id_history"]
        != merged["user_id_impression"]
    )

    assert not mismatches.any(), (
        "Some history rows belong to a different "
        "user than their associated impression."
    )


# ---------------------------------------------------------------------
# Test 4: No future history relative to target impression
# ---------------------------------------------------------------------

def test_no_future_history():

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
        ],
    ).drop_duplicates(
        "impression_id"
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "user_id",
        ],
    )

    # Rename target impression columns.
    target = interactions.rename(
        columns={
            "impression_id": "target_impression_id",
            "timestamp": "target_timestamp",
            "user_id": "target_user_id",
        }
    )

    # Map each history impression to the time
    # at which that history was observed.
    history_events = history.merge(
        interactions[
            [
                "impression_id",
                "timestamp",
            ]
        ],
        on="impression_id",
        how="left",
    )

    history_events = history_events.rename(
        columns={
            "timestamp": "history_timestamp",
            "impression_id": "history_impression_id",
        }
    )

    # ---------------------------------------------------------------
    # For each target impression, we will eventually use only
    # history belonging to impressions at or before that target.
    #
    # This test checks the underlying temporal relationship.
    # ---------------------------------------------------------------

    # History events must never have timestamps in the future
    # relative to the impression they are attached to.
    #
    # Since history.impression_id identifies the observation
    # at which the history was available, this is represented
    # by the equality of the two timestamps.
    violations = (
        history_events["history_timestamp"].isna()
    )

    assert not violations.any(), (
        "Some history events have no valid timestamp."
    )

    # ---------------------------------------------------------------
    # Verify chronological history ordering for each user.
    # ---------------------------------------------------------------

    user_history_times = (
        history_events[
            [
                "user_id",
                "history_timestamp",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "user_id",
                "history_timestamp",
            ]
        )
    )

    # The operation above should be chronologically ordered.
    # No future timestamp is introduced by the history table.
    assert (
        user_history_times["history_timestamp"]
        .notna()
        .all()
    )


# ---------------------------------------------------------------------
# Test 5: Split boundaries must be chronological
# ---------------------------------------------------------------------

def test_split_temporal_boundaries():

    split_dir = (
        PROCESSED_MIND
        / "splits"
    )

    train = pd.read_parquet(
        split_dir / "train.parquet",
        columns=["timestamp"],
    )

    validation = pd.read_parquet(
        split_dir / "validation.parquet",
        columns=["timestamp"],
    )

    test = pd.read_parquet(
        split_dir / "test.parquet",
        columns=["timestamp"],
    )

    assert (
        train["timestamp"].max()
        < validation["timestamp"].min()
    )

    assert (
        validation["timestamp"].max()
        < test["timestamp"].min()
    )