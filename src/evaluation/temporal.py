from __future__ import annotations

import pandas as pd


# ============================================================
# TEMPORAL SPLITTING
# ============================================================

TEMPORAL_PERIODS = (
    "early",
    "middle",
    "late",
)


def assign_temporal_periods(
    df: pd.DataFrame,
    n_periods: int = 3,
) -> pd.DataFrame:
    """
    Assign every MIND impression to a chronological period.

    The split is performed at the impression level, ensuring
    that all candidates belonging to one impression remain in
    the same temporal period.

    Parameters
    ----------
    df:
        Candidate-level ranking data. Must contain:

            impression_id
            timestamp

    n_periods:
        Number of chronological periods.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an additional
        ``temporal_period`` column.
    """

    required = {
        "impression_id",
        "timestamp",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if n_periods < 2:
        raise ValueError(
            "n_periods must be at least 2."
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="raise",
    )

    # --------------------------------------------------------
    # Build one row per impression.
    # --------------------------------------------------------

    impressions = (
        result[
            [
                "impression_id",
                "timestamp",
            ]
        ]
        .drop_duplicates(
            subset=["impression_id"]
        )
        .sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if impressions.empty:
        raise ValueError(
            "No impressions found."
        )

    if len(impressions) < n_periods:
        raise ValueError(
            "Number of impressions must be at least "
            "the number of temporal periods."
        )

    # --------------------------------------------------------
    # Assign chronological period by impression order.
    # --------------------------------------------------------

    period_labels = pd.qcut(
        impressions.index,
        q=n_periods,
        labels=TEMPORAL_PERIODS
        if n_periods == 3
        else False,
    )

    if n_periods == 3:
        impressions["temporal_period"] = (
            period_labels.astype(str)
        )
    else:
        impressions["temporal_period"] = (
            period_labels.astype(str)
        )

    # --------------------------------------------------------
    # Join period back to every candidate.
    # --------------------------------------------------------

    result = result.merge(
        impressions[
            [
                "impression_id",
                "temporal_period",
            ]
        ],
        on="impression_id",
        how="left",
        validate="many_to_one",
    )

    if result["temporal_period"].isna().any():
        raise ValueError(
            "Some candidates were not assigned a "
            "temporal period."
        )

    # --------------------------------------------------------
    # Verify candidate count was preserved.
    # --------------------------------------------------------

    if len(result) != len(df):
        raise ValueError(
            "Temporal splitting changed the number "
            "of candidate rows."
        )

    return result