from __future__ import annotations

import pandas as pd


MIND_BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


def load_mind_behaviors(
    path: str,
) -> pd.DataFrame:
    """
    Load a MIND behaviors.tsv file.

    Expected MIND columns:
        1. Impression ID
        2. User ID
        3. Time
        4. History
        5. Impressions
    """

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=MIND_BEHAVIOR_COLUMNS,
    )

    df["impression_id"] = df["impression_id"].astype(str)

    df["user_id"] = df["user_id"].astype(str)

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df["history"] = (
        df["history"]
        .fillna("")
        .astype(str)
    )

    df["impressions"] = (
        df["impressions"]
        .fillna("")
        .astype(str)
    )

    return df


def parse_history(
    history: str,
) -> list[str]:
    """
    Convert a MIND History field into an ordered
    list of clicked article IDs.

    Example:
        'N123 N456 N789'
        ->
        ['N123', 'N456', 'N789']
    """

    if not isinstance(history, str):
        return []

    history = history.strip()

    if not history:
        return []

    return history.split()