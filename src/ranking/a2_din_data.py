"""
Phase 3.4 — EB-NeRD Small data adapter for the local DIN baseline.

Converts chronological EB-NeRD user histories and candidate impressions
into tensors suitable for DINModel.

Important:
- Histories are strictly chronological.
- Only history available BEFORE the impression is used.
- The current candidate is never inserted into its own history.
- History is truncated to the most recent `max_history_length` events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DINExample:
    """One candidate-level training/evaluation example."""

    history_article_ids: list[int]
    candidate_article_id: int
    clicked: int


class EBNeRDDINDataset(Dataset):
    """
    Dataset connecting EB-NeRD candidate impressions with user histories.

    Parameters
    ----------
    candidates:
        Candidate-level EB-NeRD dataframe.

    user_history:
        Chronological user-history dataframe.

    article_id_to_index:
        Mapping from raw article IDs to embedding-table indices.

    max_history_length:
        Maximum number of historical events retained per example.

    """

    REQUIRED_CANDIDATE_COLUMNS = {
        "impression_id",
        "user_id",
        "article_id",
        "impression_time",
        "clicked",
    }

    REQUIRED_HISTORY_COLUMNS = {
        "user_id",
        "article_id",
        "timestamp",
    }

    def __init__(
        self,
        candidates: pd.DataFrame,
        user_history: pd.DataFrame,
        article_id_to_index: dict[int, int],
        max_history_length: int = 20,
    ) -> None:

        if max_history_length <= 0:
            raise ValueError("max_history_length must be positive.")

        missing_candidates = (
            self.REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)
        )

        if missing_candidates:
            raise ValueError(
                "Candidates dataframe is missing columns: "
                f"{sorted(missing_candidates)}"
            )

        missing_history = (
            self.REQUIRED_HISTORY_COLUMNS - set(user_history.columns)
        )

        if missing_history:
            raise ValueError(
                "User-history dataframe is missing columns: "
                f"{sorted(missing_history)}"
            )

        if not article_id_to_index:
            raise ValueError("article_id_to_index cannot be empty.")

        self.max_history_length = max_history_length
        self.article_id_to_index = article_id_to_index

        # --------------------------------------------------------------
        # Prepare candidate data
        # --------------------------------------------------------------

        self.candidates = candidates[
            [
                "impression_id",
                "user_id",
                "article_id",
                "impression_time",
                "clicked",
            ]
        ].copy()

        self.candidates["impression_time"] = pd.to_datetime(
            self.candidates["impression_time"]
        )

        self.candidates["article_id"] = self.candidates[
            "article_id"
        ].astype("int64")

        self.candidates["user_id"] = self.candidates[
            "user_id"
        ].astype("int64")

        self.candidates["clicked"] = self.candidates[
            "clicked"
        ].astype("int64")

        # --------------------------------------------------------------
        # Prepare chronological history
        # --------------------------------------------------------------

        history = user_history[
            [
                "user_id",
                "article_id",
                "timestamp",
            ]
        ].copy()

        history["timestamp"] = pd.to_datetime(
            history["timestamp"]
        )

        history["article_id"] = history[
            "article_id"
        ].astype("int64")

        history["user_id"] = history[
            "user_id"
        ].astype("int64")

        history = history.sort_values(
            ["user_id", "timestamp"]
        )

        # Store chronological events per user.
        #
        # We intentionally retain repeated article IDs because repeated
        # interactions are legitimate sequential behaviour.
        self.history_by_user: dict[int, list[tuple[pd.Timestamp, int]]] = {}

        for user_id, group in history.groupby("user_id", sort=False):
            self.history_by_user[int(user_id)] = [
                (timestamp, int(article_id))
                for timestamp, article_id in zip(
                    group["timestamp"],
                    group["article_id"],
                )
            ]

    def __len__(self) -> int:
        """Return number of candidate rows."""
        return len(self.candidates)

    def _get_history(
        self,
        user_id: int,
        impression_time: pd.Timestamp,
    ) -> list[int]:
        """
        Return the user's historical article IDs strictly before
        the current impression.
        """

        events = self.history_by_user.get(user_id, [])

        history = [
            article_id
            for timestamp, article_id in events
            if timestamp < impression_time
        ]

        # Keep only the most recent events.
        history = history[-self.max_history_length :]

        return history

    def _article_to_index(self, article_id: int) -> int:
        """
        Convert raw article ID to embedding-table index.

        Unknown articles are mapped to 0, which is the reserved
        padding/unknown index.
        """

        return self.article_id_to_index.get(article_id, 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one model-ready example."""

        row = self.candidates.iloc[index]

        user_id = int(row["user_id"])
        article_id = int(row["article_id"])
        impression_time = row["impression_time"]
        clicked = int(row["clicked"])

        history = self._get_history(
            user_id=user_id,
            impression_time=impression_time,
        )

        history_indices = [
            self._article_to_index(article_id)
            for article_id in history
        ]

        # Right-align the most recent history.
        #
        # Example max_history_length = 5:
        #
        # history = [10, 20, 30]
        #
        # becomes:
        #
        # [0, 0, 10, 20, 30]
        #
        # This makes index positions consistent across examples.
        padding_length = (
            self.max_history_length - len(history_indices)
        )

        padded_history = (
            [0] * padding_length
            + history_indices
        )

        history_mask = [
            False
        ] * padding_length + [
            True
        ] * len(history_indices)

        candidate_index = self._article_to_index(
            article_id
        )

        return {
            "history_article_ids": torch.tensor(
                padded_history,
                dtype=torch.long,
            ),
            "candidate_article_id": torch.tensor(
                candidate_index,
                dtype=torch.long,
            ),
            "clicked": torch.tensor(
                clicked,
                dtype=torch.float32,
            ),
            "user_id": torch.tensor(
                user_id,
                dtype=torch.long,
            ),
        }


def build_article_id_mapping(
    articles: pd.DataFrame,
) -> dict[int, int]:
    """
    Build a raw article-ID → embedding index mapping.

    Index 0 is reserved for padding/unknown.

    Therefore:
        raw article IDs → 1 ... N
        0              → padding/unknown
    """

    if "article_id" not in articles.columns:
        raise ValueError(
            "Articles dataframe must contain 'article_id'."
        )

    article_ids = (
        articles["article_id"]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    return {
        int(article_id): index
        for index, article_id in enumerate(
            article_ids,
            start=1,
        )
    }


def load_ebnerd_din_dataset(
    project_root: str | Path,
    split: str,
    max_history_length: int = 20,
) -> tuple[EBNeRDDINDataset, dict[int, int]]:
    """
    Load an EB-NeRD Small split and construct the DIN dataset.

    Parameters
    ----------
    project_root:
        News-Recommender project root.

    split:
        Either "train" or "validation".

    max_history_length:
        Maximum chronological history length.

    Returns
    -------
    dataset, article_id_to_index
    """

    project_root = Path(project_root)

    if split not in {"train", "validation"}:
        raise ValueError(
            "split must be 'train' or 'validation'."
        )

    feature_dir = (
        project_root
        / "data"
        / "features"
        / "ebnerd"
        / "small"
    )

    processed_dir = (
        project_root
        / "data"
        / "processed"
        / "ebnerd"
        / "small"
    )

    articles_path = feature_dir / "articles.parquet"

    candidates_path = (
        feature_dir
        / f"{split}_candidates.parquet"
    )

    history_path = (
        processed_dir
        / f"{split}_user_history.parquet"
    )

    for path in [
        articles_path,
        candidates_path,
        history_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required EB-NeRD file not found: {path}"
            )

    articles = pd.read_parquet(articles_path)
    candidates = pd.read_parquet(candidates_path)
    user_history = pd.read_parquet(history_path)

    article_id_to_index = build_article_id_mapping(
        articles
    )

    dataset = EBNeRDDINDataset(
        candidates=candidates,
        user_history=user_history,
        article_id_to_index=article_id_to_index,
        max_history_length=max_history_length,
    )

    return dataset, article_id_to_index