from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.ranking.a2_din_data import build_article_id_mapping


RETRIEVAL_FEATURE_COLUMNS = [
    "bm25_score",
    "bm25_rank",
    "semantic_score",
    "semantic_rank",
    "rrf_score",
]


class EBNeRDPhase5Dataset(Dataset):
    """
    PyTorch Dataset for the Phase 5.2 neural re-ranker.

    Each candidate example contains:

        history_article_ids
        candidate_article_id
        retrieval_features
        clicked
        user_id

    The history is constructed strictly point-in-time:

        history.timestamp < impression_time

    The most recent `max_history_length` valid events are retained.

    Retrieval features are taken directly from the already-built
    Phase 5.1 feature table.
    """

    REQUIRED_CANDIDATE_COLUMNS = {
        "impression_id",
        "user_id",
        "article_id",
        "impression_time",
        "clicked",
        *RETRIEVAL_FEATURE_COLUMNS,
    }

    REQUIRED_HISTORY_COLUMNS = {
        "user_id",
        "article_id",
        "timestamp",
    }

    def __init__(
        self,
        candidates: pd.DataFrame,
        history: pd.DataFrame,
        article_id_to_index: Dict[int, int],
        max_history_length: int = 20,
    ) -> None:

        if max_history_length <= 0:
            raise ValueError(
                "max_history_length must be positive."
            )

        missing_candidates = (
            self.REQUIRED_CANDIDATE_COLUMNS
            - set(candidates.columns)
        )

        if missing_candidates:
            raise ValueError(
                "Candidate table is missing required columns: "
                f"{sorted(missing_candidates)}"
            )

        missing_history = (
            self.REQUIRED_HISTORY_COLUMNS
            - set(history.columns)
        )

        if missing_history:
            raise ValueError(
                "History table is missing required columns: "
                f"{sorted(missing_history)}"
            )

        if not article_id_to_index:
            raise ValueError(
                "article_id_to_index cannot be empty."
            )

        self.max_history_length = max_history_length
        self.article_id_to_index = article_id_to_index

        # --------------------------------------------------------------
        # Candidate table
        # --------------------------------------------------------------

        self.candidates = candidates[
            [
                "impression_id",
                "user_id",
                "article_id",
                "impression_time",
                "clicked",
                *RETRIEVAL_FEATURE_COLUMNS,
            ]
        ].reset_index(drop=True)

        self.user_ids = self.candidates[
            "user_id"
        ].to_numpy(
            dtype=np.int64,
            copy=True,
        )

        self.candidate_article_ids = self.candidates[
            "article_id"
        ].to_numpy(
            dtype=np.int64,
            copy=True,
        )

        self.clicked = self.candidates[
            "clicked"
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        # Convert impression times to nanoseconds.
        impression_times = pd.to_datetime(
            self.candidates["impression_time"],
            errors="raise",
        )

        self.impression_times = np.asarray(
            [
                pd.Timestamp(timestamp).value
                for timestamp in impression_times
            ],
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # Retrieval features
        # --------------------------------------------------------------

        self.retrieval_features = (
            self.candidates[
                RETRIEVAL_FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        if not np.isfinite(
            self.retrieval_features
        ).all():
            raise ValueError(
                "Retrieval features contain NaN or infinite values."
            )

        # --------------------------------------------------------------
        # Candidate article-ID mapping
        #
        # Index 0 is reserved for padding/unknown articles,
        # exactly as in the existing DIN data pipeline.
        # --------------------------------------------------------------

        self.candidate_article_indices = np.asarray(
            [
                article_id_to_index.get(
                    int(article_id),
                    0,
                )
                for article_id in self.candidate_article_ids
            ],
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # Temporal history lookup
        # --------------------------------------------------------------

        history = history.copy()

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="raise",
        )

        history = history.sort_values(
            [
                "user_id",
                "timestamp",
                "article_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        self.history_by_user: Dict[
            int,
            List[Tuple[int, int]],
        ] = {}

        for row in history[
            [
                "user_id",
                "article_id",
                "timestamp",
            ]
        ].itertuples(index=False):

            user_id = int(row.user_id)
            article_id = int(row.article_id)

            timestamp_ns = int(
                pd.Timestamp(row.timestamp).value
            )

            article_index = article_id_to_index.get(
                article_id,
                0,
            )

            self.history_by_user.setdefault(
                user_id,
                [],
            ).append(
                (
                    timestamp_ns,
                    article_index,
                )
            )

        # Pre-compute timestamp lists so that temporal lookup does not
        # repeatedly construct them inside __getitem__.
        self.history_timestamps_by_user = {
            user_id: [
                timestamp
                for timestamp, _ in events
            ]
            for user_id, events
            in self.history_by_user.items()
        }

    def __len__(self) -> int:
        return len(self.candidates)

    def _get_history(
        self,
        user_id: int,
        impression_time_ns: int,
    ) -> List[int]:
        """
        Return the most recent temporally valid history.

        Only events satisfying:

            timestamp < impression_time

        are allowed.
        """

        events = self.history_by_user.get(
            user_id,
            [],
        )

        if not events:
            return []

        timestamps = self.history_timestamps_by_user[
            user_id
        ]

        # First event whose timestamp >= impression time.
        # Everything before this position is strictly earlier.
        cutoff = bisect_left(
            timestamps,
            impression_time_ns,
        )

        if cutoff == 0:
            return []

        valid_events = events[:cutoff]

        recent_events = valid_events[
            -self.max_history_length:
        ]

        return [
            article_index
            for _, article_index
            in recent_events
        ]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        user_id = int(
            self.user_ids[index]
        )

        impression_time_ns = int(
            self.impression_times[index]
        )

        history_indices = self._get_history(
            user_id=user_id,
            impression_time_ns=impression_time_ns,
        )

        # --------------------------------------------------------------
        # Right-align history with zero padding.
        #
        # Example with max_history_length=5:
        #
        # history = [12, 17, 23]
        #
        # becomes:
        #
        # [0, 0, 12, 17, 23]
        # --------------------------------------------------------------

        padded_history = np.zeros(
            self.max_history_length,
            dtype=np.int64,
        )

        if history_indices:
            history_array = np.asarray(
                history_indices,
                dtype=np.int64,
            )

            padded_history[
                -len(history_array):
            ] = history_array

        return {
            "history_article_ids": torch.tensor(
                padded_history,
                dtype=torch.long,
            ),
            "candidate_article_id": torch.tensor(
                self.candidate_article_indices[index],
                dtype=torch.long,
            ),
            "retrieval_features": torch.tensor(
                self.retrieval_features[index],
                dtype=torch.float32,
            ),
            "clicked": torch.tensor(
                self.clicked[index],
                dtype=torch.float32,
            ),
            "user_id": torch.tensor(
                user_id,
                dtype=torch.long,
            ),
        }


def load_phase5_dataset(
    project_root: Path,
    split: str = "train",
    max_history_length: int = 20,
) -> tuple[
    EBNeRDPhase5Dataset,
    Dict[int, int],
]:
    """
    Load the Phase 5.2 Dataset.

    Phase 5.1 provides:

        train_phase5_reranker.parquet

    The corresponding user-history file is loaded from:

        {split}_user_history.parquet
    """

    project_root = Path(project_root)

    feature_root = (
        project_root
        / "data"
        / "features"
        / "ebnerd"
        / "small"
    )

    processed_root = (
    project_root
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)

    articles_path = (
        feature_root
        / "articles.parquet"
    )

    if split == "train":
        candidates_path = (
            feature_root
            / "train_phase5_reranker.parquet"
        )
    elif split == "validation":
        candidates_path = (
            feature_root
            / "validation_phase5_reranker.parquet"
        )
    else:
        raise ValueError(
            "Unsupported split: "
            f"{split!r}. Expected 'train' or 'validation'."
        )

    history_path = (
        processed_root
        / f"{split}_user_history.parquet"
    )

    for path in [
        articles_path,
        candidates_path,
        history_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Phase 5.2 file not found: {path}"
            )

    articles = pd.read_parquet(
        articles_path
    )

    candidates = pd.read_parquet(
        candidates_path
    )

    history = pd.read_parquet(
        history_path
    )

    article_id_to_index = (
        build_article_id_mapping(
            articles
        )
    )

    dataset = EBNeRDPhase5Dataset(
        candidates=candidates,
        history=history,
        article_id_to_index=article_id_to_index,
        max_history_length=max_history_length,
    )

    return dataset, article_id_to_index