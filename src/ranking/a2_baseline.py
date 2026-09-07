from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLUMNS: tuple[str, ...] = (
    "category_preference",
    "subcategory_preference",
    "category_recency",
    "subcategory_recency",
    "smoothed_ctr",
    "interest_score",
    "recency_score",
    "behavioral_score",
)

ID_COLUMNS: tuple[str, ...] = (
    "impression_id",
    "user_id",
    "timestamp",
    "article_id",
)

TARGET_COLUMN = "clicked"
SCORE_COLUMN = "baseline_predicted_score"


class A2BaselineRanker:
    """Phase 2 pointwise learned scorer used for impression-level ranking."""

    def __init__(
        self,
        *,
        max_iter: int = 200,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self.feature_columns = FEATURE_COLUMNS

    @staticmethod
    def validate_training_frame(df: pd.DataFrame) -> None:
        required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("Training frame is empty.")

        labels = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
        invalid = ~labels.isin([0, 1])
        if invalid.any():
            raise ValueError("clicked must contain only binary 0/1 labels.")

        if not np.isfinite(
            df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError("Feature matrix contains non-finite values.")

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[list(self.feature_columns)].to_numpy(
            dtype=np.float32,
            copy=False,
        )

    def fit(self, df: pd.DataFrame) -> "A2BaselineRanker":
        self.validate_training_frame(df)

        X = self._matrix(df)
        y = df[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)

        self.model.fit(X, y)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Ranker must be fitted before prediction.")

        missing = sorted(set(FEATURE_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        scores = self.model.predict_proba(self._matrix(df))[:, 1]
        return np.asarray(scores, dtype=np.float32)

    def add_scores(
        self,
        df: pd.DataFrame,
        *,
        score_column: str = SCORE_COLUMN,
    ) -> pd.DataFrame:
        if "impression_id" not in df.columns:
            raise ValueError("impression_id is required for ranking output.")

        result = df.copy()
        result[score_column] = self.predict_scores(result)
        return result

    @staticmethod
    def rank_predictions(
        df: pd.DataFrame,
        *,
        score_column: str = SCORE_COLUMN,
    ) -> pd.DataFrame:
        required = {"impression_id", "article_id", score_column}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Stable sorting makes ties deterministic without changing candidate groups.
        return df.sort_values(
            ["impression_id", score_column],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "model_name": self.__class__.__name__,
                "version": 1,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "A2BaselineRanker":
        payload = joblib.load(path)
        ranker = cls()
        ranker.model = payload["model"]
        ranker.feature_columns = tuple(payload["feature_columns"])
        return ranker


def build_training_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact X/y arrays for the Phase 2 baseline."""
    A2BaselineRanker.validate_training_frame(df)
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32, copy=False)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
    return X, y
