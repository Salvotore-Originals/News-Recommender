"""
Phase 4D: Point-in-Time Fresh + Category Candidate Retrieval for EB-NeRD.

This module is intentionally independent of the existing BM25, semantic,
and RRF implementations.

Protocol:
    1. Use only user history events with timestamp < impression_time.
    2. Compute P(category | user, history_before_T).
    3. Consider only articles published at or before impression_time.
    4. Restrict to a configurable freshness window.
    5. Rank recent articles by point-in-time category preference and
       exponential freshness decay.

The module retrieves from the article corpus, not from official impression
candidate lists.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_ARTICLE_COLUMNS = {"article_id", "category_str", "published_time"}
REQUIRED_HISTORY_COLUMNS = {"user_id", "article_id", "timestamp"}

DEFAULT_FRESHNESS_WINDOW_HOURS = 48.0
DEFAULT_DECAY_HOURS = 24.0
DEFAULT_TOP_CATEGORIES = 3
DEFAULT_PER_CATEGORY = 100


def validate_articles(articles: pd.DataFrame) -> None:
    """Validate the minimum article schema required by this retriever."""
    missing = REQUIRED_ARTICLE_COLUMNS - set(articles.columns)
    if missing:
        raise ValueError(
            f"Article corpus missing required columns: {sorted(missing)}"
        )


def validate_history(history: pd.DataFrame) -> None:
    """Validate the minimum history schema required by this retriever."""
    missing = REQUIRED_HISTORY_COLUMNS - set(history.columns)
    if missing:
        raise ValueError(
            f"User history missing required columns: {sorted(missing)}"
        )


def prepare_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize article IDs/categories/timestamps without mutating the input.

    Missing category or publication time rows are excluded because they cannot
    participate safely in category-conditioned, time-aware retrieval.
    """
    validate_articles(articles)

    out = articles[
        ["article_id", "category_str", "published_time"]
    ].copy()

    out["article_id"] = out["article_id"].astype(str)
    out["category_str"] = out["category_str"].astype("string")
    out["published_time"] = pd.to_datetime(
        out["published_time"], errors="coerce"
    )

    out = out.dropna(subset=["category_str", "published_time"])
    out = out.drop_duplicates(subset=["article_id"], keep="first")
    out = out.reset_index(drop=True)

    return out


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize history IDs/timestamps without mutating the input.

    History events are retained even if their article is absent from the
    article corpus; the category lookup later ignores events without a
    matching article category.
    """
    validate_history(history)

    out = history[["user_id", "article_id", "timestamp"]].copy()
    out["article_id"] = out["article_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values(["user_id", "timestamp", "article_id"])
    out = out.reset_index(drop=True)

    return out


def compute_point_in_time_category_preferences(
    history_events: Sequence[Tuple[pd.Timestamp, str, str]],
    impression_time: pd.Timestamp,
) -> Dict[str, float]:
    """
    Compute category preferences using only history timestamps < impression_time.

    Returns:
        category -> count / total eligible categorized history events
    """
    counts: Dict[str, int] = {}
    total = 0

    for timestamp, _article_id, category in history_events:
        if timestamp >= impression_time:
            break
        if not category:
            continue
        counts[category] = counts.get(category, 0) + 1
        total += 1

    if total == 0:
        return {}

    return {
        category: count / total
        for category, count in counts.items()
    }


def freshness_score(
    age_hours: float,
    decay_hours: float = DEFAULT_DECAY_HOURS,
) -> float:
    """Exponential freshness decay; age 0 receives score 1.0."""
    if decay_hours <= 0:
        raise ValueError("decay_hours must be > 0.")
    if age_hours < 0:
        raise ValueError("age_hours must be >= 0.")
    return float(np.exp(-age_hours / decay_hours))


@dataclass(frozen=True)
class FreshCandidate:
    article_id: str
    category_str: str
    published_time: pd.Timestamp
    category_preference: float
    freshness: float
    score: float


class FreshCategoryCandidateGenerator:
    """
    Corpus-level fresh/category candidate generator.

    Resources are built once and then queried for many impressions.

    The point-in-time category state is computed from user history with a
    strict timestamp cutoff. Article eligibility is independently enforced
    with published_time <= impression_time.
    """

    def __init__(
        self,
        articles: pd.DataFrame,
        history: pd.DataFrame,
        freshness_window_hours: float = DEFAULT_FRESHNESS_WINDOW_HOURS,
        decay_hours: float = DEFAULT_DECAY_HOURS,
        top_categories: int = DEFAULT_TOP_CATEGORIES,
        per_category: int = DEFAULT_PER_CATEGORY,
    ) -> None:
        if freshness_window_hours <= 0:
            raise ValueError("freshness_window_hours must be > 0.")
        if decay_hours <= 0:
            raise ValueError("decay_hours must be > 0.")
        if top_categories <= 0:
            raise ValueError("top_categories must be > 0.")
        if per_category <= 0:
            raise ValueError("per_category must be > 0.")

        self.articles = prepare_articles(articles)
        self.history = prepare_history(history)

        self.freshness_window_hours = float(freshness_window_hours)
        self.decay_hours = float(decay_hours)
        self.top_categories = int(top_categories)
        self.per_category = int(per_category)

        # article lookup
        self._article_by_id = (
            self.articles.set_index("article_id", drop=False).to_dict("index")
        )

        # Category -> articles sorted by publication time.
        self._category_articles: Dict[
            str, List[Tuple[pd.Timestamp, str]]
        ] = {}
        for row in self.articles.itertuples(index=False):
            self._category_articles.setdefault(
                str(row.category_str), []
            ).append((row.published_time, str(row.article_id)))

        for category in self._category_articles:
            self._category_articles[category].sort(
                key=lambda x: (x[0], x[1])
            )

        # User -> chronological (timestamp, article_id, category) events.
        # Category is joined once here so query-time work stays small.
        history_with_category = self.history.merge(
            self.articles[["article_id", "category_str"]],
            on="article_id",
            how="left",
        )
        history_with_category["category_str"] = (
            history_with_category["category_str"]
            .fillna("")
            .astype(str)
        )

        self._user_history: Dict[
            object, List[Tuple[pd.Timestamp, str, str]]
        ] = {}

        for row in history_with_category.itertuples(index=False):
            self._user_history.setdefault(row.user_id, []).append(
                (
                    row.timestamp,
                    str(row.article_id),
                    row.category_str,
                )
            )

        for user_id in self._user_history:
            self._user_history[user_id].sort(
                key=lambda x: (x[0], x[1])
            )

    def get_point_in_time_preferences(
        self,
        user_id: object,
        impression_time: pd.Timestamp,
    ) -> Dict[str, float]:
        """Return P(category | user history strictly before impression_time)."""
        events = self._user_history.get(user_id, [])
        if not events:
            return {}

        timestamps = [event[0] for event in events]
        cutoff = bisect_left(timestamps, impression_time)

        counts: Dict[str, int] = {}
        total = 0

        for _timestamp, _article_id, category in events[:cutoff]:
            if not category:
                continue
            counts[category] = counts.get(category, 0) + 1
            total += 1

        if total == 0:
            return {}

        return {
            category: count / total
            for category, count in counts.items()
        }

    def _recent_articles_for_category(
        self,
        category: str,
        impression_time: pd.Timestamp,
    ) -> Iterable[Tuple[pd.Timestamp, str]]:
        """Yield articles in category with publication time in the valid window."""
        articles = self._category_articles.get(category, [])
        if not articles:
            return []

        window_start = impression_time - pd.Timedelta(
            hours=self.freshness_window_hours
        )

        publication_times = [item[0] for item in articles]
        start = bisect_left(publication_times, window_start)
        end = bisect_left(publication_times, impression_time + pd.Timedelta(nanoseconds=1))

        return articles[start:end]

    def generate_candidates(
        self,
        user_id: object,
        impression_time: pd.Timestamp,
        top_k: int = 100,
        top_categories: Optional[int] = None,
        per_category: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate fresh, category-conditioned candidates from the corpus.

        Eligibility:
            published_time >= T-window
            published_time <= T

        Ranking:
            score = category_preference * freshness

        Returns a DataFrame sorted deterministically by score descending.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")

        impression_time = pd.Timestamp(impression_time)
        if pd.isna(impression_time):
            raise ValueError("impression_time must be a valid timestamp.")

        n_categories = (
            self.top_categories
            if top_categories is None
            else int(top_categories)
        )
        n_per_category = (
            self.per_category
            if per_category is None
            else int(per_category)
        )

        if n_categories <= 0:
            raise ValueError("top_categories must be > 0.")
        if n_per_category <= 0:
            raise ValueError("per_category must be > 0.")

        preferences = self.get_point_in_time_preferences(
            user_id, impression_time
        )

        if not preferences:
            return self._empty_result()

        ranked_categories = sorted(
            preferences.items(),
            key=lambda x: (-x[1], x[0]),
        )[:n_categories]

        candidates: Dict[str, FreshCandidate] = {}

        for category, preference in ranked_categories:
            recent = self._recent_articles_for_category(
                category, impression_time
            )

            # Most recent articles first for the per-category cap.
            recent = sorted(
                recent,
                key=lambda x: (-x[0].value, x[1]),
            )[:n_per_category]

            for published_time, article_id in recent:
                age_hours = (
                    impression_time - published_time
                ).total_seconds() / 3600.0

                # Defensive temporal checks.
                if published_time > impression_time:
                    continue
                if age_hours < 0 or age_hours > self.freshness_window_hours:
                    continue

                freshness = freshness_score(
                    age_hours,
                    self.decay_hours,
                )
                score = float(preference * freshness)

                candidates[article_id] = FreshCandidate(
                    article_id=article_id,
                    category_str=category,
                    published_time=published_time,
                    category_preference=float(preference),
                    freshness=freshness,
                    score=score,
                )

        ranked = sorted(
            candidates.values(),
            key=lambda x: (
                -x.score,
                -x.category_preference,
                -x.freshness,
                -x.published_time.value,
                x.article_id,
            ),
        )[:top_k]

        return pd.DataFrame(
            [
                {
                    "article_id": item.article_id,
                    "category_str": item.category_str,
                    "published_time": item.published_time,
                    "category_preference": item.category_preference,
                    "freshness": item.freshness,
                    "score": item.score,
                }
                for item in ranked
            ]
        )

    def retrieve(
        self,
        user_id: object,
        impression_time: pd.Timestamp,
        top_k: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """Alias matching the retrieve-style API used by other retrievers."""
        return self.generate_candidates(
            user_id=user_id,
            impression_time=impression_time,
            top_k=top_k,
            **kwargs,
        )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "article_id",
                "category_str",
                "published_time",
                "category_preference",
                "freshness",
                "score",
            ]
        )


def load_small_resources(
    processed_root: Path = Path("data/processed/ebnerd/small"),
    features_root: Path = Path("data/features/ebnerd/small"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the existing Small EB-NeRD article and validation-history resources.

    This deliberately uses the point-in-time validation history rather than
    the global descriptive user_category_preferences artifact.
    """
    articles_path = features_root / "articles.parquet"
    history_path = processed_root / "validation_user_history.parquet"

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found: {articles_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Validation history not found: {history_path}")

    articles = pd.read_parquet(articles_path)
    history = pd.read_parquet(history_path)

    return articles, history


if __name__ == "__main__":
    articles, history = load_small_resources()

    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=48.0,
        decay_hours=24.0,
        top_categories=3,
        per_category=100,
    )

    print("=" * 80)
    print("EB-NeRD FRESH + CATEGORY CANDIDATE GENERATOR")
    print("=" * 80)
    print(f"Articles: {len(articles):,}")
    print(f"History rows: {len(history):,}")
    print(
        "Configuration: "
        f"window={generator.freshness_window_hours:.1f}h, "
        f"decay={generator.decay_hours:.1f}h, "
        f"top_categories={generator.top_categories}, "
        f"per_category={generator.per_category}"
    )
