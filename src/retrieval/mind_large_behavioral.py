from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_NEWS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mind"
    / "test"
    / "news.tsv"
)

TRAIN_ARTICLE_STATS = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mind"
    / "article_stats.parquet"
)


# ============================================================
# WEIGHTS
# ============================================================

CATEGORY_WEIGHT = 0.40
SUBCATEGORY_WEIGHT = 0.60

INTEREST_WEIGHT = 0.45
RECENCY_WEIGHT = 0.35
POPULARITY_WEIGHT = 0.20

DECAY_RATE = 0.10
SMOOTHING_STRENGTH = 20.0


# ============================================================
# ARTICLE LOOKUPS
# ============================================================

def load_article_metadata() -> tuple[dict, dict]:
    """
    Load official MIND Large article metadata.

    Returns
    -------
    category_lookup:
        article_id -> category

    subcategory_lookup:
        article_id -> subcategory
    """

    columns = [
        "article_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    articles = pd.read_csv(
        TEST_NEWS,
        sep="\t",
        header=None,
        names=columns,
        dtype={
            "article_id": "string",
            "category": "string",
            "subcategory": "string",
        },
        usecols=[
            "article_id",
            "category",
            "subcategory",
        ],
        keep_default_na=False,
    )

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
    )

    articles["category"] = (
        articles["category"]
        .fillna("")
        .astype(str)
    )

    articles["subcategory"] = (
        articles["subcategory"]
        .fillna("")
        .astype(str)
    )

    category_lookup = dict(
        zip(
            articles["article_id"],
            articles["category"],
        )
    )

    subcategory_lookup = dict(
        zip(
            articles["article_id"],
            articles["subcategory"],
        )
    )

    return (
        category_lookup,
        subcategory_lookup,
    )


# ============================================================
# TRAIN POPULARITY
# ============================================================

def load_train_popularity() -> dict[str, float]:
    """
    Build TRAIN-only smoothed CTR lookup.

    Unseen test articles receive global TRAIN CTR.
    """

    stats = pd.read_parquet(
        TRAIN_ARTICLE_STATS
    )

    required = {
        "article_id",
        "impression_count",
        "click_count",
    }

    missing = required - set(stats.columns)

    if missing:
        raise ValueError(
            "article_stats.parquet is missing "
            f"columns: {sorted(missing)}"
        )

    total_impressions = (
        stats["impression_count"]
        .sum()
    )

    total_clicks = (
        stats["click_count"]
        .sum()
    )

    if total_impressions <= 0:
        raise ValueError(
            "TRAIN impression count must be positive."
        )

    global_ctr = (
        total_clicks
        / total_impressions
    )

    m = SMOOTHING_STRENGTH

    stats["smoothed_ctr"] = (
        stats["click_count"]
        + m * global_ctr
    ) / (
        stats["impression_count"]
        + m
    )

    return {
        str(article_id): float(score)
        for article_id, score
        in zip(
            stats["article_id"],
            stats["smoothed_ctr"],
        )
    }, float(global_ctr)


# ============================================================
# HISTORY → BEHAVIORAL STATE
# ============================================================

def build_history_state(
    history: list[str],
    category_lookup: dict[str, str],
    subcategory_lookup: dict[str, str],
) -> tuple[
    Counter,
    Counter,
    Counter,
    Counter,
    float,
]:
    """
    Convert one MIND history into behavioral statistics.

    history is ordered oldest -> newest.

    Recency weight:

        exp(-decay_rate * distance_from_latest)
    """

    category_counts = Counter()
    subcategory_counts = Counter()

    category_recency = Counter()
    subcategory_recency = Counter()

    total_recency = 0.0

    history_length = len(history)

    for position, article_id in enumerate(history):

        category = category_lookup.get(
            article_id,
            "",
        )

        subcategory = subcategory_lookup.get(
            article_id,
            "",
        )

        if category:
            category_counts[category] += 1

        if subcategory:
            subcategory_counts[subcategory] += 1

        distance = (
            history_length
            - 1
            - position
        )

        weight = np.exp(
            -DECAY_RATE * distance
        )

        total_recency += weight

        if category:
            category_recency[category] += weight

        if subcategory:
            subcategory_recency[subcategory] += weight

    return (
        category_counts,
        subcategory_counts,
        category_recency,
        subcategory_recency,
        total_recency,
    )


# ============================================================
# CANDIDATE BEHAVIORAL SCORES
# ============================================================

def score_candidates(
    candidate_ids: list[str],
    history: list[str],
    category_lookup: dict[str, str],
    subcategory_lookup: dict[str, str],
    popularity_lookup: dict[str, float],
    global_ctr: float,
) -> np.ndarray:
    """
    Calculate MIND behavioral score for candidates.

    No clicked labels are used.
    """

    (
        category_counts,
        subcategory_counts,
        category_recency,
        subcategory_recency,
        total_recency,
    ) = build_history_state(
        history,
        category_lookup,
        subcategory_lookup,
    )

    total_clicks = len(history)

    scores = np.zeros(
        len(candidate_ids),
        dtype=np.float64,
    )

    for index, article_id in enumerate(
        candidate_ids
    ):

        category = category_lookup.get(
            article_id,
            "",
        )

        subcategory = subcategory_lookup.get(
            article_id,
            "",
        )

        # ----------------------------------------------------
        # Interest
        # ----------------------------------------------------

        if total_clicks > 0:

            category_preference = (
                category_counts.get(
                    category,
                    0,
                )
                / total_clicks
            )

            subcategory_preference = (
                subcategory_counts.get(
                    subcategory,
                    0,
                )
                / total_clicks
            )

        else:

            category_preference = 0.0
            subcategory_preference = 0.0

        interest_score = (
            CATEGORY_WEIGHT
            * category_preference
            +
            SUBCATEGORY_WEIGHT
            * subcategory_preference
        )

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

        if total_recency > 0:

            category_recency_score = (
                category_recency.get(
                    category,
                    0.0,
                )
                / total_recency
            )

            subcategory_recency_score = (
                subcategory_recency.get(
                    subcategory,
                    0.0,
                )
                / total_recency
            )

        else:

            category_recency_score = 0.0
            subcategory_recency_score = 0.0

        recency_score = (
            CATEGORY_WEIGHT
            * category_recency_score
            +
            SUBCATEGORY_WEIGHT
            * subcategory_recency_score
        )

        # ----------------------------------------------------
        # Popularity
        # ----------------------------------------------------

        popularity_score = (
            popularity_lookup.get(
                article_id,
                global_ctr,
            )
        )

        # ----------------------------------------------------
        # Final behavioral score
        # ----------------------------------------------------

        scores[index] = (
            INTEREST_WEIGHT
            * interest_score
            +
            RECENCY_WEIGHT
            * recency_score
            +
            POPULARITY_WEIGHT
            * popularity_score
        )

    return scores