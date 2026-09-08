"""
Phase 4D: Isolated Fresh + Category Candidate Recall Evaluation for EB-NeRD.

Evaluates only the fresh/category candidate-generation branch. It does not
use BM25, semantic retrieval, RRF, or the official impression candidate list
for generation.

Metrics:
    Recall@50
    Recall@100
    Recall@200
    Recall@500

A hit is counted when at least one clicked article for an impression appears
in the generated corpus-level candidate pool.

Temporal safety is also checked:
    candidate.published_time <= impression_time
    candidate.published_time >= impression_time - freshness_window
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

from src.retrieval.ebnerd_fresh_candidates import (
    FreshCategoryCandidateGenerator,
)


PROCESSED_ROOT = Path("data/processed/ebnerd/small")
FEATURES_ROOT = Path("data/features/ebnerd/small")
RESULTS_ROOT = Path("data/results/ebnerd/small")

ARTICLES_PATH = FEATURES_ROOT / "articles.parquet"
HISTORY_PATH = PROCESSED_ROOT / "validation_user_history.parquet"
INTERACTIONS_PATH = PROCESSED_ROOT / "splits" / "validation.parquet"

OUTPUT_PATH = RESULTS_ROOT / "fresh_category_recall_smoke.csv"

SMOKE_IMPRESSIONS = 1000

FRESHNESS_WINDOW_HOURS = 48.0
DECAY_HOURS = 24.0
TOP_CATEGORIES = 3
PER_CATEGORY = 200

RECALL_KS = (50, 100, 200, 500)


def validate_paths() -> None:
    for path in (ARTICLES_PATH, HISTORY_PATH, INTERACTIONS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def load_validation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_paths()

    articles = pd.read_parquet(ARTICLES_PATH)
    history = pd.read_parquet(HISTORY_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)

    interactions["article_id"] = interactions["article_id"].astype(str)

    return articles, history, interactions


def group_clicked_articles(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per impression containing all clicked article IDs.

    EB-NeRD can contain more than one clicked article for an impression.
    """
    clicked = interactions.loc[
        interactions["clicked"].astype(int) == 1,
        ["impression_id", "article_id"],
    ].copy()

    return (
        clicked.groupby("impression_id")["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .reset_index(name="clicked_articles")
    )


def recall_at_k(
    candidates: pd.DataFrame,
    clicked_articles: Set[str],
    k: int,
) -> bool:
    if not clicked_articles or candidates.empty:
        return False

    candidate_ids = set(candidates.head(k)["article_id"].astype(str))
    return bool(candidate_ids.intersection(clicked_articles))


def evaluate_smoke(
    max_impressions: int = SMOKE_IMPRESSIONS,
) -> pd.DataFrame:
    articles, history, interactions = load_validation_data()

    # Ensure deterministic impression order.
    impressions = (
        interactions[
            ["impression_id", "user_id", "impression_time"]
        ]
        .drop_duplicates("impression_id")
        .sort_values(["impression_time", "impression_id"])
        .head(max_impressions)
        .reset_index(drop=True)
    )

    clicks = group_clicked_articles(interactions)

    evaluation = impressions.merge(
        clicks,
        on="impression_id",
        how="left",
    )

    evaluation["clicked_articles"] = evaluation["clicked_articles"].apply(
        lambda value: value if isinstance(value, set) else set()
    )

    generator = FreshCategoryCandidateGenerator(
        articles=articles,
        history=history,
        freshness_window_hours=FRESHNESS_WINDOW_HOURS,
        decay_hours=DECAY_HOURS,
        top_categories=TOP_CATEGORIES,
        per_category=PER_CATEGORY,
    )

    rows = []
    total_generation_time = 0.0
    future_violations = 0
    stale_violations = 0
    empty_history_count = 0

    for row in evaluation.itertuples(index=False):
        start = time.perf_counter()

        candidates = generator.generate_candidates(
            user_id=row.user_id,
            impression_time=row.impression_time,
            top_k=max(RECALL_KS),
        )

        elapsed = time.perf_counter() - start
        total_generation_time += elapsed

        impression_time = pd.Timestamp(row.impression_time)
        window_start = (
            impression_time
            - pd.Timedelta(hours=FRESHNESS_WINDOW_HOURS)
        )

        if not candidates.empty:
            published = pd.to_datetime(
                candidates["published_time"],
                errors="coerce",
            )

            future_violations += int(
                (published > impression_time).sum()
            )
            stale_violations += int(
                (published < window_start).sum()
            )

        clicked = row.clicked_articles

        if not generator.get_point_in_time_preferences(
            row.user_id,
            impression_time,
        ):
            empty_history_count += 1

        result = {
            "impression_id": row.impression_id,
            "user_id": row.user_id,
            "impression_time": impression_time,
            "clicked_articles": len(clicked),
            "candidate_count": len(candidates),
            "generation_time_sec": elapsed,
        }

        for k in RECALL_KS:
            result[f"recall@{k}"] = int(
                recall_at_k(candidates, clicked, k)
            )

        rows.append(result)

    details = pd.DataFrame(rows)

    print("=" * 80)
    print("PHASE 4D — ISOLATED FRESH + CATEGORY CANDIDATE RECALL")
    print("=" * 80)
    print(f"Articles: {len(articles):,}")
    print(f"Validation impressions evaluated: {len(details):,}")
    print(
        "Configuration: "
        f"window={FRESHNESS_WINDOW_HOURS:.1f}h, "
        f"decay={DECAY_HOURS:.1f}h, "
        f"top_categories={TOP_CATEGORIES}, "
        f"per_category={PER_CATEGORY}"
    )
    print()

    if details.empty:
        print("No impressions evaluated.")
        return details

    print("Results:")
    for k in RECALL_KS:
        print(
            f"Recall@{k}: "
            f"{details[f'recall@{k}'].mean():.6f}"
        )

    print()
    print(
        f"Mean candidates: "
        f"{details['candidate_count'].mean():.2f}"
    )
    print(
        f"Mean generation time: "
        f"{details['generation_time_sec'].mean():.4f}s"
    )
    print(
        f"Total generation time: "
        f"{total_generation_time:.2f}s"
    )
    print(
        f"Impressions with empty point-in-time profile: "
        f"{empty_history_count:,}"
    )
    print(f"Future publication violations: {future_violations}")
    print(f"Stale-window violations: {stale_violations}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    details.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved: {OUTPUT_PATH}")

    return details


if __name__ == "__main__":
    evaluate_smoke()
