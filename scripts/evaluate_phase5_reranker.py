from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase5_reranker_evaluation import evaluate_ranking


PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "phase5"
    / "validation_predictions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "phase5"
    / "validation_metrics.json"
)


def main() -> None:
    print("=" * 70)
    print("PHASE 5.2 VALIDATION EVALUATION")
    print("=" * 70)

    print(f"Predictions: {PREDICTION_PATH}")

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_PATH}"
        )

    print("\n[1/3] Loading predictions...")
    predictions = pd.read_parquet(PREDICTION_PATH)

    print(f"       Rows:         {len(predictions):,}")
    print(f"       Impressions:  {predictions['impression_id'].nunique():,}")
    print(f"       Clicks:       {predictions['clicked'].sum():,}")

    expected_rows = 59_923
    expected_impressions = 5_000

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(predictions):,}"
        )

    if predictions["impression_id"].nunique() != expected_impressions:
        raise ValueError(
            f"Expected {expected_impressions:,} impressions, "
            f"found {predictions['impression_id'].nunique():,}"
        )

    if predictions[["impression_id", "article_id"]].duplicated().any():
        raise ValueError("Duplicate impression/article pairs detected.")

    if not predictions["score"].map(pd.notna).all():
        raise ValueError("NaN scores detected.")

    print("\n[2/3] Computing ranking metrics...")

    metrics = evaluate_ranking(predictions)

    print("\n[3/3] Results")
    print("-" * 70)

    for metric in [
        "auc",
        "mrr",
        "ndcg@5",
        "ndcg@10",
        "hit@5",
        "hit@10",
    ]:
        if metric in metrics:
            print(f"       {metric:<10}: {metrics[metric]:.6f}")

    metrics["prediction_rows"] = int(len(predictions))
    metrics["impressions"] = int(
        predictions["impression_id"].nunique()
    )
    metrics["clicks"] = int(predictions["clicked"].sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics:")
    print(f"       {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("PHASE 5.2 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()