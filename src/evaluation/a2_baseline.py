from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.behavioral import evaluate_score
from src.ranking.a2_baseline import A2BaselineRanker, SCORE_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "mind"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "a2" / "baseline_ranker.joblib"
DEFAULT_VALIDATION_PATH = FEATURE_DIR / "validation_behavioral_score.parquet"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "a2" / "validation_baseline.parquet"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "predictions" / "a2" / "validation_baseline_metrics.json"


def load_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation feature file not found: {path}")
    df = pd.read_parquet(path)
    required = {"impression_id", "user_id", "timestamp", "article_id", "clicked"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Validation data is missing required columns: {missing}")
    return df


def predict_validation(
    ranker: A2BaselineRanker,
    df: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    scored = ranker.add_scores(df, score_column=score_column)
    if len(scored) != len(df):
        raise AssertionError("Prediction changed the number of candidate rows.")
    if not scored["impression_id"].equals(df["impression_id"]):
        raise AssertionError("Prediction changed impression row alignment.")
    if not scored["article_id"].equals(df["article_id"]):
        raise AssertionError("Prediction changed article row alignment.")
    return scored


def evaluate_validation(
    scored: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
) -> dict[str, float]:
    return evaluate_score(scored, score_column=score_column, k_values=(5, 10))


def run(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, float]:
    print("=" * 70)
    print("A2 PHASE 2.7 — VALIDATION PREDICTION + EVALUATION")
    print("=" * 70)

    print(f"\nLoading model: {model_path}")
    ranker = A2BaselineRanker.load(model_path)

    print(f"Loading validation data: {validation_path}")
    df = load_validation(validation_path)
    print(f"Candidate rows: {len(df):,}")
    print(f"Impressions:    {df['impression_id'].nunique():,}")
    print(f"Clicks:         {int(df['clicked'].sum()):,}")

    scored = predict_validation(ranker, df)

    if not scored[SCORE_COLUMN].notna().all():
        raise ValueError("Validation predictions contain missing scores.")

    print(
        "\nScore statistics:"
        f"\n  min:  {scored[SCORE_COLUMN].min():.8f}"
        f"\n  max:  {scored[SCORE_COLUMN].max():.8f}"
        f"\n  mean: {scored[SCORE_COLUMN].mean():.8f}"
        f"\n  std:  {scored[SCORE_COLUMN].std():.8f}"
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(predictions_path, index=False)
    print(f"\nSaved predictions: {predictions_path}")

    metrics = evaluate_validation(scored)
    payload = {
        "phase": "A2 Phase 2.7",
        "model": "HistGradientBoostingClassifier",
        "score_column": SCORE_COLUMN,
        "evaluation_group": "impression_id",
        "validation_rows": int(len(scored)),
        "validation_impressions": int(scored["impression_id"].nunique()),
        "validation_clicks": int(scored["clicked"].sum()),
        **metrics,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics:     {metrics_path}")

    print("\nValidation ranking metrics:")
    for key in ("mrr", "hit@5", "hit@10", "ndcg@5", "ndcg@10"):
        print(f"  {key.upper():7s}: {metrics[key]:.6f}")

    print("\n" + "=" * 70)
    print("PHASE 2.7 COMPLETE")
    print("=" * 70)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()
    run(
        model_path=args.model,
        validation_path=args.validation,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
    )


if __name__ == "__main__":
    main()
