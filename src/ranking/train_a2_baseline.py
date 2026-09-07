from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ranking.a2_baseline import (
    A2BaselineRanker,
    FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "mind"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "a2" / "baseline_ranker.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "models" / "a2" / "baseline_ranker_metadata.json"


def load_split(split: str) -> pd.DataFrame:
    path = FEATURE_DIR / f"{split}_behavioral_score.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_parquet(path)


def train(
    *,
    train_path: Path,
    model_path: Path,
    metadata_path: Path,
    max_iter: int = 200,
) -> dict:
    df = pd.read_parquet(train_path)

    ranker = A2BaselineRanker(max_iter=max_iter)
    ranker.fit(df)
    ranker.save(model_path)

    metadata = {
        "phase": "A2 Phase 2",
        "model": "HistGradientBoostingClassifier",
        "objective": "pointwise click prediction used for ranking",
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "clicked",
        "group_column": "impression_id",
        "training_rows": int(len(df)),
        "training_impressions": int(df["impression_id"].nunique()),
        "training_clicks": int(df["clicked"].sum()),
        "training_non_clicks": int((df["clicked"] == 0).sum()),
        "max_iter": max_iter,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the A2 Phase 2 baseline ranker.")
    parser.add_argument("--train", type=Path, default=FEATURE_DIR / "train_behavioral_score.parquet")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    metadata = train(
        train_path=args.train,
        model_path=args.model,
        metadata_path=args.metadata,
        max_iter=args.max_iter,
    )

    print("=" * 70)
    print("A2 PHASE 2 BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Rows:        {metadata['training_rows']:,}")
    print(f"Impressions: {metadata['training_impressions']:,}")
    print(f"Clicks:      {metadata['training_clicks']:,}")
    print(f"Model:       {metadata['model']}")
    print(f"Saved:       {args.model}")


if __name__ == "__main__":
    main()
