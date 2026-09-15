from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ranking.phase5_reranker import Phase5Reranker
from src.ranking.phase5_reranker_dataset import load_phase5_dataset
from src.ranking.train_phase5_reranker import (
    collate_batch,
    make_impression_subset,
)


DEFAULT_PROJECT_ROOT = PROJECT_ROOT

DEFAULT_CHECKPOINT = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "models"
    / "phase5"
    / "phase5_reranker_best.pt"
)

DEFAULT_OUTPUT = (
    DEFAULT_PROJECT_ROOT
    / "data"
    / "predictions"
    / "phase5"
    / "validation_predictions.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 5.2 neural re-ranker "
            "predictions for the fixed validation subset."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--validation-impressions",
        type=int,
        default=5000,
        help=(
            "Number of complete validation impressions "
            "used for Experiment 1."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-history-length",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Training seed. Validation uses seed + 1, "
            "matching the training script."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def count_impressions(dataset) -> int:
    """Count unique impressions in a full dataset or Subset."""
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices

        return int(
            base_dataset.candidates.iloc[indices][
                "impression_id"
            ].nunique()
        )

    return int(
        dataset.candidates[
            "impression_id"
        ].nunique()
    )


def build_validation_subset(
    project_root: Path,
    validation_impressions: int,
    seed: int,
    max_history_length: int,
):
    """
    Load validation data and reproduce the exact
    impression-selection procedure used during training.

    Training used:
        validation seed = train seed + 1

    With Experiment 1:
        train seed = 42
        validation seed = 43
    """
    dataset, article_mapping = load_phase5_dataset(
        project_root=project_root,
        split="validation",
        max_history_length=max_history_length,
    )

    subset = make_impression_subset(
        dataset,
        validation_impressions,
        seed + 1,
    )

    return subset, article_mapping


def load_model(
    checkpoint_path: Path,
    num_articles: int,
    device: torch.device,
) -> tuple[Phase5Reranker, dict]:
    """Load the trained Phase 5.2 model and checkpoint metadata."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
    )

    embedding_dim = int(
        config.get(
            "embedding_dim",
            64,
        )
    )

    hidden_dim = int(
        config.get(
            "hidden_dim",
            64,
        )
    )

    model = Phase5Reranker(
        num_articles=num_articles,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        retrieval_dim=5,
        dropout=0.1,
        padding_idx=0,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def generate_predictions(
    model: Phase5Reranker,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Generate one prediction score for every candidate."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    scores = []

    with torch.no_grad():
        for batch in loader:
            history = batch[
                "history_article_ids"
            ].to(device)

            candidates = batch[
                "candidate_article_id"
            ].to(device)

            retrieval = batch[
                "retrieval_features"
            ].to(device)

            logits = model(
                history_article_ids=history,
                candidate_article_ids=candidates,
                retrieval_features=retrieval,
            )

            batch_scores = (
                logits.detach()
                .cpu()
                .numpy()
            )

            scores.append(batch_scores)

    if not scores:
        raise RuntimeError(
            "No predictions were generated."
        )

    return np.concatenate(
        scores,
        axis=0,
    )


def main() -> None:
    args = parse_args()

    if args.validation_impressions <= 0:
        raise ValueError(
            "--validation-impressions must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.max_history_length <= 0:
        raise ValueError(
            "--max-history-length must be positive."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    project_root = Path(
        args.project_root
    ).resolve()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            project_root / checkpoint_path
        )

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():
        output_path = (
            project_root / output_path
        )

    device = torch.device("cpu")

    print("=" * 70)
    print("PHASE 5.2 VALIDATION PREDICTION")
    print("=" * 70)
    print(f"Device:                 {device}")
    print(f"Checkpoint:             {checkpoint_path}")
    print(f"Validation impressions: {args.validation_impressions:,}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Max history:            {args.max_history_length}")
    print(f"Seed:                   {args.seed}")
    print(f"Validation seed:        {args.seed + 1}")
    print()

    print(
        "[1/5] Loading fixed validation subset..."
    )

    validation_dataset, article_mapping = (
        build_validation_subset(
            project_root=project_root,
            validation_impressions=(
                args.validation_impressions
            ),
            seed=args.seed,
            max_history_length=(
                args.max_history_length
            ),
        )
    )

    impression_count = count_impressions(
        validation_dataset
    )

    print(
        f"       Validation impressions: "
        f"{impression_count:,}"
    )

    print(
        f"       Validation candidates:   "
        f"{len(validation_dataset):,}"
    )

    if impression_count != args.validation_impressions:
        raise RuntimeError(
            "Validation impression count mismatch: "
            f"expected {args.validation_impressions:,}, "
            f"got {impression_count:,}"
        )

    print()
    print("[2/5] Loading best checkpoint...")

    model, checkpoint = load_model(
        checkpoint_path=checkpoint_path,
        num_articles=len(article_mapping),
        device=device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    checkpoint_validation_loss = checkpoint.get(
        "validation_loss"
    )

    print(
        f"       Checkpoint epoch:       "
        f"{checkpoint_epoch}"
    )

    print(
        f"       Checkpoint val loss:    "
        f"{checkpoint_validation_loss:.6f}"
    )

    print()
    print("[3/5] Generating candidate scores...")

    scores = generate_predictions(
        model=model,
        dataset=validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    if len(scores) != len(validation_dataset):
        raise RuntimeError(
            "Prediction count mismatch: "
            f"generated {len(scores):,}, "
            f"expected {len(validation_dataset):,}"
        )

    print(
        f"       Predictions generated: "
        f"{len(scores):,}"
    )

    print()
    print("[4/5] Building prediction table...")

    if isinstance(
        validation_dataset,
        Subset,
    ):
        base_dataset = validation_dataset.dataset
        indices = validation_dataset.indices

        metadata = (
            base_dataset.candidates
            .iloc[indices]
            .reset_index(drop=True)
            .copy()
        )
    else:
        metadata = (
            validation_dataset.candidates
            .reset_index(drop=True)
            .copy()
        )

    prediction_columns = [
        "impression_id",
        "user_id",
        "article_id",
        "impression_time",
        "clicked",
    ]

    missing_columns = set(
        prediction_columns
    ) - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Prediction metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions = metadata[
        prediction_columns
    ].copy()

    predictions["score"] = scores

    # Preserve a deterministic ordering.
    predictions = predictions.sort_values(
        [
            "impression_id",
            "article_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Basic integrity checks.
    if len(predictions) != len(
        validation_dataset
    ):
        raise RuntimeError(
            "Final prediction table has the wrong number "
            "of rows."
        )

    if predictions[
        ["impression_id", "article_id"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate impression/article pairs "
            "found in prediction table."
        )

    if not np.isfinite(
        predictions["score"].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Prediction scores contain NaN or infinite values."
        )

    final_impressions = predictions[
        "impression_id"
    ].nunique()

    if final_impressions != args.validation_impressions:
        raise RuntimeError(
            "Final prediction impression count mismatch: "
            f"expected {args.validation_impressions:,}, "
            f"got {final_impressions:,}"
        )

    print(
        f"       Rows:                 "
        f"{len(predictions):,}"
    )

    print(
        f"       Impressions:          "
        f"{final_impressions:,}"
    )

    print(
        f"       Clicks:               "
        f"{int(predictions['clicked'].sum()):,}"
    )

    print(
        f"       Score min:             "
        f"{predictions['score'].min():.6f}"
    )

    print(
        f"       Score max:             "
        f"{predictions['score'].max():.6f}"
    )

    print()
    print("[5/5] Saving predictions...")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"       Saved: {output_path}"
    )

    print()
    print("=" * 70)
    print("PHASE 5.2 PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
