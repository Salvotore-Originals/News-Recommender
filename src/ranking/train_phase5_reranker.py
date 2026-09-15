from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.ranking.phase5_reranker import Phase5Reranker
from src.ranking.phase5_reranker_dataset import load_phase5_dataset


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "data" / "models" / "phase5"


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset(dataset, max_rows: int | None, seed: int):
    """Return the full dataset or a deterministic row-level subset."""
    if max_rows is None:
        return dataset

    if max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")

    if max_rows >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset),
        size=max_rows,
        replace=False,
    )
    indices = np.sort(indices)

    return Subset(dataset, indices.tolist())


def make_impression_subset(
    dataset,
    max_impressions: int | None,
    seed: int,
):
    """Return a deterministic subset containing complete impressions."""
    if max_impressions is None:
        return dataset

    if max_impressions <= 0:
        raise ValueError(
            "max_impressions must be positive when provided."
        )

    # EBNeRDPhase5Dataset stores the original candidate table.
    impression_ids = dataset.candidates[
        "impression_id"
    ].drop_duplicates().to_numpy()

    if max_impressions >= len(impression_ids):
        return dataset

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        impression_ids,
        size=max_impressions,
        replace=False,
    )

    selected = np.sort(selected)

    selected_set = set(
        int(impression_id)
        for impression_id in selected
    )

    row_indices = [
        index
        for index, impression_id
        in enumerate(dataset.candidates["impression_id"])
        if int(impression_id) in selected_set
    ]

    return Subset(dataset, row_indices)


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
        dataset.candidates["impression_id"].nunique()
    )


def collate_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack dataset samples into a training batch."""
    return {
        "history_article_ids": torch.stack(
            [
                item["history_article_ids"]
                for item in batch
            ]
        ),
        "candidate_article_id": torch.stack(
            [
                item["candidate_article_id"]
                for item in batch
            ]
        ),
        "retrieval_features": torch.stack(
            [
                item["retrieval_features"]
                for item in batch
            ]
        ),
        "clicked": torch.stack(
            [
                item["clicked"]
                for item in batch
            ]
        ),
        "user_id": torch.stack(
            [
                item["user_id"]
                for item in batch
            ]
        ),
    }


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for Phase 5.2."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one training or validation epoch."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        history = batch["history_article_ids"].to(device)
        candidates = batch["candidate_article_id"].to(device)
        retrieval = batch["retrieval_features"].to(device)
        labels = batch["clicked"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(
                history_article_ids=history,
                candidate_article_ids=candidates,
                retrieval_features=retrieval,
            )

            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError(
            "The DataLoader produced zero examples."
        )

    return total_loss / total_examples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    args: argparse.Namespace,
) -> None:
    """Save the best Phase 5.2 checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "config": vars(args),
    }

    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Phase 5.2 neural re-ranker on EB-NeRD."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--train-rows",
        type=int,
        default=5000,
        help=(
            "Maximum number of training rows. "
            "Use 0 for all rows."
        ),
    )

    parser.add_argument(
        "--validation-rows",
        type=int,
        default=2000,
        help=(
            "Maximum number of validation rows. "
            "Use 0 for all rows."
        ),
    )

    parser.add_argument(
        "--train-impressions",
        type=int,
        default=None,
        help=(
            "Maximum number of complete training impressions. "
            "Overrides --train-rows when provided."
        ),
    )

    parser.add_argument(
        "--validation-impressions",
        type=int,
        default=None,
        help=(
            "Maximum number of complete validation impressions. "
            "Overrides --validation-rows when provided."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
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
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. Keep 0 for the first "
            "Windows smoke test."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a tiny controlled smoke-training configuration."
        ),
    )

    return parser.parse_args()


def apply_smoke_settings(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Apply intentionally small settings for the smoke test."""
    if not args.smoke:
        return args

    args.train_rows = 1000
    args.validation_rows = 500

    # Explicitly disable impression-based selection in smoke mode.
    args.train_impressions = None
    args.validation_impressions = None

    args.epochs = 1
    args.batch_size = 32
    args.embedding_dim = 32
    args.hidden_dim = 32
    args.max_history_length = 20
    args.learning_rate = 1e-3

    return args


def main() -> None:
    args = parse_args()
    args = apply_smoke_settings(args)

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.max_history_length <= 0:
        raise ValueError(
            "--max-history-length must be positive."
        )

    if (
        args.train_impressions is not None
        and args.train_impressions <= 0
    ):
        raise ValueError(
            "--train-impressions must be positive."
        )

    if (
        args.validation_impressions is not None
        and args.validation_impressions <= 0
    ):
        raise ValueError(
            "--validation-impressions must be positive."
        )

    if args.train_rows == 0:
        args.train_rows = None

    if args.validation_rows == 0:
        args.validation_rows = None

    set_seed(args.seed)

    # Phase 5.2 is deliberately CPU-first.
    device = torch.device("cpu")

    print("=" * 70)
    print("PHASE 5.2 NEURAL RE-RANKER TRAINING")
    print("=" * 70)
    print(f"Device:              {device}")
    print(f"Seed:                {args.seed}")
    print(f"Train rows:          {args.train_rows}")
    print(f"Validation rows:     {args.validation_rows}")
    print(f"Train impressions:   {args.train_impressions}")
    print(
        f"Validation impressions: "
        f"{args.validation_impressions}"
    )
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Learning rate:       {args.learning_rate}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden dimension:    {args.hidden_dim}")
    print(f"Max history:         {args.max_history_length}")
    print(f"Smoke mode:          {args.smoke}")
    print()

    print(
        "[1/6] Loading Phase 5.2 training dataset..."
    )

    train_dataset, article_mapping = load_phase5_dataset(
        project_root=args.project_root,
        split="train",
        max_history_length=args.max_history_length,
    )

    print(
        f"       Full train rows: "
        f"{len(train_dataset):,}"
    )

    print(
        f"       Articles:        "
        f"{len(article_mapping):,}"
    )

    print()
    print(
        "[2/6] Loading Phase 5.2 validation dataset..."
    )

    validation_dataset, _ = load_phase5_dataset(
        project_root=args.project_root,
        split="validation",
        max_history_length=args.max_history_length,
    )

    print(
        f"       Full validation rows: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------------
    # Select complete impressions when requested.
    # Otherwise retain the original row-level selection behaviour.
    # --------------------------------------------------------------
    if args.train_impressions is not None:
        train_dataset = make_impression_subset(
            train_dataset,
            args.train_impressions,
            args.seed,
        )
    else:
        train_dataset = make_subset(
            train_dataset,
            args.train_rows,
            args.seed,
        )

    if args.validation_impressions is not None:
        validation_dataset = make_impression_subset(
            validation_dataset,
            args.validation_impressions,
            args.seed + 1,
        )
    else:
        validation_dataset = make_subset(
            validation_dataset,
            args.validation_rows,
            args.seed + 1,
        )

    train_impression_count = count_impressions(
        train_dataset
    )

    validation_impression_count = count_impressions(
        validation_dataset
    )

    print()
    print(
        f"       Selected train rows:      "
        f"{len(train_dataset):,}"
    )

    print(
        f"       Selected validation rows: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"       Train impressions:         "
        f"{train_impression_count:,}"
    )

    print(
        f"       Validation impressions:    "
        f"{validation_impression_count:,}"
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validation_loader = build_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print()
    print(
        "[3/6] Building Phase 5.2 model..."
    )

    model = Phase5Reranker(
        num_articles=len(article_mapping),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        retrieval_dim=5,
        dropout=0.1,
        padding_idx=0,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"       Trainable parameters: "
        f"{parameter_count:,}"
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    print()
    print("[4/6] Training...")
    print("-" * 70)

    best_validation_loss = float("inf")
    best_epoch = None
    history = []

    checkpoint_path = (
        args.output_dir
        / "phase5_reranker_best.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }

        history.append(history_entry)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                args=args,
            )

            print(
                f"             New best checkpoint: "
                f"{checkpoint_path}"
            )

    print("-" * 70)

    metrics_path = (
        args.output_dir
        / "phase5_reranker_training_metrics.json"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "device": str(device),
        "seed": args.seed,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "train_impressions": train_impression_count,
        "validation_impressions": validation_impression_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "max_history_length": args.max_history_length,
        "retrieval_dim": 5,
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    print()
    print("[5/6] Saved training artifacts")
    print(
        f"       Best checkpoint: "
        f"{checkpoint_path}"
    )
    print(
        f"       Metrics:         "
        f"{metrics_path}"
    )

    print()
    print("[6/6] Training complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

