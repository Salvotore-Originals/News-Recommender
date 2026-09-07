"""
Phase 3.5 — CPU smoke training for the local DIN baseline.

This script performs a small real-data training run on EB-NeRD Small.

Purpose:
    Validate the complete training pipeline before full training.

This is NOT the final Phase 3 training configuration.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.ranking.a2_din import DINModel
from src.ranking.a2_din_data import load_ebnerd_din_dataset


def set_seed(seed: int) -> None:
    """Make the smoke run reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Combine individual DIN examples into a mini-batch."""

    return {
        "history_article_ids": torch.stack(
            [item["history_article_ids"] for item in batch]
        ),
        "candidate_article_id": torch.stack(
            [item["candidate_article_id"] for item in batch]
        ),
        "clicked": torch.stack(
            [item["clicked"] for item in batch]
        ),
        "user_id": torch.stack(
            [item["user_id"] for item in batch]
        ),
    }


def train_one_epoch(
    model: DINModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        histories = batch["history_article_ids"].to(device)
        candidates = batch["candidate_article_id"].to(device)
        labels = batch["clicked"].to(device)

        optimizer.zero_grad()

        logits = model(
            history_article_ids=histories,
            candidate_article_ids=candidates,
        )

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    model: DINModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate average BCE loss."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        histories = batch["history_article_ids"].to(device)
        candidates = batch["candidate_article_id"].to(device)
        labels = batch["clicked"].to(device)

        logits = model(
            history_article_ids=histories,
            candidate_article_ids=candidates,
        )

        loss = criterion(logits, labels)

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU smoke training for EB-NeRD DIN."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--train-rows",
        type=int,
        default=10000,
        help="Number of candidate rows used for smoke training.",
    )

    parser.add_argument(
        "--validation-rows",
        type=int,
        default=2000,
        help="Number of validation candidate rows used for smoke evaluation.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
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
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    set_seed(args.seed)

    device = torch.device("cpu")

    print("=" * 70)
    print("PHASE 3.5 — DIN CPU SMOKE TRAINING")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Seed: {args.seed}")
    print(f"Training rows: {args.train_rows}")
    print(f"Validation rows: {args.validation_rows}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Max history length: {args.max_history_length}")
    print(f"Learning rate: {args.learning_rate}")
    print()

    # --------------------------------------------------------------
    # Load real EB-NeRD data
    # --------------------------------------------------------------

    print("[1/5] Loading EB-NeRD Small training data...")

    train_dataset, article_mapping = load_ebnerd_din_dataset(
        project_root=args.project_root,
        split="train",
        max_history_length=args.max_history_length,
    )

    print(f"       Full training candidates: {len(train_dataset):,}")

    # --------------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------------

    print("[2/5] Loading EB-NeRD Small validation data...")

    validation_dataset, validation_mapping = load_ebnerd_din_dataset(
        project_root=args.project_root,
        split="validation",
        max_history_length=args.max_history_length,
    )

    print(
        f"       Full validation candidates: "
        f"{len(validation_dataset):,}"
    )

    # Both splits must use the same article vocabulary for the model.
    if article_mapping != validation_mapping:
        raise RuntimeError(
            "Train and validation article mappings differ."
        )

    # --------------------------------------------------------------
    # Create deterministic subsets
    # --------------------------------------------------------------

    train_count = min(
        args.train_rows,
        len(train_dataset),
    )

    validation_count = min(
        args.validation_rows,
        len(validation_dataset),
    )

    train_indices = list(range(train_count))
    validation_indices = list(range(validation_count))

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    print(
        f"       Smoke training subset: "
        f"{len(train_subset):,}"
    )

    print(
        f"       Smoke validation subset: "
        f"{len(validation_subset):,}"
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    print("[3/5] Building DataLoaders...")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print("[4/5] Building DIN model...")

    model = DINModel(
        num_articles=len(article_mapping) + 1,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
        padding_idx=0,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"       Articles in vocabulary: "
        f"{len(article_mapping):,}"
    )

    print(
        f"       Trainable parameters: "
        f"{parameter_count:,}"
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # --------------------------------------------------------------
    # Training with best-checkpoint selection
    # --------------------------------------------------------------

    print("[5/5] Training...")
    print()

    initial_validation_loss = evaluate_loss(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"       Initial validation BCE: "
        f"{initial_validation_loss:.6f}"
    )

    training_losses = []
    validation_losses = []

    # The initial model is our first baseline for comparison.
    best_validation_loss = initial_validation_loss
    best_epoch = 0

    best_state_dict = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)

        # ----------------------------------------------------------
        # Best-checkpoint selection
        # ----------------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

            marker = " ← BEST"

        else:
            marker = ""

        print(
            f"       Epoch {epoch}/{args.epochs} | "
            f"train BCE: {train_loss:.6f} | "
            f"validation BCE: {validation_loss:.6f}"
            f"{marker}"
        )

    # --------------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------------

    model.load_state_dict(best_state_dict)

    # --------------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------------

    output_dir = (
        args.project_root
        / "data"
        / "models"
        / "a2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "din_controlled_100k_best.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_articles": len(article_mapping) + 1,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "max_history_length": args.max_history_length,
            "seed": args.seed,
            "train_rows": len(train_subset),
            "validation_rows": len(validation_subset),
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "initial_validation_loss": initial_validation_loss,
            "best_validation_loss": best_validation_loss,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        },
        checkpoint_path,
    )

    print()
    print("=" * 70)
    print("CONTROLLED DIN TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Initial validation BCE: "
        f"{initial_validation_loss:.6f}"
    )

    print(
        f"Best validation BCE:    "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Best epoch:              "
        f"{best_epoch}"
    )

    print(
        f"Final epoch BCE:         "
        f"{validation_losses[-1]:.6f}"
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )

    print("=" * 70)

if __name__ == "__main__":
    main()