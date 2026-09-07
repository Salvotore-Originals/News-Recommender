"""
Phase 3.6 — DIN validation and impression-level ranking metrics.

Metrics:
    - avgAUC
    - MRR
    - Hit@5
    - Hit@10
    - NDCG@5
    - NDCG@10

Evaluation is performed at the impression level.

Important:
    The published official DIN baseline reports avgAUC.
    Our local reproduction reports avgAUC as well, while retaining
    the ranking metrics used elsewhere in this project.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.ranking.a2_din import DINModel
from src.ranking.a2_din_data import (
    build_article_id_mapping,
    EBNeRDDINDataset,
    load_ebnerd_din_dataset,
)

def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Combine DIN examples into a batch."""

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


class IndexedDataset(Dataset):
    """
    Dataset wrapper returning the original candidate index together
    with the DIN tensors.

    The original index is required to reconstruct impression groups.
    """

    def __init__(
        self,
        dataset: EBNeRDDINDataset,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        original_index = self.indices[index]

        item = self.dataset[original_index]

        item["candidate_index"] = torch.tensor(
            original_index,
            dtype=torch.long,
        )

        return item


def collate_indexed_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate indexed DIN examples."""

    result = collate_fn(batch)

    result["candidate_index"] = torch.stack(
        [item["candidate_index"] for item in batch]
    )

    return result


def roc_auc_for_impression(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """
    Calculate AUC for one impression.

    AUC can only be defined when the impression contains at least
    one positive and one negative candidate.

    For binary labels, AUC is equivalent to:

        P(score_positive > score_negative)

    with ties receiving half credit.
    """

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return None

    # Compare every positive against every negative.
    comparisons = (
        positive_scores[:, None]
        - negative_scores[None, :]
    )

    greater = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)

    total_pairs = (
        len(positive_scores)
        * len(negative_scores)
    )

    return float(
        (greater + 0.5 * ties) / total_pairs
    )


def reciprocal_rank(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Calculate reciprocal rank of the first clicked article."""

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranked_labels = labels[order]

    positive_positions = np.flatnonzero(
        ranked_labels == 1
    )

    if len(positive_positions) == 0:
        return 0.0

    return float(
        1.0 / (positive_positions[0] + 1)
    )


def hit_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Return 1 when a clicked article appears in top-k."""

    order = np.argsort(
        -scores,
        kind="stable",
    )

    top_k = order[:k]

    return float(
        np.any(labels[top_k] == 1)
    )


def ndcg_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Calculate binary-label NDCG@k.

    Since clicked is binary, DCG is:

        sum((2^rel - 1) / log2(rank + 1))
    """

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranked_labels = labels[order][:k]

    if len(ranked_labels) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(ranked_labels) + 2)
    )

    dcg = np.sum(
        ((2 ** ranked_labels) - 1)
        / discounts
    )

    ideal_labels = np.sort(
        labels
    )[::-1][:k]

    ideal_discounts = np.log2(
        np.arange(2, len(ideal_labels) + 2)
    )

    idcg = np.sum(
        ((2 ** ideal_labels) - 1)
        / ideal_discounts
    )

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def evaluate_impressions(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Evaluate prediction rows grouped by impression_id.

    Required columns:
        impression_id
        score
        clicked
    """

    required = {
        "impression_id",
        "score",
        "clicked",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Predictions missing columns: {sorted(missing)}"
        )

    impression_metrics: list[dict[str, float]] = []

    for _, group in predictions.groupby(
        "impression_id",
        sort=False,
    ):
        scores = group["score"].to_numpy(
            dtype=np.float64
        )

        labels = group["clicked"].to_numpy(
            dtype=np.int64
        )

        auc = roc_auc_for_impression(
            scores,
            labels,
        )

        rr = reciprocal_rank(
            scores,
            labels,
        )

        hit5 = hit_at_k(
            scores,
            labels,
            5,
        )

        hit10 = hit_at_k(
            scores,
            labels,
            10,
        )

        ndcg5 = ndcg_at_k(
            scores,
            labels,
            5,
        )

        ndcg10 = ndcg_at_k(
            scores,
            labels,
            10,
        )

        impression_metrics.append(
            {
                "mrr": rr,
                "hit@5": hit5,
                "hit@10": hit10,
                "ndcg@5": ndcg5,
                "ndcg@10": ndcg10,
                "auc": (
                    np.nan
                    if auc is None
                    else auc
                ),
            }
        )

    metrics_df = pd.DataFrame(
        impression_metrics
    )

    auc_values = metrics_df["auc"].dropna()

    return {
        "impressions": int(
            len(metrics_df)
        ),
        "auc_valid_impressions": int(
            len(auc_values)
        ),
        "avgAUC": float(
            auc_values.mean()
        )
        if len(auc_values) > 0
        else float("nan"),
        "MRR": float(
            metrics_df["mrr"].mean()
        ),
        "Hit@5": float(
            metrics_df["hit@5"].mean()
        ),
        "Hit@10": float(
            metrics_df["hit@10"].mean()
        ),
        "NDCG@5": float(
            metrics_df["ndcg@5"].mean()
        ),
        "NDCG@10": float(
            metrics_df["ndcg@10"].mean()
        ),
    }


@torch.no_grad()
def generate_predictions(
    model: DINModel,
    dataset: IndexedDataset,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Generate predictions for a set of candidate rows.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_indexed_fn,
    )

    model.eval()

    rows: list[dict[str, float | int]] = []

    for batch in loader:

        histories = batch[
            "history_article_ids"
        ].to(device)

        candidates = batch[
            "candidate_article_id"
        ].to(device)

        logits = model(
            history_article_ids=histories,
            candidate_article_ids=candidates,
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

        candidate_indices = (
            batch["candidate_index"]
            .numpy()
        )

        labels = (
            batch["clicked"]
            .numpy()
        )

        for candidate_index, score, label in zip(
            candidate_indices,
            probabilities,
            labels,
        ):
            row = dataset.dataset.candidates.iloc[
                int(candidate_index)
            ]

            rows.append(
                {
                    "impression_id": int(
                        row["impression_id"]
                    ),
                    "article_id": int(
                        row["article_id"]
                    ),
                    "score": float(score),
                    "clicked": int(label),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local DIN on EB-NeRD."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help=(
            "Maximum number of validation candidate rows. "
            "Complete impressions are selected."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-history-length",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    project_root = args.project_root

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = (
            project_root
            / "data"
            / "models"
            / "a2"
            / "din_smoke_test.pt"
        )

    print("=" * 70)
    print("PHASE 3.6 — DIN VALIDATION")
    print("=" * 70)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: cpu")
    print(f"Requested candidate rows: {args.rows}")
    print()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # --------------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------------

    print("[1/4] Loading EB-NeRD validation data...")

    dataset, article_mapping = (
        load_ebnerd_din_dataset(
            project_root=project_root,
            split="validation",
            max_history_length=args.max_history_length,
        )
    )

    print(
        f"       Full validation candidates: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------------
    # Select complete impressions
    # --------------------------------------------------------------

    print("[2/4] Selecting complete impressions...")

    candidates = dataset.candidates

    impression_sizes = (
        candidates
        .groupby("impression_id", sort=False)
        .size()
    )

    selected_indices: list[int] = []

    selected_rows = 0

    for impression_id, size in impression_sizes.items():

        if (
            selected_rows > 0
            and selected_rows + int(size)
            > args.rows
        ):
            break

        indices = candidates.index[
            candidates["impression_id"]
            == impression_id
        ].tolist()

        selected_indices.extend(indices)

        selected_rows += int(size)

    if not selected_indices:
        raise RuntimeError(
            "No validation impressions selected."
        )

    print(
        f"       Selected candidate rows: "
        f"{len(selected_indices):,}"
    )

    print(
        f"       Selected impressions: "
        f"{candidates.loc[selected_indices, 'impression_id'].nunique():,}"
    )

    # --------------------------------------------------------------
    # Build model
    # --------------------------------------------------------------

    print("[3/4] Loading DIN checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = DINModel(
        num_articles=checkpoint["num_articles"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=0.0,
        padding_idx=0,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    indexed_dataset = IndexedDataset(
        dataset=dataset,
        indices=selected_indices,
    )

    # --------------------------------------------------------------
    # Prediction + evaluation
    # --------------------------------------------------------------

    print("[4/4] Generating predictions...")

    predictions = generate_predictions(
        model=model,
        dataset=indexed_dataset,
        batch_size=args.batch_size,
        device=torch.device("cpu"),
    )

    metrics = evaluate_impressions(
        predictions
    )

    print()
    print("=" * 70)
    print("DIN VALIDATION RESULTS")
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value:,}")

    # --------------------------------------------------------------
    # Save predictions and metrics
    # --------------------------------------------------------------

    output_dir = (
        project_root
        / "data"
        / "predictions"
        / "a2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "validation_din_smoke.parquet"
    )

    metrics_path = (
        output_dir
        / "validation_din_smoke_metrics.json"
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=True,
        )

    print()
    print(
        f"Predictions saved: {predictions_path}"
    )

    print(
        f"Metrics saved: {metrics_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()