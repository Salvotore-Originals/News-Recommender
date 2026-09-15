from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.a2_din import DINModel
from src.ranking.a2_din_data import load_ebnerd_din_dataset
from src.ranking.train_phase5_reranker import (
    make_impression_subset,
)
from src.ranking.phase5_reranker_dataset import load_phase5_dataset


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "a2"
    / "din_controlled_100k_best.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "a2"
    / "validation_din_fixed_phase5_subset.parquet"
)

SEED = 42
VALIDATION_SEED = SEED + 1
VALIDATION_IMPRESSIONS = 5_000
BATCH_SIZE = 256


def collate_fn(batch):
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
        "candidate_index": torch.tensor(
            [item["candidate_index"] for item in batch],
            dtype=torch.long,
        ),
    }


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        item = self.dataset[original_index]

        return {
            **item,
            "candidate_index": original_index,
        }


def main():
    print("=" * 70)
    print("DIN BASELINE — FIXED PHASE 5.2 VALIDATION SUBSET")
    print("=" * 70)

    print(f"Checkpoint:              {CHECKPOINT_PATH}")
    print(f"Validation impressions: {VALIDATION_IMPRESSIONS:,}")
    print(f"Seed:                    {SEED}")
    print(f"Validation seed:         {VALIDATION_SEED}")
    print(f"Batch size:              {BATCH_SIZE}")
    print()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"DIN checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------------
    # 1. Reproduce the exact Phase 5.2 validation impression subset
    # --------------------------------------------------------------

    print("[1/5] Reproducing fixed Phase 5.2 validation subset...")

    phase5_dataset, _ = load_phase5_dataset(
        project_root=PROJECT_ROOT,
        split="validation",
        max_history_length=20,
    )

    phase5_subset = make_impression_subset(
        phase5_dataset,
        max_impressions=VALIDATION_IMPRESSIONS,
        seed=VALIDATION_SEED,
    )

    selected_phase5_indices = phase5_subset.indices

    phase5_candidates = phase5_dataset.candidates

    selected_impression_ids = set(
        phase5_candidates.iloc[
            selected_phase5_indices
        ]["impression_id"].astype(int)
    )

    print(
        f"       Selected impressions: "
        f"{len(selected_impression_ids):,}"
    )

    print(
        f"       Selected candidates: "
        f"{len(selected_phase5_indices):,}"
    )

    if len(selected_impression_ids) != VALIDATION_IMPRESSIONS:
        raise RuntimeError(
            "Incorrect number of selected validation impressions."
        )

    # --------------------------------------------------------------
    # 2. Load DIN validation dataset
    # --------------------------------------------------------------

    print("\n[2/5] Loading DIN validation dataset...")

    din_dataset, article_mapping = load_ebnerd_din_dataset(
        project_root=PROJECT_ROOT,
        split="validation",
        max_history_length=20,
    )

    print(
        f"       Full validation candidates: "
        f"{len(din_dataset):,}"
    )

    # --------------------------------------------------------------
    # 3. Select EXACT same impressions
    # --------------------------------------------------------------

    print("\n[3/5] Selecting exact same impressions...")

    din_candidates = din_dataset.candidates

    din_mask = din_candidates["impression_id"].isin(
        selected_impression_ids
    )

    din_indices = din_candidates.index[din_mask].tolist()

    din_selected = din_candidates.loc[din_indices]

    print(
        f"       DIN candidates: "
        f"{len(din_indices):,}"
    )

    print(
        f"       DIN impressions: "
        f"{din_selected['impression_id'].nunique():,}"
    )

    print(
        f"       DIN clicks: "
        f"{din_selected['clicked'].sum():,}"
    )

    if len(din_indices) != len(selected_phase5_indices):
        raise RuntimeError(
            "DIN and Phase 5.2 candidate counts differ."
        )

    if (
        din_selected["impression_id"].nunique()
        != VALIDATION_IMPRESSIONS
    ):
        raise RuntimeError(
            "DIN subset does not contain exactly 5,000 impressions."
        )

    # --------------------------------------------------------------
    # 4. Load checkpoint and generate predictions
    # --------------------------------------------------------------

    print("\n[4/5] Loading DIN checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print(
        f"       Best epoch: "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"       Best validation BCE: "
        f"{checkpoint.get('best_validation_loss'):.6f}"
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
        din_dataset,
        din_indices,
    )

    loader = DataLoader(
        indexed_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    prediction_rows = []

    with torch.no_grad():
        for batch in loader:

            histories = batch[
                "history_article_ids"
            ]

            candidates = batch[
                "candidate_article_id"
            ]

            logits = model(
                history_article_ids=histories,
                candidate_article_ids=candidates,
            )

            scores = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            labels = (
                batch["clicked"]
                .cpu()
                .numpy()
            )

            candidate_indices = batch[
                "candidate_index"
            ].numpy()

            for candidate_index, score, label in zip(
                candidate_indices,
                scores,
                labels,
            ):
                row = din_dataset.candidates.iloc[
                    int(candidate_index)
                ]

                prediction_rows.append(
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

    predictions = pd.DataFrame(
        prediction_rows
    )

    print(
        f"       Predictions generated: "
        f"{len(predictions):,}"
    )

    # --------------------------------------------------------------
    # 5. Validate and save
    # --------------------------------------------------------------

    print("\n[5/5] Validating predictions...")

    if len(predictions) != len(selected_phase5_indices):
        raise RuntimeError(
            "Prediction row count does not match selected candidates."
        )

    if (
        predictions["impression_id"].nunique()
        != VALIDATION_IMPRESSIONS
    ):
        raise RuntimeError(
            "Prediction impression count is incorrect."
        )

    if predictions[
        ["impression_id", "article_id"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate impression/article pairs detected."
        )

    if not np.isfinite(
        predictions["score"].to_numpy()
    ).all():
        raise RuntimeError(
            "NaN or infinite prediction scores detected."
        )

    if predictions["clicked"].sum() != 5021:
        raise RuntimeError(
            "Click count does not match the fixed Phase 5.2 subset."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"       Rows:        {len(predictions):,}"
    )

    print(
        f"       Impressions: "
        f"{predictions['impression_id'].nunique():,}"
    )

    print(
        f"       Clicks:      "
        f"{predictions['clicked'].sum():,}"
    )

    print(
        f"       Score min:   "
        f"{predictions['score'].min():.6f}"
    )

    print(
        f"       Score max:   "
        f"{predictions['score'].max():.6f}"
    )

    print(
        f"       Saved:       {OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("DIN FIXED-SUBSET PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()