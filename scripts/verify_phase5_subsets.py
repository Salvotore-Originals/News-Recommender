from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.phase5_reranker_dataset import load_phase5_dataset
from src.ranking.train_phase5_reranker import make_impression_subset


def verify_split(
    split: str,
    max_impressions: int,
    seed: int,
) -> None:
    print()
    print("=" * 72)
    print(f"{split.upper()} SUBSET VERIFICATION")
    print("=" * 72)

    dataset, _ = load_phase5_dataset(
        PROJECT_ROOT,
        split=split,
        max_history_length=20,
    )

    print(f"Full dataset rows:        {len(dataset):,}")

    full_candidates = dataset.candidates

    full_impressions = (
        full_candidates["impression_id"]
        .drop_duplicates()
        .to_numpy()
    )

    print(
        f"Full dataset impressions: {len(full_impressions):,}"
    )

    subset = make_impression_subset(
        dataset,
        max_impressions=max_impressions,
        seed=seed,
    )

    # Recover the selected row indices from the Subset.
    row_indices = np.asarray(
        subset.indices,
        dtype=np.int64,
    )

    selected = full_candidates.iloc[row_indices].copy()

    selected_impressions = (
        selected["impression_id"]
        .drop_duplicates()
        .to_numpy()
    )

    print(
        f"Requested impressions:    {max_impressions:,}"
    )
    print(
        f"Selected impressions:     {len(selected_impressions):,}"
    )
    print(
        f"Selected candidate rows:  {len(selected):,}"
    )

    # ------------------------------------------------------------------
    # 1. Exact impression count
    # ------------------------------------------------------------------
    assert len(selected_impressions) == max_impressions, (
        f"Expected {max_impressions:,} impressions, "
        f"got {len(selected_impressions):,}"
    )

    # ------------------------------------------------------------------
    # 2. Every selected impression must be complete.
    #
    # Compare the number of selected rows per impression against
    # the number of rows for that same impression in the full dataset.
    # ------------------------------------------------------------------
    full_counts = (
        full_candidates
        .groupby("impression_id")
        .size()
    )

    selected_counts = (
        selected
        .groupby("impression_id")
        .size()
    )

    expected_counts = full_counts.loc[
        selected_counts.index
    ]

    incomplete = selected_counts[
        selected_counts != expected_counts
    ]

    assert incomplete.empty, (
        "Some impressions were only partially selected:\n"
        f"{incomplete.head(10)}"
    )

    print("Complete impressions:     PASS")

    # ------------------------------------------------------------------
    # 3. No duplicate candidate rows.
    # ------------------------------------------------------------------
    duplicate_count = selected.duplicated(
        subset=["impression_id", "article_id"]
    ).sum()

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate "
        "impression/article pairs."
    )

    print("Duplicate candidates:     PASS")

    # ------------------------------------------------------------------
    # 4. Click statistics
    # ------------------------------------------------------------------
    clicks = int(selected["clicked"].sum())
    negatives = int(len(selected) - clicks)

    print(f"Clicks:                    {clicks:,}")
    print(f"Non-clicks:                {negatives:,}")
    print(
        f"Positive rate:             "
        f"{100.0 * clicks / len(selected):.4f}%"
    )

    candidates_per_impression = (
        selected.groupby("impression_id")
        .size()
    )

    print(
        f"Candidates / impression:   "
        f"mean={candidates_per_impression.mean():.2f}, "
        f"median={candidates_per_impression.median():.0f}, "
        f"min={candidates_per_impression.min()}, "
        f"max={candidates_per_impression.max()}"
    )

    # ------------------------------------------------------------------
    # 5. Every selected impression should have >=1 click.
    # ------------------------------------------------------------------
    clicks_per_impression = (
        selected.groupby("impression_id")["clicked"]
        .sum()
    )

    zero_click_impressions = (
        clicks_per_impression == 0
    ).sum()

    assert zero_click_impressions == 0, (
        f"Found {zero_click_impressions} "
        "zero-click impressions."
    )

    print("Every impression has click: PASS")

    print()
    print(f"{split.upper()} VERIFICATION: PASS")


def main() -> None:
    verify_split(
        split="train",
        max_impressions=10_000,
        seed=42,
    )

    verify_split(
        split="validation",
        max_impressions=5_000,
        seed=43,
    )

    print()
    print("=" * 72)
    print("ALL PHASE 5 EXPERIMENT SUBSET CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()