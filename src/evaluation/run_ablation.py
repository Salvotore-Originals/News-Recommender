from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.evaluation.ablation import evaluate_ablation


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "validation_hybrid_candidates.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "validation_ablation_results.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("MIND HYBRID ABLATION — VALIDATION")
    print("=" * 70)

    start = time.perf_counter()

    # --------------------------------------------------------
    # Load candidate scores
    # --------------------------------------------------------

    print(
        "\n[1/2] Loading candidate-level scores..."
    )

    candidate_scores = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"       Candidate rows: "
        f"{len(candidate_scores):,}"
    )

    print(
        f"       Impressions: "
        f"{candidate_scores['impression_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Run ablation
    # --------------------------------------------------------

    print(
        "\n[2/2] Running seven ablation configurations..."
    )

    evaluation_start = time.perf_counter()

    results = evaluate_ablation(
        candidate_scores
    )

    evaluation_elapsed = (
        time.perf_counter()
        - evaluation_start
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - start
    )

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)

    print(
        f"Evaluation runtime: "
        f"{evaluation_elapsed:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()