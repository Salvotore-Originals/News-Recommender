from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.evaluation.ablation import (
    ABLATION_CONFIGS,
    evaluate_ablation_config,
)
from src.evaluation.temporal import (
    assign_temporal_periods,
)


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
    / "validation_temporal_results.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("MIND HYBRID TEMPORAL EVALUATION")
    print("=" * 70)

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Load candidate scores
    # --------------------------------------------------------

    print(
        "\n[1/3] Loading candidate-level scores..."
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
    # Assign temporal periods
    # --------------------------------------------------------

    print(
        "\n[2/3] Assigning chronological periods..."
    )

    temporal_scores = assign_temporal_periods(
        candidate_scores
    )

    summary = (
        temporal_scores
        .groupby("temporal_period")
        .agg(
            impressions=(
                "impression_id",
                "nunique",
            ),
            candidates=(
                "impression_id",
                "size",
            ),
            first_time=(
                "timestamp",
                "min",
            ),
            last_time=(
                "timestamp",
                "max",
            ),
        )
        .reindex(
            ["early", "middle", "late"]
        )
    )

    print(
        "\n"
        + summary.to_string()
    )

    # --------------------------------------------------------
    # Evaluate all configurations
    # --------------------------------------------------------

    print(
        "\n[3/3] Evaluating seven configurations "
        "across three temporal periods..."
    )

    results = []

    evaluation_start = time.perf_counter()

    for period in [
        "early",
        "middle",
        "late",
    ]:

        period_data = temporal_scores[
            temporal_scores[
                "temporal_period"
            ]
            == period
        ].copy()

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"PERIOD: {period.upper()}"
        )

        print(
            f"Impressions: "
            f"{period_data['impression_id'].nunique():,}"
        )

        print(
            f"Candidates: "
            f"{len(period_data):,}"
        )

        for config in ABLATION_CONFIGS:

            print(
                f"\nEvaluating "
                f"{config.name}..."
            )

            result = (
                evaluate_ablation_config(
                    period_data,
                    config,
                )
            )

            result["temporal_period"] = period

            results.append(
                result
            )

            print(
                f"       MRR:     "
                f"{result['MRR']:.6f}"
            )

            print(
                f"       Hit@5:   "
                f"{result['Hit@5']:.6f}"
            )

            print(
                f"       Hit@10:  "
                f"{result['Hit@10']:.6f}"
            )

            print(
                f"       NDCG@5:  "
                f"{result['NDCG@5']:.6f}"
            )

            print(
                f"       NDCG@10: "
                f"{result['NDCG@10']:.6f}"
            )

    evaluation_elapsed = (
        time.perf_counter()
        - evaluation_start
    )

    # --------------------------------------------------------
    # Build results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "temporal_period",
            "model",
            "lexical_weight",
            "semantic_weight",
            "behavioral_weight",
            "MRR",
            "Hit@5",
            "Hit@10",
            "NDCG@5",
            "NDCG@10",
        ]
    ]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
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
        - total_start
    )

    print("\n" + "=" * 70)
    print("TEMPORAL EVALUATION COMPLETE")
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