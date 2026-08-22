from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_RESULTS_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "validation_temporal_results.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "temporal_stability_analysis.parquet"
)


# ============================================================
# CONSTANTS
# ============================================================

PERIOD_ORDER = (
    "early",
    "middle",
    "late",
)

METRICS = (
    "MRR",
    "Hit@5",
    "Hit@10",
    "NDCG@5",
    "NDCG@10",
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_temporal_results(
    path: Path = TEMPORAL_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Load temporal evaluation results.

    Expected structure:

        temporal_period
        model
        lexical_weight
        semantic_weight
        behavioral_weight
        MRR
        Hit@5
        Hit@10
        NDCG@5
        NDCG@10
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Temporal results not found:\n{path}"
        )

    df = pd.read_parquet(path)

    required = {
        "temporal_period",
        "model",
        *METRICS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Temporal results are missing required "
            f"columns: {sorted(missing)}"
        )

    return df


# ============================================================
# VALIDATE TEMPORAL RESULTS
# ============================================================

def validate_temporal_results(
    df: pd.DataFrame,
) -> None:
    """
    Validate the expected 7 × 3 temporal experiment.
    """

    if df.empty:
        raise ValueError(
            "Temporal results are empty."
        )

    periods = set(
        df["temporal_period"]
    )

    expected_periods = set(
        PERIOD_ORDER
    )

    if periods != expected_periods:
        raise ValueError(
            "Unexpected temporal periods: "
            f"{sorted(periods)}"
        )

    models = set(
        df["model"]
    )

    if len(models) != 7:
        raise ValueError(
            "Expected exactly seven models, "
            f"found {len(models)}."
        )

    expected_rows = (
        len(expected_periods)
        * len(models)
    )

    if len(df) != expected_rows:
        raise ValueError(
            "Unexpected number of temporal result rows: "
            f"{len(df)}; expected {expected_rows}."
        )

    duplicated = df.duplicated(
        [
            "temporal_period",
            "model",
        ]
    )

    if duplicated.any():
        raise ValueError(
            "Duplicate temporal-period/model results found."
        )

    for metric in METRICS:

        values = df[metric].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{metric} contains non-finite values."
            )

        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(
                f"{metric} contains values outside [0, 1]."
            )


# ============================================================
# PERIOD-WISE WINNERS
# ============================================================

def find_period_winners(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the best model for every metric
    in every temporal period.
    """

    rows = []

    for period in PERIOD_ORDER:

        period_df = df[
            df["temporal_period"] == period
        ]

        for metric in METRICS:

            best_index = period_df[
                metric
            ].idxmax()

            best_row = period_df.loc[
                best_index
            ]

            rows.append(
                {
                    "temporal_period": period,
                    "metric": metric,
                    "best_model": best_row["model"],
                    "best_value": float(
                        best_row[metric]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# EARLY → LATE STABILITY
# ============================================================

def calculate_stability(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate temporal stability for every model.

    For each metric:

        absolute_change =
            late - early

        relative_change =
            (late - early) / early

    A negative value indicates degradation.
    """

    rows = []

    for model in sorted(
        df["model"].unique()
    ):

        model_df = (
            df[
                df["model"] == model
            ]
            .set_index(
                "temporal_period"
            )
        )

        row = {
            "model": model,
        }

        for metric in METRICS:

            early = float(
                model_df.loc[
                    "early",
                    metric,
                ]
            )

            middle = float(
                model_df.loc[
                    "middle",
                    metric,
                ]
            )

            late = float(
                model_df.loc[
                    "late",
                    metric,
                ]
            )

            absolute_change = (
                late - early
            )

            if early != 0:

                relative_change = (
                    (
                        late - early
                    )
                    /
                    early
                    * 100
                )

            else:

                relative_change = np.nan

            row[
                f"{metric}_early"
            ] = early

            row[
                f"{metric}_middle"
            ] = middle

            row[
                f"{metric}_late"
            ] = late

            row[
                f"{metric}_change"
            ] = absolute_change

            row[
                f"{metric}_relative_change_pct"
            ] = relative_change

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# TEMPORAL VARIABILITY
# ============================================================

def calculate_temporal_variability(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean, standard deviation and range
    across the three temporal periods.

    Lower standard deviation indicates greater
    temporal consistency.
    """

    rows = []

    for model in sorted(
        df["model"].unique()
    ):

        model_df = df[
            df["model"] == model
        ]

        row = {
            "model": model,
        }

        for metric in METRICS:

            values = model_df[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=0,
                )
            )

            row[
                f"{metric}_range"
            ] = float(
                np.max(values)
                -
                np.min(values)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SEMANTIC + BEHAVIOURAL VS FULL HYBRID
# ============================================================

def compare_key_models(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Semantic + Behavioural against Full Hybrid
    for every temporal period and metric.
    """

    required_models = {
        "Semantic + Behavioural",
        "Full Hybrid",
    }

    available = set(
        df["model"]
    )

    missing = (
        required_models
        - available
    )

    if missing:
        raise ValueError(
            "Required comparison models are missing: "
            f"{sorted(missing)}"
        )

    rows = []

    for period in PERIOD_ORDER:

        period_df = df[
            df["temporal_period"] == period
        ]

        s_b = period_df[
            period_df["model"]
            == "Semantic + Behavioural"
        ].iloc[0]

        full = period_df[
            period_df["model"]
            == "Full Hybrid"
        ].iloc[0]

        for metric in METRICS:

            sb_value = float(
                s_b[metric]
            )

            full_value = float(
                full[metric]
            )

            absolute_difference = (
                sb_value
                -
                full_value
            )

            if full_value != 0:

                relative_difference = (
                    absolute_difference
                    /
                    full_value
                    * 100
                )

            else:

                relative_difference = np.nan

            rows.append(
                {
                    "temporal_period": period,
                    "metric": metric,
                    "semantic_behavioral": sb_value,
                    "full_hybrid": full_value,
                    "difference": absolute_difference,
                    "relative_difference_pct": relative_difference,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_temporal_results(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Produce all temporal stability analyses.
    """

    validate_temporal_results(
        df
    )

    return {
        "winners": find_period_winners(
            df
        ),
        "stability": calculate_stability(
            df
        ),
        "variability": calculate_temporal_variability(
            df
        ),
        "key_comparison": compare_key_models(
            df
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("MIND TEMPORAL STABILITY ANALYSIS")
    print("=" * 70)

    df = load_temporal_results()

    print(
        f"\nTemporal result rows: "
        f"{len(df)}"
    )

    analysis = analyze_temporal_results(
        df
    )

    # --------------------------------------------------------
    # Winners
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST MODEL BY TEMPORAL PERIOD")
    print("=" * 70)

    print(
        analysis["winners"].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EARLY → LATE STABILITY")
    print("=" * 70)

    stability = analysis[
        "stability"
    ]

    stability_display = stability[
        [
            "model",
            "MRR_early",
            "MRR_middle",
            "MRR_late",
            "MRR_change",
            "MRR_relative_change_pct",
        ]
    ]

    print(
        stability_display.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Variability
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL VARIABILITY")
    print("=" * 70)

    variability = analysis[
        "variability"
    ]

    variability_display = variability[
        [
            "model",
            "MRR_mean",
            "MRR_std",
            "MRR_range",
        ]
    ].sort_values(
        "MRR_std"
    )

    print(
        variability_display.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Key comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "SEMANTIC + BEHAVIOURAL "
        "VS FULL HYBRID"
    )
    print("=" * 70)

    key_comparison = analysis[
        "key_comparison"
    ]

    print(
        key_comparison.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Save complete stability table
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stability.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TEMPORAL STABILITY ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()