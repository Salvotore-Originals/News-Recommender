import numpy as np
import pandas as pd
import pytest

from src.evaluation.temporal_analysis import (
    calculate_stability,
    calculate_temporal_variability,
    compare_key_models,
    find_period_winners,
    validate_temporal_results,
)


MODELS = [
    "BM25",
    "Semantic",
    "Behavioural",
    "BM25 + Semantic",
    "BM25 + Behavioural",
    "Semantic + Behavioural",
    "Full Hybrid",
]


def make_results() -> pd.DataFrame:

    rows = []

    for period_index, period in enumerate(
        ["early", "middle", "late"]
    ):

        for model_index, model in enumerate(
            MODELS
        ):

            base = (
                0.20
                + 0.01 * model_index
                - 0.005 * period_index
            )

            rows.append(
                {
                    "temporal_period": period,
                    "model": model,
                    "MRR": base,
                    "Hit@5": min(base + 0.05, 1.0),
                    "Hit@10": min(base + 0.10, 1.0),
                    "NDCG@5": min(base + 0.03, 1.0),
                    "NDCG@10": min(base + 0.08, 1.0),
                }
            )

    return pd.DataFrame(rows)


def test_temporal_results_validate():

    df = make_results()

    validate_temporal_results(
        df
    )


def test_invalid_row_count_raises():

    df = make_results().iloc[:-1]

    with pytest.raises(
        ValueError,
        match="Unexpected number",
    ):
        validate_temporal_results(
            df
        )


def test_invalid_metric_values_raise():

    df = make_results()

    df.loc[
        0,
        "MRR",
    ] = 2.0

    with pytest.raises(
        ValueError,
        match=r"outside \[0, 1\]",
    ):
        validate_temporal_results(
            df
        )


def test_period_winners():

    df = make_results()

    winners = find_period_winners(
        df
    )

    assert len(winners) == 15

    assert set(
        winners["temporal_period"]
    ) == {
        "early",
        "middle",
        "late",
    }


def test_stability_calculation():

    df = make_results()

    stability = calculate_stability(
        df
    )

    assert len(stability) == 7

    assert "MRR_change" in stability.columns

    assert "MRR_relative_change_pct" in stability.columns


def test_temporal_variability():

    df = make_results()

    variability = calculate_temporal_variability(
        df
    )

    assert len(variability) == 7

    assert (
        variability["MRR_std"]
        >= 0
    ).all()

    assert (
        variability["MRR_range"]
        >= 0
    ).all()


def test_key_model_comparison():

    df = make_results()

    comparison = compare_key_models(
        df
    )

    assert len(comparison) == 15

    assert set(
        comparison["metric"]
    ) == {
        "MRR",
        "Hit@5",
        "Hit@10",
        "NDCG@5",
        "NDCG@10",
    }


def test_missing_required_model_raises():

    df = make_results()

    df = df[
        df["model"] != "Full Hybrid"
    ]

    with pytest.raises(
        ValueError,
        match="Required comparison models",
    ):
        compare_key_models(
            df
        )