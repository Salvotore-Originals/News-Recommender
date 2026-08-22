from pathlib import Path

import pandas as pd

from src.retrieval.ebnerd_hybrid import (
    build_resources,
    load_articles,
)

from src.retrieval.ebnerd_weight_tuning import (
    evaluate_weight_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)
def generate_fine_weight_grid():
    """
    Fine search around the coarse optimum.

    Stage A:
        L = 0
        S = 0.00 ... 0.20
        B = 1 - S

    Stage B:
        Small lexical contributions around
        the best semantic/behavioral region.
    """

    configs = []

    # ------------------------------------------------------------------
    # Stage A: Semantic + Behavioral
    # ------------------------------------------------------------------

    for semantic_weight in [
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
    ]:

        configs.append(
            {
                "lexical_weight": 0.0,
                "semantic_weight": semantic_weight,
                "behavioral_weight":
                    round(
                        1.0 - semantic_weight,
                        10,
                    ),
            }
        )

    # ------------------------------------------------------------------
    # Stage B: Small lexical contribution
    # ------------------------------------------------------------------

    configs.extend(
        [
            {
                "lexical_weight": 0.02,
                "semantic_weight": 0.08,
                "behavioral_weight": 0.90,
            },
            {
                "lexical_weight": 0.02,
                "semantic_weight": 0.10,
                "behavioral_weight": 0.88,
            },
            {
                "lexical_weight": 0.02,
                "semantic_weight": 0.12,
                "behavioral_weight": 0.86,
            },
            {
                "lexical_weight": 0.05,
                "semantic_weight": 0.05,
                "behavioral_weight": 0.90,
            },
            {
                "lexical_weight": 0.05,
                "semantic_weight": 0.10,
                "behavioral_weight": 0.85,
            },
            {
                "lexical_weight": 0.05,
                "semantic_weight": 0.15,
                "behavioral_weight": 0.80,
            },
        ]
    )

    return configs

def main():

    print("=" * 80)

    print(
        "EB-NeRD SMALL — FINE HYBRID WEIGHT SEARCH"
    )

    print("=" * 80)

    configs = (
        generate_fine_weight_grid()
    )

    print(
        f"\nFine configurations: "
        f"{len(configs)}"
    )

    articles = load_articles()

    # ==================================================================
    # TRAIN
    # ==================================================================

    print(
        "\nBuilding train resources..."
    )

    train_resources = build_resources(
        "train",
        articles,
    )

    print(
        "\nTRAIN FINE WEIGHT EVALUATION"
    )

    train_results = (
        evaluate_weight_grid(
            "train",
            train_resources,
            configs,
        )
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    print(
        "\nBuilding validation resources..."
    )

    validation_resources = (
        build_resources(
            "validation",
            articles,
        )
    )

    print(
        "\nVALIDATION FINE WEIGHT EVALUATION"
    )

    validation_results = (
        evaluate_weight_grid(
            "validation",
            validation_resources,
            configs,
        )
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    results = pd.concat(
        [
            train_results,
            validation_results,
        ],
        ignore_index=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULT_ROOT
        / "hybrid_fine_weight_tuning.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )

    # ==================================================================
    # BEST VALIDATION RESULT
    # ==================================================================

    best = (
        validation_results
        .sort_values(
            "mrr",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "BEST FINE VALIDATION WEIGHTS"
    )

    print("=" * 80)

    print(
        f"Lexical:    "
        f"{best['lexical_weight']:.2f}"
    )

    print(
        f"Semantic:   "
        f"{best['semantic_weight']:.2f}"
    )

    print(
        f"Behavioral: "
        f"{best['behavioral_weight']:.2f}"
    )

    print(
        f"Hit@5:      "
        f"{best['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:     "
        f"{best['hit_at_10']:.6f}"
    )

    print(
        f"MRR:        "
        f"{best['mrr']:.6f}"
    )


if __name__ == "__main__":
    main()
