from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.retrieval.ebnerd_hybrid import (
    build_resources,
    load_articles,
    load_interactions,
    build_query_from_history,
    build_query_embedding,
    score_candidates_restricted,
    score_candidates_vectorized,
    build_behavioral_scores,
    combine_scores,
    rank_candidates,
    evaluate_ranking,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)

def generate_weight_grid(
    step=0.1,
):
    """
    Generate all non-negative three-way weight
    combinations satisfying:

        w_L + w_S + w_B = 1
    """

    values = np.arange(
        0.0,
        1.0 + step / 2,
        step,
    )

    configs = []

    for lexical_weight in values:

        for semantic_weight in values:

            behavioral_weight = (
                1.0
                - lexical_weight
                - semantic_weight
            )

            if behavioral_weight < -1e-9:
                continue

            if behavioral_weight > 1.0 + 1e-9:
                continue

            behavioral_weight = round(
                behavioral_weight,
                10,
            )

            configs.append(
                {
                    "lexical_weight":
                        round(
                            float(
                                lexical_weight
                            ),
                            10,
                        ),

                    "semantic_weight":
                        round(
                            float(
                                semantic_weight
                            ),
                            10,
                        ),

                    "behavioral_weight":
                        behavioral_weight,
                }
            )

    return configs

def evaluate_weight_grid(
    split,
    resources,
    configs,
):
    """
    Evaluate every weight configuration using the
    same L/S/B component scores.
    """

    interactions = load_interactions(
        split
    )

    grouped = interactions.groupby(
        "impression_id",
        sort=False,
    )

    accumulators = {}

    for index, config in enumerate(
        configs
    ):

        accumulators[index] = {
            "hit_at_5": 0,
            "hit_at_10": 0,
            "mrr": 0.0,
        }

    evaluated = 0
    total_candidates = 0

    start = time.perf_counter()

    for (
        impression_id,
        group,
    ) in grouped:

        user_id = group.iloc[0][
            "user_id"
        ]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidate_ids = (
            group["article_id"]
            .tolist()
        )

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        # ================================================================
        # LEXICAL
        # ================================================================

        bm25_history = (
            resources[
                "bm25_history_lookup"
            ].get(
                user_id,
                [],
            )
        )

        query_tokens = (
            build_query_from_history(
                bm25_history,
                impression_time,
            )
        )

        lexical_ranked = (
            score_candidates_restricted(
                query_tokens,
                candidate_ids,
                resources[
                    "bm25_article_to_index"
                ],
                resources[
                    "bm25_statistics"
                ],
            )
        )

        lexical_map = dict(
            lexical_ranked
        )

        lexical_scores = np.asarray(
            [
                lexical_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ================================================================
        # SEMANTIC
        # ================================================================

        semantic_history = (
            resources[
                "semantic_history_lookup"
            ].get(
                user_id,
                [],
            )
        )

        query_embedding = (
            build_query_embedding(
                semantic_history,
                impression_time,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_ranked = (
            score_candidates_vectorized(
                query_embedding,
                candidate_ids,
                resources[
                    "semantic_article_to_index"
                ],
                resources[
                    "semantic_embeddings"
                ],
            )
        )

        semantic_map = dict(
            semantic_ranked
        )

        semantic_scores = np.asarray(
            [
                semantic_map.get(
                    article_id,
                    0.0,
                )
                for article_id
                in candidate_ids
            ],
            dtype=np.float64,
        )

        # ================================================================
        # BEHAVIORAL
        # ================================================================

        behavioral_scores = (
            build_behavioral_scores(
                user_id,
                impression_time,
                candidate_ids,
                resources[
                    "article_categories"
                ],
                resources[
                    "temporal_lookup"
                ],
            )
        )

        # ================================================================
        # ALL WEIGHT CONFIGURATIONS
        # ================================================================

        for index, config in enumerate(
            configs
        ):

            hybrid_scores = (
                combine_scores(
                    lexical_scores,
                    semantic_scores,
                    behavioral_scores,
                    config[
                        "lexical_weight"
                    ],
                    config[
                        "semantic_weight"
                    ],
                    config[
                        "behavioral_weight"
                    ],
                )
            )

            ranked = rank_candidates(
                candidate_ids,
                hybrid_scores,
            )

            ranked_ids = [
                article_id
                for article_id, _score
                in ranked
            ]

            (
                hit5,
                hit10,
                rr,
            ) = evaluate_ranking(
                ranked_ids,
                clicked,
            )

            accumulators[index][
                "hit_at_5"
            ] += hit5

            accumulators[index][
                "hit_at_10"
            ] += hit10

            accumulators[index][
                "mrr"
            ] += rr

        evaluated += 1
        total_candidates += len(
            candidate_ids
        )

        if evaluated % 10_000 == 0:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{split}: "
                f"{evaluated:,} impressions | "
                f"{elapsed:.1f}s"
            )

    runtime = (
        time.perf_counter()
        - start
    )

    results = []

    for index, config in enumerate(
        configs
    ):

        results.append(
            {
                "split": split,

                "lexical_weight":
                    config[
                        "lexical_weight"
                    ],

                "semantic_weight":
                    config[
                        "semantic_weight"
                    ],

                "behavioral_weight":
                    config[
                        "behavioral_weight"
                    ],

                "impressions":
                    evaluated,

                "hit_at_5":
                    accumulators[index][
                        "hit_at_5"
                    ] / evaluated,

                "hit_at_10":
                    accumulators[index][
                        "hit_at_10"
                    ] / evaluated,

                "mrr":
                    accumulators[index][
                        "mrr"
                    ] / evaluated,

                "mean_candidates":
                    total_candidates
                    / evaluated,

                "runtime_seconds":
                    runtime,
            }
        )

    return pd.DataFrame(
        results
    )

def main():

    print(
        "=" * 80
    )

    print(
        "EB-NeRD SMALL — HYBRID WEIGHT ANALYSIS"
    )

    print(
        "=" * 80
    )

    articles = load_articles()

    configs = generate_weight_grid(
        step=0.1
    )

    print(
        f"\nWeight configurations: "
        f"{len(configs)}"
    )

    # ================================================================
    # TRAIN
    # ================================================================

    print(
        "\nBuilding train resources..."
    )

    train_resources = build_resources(
        "train",
        articles,
    )

    print(
        "\nTRAIN WEIGHT EVALUATION"
    )

    train_results = (
        evaluate_weight_grid(
            "train",
            train_resources,
            configs,
        )
    )

    # ================================================================
    # VALIDATION
    # ================================================================

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
        "\nVALIDATION WEIGHT EVALUATION"
    )

    validation_results = (
        evaluate_weight_grid(
            "validation",
            validation_resources,
            configs,
        )
    )

    # ================================================================
    # COMBINE
    # ================================================================

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
        / "hybrid_weight_tuning.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )

    # ================================================================
    # BEST VALIDATION CONFIGURATION
    # ================================================================

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
        "BEST VALIDATION WEIGHTS"
    )

    print(
        "=" * 80
    )

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