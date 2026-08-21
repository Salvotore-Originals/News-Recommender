from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.hybrid import evaluate_score
from src.ranking.hybrid import build_hybrid_scores


# ============================================================
# ABLATION CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AblationConfig:
    """
    Configuration for one hybrid-ranking ablation experiment.

    The three weights correspond to:

        lexical_weight
        semantic_weight
        behavioral_weight
    """

    name: str
    lexical_weight: float
    semantic_weight: float
    behavioral_weight: float


ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="BM25",
        lexical_weight=1.0,
        semantic_weight=0.0,
        behavioral_weight=0.0,
    ),
    AblationConfig(
        name="Semantic",
        lexical_weight=0.0,
        semantic_weight=1.0,
        behavioral_weight=0.0,
    ),
    AblationConfig(
        name="Behavioural",
        lexical_weight=0.0,
        semantic_weight=0.0,
        behavioral_weight=1.0,
    ),
    AblationConfig(
        name="BM25 + Semantic",
        lexical_weight=0.5,
        semantic_weight=0.5,
        behavioral_weight=0.0,
    ),
    AblationConfig(
        name="BM25 + Behavioural",
        lexical_weight=0.5,
        semantic_weight=0.0,
        behavioral_weight=0.5,
    ),
    AblationConfig(
        name="Semantic + Behavioural",
        lexical_weight=0.0,
        semantic_weight=0.5,
        behavioral_weight=0.5,
    ),
    AblationConfig(
        name="Full Hybrid",
        lexical_weight=0.40,
        semantic_weight=0.30,
        behavioral_weight=0.30,
    ),
)


# ============================================================
# VALIDATION
# ============================================================

def validate_ablation_configs() -> None:
    """
    Validate all predefined ablation configurations.
    """

    if not ABLATION_CONFIGS:
        raise ValueError(
            "No ablation configurations defined."
        )

    names = [
        config.name
        for config in ABLATION_CONFIGS
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "Ablation configuration names must be unique."
        )

    for config in ABLATION_CONFIGS:

        weights = (
            config.lexical_weight,
            config.semantic_weight,
            config.behavioral_weight,
        )

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                f"Negative weight in configuration "
                f"'{config.name}'."
            )

        total = sum(weights)

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Weights for '{config.name}' "
                f"must sum to 1. "
                f"Got {total}."
            )


# ============================================================
# SINGLE ABLATION EXPERIMENT
# ============================================================

def evaluate_ablation_config(
    candidate_scores: pd.DataFrame,
    config: AblationConfig,
) -> dict[str, float | str]:
    """
    Evaluate one ablation configuration.

    The candidate set remains unchanged. Only the fusion
    weights are changed.
    """

    required_columns = {
        "impression_id",
        "clicked",
        "bm25_score",
        "semantic_score",
        "behavioral_score",
    }

    missing = (
        required_columns
        - set(candidate_scores.columns)
    )

    if missing:
        raise ValueError(
            "Candidate scores are missing required "
            f"columns: {sorted(missing)}"
        )

    scored = build_hybrid_scores(
        candidate_scores,
        lexical_weight=config.lexical_weight,
        semantic_weight=config.semantic_weight,
        behavioral_weight=config.behavioral_weight,
    )

    metrics = evaluate_score(
        scored,
        "hybrid_score",
    )

    return {
        "model": config.name,
        "lexical_weight": config.lexical_weight,
        "semantic_weight": config.semantic_weight,
        "behavioral_weight": config.behavioral_weight,
        **metrics,
    }


# ============================================================
# FULL ABLATION STUDY
# ============================================================

def evaluate_ablation(
    candidate_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete seven-configuration ablation study.

    Every configuration is evaluated on exactly the same
    candidate-level impression table.
    """

    validate_ablation_configs()

    if candidate_scores.empty:
        raise ValueError(
            "candidate_scores is empty."
        )

    print("=" * 70)
    print("MIND HYBRID ABLATION STUDY")
    print("=" * 70)

    print(
        f"\nCandidate rows: "
        f"{len(candidate_scores):,}"
    )

    print(
        f"Impressions: "
        f"{candidate_scores['impression_id'].nunique():,}"
    )

    results = []

    for config in ABLATION_CONFIGS:

        print(
            f"\nEvaluating {config.name}..."
        )

        result = evaluate_ablation_config(
            candidate_scores,
            config,
        )

        results.append(result)

        print(
            f"       Hit@5:   "
            f"{result['Hit@5']:.6f}"
        )

        print(
            f"       Hit@10:  "
            f"{result['Hit@10']:.6f}"
        )

        print(
            f"       MRR:     "
            f"{result['MRR']:.6f}"
        )

        print(
            f"       NDCG@5:  "
            f"{result['NDCG@5']:.6f}"
        )

        print(
            f"       NDCG@10: "
            f"{result['NDCG@10']:.6f}"
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df