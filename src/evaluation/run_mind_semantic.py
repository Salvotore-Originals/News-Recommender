from __future__ import annotations

import time

import pandas as pd

from src.evaluation.mind_semantic import (
    build_history_lookup,
    evaluate_semantic_retrieval,
    load_semantic_feature_store,
)


ARTICLES_EMBEDDINGS = (
    "data/features/mind/semantic/article_embeddings.npy"
)

ARTICLE_IDS = (
    "data/features/mind/semantic/article_ids.npy"
)

HISTORY_PATH = (
    "data/processed/mind/user_history.parquet"
)

VALIDATION_PATH = (
    "data/processed/mind/splits/validation.parquet"
)


def main() -> None:

    print("=" * 70)
    print("MIND SEMANTIC RETRIEVAL — FULL VALIDATION")
    print("=" * 70)

    # ==================================================
    # 1. Load semantic feature store
    # ==================================================

    print(
        "\n[1/5] Loading semantic feature store..."
    )

    embeddings, article_ids = (
        load_semantic_feature_store(
            ARTICLES_EMBEDDINGS,
            ARTICLE_IDS,
        )
    )

    print(
        f"       Embeddings: {embeddings.shape}"
    )

    print(
        f"       Articles: {len(article_ids):,}"
    )

    # ==================================================
    # 2. Load user history
    # ==================================================

    print(
        "\n[2/5] Loading user history..."
    )

    history = pd.read_parquet(
        HISTORY_PATH
    )

    print(
        f"       History rows: {len(history):,}"
    )

    history_lookup = build_history_lookup(
        history
    )

    print(
        f"       Impression histories: "
        f"{len(history_lookup):,}"
    )

    # ==================================================
    # 3. Load validation split
    # ==================================================

    print(
        "\n[3/5] Loading validation split..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    total_impressions = (
        validation["impression_id"]
        .nunique()
    )

    print(
        f"       Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"       Validation impressions: "
        f"{total_impressions:,}"
    )

    # ==================================================
    # 4. Prepare FULL validation set
    # ==================================================

    print(
        "\n[4/5] Preparing full validation set..."
    )

    sample = validation

    print(
        f"       Validation rows: "
        f"{len(sample):,}"
    )

    print(
        f"       Validation impressions: "
        f"{sample['impression_id'].nunique():,}"
    )

    # ==================================================
    # 5. Evaluate semantic retrieval
    # ==================================================

    print(
        "\n[5/5] Evaluating semantic retrieval..."
    )

    print(
        "       This may take several minutes..."
    )

    start = time.perf_counter()

    metrics = evaluate_semantic_retrieval(
        candidates=sample,
        history_lookup=history_lookup,
        article_embeddings=embeddings,
        article_ids=article_ids,
        max_history=10,
        ks=(5, 10),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    # ==================================================
    # Results
    # ==================================================

    print("\n" + "=" * 70)
    print("SEMANTIC FULL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Impressions evaluated : "
        f"{int(metrics['impressions']):,}"
    )

    print(
        f"Hit@5                 : "
        f"{metrics['Hit@5']:.6f}"
    )

    print(
        f"Hit@10                : "
        f"{metrics['Hit@10']:.6f}"
    )

    print(
        f"MRR                   : "
        f"{metrics['MRR']:.6f}"
    )

    print(
        f"Runtime               : "
        f"{elapsed:.2f} seconds"
    )

    # ==================================================
    # Coverage / Skip Rate
    # ==================================================

    total = int(
        metrics["total_impressions"]
    )

    evaluated = int(
        metrics["impressions"]
    )

    skipped = int(
        metrics["skipped_impressions"]
    )

    print("\n" + "=" * 70)
    print("SEMANTIC VALIDATION COVERAGE")
    print("=" * 70)

    print(
        f"Total impressions          : "
        f"{total:,}"
    )

    print(
        f"Evaluated impressions      : "
        f"{evaluated:,}"
    )

    print(
        f"Skipped impressions        : "
        f"{skipped:,}"
    )

    print(
        f"Coverage                   : "
        f"{metrics['coverage']:.4%}"
    )

    print(
        f"Skip rate                  : "
        f"{metrics['skip_rate']:.4%}"
    )

    # ==================================================
    # Skip reasons
    # ==================================================

    print("\nSkip reasons:")

    print(
        f"  no_history                : "
        f"{int(metrics['skip_no_history']):,}"
    )

    print(
        f"  unknown_history_articles  : "
        f"{int(metrics['skip_unknown_history_articles']):,}"
    )

    print(
        f"  zero_user_embedding       : "
        f"{int(metrics['skip_zero_user_embedding']):,}"
    )

    print(
        f"  no_valid_candidates       : "
        f"{int(metrics['skip_no_valid_candidates']):,}"
    )

    print("=" * 70)

    # ==================================================
    # Consistency check
    # ==================================================

    skip_reason_total = (
        int(metrics["skip_no_history"])
        + int(
            metrics[
                "skip_unknown_history_articles"
            ]
        )
        + int(
            metrics[
                "skip_zero_user_embedding"
            ]
        )
        + int(
            metrics[
                "skip_no_valid_candidates"
            ]
        )
    )

    print("\n" + "=" * 70)
    print("SKIP COUNT CONSISTENCY CHECK")
    print("=" * 70)

    print(
        f"Skipped impressions        : "
        f"{skipped:,}"
    )

    print(
        f"Sum of skip reasons       : "
        f"{skip_reason_total:,}"
    )

    if skip_reason_total == skipped:

        print(
            "Status                    : PASS"
        )

    else:

        print(
            "Status                    : FAIL"
        )

        raise RuntimeError(
            "Skip reason counts do not match "
            "the total skipped impressions."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()