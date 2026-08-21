from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ranking.candidates import (
    build_hybrid_candidate_scores,
)
from src.evaluation.hybrid import (
    evaluate_hybrid,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "mind"
    / "splits"
    / "validation.parquet"
)

HISTORY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "mind"
    / "user_history.parquet"
)

ARTICLES_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "articles.parquet"
)

EMBEDDINGS_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "semantic"
    / "article_embeddings.npy"
)

ARTICLE_IDS_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "semantic"
    / "article_ids.npy"
)

BEHAVIORAL_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "validation_behavioral_score.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "features"
    / "mind"
    / "validation_hybrid_candidates.parquet"
)


# ============================================================
# IMPORT BM25
# ============================================================

from src.retrieval.mind_bm25 import (
    build_mind_bm25,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("MIND HYBRID RANKING — FULL VALIDATION")
    print("=" * 70)

    total_start = time.perf_counter()

    # ========================================================
    # 1. Load validation candidates
    # ========================================================

    print(
        "\n[1/7] Loading validation candidates..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    print(
        f"       Candidate rows: "
        f"{len(validation):,}"
    )

    print(
        f"       Impressions: "
        f"{validation['impression_id'].nunique():,}"
    )

    # ========================================================
    # 2. Load history
    # ========================================================

    print(
        "\n[2/7] Loading MIND history..."
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "article_id",
            "history_position",
        ],
    )

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    # ========================================================
    # 3. Load article metadata
    # ========================================================

    print(
        "\n[3/7] Loading article feature store..."
    )

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    # ========================================================
    # 4. Build BM25 index
    # ========================================================

    print(
        "\n[4/7] Building BM25 index..."
    )

    bm25 = build_mind_bm25(
        str(ARTICLES_PATH)
    )

    print(
        f"       BM25 articles: "
        f"{len(bm25.article_ids):,}"
    )

    # ========================================================
    # 5. Load semantic feature store
    # ========================================================

    print(
        "\n[5/7] Loading semantic embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    article_ids = np.load(
        ARTICLE_IDS_PATH,
        allow_pickle=True,
    )

    print(
        f"       Embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"       Article IDs: "
        f"{len(article_ids):,}"
    )

    # ========================================================
    # 6. Load behavioural scores
    # ========================================================

    print(
        "\n[6/7] Loading behavioural scores..."
    )

    behavioral = pd.read_parquet(
        BEHAVIORAL_PATH
    )

    print(
        f"       Behavioural rows: "
        f"{len(behavioral):,}"
    )

    print(
        f"       Behavioural impressions: "
        f"{behavioral['impression_id'].nunique():,}"
    )

    # ========================================================
    # 7. Build candidate-level scores
    # ========================================================

    print(
        "\n[7/7] Building candidate-level "
        "lexical + semantic + behavioural scores..."
    )

    print(
        "       This may take several minutes."
    )

    start = time.perf_counter()

    candidate_scores = (
        build_hybrid_candidate_scores(
            candidates=validation,
            history=history,
            articles=articles,
            bm25_retriever=bm25,
            article_embeddings=embeddings,
            article_ids=article_ids,
            behavioral_scores=behavioral,
            max_history=10,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\n       Candidate scoring runtime: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"       Candidate rows generated: "
        f"{len(candidate_scores):,}"
    )

    # ========================================================
    # Save candidate scores
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_scores.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"       Saved: "
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # Evaluate hybrid ranking
    # ========================================================

    print(
        "\nEvaluating ranking models..."
    )

    evaluation_start = time.perf_counter()

    results = evaluate_hybrid(
        candidate_scores,
        lexical_weight=0.40,
        semantic_weight=0.30,
        behavioral_weight=0.30,
    )

    evaluation_elapsed = (
        time.perf_counter()
        - evaluation_start
    )

    # ========================================================
    # Display results
    # ========================================================

    print("\n" + "=" * 70)
    print("MIND HYBRID VALIDATION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # ========================================================
    # Best baseline comparison
    # ========================================================

    baseline = results[
        results["model"] != "Hybrid"
    ]

    hybrid = results[
        results["model"] == "Hybrid"
    ]

    if not hybrid.empty:

        hybrid_mrr = float(
            hybrid.iloc[0]["MRR"]
        )

        best_baseline_mrr = float(
            baseline["MRR"].max()
        )

        if best_baseline_mrr > 0:

            improvement = (
                (
                    hybrid_mrr
                    - best_baseline_mrr
                )
                /
                best_baseline_mrr
                * 100
            )

        else:

            improvement = float("nan")

        print("\n" + "=" * 70)
        print("HYBRID IMPROVEMENT")
        print("=" * 70)

        print(
            f"Hybrid MRR          : "
            f"{hybrid_mrr:.6f}"
        )

        print(
            f"Best baseline MRR   : "
            f"{best_baseline_mrr:.6f}"
        )

        print(
            f"MRR improvement     : "
            f"{improvement:.2f}%"
        )

    # ========================================================
    # Runtime
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 70)
    print("HYBRID VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Scoring runtime     : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Evaluation runtime  : "
        f"{evaluation_elapsed:.2f} seconds"
    )

    print(
        f"Total runtime       : "
        f"{total_elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()