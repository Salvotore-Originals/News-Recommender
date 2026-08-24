from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.mind_inference import (
    predict_mind_candidates,
)
from src.retrieval.mind_bm25 import (
    build_mind_bm25,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind"
    / "splits"
    / "test.parquet"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind"
    / "user_history.parquet"
)

ARTICLES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mind"
    / "articles.parquet"
)

BEHAVIORAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mind"
    / "test_behavioral_score.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mind"
    / "semantic"
    / "article_embeddings.npy"
)

ARTICLE_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "mind"
    / "semantic"
    / "article_ids.npy"
)


def check_required_files() -> None:
    """Verify all required inference artifacts exist."""

    paths = {
        "test candidates": TEST_PATH,
        "user history": HISTORY_PATH,
        "articles": ARTICLES_PATH,
        "behavioral scores": BEHAVIORAL_PATH,
        "article embeddings": EMBEDDINGS_PATH,
        "article IDs": ARTICLE_IDS_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required MIND smoke-test files are missing:\n"
            + "\n".join(missing)
        )


def load_candidates(
    max_impressions: int,
) -> pd.DataFrame:
    """
    Load a small number of test impressions.

    The original clicked labels are deliberately removed
    before inference.
    """

    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "clicked",
    ]

    candidates = pd.read_parquet(
        TEST_PATH,
        columns=columns,
    )

    # --------------------------------------------------------
    # Select complete impressions, rather than arbitrary rows.
    # --------------------------------------------------------

    impression_ids = (
        candidates["impression_id"]
        .drop_duplicates()
        .head(max_impressions)
    )

    candidates = candidates[
        candidates["impression_id"].isin(
            impression_ids
        )
    ].copy()

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Keep the original labels out of the inference pipeline.
    # --------------------------------------------------------

    candidates = candidates.drop(
        columns=["clicked"]
    )

    return candidates


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--impressions",
        type=int,
        default=100,
        help="Number of test impressions to score.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MIND LABEL-FREE INFERENCE SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\n[1/7] Checking inference artifacts...")

    check_required_files()

    print("       All required files found.")

    # --------------------------------------------------------
    # 2. Load candidates WITHOUT labels
    # --------------------------------------------------------

    print("\n[2/7] Loading label-free test candidates...")

    candidates = load_candidates(
        max_impressions=args.impressions
    )

    print(
        f"       Impressions: "
        f"{candidates['impression_id'].nunique():,}"
    )

    print(
        f"       Candidate rows: "
        f"{len(candidates):,}"
    )

    if "clicked" in candidates.columns:
        raise AssertionError(
            "clicked must not enter the inference pipeline."
        )

    print("       clicked column: HIDDEN")

    # --------------------------------------------------------
    # 3. Load article metadata
    # --------------------------------------------------------

    print("\n[3/7] Loading articles...")

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    # --------------------------------------------------------
    # 4. Load history
    # --------------------------------------------------------

    print("\n[4/7] Loading user history...")

    history = pd.read_parquet(
        HISTORY_PATH
    )

    selected_impressions = set(
        candidates["impression_id"].astype(str)
    )

    history = history[
        history["impression_id"]
        .astype(str)
        .isin(selected_impressions)
    ].copy()

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    # --------------------------------------------------------
    # 5. Load behavioral scores
    # --------------------------------------------------------

    print("\n[5/7] Loading behavioural scores...")

    behavioral_scores = pd.read_parquet(
        BEHAVIORAL_PATH
    )

    behavioral_scores = behavioral_scores[
        behavioral_scores["impression_id"]
        .astype(str)
        .isin(selected_impressions)
    ].copy()

    print(
        f"       Behavioural rows: "
        f"{len(behavioral_scores):,}"
    )

    # --------------------------------------------------------
    # 6. Load semantic model artifacts + BM25
    # --------------------------------------------------------

    print("\n[6/7] Loading retrieval artifacts...")

    print("       Building BM25 index...")

    bm25_retriever = build_mind_bm25(
        str(ARTICLES_PATH)
    )

    article_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    article_ids = np.load(
        ARTICLE_IDS_PATH,
        allow_pickle=True,
    )

    print(
        f"       Embedding shape: "
        f"{article_embeddings.shape}"
    )

    print(
        f"       Article IDs: "
        f"{len(article_ids):,}"
    )

    # --------------------------------------------------------
    # 7. Run REAL label-free inference
    # --------------------------------------------------------

    print("\n[7/7] Running hybrid inference...")

    predictions = predict_mind_candidates(
        candidates=candidates,
        history=history,
        articles=articles,
        bm25_retriever=bm25_retriever,
        article_embeddings=article_embeddings,
        article_ids=article_ids,
        behavioral_scores=behavioral_scores,
        max_history=10,
        lexical_weight=0.40,
        semantic_weight=0.30,
        behavioral_weight=0.30,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    required_columns = {
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
        "bm25_score",
        "semantic_score",
        "behavioral_score",
        "bm25_normalized",
        "semantic_normalized",
        "behavioral_normalized",
        "hybrid_score",
        "rank",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise AssertionError(
            "Prediction output is missing columns: "
            f"{sorted(missing)}"
        )

    if "clicked" in predictions.columns:
        raise AssertionError(
            "Prediction output must not contain clicked."
        )

    if predictions.empty:
        raise AssertionError(
            "Prediction output is empty."
        )

    if predictions["hybrid_score"].isna().any():
        raise AssertionError(
            "hybrid_score contains NaN."
        )

    if not np.isfinite(
        predictions["hybrid_score"].to_numpy()
    ).all():
        raise AssertionError(
            "hybrid_score contains non-finite values."
        )

    # --------------------------------------------------------
    # Verify ranking
    # --------------------------------------------------------

    rank_counts = (
        predictions
        .groupby("impression_id")["rank"]
        .count()
    )

    if (
        predictions
        .groupby("impression_id")["rank"]
        .apply(lambda x: x.min() != 1)
        .any()
    ):
        raise AssertionError(
            "Ranking does not start at rank 1."
        )

    if (
        predictions
        .groupby("impression_id")["rank"]
        .apply(
            lambda x:
                len(x)
                != len(set(x))
        )
        .any()
    ):
        raise AssertionError(
            "Duplicate ranks detected."
        )

    # --------------------------------------------------------
    # Verify candidate preservation
    # --------------------------------------------------------

    input_pairs = set(
        zip(
            candidates["impression_id"].astype(str),
            candidates["article_id"].astype(str),
        )
    )

    output_pairs = set(
        zip(
            predictions["impression_id"].astype(str),
            predictions["article_id"].astype(str),
        )
    )

    if input_pairs != output_pairs:
        raise AssertionError(
            "Prediction candidates do not exactly match "
            "the input candidate universe."
        )

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)

    print(
        f"Impressions scored: "
        f"{predictions['impression_id'].nunique():,}"
    )

    print(
        f"Candidates scored: "
        f"{len(predictions):,}"
    )

    print(
        f"Hybrid score range: "
        f"{predictions['hybrid_score'].min():.6f}"
        f" to "
        f"{predictions['hybrid_score'].max():.6f}"
    )

    print(
        f"Rank range: "
        f"{predictions['rank'].min()}"
        f" to "
        f"{predictions['rank'].max()}"
    )

    print("\nTop-ranked candidates:")

    print(
        predictions[
            [
                "impression_id",
                "article_id",
                "bm25_score",
                "semantic_score",
                "behavioral_score",
                "hybrid_score",
                "rank",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MIND LABEL-FREE SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()