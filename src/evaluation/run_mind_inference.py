from __future__ import annotations

import argparse
import time
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
    / "data/processed/mind/splits/test.parquet"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data/processed/mind/user_history.parquet"
)

ARTICLES_PATH = (
    PROJECT_ROOT
    / "data/features/mind/articles.parquet"
)

BEHAVIORAL_PATH = (
    PROJECT_ROOT
    / "data/features/mind/test_behavioral_score.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data/features/mind/semantic/article_embeddings.npy"
)

ARTICLE_IDS_PATH = (
    PROJECT_ROOT
    / "data/features/mind/semantic/article_ids.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/results/mind/test_predictions.parquet"
)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-impressions",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-impressions",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MIND PRODUCTION INFERENCE")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Load static artifacts ONCE
    # --------------------------------------------------------

    print("\n[1/6] Loading static artifacts...")

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    history = pd.read_parquet(
        HISTORY_PATH
    )

    behavioral = pd.read_parquet(
        BEHAVIORAL_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    article_ids = np.load(
        ARTICLE_IDS_PATH,
        allow_pickle=True,
    )

    print(
        f"       Articles: {len(articles):,}"
    )

    print(
        f"       History rows: {len(history):,}"
    )

    print(
        f"       Behavioural rows: {len(behavioral):,}"
    )

    print(
        f"       Embeddings: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build BM25 ONCE
    # --------------------------------------------------------

    print("\n[2/6] Building BM25 index...")

    bm25 = build_mind_bm25(
        str(ARTICLES_PATH)
    )

    print("       BM25 ready.")

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    print("\n[3/6] Loading candidates...")

    candidates = pd.read_parquet(
        TEST_PATH,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
        ],
    )

    if args.max_impressions is not None:

        selected = (
            candidates[
                "impression_id"
            ]
            .drop_duplicates()
            .head(args.max_impressions)
        )

        candidates = candidates[
            candidates["impression_id"].isin(
                selected
            )
        ].copy()

    impression_ids = (
        candidates[
            "impression_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    total_impressions = len(
        impression_ids
    )

    print(
        f"       Impressions: "
        f"{total_impressions:,}"
    )

    print(
        f"       Candidates: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    print("\n[4/6] Running inference...")

    results = []

    for start in range(
        0,
        total_impressions,
        args.batch_impressions,
    ):

        batch_ids = impression_ids[
            start:
            start + args.batch_impressions
        ]

        batch_candidates = candidates[
            candidates["impression_id"].isin(
                batch_ids
            )
        ].copy()

        batch_history = history[
            history["impression_id"].isin(
                batch_ids
            )
        ].copy()

        batch_behavioral = behavioral[
            behavioral["impression_id"].isin(
                batch_ids
            )
        ].copy()

        batch_predictions = (
            predict_mind_candidates(
                candidates=batch_candidates,
                history=batch_history,
                articles=articles,
                bm25_retriever=bm25,
                article_embeddings=embeddings,
                article_ids=article_ids,
                behavioral_scores=batch_behavioral,
                max_history=10,
                lexical_weight=0.40,
                semantic_weight=0.30,
                behavioral_weight=0.30,
            )
        )

        results.append(
            batch_predictions
        )

        completed = min(
            start
            + len(batch_ids),
            total_impressions,
        )

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            completed / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"       {completed:,}/"
            f"{total_impressions:,} impressions "
            f"({rate:.2f} impressions/sec)"
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print("\n[5/6] Combining predictions...")

    predictions = pd.concat(
        results,
        ignore_index=True,
    )

    predictions = predictions.sort_values(
        [
            "impression_id",
            "rank",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if "clicked" in predictions.columns:
        raise ValueError(
            "Prediction output contains clicked."
        )

    if predictions["hybrid_score"].isna().any():
        raise ValueError(
            "Prediction contains NaN scores."
        )

    if not np.isfinite(
        predictions[
            "hybrid_score"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Prediction contains non-finite scores."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[6/6] Saving predictions...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"       Rows: "
        f"{len(predictions):,}"
    )

    print(
        f"       Saved: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"       Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print("\n" + "=" * 70)
    print("MIND INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()