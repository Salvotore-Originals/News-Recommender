from __future__ import annotations

import gc
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.mind_bm25 import build_mind_bm25
from src.retrieval.mind_query import MINDQueryBuilder
from src.retrieval.mind_large_behavioral import (
    load_article_metadata,
    load_train_popularity,
    score_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_BEHAVIORS = (
    PROJECT_ROOT
    / "data/raw/mind/test/behaviors.tsv"
)

TEST_ARTICLES = (
    PROJECT_ROOT
    / "data/processed/mind_test/articles.parquet"
)

SEMANTIC_DIR = (
    PROJECT_ROOT
    / "data/features/mind_test/semantic"
)

EMBEDDINGS_PATH = (
    SEMANTIC_DIR / "article_embeddings.npy"
)

ARTICLE_IDS_PATH = (
    SEMANTIC_DIR / "article_ids.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/results/mind"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mind_large_test_predictions.parquet"
)


LEXICAL_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.30
BEHAVIORAL_WEIGHT = 0.30


def min_max(values: np.ndarray) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return np.zeros(0)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros(
            len(values),
            dtype=np.float64,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--impressions",
        type=int,
        default=None,
        help=(
            "Number of test impressions to process. "
            "Omit for the complete test set."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help=(
            "Number of impressions kept in memory before "
            "writing a prediction chunk."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MIND LARGE HYBRID INFERENCE")
    print("=" * 70)

    # ============================================================
    # 1. Load articles
    # ============================================================

    print("\n[1/8] Loading official test articles...")

    articles = pd.read_parquet(
        TEST_ARTICLES
    )

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
    )

    articles["title"] = (
        articles["title"]
        .fillna("")
        .astype(str)
    )

    articles["abstract"] = (
        articles["abstract"]
        .fillna("")
        .astype(str)
    )

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    # ============================================================
    # 2. Build BM25
    # ============================================================

    print("\n[2/8] Building BM25 index...")

    bm25 = build_mind_bm25(
        str(TEST_ARTICLES)
    )

    print("       BM25 ready.")

    # ============================================================
    # 3. Load semantic embeddings
    # ============================================================

    print("\n[3/8] Loading semantic embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    article_ids = np.load(
        ARTICLE_IDS_PATH,
        allow_pickle=True,
    ).astype(str)

    if embeddings.shape[0] != len(
        article_ids
    ):
        raise RuntimeError(
            "Embedding/article-ID mismatch."
        )

    embedding_lookup = {
        article_id: index
        for index, article_id
        in enumerate(article_ids)
    }

    # Normalize the complete embedding matrix ONCE.
    # This removes repeated candidate normalization work.
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    ).copy()

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embedding_norms[
        embedding_norms == 0
    ] = 1.0

    embeddings /= embedding_norms

    print(
        f"       Embeddings: "
        f"{embeddings.shape}"
    )

    # ============================================================
    # 4. Load behavioral artifacts
    # ============================================================

    print("\n[4/8] Loading behavioral artifacts...")

    (
        category_lookup,
        subcategory_lookup,
    ) = load_article_metadata()

    (
        popularity_lookup,
        global_ctr,
    ) = load_train_popularity()

    print(
        f"       Categories: "
        f"{len(category_lookup):,}"
    )

    print(
        f"       Popularity entries: "
        f"{len(popularity_lookup):,}"
    )

    print(
        f"       Global CTR: "
        f"{global_ctr:.8f}"
    )

    # ============================================================
    # 5. Query builder
    # ============================================================

    query_builder = MINDQueryBuilder(
        articles=articles,
        text_field="title",
        max_history=10,
        max_query_terms=80,
    )

    # ============================================================
    # 6. Open official test behaviors
    # ============================================================

    print("\n[5/8] Loading official test impressions...")

    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "impressions",
    ]

    behaviors = pd.read_csv(
        TEST_BEHAVIORS,
        sep="\t",
        header=None,
        names=columns,
        dtype="string",
        keep_default_na=False,
        nrows=args.impressions,
    )

    print(
        f"       Impressions: "
        f"{len(behaviors):,}"
    )

    # ============================================================
    # 7. Inference
    # ============================================================

    print("\n[6/8] Running hybrid inference...")

    # Remove stale chunk files from an interrupted/previous run.
    for stale_path in OUTPUT_DIR.glob(
        "mind_large_predictions_part_*.parquet"
    ):
        stale_path.unlink()

    chunk_rows = []

    CHUNK_IMPRESSIONS = args.chunk_size

    if CHUNK_IMPRESSIONS <= 0:
        raise ValueError(
            "--chunk-size must be positive."
        )

    part_number = 0
    total_rows = 0

    start_time = time.time()

    for count, row in enumerate(
        behaviors.itertuples(index=False),
        start=1,
    ):

        impression_id = str(
            row.impression_id
        )

        candidate_tokens = (
            str(row.impressions)
            .split()
        )

        candidate_ids = [
            token.rsplit("-", 1)[0]
            for token in candidate_tokens
        ]

        history = (
            str(row.history).split()
            if row.history
            else []
        )

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        query = query_builder.build_query(
            history
        )

        if query:

            lexical_results = bm25.score_candidates(
                query=query,
                article_ids=candidate_ids,
            )

            lexical_scores = np.asarray(
                [
                    score
                    for _, score in lexical_results
            ],
            dtype=np.float64,
        )

        else:

            lexical_scores = np.zeros(
                len(candidate_ids),
                dtype=np.float64,
            )

        # --------------------------------------------------------
        # Semantic
        # --------------------------------------------------------

        candidate_indices = np.fromiter(
            (
                embedding_lookup.get(
                    article_id,
                    -1,
                )
                for article_id in candidate_ids
            ),
            dtype=np.int32,
            count=len(candidate_ids),
        )

        valid = candidate_indices >= 0

        semantic_scores = np.zeros(
            len(candidate_ids),
            dtype=np.float64,
        )

        if query and valid.any():

            history_indices = np.fromiter(
                (
                    embedding_lookup.get(
                        article_id,
                        -1,
                    )
                    for article_id in history[-10:]
                ),
                dtype=np.int32,
            )

            history_indices = history_indices[
                history_indices >= 0
            ]

            if len(history_indices) > 0:

                query_embedding = embeddings[
                    history_indices
                ].mean(
                    axis=0
                )

                query_norm = np.linalg.norm(
                    query_embedding
                )

                if query_norm > 0:

                    query_embedding = (
                        query_embedding
                        / query_norm
                    )

                    valid_indices = (
                        candidate_indices[valid]
                    )

                    # Embeddings are already L2-normalized.
                    # Semantic cosine similarity is therefore
                    # a direct dot product.
                    semantic_scores[valid] = (
                        embeddings[
                            valid_indices
                        ]
                        @ query_embedding
                    ).astype(
                        np.float64
                    )

        # --------------------------------------------------------
        # Behavioral
        # --------------------------------------------------------

        behavioral_scores = score_candidates(
            candidate_ids=candidate_ids,
            history=history,
            category_lookup=category_lookup,
            subcategory_lookup=subcategory_lookup,
            popularity_lookup=popularity_lookup,
            global_ctr=global_ctr,
        )

        behavioral_scores = np.asarray(
            behavioral_scores,
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # Normalize per impression
        # --------------------------------------------------------

        lexical_normalized = min_max(
            lexical_scores
        )

        semantic_normalized = min_max(
            semantic_scores
        )

        behavioral_normalized = min_max(
            behavioral_scores
        )

        hybrid_scores = (
            LEXICAL_WEIGHT
            * lexical_normalized
            +
            SEMANTIC_WEIGHT
            * semantic_normalized
            +
            BEHAVIORAL_WEIGHT
            * behavioral_normalized
        )

        # --------------------------------------------------------
        # Ranking
        # --------------------------------------------------------

        order = np.argsort(
            -hybrid_scores,
            kind="stable",
        )

        for rank, index in enumerate(
            order,
            start=1,
        ):

            chunk_rows.append(
                {
                    "impression_id":
                        impression_id,
                    "article_id":
                        candidate_ids[index],
                    "score":
                        float(hybrid_scores[index]),
                    "rank":
                        rank,
                }
            )

        if (
            count <= 3
            or count % 1000 == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                count / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"       {count:,}/"
                f"{len(behaviors):,} "
                f"impressions "
                f"({rate:.2f} impressions/sec)"
            )

        # --------------------------------------------------------
        # Write prediction chunk
        # --------------------------------------------------------

        if (
            count % CHUNK_IMPRESSIONS == 0
            or count == len(behaviors)
        ):

            part_number += 1

            chunk = pd.DataFrame(
                chunk_rows
            )

            part_path = (
                OUTPUT_DIR
                / (
                    "mind_large_predictions_part_"
                    f"{part_number:04d}.parquet"
                )
            )

            chunk.to_parquet(
                part_path,
                index=False,
            )

            total_rows += len(chunk)

            print(
                f"       Wrote chunk "
                f"{part_number}: "
                f"{len(chunk):,} rows"
            )

            chunk_rows.clear()

            del chunk

            # Full GC every chunk is expensive. Run it periodically.
            if part_number % 5 == 0:
                gc.collect()

    # ============================================================
    # 8. Save
    # ============================================================

    print("\n[7/8] Prediction chunks written.")

    print(
        f"       Total rows: "
        f"{total_rows:,}"
    )

    print(
        f"       Parts: "
        f"{part_number}"
    )

    print("\n[8/8] Combining prediction chunks...")

    parts = sorted(
        OUTPUT_DIR.glob(
            "mind_large_predictions_part_*.parquet"
        )
    )

    if not parts:
        raise RuntimeError(
            "No prediction chunks were written."
        )

    frames = [
        pd.read_parquet(path)
        for path in parts
    ]

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    predictions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"       Final rows: "
        f"{len(predictions):,}"
    )

    print(
        f"       Saved: "
        f"{OUTPUT_PATH}"
    )

    del predictions
    del frames

    gc.collect()

    for path in parts:
        path.unlink()

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"       Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print("\n" + "=" * 70)
    print("MIND LARGE HYBRID INFERENCE COMPLETE")
    print("=" * 70)



if __name__ == "__main__":
    main()