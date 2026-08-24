from __future__ import annotations

"""
Memory-efficient EB-NeRD official-test hybrid inference.

Model logic preserved from the validated EB-NeRD hybrid:
    H = 0.0 * L_hat + 0.02 * S_hat + 0.98 * B_hat

The official-test runner avoids constructing the full temporal behavioral
lookup in RAM.  It uses:
  * official test article features
  * precomputed official-test semantic embeddings
  * precomputed official-test BM25 artifacts (kept for consistency)
  * validated compressed behavioral artifact
  * a compact per-user recent-history artifact for the semantic query

Before a full run, execute:
    python -m src.evaluation.run_ebnerd_test_hybrid --impressions 100

Then, if the smoke test passes:
    python -m src.evaluation.run_ebnerd_test_hybrid

The runner refuses to build an approximate semantic history if the official
history overlaps the test-impression time range.  This prevents silent
temporal leakage.
"""

from collections import defaultdict, deque
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset"
)

ARTICLE_FEATURES = (
    PROJECT_ROOT
    / "data/features/ebnerd/test/articles.parquet"
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "data/embeddings/ebnerd/test"
)

BM25_ROOT = (
    PROJECT_ROOT
    / "data/features/ebnerd/test/bm25"
)

BEHAVIORAL_ROOT = (
    PROJECT_ROOT
    / "data/features/ebnerd/test/behavioral"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "data/results/ebnerd/test"
)

SEMANTIC_HISTORY_PATH = (
    PROJECT_ROOT
    / "data/features/ebnerd/test/semantic_history_recent10.parquet"
)

BEHAVIORS_PATH = TEST_ROOT / "test/behaviors.parquet"
HISTORY_PATH = TEST_ROOT / "test/history.parquet"

MODEL_NAME = "all-MiniLM-L6-v2"

LEXICAL_WEIGHT = 0.0
SEMANTIC_WEIGHT = 0.02
BEHAVIORAL_WEIGHT = 0.98

HISTORY_QUERY_LENGTH = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impressions",
        type=int,
        default=None,
        help="Number of test impressions. Omit for the full test set.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Rows accumulated before a prediction chunk is written.",
    )
    parser.add_argument(
        "--rebuild-semantic-history",
        action="store_true",
        help="Rebuild the compact recent-history artifact.",
    )
    return parser.parse_args()


def normalize_scores(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return values
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def combine_scores(
    semantic_scores,
    behavioral_scores,
) -> np.ndarray:
    """
    Exact reduction of the validated hybrid because lexical_weight == 0.

    H = 0.02*S_hat + 0.98*B_hat
    """
    if not np.isclose(
        SEMANTIC_WEIGHT + BEHAVIORAL_WEIGHT,
        1.0,
    ):
        raise ValueError("Semantic + behavioral weights must sum to 1.")

    return (
        SEMANTIC_WEIGHT * normalize_scores(semantic_scores)
        + BEHAVIORAL_WEIGHT * normalize_scores(behavioral_scores)
    )


def load_articles():
    articles = pd.read_parquet(
        ARTICLE_FEATURES,
        columns=["article_id", "category"],
    )
    articles["article_id"] = articles["article_id"].astype(str)
    articles["category"] = articles["category"].fillna("").astype(str)

    article_categories = dict(
        zip(articles["article_id"], articles["category"])
    )
    return articles, article_categories


def load_embeddings():
    embeddings = np.load(
        EMBEDDING_ROOT / "article_embeddings.npy",
        mmap_mode="r",
    )
    article_ids = np.load(
        EMBEDDING_ROOT / "article_ids.npy",
        allow_pickle=True,
    ).astype(str)

    if embeddings.shape[0] != len(article_ids):
        raise RuntimeError("Embedding/article-ID mismatch.")

    article_to_index = {
        article_id: i
        for i, article_id in enumerate(article_ids)
    }

    return embeddings, article_to_index


def load_category_map():
    with open(
        BEHAVIORAL_ROOT / "category_map.json",
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return {
        str(category): int(category_id)
        for category, category_id
        in payload["category_to_id"].items()
    }


def load_behavioral_artifact():
    table = pq.read_table(
        BEHAVIORAL_ROOT / "behavioral_history.parquet"
    )
    df = table.to_pandas()

    return {
        str(row.user_id): row
        for row in df.itertuples(index=False)
    }


def compressed_behavioral_counts(
    user_row,
    impression_time,
    category_to_id,
):
    """
    Compute all category counts once for one impression.

    The previous implementation called compressed_behavioral_score()
    separately for every candidate. That meant repeatedly scanning the
    same user's category-position lists. For an impression with N
    candidates, this could repeat the same work N times.

    This implementation performs one binary search per category using
    the globally chronological timestamp array and the category's
    sorted event positions:

        count(category, t)
          = number of category positions whose timestamp < t

    Because positions are chronological, this is O(C log E) per
    impression, where C is the number of categories (32 here), rather
    than scanning the category history once per candidate.
    """
    if user_row is None:
        return {}, 0

    cutoff_ns = pd.Timestamp(impression_time).value

    timestamps = user_row.timestamps_ns
    category_ids = user_row.category_ids
    category_positions = user_row.category_positions

    counts = {}
    total = 0

    # Binary search without materializing expanded timestamps.
    # Each category's positions are in chronological order.
    for cid, positions in zip(
        category_ids,
        category_positions,
    ):
        lo = 0
        hi = len(positions)

        while lo < hi:
            mid = (lo + hi) // 2
            ts = int(timestamps[int(positions[mid])])

            if ts < cutoff_ns:
                lo = mid + 1
            else:
                hi = mid

        count = lo

        if count:
            category_id = int(cid)
            counts[category_id] = count
            total += count

    return counts, total


def compressed_behavioral_scores(
    user_row,
    candidate_ids,
    article_categories,
    impression_time,
    category_to_id,
):
    """
    Score all candidates for one impression using one compressed-history
    lookup.

    This preserves the validated formula:

        B(article, t) =
            count(user history in article category before t)
            ------------------------------------------------
            total known-category history events before t
    """
    scores = np.zeros(
        len(candidate_ids),
        dtype=np.float64,
    )

    if user_row is None:
        return scores

    counts, total = compressed_behavioral_counts(
        user_row,
        impression_time,
        category_to_id,
    )

    if total == 0:
        return scores

    denominator = float(total)

    for i, article_id in enumerate(candidate_ids):
        category = article_categories.get(
            str(article_id),
            "",
        )

        category_id = category_to_id.get(
            category
        )

        if category_id is not None:
            scores[i] = (
                counts.get(
                    category_id,
                    0,
                )
                / denominator
            )

    return scores

def build_semantic_history_artifact():
    """
    Stream official history and retain the most recent 10 unique articles
    per user.

    This is exact for the semantic query when all official history events
    occur before the earliest test impression.  The caller checks that
    condition before allowing this artifact to be used.
    """
    print("\n[SEMANTIC HISTORY] Building compact recent-10 artifact...")

    pf = pq.ParquetFile(HISTORY_PATH)

    # user -> deque[(timestamp_ns, article_id)]
    recent = defaultdict(lambda: deque(maxlen=HISTORY_QUERY_LENGTH))

    rows = 0
    start = time.perf_counter()

    for batch in pf.iter_batches(
        batch_size=2_000,
        columns=[
            "user_id",
            "impression_time_fixed",
            "article_id_fixed",
        ],
    ):
        frame = batch.to_pandas()

        for row in frame.itertuples(index=False):
            user_id = str(row.user_id)
            timestamps = row.impression_time_fixed
            article_ids = row.article_id_fixed

            events = recent[user_id]

            # Preserve chronological order even if the source row is not
            # guaranteed to be sorted.
            local = [
                (
                    pd.Timestamp(ts).value,
                    str(article_id),
                )
                for ts, article_id in zip(
                    timestamps,
                    article_ids,
                )
            ]

            local.sort(key=lambda x: x[0])

            for event in local:
                events.append(event)

            # deque already bounds the number of retained events.

        rows += len(frame)

        if rows % 50_000 < len(frame):
            print(
                f"       History rows processed: {rows:,}"
            )

    user_ids = []
    timestamps_col = []
    article_ids_col = []

    for user_id, events in recent.items():
        ordered = list(events)
        ordered.sort(key=lambda x: x[0])

        user_ids.append(user_id)
        timestamps_col.append([int(x[0]) for x in ordered])
        article_ids_col.append([x[1] for x in ordered])

    table = pa.Table.from_pydict(
        {
            "user_id": user_ids,
            "timestamps_ns": timestamps_col,
            "article_ids": article_ids_col,
        },
        schema=pa.schema(
            [
                pa.field("user_id", pa.string()),
                pa.field("timestamps_ns", pa.list_(pa.int64())),
                pa.field("article_ids", pa.list_(pa.string())),
            ]
        ),
    )

    SEMANTIC_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        SEMANTIC_HISTORY_PATH,
        compression="zstd",
        use_dictionary=False,
    )

    elapsed = time.perf_counter() - start

    print(
        f"       Users: {len(user_ids):,}"
    )
    print(
        f"       Saved: {SEMANTIC_HISTORY_PATH}"
    )
    print(
        f"       Runtime: {elapsed:.1f}s"
    )


def load_semantic_history():
    table = pq.read_table(
        SEMANTIC_HISTORY_PATH
    )

    return {
        str(row.user_id): row
        for row in table.to_pandas().itertuples(index=False)
    }


def validate_temporal_assumption():
    """
    The compact recent-10 semantic artifact is exact only if test history
    ends before test impressions begin.

    We scan Parquet in batches, so no huge dataframe is materialized.
    """
    print("\n[VALIDATION] Checking history/test temporal boundary...")

    behavior_pf = pq.ParquetFile(BEHAVIORS_PATH)
    history_pf = pq.ParquetFile(HISTORY_PATH)

    min_test = None
    max_test = None

    for batch in behavior_pf.iter_batches(
        batch_size=50_000,
        columns=["impression_time"],
    ):
        frame = batch.to_pandas()
        times = pd.to_datetime(frame["impression_time"])
        batch_min = times.min()
        batch_max = times.max()

        min_test = batch_min if min_test is None else min(min_test, batch_min)
        max_test = batch_max if max_test is None else max(max_test, batch_max)

    max_history = None

    for batch in history_pf.iter_batches(
        batch_size=2_000,
        columns=["impression_time_fixed"],
    ):
        frame = batch.to_pandas()

        for value in frame["impression_time_fixed"]:
            if len(value) == 0:
                continue
            local_max = pd.to_datetime(value).max()
            max_history = (
                local_max
                if max_history is None
                else max(max_history, local_max)
            )

    print(f"       Earliest test impression: {min_test}")
    print(f"       Latest test impression:   {max_test}")
    print(f"       Latest history event:     {max_history}")

    if max_history is None or min_test is None:
        raise RuntimeError("Could not determine temporal boundary.")

    if max_history >= min_test:
        raise RuntimeError(
            "Official history overlaps the test-impression period. "
            "The compact recent-10 semantic artifact could then lose "
            "earlier point-in-time events and is not safe to use. "
            "Do not launch the full test until a temporal semantic "
            "history index is built."
        )

    print("       Temporal boundary: PASS")


def semantic_query(
    user_row,
    impression_time,
    article_to_index,
    embeddings,
):
    if user_row is None:
        return None

    cutoff_ns = pd.Timestamp(impression_time).value

    selected = []
    seen = set()

    # History is chronological; process newest first.
    for timestamp_ns, article_id in reversed(
        list(zip(user_row.timestamps_ns, user_row.article_ids))
    ):
        if int(timestamp_ns) >= cutoff_ns:
            continue

        article_id = str(article_id)

        if article_id in seen:
            continue

        index = article_to_index.get(article_id)
        if index is None:
            continue

        seen.add(article_id)
        selected.append(index)

        if len(selected) >= HISTORY_QUERY_LENGTH:
            break

    if not selected:
        return None

    query = np.asarray(
        embeddings[selected],
        dtype=np.float32,
    ).mean(axis=0)

    norm = np.linalg.norm(query)

    if norm == 0:
        return None

    return (query / norm).astype(np.float32)


def semantic_scores(
    query,
    candidate_ids,
    article_to_index,
    embeddings,
):
    scores = np.zeros(
        len(candidate_ids),
        dtype=np.float64,
    )

    if query is None:
        return scores

    valid_positions = []
    indices = []

    for position, article_id in enumerate(candidate_ids):
        index = article_to_index.get(str(article_id))
        if index is not None:
            valid_positions.append(position)
            indices.append(index)

    if not indices:
        return scores

    matrix = np.asarray(
        embeddings[indices],
        dtype=np.float32,
    )

    values = matrix @ query

    for position, value in zip(valid_positions, values):
        scores[position] = float(value)

    return scores


def rank_candidates(candidate_ids, scores):
    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: (-float(x[1]), str(x[0])),
    )
    return ranked


def load_behaviors():
    return pq.ParquetFile(BEHAVIORS_PATH)


def count_test_impressions():
    """Return the exact official test impression count from parquet metadata."""
    return pq.ParquetFile(BEHAVIORS_PATH).metadata.num_rows


def process(
    max_impressions,
    chunk_size,
    semantic_history,
    article_categories,
    category_to_id,
    behavioral_lookup,
    embeddings,
    embedding_lookup,
    total_impressions,
):
    """
    Process the official test set without materializing all 2.37M
    impressions or 93M prediction rows in RAM.
    """
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    parts = []
    output_rows = 0
    processed = 0
    start = time.perf_counter()

    chunk_rows = []

    behaviors = load_behaviors()

    for batch in behaviors.iter_batches(
        batch_size=2_000,
        columns=[
            "impression_id",
            "impression_time",
            "article_ids_inview",
            "user_id",
        ],
    ):
        frame = batch.to_pandas()

        for row in frame.itertuples(index=False):
            if (
                max_impressions is not None
                and processed >= max_impressions
            ):
                break

            impression_id = str(row.impression_id)
            user_id = str(row.user_id)
            impression_time = pd.Timestamp(row.impression_time)

            # Remove repeated article IDs while preserving the
            # original in-view order. This guarantees one
            # prediction per impression/article pair.
            candidate_ids = list(
                dict.fromkeys(
                    str(article_id)
                    for article_id in row.article_ids_inview
                )
            )

            if not candidate_ids:
                continue

            # Semantic component.
            user_history = semantic_history.get(user_id)

            query = semantic_query(
                user_history,
                impression_time,
                embedding_lookup,
                embeddings,
            )

            semantic = semantic_scores(
                query,
                candidate_ids,
                embedding_lookup,
                embeddings,
            )

            # Behavioral component.
            #
            # IMPORTANT PERFORMANCE OPTIMIZATION:
            # Compute the user's category counts ONCE per impression.
            # The old implementation recomputed the same counts for
            # every candidate, which made full-test inference extremely
            # slow.
            user_behavior = behavioral_lookup.get(user_id)

            behavioral = compressed_behavioral_scores(
                user_behavior,
                candidate_ids,
                article_categories,
                impression_time,
                category_to_id,
            )

            final_scores = combine_scores(
                semantic,
                behavioral,
            )

            ranked = rank_candidates(
                candidate_ids,
                final_scores,
            )

            for rank, (article_id, score) in enumerate(
                ranked,
                start=1,
            ):
                chunk_rows.append(
                    (
                        impression_id,
                        article_id,
                        rank,
                        float(score),
                    )
                )

            processed += 1
            output_rows += len(ranked)

            if len(chunk_rows) >= chunk_size:
                part_number = len(parts) + 1
                part_path = (
                    RESULT_ROOT
                    / f"predictions_part_{part_number:04d}.parquet"
                )

                part_df = pd.DataFrame(
                    chunk_rows,
                    columns=[
                        "impression_id",
                        "article_id",
                        "rank",
                        "score",
                    ],
                )

                part_df.to_parquet(
                    part_path,
                    index=False,
                )

                parts.append(part_path)
                chunk_rows.clear()

            if processed in {1, 2, 3} or processed % 10_000 == 0:
                elapsed = time.perf_counter() - start
                rate = processed / elapsed if elapsed else 0.0
                target_count = (
                    max_impressions
                    if max_impressions is not None
                    else total_impressions
                )
                target = f"{target_count:,}"

                print(
                    f"       {processed:,}/{target} impressions "
                    f"({rate:.2f} impressions/sec)"
                )

        if (
            max_impressions is not None
            and processed >= max_impressions
        ):
            break

    if chunk_rows:
        part_number = len(parts) + 1
        part_path = (
            RESULT_ROOT
            / f"predictions_part_{part_number:04d}.parquet"
        )

        pd.DataFrame(
            chunk_rows,
            columns=[
                "impression_id",
                "article_id",
                "rank",
                "score",
            ],
        ).to_parquet(
            part_path,
            index=False,
        )

        parts.append(part_path)
        chunk_rows.clear()

    if not parts:
        raise RuntimeError("No prediction rows were generated.")

    final_path = (
        RESULT_ROOT
        / "ebnerd_test_hybrid_predictions.parquet"
    )

    # Combine through ParquetWriter to avoid loading all predictions
    # simultaneously.
    writer = None

    for part_path in parts:
        part_pf = pq.ParquetFile(part_path)

        for row_group in range(part_pf.num_row_groups):
            table = part_pf.read_row_group(row_group)

            if writer is None:
                writer = pq.ParquetWriter(
                    final_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

    if writer is not None:
        writer.close()

    elapsed = time.perf_counter() - start

    return {
        "impressions": processed,
        "prediction_rows": output_rows,
        "parts": len(parts),
        "runtime_seconds": elapsed,
        "output": str(final_path),
    }


def smoke_checks(prediction_path, expected_impressions):
    """Memory-efficient final validation of the prediction parquet."""
    print("\n[VALIDATION] Checking predictions...")

    parquet_file = pq.ParquetFile(prediction_path)

    required = {
        "impression_id",
        "article_id",
        "rank",
        "score",
    }

    available = set(parquet_file.schema_arrow.names)
    missing = required - available

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    total_rows = 0
    impression_ids = set()
    pair_seen = set()
    duplicate_pairs = 0
    nan_scores = 0
    infinite_scores = 0

    for batch in parquet_file.iter_batches(
        batch_size=250_000,
        columns=["impression_id", "article_id", "score"],
    ):
        frame = batch.to_pandas()
        total_rows += len(frame)

        nan_scores += int(frame["score"].isna().sum())

        values = frame["score"].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        infinite_scores += int(np.isinf(values).sum())

        for impression_id, article_id in zip(
            frame["impression_id"],
            frame["article_id"],
        ):
            impression_id = str(impression_id)
            article_id = str(article_id)
            impression_ids.add(impression_id)

            key = (impression_id, article_id)
            if key in pair_seen:
                duplicate_pairs += 1
            else:
                pair_seen.add(key)

    print(f"       Prediction rows: {total_rows:,}")
    print(f"       Impressions: {len(impression_ids):,}")
    print(f"       NaN scores: {nan_scores:,}")
    print(f"       Infinite scores: {infinite_scores:,}")
    print(
        "       Duplicate impression/article pairs: "
        f"{duplicate_pairs:,}"
    )

    if total_rows == 0:
        raise RuntimeError("Prediction file is empty.")

    if nan_scores:
        raise RuntimeError(
            f"NaN hybrid scores detected: {nan_scores:,}"
        )

    if infinite_scores:
        raise RuntimeError(
            f"Infinite hybrid scores detected: {infinite_scores:,}"
        )

    if duplicate_pairs:
        raise RuntimeError(
            "Duplicate impression/article predictions detected: "
            f"{duplicate_pairs:,}"
        )

    if len(impression_ids) != expected_impressions:
        raise RuntimeError(
            f"Expected {expected_impressions:,} impressions, "
            f"got {len(impression_ids):,}."
        )

    print("       PASS")

def main():
    args = parse_args()

    print("=" * 80)
    print("EB-NeRD OFFICIAL TEST — MEMORY-EFFICIENT HYBRID INFERENCE")
    print("=" * 80)

    print("\n[1/8] Loading official test article features...")

    _, article_categories = load_articles()

    print(f"       Articles: {len(article_categories):,}")

    print("\n[2/8] Loading semantic embeddings...")

    embeddings, embedding_lookup = load_embeddings()

    print(f"       Embeddings: {embeddings.shape}")

    print("\n[3/8] Loading compressed behavioral artifact...")

    category_to_id = load_category_map()
    behavioral_lookup = load_behavioral_artifact()

    print(f"       Behavioral users: {len(behavioral_lookup):,}")
    print(f"       Categories: {len(category_to_id):,}")

    print("\n[4/8] Checking temporal safety...")

    validate_temporal_assumption()

    print("\n[5/8] Preparing compact semantic history...")

    if (
        args.rebuild_semantic_history
        or not SEMANTIC_HISTORY_PATH.exists()
    ):
        build_semantic_history_artifact()
    else:
        print(
            f"       Using existing: {SEMANTIC_HISTORY_PATH}"
        )

    semantic_history = load_semantic_history()

    print(
        f"       Semantic-history users: "
        f"{len(semantic_history):,}"
    )

    # Ensure BM25 artifacts exist even though lexical weight is zero.
    # This catches accidental missing official-test artifacts without
    # paying the enormous cost of building a second BM25 corpus.
    print("\n[6/8] Checking official BM25 artifacts...")

    required_bm25 = [
        BM25_ROOT / "article_ids.npy",
        BM25_ROOT / "document_lengths.npy",
        BM25_ROOT / "idf.json",
        BM25_ROOT / "token_frequencies.pkl",
        BM25_ROOT / "metadata.json",
    ]

    missing = [
        str(path)
        for path in required_bm25
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing official-test BM25 artifacts:\n"
            + "\n".join(missing)
        )

    print("       BM25 artifacts: OK")
    print(
        "       Lexical weight: "
        f"{LEXICAL_WEIGHT:.2f} "
        "(no lexical scoring required)"
    )

    print("\n[7/8] Running inference...")

    total_test_impressions = count_test_impressions()

    if args.impressions is not None:
        if args.impressions <= 0:
            raise ValueError("--impressions must be positive.")
        if args.impressions > total_test_impressions:
            raise ValueError(
                f"--impressions={args.impressions:,} exceeds the "
                f"official test set size of {total_test_impressions:,}."
            )

    print(
        f"       Official test impressions: "
        f"{total_test_impressions:,}"
    )

    stats = process(
        args.impressions,
        args.chunk_size,
        semantic_history,
        article_categories,
        category_to_id,
        behavioral_lookup,
        embeddings,
        embedding_lookup,
        total_test_impressions,
    )

    print("\n[8/8] Validating result...")

    smoke_checks(
        stats["output"],
        stats["impressions"],
    )

    print("\n" + "=" * 80)
    print("EB-NeRD OFFICIAL TEST HYBRID INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Impressions: {stats['impressions']:,}")
    print(f"Prediction rows: {stats['prediction_rows']:,}")
    print(f"Parts: {stats['parts']:,}")
    print(f"Runtime: {stats['runtime_seconds']:.2f}s")
    print(f"Saved: {stats['output']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
