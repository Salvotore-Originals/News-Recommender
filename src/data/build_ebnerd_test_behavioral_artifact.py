from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_testset"
    / "ebnerd_testset"
)

ARTICLES_PATH = TEST_ROOT / "articles.parquet"
HISTORY_PATH = TEST_ROOT / "test" / "history.parquet"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "test"
    / "behavioral"
)

OUTPUT_PATH = OUTPUT_DIR / "behavioral_history.parquet"
CATEGORY_MAP_PATH = OUTPUT_DIR / "category_map.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


BATCH_SIZE = 512
PROGRESS_EVERY = 10_000


def load_category_lookup() -> dict[str, int]:
    print("[1/5] Loading article categories...")

    articles = pd.read_parquet(
        ARTICLES_PATH,
        columns=[
            "article_id",
            "category",
        ],
    )

    categories = (
        articles["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    categories.sort()

    category_to_id = {
        category: index
        for index, category in enumerate(categories)
    }

    article_to_category = {}

    for row in articles.itertuples(index=False):
        if pd.isna(row.category):
            continue

        category = str(row.category)

        article_to_category[
            str(row.article_id)
        ] = category_to_id[category]

    print(
        f"       Articles with category: "
        f"{len(article_to_category):,}"
    )

    print(
        f"       Unique categories: "
        f"{len(category_to_id):,}"
    )

    return article_to_category, category_to_id


def make_schema() -> pa.Schema:
    return pa.schema(
        [
            (
                "user_id",
                pa.uint32(),
            ),
            (
                "timestamps_ns",
                pa.list_(pa.int64()),
            ),
            (
                "category_ids",
                pa.list_(pa.uint16()),
            ),
            (
                "category_positions",
                pa.list_(
                    pa.list_(pa.uint32())
                ),
            ),
        ]
    )


def process_batch(
    batch: pd.DataFrame,
    article_to_category: dict[str, int],
):
    user_ids = []
    timestamps_column = []
    category_ids_column = []
    category_positions_column = []

    for row in batch.itertuples(index=False):

        user_id = int(row.user_id)

        timestamps = row.impression_time_fixed
        article_ids = row.article_id_fixed

        if timestamps is None or article_ids is None:
            continue

        if len(timestamps) != len(article_ids):
            raise ValueError(
                "History array length mismatch "
                f"for user {user_id}: "
                f"{len(timestamps)} timestamps vs "
                f"{len(article_ids)} articles"
            )

        events = []

        for timestamp, article_id in zip(
            timestamps,
            article_ids,
        ):
            category_id = article_to_category.get(
                str(article_id)
            )

            if category_id is None:
                continue

            timestamp_ns = (
                pd.Timestamp(timestamp).value
            )

            events.append(
                (
                    timestamp_ns,
                    category_id,
                )
            )

        if not events:
            continue

        # Keep the temporal representation deterministic.
        events.sort(key=lambda x: x[0])

        timestamps_ns = [
            event[0]
            for event in events
        ]

        category_order = []
        positions_by_category = {}

        for position, (_, category_id) in enumerate(
            events
        ):

            if category_id not in positions_by_category:
                positions_by_category[
                    category_id
                ] = []

                category_order.append(
                    category_id
                )

            positions_by_category[
                category_id
            ].append(position)

        category_order.sort()

        category_positions = [
            positions_by_category[
                category_id
            ]
            for category_id in category_order
        ]

        user_ids.append(user_id)
        timestamps_column.append(timestamps_ns)
        category_ids_column.append(
            category_order
        )
        category_positions_column.append(
            category_positions
        )

    return {
        "user_id": user_ids,
        "timestamps_ns": timestamps_column,
        "category_ids": category_ids_column,
        "category_positions": category_positions_column,
    }


def write_batch(
    writer: pq.ParquetWriter | None,
    data: dict,
    schema: pa.Schema,
):
    if not data["user_id"]:
        return writer

    table = pa.Table.from_pydict(
        data,
        schema=schema,
    )

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_PATH,
            schema,
            compression="zstd",
            use_dictionary=False,
        )

    writer.write_table(table)

    return writer


def main() -> None:

    print("=" * 80)
    print(
        "EB-NeRD OFFICIAL TEST — "
        "MEMORY-EFFICIENT BEHAVIORAL ARTIFACT"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Article/category lookup
    # --------------------------------------------------------

    article_to_category, category_to_id = (
        load_category_lookup()
    )

    # --------------------------------------------------------
    # Stream history
    # --------------------------------------------------------

    print(
        "\n[2/5] Streaming official test history..."
    )

    parquet_file = pq.ParquetFile(
        HISTORY_PATH
    )

    print(
        f"       History rows: "
        f"{parquet_file.metadata.num_rows:,}"
    )

    schema = make_schema()

    writer = None

    users_written = 0
    events_written = 0
    batches = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "user_id",
            "impression_time_fixed",
            "article_id_fixed",
        ],
    ):

        batch = record_batch.to_pandas()

        data = process_batch(
            batch,
            article_to_category,
        )

        writer = write_batch(
            writer,
            data,
            schema,
        )

        users_in_batch = len(
            data["user_id"]
        )

        users_written += users_in_batch

        events_written += sum(
            len(values)
            for values
            in data["timestamps_ns"]
        )

        batches += 1

        if (
            users_written > 0
            and users_written % PROGRESS_EVERY
            < users_in_batch
        ):
            print(
                f"       Users written: "
                f"{users_written:,} | "
                f"Events: "
                f"{events_written:,}"
            )

    if writer is None:
        raise RuntimeError(
            "No behavioral history rows were written."
        )

    writer.close()

    # --------------------------------------------------------
    # Validate artifact
    # --------------------------------------------------------

    print(
        "\n[3/5] Validating artifact..."
    )

    output_file = pq.ParquetFile(
        OUTPUT_PATH
    )

    artifact_rows = (
        output_file.metadata.num_rows
    )

    if artifact_rows != users_written:
        raise RuntimeError(
            "Artifact row count mismatch: "
            f"expected {users_written:,}, "
            f"got {artifact_rows:,}"
        )

    print(
        f"       Users: "
        f"{artifact_rows:,}"
    )

    print(
        f"       Category-known events: "
        f"{events_written:,}"
    )

    # --------------------------------------------------------
    # Save category mapping
    # --------------------------------------------------------

    print(
        "\n[4/5] Saving category mapping..."
    )

    id_to_category = {
        str(category_id): category
        for category, category_id
        in category_to_id.items()
    }

    with open(
        CATEGORY_MAP_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "category_to_id": category_to_id,
                "id_to_category": id_to_category,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "EB-NeRD official test",
        "source_history_rows": int(
            parquet_file.metadata.num_rows
        ),
        "artifact_users": int(
            artifact_rows
        ),
        "category_known_events": int(
            events_written
        ),
        "unique_categories": int(
            len(category_to_id)
        ),
        "representation": {
            "timestamps": (
                "int64 nanoseconds since Unix epoch"
            ),
            "category_ids": (
                "uint16 category codes"
            ),
            "category_positions": (
                "positions of each category "
                "inside the user's chronological history"
            ),
        },
        "temporal_rule": (
            "Only events with timestamp < "
            "impression_time are eligible."
        ),
        "compression": "zstd",
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n[5/5] Complete."
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "MEMORY-EFFICIENT BEHAVIORAL ARTIFACT COMPLETE"
    )

    print(
        f"Parquet: {OUTPUT_PATH}"
    )

    print(
        f"Category map: {CATEGORY_MAP_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )

    print(
        f"Users: {artifact_rows:,}"
    )

    print(
        f"Events: {events_written:,}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
