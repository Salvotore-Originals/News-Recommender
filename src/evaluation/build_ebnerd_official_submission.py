from __future__ import annotations

"""
Build the official EB-NeRD / RecSys Challenge 2024 submission.

IMPORTANT:
The official FuxiCTR EB-NeRD submission writer outputs:

    <impression_id> [r1,r2,r3,...]

where r_i is the ORIGINAL 1-based position of the candidate article in
that impression's article_ids_inview list, ordered by predicted score.

Therefore this is NOT an article-ID submission.

This script converts the already-generated prediction parquet into the
official predictions.txt and ZIP without loading the 2.15 GB prediction
file or all 13.5M impressions into RAM.
"""

from pathlib import Path
import ast
import json
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = (
    PROJECT_ROOT
    / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
)

BEHAVIORS = (
    PROJECT_ROOT
    / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/results/ebnerd/test/submission"
)

PREDICTIONS_TXT = OUTPUT_DIR / "predictions.txt"
PREDICTIONS_ZIP = OUTPUT_DIR / "predictions.zip"

EXPECTED_IMPRESSIONS = 13_536_710
PRED_BATCH_SIZE = 1_000_000
BEHAVIOR_BATCH_SIZE = 250_000


def normalize_id(value):
    return int(value)


def as_article_list(value):
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return [int(x) for x in parsed]
    return [int(x) for x in list(value)]


def prediction_group_key(row):
    return int(row["impression_id"])


class PredictionStream:
    """Streaming iterator over prediction rows."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        self.iterator = self.file.iter_batches(
            batch_size=PRED_BATCH_SIZE,
            columns=["impression_id", "article_id", "rank"],
        )
        self.current = None
        self.index = 0
        self.batch = None
        self._load_next_batch()

    def _load_next_batch(self):
        try:
            self.batch = next(self.iterator).to_pandas()
            self.index = 0
        except StopIteration:
            self.batch = None
            self.index = 0

    def pop_impression(self):
        if self.batch is None:
            return None

        impression_id = int(
            self.batch.iloc[self.index]["impression_id"]
        )

        rows = []

        while True:
            if self.batch is None:
                break

            while self.index < len(self.batch):
                row = self.batch.iloc[self.index]
                current_id = int(row["impression_id"])

                if current_id != impression_id:
                    return impression_id, rows

                rows.append(
                    (
                        int(row["article_id"]),
                        int(row["rank"]),
                    )
                )
                self.index += 1

            self._load_next_batch()

            if self.batch is None:
                break

        return impression_id, rows


def build_prediction_positions(prediction_rows, inview):
    """
    Convert model-ranked article IDs to original 1-based in-view positions.

    The official EB-NeRD challenge submission uses candidate positions,
    not article IDs. The public FuxiCTR submission writer confirms this:
    it ranks candidates, stores their original indices, then writes
    [1-based positions] per impression.
    """
    position_by_article = {}

    for position, article_id in enumerate(inview, start=1):
        article_id = int(article_id)

        # If the source impression itself contains duplicate article IDs,
        # preserve the first occurrence, matching the candidate ordering.
        if article_id not in position_by_article:
            position_by_article[article_id] = position

    ranked = sorted(
        prediction_rows,
        key=lambda item: (item[1], item[0]),
    )

    positions = []

    for article_id, _rank in ranked:
        position = position_by_article.get(article_id)

        if position is None:
            raise RuntimeError(
                "Prediction article is not present in official "
                f"article_ids_inview. article_id={article_id}"
            )

        positions.append(position)

    return positions


def main():
    print("=" * 80)
    print("EB-NeRD OFFICIAL SUBMISSION BUILDER")
    print("=" * 80)

    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)

    if not BEHAVIORS.exists():
        raise FileNotFoundError(BEHAVIORS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_file = pq.ParquetFile(PREDICTIONS)
    behavior_file = pq.ParquetFile(BEHAVIORS)

    print("\n[1/5] Checking prediction artifact...")
    print(
        f"       Prediction rows: "
        f"{prediction_file.metadata.num_rows:,}"
    )
    print(
        f"       Prediction row groups: "
        f"{prediction_file.num_row_groups:,}"
    )

    print("\n[2/5] Checking official test behaviors...")
    behavior_rows = behavior_file.metadata.num_rows
    print(
        f"       Official impressions: "
        f"{behavior_rows:,}"
    )

    if behavior_rows != EXPECTED_IMPRESSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPRESSIONS:,} impressions, "
            f"found {behavior_rows:,}."
        )

    print("\n[3/5] Building official predictions.txt...")
    print(
        "       Format: "
        "<impression_id> [1-based candidate positions]"
    )

    prediction_stream = PredictionStream(PREDICTIONS)

    behavior_iterator = behavior_file.iter_batches(
        batch_size=BEHAVIOR_BATCH_SIZE,
        columns=["impression_id", "article_ids_inview"],
    )

    impression_count = 0
    prediction_rows = 0
    started = time.time()

    with open(
        PREDICTIONS_TXT,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fout:

        for behavior_batch in behavior_iterator:
            behavior_df = behavior_batch.to_pandas()

            for behavior_row in behavior_df.itertuples(index=False):
                impression_id = int(
                    behavior_row.impression_id
                )

                prediction_result = (
                    prediction_stream.pop_impression()
                )

                if prediction_result is None:
                    raise RuntimeError(
                        "Prediction file ended before official "
                        f"impression {impression_id}."
                    )

                predicted_id, prediction_rows_for_impression = (
                    prediction_result
                )

                if predicted_id != impression_id:
                    raise RuntimeError(
                        "Prediction/behavior impression mismatch: "
                        f"behaviors={impression_id}, "
                        f"predictions={predicted_id}"
                    )

                inview = as_article_list(
                    behavior_row.article_ids_inview
                )

                positions = build_prediction_positions(
                    prediction_rows_for_impression,
                    inview,
                )

                # Official EB-NeRD format:
                # impression_id [position1,position2,...]
                fout.write(
                    f"{impression_id} "
                    f"[{','.join(map(str, positions))}]\n"
                )

                impression_count += 1
                prediction_rows += len(
                    prediction_rows_for_impression
                )

                if impression_count % 100_000 == 0:
                    elapsed = time.time() - started
                    rate = impression_count / max(elapsed, 1e-9)

                    print(
                        f"       {impression_count:,}/"
                        f"{behavior_rows:,} impressions "
                        f"({rate:,.0f} impressions/sec)"
                    )

    # Ensure prediction stream has no extra impressions.
    extra = prediction_stream.pop_impression()

    if extra is not None:
        raise RuntimeError(
            "Prediction file contains impressions beyond the official "
            f"test set. First extra impression: {extra[0]}"
        )

    print("\n[4/5] Validating submission...")
    print(
        f"       Submission lines written: "
        f"{impression_count:,}"
    )

    if impression_count != EXPECTED_IMPRESSIONS:
        raise RuntimeError(
            f"Submission line count mismatch: "
            f"{impression_count:,} != "
            f"{EXPECTED_IMPRESSIONS:,}"
        )

    print("       PASS")

    print("\n[5/5] Creating ZIP...")

    with zipfile.ZipFile(
        PREDICTIONS_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(
            PREDICTIONS_TXT,
            arcname="predictions.txt",
        )

    print()
    print(f"       TXT: {PREDICTIONS_TXT}")
    print(f"       ZIP: {PREDICTIONS_ZIP}")
    print()
    print("=" * 80)
    print("EB-NeRD OFFICIAL SUBMISSION BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
