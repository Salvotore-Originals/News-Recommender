from __future__ import annotations

from pathlib import Path
import zipfile
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUT = ROOT / "data/results/ebnerd/test/submission"
TXT = OUT / "predictions.txt"
ZIP = OUT / "predictions.zip"

TOTAL = 13_536_710
DONE = 13_325_000
BEHAVIOR_BATCH = 50_000
PRED_BATCH = 1_000_000


def main():
    print("=" * 80)
    print("EB-NeRD EXACT-ROW TAIL SUBMISSION BUILDER")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    # ---------------------------------------------------------------
    # Verify the existing submission checkpoint.
    # ---------------------------------------------------------------
    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)

    print(f"Existing submission lines: {existing:,}")

    if existing != DONE:
        raise RuntimeError(
            f"Expected {DONE:,} existing lines, got {existing:,}. "
            "No changes made."
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # The submission file has already consumed the FIRST DONE rows of
    # the official behaviors file. Impression IDs are NOT assumed to
    # be monotonic.
    #
    # Therefore we locate the exact row boundary by row count.
    # ---------------------------------------------------------------
    beh_pf = pq.ParquetFile(BEH)

    if beh_pf.metadata.num_rows != TOTAL:
        raise RuntimeError(
            f"Expected {TOTAL:,} official behavior rows, "
            f"got {beh_pf.metadata.num_rows:,}"
        )

    print("\n[1/4] Reading exact remaining official behavior rows...")
    print(f"       Starting row: {DONE:,}")
    print(f"       Remaining:    {TOTAL - DONE:,}")

    reader = beh_pf.iter_batches(
        batch_size=BEHAVIOR_BATCH,
        columns=["impression_id", "article_ids_inview"],
    )

    skipped = 0

    # Discard exactly DONE rows.
    while skipped < DONE:
        batch = next(reader)
        skipped += len(batch)

    tail_rows = []

    # The reader is now positioned exactly at row DONE.
    for batch in reader:
        data = batch.to_pydict()
        tail_rows.extend(
            zip(
                data["impression_id"],
                data["article_ids_inview"],
            )
        )

    print(f"       Exact tail rows loaded: {len(tail_rows):,}")

    expected_tail = TOTAL - DONE

    if len(tail_rows) != expected_tail:
        raise RuntimeError(
            f"Expected {expected_tail:,} tail rows, "
            f"got {len(tail_rows):,}"
        )

    # Preserve official behavior-file order.
    tail_ids = [int(x[0]) for x in tail_rows]

    if len(set(tail_ids)) != len(tail_ids):
        raise RuntimeError(
            "Duplicate impression IDs detected in official tail."
        )

    # Article -> 1-based candidate position.
    position_lookup = {}

    for impression_id, article_ids in tail_rows:
        position_lookup[int(impression_id)] = {
            int(article_id): position
            for position, article_id
            in enumerate(article_ids, start=1)
        }

    print("       Official tail boundary verified.")
    print(
        f"       First tail impression: {tail_ids[0]}"
    )
    print(
        f"       Last tail impression:  {tail_ids[-1]}"
    )

    # ---------------------------------------------------------------
    # Stream predictions ONCE and retain only tail impressions.
    # ---------------------------------------------------------------
    print("\n[2/4] Streaming completed prediction artifact...")

    prediction_pf = pq.ParquetFile(PRED)

    ranked = {}

    for batch_no, batch in enumerate(
        prediction_pf.iter_batches(
            batch_size=PRED_BATCH,
            columns=["impression_id", "article_id", "rank"],
        ),
        start=1,
    ):
        data = batch.to_pydict()

        for impression_id, article_id, rank in zip(
            data["impression_id"],
            data["article_id"],
            data["rank"],
        ):
            impression_id = int(impression_id)

            if impression_id not in position_lookup:
                continue

            article_id = int(article_id)

            position = position_lookup[impression_id].get(
                article_id
            )

            if position is None:
                raise RuntimeError(
                    f"Prediction article {article_id} does not occur "
                    f"in official candidates for impression "
                    f"{impression_id}."
                )

            ranked.setdefault(
                impression_id,
                []
            ).append(
                (
                    int(rank),
                    position,
                )
            )

        if batch_no % 5 == 0:
            print(
                f"       Prediction batches scanned: {batch_no:,} | "
                f"tail impressions matched: "
                f"{len(ranked):,}/{expected_tail:,}"
            )

    if len(ranked) != expected_tail:
        missing = [
            impression_id
            for impression_id in tail_ids
            if impression_id not in ranked
        ]

        raise RuntimeError(
            f"Missing predictions for {len(missing):,} "
            f"tail impressions. Examples: {missing[:10]}"
        )

    print(
        f"       All {expected_tail:,} tail impressions matched."
    )

    # ---------------------------------------------------------------
    # Append EXACTLY in official behavior-file order.
    # ---------------------------------------------------------------
    print("\n[3/4] Appending final submission rows...")

    with open(
        TXT,
        "a",
        encoding="utf-8",
        newline="\n",
    ) as output:

        for index, impression_id in enumerate(
            tail_ids,
            start=1,
        ):

            candidates = ranked[impression_id]

            candidates.sort(
                key=lambda x: (
                    x[0],  # model rank
                    x[1],  # deterministic position tie-break
                )
            )

            positions = ",".join(
                str(position)
                for _, position in candidates
            )

            output.write(
                f"{impression_id} [{positions}]\n"
            )

            if index % 25_000 == 0:
                print(
                    f"       Appended: "
                    f"{DONE + index:,}/{TOTAL:,}"
                )

    # ---------------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------------
    print("\n[4/4] Final validation...")

    with open(TXT, "rb") as f:
        final_count = sum(1 for _ in f)

    print(f"       Final submission lines: {final_count:,}")

    if final_count != TOTAL:
        raise RuntimeError(
            f"FINAL VALIDATION FAILED: "
            f"{final_count:,} != {TOTAL:,}"
        )

    print("       LINE COUNT: PASS")

    # Verify the final line belongs to the official final behavior row.
    final_expected_id = tail_ids[-1]

    with open(TXT, "rb") as f:
        f.seek(0, 2)
        position = f.tell()
        f.seek(max(0, position - 1000))
        final_text = f.read().decode(
            "utf-8",
            errors="replace",
        )

    final_line = final_text.strip().splitlines()[-1]
    actual_final_id = int(
        final_line.split(" ", 1)[0]
    )

    print(
        f"       Final official impression: "
        f"{final_expected_id}"
    )
    print(
        f"       Final submission impression: "
        f"{actual_final_id}"
    )

    if actual_final_id != final_expected_id:
        raise RuntimeError(
            "FINAL IMPRESSION ORDER VALIDATION FAILED."
        )

    print("       ORDER: PASS")

    # ---------------------------------------------------------------
    # ZIP
    # ---------------------------------------------------------------
    print("\nCreating final predictions.zip...")

    ZIP.unlink(missing_ok=True)

    with zipfile.ZipFile(
        ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:

        archive.write(
            TXT,
            arcname="predictions.txt",
        )

    print(f"TXT: {TXT}")
    print(f"ZIP: {ZIP}")

    print("\n" + "=" * 80)
    print("EB-NeRD FINAL SUBMISSION READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
