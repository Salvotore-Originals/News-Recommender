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

DONE = 13_325_000
TOTAL = 13_536_710
BEHAVIOR_BATCH = 25_000
PRED_BATCH = 1_000_000


def main():
    print("=" * 80)
    print("EB-NeRD FINAL VERIFIED TAIL SUBMISSION BUILDER")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)

    print(f"Existing submission lines: {existing:,}")

    if existing != DONE:
        raise RuntimeError(
            f"Expected exactly {DONE:,} existing lines; "
            f"found {existing:,}. Nothing changed."
        )

    beh_pf = pq.ParquetFile(BEH)

    if beh_pf.metadata.num_rows != TOTAL:
        raise RuntimeError(
            f"Expected {TOTAL:,} official behavior rows; "
            f"found {beh_pf.metadata.num_rows:,}."
        )

    # ---------------------------------------------------------------
    # Read EXACTLY the final 211,710 behavior rows.
    # The 25,000 batch size divides 13,325,000 exactly.
    # ---------------------------------------------------------------
    print("\n[1/4] Reading exact official tail...")

    reader = beh_pf.iter_batches(
        batch_size=BEHAVIOR_BATCH,
        columns=["impression_id", "article_ids_inview"],
    )

    skipped = 0

    while skipped < DONE:
        batch = next(reader)
        skipped += len(batch)

    if skipped != DONE:
        raise RuntimeError(
            f"Boundary error: positioned at {skipped:,}, "
            f"expected {DONE:,}."
        )

    tail = []

    for batch in reader:
        data = batch.to_pydict()

        for impression_id, articles in zip(
            data["impression_id"],
            data["article_ids_inview"],
        ):
            tail.append(
                (
                    str(int(impression_id)),
                    tuple(int(x) for x in articles),
                )
            )

    print(f"       Tail rows: {len(tail):,}")

    if len(tail) != TOTAL - DONE:
        raise RuntimeError(
            f"Expected {TOTAL-DONE:,} tail rows; "
            f"found {len(tail):,}."
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # impression_id can repeat in the raw behavior file.
    # Build candidate-position maps for each occurrence, but the
    # prediction artifact is keyed by impression_id/article_id.
    #
    # If repeated rows have identical candidate lists, they represent
    # the same official candidate ordering and only one output line is
    # permitted per impression ID.
    # ---------------------------------------------------------------
    candidate_lists = {}

    for impression_id, articles in tail:
        previous = candidate_lists.get(impression_id)

        if previous is not None and previous != articles:
            raise RuntimeError(
                f"Conflicting candidate lists for impression "
                f"{impression_id}."
            )

        candidate_lists[impression_id] = articles

    tail_ids = list(candidate_lists.keys())

    print(
        f"       Unique tail impression IDs: "
        f"{len(tail_ids):,}"
    )

    duplicate_rows = len(tail) - len(tail_ids)

    print(
        f"       Repeated impression rows: "
        f"{duplicate_rows:,}"
    )

    # ---------------------------------------------------------------
    # Build official candidate-position lookup.
    # ---------------------------------------------------------------
    positions = {
        impression_id: {
            str(article_id): position
            for position, article_id in enumerate(
                articles,
                start=1,
            )
        }
        for impression_id, articles in candidate_lists.items()
    }

    # ---------------------------------------------------------------
    # Stream the prediction artifact once.
    #
    # Prediction IDs are LARGE_STRING, so they are deliberately compared
    # as strings. This is the key verified type difference.
    # ---------------------------------------------------------------
    print("\n[2/4] Streaming prediction artifact...")

    pred_pf = pq.ParquetFile(PRED)

    ranked = {}

    for batch_number, batch in enumerate(
        pred_pf.iter_batches(
            batch_size=PRED_BATCH,
            columns=[
                "impression_id",
                "article_id",
                "rank",
            ],
        ),
        start=1,
    ):
        data = batch.to_pydict()

        for impression_id_raw, article_id_raw, rank_raw in zip(
            data["impression_id"],
            data["article_id"],
            data["rank"],
        ):
            impression_id = str(impression_id_raw)

            if impression_id not in positions:
                continue

            article_id = str(article_id_raw)

            candidate_position = positions[
                impression_id
            ].get(article_id)

            if candidate_position is None:
                raise RuntimeError(
                    f"Prediction article {article_id} is not in "
                    f"official candidates for impression "
                    f"{impression_id}."
                )

            ranked.setdefault(
                impression_id,
                [],
            ).append(
                (
                    int(rank_raw),
                    candidate_position,
                )
            )

        if batch_number % 10 == 0:
            print(
                f"       Prediction batches: {batch_number:,} | "
                f"matched impressions: "
                f"{len(ranked):,}/{len(tail_ids):,}"
            )

    missing = [
        impression_id
        for impression_id in tail_ids
        if impression_id not in ranked
    ]

    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing):,} tail "
            f"impressions. Examples: {missing[:20]}"
        )

    print(
        f"       All {len(tail_ids):,} unique tail impressions matched."
    )

    # ---------------------------------------------------------------
    # Append one line per unique official impression ID.
    # ---------------------------------------------------------------
    print("\n[3/4] Appending final tail...")

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
            rows = ranked[impression_id]

            rows.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            rank_positions = ",".join(
                str(position)
                for _, position in rows
            )

            output.write(
                f"{impression_id} [{rank_positions}]\n"
            )

            if index % 25_000 == 0:
                print(
                    f"       Written unique impressions: "
                    f"{index:,}/{len(tail_ids):,}"
                )

    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------
    print("\n[4/4] Final validation...")

    with open(TXT, "rb") as f:
        final_lines = sum(1 for _ in f)

    expected_final = DONE + len(tail_ids)

    print(f"       Final lines: {final_lines:,}")
    print(f"       Expected:    {expected_final:,}")

    if final_lines != expected_final:
        raise RuntimeError(
            f"Final line count mismatch: "
            f"{final_lines:,} != {expected_final:,}"
        )

    print("       LINE COUNT: PASS")

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

    print(f"\nTXT: {TXT}")
    print(f"ZIP: {ZIP}")

    print("\n" + "=" * 80)
    print("EB-NeRD FINAL SUBMISSION CREATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
