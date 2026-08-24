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
BEHAVIOR_BATCH = 25_000
PRED_BATCH = 1_000_000


def main():
    print("=" * 80)
    print("EB-NeRD FINAL SUBMISSION — ACTUAL OFFICIAL BEHAVIOR COUNT")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)

    print(f"Existing submission lines: {existing:,}")
    if existing != DONE:
        raise RuntimeError(
            f"Expected {DONE:,} existing lines; got {existing:,}. "
            "No changes made."
        )

    beh_pf = pq.ParquetFile(BEH)
    total = beh_pf.metadata.num_rows

    print(f"Official behaviors.parquet rows: {total:,}")

    if total < DONE:
        raise RuntimeError(
            f"Official behavior file has only {total:,} rows, "
            f"less than the existing {DONE:,} submission lines."
        )

    remaining = total - DONE
    print(f"Remaining official rows: {remaining:,}")

    # This is the key correction:
    # do NOT hard-code 13,536,710. The actual behavior parquet is the
    # authoritative row source for the submission ordering.
    reader = beh_pf.iter_batches(
        batch_size=BEHAVIOR_BATCH,
        columns=["impression_id", "article_ids_inview"],
    )

    skipped = 0
    while skipped < DONE:
        batch = next(reader)
        skipped += len(batch)
        if skipped > DONE:
            raise RuntimeError(
                f"Internal row-boundary error: skipped {skipped:,} "
                f"rows while target is {DONE:,}."
            )

    tail_rows = []
    for batch in reader:
        d = batch.to_pydict()
        tail_rows.extend(
            zip(d["impression_id"], d["article_ids_inview"])
        )

    print(f"Exact tail rows loaded: {len(tail_rows):,}")

    if len(tail_rows) != remaining:
        raise RuntimeError(
            f"Tail mismatch: expected {remaining:,}, "
            f"got {len(tail_rows):,}"
        )

    tail_ids = [int(x[0]) for x in tail_rows]

    if len(set(tail_ids)) != len(tail_ids):
        raise RuntimeError("Duplicate impression IDs in official tail.")

    position_lookup = {}
    for imp, articles in tail_rows:
        position_lookup[int(imp)] = {
            int(article): pos
            for pos, article in enumerate(articles, start=1)
        }

    print(
        f"First tail impression: {tail_ids[0]}"
    )
    print(
        f"Last tail impression:  {tail_ids[-1]}"
    )

    print("\n[2/4] Streaming prediction artifact once...")
    pred_pf = pq.ParquetFile(PRED)
    ranked = {}

    for batch_no, batch in enumerate(
        pred_pf.iter_batches(
            batch_size=PRED_BATCH,
            columns=["impression_id", "article_id", "rank"],
        ),
        start=1,
    ):
        d = batch.to_pydict()

        for imp_raw, article_raw, rank_raw in zip(
            d["impression_id"],
            d["article_id"],
            d["rank"],
        ):
            imp = int(imp_raw)

            if imp not in position_lookup:
                continue

            article = int(article_raw)
            position = position_lookup[imp].get(article)

            if position is None:
                raise RuntimeError(
                    f"Article {article} is not in official candidates "
                    f"for impression {imp}."
                )

            ranked.setdefault(imp, []).append(
                (int(rank_raw), position)
            )

        if batch_no % 5 == 0:
            print(
                f"       Prediction batches scanned: {batch_no:,} | "
                f"matched: {len(ranked):,}/{remaining:,}"
            )

    missing = [imp for imp in tail_ids if imp not in ranked]
    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing):,} impressions. "
            f"Examples: {missing[:10]}"
        )

    print(f"       All {remaining:,} tail impressions matched.")

    print("\n[3/4] Appending final rows in official order...")
    with open(TXT, "a", encoding="utf-8", newline="\n") as f:
        for i, imp in enumerate(tail_ids, start=1):
            rows = ranked[imp]
            rows.sort(key=lambda x: (x[0], x[1]))
            positions = ",".join(str(pos) for _, pos in rows)
            f.write(f"{imp} [{positions}]\n")

            if i % 25_000 == 0:
                print(f"       Appended: {DONE+i:,}/{total:,}")

    with open(TXT, "rb") as f:
        final_count = sum(1 for _ in f)

    print(f"\nFinal submission lines: {final_count:,}")

    if final_count != total:
        raise RuntimeError(
            f"Final count mismatch: {final_count:,} != {total:,}"
        )

    print("LINE COUNT: PASS")

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(
        ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as z:
        z.write(TXT, arcname="predictions.txt")

    print(f"TXT: {TXT}")
    print(f"ZIP: {ZIP}")
    print("=" * 80)
    print("EB-NeRD FINAL SUBMISSION READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
