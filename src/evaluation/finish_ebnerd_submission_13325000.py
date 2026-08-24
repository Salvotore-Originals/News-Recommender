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
BEHAVIOR_BATCH = 25_000
PRED_BATCH = 1_000_000


def main():
    print("=" * 80)
    print("EB-NeRD FINAL STREAMING SUBMISSION BUILDER")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    # Confirm existing submission checkpoint.
    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)

    print(f"Existing lines: {existing:,}")
    if existing != DONE:
        raise RuntimeError(
            f"Expected {DONE:,} existing lines; found {existing:,}. "
            "No changes made."
        )

    behavior_pf = pq.ParquetFile(BEH)
    if behavior_pf.metadata.num_rows != TOTAL:
        raise RuntimeError(
            f"Expected {TOTAL:,} behavior rows; "
            f"got {behavior_pf.metadata.num_rows:,}"
        )

    # ------------------------------------------------------------------
    # Build only the candidate-position lookup for the remaining tail.
    # This is small enough to hold in memory and avoids the enormous
    # DuckDB many-to-many temporary relation that caused the 58.9 GiB
    # temp-directory failure.
    # ------------------------------------------------------------------
    print("\n[1/4] Loading final 211,710 official impressions...")

    reader = behavior_pf.iter_batches(
        batch_size=BEHAVIOR_BATCH,
        columns=["impression_id", "article_ids_inview"],
    )

    skipped = 0
    while skipped < DONE:
        batch = next(reader)
        skipped += len(batch)

    tail = {}

    for batch in reader:
        data = batch.to_pydict()
        ids = data["impression_id"]
        inviews = data["article_ids_inview"]

        for imp, articles in zip(ids, inviews):
            imp = int(imp)
            # article_id -> 1-based official candidate position
            tail[imp] = {
                int(article): pos
                for pos, article in enumerate(articles, start=1)
            }

    remaining = len(tail)
    print(f"       Tail impressions loaded: {remaining:,}")

    if remaining != TOTAL - DONE:
        raise RuntimeError(
            f"Expected {TOTAL-DONE:,} tail impressions; got {remaining:,}"
        )

    tail_ids = set(tail)

    # ------------------------------------------------------------------
    # Stream the prediction parquet ONCE.
    # No DuckDB join and no huge temporary files.
    #
    # Predictions were generated ordered by impression_id/rank. We retain
    # that ordering in the final output by collecting each impression's
    # matching rows and emitting it when the impression changes.
    # ------------------------------------------------------------------
    print("\n[2/4] Streaming prediction artifact once...")
    pred_pf = pq.ParquetFile(PRED)

    written = 0
    current_imp = None
    current_rows = []

    def emit(imp, rows, fout):
        if imp is None:
            return 0
        rows.sort(key=lambda x: (x[0], x[1]))
        positions = [str(position) for _, position in rows]
        fout.write(
            str(imp) + " [" + ",".join(positions) + "]\n"
        )
        return 1

    with open(TXT, "a", encoding="utf-8", newline="\n") as fout:
        for batch_no, batch in enumerate(
            pred_pf.iter_batches(
                batch_size=PRED_BATCH,
                columns=["impression_id", "article_id", "rank"],
            ),
            start=1,
        ):
            data = batch.to_pydict()

            for imp_raw, article_raw, rank_raw in zip(
                data["impression_id"],
                data["article_id"],
                data["rank"],
            ):
                imp = int(imp_raw)

                if imp not in tail_ids:
                    continue

                article = int(article_raw)
                position = tail[imp].get(article)

                if position is None:
                    raise RuntimeError(
                        f"Prediction article {article} is not present "
                        f"in official inview list for impression {imp}."
                    )

                if current_imp is None:
                    current_imp = imp

                if imp != current_imp:
                    written += emit(
                        current_imp,
                        current_rows,
                        fout,
                    )
                    current_rows = []
                    current_imp = imp

                current_rows.append(
                    (int(rank_raw), position)
                )

            if batch_no % 5 == 0:
                print(
                    f"       Prediction batches scanned: {batch_no:,} | "
                    f"tail impressions written: {written:,}/{remaining:,}"
                )

        if current_imp is not None:
            written += emit(
                current_imp,
                current_rows,
                fout,
            )

    print(f"       Tail impressions written: {written:,}")

    if written != remaining:
        raise RuntimeError(
            f"Expected {remaining:,} tail lines; wrote {written:,}"
        )

    # ------------------------------------------------------------------
    # Final validation.
    # ------------------------------------------------------------------
    print("\n[3/4] Final validation...")
    with open(TXT, "rb") as f:
        final_lines = sum(1 for _ in f)

    print(f"       Final lines: {final_lines:,}")

    if final_lines != TOTAL:
        raise RuntimeError(
            f"Expected {TOTAL:,}; got {final_lines:,}"
        )

    print("       PASS")

    # ------------------------------------------------------------------
    # ZIP.
    # ------------------------------------------------------------------
    print("\n[4/4] Creating final ZIP...")
    ZIP.unlink(missing_ok=True)

    with zipfile.ZipFile(
        ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as z:
        z.write(TXT, arcname="predictions.txt")

    print(f"       TXT: {TXT}")
    print(f"       ZIP: {ZIP}")

    print("\n" + "=" * 80)
    print("EB-NeRD FINAL SUBMISSION READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
