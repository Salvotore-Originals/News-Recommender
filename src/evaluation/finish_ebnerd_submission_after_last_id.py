from pathlib import Path
import zipfile
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUT = ROOT / "data/results/ebnerd/test/submission"
TXT = OUT / "predictions.txt"
ZIP = OUT / "predictions.zip"

TOTAL = 13_536_710
EXPECTED_EXISTING = 13_325_000
BATCH = 50_000

def main():
    print("=" * 80)
    print("EB-NeRD FINAL SUBMISSION — LAST-ID TAIL")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    # Get exact current checkpoint.
    with open(TXT, "rb") as f:
        lines = f.readlines()

    existing = len(lines)
    if existing != EXPECTED_EXISTING:
        raise RuntimeError(
            f"Expected {EXPECTED_EXISTING:,} existing lines, got {existing:,}"
        )

    last = lines[-1].decode("utf-8").strip()
    last_id = int(last.split(" ", 1)[0])

    print(f"Existing lines: {existing:,}")
    print(f"Last written impression_id: {last_id}")

    # Load only the official behavior tail whose impression_id is greater
    # than the last written ID. This avoids assuming row offsets.
    print("\n[1/3] Finding remaining official impressions...")

    beh_pf = pq.ParquetFile(BEH)
    tail_frames = []

    for batch in beh_pf.iter_batches(
        batch_size=BATCH,
        columns=["impression_id", "article_ids_inview"],
    ):
        df = batch.to_pandas()
        df["impression_id"] = df["impression_id"].astype("int64")
        part = df[df["impression_id"] > last_id]
        if not part.empty:
            tail_frames.append(part)

    if not tail_frames:
        raise RuntimeError(
            "No official impressions found after the current last impression ID."
        )

    tail = pd.concat(tail_frames, ignore_index=True)

    # Guard against duplicate official IDs.
    if tail["impression_id"].duplicated().any():
        dup = int(tail["impression_id"].duplicated().sum())
        raise RuntimeError(f"Duplicate official impression IDs in tail: {dup}")

    print(f"       Remaining official impressions: {len(tail):,}")

    expected_remaining = TOTAL - EXPECTED_EXISTING
    if len(tail) != expected_remaining:
        raise RuntimeError(
            f"Expected {expected_remaining:,} remaining impressions, "
            f"but found {len(tail):,}. "
            f"Current last ID is {last_id}."
        )

    # Build article -> official 1-based position lookup for only the tail.
    print("\n[2/3] Matching predictions for the final tail...")

    lookup = {}
    for imp, articles in zip(
        tail["impression_id"].tolist(),
        tail["article_ids_inview"].tolist(),
    ):
        lookup[int(imp)] = {
            int(article): pos
            for pos, article in enumerate(articles, start=1)
        }

    # Stream prediction parquet once. Only rows for the tail are retained.
    pred_pf = pq.ParquetFile(PRED)

    grouped = {}
    matched_rows = 0

    for batch_no, batch in enumerate(
        pred_pf.iter_batches(
            batch_size=1_000_000,
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

            if imp not in lookup:
                continue

            article = int(article_raw)
            pos = lookup[imp].get(article)

            if pos is None:
                raise RuntimeError(
                    f"Prediction article {article} not found in "
                    f"official inview for impression {imp}"
                )

            grouped.setdefault(imp, []).append(
                (int(rank_raw), pos)
            )
            matched_rows += 1

        print(
            f"       Prediction batches scanned: {batch_no:,} | "
            f"tail impressions matched: {len(grouped):,}/{len(tail):,}"
        )

    if len(grouped) != len(tail):
        missing = set(lookup) - set(grouped)
        sample = list(missing)[:10]
        raise RuntimeError(
            f"Missing predictions for {len(missing):,} tail impressions. "
            f"Examples: {sample}"
        )

    # Append in official behavior order, not numeric ID order.
    print("\n[3/3] Appending and validating...")
    with open(TXT, "a", encoding="utf-8", newline="\n") as f:
        for imp in tail["impression_id"].tolist():
            rows = grouped[int(imp)]
            rows.sort(key=lambda x: (x[0], x[1]))
            positions = ",".join(str(pos) for _, pos in rows)
            f.write(f"{int(imp)} [{positions}]\n")

    with open(TXT, "rb") as f:
        final_count = sum(1 for _ in f)

    print(f"       Final lines: {final_count:,}")

    if final_count != TOTAL:
        raise RuntimeError(
            f"Final line count mismatch: {final_count:,} != {TOTAL:,}"
        )

    print("       VALIDATION PASSED")

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(
        ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as z:
        z.write(TXT, arcname="predictions.txt")

    print(f"       ZIP: {ZIP}")
    print("=" * 80)
    print("EB-NeRD FINAL SUBMISSION READY")
    print("=" * 80)

if __name__ == "__main__":
    main()
