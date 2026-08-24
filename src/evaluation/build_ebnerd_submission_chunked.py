from __future__ import annotations

from pathlib import Path
import time
import zipfile
import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]

PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUT = ROOT / "data/results/ebnerd/test/submission"
TXT = OUT / "predictions.txt"
ZIP = OUT / "predictions.zip"

TOTAL = 13_536_710
BEHAVIOR_BATCH = 100_000


def main():
    print("=" * 80)
    print("EB-NeRD CHUNKED SUBMISSION BUILDER")
    print("=" * 80)

    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not BEH.exists():
        raise FileNotFoundError(BEH)

    OUT.mkdir(parents=True, exist_ok=True)

    # Only the incomplete submission products are removed.
    TXT.unlink(missing_ok=True)
    ZIP.unlink(missing_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")

    pf = pq.ParquetFile(BEH)
    print(f"\nOfficial impressions: {pf.metadata.num_rows:,}")
    if pf.metadata.num_rows != TOTAL:
        raise RuntimeError(
            f"Expected {TOTAL:,} impressions, "
            f"got {pf.metadata.num_rows:,}"
        )

    print(f"Prediction file: {PRED}")
    print("Chunk size:", f"{BEHAVIOR_BATCH:,}", "impressions")
    print("\nStarting chunked conversion...")

    start = time.time()
    written = 0
    first_chunk = True

    for batch_no, batch in enumerate(
        pf.iter_batches(
            batch_size=BEHAVIOR_BATCH,
            columns=["impression_id", "article_ids_inview"],
        ),
        start=1,
    ):
        # Arrow -> pandas is only ~100k rows, not the whole dataset.
        df = batch.to_pandas()

        lo = int(df["impression_id"].min())
        hi = int(df["impression_id"].max())

        # Materialize the current behavior chunk into DuckDB.
        con.register("beh_chunk", df)

        # Crucial optimization:
        # query only the prediction range corresponding to this behavior
        # chunk. The prediction parquet is sorted by impression_id, so
        # Parquet row-group pruning avoids repeatedly reading the full 2.15GB.
        sql = f"""
        WITH candidates AS (
            SELECT
                b.impression_id,
                CAST(u.article_id AS BIGINT) AS article_id,
                CAST(u.position AS INTEGER) AS position
            FROM beh_chunk b,
            UNNEST(b.article_ids_inview)
                WITH ORDINALITY AS u(article_id, position)
        ),
        ranked AS (
            SELECT
                p.impression_id,
                c.position,
                CAST(p.rank AS INTEGER) AS model_rank
            FROM read_parquet(
                '{PRED.as_posix()}',
                hive_partitioning=false
            ) p
            INNER JOIN candidates c
              ON CAST(p.impression_id AS BIGINT) = c.impression_id
             AND CAST(p.article_id AS BIGINT) = c.article_id
            WHERE CAST(p.impression_id AS BIGINT)
                  BETWEEN {lo} AND {hi}
        )
        SELECT
            CAST(impression_id AS VARCHAR) || ' [' ||
            string_agg(
                CAST(position AS VARCHAR),
                ','
                ORDER BY model_rank, position
            ) ||
            ']' AS line
        FROM ranked
        GROUP BY impression_id
        ORDER BY impression_id
        """

        result = con.execute(sql).fetchall()

        if len(result) != len(df):
            raise RuntimeError(
                f"Chunk {batch_no}: expected {len(df):,} lines, "
                f"got {len(result):,}. "
                f"Impression range {lo}-{hi}."
            )

        mode = "w" if first_chunk else "a"
        with open(TXT, mode, encoding="utf-8", newline="\n") as f:
            for (line,) in result:
                f.write(line)
                f.write("\n")

        first_chunk = False
        written += len(result)

        elapsed = time.time() - start
        rate = written / max(elapsed, 1e-9)
        remaining = TOTAL - written
        eta = remaining / max(rate, 1e-9)

        print(
            f"       Chunk {batch_no:03d}: "
            f"{written:,}/{TOTAL:,} "
            f"({rate:,.0f} impressions/sec, "
            f"ETA {eta/60:.1f} min)"
        )

        con.unregister("beh_chunk")

    if written != TOTAL:
        raise RuntimeError(
            f"Final line count {written:,} != {TOTAL:,}"
        )

    print("\n[VALIDATION] Checking line count...")
    with open(TXT, "rb") as f:
        lines = sum(1 for _ in f)

    print(f"       Lines: {lines:,}")
    if lines != TOTAL:
        raise RuntimeError(
            f"Submission lines {lines:,} != {TOTAL:,}"
        )
    print("       PASS")

    print("\n[ZIP] Creating predictions.zip...")
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
    print("EB-NeRD SUBMISSION COMPLETE")
    print("=" * 80)

    con.close()


if __name__ == "__main__":
    main()
