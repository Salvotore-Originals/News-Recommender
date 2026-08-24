from pathlib import Path
import time, zipfile, duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUT = ROOT / "data/results/ebnerd/test/submission"
TXT = OUT / "predictions.txt"
ZIP = OUT / "predictions.zip"
TOTAL = 13_536_710
ALREADY = 13_300_000
BATCH = 25_000

def main():
    print("="*80)
    print("EB-NeRD FINAL TAIL SUBMISSION BUILDER")
    print("="*80)

    if not PRED.exists() or not BEH.exists() or not TXT.exists():
        raise FileNotFoundError("Required prediction/behavior/submission file missing.")

    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)
    print(f"Existing submission lines: {existing:,}")
    if existing != ALREADY:
        raise RuntimeError(f"Expected {ALREADY:,} existing lines, found {existing:,}.")

    pf = pq.ParquetFile(BEH)
    if pf.metadata.num_rows != TOTAL:
        raise RuntimeError(f"Expected {TOTAL:,} behaviors, got {pf.metadata.num_rows:,}.")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")

    reader = pf.iter_batches(batch_size=BATCH,
                             columns=["impression_id","article_ids_inview"])
    skipped = 0
    while skipped < ALREADY:
        b = next(reader)
        skipped += len(b)

    remaining = TOTAL - ALREADY
    written = 0
    start = time.time()

    print(f"Remaining impressions: {remaining:,}")

    with open(TXT, "a", encoding="utf-8", newline="\n") as fout:
        for n, batch in enumerate(reader, 1):
            df = batch.to_pandas()
            con.register("beh_tail", df)
            lo = int(df["impression_id"].min())
            hi = int(df["impression_id"].max())

            sql = f"""
            WITH candidates AS (
                SELECT b.impression_id,
                       CAST(u.article_id AS BIGINT) article_id,
                       CAST(u.position AS INTEGER) AS candidate_position
                FROM beh_tail b,
                UNNEST(b.article_ids_inview)
                  WITH ORDINALITY AS u(article_id, position)
            ),
            ranked AS (
                SELECT p.impression_id, c.candidate_position,
                       CAST(p.rank AS INTEGER) model_rank
                FROM read_parquet('{PRED.as_posix()}') p
                JOIN candidates c
                  ON CAST(p.impression_id AS BIGINT)=c.impression_id
                 AND CAST(p.article_id AS BIGINT)=c.article_id
                WHERE CAST(p.impression_id AS BIGINT) BETWEEN {lo} AND {hi}
            )
            SELECT CAST(impression_id AS VARCHAR) || ' [' ||
                   string_agg(CAST(candidate_position AS VARCHAR), ','
                              ORDER BY model_rank, candidate_position) || ']'
            FROM ranked
            GROUP BY impression_id
            ORDER BY impression_id
            """
            rows = con.execute(sql).fetchall()
            if len(rows) != len(df):
                raise RuntimeError(f"Chunk mismatch: expected {len(df):,}, got {len(rows):,}")
            for (line,) in rows:
                fout.write(line + "\n")
            con.unregister("beh_tail")
            written += len(rows)
            rate = written / max(time.time()-start, 1e-9)
            print(f"Tail {written:,}/{remaining:,} ({rate:,.0f}/sec)")

    with open(TXT, "rb") as f:
        total_lines = sum(1 for _ in f)
    print(f"Final lines: {total_lines:,}")
    if total_lines != TOTAL:
        raise RuntimeError(f"Expected {TOTAL:,}, got {total_lines:,}")

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(TXT, arcname="predictions.txt")

    print(f"TXT: {TXT}")
    print(f"ZIP: {ZIP}")
    print("EB-NeRD FINAL SUBMISSION COMPLETE")
    con.close()

if __name__ == "__main__":
    main()
