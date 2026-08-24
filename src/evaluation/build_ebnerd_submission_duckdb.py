from pathlib import Path
import zipfile
import time
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = PROJECT_ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEHAVIORS = PROJECT_ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/results/ebnerd/test/submission"
TXT = OUTPUT_DIR / "predictions.txt"
ZIP = OUTPUT_DIR / "predictions.zip"

EXPECTED = 13_536_710

def main():
    print("=" * 80)
    print("EB-NeRD FAST DUCKDB SUBMISSION BUILDER")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)
    if not BEHAVIORS.exists():
        raise FileNotFoundError(BEHAVIORS)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")

    print("\n[1/4] Checking input files...")
    pred_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PREDICTIONS.as_posix()}')"
    ).fetchone()[0]
    beh_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{BEHAVIORS.as_posix()}')"
    ).fetchone()[0]

    print(f"       Prediction rows: {pred_rows:,}")
    print(f"       Official impressions: {beh_rows:,}")

    if beh_rows != EXPECTED:
        raise RuntimeError(
            f"Expected {EXPECTED:,} impressions, got {beh_rows:,}"
        )

    print("\n[2/4] Converting ranked article IDs to official candidate positions...")
    print("       DuckDB vectorized execution; no Python impression loop.")

    # Keep the prediction's model rank. Join to the official in-view list,
    # then derive the original 1-based candidate position.
    #
    # DuckDB handles list UNNEST and the large parquet scan in its vectorized
    # engine. We write the final text directly from SQL.
    start = time.time()

    sql = f"""
    COPY (
        WITH pred AS (
            SELECT
                CAST(impression_id AS BIGINT) AS impression_id,
                CAST(article_id AS BIGINT) AS article_id,
                CAST(rank AS INTEGER) AS model_rank
            FROM read_parquet('{PREDICTIONS.as_posix()}')
        ),
        beh AS (
            SELECT
                CAST(impression_id AS BIGINT) AS impression_id,
                article_ids_inview
            FROM read_parquet('{BEHAVIORS.as_posix()}')
        ),
        candidates AS (
            SELECT
                b.impression_id,
                u.article_id,
                u.position
            FROM beh b,
            UNNEST(b.article_ids_inview)
                WITH ORDINALITY AS u(article_id, position)
        ),
        ranked AS (
            SELECT
                p.impression_id,
                c.position,
                p.model_rank
            FROM pred p
            INNER JOIN candidates c
                ON p.impression_id = c.impression_id
               AND p.article_id = c.article_id
        ),
        lines AS (
            SELECT
                impression_id,
                '[' ||
                string_agg(
                    CAST(position AS VARCHAR),
                    ','
                    ORDER BY model_rank, position
                ) ||
                ']' AS ranked_positions
            FROM ranked
            GROUP BY impression_id
        )
        SELECT
            CAST(impression_id AS VARCHAR) || ' ' || ranked_positions
                AS line
        FROM lines
        ORDER BY impression_id
    )
    TO '{TXT.as_posix()}'
    (FORMAT CSV, HEADER FALSE, QUOTE '');
    """

    con.execute(sql)

    elapsed = time.time() - start

    print(f"       Conversion time: {elapsed:.1f}s")

    print("\n[3/4] Validating output line count...")
    line_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{TXT.as_posix()}',
            columns={{'line':'VARCHAR'}},
            header=false,
            quote='',
            escape='\\\\'
        )
        """
    ).fetchone()[0]

    print(f"       Submission lines: {line_count:,}")

    if line_count != EXPECTED:
        raise RuntimeError(
            f"Submission line count mismatch: "
            f"{line_count:,} != {EXPECTED:,}"
        )

    print("       PASS")

    print("\n[4/4] Creating ZIP...")

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
    print("EB-NeRD FAST SUBMISSION BUILD COMPLETE")
    print("=" * 80)

    con.close()

if __name__ == "__main__":
    main()
