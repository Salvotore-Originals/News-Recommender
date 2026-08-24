from pathlib import Path
import zipfile
import duckdb

ROOT = Path(__file__).resolve().parents[2]

PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
TXT = ROOT / "data/results/ebnerd/test/submission/predictions.txt"
ZIP = ROOT / "data/results/ebnerd/test/submission/predictions.zip"

DONE = 13_325_000
TOTAL = 13_536_710
BATCH = 25_000

def main():
    print("=" * 80)
    print("EB-NeRD FINAL DUCKDB TAIL CONVERTER")
    print("=" * 80)

    for p in (PRED, BEH, TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    with open(TXT, "rb") as f:
        existing = sum(1 for _ in f)

    print(f"Existing submission lines: {existing:,}")

    if existing != DONE:
        raise RuntimeError(
            f"Expected {DONE:,} existing lines; found {existing:,}. "
            "Nothing changed."
        )

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA preserve_insertion_order=false")

    beh = BEH.as_posix().replace("\\", "/")
    pred = PRED.as_posix().replace("\\", "/")

    # Materialize ONLY the final 211,710 official behavior rows.
    # DuckDB row_number() is used against the official behavior file,
    # so impression_id ordering/duplication is irrelevant.
    print("\n[1/4] Materializing exact official tail...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tail AS
        SELECT
            CAST(impression_id AS VARCHAR) AS impression_id,
            article_ids_inview
        FROM read_parquet('{beh}')
        QUALIFY row_number() OVER () > {DONE}
    """)

    tail_count = con.execute(
        "SELECT COUNT(*) FROM tail"
    ).fetchone()[0]

    print(f"       Tail rows: {tail_count:,}")

    if tail_count != TOTAL - DONE:
        raise RuntimeError(
            f"Expected {TOTAL-DONE:,} tail rows; got {tail_count:,}"
        )

    unique_count = con.execute(
        "SELECT COUNT(DISTINCT impression_id) FROM tail"
    ).fetchone()[0]

    print(f"       Unique tail impressions: {unique_count:,}")

    # Build one official candidate-position row per unique impression.
    # Duplicate behavior rows are intentionally deduplicated only when their
    # candidate lists are identical.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE candidates AS
        SELECT
            impression_id,
            article_id,
            candidate_position
        FROM (
            SELECT
                impression_id,
                UNNEST(article_ids_inview) AS article_id,
                generate_subscripts(article_ids_inview, 1)
                    AS candidate_position
            FROM (
                SELECT DISTINCT
                    impression_id,
                    article_ids_inview
                FROM tail
            )
        )
    """)

    candidate_count = con.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"       Candidate mappings: {candidate_count:,}")

    # IMPORTANT:
    # prediction impression_id/article_id are LARGE_STRING. Both sides are
    # explicitly VARCHAR. The join is restricted to the 11,711 tail IDs,
    # not the entire official test.
    print("\n[2/4] Joining tail predictions...")
    print("       DuckDB vectorized join; no Python prediction scan.")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tail_ranked AS
        SELECT
            p.impression_id,
            c.candidate_position,
            CAST(p.rank AS BIGINT) AS model_rank
        FROM read_parquet('{pred}') p
        INNER JOIN candidates c
            ON CAST(p.impression_id AS VARCHAR) = c.impression_id
           AND CAST(p.article_id AS BIGINT) = CAST(c.article_id AS BIGINT)
    """)

    matched = con.execute(
        "SELECT COUNT(DISTINCT impression_id) FROM tail_ranked"
    ).fetchone()[0]

    print(
        f"       Matched impressions: {matched:,}/{unique_count:,}"
    )

    if matched != unique_count:
        missing = con.execute("""
            SELECT impression_id
            FROM candidates
            EXCEPT
            SELECT DISTINCT impression_id
            FROM tail_ranked
            LIMIT 20
        """).fetchall()

        raise RuntimeError(
            f"Missing predictions for "
            f"{unique_count-matched:,} impressions. "
            f"Examples: {missing}"
        )

    # Produce one line per unique impression, ordered by the model rank.
    print("\n[3/4] Generating final tail lines...")

    rows = con.execute("""
        SELECT
            impression_id,
            '[' ||
            string_agg(
                CAST(candidate_position AS VARCHAR),
                ','
                ORDER BY model_rank, candidate_position
            ) ||
            ']' AS ranking
        FROM tail_ranked
        GROUP BY impression_id
        ORDER BY MIN(CAST(impression_id AS UBIGINT))
    """).fetchall()

    print(f"       Lines generated: {len(rows):,}")

    with open(TXT, "a", encoding="utf-8", newline="\n") as f:
        for impression_id, ranking in rows:
            f.write(f"{impression_id} {ranking}\n")

    # Final checks.
    print("\n[4/4] Final validation...")

    with open(TXT, "rb") as f:
        final_lines = sum(1 for _ in f)

    expected = DONE + unique_count

    print(f"       Final lines: {final_lines:,}")
    print(f"       Expected:    {expected:,}")

    if final_lines != expected:
        raise RuntimeError(
            f"Final submission count mismatch: "
            f"{final_lines:,} != {expected:,}"
        )

    # Validate every generated line has an ID and a bracketed ranking.
    with open(TXT, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if " [" not in line or not line.endswith("]"):
                raise RuntimeError(
                    f"Malformed submission line at {line_no}: {line[:100]}"
                )

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

    con.close()

if __name__ == "__main__":
    main()
