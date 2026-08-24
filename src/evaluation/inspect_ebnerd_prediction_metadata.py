from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]
PRED=ROOT/"data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"

con=duckdb.connect()
print("="*80)
print("EB-NeRD PREDICTION METADATA INSPECTION")
print("="*80)

q=f"""
SELECT
  file_name,
  row_group_id,
  row_group_num_rows,
  stats_min,
  stats_max
FROM parquet_metadata('{PRED.as_posix()}')
WHERE path_in_schema='impression_id'
ORDER BY row_group_id
"""
rows=con.execute(q).fetchall()

print(f"Row-group metadata entries: {len(rows):,}")
print("\nFirst 5:")
for r in rows[:5]:
    print(r)

print("\nLast 10:")
for r in rows[-10:]:
    print(r)

# Numeric ranges, ignoring malformed/zero stats where possible.
q2=f"""
SELECT
  COUNT(*) AS groups,
  MAX(TRY_CAST(stats_max AS BIGINT)) AS max_numeric,
  MIN(TRY_CAST(NULLIF(stats_min,'0') AS BIGINT)) AS min_nonzero_numeric
FROM parquet_metadata('{PRED.as_posix()}')
WHERE path_in_schema='impression_id'
"""
print("\nOverall metadata numeric range:")
print(con.execute(q2).fetchone())

con.close()
