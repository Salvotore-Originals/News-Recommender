from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]
PRED=ROOT/"data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"

ids = [
"573647300","573647310","573647318","573647320","573647321",
"573647322","573647323","573647325","573647348","573647349"
]

con=duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='4GB'")
vals=",".join("'" + x + "'" for x in ids)

print("="*80)
print("CHECKING EXACT TAIL IMPRESSION IDs IN PREDICTION ARTIFACT")
print("="*80)
print("Searching 10 official tail IDs...")

q=f"""
SELECT impression_id, COUNT(*) AS prediction_rows
FROM read_parquet('{PRED.as_posix()}')
WHERE impression_id IN ({vals})
GROUP BY impression_id
ORDER BY impression_id
"""
rows=con.execute(q).fetchall()

print("\nMATCHES:")
if rows:
    for r in rows:
        print(r)
else:
    print("NO MATCHES")

print(f"\nMatched IDs: {len(rows)}/{len(ids)}")
print("="*80)
con.close()
