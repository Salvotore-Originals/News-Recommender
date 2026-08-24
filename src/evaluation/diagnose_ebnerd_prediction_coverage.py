from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH = ROOT / "data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"

DONE = 13_325_000
SAMPLE = 20

def main():
    print("=" * 80)
    print("EB-NeRD PREDICTION COVERAGE DIAGNOSTIC")
    print("=" * 80)

    bp = pq.ParquetFile(BEH)
    pp = pq.ParquetFile(PRED)

    print(f"Behavior rows:   {bp.metadata.num_rows:,}")
    print(f"Prediction rows: {pp.metadata.num_rows:,}")
    print(f"Prediction row groups: {pp.metadata.num_row_groups:,}")

    # Read exactly the boundary area: last 20 already-submitted rows and
    # first 20 rows after the 13,325,000-row checkpoint.
    reader = bp.iter_batches(
        batch_size=25_000,
        columns=["impression_id", "article_ids_inview"],
    )

    skipped = 0
    boundary = []

    while skipped < DONE:
        b = next(reader)
        skipped += len(b)

    # Last batch ended exactly at the checkpoint because 25k divides 13.325m.
    # Read the first 20 rows of the remaining official tail.
    b = next(reader)
    d = b.to_pydict()

    tail_ids = [str(x) for x in d["impression_id"][:SAMPLE]]
    print("\nFirst 20 IDs AFTER submission checkpoint:")
    print(tail_ids)

    # Also get IDs from the immediately preceding batch.
    # Re-read from start only for the checkpoint batch.
    reader2 = bp.iter_batches(
        batch_size=25_000,
        columns=["impression_id"],
    )
    prev = None
    for _ in range(DONE // 25_000):
        prev = next(reader2)

    prev_ids = [str(x) for x in prev["impression_id"][-SAMPLE:]]
    print("\nLast 20 IDs ALREADY represented by submission checkpoint:")
    print(prev_ids)

    # Read the first prediction row group only. This establishes physical
    # ordering without scanning the entire 205.9M-row artifact.
    first_rg = pp.read_row_group(
        0,
        columns=["impression_id", "article_id", "rank"],
    ).to_pydict()

    last_rg = pp.read_row_group(
        pp.metadata.num_row_groups - 1,
        columns=["impression_id", "article_id", "rank"],
    ).to_pydict()

    print("\nPrediction first-row-group impression ID range:")
    print("  first:", first_rg["impression_id"][0])
    print("  last: ", first_rg["impression_id"][-1])

    print("\nPrediction last-row-group impression ID range:")
    print("  first:", last_rg["impression_id"][0])
    print("  last: ", last_rg["impression_id"][-1])

    pred_first_ids = set(str(x) for x in first_rg["impression_id"])
    pred_last_ids = set(str(x) for x in last_rg["impression_id"])

    print("\nBoundary sample membership:")
    for x in tail_ids[:10]:
        print(f"  tail {x}: first_RG={x in pred_first_ids}, last_RG={x in pred_last_ids}")

    print("\nCheckpoint sample membership:")
    for x in prev_ids[:10]:
        print(f"  prev {x}: first_RG={x in pred_first_ids}, last_RG={x in pred_last_ids}")

    print("\nIMPORTANT:")
    print("This diagnostic does NOT modify predictions.txt.")
    print("=" * 80)

if __name__ == "__main__":
    main()
