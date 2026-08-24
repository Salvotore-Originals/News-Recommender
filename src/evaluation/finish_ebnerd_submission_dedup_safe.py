from pathlib import Path
import zipfile
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
PRED=ROOT/"data/results/ebnerd/test/ebnerd_test_hybrid_predictions.parquet"
BEH=ROOT/"data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test/behaviors.parquet"
OUT=ROOT/"data/results/ebnerd/test/submission"
TXT=OUT/"predictions.txt"
ZIP=OUT/"predictions.zip"

DONE=13_325_000
BATCH=25_000

def main():
    print("="*80)
    print("EB-NeRD FINAL SUBMISSION — DUPLICATE-SAFE TAIL")
    print("="*80)

    if not all(p.exists() for p in (PRED,BEH,TXT)):
        raise FileNotFoundError("Prediction, behavior, or submission file missing.")

    with open(TXT,"rb") as f:
        existing=sum(1 for _ in f)
    if existing!=DONE:
        raise RuntimeError(f"Expected {DONE:,} existing lines; found {existing:,}.")

    pf=pq.ParquetFile(BEH)
    total=pf.metadata.num_rows
    print(f"Official behavior rows: {total:,}")
    print(f"Existing submission rows: {existing:,}")
    print(f"Remaining behavior rows: {total-existing:,}")

    # Exact row tail. Duplicated impression IDs are retained only if their
    # candidate lists differ; identical duplicate rows are collapsed because
    # an official submission cannot contain duplicate impression IDs.
    reader=pf.iter_batches(batch_size=BATCH,
                           columns=["impression_id","article_ids_inview"])
    skipped=0
    while skipped<DONE:
        b=next(reader)
        skipped+=len(b)

    tail=[]
    for b in reader:
        d=b.to_pydict()
        tail.extend(zip(d["impression_id"],d["article_ids_inview"]))

    print(f"Tail rows: {len(tail):,}")

    # Group duplicate impression IDs. Require duplicate occurrences to have
    # identical candidate lists; otherwise the supplied behavior file is
    # internally ambiguous and must not be silently corrupted.
    groups={}
    for imp,articles in tail:
        imp=int(imp)
        cand=tuple(int(x) for x in articles)
        groups.setdefault(imp,[]).append(cand)

    ambiguous=[]
    for imp,vals in groups.items():
        if len(set(vals))>1:
            ambiguous.append(imp)

    print(f"Unique tail impression IDs: {len(groups):,}")
    print(f"Duplicate-ID groups: {sum(len(v)>1 for v in groups.values()):,}")

    if ambiguous:
        raise RuntimeError(
            f"{len(ambiguous):,} impression IDs have conflicting candidate "
            f"lists. Cannot safely create an official submission. Examples: "
            f"{ambiguous[:10]}"
        )

    # Official submission must contain one line per unique impression ID.
    tail_order=[]
    seen=set()
    for imp,articles in tail:
        imp=int(imp)
        if imp not in seen:
            seen.add(imp)
            tail_order.append(imp)

    # Candidate position lookup.
    positions={
        imp:{article:pos for pos,article in enumerate(groups[imp][0],start=1)}
        for imp in tail_order
    }

    print("\n[1/3] Streaming prediction artifact...")
    pred=pq.ParquetFile(PRED)
    ranked={}
    for n,b in enumerate(pred.iter_batches(
        batch_size=1_000_000,
        columns=["impression_id","article_id","rank"]
    ),1):
        d=b.to_pydict()
        for imp0,art0,rank0 in zip(
            d["impression_id"],d["article_id"],d["rank"]
        ):
            imp=int(imp0)
            if imp not in positions:
                continue
            art=int(art0)
            pos=positions[imp].get(art)
            if pos is None:
                raise RuntimeError(
                    f"Prediction article {art} absent from official "
                    f"candidates for impression {imp}."
                )
            ranked.setdefault(imp,[]).append((int(rank0),pos))
        if n%5==0:
            print(f"       Batches: {n:,} | matched: {len(ranked):,}/{len(tail_order):,}")

    missing=[x for x in tail_order if x not in ranked]
    if missing:
        raise RuntimeError(f"Missing {len(missing):,} predictions; examples {missing[:10]}")

    print("\n[2/3] Appending tail...")
    with open(TXT,"a",encoding="utf-8",newline="\n") as f:
        for imp in tail_order:
            rows=ranked[imp]
            rows.sort(key=lambda x:(x[0],x[1]))
            f.write(f"{imp} [{','.join(str(p) for _,p in rows)}]\n")

    with open(TXT,"rb") as f:
        final=sum(1 for _ in f)

    print(f"Final submission lines: {final:,}")
    print(f"Official unique impression IDs represented: {DONE+len(tail_order):,}")

    if final != DONE+len(tail_order):
        raise RuntimeError("Final line-count validation failed.")

    print("\n[3/3] Creating ZIP...")
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.write(TXT,arcname="predictions.txt")
    print(f"ZIP: {ZIP}")
    print("="*80)
    print("SUBMISSION FILE CREATED")
    print("="*80)

if __name__=="__main__":
    main()
