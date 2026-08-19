import pandas as pd

from src.retrieval.mind_bm25 import build_mind_bm25


def test_mind_bm25():

    path = "data/features/mind/articles.parquet"

    articles = pd.read_parquet(path)

    retriever = build_mind_bm25(path)

    results = retriever.search(
        "artificial intelligence",
        top_k=10,
    )

    article_lookup = articles.set_index("article_id")

    print("\n" + "=" * 80)
    print("BM25 RETRIEVAL TEST")
    print("=" * 80)

    for article_id, score in results:

        title = article_lookup.loc[
            article_id,
            "title"
        ]

        print(f"\nScore: {score:.4f}")
        print(f"ID:    {article_id}")
        print(f"Title: {title}")

    assert len(results) == 10