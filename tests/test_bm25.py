from src.retrieval.bm25 import BM25Retriever


def test_bm25_retrieval():

    article_ids = [
        "A1",
        "A2",
        "A3",
    ]

    documents = [
        "artificial intelligence machine learning neural networks",
        "football premier league Manchester United",
        "stock market investment financial markets",
    ]

    retriever = BM25Retriever(
        article_ids=article_ids,
        documents=documents,
    )

    results = retriever.search(
        "artificial intelligence",
        top_k=2,
    )

    assert len(results) == 1
    assert results[0][0] == "A1"