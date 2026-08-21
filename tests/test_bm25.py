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

def test_score_candidates_scores_requested_articles():
    retriever = BM25Retriever(
        article_ids=[
            "A",
            "B",
            "C",
        ],
        documents=[
            "apple banana",
            "banana orange",
            "apple orange",
        ],
    )

    scores = retriever.score_candidates(
        query="apple",
        article_ids=[
            "A",
            "B",
            "C",
        ],
    )

    # The requested candidate order must be preserved.
    assert [article_id for article_id, _ in scores] == [
        "A",
        "B",
        "C",
    ]

    # Candidate scores must agree with the existing
    # BM25 search implementation for matching documents.
    search_scores = dict(
        retriever.search(
            query="apple",
            top_k=3,
        )
    )

    assert scores[0][1] == search_scores["A"]
    assert scores[2][1] == search_scores["C"]

    # B does not contain "apple", so it receives zero.
    assert scores[1][1] == 0.0


def test_score_candidates_handles_unknown_articles():
    retriever = BM25Retriever(
        article_ids=[
            "A",
            "B",
        ],
        documents=[
            "apple banana",
            "banana orange",
        ],
    )

    scores = retriever.score_candidates(
        query="apple",
        article_ids=[
            "A",
            "UNKNOWN",
        ],
    )

    assert scores[0][0] == "A"
    assert scores[1][0] == "UNKNOWN"

    # The unknown article is not in the BM25 corpus,
    # so it receives the defined fallback score of 0.
    assert scores[1][1] == 0.0