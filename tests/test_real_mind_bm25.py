import pandas as pd

from src.data.mind_behaviors import (
    load_mind_behaviors,
    parse_history,
)

from src.data.mind_impressions import (
    get_positive_articles,
)

from src.retrieval.mind_query import (
    MINDQueryBuilder,
)

from src.retrieval.mind_bm25 import (
    build_mind_bm25,
)

from src.retrieval.filtering import (
    filter_seen_articles,
)

from src.evaluation.retrieval import (
    hit_at_k,
    rank_of_article,
)

ARTICLES_PATH = (
    "data/features/mind/articles.parquet"
)

BEHAVIORS_PATH = (
    "data/raw/mind/train/behaviors.tsv"
)


def test_real_mind_bm25():

    # ==================================================
    # 1. Load article data
    # ==================================================

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    article_lookup = articles.set_index(
        "article_id"
    )

    # ==================================================
    # 2. Load MIND behaviors
    # ==================================================

    behaviors = load_mind_behaviors(
        BEHAVIORS_PATH
    )

    # ==================================================
    # 3. Select a real user impression
    # ==================================================

    row = behaviors[
        behaviors["history"].str.strip() != ""
    ].iloc[0]

    # ==================================================
    # 4. Parse click history
    # ==================================================

    history = parse_history(
        row["history"]
    )

    # ==================================================
    # 5. Build BM25 query
    # ==================================================

    query_builder = MINDQueryBuilder(
        articles,
        text_field="title",
    )

    query = query_builder.build_query(
        history
    )

    # ==================================================
    # 6. Build BM25 retriever
    # ==================================================

    retriever = build_mind_bm25(
        ARTICLES_PATH
    )

    # ==================================================
    # 7. Retrieve Top-100 candidates
    # ==================================================

    results = retriever.search(
        query,
        top_k=200,
    )

    # ==================================================
    # 8. Remove previously seen articles
    # ==================================================

    filtered_results = filter_seen_articles(
        results,
        history,
    )[:100]

    # ==================================================
    # 9. Extract positive impression articles
    # ==================================================

    positive_articles = get_positive_articles(
        row["impressions"]
    )

    # ==================================================
    # 10. Print basic information
    # ==================================================

    print("\n" + "=" * 90)
    print("MIND → CLICK HISTORY → BM25 → EVALUATION")
    print("=" * 90)

    print(
        f"\nUser: {row['user_id']}"
    )

    print(
        f"Time: {row['time']}"
    )

    print(
        f"History length: {len(history)}"
    )

    print("\nHistory:")
    print(" ".join(history))

    print("\nBM25 Query:")
    print(query)

    # ==================================================
    # 11. Original BM25 results
    # ==================================================

    print("\nTop 10 BM25 Results:")

    for rank, (article_id, score) in enumerate(
        results[:10],
        start=1,
    ):

        title = article_lookup.loc[
            article_id,
            "title",
        ]

        print(
            f"{rank:02d}. "
            f"{article_id} | "
            f"{score:.4f} | "
            f"{title}"
        )

    # ==================================================
    # 12. Filtered BM25 results
    # ==================================================

    print(
        "\nTop 10 After Seen-Item Filtering:"
    )

    for rank, (article_id, score) in enumerate(
        filtered_results[:10],
        start=1,
    ):

        title = article_lookup.loc[
            article_id,
            "title",
        ]

        print(
            f"{rank:02d}. "
            f"{article_id} | "
            f"{score:.4f} | "
            f"{title}"
        )

    # ==================================================
    # 13. Candidate counts
    # ==================================================

    print(
        f"\nCandidates before filtering: "
        f"{len(results)}"
    )

    print(
        f"Candidates after filtering: "
        f"{len(filtered_results)}"
    )

    # ==================================================
    # 14. Actual impressions
    # ==================================================

    print("\nActual impressions:")
    print(row["impressions"])

    print("\nPositive impression articles:")
    print(positive_articles)

    # ==================================================
    # 15. Hit@K evaluation
    # ==================================================

    print("\nBM25 Hit Evaluation:")

    for k in [50, 100]:

        hit = hit_at_k(
            filtered_results,
            positive_articles,
            k,
        )

        print(
            f"Hit@{k}: {hit}"
        )

    # ==================================================
    # 16. Positive article rank
    # ==================================================

    print("\nPositive Article Ranks:")

    for article_id in positive_articles:

        rank = rank_of_article(
            filtered_results,
            article_id,
        )

        if rank is None:
            rank_display = "NOT RETRIEVED"
        else:
            rank_display = rank

        print(
            f"{article_id}: "
            f"{rank_display}"
        )

    # ==================================================
    # 17. Assertions
    # ==================================================

    seen_set = set(history)

    # No seen article may survive filtering.
    assert all(
        article_id not in seen_set
        for article_id, _ in filtered_results
    )

    # History and query must exist.
    assert len(history) > 0
    assert query.strip() != ""

    # We requested 200 candidates.
    assert len(results) == 200

    # Filtering cannot increase candidate count.
    assert len(filtered_results) <= 100
