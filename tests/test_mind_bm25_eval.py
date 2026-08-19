import pandas as pd

from src.data.mind_behaviors import (
    load_mind_behaviors,
    parse_history,
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
    "data/features/mind/dev_articles.parquet"
)

BEHAVIORS_PATH = (
    "data/raw/mind/dev/behaviors.tsv"
)


def test_mind_bm25_dev_evaluation():

    # ==================================================
    # 1. Load article feature store
    # ==================================================

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    print(
        f"\nArticles loaded: {len(articles):,}"
    )

    # Article lookup for diagnostics
    article_lookup = articles.set_index(
        "article_id"
    )

    # ==================================================
    # 2. Load MIND dev behaviors
    # ==================================================

    behaviors = load_mind_behaviors(
        BEHAVIORS_PATH
    )

    print(
        f"Dev impressions loaded: {len(behaviors):,}"
    )

    # ==================================================
    # 3. Build query builder ONCE
    # ==================================================

    query_builder = MINDQueryBuilder(
        articles,
        text_field="title",
    )

    # ==================================================
    # 4. Build BM25 index ONCE
    # ==================================================

    print("\nBuilding BM25 index...")

    retriever = build_mind_bm25(
        ARTICLES_PATH
    )

    print("BM25 index ready.")

    # ==================================================
    # 5. Evaluation counters
    # ==================================================

    total = 0

    skipped_empty_history = 0
    skipped_no_positive = 0

    hit10 = 0
    hit50 = 0
    hit100 = 0

    reciprocal_rank_sum = 0.0

    # ==================================================
    # 6. Missing impression article diagnostics
    # ==================================================

    missing_impression_ids = set()

    # ==================================================
    # 7. Iterate through first 100 dev impressions
    # ==================================================

    for _, row in behaviors.iterrows():

        # ----------------------------------------------
        # Check impression article availability
        # ----------------------------------------------

        for item in row["impressions"].split():

            article_id = item.rsplit(
                "-",
                1,
            )[0]

            if article_id not in article_lookup.index:

                missing_impression_ids.add(
                    article_id
                )

        # ----------------------------------------------
        # Parse click history
        # ----------------------------------------------

        history = parse_history(
            row["history"]
        )

        if not history:

            skipped_empty_history += 1
            continue

        # ----------------------------------------------
        # Positive impression articles
        # ----------------------------------------------

        positive_articles = [
            article_id
            for article_id, clicked
            in (
                item.rsplit("-", 1)
                for item in row["impressions"].split()
            )
            if clicked == "1"
        ]

        if not positive_articles:

            skipped_no_positive += 1
            continue

        # ----------------------------------------------
        # Build BM25 query
        # ----------------------------------------------

        query = query_builder.build_query(
            history
        )

        if not query.strip():
            continue

        # ----------------------------------------------
        # Retrieve Top-200
        # ----------------------------------------------

        results = retriever.search(
            query,
            top_k=200,
        )

        # ----------------------------------------------
        # Remove previously seen articles
        # ----------------------------------------------

        filtered_results = filter_seen_articles(
            results,
            history,
        )[:100]

        candidate_ids = [
            article_id
            for article_id, _ in filtered_results
        ]

        # ----------------------------------------------
        # Evaluation
        # ----------------------------------------------

        total += 1

        if total % 1000 == 0:

            print(
                f"Evaluated: {total:,}"
            )

        # ----------------------------------------------
        # Hit@10
        # ----------------------------------------------

        if any(
            article_id in candidate_ids[:10]
            for article_id in positive_articles
        ):

            hit10 += 1

        # ----------------------------------------------
        # Hit@50
        # ----------------------------------------------

        if any(
            article_id in candidate_ids[:50]
            for article_id in positive_articles
        ):

            hit50 += 1

        # ----------------------------------------------
        # Hit@100
        # ----------------------------------------------

        if any(
            article_id in candidate_ids[:100]
            for article_id in positive_articles
        ):

            hit100 += 1

        # ----------------------------------------------
        # Reciprocal Rank
        # ----------------------------------------------

        best_rank = None

        for article_id in positive_articles:

            rank = rank_of_article(
                filtered_results,
                article_id,
            )

            if rank is not None:

                if (
                    best_rank is None
                    or rank < best_rank
                ):

                    best_rank = rank

        if best_rank is not None:

            reciprocal_rank_sum += (
                1.0 / best_rank
            )

    # ==================================================
    # 8. Final metrics
    # ==================================================

    assert total > 0

    hit10_score = hit10 / total
    hit50_score = hit50 / total
    hit100_score = hit100 / total

    mrr = (
        reciprocal_rank_sum / total
    )

    # ==================================================
    # 9. Final report
    # ==================================================

    print("\n")

    print("=" * 90)

    print(
        "MIND DEV → BM25 EVALUATION"
    )

    print("=" * 90)

    print(
        f"\nEvaluated impressions: "
        f"{total:,}"
    )

    print(
        f"Skipped empty histories: "
        f"{skipped_empty_history:,}"
    )

    print(
        f"Skipped without positive: "
        f"{skipped_no_positive:,}"
    )

    print(
        f"Missing impression articles: "
        f"{len(missing_impression_ids):,}"
    )

    # ==================================================
    # 10. Missing article diagnostic
    # ==================================================

    print("\n" + "-" * 90)

    print(
        "MISSING IMPRESSION ARTICLE DIAGNOSTIC"
    )

    print("-" * 90)

    if missing_impression_ids:

        print(
            "\nFirst 30 missing article IDs:"
        )

        for article_id in sorted(
            missing_impression_ids
        )[:30]:

            print(
                f"  {article_id}"
            )

    else:

        print(
            "\nNo missing impression articles found."
        )

    # ==================================================
    # 11. BM25 results
    # ==================================================

    print("\n" + "-" * 90)

    print(
        "CORPUS-LEVEL BM25 RETRIEVAL"
    )

    print("-" * 90)

    print(
        "Initial retrieval: Top-200"
    )

    print(
        "Seen-item filtering: enabled"
    )

    print(
        "Final candidate pool: Top-100"
    )

    print(
        f"\nHit@10:  {hit10_score:.4f}"
    )

    print(
        f"Hit@50:  {hit50_score:.4f}"
    )

    print(
        f"Hit@100: {hit100_score:.4f}"
    )

    print(
        f"MRR:     {mrr:.4f}"
    )

    print("=" * 90)

    # ==================================================
    # 12. Sanity checks
    # ==================================================

    assert 0.0 <= hit10_score <= 1.0
    assert 0.0 <= hit50_score <= 1.0
    assert 0.0 <= hit100_score <= 1.0
    assert 0.0 <= mrr <= 1.0

    assert hit10_score <= hit50_score
    assert hit50_score <= hit100_score