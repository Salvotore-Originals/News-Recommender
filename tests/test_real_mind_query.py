import pandas as pd

from src.data.mind_behaviors import (
    load_mind_behaviors,
    parse_history,
)

from src.retrieval.mind_query import (
    MINDQueryBuilder,
)


ARTICLES_PATH = (
    "data/features/mind/articles.parquet"
)

BEHAVIORS_PATH = (
    "data/raw/mind/train/behaviors.tsv"
)


def test_real_mind_query():

    # Load article feature store
    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    # Load actual MIND behavior data
    behaviors = load_mind_behaviors(
        BEHAVIORS_PATH
    )

    # Find first impression with history
    row = behaviors[
        behaviors["history"].str.strip() != ""
    ].iloc[0]

    # Parse click history
    history = parse_history(
        row["history"]
    )

    # Build query
    query_builder = MINDQueryBuilder(
        articles
    )

    query = query_builder.build_query(
        history
    )

    print("\n" + "=" * 80)
    print("REAL MIND USER QUERY")
    print("=" * 80)

    print(f"User: {row['user_id']}")
    print(f"Time: {row['time']}")

    print("\nClicked article IDs:")
    print(history)

    print("\nNumber of clicked articles:")
    print(len(history))

    print("\nBM25 Query:")
    print(query)

    assert len(history) > 0
    assert query.strip() != ""