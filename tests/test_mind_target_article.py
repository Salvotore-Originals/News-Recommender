import pandas as pd

from src.data.mind_behaviors import (
    load_mind_behaviors,
    parse_history,
)

from src.retrieval.mind_query import (
    MINDQueryBuilder,
)


ARTICLES_PATH = "data/features/mind/articles.parquet"
BEHAVIORS_PATH = "data/raw/mind/train/behaviors.tsv"


def test_inspect_positive_article():

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    behaviors = load_mind_behaviors(
        BEHAVIORS_PATH
    )

    row = behaviors[
        behaviors["history"].str.strip() != ""
    ].iloc[0]

    history = parse_history(
        row["history"]
    )

    positive = [
        item.rsplit("-", 1)[0]
        for item in row["impressions"].split()
        if item.endswith("-1")
    ]

    article_lookup = articles.set_index(
        "article_id"
    )

    query_builder = MINDQueryBuilder(
        articles,
        text_field="title",
    )

    query = query_builder.build_query(
        history
    )

    print("\n" + "=" * 90)
    print("TARGET ARTICLE DIAGNOSTIC")
    print("=" * 90)

    print("\nUser:", row["user_id"])

    print("\nQuery:")
    print(query)

    print("\nPositive articles:")

    for article_id in positive:

        print("\nArticle ID:", article_id)

        if article_id not in article_lookup.index:
            print("NOT FOUND IN ARTICLE STORE")
            continue

        article = article_lookup.loc[
            article_id
        ]

        print("Title:")
        print(article["title"])

        print("\nAbstract:")

        if "abstract" in article:
            print(article["abstract"])
        else:
            print("No abstract column.")

    assert len(positive) > 0