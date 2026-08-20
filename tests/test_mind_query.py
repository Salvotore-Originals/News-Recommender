import pandas as pd

from src.retrieval.mind_query import MINDQueryBuilder


def test_mind_query_builder():

    articles = pd.DataFrame({
        "article_id": [
            "N1",
            "N2",
            "N3",
        ],
        "title": [
            "Artificial Intelligence Advances",
            "Microsoft Announces New AI Model",
            "Premier League Football Results",
        ],
    })

    builder = MINDQueryBuilder(articles)

    query = builder.build_query(
        ["N1", "N2"]
    )

    assert query
    assert "artificial" in query
    assert "intelligence" in query
    assert "advances" in query
    assert "microsoft" in query
    assert "announces" in query
    assert "model" in query

    assert "the" not in query.split()
    assert "and" not in query.split()

def test_unknown_articles_are_ignored():

    articles = pd.DataFrame({
        "article_id": ["N1"],
        "title": ["Artificial Intelligence Advances"],
    })

    builder = MINDQueryBuilder(articles)

    query = builder.build_query(
        ["N1", "N999999"]
    )

    assert query
    assert "artificial" in query
    assert "intelligence" in query
    assert "advances" in query
    assert "n999999" not in query