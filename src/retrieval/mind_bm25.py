from __future__ import annotations

import pandas as pd

from src.retrieval.bm25 import BM25Retriever


def build_mind_bm25(
    articles_path: str,
) -> BM25Retriever:

    articles = pd.read_parquet(articles_path)

    required_columns = {
        "article_id",
        "title",
        "abstract",
    }

    missing = required_columns - set(articles.columns)

    if missing:
        raise ValueError(
            f"Missing required article columns: {sorted(missing)}"
        )

    articles["title"] = articles["title"].fillna("")
    articles["abstract"] = articles["abstract"].fillna("")

    documents = (
        articles["title"].astype(str)
        + " "
        + articles["abstract"].astype(str)
    ).tolist()

    article_ids = (
        articles["article_id"]
        .astype(str)
        .tolist()
    )

    return BM25Retriever(
        article_ids=article_ids,
        documents=documents,
    )