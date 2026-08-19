from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import pandas as pd


class MINDQueryBuilder:
    """
    Constructs an improved lexical BM25 query from MIND click history.

    Improvements:
    - Uses recent clicks more heavily than old clicks.
    - Removes very common stopwords.
    - Removes duplicate terms.
    - Limits query length.
    - Preserves informative title terms.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but",
        "of", "to", "in", "on", "for", "with",
        "at", "by", "from", "as", "is", "are",
        "was", "were", "be", "been", "being",
        "this", "that", "these", "those",
        "it", "its", "he", "she", "they",
        "his", "her", "their", "you", "your",
        "i", "we", "our", "us",
        "about", "after", "before", "into",
        "over", "under", "than", "then",
        "who", "what", "when", "where", "why",
        "how"
    }

    def __init__(
        self,
        articles: pd.DataFrame,
        text_field: str = "title",
        max_history: int = 10,
        max_query_terms: int = 80,
    ) -> None:

        required_columns = {
            "article_id",
            text_field,
        }

        missing = required_columns - set(articles.columns)

        if missing:
            raise ValueError(
                f"Missing required article columns: {sorted(missing)}"
            )

        self.text_field = text_field
        self.max_history = max_history
        self.max_query_terms = max_query_terms

        self.article_text = (
            articles
            .assign(
                article_id=lambda df:
                    df["article_id"].astype(str),

                text=lambda df:
                    df[text_field]
                    .fillna("")
                    .astype(str),
            )
            .set_index("article_id")["text"]
            .to_dict()
        )

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Tokenize title text while removing stopwords.
        """

        tokens = re.findall(
            r"\b[a-z0-9]+\b",
            text.lower(),
        )

        return [
            token
            for token in tokens
            if token not in cls.STOPWORDS
            and len(token) > 1
        ]

    def build_query(
        self,
        history: Sequence[str],
    ) -> str:
        """
        Build a compact lexical query from click history.

        More recent clicks receive higher weight by appearing
        earlier/more frequently in the query.
        """

        if not history:
            return ""

        # --------------------------------------------------
        # Keep the most recent history
        # --------------------------------------------------

        history = list(history)[-self.max_history:]

        term_weights = Counter()

        # --------------------------------------------------
        # Recent clicks receive higher importance
        #
        # Example for 5 articles:
        # oldest -> weight 1
        # newest -> weight 5
        # --------------------------------------------------

        for position, article_id in enumerate(history, start=1):

            text = self.article_text.get(
                str(article_id),
                "",
            )

            if not text:
                continue

            tokens = self.tokenize(text)

            # More recent article = larger weight
            weight = position

            for token in set(tokens):
                term_weights[token] += weight

        if not term_weights:
            return ""

        # --------------------------------------------------
        # Select strongest lexical terms
        # --------------------------------------------------

        ranked_terms = sorted(
            term_weights.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        selected_terms = [
            term
            for term, _ in ranked_terms[
                :self.max_query_terms
            ]
        ]

        return " ".join(selected_terms)