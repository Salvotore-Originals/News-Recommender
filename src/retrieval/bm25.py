from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Fast BM25 lexical retriever using an inverted index.

    Unlike rank_bm25.BM25Okapi.get_scores(), this implementation
    scores only documents that contain at least one query term.

    BM25 parameters:
        k1 = 1.5
        b  = 0.75
    """

    def __init__(
        self,
        article_ids: Sequence[str],
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:

        if len(article_ids) != len(documents):
            raise ValueError(
                "article_ids and documents must have the same length."
            )

        if len(article_ids) == 0:
            raise ValueError(
                "Cannot build BM25 index from an empty corpus."
            )

        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.article_ids = list(article_ids)
        self.documents = list(documents)

        
        self.article_to_index = {
            str(article_id): index
            for index, article_id
            in enumerate(self.article_ids)
        }

        self.k1 = k1
        self.b = b

        # --------------------------------------------------
        # Tokenize documents
        # --------------------------------------------------

        self.tokenized_documents = [
            self.tokenize(document)
            for document in self.documents
        ]

        # Keep the BM25 index available
        self.index = BM25Okapi(
        self.tokenized_documents
        )

        # Inverted index for candidate lookup
        self.inverted_index: dict[str, list[int]] = {}

        for doc_index, tokens in enumerate(
            self.tokenized_documents
        ):
            for token in set(tokens):
                self.inverted_index.setdefault(
            token, []
        ).append(doc_index)

        # Document lengths
        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.num_documents = len(
            self.tokenized_documents
        )

        self.avg_document_length = (
            sum(self.document_lengths)
            / self.num_documents
        )

        # --------------------------------------------------
        # Inverted index
        #
        # token ->
        # {
        #     document_index: term_frequency
        # }
        # --------------------------------------------------

        self.inverted_index: dict[
            str,
            dict[int, int]
        ] = defaultdict(dict)

        # Document frequency
        document_frequency: Counter[str] = Counter()

        for doc_index, tokens in enumerate(
            self.tokenized_documents
        ):

            term_counts = Counter(tokens)

            for token, frequency in term_counts.items():

                self.inverted_index[token][
                    doc_index
                ] = frequency

                document_frequency[token] += 1

        # --------------------------------------------------
        # BM25 IDF
        #
        # Same standard formulation used by BM25Okapi:
        #
        # IDF = log(
        #     (N - df + 0.5) /
        #     (df + 0.5)
        # )
        # --------------------------------------------------

        raw_idf: dict[str, float] = {}

        for token, df in document_frequency.items():

            raw_idf[token] = math.log(
                (
                    self.num_documents
                    - df
                    + 0.5
                )
                /
                (
                    df
                    + 0.5
                )
            )

        # --------------------------------------------------
        # Handle negative IDF values in the same spirit as
        # rank_bm25.BM25Okapi.
        # --------------------------------------------------

        if raw_idf:

            average_idf = (
                sum(raw_idf.values())
                / len(raw_idf)
            )

            epsilon = 0.25 * average_idf

            self.idf = {
                token: (
                    epsilon
                    if value < 0
                    else value
                )
                for token, value in raw_idf.items()
            }

        else:

            self.idf = {}

    # ======================================================
    # TOKENIZATION
    # ======================================================

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:

        if not isinstance(text, str):
            return []

        text = text.lower()

        return re.findall(
            r"\b[a-z0-9]+\b",
            text,
        )

    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # --------------------------------------------------
        # Query term frequencies
        #
        # Keeping duplicate terms makes the behaviour closer
        # to rank_bm25.BM25Okapi.
        # --------------------------------------------------

        query_counts = Counter(
            query_tokens
        )

        # --------------------------------------------------
        # Candidate generation
        #
        # Only documents containing at least one query term
        # are considered.
        # --------------------------------------------------

        candidate_documents: set[int] = set()

        for token in query_counts:

            postings = self.inverted_index.get(
                token
            )

            if postings is not None:

                candidate_documents.update(
                    postings.keys()
                )

        if not candidate_documents:
            return []

        # --------------------------------------------------
        # BM25 scoring
        #
        # We accumulate scores directly from the inverted
        # index.
        #
        # No full-corpus get_scores() call.
        # --------------------------------------------------

        scores: dict[int, float] = defaultdict(float)

        avgdl = self.avg_document_length
        k1 = self.k1
        b = self.b

        for token, query_frequency in query_counts.items():

            postings = self.inverted_index.get(
                token
            )

            if postings is None:
                continue

            idf = self.idf.get(
                token,
                0.0,
            )

            # --------------------------------------------------
            # Only documents containing this query term
            # receive a contribution.
            # --------------------------------------------------

            for doc_index, term_frequency in postings.items():

                # Skip impossible candidates defensively.
                if doc_index not in candidate_documents:
                    continue

                document_length = (
                    self.document_lengths[
                        doc_index
                    ]
                )

                denominator = (
                    term_frequency
                    +
                    k1
                    *
                    (
                        1
                        -
                        b
                        +
                        b
                        *
                        (
                            document_length
                            /
                            avgdl
                        )
                    )
                )

                contribution = (
                    idf
                    *
                    (
                        term_frequency
                        *
                        (k1 + 1)
                    )
                    /
                    denominator
                )

                # Account for repeated query terms.
                scores[doc_index] += (
                    query_frequency
                    * contribution
                )

        # --------------------------------------------------
        # Sort only candidate documents.
        # --------------------------------------------------

        ranked_candidates = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked_candidates = ranked_candidates[
            :top_k
        ]

        # --------------------------------------------------
        # Convert document indices to article IDs.
        # --------------------------------------------------

        return [
            (
                self.article_ids[doc_index],
                float(score),
            )
            for doc_index, score
            in ranked_candidates
        ]

    def score_candidates(
        self,
        query: str,
        article_ids: Sequence[str],
    ) -> list[tuple[str, float]]:

        if not query or not query.strip():
            return [
                (
                    str(article_id),
                    0.0,
                )
                for article_id in article_ids
            ]

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return [
                (
                    str(article_id),
                    0.0,
                )
                for article_id in article_ids
            ]

        query_counts = Counter(
            query_tokens
        )

    # --------------------------------------------------
    # Article ID -> document index
    # --------------------------------------------------

        requested_ids = [
            str(article_id)
            for article_id in article_ids
        ]

        requested_indices = {
            article_id: self.article_to_index.get(article_id)
            for article_id in requested_ids
        }

    # --------------------------------------------------
    # Score only requested candidates.
    # --------------------------------------------------

        scores: dict[int, float] = defaultdict(float)

        avgdl = self.avg_document_length
        k1 = self.k1
        b = self.b

        candidate_indices = {
            index
            for index in requested_indices.values()
            if index is not None
        }

        for token, query_frequency in query_counts.items():

            postings = self.inverted_index.get(token)

            if postings is None:
                continue

            idf = self.idf.get(
                token,
                0.0,
            )

            for doc_index, term_frequency in postings.items():

                if doc_index not in candidate_indices:
                    continue

                document_length = (
                    self.document_lengths[doc_index]
                )

                denominator = (
                    term_frequency
                    +
                    k1
                    *
                    (
                        1
                        - b
                        +
                        b
                        *
                        (
                            document_length
                            / avgdl
                        )
                    )
                )
            

                contribution = (
                    idf
                    *
                    (
                        term_frequency
                        * (k1 + 1)
                    )
                    /
                    denominator
                )

                scores[doc_index] += (
                    query_frequency
                    * contribution
                )

    # --------------------------------------------------
    # Preserve requested candidate order.
    # --------------------------------------------------

        return [
            (
                article_id,
                float(
                    scores.get(
                        requested_indices[article_id],
                        0.0,
                    )
                    if requested_indices[article_id]
                    is not None
                    else 0.0
                ),
            )
            for article_id in requested_ids
        ]