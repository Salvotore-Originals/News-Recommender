from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTICLES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "test"
    / "articles.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "test"
    / "bm25"
)

ARTICLE_IDS_PATH = (
    OUTPUT_DIR / "article_ids.npy"
)

DOCUMENT_LENGTHS_PATH = (
    OUTPUT_DIR / "document_lengths.npy"
)

IDF_PATH = (
    OUTPUT_DIR / "idf.json"
)

TOKEN_FREQUENCIES_PATH = (
    OUTPUT_DIR / "token_frequencies.pkl"
)

METADATA_PATH = (
    OUTPUT_DIR / "metadata.json"
)


# ============================================================
# BM25 PARAMETERS
# ============================================================

K1 = 1.5
B = 0.75


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(
    text: str,
) -> list[str]:
    """
    Same tokenizer used by the validated
    EB-NeRD BM25 implementation.

    Lowercase text and retain:
        a-z
        0-9
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    return re.findall(
        r"\b[a-z0-9]+\b",
        text,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "EB-NeRD OFFICIAL TEST — BM25 ARTIFACT BUILDER"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD ARTICLES
    # ========================================================

    print(
        "\n[1/5] Loading official test articles..."
    )

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    required_columns = {
        "article_id",
        "text",
    }

    missing = (
        required_columns
        - set(articles.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    articles["article_id"] = (
        articles["article_id"]
        .astype(str)
    )

    articles["text"] = (
        articles["text"]
        .fillna("")
        .astype(str)
    )

    if articles["article_id"].isna().any():
        raise ValueError(
            "Missing article IDs detected."
        )

    if articles["article_id"].duplicated().any():
        raise ValueError(
            "Duplicate article IDs detected."
        )

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    # ========================================================
    # 2. TOKENIZE
    # ========================================================

    print(
        "\n[2/5] Tokenizing article corpus..."
    )

    tokenized_documents = [
        tokenize(text)
        for text in articles["text"]
    ]

    document_lengths = np.asarray(
        [
            len(tokens)
            for tokens in tokenized_documents
        ],
        dtype=np.float64,
    )

    average_document_length = float(
        document_lengths.mean()
    )

    print(
        f"       Average document length: "
        f"{average_document_length:.4f}"
    )

    # ========================================================
    # 3. BUILD EXACT BM25 STATISTICS
    # ========================================================

    print(
        "\n[3/5] Building BM25 reference statistics..."
    )

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=K1,
        b=B,
    )

    # IMPORTANT:
    #
    # Use rank_bm25's own IDF values.
    #
    # This keeps the official-test scorer mathematically
    # consistent with the validated EB-NeRD BM25 implementation.
    #

    idf = {
        token: float(value)
        for token, value
        in bm25.idf.items()
    }

    print(
        f"       Vocabulary: "
        f"{len(idf):,}"
    )

    # --------------------------------------------------------
    # Token frequencies
    # --------------------------------------------------------

    print(
        "       Building token frequencies..."
    )

    token_frequencies = []

    for tokens in tokenized_documents:

        frequencies = defaultdict(int)

        for token in tokens:
            frequencies[token] += 1

        token_frequencies.append(
            dict(frequencies)
        )

    # ========================================================
    # 4. VALIDATE
    # ========================================================

    print(
        "\n[4/5] Validating statistics..."
    )

    if len(document_lengths) != len(
        articles
    ):
        raise RuntimeError(
            "Document-length count does not "
            "match article count."
        )

    if len(token_frequencies) != len(
        articles
    ):
        raise RuntimeError(
            "Token-frequency count does not "
            "match article count."
        )

    if not np.all(
        np.isfinite(document_lengths)
    ):
        raise RuntimeError(
            "Document lengths contain "
            "non-finite values."
        )

    if average_document_length <= 0:
        raise RuntimeError(
            "Average document length must "
            "be positive."
        )

    # Check a sample of IDF values against
    # the actual BM25Okapi object.

    sample_tokens = list(idf)[:100]

    for token in sample_tokens:

        if not np.isclose(
            idf[token],
            float(
                bm25.idf[token]
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "IDF mismatch detected for "
                f"token: {token}"
            )

    print(
        f"       Document lengths: "
        f"{len(document_lengths):,}"
    )

    print(
        f"       Token-frequency rows: "
        f"{len(token_frequencies):,}"
    )

    print(
        f"       IDF entries: "
        f"{len(idf):,}"
    )

    # ========================================================
    # 5. SAVE
    # ========================================================

    print(
        "\n[5/5] Saving BM25 artifacts..."
    )

    # --------------------------------------------------------
    # Article IDs
    # --------------------------------------------------------

    np.save(
        ARTICLE_IDS_PATH,
        articles[
            "article_id"
        ].to_numpy(
            dtype=str
        ),
    )

    # --------------------------------------------------------
    # Document lengths
    # --------------------------------------------------------

    np.save(
        DOCUMENT_LENGTHS_PATH,
        document_lengths,
    )

    # --------------------------------------------------------
    # IDF
    # --------------------------------------------------------

    with open(
        IDF_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            idf,
            file,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Token frequencies
    # --------------------------------------------------------

    with open(
        TOKEN_FREQUENCIES_PATH,
        "wb",
    ) as file:

        pickle.dump(
            token_frequencies,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "article_count": int(
            len(articles)
        ),
        "vocabulary_size": int(
            len(idf)
        ),
        "average_document_length": (
            average_document_length
        ),
        "k1": K1,
        "b": B,
        "tokenizer": (
            r"\b[a-z0-9]+\b"
        ),
        "idf_source": (
            "rank_bm25.BM25Okapi"
        ),
        "article_ids_dtype": "str",
        "document_lengths_dtype": (
            "float64"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "EB-NeRD OFFICIAL TEST BM25 "
        "BUILD COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nArticle IDs:\n"
        f"  {ARTICLE_IDS_PATH}"
    )

    print(
        f"\nDocument lengths:\n"
        f"  {DOCUMENT_LENGTHS_PATH}"
    )

    print(
        f"\nIDF:\n"
        f"  {IDF_PATH}"
    )

    print(
        f"\nToken frequencies:\n"
        f"  {TOKEN_FREQUENCIES_PATH}"
    )

    print(
        f"\nMetadata:\n"
        f"  {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()