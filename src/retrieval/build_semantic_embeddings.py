from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

ARTICLES_PATH = Path(
    "data/features/mind/articles.parquet"
)

OUTPUT_DIR = Path(
    "data/features/mind/semantic"
)

EMBEDDINGS_PATH = (
    OUTPUT_DIR / "article_embeddings.npy"
)

ARTICLE_IDS_PATH = (
    OUTPUT_DIR / "article_ids.npy"
)

METADATA_PATH = (
    OUTPUT_DIR / "metadata.json"
)

BATCH_SIZE = 32


def main() -> None:

    # --------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load articles
    # --------------------------------------------------

    print("=" * 70)
    print("MIND SEMANTIC EMBEDDING BUILDER")
    print("=" * 70)

    print()
    print("[1/5] Loading article feature store...")

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
            f"Missing required columns: {sorted(missing)}"
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

    if (
        articles["text"]
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Article text contains empty documents."
        )

    print(
        f"       Articles: {len(articles):,}"
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print()
    print("[2/5] Loading Sentence Transformer model...")
    print(
        f"       Model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    print()
    print("[3/5] Generating article embeddings...")
    print(
        f"       Batch size: {BATCH_SIZE}"
    )

    embeddings = model.encode(
        articles["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------

    print()
    print("[4/5] Validating embeddings...")

    expected_rows = len(articles)

    if embeddings.shape[0] != expected_rows:
        raise RuntimeError(
            "Embedding row count does not match "
            "article count."
        )

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Embeddings must be a 2D matrix."
        )

    if not np.all(
        np.isfinite(embeddings)
    ):
        raise RuntimeError(
            "Embeddings contain non-finite values."
        )

    embedding_dimension = embeddings.shape[1]

    print(
        f"       Shape: {embeddings.shape}"
    )

    print(
        f"       Dimension: {embedding_dimension}"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print()
    print("[5/5] Saving semantic feature store...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    np.save(
        ARTICLE_IDS_PATH,
        articles["article_id"].to_numpy(),
    )

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(
            embedding_dimension
        ),
        "article_count": int(
            len(articles)
        ),
        "text_field": "text",
        "batch_size": BATCH_SIZE,
        "dtype": "float32",
        "normalized": False,
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

    print()
    print("=" * 70)
    print("SEMANTIC EMBEDDING BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Embeddings: {EMBEDDINGS_PATH}"
    )

    print(
        f"Article IDs: {ARTICLE_IDS_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()