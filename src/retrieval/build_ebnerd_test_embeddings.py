from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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
    / "embeddings"
    / "ebnerd"
    / "test"
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

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main() -> None:

    print("=" * 80)
    print("EB-NeRD OFFICIAL TEST — SEMANTIC EMBEDDINGS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/5] Loading test article features...")

    articles = pd.read_parquet(
        ARTICLES_PATH
    )

    required = {
        "article_id",
        "text",
    }

    missing = (
        required
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

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    print("\n[2/5] Loading Sentence Transformer...")

    print(
        f"       Model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("\n[3/5] Generating embeddings...")

    print(
        f"       Batch size: {BATCH_SIZE}"
    )

    embeddings = model.encode(
        articles["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print("\n[4/5] Validating embeddings...")

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Embeddings must be 2-dimensional."
        )

    if embeddings.shape[0] != len(articles):
        raise RuntimeError(
            "Embedding count does not match "
            "article count."
        )

    if embeddings.shape[1] != 384:
        raise RuntimeError(
            f"Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    if not np.all(
        np.isfinite(embeddings)
    ):
        raise RuntimeError(
            "Embeddings contain non-finite values."
        )

    print(
        f"       Shape: {embeddings.shape}"
    )

    print(
        f"       Dimension: {embeddings.shape[1]}"
    )

    print("\n[5/5] Saving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    np.save(
        ARTICLE_IDS_PATH,
        articles["article_id"].to_numpy(
            dtype=str
        ),
    )

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "article_count": int(
            len(articles)
        ),
        "text_field": "text",
        "batch_size": BATCH_SIZE,
        "dtype": "float32",
        "normalized": True,
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
    print("=" * 80)
    print("EB-NeRD TEST SEMANTIC EMBEDDINGS COMPLETE")
    print("=" * 80)

    print(
        f"\nEmbeddings: {EMBEDDINGS_PATH}"
    )

    print(
        f"Article IDs: {ARTICLE_IDS_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()