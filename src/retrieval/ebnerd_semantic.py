from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "ebnerd"
    / "small"
)

MODEL_NAME = "all-MiniLM-L6-v2"

HISTORY_QUERY_LENGTH = 10


def load_articles() -> pd.DataFrame:
    return pd.read_parquet(
        FEATURE_ROOT / "articles.parquet"
    )


def load_history(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / f"{split}_user_history.parquet"
    )


def load_interactions(
    split: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        PROCESSED_ROOT
        / "splits"
        / f"{split}.parquet"
    )


def build_embeddings(
    articles: pd.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode every EB-NeRD article once.

    The article representation uses the canonical `text`
    field from the EB-NeRD feature store.
    """

    texts = (
        articles["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(
        np.float32
    )


def save_embeddings(
    embeddings: np.ndarray,
    articles: pd.DataFrame,
) -> None:

    EMBEDDING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        EMBEDDING_ROOT
        / "article_embeddings.npy"
    )

    ids_path = (
        EMBEDDING_ROOT
        / "article_ids.parquet"
    )

    np.save(
        embedding_path,
        embeddings,
    )

    articles[
        ["article_id"]
    ].to_parquet(
        ids_path,
        index=False,
    )

    print(
        f"Saved embeddings: "
        f"{embedding_path}"
    )

    print(
        f"Saved article IDs: "
        f"{ids_path}"
    )


def load_or_build_embeddings(
    articles: pd.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:

    embedding_path = (
        EMBEDDING_ROOT
        / "article_embeddings.npy"
    )

    ids_path = (
        EMBEDDING_ROOT
        / "article_ids.parquet"
    )

    if (
        embedding_path.exists()
        and ids_path.exists()
    ):

        saved_ids = pd.read_parquet(
            ids_path
        )["article_id"].tolist()

        current_ids = (
            articles["article_id"]
            .tolist()
        )

        if saved_ids != current_ids:

            raise ValueError(
                "Saved embedding article IDs "
                "do not match the current article "
                "corpus."
            )

        embeddings = np.load(
            embedding_path
        )

        if len(embeddings) != len(
            articles
        ):

            raise ValueError(
                "Embedding count does not "
                "match article count."
            )

        print(
            "Loaded existing article embeddings."
        )

        return embeddings

    print(
        "Building article embeddings..."
    )

    start = time.perf_counter()

    embeddings = build_embeddings(
        articles,
        model,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Embedding generation completed "
        f"in {elapsed:.2f}s"
    )

    save_embeddings(
        embeddings,
        articles,
    )

    return embeddings


def build_history_lookup(
    history: pd.DataFrame,
) -> dict:

    history = history.sort_values(
        [
            "user_id",
            "timestamp",
        ]
    )

    lookup = defaultdict(list)

    for row in history.itertuples(
        index=False
    ):

        lookup[row.user_id].append(
            (
                row.timestamp,
                row.article_id,
            )
        )

    return lookup


def build_query_embedding(
    history_events: list,
    impression_time,
    article_to_index: dict,
    article_embeddings: np.ndarray,
    max_articles: int = HISTORY_QUERY_LENGTH,
) -> np.ndarray | None:
    """
    Build a point-in-time semantic user query.

    Only history events strictly before the impression
    are allowed.

    The query is the normalized mean of the most recent
    unique article embeddings.
    """

    eligible = [
        event
        for event in history_events
        if event[0] < impression_time
    ]

    eligible = eligible[
        -max_articles:
    ]

    selected_indices = []

    seen_articles = set()

    for (
        timestamp,
        article_id,
    ) in reversed(eligible):

        if article_id in seen_articles:
            continue

        index = article_to_index.get(
            article_id
        )

        if index is None:
            continue

        seen_articles.add(
            article_id
        )

        selected_indices.append(
            index
        )

    if not selected_indices:
        return None

    query = article_embeddings[
        selected_indices
    ].mean(
        axis=0
    )

    norm = np.linalg.norm(
        query
    )

    if norm == 0:
        return None

    return (
        query / norm
    ).astype(
        np.float32
    )


def score_candidates(
    query_embedding: np.ndarray | None,
    candidate_ids,
    article_to_index: dict,
    article_embeddings: np.ndarray,
) -> list[tuple]:

    if not candidate_ids:
        return []

    if query_embedding is None:

        return [
            (
                article_id,
                0.0,
            )
            for article_id in candidate_ids
            if article_id in article_to_index
        ]

    scored = []

    for article_id in candidate_ids:

        index = article_to_index.get(
            article_id
        )

        if index is None:
            continue

        similarity = float(
            np.dot(
                query_embedding,
                article_embeddings[index],
            )
        )

        scored.append(
            (
                article_id,
                similarity,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[1],
            str(x[0]),
        )
    )

    return scored

def score_candidates_vectorized(
    query_embedding: np.ndarray | None,
    candidate_ids,
    article_to_index: dict,
    article_embeddings: np.ndarray,
) -> list[tuple]:
    """
    Vectorized semantic scoring over ONLY the candidate set.

    Because both query and article embeddings are normalized,
    the dot product is cosine similarity.
    """

    if not candidate_ids:
        return []

    valid_candidates = []
    candidate_indices = []

    for article_id in candidate_ids:

        index = article_to_index.get(
            article_id
        )

        if index is None:
            continue

        valid_candidates.append(
            article_id
        )

        candidate_indices.append(
            index
        )

    if not valid_candidates:
        return []

    if query_embedding is None:

        return [
            (
                article_id,
                0.0,
            )
            for article_id in valid_candidates
        ]

    candidate_matrix = (
        article_embeddings[
            candidate_indices
        ]
    )

    scores = candidate_matrix @ query_embedding

    ranked = [
        (
            article_id,
            float(score),
        )
        for article_id, score
        in zip(
            valid_candidates,
            scores,
        )
    ]

    ranked.sort(
        key=lambda x: (
            -x[1],
            str(x[0]),
        )
    )

    return ranked


def evaluate_split(
    split: str,
    article_embeddings: np.ndarray,
    article_to_index: dict,
    articles: pd.DataFrame,
    max_impressions: int | None = None,
):
    """
    Evaluate semantic ranking over EB-NeRD's actual
    impression candidate sets.
    """

    interactions = load_interactions(
        split
    )

    history = load_history(
        split
    )

    history_lookup = build_history_lookup(
        history
    )

    impression_groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    hits_5 = 0
    hits_10 = 0
    reciprocal_rank_sum = 0.0

    evaluated = 0
    total_candidates = 0

    start = time.perf_counter()

    for impression_id, group in impression_groups:

        if (
            max_impressions is not None
            and evaluated >= max_impressions
        ):
            break

        group = group.sort_values(
            "article_id"
        )

        user_id = group.iloc[0][
            "user_id"
        ]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidates = (
            group["article_id"]
            .tolist()
        )

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        user_history = history_lookup.get(
            user_id,
            [],
        )

        query_embedding = (
            build_query_embedding(
                user_history,
                impression_time,
                article_to_index,
                article_embeddings,
            )
        )

        ranked = score_candidates_vectorized(
            query_embedding,
            candidates,
            article_to_index,
            article_embeddings,
        )

        ranked_ids = [
            article_id
            for article_id, score
            in ranked
        ]

        total_candidates += len(
            ranked_ids
        )

        if clicked and any(
            article_id in clicked
            for article_id in ranked_ids[:5]
        ):
            hits_5 += 1

        if clicked and any(
            article_id in clicked
            for article_id in ranked_ids[:10]
        ):
            hits_10 += 1

        reciprocal_rank = 0.0

        for rank, article_id in enumerate(
            ranked_ids,
            start=1,
        ):

            if article_id in clicked:

                reciprocal_rank = (
                    1.0 / rank
                )

                break

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        evaluated += 1

        if evaluated % 10_000 == 0:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{split}: "
                f"{evaluated:,} impressions | "
                f"{elapsed:.1f}s"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    if evaluated == 0:

        raise ValueError(
            "No impressions evaluated."
        )

    return {
        "split": split,
        "impressions": evaluated,
        "hit_at_5": hits_5 / evaluated,
        "hit_at_10": hits_10 / evaluated,
        "mrr": reciprocal_rank_sum / evaluated,
        "mean_candidates": (
            total_candidates / evaluated
        ),
        "runtime_seconds": elapsed,
    }


def main():

    print("=" * 80)
    print("EB-NeRD SMALL — SEMANTIC RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load article corpus
    # --------------------------------------------------------------

    print("\nLoading article corpus...")

    articles = load_articles()

    print(
        f"Articles: {len(articles):,}"
    )

    # --------------------------------------------------------------
    # Load Sentence-Transformer
    # --------------------------------------------------------------

    print(
        "\nLoading Sentence-Transformer..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------------
    # Load or build article embeddings
    # --------------------------------------------------------------

    print(
        "\nBuilding/loading article embeddings..."
    )

    article_embeddings = (
        load_or_build_embeddings(
            articles,
            model,
        )
    )

    print(
        f"Embedding shape: "
        f"{article_embeddings.shape}"
    )

    # --------------------------------------------------------------
    # Article ID → embedding index
    # --------------------------------------------------------------

    article_to_index = {
        article_id: index
        for index, article_id
        in enumerate(
            articles["article_id"]
        )
    }

    # --------------------------------------------------------------
    # FULL TRAIN EVALUATION
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAIN FULL EVALUATION"
    )

    print(
        "=" * 80
    )

    train_results = evaluate_split(
        "train",
        article_embeddings,
        article_to_index,
        articles,
        max_impressions=None,
    )

    print(
        "\nTRAIN RESULTS"
    )

    print(
        f"Impressions: "
        f"{train_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{train_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{train_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{train_results['mrr']:.6f}"
    )

    print(
        f"Mean candidates: "
        f"{train_results['mean_candidates']:.2f}"
    )

    print(
        f"Runtime:     "
        f"{train_results['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------------
    # FULL VALIDATION EVALUATION
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION FULL EVALUATION"
    )

    print(
        "=" * 80
    )

    validation_results = evaluate_split(
        "validation",
        article_embeddings,
        article_to_index,
        articles,
        max_impressions=None,
    )

    print(
        "\nVALIDATION RESULTS"
    )

    print(
        f"Impressions: "
        f"{validation_results['impressions']:,}"
    )

    print(
        f"Hit@5:       "
        f"{validation_results['hit_at_5']:.6f}"
    )

    print(
        f"Hit@10:      "
        f"{validation_results['hit_at_10']:.6f}"
    )

    print(
        f"MRR:         "
        f"{validation_results['mrr']:.6f}"
    )

    print(
        f"Mean candidates: "
        f"{validation_results['mean_candidates']:.2f}"
    )

    print(
        f"Runtime:     "
        f"{validation_results['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------------
    # Validate impression counts
    # --------------------------------------------------------------

    expected_train_impressions = 232_887

    expected_validation_impressions = 244_647

    if (
        train_results["impressions"]
        != expected_train_impressions
    ):
        raise AssertionError(
            "Unexpected train impression count: "
            f"{train_results['impressions']:,}; "
            f"expected "
            f"{expected_train_impressions:,}"
        )

    if (
        validation_results["impressions"]
        != expected_validation_impressions
    ):
        raise AssertionError(
            "Unexpected validation impression count: "
            f"{validation_results['impressions']:,}; "
            f"expected "
            f"{expected_validation_impressions:,}"
        )

    # --------------------------------------------------------------
    # Save full results
    # --------------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            train_results,
            validation_results,
        ]
    )

    output_path = (
        RESULT_ROOT
        / "semantic_full.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Full semantic results saved to:"
    )

    print(
        output_path
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()