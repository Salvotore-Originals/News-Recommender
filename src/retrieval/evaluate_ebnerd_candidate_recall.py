from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


from src.ranking import candidates
from src.retrieval.ebnerd_bm25 import (
    build_history_lookup,
    build_query_from_history,
    load_history,
    load_interactions,
)

from src.retrieval.ebnerd_semantic import (
    build_history_lookup as build_semantic_history_lookup,
    build_query_embedding,
)

from src.retrieval.ebnerd_candidate_generation import (
    EBNeRDCandidateGenerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
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

RESULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "ebnerd"
    / "small"
)

SMOKE_IMPRESSIONS = 100

BM25_TOP_K = 100

SEMANTIC_TOP_K = 100

RECALL_KS = [
    50,
    100,
    200,
    500,
]

RRF_K = 60


def load_articles() -> pd.DataFrame:
    return pd.read_parquet(
        FEATURE_ROOT / "articles.parquet"
    )


def load_embeddings(
    articles: pd.DataFrame,
) -> np.ndarray:
    embedding_path = (
        EMBEDDING_ROOT
        / "article_embeddings.npy"
    )

    ids_path = (
        EMBEDDING_ROOT
        / "article_ids.parquet"
    )

    embeddings = np.load(
        embedding_path
    )

    saved_ids = pd.read_parquet(
        ids_path
    )["article_id"].tolist()

    current_ids = (
        articles["article_id"]
        .tolist()
    )

    if saved_ids != current_ids:
        raise ValueError(
            "Embedding article IDs do not "
            "match the article corpus."
        )

    if embeddings.shape[0] != len(
        articles
    ):
        raise ValueError(
            "Embedding count does not match "
            "article count."
        )

    return embeddings.astype(
        np.float32,
        copy=False,
    )


def evaluate_smoke(
    interactions: pd.DataFrame,
    history_lookup: dict,
    semantic_history_lookup: dict,
    article_embeddings: np.ndarray,
    article_to_index: dict,
    candidate_generator: EBNeRDCandidateGenerator,
) -> dict:

    impression_groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    recall_hits = {
        k: 0
        for k in RECALL_KS
    }

    evaluated = 0

    impressions_with_click = 0

    total_generated = 0

    future_article_violations = 0

    generation_times = []

    start = time.perf_counter()

    for impression_id, group in impression_groups:

        if evaluated >= SMOKE_IMPRESSIONS:
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

        clicked = set(
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
        )

        # ----------------------------------------------------------
        # BM25 query
        # ----------------------------------------------------------

        user_history = history_lookup.get(
            user_id,
            [],
        )

        query_tokens = (
            build_query_from_history(
                user_history,
                impression_time,
            )
        )

        query = " ".join(
            query_tokens
        )

        # ----------------------------------------------------------
        # Semantic query
        # ----------------------------------------------------------

        semantic_history = (
            semantic_history_lookup.get(
                user_id,
                [],
            )
        )

        query_embedding = (
            build_query_embedding(
                semantic_history,
                impression_time,
                article_to_index,
                article_embeddings,
            )
        )

        # ----------------------------------------------------------
        # Cold-start handling
        # ----------------------------------------------------------

        if query_embedding is None:
            evaluated += 1
            continue

        # ----------------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------------

        generation_start = (
            time.perf_counter()
        )

        candidates = (
            candidate_generator.retrieve(
                query=query,
                query_embedding=query_embedding,
                impression_time=impression_time,
                final_top_k=max(RECALL_KS),
            )
        )

        generation_elapsed = (
            time.perf_counter()
            - generation_start
        )

        generation_times.append(
            generation_elapsed
        )

        candidate_ids = set(
            candidates["article_id"]
        )

        total_generated += len(
            candidate_ids
        )

        # ----------------------------------------------------------
        # Temporal leakage check
        # ----------------------------------------------------------

        if not candidates.empty:

            article_metadata = candidate_generator.articles[
                [
                    "article_id",
                    "published_time",
                ]
            ].copy()

            article_metadata["article_id"] = (
                article_metadata["article_id"].astype(str)
            )

            candidate_metadata = (
                candidates.merge(
                    article_metadata,
                    on="article_id",
                    how="left",
                )
            )

            publication_times = pd.to_datetime(
                candidate_metadata[
                    "published_time"
                ],
                errors="coerce",
            )

            violations = (
                publication_times
                > pd.Timestamp(
                    impression_time
                )
            )

            future_article_violations += int(
                violations.sum()
            )

        # ----------------------------------------------------------
        # Recall
        # ----------------------------------------------------------

        if clicked:

            impressions_with_click += 1

            for k in RECALL_KS:

                top_k_ids = set(
                    candidates.head(k)[
                        "article_id"
                    ]
                )

                if clicked & top_k_ids:
                    recall_hits[k] += 1

        evaluated += 1

        if evaluated % 25 == 0:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"Evaluated: {evaluated}/{SMOKE_IMPRESSIONS} "
                f"| elapsed={elapsed:.2f}s"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    if evaluated == 0:
        raise ValueError(
            "No impressions were evaluated."
        )

    results = {
        "impressions": evaluated,
        "impressions_with_click": (
            impressions_with_click
        ),
        "mean_candidates": (
            total_generated / evaluated
        ),
        "future_article_violations": (
            future_article_violations
        ),
        "mean_generation_time_seconds": (
            float(
                np.mean(
                    generation_times
                )
            )
            if generation_times
            else 0.0
        ),
        "total_runtime_seconds": elapsed,
    }

    for k in RECALL_KS:

        denominator = (
            impressions_with_click
        )

        results[
            f"recall_at_{k}"
        ] = (
            recall_hits[k] / denominator
            if denominator
            else 0.0
        )

    return results


def main():

    print("=" * 80)
    print(
        "EB-NeRD SMALL — CANDIDATE GENERATION "
        "RECALL SMOKE TEST"
    )
    print("=" * 80)

    # --------------------------------------------------------------
    # Load articles
    # --------------------------------------------------------------

    print(
        "\nLoading article corpus..."
    )

    articles = load_articles()

    print(
        f"Articles: {len(articles):,}"
    )

    # --------------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------------

    print(
        "\nLoading semantic embeddings..."
    )

    article_embeddings = load_embeddings(
        articles
    )

    print(
        f"Embedding shape: "
        f"{article_embeddings.shape}"
    )

    # --------------------------------------------------------------
    # Build article index
    # --------------------------------------------------------------

    article_to_index = {
        article_id: index
        for index, article_id
        in enumerate(
            articles["article_id"]
        )
    }

    # --------------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------------

    print(
        "\nLoading validation interactions..."
    )

    interactions = load_interactions(
        "validation"
    )

    # Candidate generator uses canonical string article IDs.
    # Normalize validation IDs to the same representation.
    interactions["article_id"] = (
        interactions["article_id"].astype(str)
    )

    history = load_history(
        "validation"
    )

    print(
        f"Interaction rows: "
        f"{len(interactions):,}"
    )

    print(
        f"Validation impressions: "
        f"{interactions['impression_id'].nunique():,}"
    )

    # --------------------------------------------------------------
    # Build history lookups
    # --------------------------------------------------------------

    print(
        "\nBuilding history lookups..."
    )

    history_lookup = (
        build_history_lookup(
            history,
            articles,
        )
    )

    semantic_history_lookup = (
        build_semantic_history_lookup(
            history
        )
    )

    print(
        f"BM25 history users: "
        f"{len(history_lookup):,}"
    )

    print(
        f"Semantic history users: "
        f"{len(semantic_history_lookup):,}"
    )

    # --------------------------------------------------------------
    # Build reusable candidate generator
    # --------------------------------------------------------------

    print(
        "\nBuilding reusable candidate generator..."
    )

    generator_start = (
        time.perf_counter()
    )

    candidate_generator = (
        EBNeRDCandidateGenerator(
            articles,
            article_embeddings,
            bm25_top_k=BM25_TOP_K,
            semantic_top_k=SEMANTIC_TOP_K,
            rrf_k=RRF_K,
        )
    )

    generator_build_time = (
        time.perf_counter()
        - generator_start
    )

    print(
        "Candidate generator built in "
        f"{generator_build_time:.2f}s"
    )

    # --------------------------------------------------------------
    # Run smoke evaluation
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        f"Evaluating first "
        f"{SMOKE_IMPRESSIONS} validation impressions..."
    )

    print(
        "=" * 80
    )

    results = evaluate_smoke(
        interactions,
        history_lookup,
        semantic_history_lookup,
        article_embeddings,
        article_to_index,
        candidate_generator,
    )


    results[
        "generator_build_time_seconds"
    ] = generator_build_time

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "CANDIDATE GENERATION RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Impressions evaluated: "
        f"{results['impressions']:,}"
    )

    print(
        f"Impressions with click: "
        f"{results['impressions_with_click']:,}"
    )

    print(
        f"Mean candidates: "
        f"{results['mean_candidates']:.2f}"
    )

    print(
        f"Recall@50:  "
        f"{results['recall_at_50']:.6f}"
    )

    print(
        f"Recall@100: "
        f"{results['recall_at_100']:.6f}"
    )

    print(
        f"Recall@200: "
        f"{results['recall_at_200']:.6f}"
    )

    print(
        f"Recall@500: "
        f"{results['recall_at_500']:.6f}"
    )

    print(
        f"Future article violations: "
        f"{results['future_article_violations']}"
    )

    print(
        f"Mean generation time: "
        f"{results['mean_generation_time_seconds']:.4f}s"
    )

    print(
        f"Total runtime: "
        f"{results['total_runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------------
    # Hard correctness assertion
    # --------------------------------------------------------------

    if results[
        "future_article_violations"
    ] != 0:

        raise AssertionError(
            "Temporal leakage detected: "
            f"{results['future_article_violations']} "
            "future articles entered the candidate pool."
        )

    # --------------------------------------------------------------
    # Save result
    # --------------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_ROOT
        / "candidate_recall_smoke.csv"
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        output_path,
        index=False,
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()