from pathlib import Path

from src.ranking.phase5_reranker_data import (
    Phase5Config,
    build_phase5_reranker_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
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


def main() -> None:
    """Build the Phase 5.1 validation retrieval-feature table."""

    config = Phase5Config(
        candidate_path=(
            FEATURE_ROOT
            / "validation_candidates.parquet"
        ),
        history_path=(
            PROCESSED_ROOT
            / "validation_user_history.parquet"
        ),
        article_path=(
            FEATURE_ROOT
            / "articles.parquet"
        ),
        embedding_path=(
            EMBEDDING_ROOT
            / "article_embeddings.npy"
        ),
        embedding_ids_path=(
            EMBEDDING_ROOT
            / "article_ids.parquet"
        ),
        output_path=(
            FEATURE_ROOT
            / "validation_phase5_reranker.parquet"
        ),
        max_impressions=None,
    )

    print("=" * 70)
    print("PHASE 5.1 — VALIDATION RETRIEVAL FEATURES")
    print("=" * 70)

    print(f"Candidate file : {config.candidate_path}")
    print(f"History file   : {config.history_path}")
    print(f"Article file   : {config.article_path}")
    print(f"Embedding file : {config.embedding_path}")
    print(f"Output file    : {config.output_path}")
    print()

    result = build_phase5_reranker_features(config)

    config.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        config.output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 5.1 VALIDATION COMPLETE")
    print("=" * 70)

    print(f"Rows        : {len(result):,}")
    print(
        f"Impressions : "
        f"{result['impression_id'].nunique():,}"
    )
    print(
        f"Clicks      : "
        f"{int(result['clicked'].sum()):,}"
    )
    print(f"Output      : {config.output_path}")

    print()
    print("Retrieval features:")

    for column in [
        "bm25_score",
        "bm25_rank",
        "semantic_score",
        "semantic_rank",
        "rrf_score",
    ]:
        print(f"  - {column}")

    print()
    print(
        result[
            [
                "bm25_score",
                "bm25_rank",
                "semantic_score",
                "semantic_rank",
                "rrf_score",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()