from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.competition_metrics import (
    evaluate_competition_ranking,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data/results/mind/test_predictions.parquet"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data/processed/mind/splits/test.parquet"
)


def main() -> None:

    print("=" * 70)
    print("MIND FULL TEST OFFLINE EVALUATION")
    print("=" * 70)

    # ========================================================
    # 1. Load predictions
    # ========================================================

    print("\n[1/4] Loading predictions...")

    predictions = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=[
            "impression_id",
            "article_id",
            "hybrid_score",
            "rank",
        ],
    )

    print(
        f"       Prediction rows: "
        f"{len(predictions):,}"
    )

    print(
        f"       Impressions: "
        f"{predictions['impression_id'].nunique():,}"
    )

    # ========================================================
    # 2. Load labels
    # ========================================================

    print("\n[2/4] Loading test labels...")

    labels = pd.read_parquet(
        TEST_PATH,
        columns=[
            "impression_id",
            "article_id",
            "clicked",
        ],
    )

    print(
        f"       Label rows: "
        f"{len(labels):,}"
    )

    # ========================================================
    # 3. Normalize IDs and join
    # ========================================================

    print("\n[3/4] Joining labels...")

    predictions["impression_id"] = (
        predictions["impression_id"]
        .astype(str)
    )

    predictions["article_id"] = (
        predictions["article_id"]
        .astype(str)
    )

    labels["impression_id"] = (
        labels["impression_id"]
        .astype(str)
    )

    labels["article_id"] = (
        labels["article_id"]
        .astype(str)
    )

    evaluation = predictions.merge(
        labels,
        on=[
            "impression_id",
            "article_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if evaluation["clicked"].isna().any():

        missing = int(
            evaluation["clicked"].isna().sum()
        )

        raise ValueError(
            f"{missing:,} prediction rows "
            "could not be matched to labels."
        )

    print(
        f"       Evaluation rows: "
        f"{len(evaluation):,}"
    )

    # ========================================================
    # 4. Evaluate each impression
    # ========================================================

    print("\n[4/4] Computing ranking metrics...")

    metric_rows = []

    total_impressions = (
        evaluation["impression_id"]
        .nunique()
    )

    for counter, (
        impression_id,
        group,
    ) in enumerate(
        evaluation.groupby(
            "impression_id",
            sort=False,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # Ensure predictions are ranked correctly.
        # ----------------------------------------------------

        group = group.sort_values(
            "rank"
        )

        results = [
            (
                str(article_id),
                float(score),
            )
            for article_id, score in zip(
                group["article_id"],
                group["hybrid_score"],
            )
        ]

        # ----------------------------------------------------
        # Ground-truth clicked articles.
        # ----------------------------------------------------

        relevant_articles = (
            group.loc[
                group["clicked"] == 1,
                "article_id",
            ]
            .astype(str)
            .tolist()
        )

        metrics = evaluate_competition_ranking(
            results,
            relevant_articles,
        )

        metrics["impression_id"] = (
            impression_id
        )

        metric_rows.append(
            metrics
        )

        if counter % 5000 == 0:

            print(
                f"       Evaluated "
                f"{counter:,}/"
                f"{total_impressions:,} impressions"
            )

    # ========================================================
    # Aggregate
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metric_columns = [
        "auc",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    print("\n" + "=" * 70)
    print("MIND FULL TEST RESULTS")
    print("=" * 70)

    for column in metric_columns:

        print(
            f"{column:15s}: "
            f"{metrics_df[column].mean():.6f}"
        )

    print(
        f"\nImpressions evaluated: "
        f"{len(metrics_df):,}"
    )

    print("=" * 70)

    # ========================================================
    # Save per-impression metrics
    # ========================================================

    output_path = (
        PROJECT_ROOT
        / "data/results/mind/"
        / "test_metrics_by_impression.parquet"
    )

    metrics_df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nPer-impression metrics saved:"
    )

    print(
        f"       {output_path}"
    )


if __name__ == "__main__":
    main()