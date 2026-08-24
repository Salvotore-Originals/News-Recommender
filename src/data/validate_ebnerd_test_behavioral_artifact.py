from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ebnerd"
    / "ebnerd_testset"
    / "ebnerd_testset"
)

ARTICLES_PATH = TEST_ROOT / "articles.parquet"
HISTORY_PATH = TEST_ROOT / "test" / "history.parquet"
BEHAVIORS_PATH = TEST_ROOT / "test" / "behaviors.parquet"

ARTIFACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "test"
    / "behavioral"
    / "behavioral_history.parquet"
)


def load_category_lookup():
    articles = pd.read_parquet(
        ARTICLES_PATH,
        columns=["article_id", "category"],
    )

    return {
        str(row.article_id): str(row.category)
        for row in articles.itertuples(index=False)
        if not pd.isna(row.category)
    }


def load_target_behaviors(n=100):
    return pd.read_parquet(
        BEHAVIORS_PATH,
        columns=[
            "impression_id",
            "impression_time",
            "article_ids_inview",
            "user_id",
        ],
    ).head(n)


def build_direct_reference(
    history,
    target_users,
    article_categories,
):
    """
    Build the reference behavioral scores directly from the
    official array-based history. This is intentionally the
    same simple logic used by the targeted smoke test.
    """

    result = {}

    relevant = history[
        history["user_id"].astype(str).isin(target_users)
    ]

    for row in relevant[
        [
            "user_id",
            "impression_time_fixed",
            "article_id_fixed",
        ]
    ].itertuples(index=False):

        user_id = str(row.user_id)

        timestamps = row.impression_time_fixed
        article_ids = row.article_id_fixed

        events = []

        for timestamp, article_id in zip(
            timestamps,
            article_ids,
        ):
            category = article_categories.get(
                str(article_id)
            )

            if category is not None:
                events.append(
                    (
                        pd.Timestamp(timestamp),
                        category,
                    )
                )

        events.sort(key=lambda x: x[0])

        result[user_id] = events

    return result


def direct_score(
    user_id,
    impression_time,
    candidate_ids,
    direct_history,
    article_categories,
):
    events = direct_history.get(
        user_id,
        [],
    )

    counts = {}
    total = 0

    for timestamp, category in events:

        # Critical temporal-leakage rule.
        if timestamp >= impression_time:
            break

        counts[category] = (
            counts.get(category, 0) + 1
        )
        total += 1

    ranked = []

    for article_id in candidate_ids:

        article_id = str(article_id)

        category = article_categories.get(
            article_id
        )

        score = (
            counts.get(category, 0) / total
            if category is not None and total > 0
            else 0.0
        )

        ranked.append(
            (article_id, float(score))
        )

    ranked.sort(
        key=lambda x: (-x[1], str(x[0]))
    )

    return ranked


def load_compressed_rows(target_users):
    """
    Read only the required user rows from the compressed artifact.
    """

    table = pq.read_table(
        ARTIFACT_PATH,
        filters=[
            (
                "user_id",
                "in",
                [int(user_id) for user_id in target_users],
            )
        ],
    )

    return table.to_pandas()


def compressed_score(
    user_id,
    impression_time,
    candidate_ids,
    compressed_row,
    article_categories,
):
    """
    Reconstruct the same category preference from the compressed
    representation.

    Artifact representation:

        timestamps_ns
        category_ids
        category_positions

    For each category we binary-search its event positions against
    the impression timestamp.
    """

    if compressed_row is None:
        counts = {}
        total = 0
    else:
        timestamps = [
            int(value)
            for value in compressed_row.timestamps_ns
        ]

        category_ids = [
            int(value)
            for value in compressed_row.category_ids
]

        category_positions = (
            compressed_row.category_positions
        )

        counts = {}
        total = 0

        impression_ns = pd.Timestamp(
            impression_time
        ).value

        for category_id, positions in zip(
            category_ids,
            category_positions,
        ):

            # Convert category positions to timestamps.
            category_timestamps = [
                timestamps[int(position)]
                for position in positions
            ]

            count = bisect_left(
                category_timestamps,
                impression_ns,
            )

            if count > 0:
                counts[category_id] = count
                total += count

    # Map article category string -> compressed integer ID.
    # Build this mapping outside this function in main.
    category_to_id = compressed_score.category_to_id

    ranked = []

    for article_id in candidate_ids:

        article_id = str(article_id)

        category = article_categories.get(
            article_id
        )

        category_id = (
            category_to_id.get(category)
            if category is not None
            else None
        )

        score = (
            counts.get(category_id, 0) / total
            if category_id is not None and total > 0
            else 0.0
        )

        ranked.append(
            (article_id, float(score))
        )

    ranked.sort(
        key=lambda x: (-x[1], str(x[0]))
    )

    return ranked


def main():

    print("=" * 80)
    print(
        "EB-NeRD OFFICIAL TEST — "
        "COMPRESSED BEHAVIORAL VALIDATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load test impressions
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading 100 reference impressions..."
    )

    behaviors = load_target_behaviors(100)

    target_users = {
        str(user_id)
        for user_id in behaviors["user_id"]
    }

    print(
        f"       Impressions: {len(behaviors):,}"
    )

    print(
        f"       Target users: {len(target_users):,}"
    )

    # --------------------------------------------------------
    # 2. Article categories
    # --------------------------------------------------------

    print(
        "\n[2/6] Loading article categories..."
    )

    article_categories = load_category_lookup()

    print(
        f"       Category mappings: "
        f"{len(article_categories):,}"
    )

    # --------------------------------------------------------
    # 3. Direct reference
    # --------------------------------------------------------

    print(
        "\n[3/6] Building direct reference scores..."
    )

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "user_id",
            "impression_time_fixed",
            "article_id_fixed",
        ],
    )

    direct_history = build_direct_reference(
        history,
        target_users,
        article_categories,
    )

    print(
        f"       Reference users: "
        f"{len(direct_history):,}"
    )

    # --------------------------------------------------------
    # 4. Load compressed artifact
    # --------------------------------------------------------

    print(
        "\n[4/6] Loading compressed artifact rows..."
    )

    compressed = load_compressed_rows(
        target_users
    )

    print(
        f"       Artifact rows loaded: "
        f"{len(compressed):,}"
    )

    compressed_by_user = {
        str(row.user_id): row
        for row in compressed.itertuples(
            index=False
        )
    }

    # Load category ID mapping from the artifact's compact
    # category representation.
    category_counts = {}

    artifact = pq.read_table(
        ARTIFACT_PATH,
        columns=["category_ids"],
    )

    # The category IDs themselves are global codes. Their exact
    # mapping is recovered from category_map.json.
    import json

    category_map_path = (
        ARTIFACT_PATH.parent
        / "category_map.json"
    )

    with open(
        category_map_path,
        "r",
        encoding="utf-8",
    ) as file:
        category_map = json.load(file)

    category_to_id = {
        str(category): int(category_id)
        for category, category_id
        in category_map["category_to_id"].items()
    }

    compressed_score.category_to_id = category_to_id

    # --------------------------------------------------------
    # 5. Compare
    # --------------------------------------------------------

    print(
        "\n[5/6] Comparing direct vs compressed rankings..."
    )

    impressions_checked = 0
    candidates_checked = 0
    max_score_error = 0.0
    ranking_mismatches = 0
    score_mismatches = 0

    for row in behaviors.itertuples(index=False):

        user_id = str(row.user_id)
        impression_time = pd.Timestamp(
            row.impression_time
        )

        candidate_ids = [
            str(article_id)
            for article_id in row.article_ids_inview
        ]

        reference = direct_score(
            user_id,
            impression_time,
            candidate_ids,
            direct_history,
            article_categories,
        )

        compressed_row = compressed_by_user.get(
            user_id
        )

        candidate_result = compressed_score(
            user_id,
            impression_time,
            candidate_ids,
            compressed_row,
            article_categories,
        )

        if len(reference) != len(candidate_result):
            raise AssertionError(
                f"Candidate count mismatch for "
                f"impression {row.impression_id}"
            )

        for left, right in zip(
            reference,
            candidate_result,
        ):

            candidates_checked += 1

            error = abs(
                left[1] - right[1]
            )

            max_score_error = max(
                max_score_error,
                error,
            )

            if left[0] != right[0]:
                ranking_mismatches += 1

            if error > 1e-12:
                score_mismatches += 1

        impressions_checked += 1

    # --------------------------------------------------------
    # 6. Result
    # --------------------------------------------------------

    print(
        "\n[6/6] Validation result..."
    )

    print(
        f"       Impressions checked: "
        f"{impressions_checked:,}"
    )

    print(
        f"       Candidates checked: "
        f"{candidates_checked:,}"
    )

    print(
        f"       Maximum score error: "
        f"{max_score_error:.16f}"
    )

    print(
        f"       Score mismatches: "
        f"{score_mismatches:,}"
    )

    print(
        f"       Ranking mismatches: "
        f"{ranking_mismatches:,}"
    )

    if score_mismatches != 0:
        raise AssertionError(
            "Compressed behavioral scores do not "
            "match the direct reference."
        )

    if ranking_mismatches != 0:
        raise AssertionError(
            "Compressed behavioral rankings do not "
            "match the direct reference."
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPRESSED BEHAVIORAL VALIDATION PASSED"
    )

    print(
        "The compressed artifact reproduces the "
        "direct behavioral scores and rankings."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
