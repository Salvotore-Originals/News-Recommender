from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


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
TEST_HISTORY_PATH = TEST_ROOT / "test" / "history.parquet"
TEST_BEHAVIORS_PATH = TEST_ROOT / "test" / "behaviors.parquet"


def load_articles() -> pd.DataFrame:
    return pd.read_parquet(ARTICLES_PATH)


def load_test_history() -> pd.DataFrame:
    return pd.read_parquet(TEST_HISTORY_PATH)


def load_test_behaviors(
    nrows: int = 100,
) -> pd.DataFrame:
    """
    Load only the impressions required by the smoke test.
    """

    return pd.read_parquet(
        TEST_BEHAVIORS_PATH
    ).head(nrows)


def build_article_category_lookup(
    articles: pd.DataFrame,
) -> dict[str, str]:

    required = {
        "article_id",
        "category",
    }

    missing = required - set(articles.columns)

    if missing:
        raise ValueError(
            "Article table is missing columns: "
            f"{sorted(missing)}"
        )

    return {
        str(row.article_id): str(row.category)
        for row in articles[
            ["article_id", "category"]
        ].itertuples(index=False)
        if not pd.isna(row.category)
    }


def build_target_users(
    behaviors: pd.DataFrame,
) -> set[str]:

    return {
        str(user_id)
        for user_id in behaviors["user_id"].tolist()
    }


def build_user_category_preferences(
    history: pd.DataFrame,
    target_users: set[str],
    article_categories: dict[str, str],
) -> dict[str, dict[str, float]]:
    """
    Build category preferences only for users occurring in
    the smoke-test impressions.

    For each target user:

        preference(category)
          =
        historical clicks in category
        --------------------------------
        historical events with known category

    The official test history stores arrays per user, so those
    arrays are processed directly without expanding the complete
    history into a pandas DataFrame.
    """

    preferences = {}

    relevant = history[
        history["user_id"].astype(str).isin(
            target_users
        )
    ]

    print(
        f"       Matching history users: "
        f"{len(relevant):,}"
    )

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

        if timestamps is None or article_ids is None:
            preferences[user_id] = {}
            continue

        if len(timestamps) != len(article_ids):
            raise ValueError(
                "History array length mismatch "
                f"for user {user_id}: "
                f"{len(timestamps)} timestamps vs "
                f"{len(article_ids)} articles"
            )

        category_counts = defaultdict(int)
        total = 0

        for article_id in article_ids:

            category = article_categories.get(
                str(article_id)
            )

            if category is None:
                continue

            category_counts[category] += 1
            total += 1

        if total == 0:
            preferences[user_id] = {}
        else:
            preferences[user_id] = {
                category: count / total
                for category, count
                in category_counts.items()
            }

    return preferences


def score_candidates(
    user_id: str,
    candidate_ids,
    article_categories,
    user_preferences,
) -> list[tuple[str, float]]:

    preferences = user_preferences.get(
        user_id,
        {},
    )

    scored = []

    for article_id in candidate_ids:

        article_id = str(article_id)

        category = article_categories.get(
            article_id
        )

        score = preferences.get(
            category,
            0.0,
        )

        scored.append(
            (
                article_id,
                float(score),
            )
        )

    scored.sort(
        key=lambda x: (
            -x[1],
            str(x[0]),
        )
    )

    return scored


def run_smoke_test(
    max_impressions: int = 100,
) -> None:

    print("=" * 80)
    print(
        "EB-NeRD OFFICIAL TEST — "
        "TARGETED BEHAVIORAL SMOKE TEST"
    )
    print("=" * 80)

    # ========================================================
    # 1. ARTICLES
    # ========================================================

    print(
        "\n[1/5] Loading official test articles..."
    )

    articles = load_articles()

    print(
        f"       Articles: "
        f"{len(articles):,}"
    )

    article_categories = (
        build_article_category_lookup(
            articles
        )
    )

    print(
        f"       Category mappings: "
        f"{len(article_categories):,}"
    )

    # ========================================================
    # 2. BEHAVIORS — ONLY 100
    # ========================================================

    print(
        "\n[2/5] Loading first "
        f"{max_impressions:,} test impressions..."
    )

    behaviors = load_test_behaviors(
        nrows=max_impressions
    )

    print(
        f"       Impressions: "
        f"{len(behaviors):,}"
    )

    target_users = build_target_users(
        behaviors
    )

    print(
        f"       Target users: "
        f"{len(target_users):,}"
    )

    # ========================================================
    # 3. HISTORY — FILTER TO TARGET USERS
    # ========================================================

    print(
        "\n[3/5] Loading official test history..."
    )

    history = load_test_history()

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    print(
        "       Building preferences only for "
        f"{len(target_users):,} target users..."
    )

    user_preferences = (
        build_user_category_preferences(
            history,
            target_users,
            article_categories,
        )
    )

    print(
        f"       Preference users: "
        f"{len(user_preferences):,}"
    )

    # ========================================================
    # 4. SCORE
    # ========================================================

    print(
        "\n[4/5] Scoring test impressions..."
    )

    total_rows = 0

    for count, row in enumerate(
        behaviors.itertuples(index=False),
        start=1,
    ):

        user_id = str(row.user_id)

        impression_id = str(
            row.impression_id
        )

        candidate_ids = [
            str(article_id)
            for article_id
            in row.article_ids_inview
        ]

        ranked = score_candidates(
            user_id,
            candidate_ids,
            article_categories,
            user_preferences,
        )

        if len(ranked) != len(candidate_ids):
            raise RuntimeError(
                "Candidate count changed for "
                f"impression {impression_id}: "
                f"{len(candidate_ids)} -> "
                f"{len(ranked)}"
            )

        ranked_ids = [
            article_id
            for article_id, _ in ranked
        ]

        if len(ranked_ids) != len(
            set(ranked_ids)
        ):
            raise RuntimeError(
                "Duplicate article IDs after "
                f"ranking impression "
                f"{impression_id}"
            )

        total_rows += len(ranked)

        if count <= 3:

            print()
            print(
                f"       Impression {count}"
            )

            print(
                f"       ID: {impression_id}"
            )

            print(
                f"       User: {user_id}"
            )

            print(
                f"       Candidates: "
                f"{len(candidate_ids)}"
            )

            print(
                pd.DataFrame(
                    ranked[:10],
                    columns=[
                        "article_id",
                        "behavioral_score",
                    ],
                ).to_string(index=False)
            )

    # ========================================================
    # 5. COMPLETE
    # ========================================================

    print(
        "\n[5/5] Validation complete."
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "TARGETED BEHAVIORAL SMOKE TEST COMPLETE"
    )

    print(
        f"       Impressions scored: "
        f"{len(behaviors):,}"
    )

    print(
        f"       Candidates scored: "
        f"{total_rows:,}"
    )

    print(
        f"       Target users: "
        f"{len(target_users):,}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    run_smoke_test(
        max_impressions=100
    )
