from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.retrieval.mind_large_behavioral import (
    load_article_metadata,
    load_train_popularity,
    score_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEHAVIORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mind"
    / "test"
    / "behaviors.tsv"
)


def parse_behaviors(
    path: Path,
    limit: int,
) -> pd.DataFrame:

    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "impressions",
    ]

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        nrows=limit,
        dtype="string",
        keep_default_na=False,
    )

    return df


def main():

    print("=" * 70)
    print("MIND LARGE LABEL-FREE BEHAVIORAL SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load article metadata
    # --------------------------------------------------------

    print("\n[1/4] Loading article metadata...")

    (
        category_lookup,
        subcategory_lookup,
    ) = load_article_metadata()

    print(
        f"       Articles: "
        f"{len(category_lookup):,}"
    )

    # --------------------------------------------------------
    # 2. Load popularity
    # --------------------------------------------------------

    print("\n[2/4] Loading TRAIN popularity...")

    (
        popularity_lookup,
        global_ctr,
    ) = load_train_popularity()

    print(
        f"       Popularity entries: "
        f"{len(popularity_lookup):,}"
    )

    print(
        f"       Global CTR: "
        f"{global_ctr:.8f}"
    )

    # --------------------------------------------------------
    # 3. Load official test impressions
    # --------------------------------------------------------

    print("\n[3/4] Loading test impressions...")

    behaviors = parse_behaviors(
        BEHAVIORS_PATH,
        limit=100,
    )

    print(
        f"       Impressions: "
        f"{len(behaviors):,}"
    )

    # --------------------------------------------------------
    # 4. Score candidates
    # --------------------------------------------------------

    print("\n[4/4] Scoring candidates...")

    total_candidates = 0

    score_min = float("inf")
    score_max = float("-inf")

    scored_impressions = 0

    for row in behaviors.itertuples(
        index=False
    ):

        history = (
            str(row.history).split()
            if row.history
            else []
        )

        candidate_ids = [
            item.rsplit("-", 1)[0]
            for item in str(
                row.impressions
            ).split()
            if item
        ]

        scores = score_candidates(
            candidate_ids=candidate_ids,
            history=history,
            category_lookup=category_lookup,
            subcategory_lookup=subcategory_lookup,
            popularity_lookup=popularity_lookup,
            global_ctr=global_ctr,
        )

        total_candidates += len(
            candidate_ids
        )

        if len(scores) > 0:

            score_min = min(
                score_min,
                float(scores.min()),
            )

            score_max = max(
                score_max,
                float(scores.max()),
            )

        scored_impressions += 1

        if scored_impressions <= 3:

            preview = pd.DataFrame(
                {
                    "article_id":
                        candidate_ids,
                    "behavioral_score":
                        scores,
                }
            ).sort_values(
                "behavioral_score",
                ascending=False,
            )

            print(
                f"\n       Impression "
                f"{row.impression_id}"
            )

            print(
                f"       User: "
                f"{row.user_id}"
            )

            print(
                f"       History: "
                f"{len(history)}"
            )

            print(
                f"       Candidates: "
                f"{len(candidate_ids)}"
            )

            print(
                preview.head(10)
                .to_string(index=False)
            )

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)

    print(
        f"Impressions scored: "
        f"{scored_impressions:,}"
    )

    print(
        f"Candidates scored: "
        f"{total_candidates:,}"
    )

    print(
        f"Behavioural score range: "
        f"{score_min:.6f} "
        f"to "
        f"{score_max:.6f}"
    )

    if scored_impressions != 100:
        raise RuntimeError(
            "Expected exactly 100 impressions."
        )

    if total_candidates == 0:
        raise RuntimeError(
            "No candidates were scored."
        )

    if not (
        0.0 <= score_min <= 1.0
        and
        0.0 <= score_max <= 1.0
    ):
        raise RuntimeError(
            "Behavioural scores outside [0, 1]."
        )

    print("\n" + "=" * 70)
    print(
        "MIND LARGE BEHAVIORAL "
        "SMOKE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()