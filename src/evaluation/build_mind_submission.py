from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = (
    PROJECT_ROOT
    / "data/results/mind/mind_large_test_predictions.parquet"
)

BEHAVIORS = (
    PROJECT_ROOT
    / "data/raw/mind/test/behaviors.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/results/mind/submission"
)

PREDICTION_TXT = OUTPUT_DIR / "prediction.txt"
PREDICTION_ZIP = OUTPUT_DIR / "prediction.zip"


def load_candidate_positions() -> dict[str, dict[str, int]]:
    """
    Build:

        impression_id -> article_id -> original candidate position

    MIND candidate positions are 1-based.
    """

    positions = {}

    with open(
        BEHAVIORS,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 5:
                continue

            impression_id = parts[0]
            candidates = parts[4].split()

            article_positions = {}

            for position, article_id in enumerate(
                candidates,
                start=1,
            ):
                article_positions[article_id] = position

            positions[impression_id] = article_positions

    return positions


def main() -> None:

    print("=" * 70)
    print("MIND OFFICIAL SUBMISSION BUILDER")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------

    print("\n[1/6] Loading predictions...")

    df = pd.read_parquet(
        PREDICTIONS,
        columns=[
            "impression_id",
            "article_id",
            "rank",
        ],
    )

    print(
        f"       Prediction rows: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Validate predictions
    # ------------------------------------------------------------

    print("\n[2/6] Validating predictions...")

    df["impression_id"] = df["impression_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    df["rank"] = df["rank"].astype(int)

    if df.duplicated(
        ["impression_id", "article_id"]
    ).any():
        raise ValueError(
            "Duplicate impression/article pairs found."
        )

    # ------------------------------------------------------------
    # Load original candidate positions
    # ------------------------------------------------------------

    print("\n[3/6] Loading original MIND candidates...")

    candidate_positions = load_candidate_positions()

    print(
        f"       Behavior impressions: "
        f"{len(candidate_positions):,}"
    )

    # ------------------------------------------------------------
    # Sort predictions by model rank
    # ------------------------------------------------------------

   # ------------------------------------------------------------
   # Sort predictions numerically by impression ID
   # ------------------------------------------------------------

    print("\n[4/6] Building official rank lists...")

    df["_impression_id_num"] = (
        df["impression_id"].astype(int)
    )

    df = df.sort_values(
        [
            "_impression_id_num",
            "rank",
        ],
        kind="stable",
    )

    grouped = df.groupby(
        "impression_id",
        sort=False,
    )

    # ------------------------------------------------------------
    # Write official MIND format
    # ------------------------------------------------------------

    print("\n[5/6] Writing prediction.txt...")

    impression_count = 0
    total_predictions = 0

    with open(
        PREDICTION_TXT,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for impression_id, group in grouped:

            if impression_id not in candidate_positions:
                raise ValueError(
                    f"Prediction impression {impression_id} "
                    f"not found in test behaviors."
                )

            article_position_map = candidate_positions[
                impression_id
            ]

            ranked_positions = []

            for article_id in group["article_id"]:

                if article_id not in article_position_map:
                    raise ValueError(
                        f"Article {article_id} for impression "
                        f"{impression_id} is not in the "
                        f"original candidate list."
                    )

                ranked_positions.append(
                    article_position_map[article_id]
                )

            # MIND official format:
            #
            # impression_id [position1,position2,...]
            #
            file.write(
                f"{impression_id} "
                f"[{','.join(map(str, ranked_positions))}]\n"
            )

            impression_count += 1
            total_predictions += len(ranked_positions)

    print(
        f"       Impressions written: "
        f"{impression_count:,}"
    )

    print(
        f"       Predictions written: "
        f"{total_predictions:,}"
    )

    # ------------------------------------------------------------
    # Create ZIP
    # ------------------------------------------------------------

    print("\n[6/6] Creating prediction.zip...")

    with zipfile.ZipFile(
        PREDICTION_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.write(
            PREDICTION_TXT,
            arcname="prediction.txt",
        )

    print()
    print(f"       TXT: {PREDICTION_TXT}")
    print(f"       ZIP: {PREDICTION_ZIP}")

    print("\n" + "=" * 70)
    print("MIND OFFICIAL SUBMISSION BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()