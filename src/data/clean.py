from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the root directory of the News-Recommender project.

    This file is located at:
        project_root/src/data/clean.py

    Therefore:
        parent     -> src/data
        parents[1] -> src
        parents[2] -> project root
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# MIND paths
# ---------------------------------------------------------------------

def get_mind_paths(
    project_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Return the paths for MIND raw data and processed output.

    Returns
    -------
    mind_raw_dir:
        Directory containing raw MIND files.

    mind_train_dir:
        Directory containing the MIND training files.

    mind_processed_dir:
        Directory where cleaned Parquet files will be written.
    """

    if project_root is None:
        project_root = get_project_root()

    mind_raw_dir = project_root / "data" / "raw" / "mind"
    mind_train_dir = mind_raw_dir / "train"
    mind_processed_dir = (
        project_root / "data" / "processed" / "mind"
    )

    return (
        mind_raw_dir,
        mind_train_dir,
        mind_processed_dir,
    )


# ---------------------------------------------------------------------
# Load news.tsv
# ---------------------------------------------------------------------

def load_mind_news(news_path: Path) -> pd.DataFrame:
    """
    Load and clean the MIND news.tsv file.

    MIND news.tsv contains eight fields:

        0. news ID
        1. category
        2. subcategory
        3. title
        4. abstract
        5. URL
        6. title entities
        7. abstract entities
    """

    if not news_path.exists():
        raise FileNotFoundError(
            f"MIND news file not found: {news_path}"
        )

    news = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    expected_columns = 8

    if news.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in news.tsv, "
            f"but found {news.shape[1]}."
        )

    news.columns = [
        "article_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    # Convert empty strings into missing values for optional fields.
    optional_columns = [
        "abstract",
        "title_entities",
        "abstract_entities",
    ]

    for column in optional_columns:
        news[column] = news[column].replace("", pd.NA)

    # The title is mandatory for our article representation.
    if news["article_id"].isna().any():
        raise ValueError(
            "Found missing article IDs in news.tsv."
        )

    if news["title"].isna().any():
        raise ValueError(
            "Found missing article titles in news.tsv."
        )

    # Article IDs must uniquely identify articles.
    duplicate_count = news["article_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate article IDs."
        )

    # Missing abstracts are allowed.
    news["abstract"] = news["abstract"].fillna("")

    # Build the text representation used later by BM25,
    # TF-IDF, and semantic models.
    news["text"] = (
        news["title"].fillna("")
        + " "
        + news["abstract"].fillna("")
    ).str.strip()

    # Every article should have usable text because title is mandatory.
    empty_text_count = news["text"].str.strip().eq("").sum()

    if empty_text_count > 0:
        raise ValueError(
            f"Found {empty_text_count} articles with empty text."
        )

    return news


# ---------------------------------------------------------------------
# Load behaviors.tsv
# ---------------------------------------------------------------------

def load_mind_behaviors(
    behaviors_path: Path,
) -> pd.DataFrame:
    """
    Load the MIND behaviors.tsv file.

    MIND behaviors.tsv contains:

        0. impression ID
        1. user ID
        2. timestamp
        3. history
        4. impressions
    """

    if not behaviors_path.exists():
        raise FileNotFoundError(
            f"MIND behaviors file not found: {behaviors_path}"
        )

    behaviors = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    expected_columns = 5

    if behaviors.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in behaviors.tsv, "
            f"but found {behaviors.shape[1]}."
        )

    behaviors.columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "impressions",
    ]

    # Required fields.
    required_columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "impressions",
    ]

    for column in required_columns:
        if behaviors[column].isna().any():
            raise ValueError(
                f"Required column '{column}' contains missing values."
            )

    # Convert timestamp to an actual datetime.
    behaviors["timestamp"] = pd.to_datetime(
        behaviors["timestamp"],
        errors="raise",
    )

    # A missing history means no observed previous click history.
    # We represent it as an empty string so that it can later become
    # an empty list of clicks.
    behaviors["history"] = behaviors["history"].fillna("")

    return behaviors


# ---------------------------------------------------------------------
# Parse impressions
# ---------------------------------------------------------------------

def parse_impressions(
    behaviors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the compact MIND impression representation into
    candidate-level interaction rows.

    Example raw value:

        N123-0 N456-1 N789-0

    becomes:

        article_id    clicked
        N123          0
        N456          1
        N789          0
    """

    rows = []

    for row in behaviors.itertuples(index=False):
        impression_items = row.impressions.split()

        if not impression_items:
            raise ValueError(
                f"Impression {row.impression_id} contains no candidates."
            )

        for item in impression_items:
            try:
                article_id, label = item.rsplit("-", 1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid impression item '{item}' "
                    f"in impression {row.impression_id}."
                ) from exc

            if label not in {"0", "1"}:
                raise ValueError(
                    f"Invalid click label '{label}' "
                    f"in impression {row.impression_id}."
                )

            rows.append(
                {
                    "impression_id": row.impression_id,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "article_id": article_id,
                    "clicked": int(label),
                }
            )

    interactions = pd.DataFrame(rows)

    if interactions.empty:
        raise ValueError(
            "No candidate interactions were produced."
        )

    return interactions


# ---------------------------------------------------------------------
# Parse user click history
# ---------------------------------------------------------------------

def parse_user_history(
    behaviors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert each user's historical clicked article IDs into
    individual rows.

    The timestamp assigned to each history event is the timestamp
    of the impression at which that history was observed.

    This representation is useful for constructing historical
    behaviour features while enforcing temporal boundaries.
    """

    rows = []

    for row in behaviors.itertuples(index=False):
        if not row.history:
            continue

        history_articles = row.history.split()

        for position, article_id in enumerate(history_articles):
            rows.append(
                {
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "article_id": article_id,
                    "history_position": position,
                    "impression_id": row.impression_id,
                }
            )

    history = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "timestamp",
            "article_id",
            "history_position",
            "impression_id",
        ],
    )

    return history


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_data(
    articles: pd.DataFrame,
    interactions: pd.DataFrame,
    user_history: pd.DataFrame,
) -> None:
    """
    Perform integrity checks before writing processed data.
    """

    # ---------------------------------------------------------------
    # Article uniqueness
    # ---------------------------------------------------------------

    if articles["article_id"].duplicated().any():
        raise ValueError(
            "articles contains duplicate article IDs."
        )

    article_ids = set(articles["article_id"])

    # ---------------------------------------------------------------
    # Candidate article integrity
    # ---------------------------------------------------------------

    interaction_article_ids = set(
        interactions["article_id"]
    )

    missing_interaction_articles = (
        interaction_article_ids - article_ids
    )

    if missing_interaction_articles:
        sample = list(
            missing_interaction_articles
        )[:10]

        raise ValueError(
            "Interactions contain article IDs that are not "
            f"present in articles. Example: {sample}"
        )

    # ---------------------------------------------------------------
    # History article integrity
    # ---------------------------------------------------------------

    if not user_history.empty:
        history_article_ids = set(
            user_history["article_id"]
        )

        missing_history_articles = (
            history_article_ids - article_ids
        )

        if missing_history_articles:
            sample = list(
                missing_history_articles
            )[:10]

            raise ValueError(
                "User history contains article IDs that are "
                f"not present in articles. Example: {sample}"
            )

    # ---------------------------------------------------------------
    # Click labels
    # ---------------------------------------------------------------

    valid_labels = {0, 1}

    if not set(interactions["clicked"].unique()).issubset(
        valid_labels
    ):
        raise ValueError(
            "Interactions contain click labels other than 0 or 1."
        )

    # ---------------------------------------------------------------
    # Every impression must have at least one candidate.
    # ---------------------------------------------------------------

    candidate_counts = (
        interactions
        .groupby("impression_id")
        .size()
    )

    if (candidate_counts < 1).any():
        raise ValueError(
            "Found an impression with no candidate articles."
        )

    # ---------------------------------------------------------------
    # Every impression should have at least one click in MIND train.
    # ---------------------------------------------------------------

    clicks_per_impression = (
        interactions
        .groupby("impression_id")["clicked"]
        .sum()
    )

    zero_click_impressions = (
        clicks_per_impression == 0
    ).sum()

    if zero_click_impressions > 0:
        raise ValueError(
            f"Found {zero_click_impressions} impressions "
            "with zero clicked candidates."
        )

    # ---------------------------------------------------------------
    # Timestamp validity
    # ---------------------------------------------------------------

    if interactions["timestamp"].isna().any():
        raise ValueError(
            "Interactions contain missing timestamps."
        )


# ---------------------------------------------------------------------
# Main MIND cleaning pipeline
# ---------------------------------------------------------------------

def clean_mind(
    project_root: Path | None = None,
) -> dict[str, Path]:
    """
    Run the complete MIND cleaning pipeline.

    Returns
    -------
    dict
        Paths to the generated Parquet files.
    """

    (
        _mind_raw_dir,
        mind_train_dir,
        mind_processed_dir,
    ) = get_mind_paths(project_root)

    mind_processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    news_path = mind_train_dir / "news.tsv"
    behaviors_path = mind_train_dir / "behaviors.tsv"

    print("=" * 70)
    print("MIND CLEANING PIPELINE")
    print("=" * 70)

    print("\n[1/6] Loading news.tsv...")
    articles = load_mind_news(news_path)

    print(
        f"       Articles loaded: {len(articles):,}"
    )

    print("\n[2/6] Loading behaviors.tsv...")
    behaviors = load_mind_behaviors(
        behaviors_path
    )

    print(
        f"       Impressions loaded: {len(behaviors):,}"
    )

    print("\n[3/6] Parsing impressions...")
    interactions = parse_impressions(
        behaviors
    )

    print(
        f"       Candidate interactions: "
        f"{len(interactions):,}"
    )

    print("\n[4/6] Parsing user history...")
    user_history = parse_user_history(
        behaviors
    )

    print(
        f"       History events: "
        f"{len(user_history):,}"
    )

    print("\n[5/6] Validating data...")
    validate_data(
        articles,
        interactions,
        user_history,
    )

    print("       Validation passed.")

    print("\n[6/6] Writing Parquet files...")

    articles_path = (
        mind_processed_dir / "articles.parquet"
    )

    interactions_path = (
        mind_processed_dir / "interactions.parquet"
    )

    history_path = (
        mind_processed_dir / "user_history.parquet"
    )

    articles.to_parquet(
        articles_path,
        index=False,
    )

    interactions.to_parquet(
        interactions_path,
        index=False,
    )

    user_history.to_parquet(
        history_path,
        index=False,
    )

    print(
        f"       {articles_path}"
    )

    print(
        f"       {interactions_path}"
    )

    print(
        f"       {history_path}"
    )

    print("\n" + "=" * 70)
    print("MIND CLEANING COMPLETE")
    print("=" * 70)

    return {
        "articles": articles_path,
        "interactions": interactions_path,
        "user_history": history_path,
    }


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    clean_mind()