from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIND = (
    ROOT
    / "data"
    / "processed"
    / "mind"
)

FEATURE_MIND = (
    ROOT
    / "data"
    / "features"
    / "mind"
)

HISTORY_PATH = (
    PROCESSED_MIND
    / "user_history.parquet"
)

INTERACTIONS_PATH = (
    PROCESSED_MIND
    / "interactions.parquet"
)

TRAIN_ARTICLES_PATH = (
    FEATURE_MIND
    / "train_articles.parquet"
)


# ============================================================
# GLOBAL USER × CATEGORY PREFERENCE
# ============================================================

def build_user_category_preferences() -> pd.DataFrame:
    """
    Build global historical user-category preferences.

    This feature describes a user's overall historical
    category distribution.

    It is useful as a descriptive behavioural feature, but
    it should not be used directly for temporal evaluation
    because it aggregates history across all observations.
    """

    print("\nBuilding user-category preferences...")

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "user_id",
            "article_id",
        ],
    )

    print(
        f"       History rows: {len(history):,}"
    )

    articles = pd.read_parquet(
        TRAIN_ARTICLES_PATH,
        columns=[
            "article_id",
            "category",
        ],
    )

    print(
        f"       Articles: {len(articles):,}"
    )

    history_with_category = history.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_category = (
        history_with_category["category"]
        .isna()
        .sum()
    )

    print(
        f"       Missing category mappings: "
        f"{missing_category:,}"
    )

    if missing_category > 0:
        raise ValueError(
            "Some historical articles could not be "
            "mapped to a category."
        )

    category_counts = (
        history_with_category
        .groupby(
            ["user_id", "category"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "click_count"
            }
        )
    )

    total_clicks = (
        category_counts
        .groupby("user_id")["click_count"]
        .sum()
        .rename("total_clicks")
        .reset_index()
    )

    preferences = category_counts.merge(
        total_clicks,
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    preferences["preference"] = (
        preferences["click_count"]
        / preferences["total_clicks"]
    )

    preferences = preferences[
        [
            "user_id",
            "category",
            "click_count",
            "total_clicks",
            "preference",
        ]
    ]

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FEATURE_MIND
        / "user_category_preferences.parquet"
    )

    preferences.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Users: "
        f"{preferences['user_id'].nunique():,}"
    )

    print(
        f"       User-category pairs: "
        f"{len(preferences):,}"
    )

    print(
        f"       Saved: {output}"
    )

    return preferences


# ============================================================
# IMPRESSION-TIME USER × CATEGORY PREFERENCE
# ============================================================

def build_impression_category_preferences(
    split: str,
) -> pd.DataFrame:
    """
    Build candidate-level category preference features
    using the history available at each impression.

    Parameters
    ----------
    split:
        One of:
            train
            validation
            test

    Returns
    -------
    pd.DataFrame
        Candidate-level behavioural features containing:

            impression_id
            user_id
            timestamp
            article_id
            category
            user_category_clicks
            user_total_clicks
            category_preference

    Notes
    -----
    The user's history is taken from the history snapshot
    associated with each impression.

    Therefore, the feature represents the information
    available to the recommender at that impression.
    """

    print(
        f"\nBuilding impression-time category "
        f"preferences for {split.upper()}..."
    )

    # --------------------------------------------------------
    # Load candidate interactions
    # --------------------------------------------------------

    interactions_path = (
        PROCESSED_MIND
        / "splits"
        / f"{split}.parquet"
    )

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{interactions_path}"
        )

    interactions = pd.read_parquet(
        interactions_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
        ],
    )

    print(
        f"       Candidate rows: "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # Load history snapshots
    # --------------------------------------------------------

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "user_id",
            "article_id",
        ],
    )

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    # --------------------------------------------------------
    # Load article categories
    # --------------------------------------------------------

    articles = pd.read_parquet(
        TRAIN_ARTICLES_PATH,
        columns=[
            "article_id",
            "category",
        ],
    )

    # --------------------------------------------------------
    # Map historical articles to categories
    # --------------------------------------------------------

    history = history.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_history_categories = (
        history["category"]
        .isna()
        .sum()
    )

    if missing_history_categories > 0:
        raise ValueError(
            "Some history articles could not be mapped "
            "to categories."
        )

    # --------------------------------------------------------
    # Count category clicks inside each impression's
    # history snapshot
    # --------------------------------------------------------

    history_category_counts = (
        history
        .groupby(
            [
                "impression_id",
                "user_id",
                "category",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "user_category_clicks"
            }
        )
    )

    # --------------------------------------------------------
    # Count total historical clicks for each impression
    # --------------------------------------------------------

    history_totals = (
        history_category_counts
        .groupby(
            [
                "impression_id",
                "user_id",
            ],
            as_index=False,
        )["user_category_clicks"]
        .sum()
        .rename(
            columns={
                "user_category_clicks":
                    "user_total_clicks"
            }
        )
    )

    # --------------------------------------------------------
    # Load candidate article categories
    # --------------------------------------------------------

    candidates = interactions.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_candidate_categories = (
        candidates["category"]
        .isna()
        .sum()
    )

    if missing_candidate_categories > 0:
        raise ValueError(
            "Some candidate articles could not be "
            "mapped to categories."
        )

    # --------------------------------------------------------
    # Attach the user's category history to each candidate
    # --------------------------------------------------------

    features = candidates.merge(
        history_category_counts,
        on=[
            "impression_id",
            "user_id",
            "category",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Attach total history size
    # --------------------------------------------------------

    features = features.merge(
        history_totals,
        on=[
            "impression_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Cold-start handling
    # --------------------------------------------------------

    features["user_category_clicks"] = (
        features["user_category_clicks"]
        .fillna(0)
        .astype("int64")
    )

    features["user_total_clicks"] = (
        features["user_total_clicks"]
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # Calculate category preference
    #
    # If there is no history, preference is 0.
    # The popularity signal will later provide a fallback.
    # --------------------------------------------------------

    features["category_preference"] = 0.0

    has_history = (
        features["user_total_clicks"] > 0
    )

    features.loc[
        has_history,
        "category_preference",
    ] = (
        features.loc[
            has_history,
            "user_category_clicks",
        ]
        / features.loc[
            has_history,
            "user_total_clicks",
        ]
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    features = features[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "category",
            "clicked",
            "user_category_clicks",
            "user_total_clicks",
            "category_preference",
        ]
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if features["category_preference"].isna().any():
        raise ValueError(
            "category_preference contains missing values."
        )

    if (
        features["category_preference"] < 0
    ).any():
        raise ValueError(
            "category_preference contains negative values."
        )

    if (
        features["category_preference"] > 1
    ).any():
        raise ValueError(
            "category_preference contains values above 1."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FEATURE_MIND
        / f"{split}_behavioral_category.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Output rows: "
        f"{len(features):,}"
    )

    print(
        f"       Impressions: "
        f"{features['impression_id'].nunique():,}"
    )

    print(
        f"       Users: "
        f"{features['user_id'].nunique():,}"
    )

    print(
        f"       Cold-start candidate rows: "
        f"{(~has_history).sum():,}"
    )

    print(
        f"       Saved: {output}"
    )

    return features

# ============================================================
# IMPRESSION-TIME USER × SUBCATEGORY PREFERENCE
# ============================================================

def build_impression_subcategory_preferences(
    split: str,
) -> pd.DataFrame:
    """
    Build candidate-level subcategory preference features
    using the history available at each impression.

    The user's history is taken from the history snapshot
    associated with each impression.

    Returns candidate-level features containing:

        impression_id
        user_id
        timestamp
        article_id
        category
        subcategory
        clicked
        user_subcategory_clicks
        user_total_clicks
        subcategory_preference
    """

    print(
        f"\nBuilding impression-time subcategory "
        f"preferences for {split.upper()}..."
    )

    # --------------------------------------------------------
    # Load candidate interactions
    # --------------------------------------------------------

    interactions_path = (
        PROCESSED_MIND
        / "splits"
        / f"{split}.parquet"
    )

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{interactions_path}"
        )

    interactions = pd.read_parquet(
        interactions_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
        ],
    )

    print(
        f"       Candidate rows: "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # Load impression-specific history
    # --------------------------------------------------------

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "user_id",
            "article_id",
        ],
    )

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    # --------------------------------------------------------
    # Load article metadata
    # --------------------------------------------------------

    articles = pd.read_parquet(
        TRAIN_ARTICLES_PATH,
        columns=[
            "article_id",
            "category",
            "subcategory",
        ],
    )

    # --------------------------------------------------------
    # Map historical articles to subcategories
    # --------------------------------------------------------

    history = history.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_history_subcategories = (
        history["subcategory"]
        .isna()
        .sum()
    )

    if missing_history_subcategories > 0:
        raise ValueError(
            "Some history articles could not be mapped "
            "to subcategories."
        )

    # --------------------------------------------------------
    # Count subcategory clicks within each impression's
    # history snapshot
    # --------------------------------------------------------

    history_subcategory_counts = (
        history
        .groupby(
            [
                "impression_id",
                "user_id",
                "subcategory",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "user_subcategory_clicks"
            }
        )
    )

    # --------------------------------------------------------
    # Total historical clicks for each impression
    # --------------------------------------------------------

    history_totals = (
        history_subcategory_counts
        .groupby(
            [
                "impression_id",
                "user_id",
            ],
            as_index=False,
        )["user_subcategory_clicks"]
        .sum()
        .rename(
            columns={
                "user_subcategory_clicks":
                    "user_total_clicks"
            }
        )
    )

    # --------------------------------------------------------
    # Map candidate articles to category/subcategory
    # --------------------------------------------------------

    candidates = interactions.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    missing_candidate_subcategories = (
        candidates["subcategory"]
        .isna()
        .sum()
    )

    if missing_candidate_subcategories > 0:
        raise ValueError(
            "Some candidate articles could not be mapped "
            "to subcategories."
        )

    # --------------------------------------------------------
    # Attach historical subcategory preference
    # --------------------------------------------------------

    features = candidates.merge(
        history_subcategory_counts,
        on=[
            "impression_id",
            "user_id",
            "subcategory",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Attach total history size
    # --------------------------------------------------------

    features = features.merge(
        history_totals,
        on=[
            "impression_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Cold-start handling
    # --------------------------------------------------------

    features["user_subcategory_clicks"] = (
        features["user_subcategory_clicks"]
        .fillna(0)
        .astype("int64")
    )

    features["user_total_clicks"] = (
        features["user_total_clicks"]
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # Calculate subcategory preference
    # --------------------------------------------------------

    features["subcategory_preference"] = 0.0

    has_history = (
        features["user_total_clicks"] > 0
    )

    features.loc[
        has_history,
        "subcategory_preference",
    ] = (
        features.loc[
            has_history,
            "user_subcategory_clicks",
        ]
        / features.loc[
            has_history,
            "user_total_clicks",
        ]
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    features = features[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "category",
            "subcategory",
            "clicked",
            "user_subcategory_clicks",
            "user_total_clicks",
            "subcategory_preference",
        ]
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if features["subcategory_preference"].isna().any():
        raise ValueError(
            "subcategory_preference contains missing values."
        )

    if (
        features["subcategory_preference"] < 0
    ).any():
        raise ValueError(
            "subcategory_preference contains negative values."
        )

    if (
        features["subcategory_preference"] > 1
    ).any():
        raise ValueError(
            "subcategory_preference contains values above 1."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FEATURE_MIND
        / f"{split}_behavioral_subcategory.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Output rows: "
        f"{len(features):,}"
    )

    print(
        f"       Impressions: "
        f"{features['impression_id'].nunique():,}"
    )

    print(
        f"       Users: "
        f"{features['user_id'].nunique():,}"
    )

    print(
        f"       Cold-start candidate rows: "
        f"{(~has_history).sum():,}"
    )

    print(
        f"       Saved: {output}"
    )

    return features

# ============================================================
# IMPRESSION-TIME RECENCY FEATURES
# ============================================================

def build_impression_recency_features(
    split: str,
    decay_rate: float = 0.1,
) -> pd.DataFrame:
    """
    Build impression-time recency-weighted behavioural
    features using ordered MIND history.

    Since MIND does not provide individual timestamps for
    historical clicks, history_position is used as a
    relative recency proxy.

    Parameters
    ----------
    split:
        One of:
            train
            validation
            test

    decay_rate:
        Exponential decay strength.

    Returns
    -------
    pd.DataFrame
        Candidate-level recency features.
    """

    if decay_rate <= 0:
        raise ValueError(
            "decay_rate must be greater than zero."
        )

    print(
        f"\nBuilding impression-time recency "
        f"features for {split.upper()}..."
    )

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    interactions_path = (
        PROCESSED_MIND
        / "splits"
        / f"{split}.parquet"
    )

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{interactions_path}"
        )

    interactions = pd.read_parquet(
        interactions_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
        ],
    )

    print(
        f"       Candidate rows: "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # Load history
    # --------------------------------------------------------

    history = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "impression_id",
            "user_id",
            "article_id",
            "history_position",
        ],
    )

    print(
        f"       History rows: "
        f"{len(history):,}"
    )

    # --------------------------------------------------------
    # Load article metadata
    # --------------------------------------------------------

    articles = pd.read_parquet(
        TRAIN_ARTICLES_PATH,
        columns=[
            "article_id",
            "category",
            "subcategory",
        ],
    )

    # --------------------------------------------------------
    # Join history with article metadata
    # --------------------------------------------------------

    history = history.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    if history["category"].isna().any():
        raise ValueError(
            "Historical articles contain missing categories."
        )

    if history["subcategory"].isna().any():
        raise ValueError(
            "Historical articles contain missing "
            "subcategories."
        )

    # --------------------------------------------------------
    # Find newest history position for each impression
    # --------------------------------------------------------

    latest_position = (
        history
        .groupby("impression_id")["history_position"]
        .transform("max")
    )

    # --------------------------------------------------------
    # Calculate distance from the newest history item
    #
    # Example:
    #
    # positions: 0 1 2 3
    # distances: 3 2 1 0
    #
    # Therefore position 3 receives the largest weight.
    # --------------------------------------------------------

    history["position_distance"] = (
        latest_position
        - history["history_position"]
    )

    # --------------------------------------------------------
    # Exponential decay
    # --------------------------------------------------------

    history["recency_weight"] = np.exp(
        -decay_rate
        * history["position_distance"]
    )

    # --------------------------------------------------------
    # Total recency weight per impression
    # --------------------------------------------------------

    total_weight = (
        history
        .groupby(
            ["impression_id", "user_id"],
            as_index=False,
        )["recency_weight"]
        .sum()
        .rename(
            columns={
                "recency_weight":
                    "total_recency_weight"
            }
        )
    )

    # --------------------------------------------------------
    # Category recency weight
    # --------------------------------------------------------

    category_weight = (
        history
        .groupby(
            [
                "impression_id",
                "user_id",
                "category",
            ],
            as_index=False,
        )["recency_weight"]
        .sum()
        .rename(
            columns={
                "recency_weight":
                    "category_recency_weight"
            }
        )
    )

    # --------------------------------------------------------
    # Subcategory recency weight
    # --------------------------------------------------------

    subcategory_weight = (
        history
        .groupby(
            [
                "impression_id",
                "user_id",
                "subcategory",
            ],
            as_index=False,
        )["recency_weight"]
        .sum()
        .rename(
            columns={
                "recency_weight":
                    "subcategory_recency_weight"
            }
        )
    )

    # --------------------------------------------------------
    # Load candidate metadata
    # --------------------------------------------------------

    candidates = interactions.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    if candidates["category"].isna().any():
        raise ValueError(
            "Candidate articles contain missing categories."
        )

    if candidates["subcategory"].isna().any():
        raise ValueError(
            "Candidate articles contain missing "
            "subcategories."
        )

    # --------------------------------------------------------
    # Attach total weight
    # --------------------------------------------------------

    features = candidates.merge(
        total_weight,
        on=[
            "impression_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Attach category recency
    # --------------------------------------------------------

    features = features.merge(
        category_weight,
        on=[
            "impression_id",
            "user_id",
            "category",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Attach subcategory recency
    # --------------------------------------------------------

    features = features.merge(
        subcategory_weight,
        on=[
            "impression_id",
            "user_id",
            "subcategory",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Cold-start handling
    # --------------------------------------------------------

    features["total_recency_weight"] = (
        features["total_recency_weight"]
        .fillna(0.0)
    )

    features["category_recency_weight"] = (
        features["category_recency_weight"]
        .fillna(0.0)
    )

    features["subcategory_recency_weight"] = (
        features["subcategory_recency_weight"]
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    features["category_recency"] = 0.0

    features["subcategory_recency"] = 0.0

    has_history = (
        features["total_recency_weight"] > 0
    )

    features.loc[
        has_history,
        "category_recency",
    ] = (
        features.loc[
            has_history,
            "category_recency_weight",
        ]
        / features.loc[
            has_history,
            "total_recency_weight",
        ]
    )

    features.loc[
        has_history,
        "subcategory_recency",
    ] = (
        features.loc[
            has_history,
            "subcategory_recency_weight",
        ]
        / features.loc[
            has_history,
            "total_recency_weight",
        ]
    )

    # --------------------------------------------------------
    # Final columns
    # --------------------------------------------------------

    features = features[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "category",
            "subcategory",
            "clicked",
            "total_recency_weight",
            "category_recency",
            "subcategory_recency",
        ]
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if features["category_recency"].isna().any():
        raise ValueError(
            "category_recency contains missing values."
        )

    if features["subcategory_recency"].isna().any():
        raise ValueError(
            "subcategory_recency contains missing values."
        )

    if (
        features["category_recency"] < 0
    ).any():
        raise ValueError(
            "category_recency contains negative values."
        )

    if (
        features["category_recency"] > 1
    ).any():
        raise ValueError(
            "category_recency contains values above 1."
        )

    if (
        features["subcategory_recency"] < 0
    ).any():
        raise ValueError(
            "subcategory_recency contains negative values."
        )

    if (
        features["subcategory_recency"] > 1
    ).any():
        raise ValueError(
            "subcategory_recency contains values above 1."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FEATURE_MIND
        / f"{split}_behavioral_recency.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Output rows: "
        f"{len(features):,}"
    )

    print(
        f"       Impressions: "
        f"{features['impression_id'].nunique():,}"
    )

    print(
        f"       Users: "
        f"{features['user_id'].nunique():,}"
    )

    print(
        f"       Cold-start candidate rows: "
        f"{(~has_history).sum():,}"
    )

    print(
        f"       Decay rate: "
        f"{decay_rate}"
    )

    print(
        f"       Saved: {output}"
    )

    return features

# ============================================================
# ARTICLE POPULARITY FEATURES
# ============================================================

def build_article_popularity_features(
    split: str,
    smoothing_strength: float = 20.0,
) -> pd.DataFrame:
    """
    Build candidate-level article popularity features.

    Popularity statistics are derived from TRAIN only.

    Raw CTR:
        click_count / impression_count

    Smoothed CTR:
        (click_count + m * global_ctr)
        / (impression_count + m)

    Articles with no training statistics fall back to the
    global training CTR.

    Parameters
    ----------
    split:
        One of:
            train
            validation
            test

    smoothing_strength:
        Strength of the global CTR prior.

    Returns
    -------
    pd.DataFrame
        Candidate-level popularity features.
    """

    if smoothing_strength <= 0:
        raise ValueError(
            "smoothing_strength must be greater than zero."
        )

    print(
        f"\nBuilding article popularity "
        f"features for {split.upper()}..."
    )

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    interactions_path = (
        PROCESSED_MIND
        / "splits"
        / f"{split}.parquet"
    )

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{interactions_path}"
        )

    interactions = pd.read_parquet(
        interactions_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
        ],
    )

    print(
        f"       Candidate rows: "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # Load TRAIN-only article statistics
    # --------------------------------------------------------

    article_stats = pd.read_parquet(
        FEATURE_MIND
        / "article_stats.parquet"
    )

    required_columns = {
        "article_id",
        "impression_count",
        "click_count",
        "ctr",
    }

    missing_columns = (
        required_columns
        - set(article_stats.columns)
    )

    if missing_columns:
        raise ValueError(
            "article_stats.parquet is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Calculate global TRAIN CTR
    # --------------------------------------------------------

    total_impressions = (
        article_stats["impression_count"]
        .sum()
    )

    total_clicks = (
        article_stats["click_count"]
        .sum()
    )

    if total_impressions <= 0:
        raise ValueError(
            "Training impression count must be positive."
        )

    global_ctr = (
        total_clicks
        / total_impressions
    )

    print(
        f"       Training global CTR: "
        f"{global_ctr:.8f}"
    )

    # --------------------------------------------------------
    # Calculate smoothed CTR
    # --------------------------------------------------------

    article_stats["smoothed_ctr"] = (
        article_stats["click_count"]
        + smoothing_strength * global_ctr
    ) / (
        article_stats["impression_count"]
        + smoothing_strength
    )

    # --------------------------------------------------------
    # Keep only required popularity columns
    # --------------------------------------------------------

    article_stats = article_stats[
        [
            "article_id",
            "impression_count",
            "click_count",
            "ctr",
            "smoothed_ctr",
        ]
    ]

    # --------------------------------------------------------
    # Attach popularity to candidates
    # --------------------------------------------------------

    features = interactions.merge(
        article_stats,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Handle articles unseen during TRAIN
    # --------------------------------------------------------

    unseen = (
        features["impression_count"]
        .isna()
    )

    unseen_count = unseen.sum()

    features.loc[
        unseen,
        "impression_count",
    ] = 0

    features.loc[
        unseen,
        "click_count",
    ] = 0

    features.loc[
        unseen,
        "ctr",
    ] = 0.0

    features.loc[
        unseen,
        "smoothed_ctr",
    ] = global_ctr

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    features["impression_count"] = (
        features["impression_count"]
        .astype("int64")
    )

    features["click_count"] = (
        features["click_count"]
        .astype("int64")
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if features["ctr"].isna().any():
        raise ValueError(
            "CTR contains missing values."
        )

    if features["smoothed_ctr"].isna().any():
        raise ValueError(
            "Smoothed CTR contains missing values."
        )

    if (
        features["ctr"] < 0
    ).any() or (
        features["ctr"] > 1
    ).any():
        raise ValueError(
            "CTR must be between 0 and 1."
        )

    if (
        features["smoothed_ctr"] < 0
    ).any() or (
        features["smoothed_ctr"] > 1
    ).any():
        raise ValueError(
            "Smoothed CTR must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Final columns
    # --------------------------------------------------------

    features = features[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
            "impression_count",
            "click_count",
            "ctr",
            "smoothed_ctr",
        ]
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    FEATURE_MIND.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FEATURE_MIND
        / f"{split}_behavioral_popularity.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Output rows: "
        f"{len(features):,}"
    )

    print(
        f"       Articles with training statistics: "
        f"{features['impression_count'].gt(0).sum():,}"
    )

    print(
        f"       Unseen during training: "
        f"{unseen_count:,}"
    )

    print(
        f"       Smoothing strength: "
        f"{smoothing_strength}"
    )

    print(
        f"       Saved: {output}"
    )

    return features

# ============================================================
# BEHAVIOURAL CANDIDATE SCORE
# ============================================================

def build_behavioral_score(
    split: str,
    category_weight: float = 0.4,
    subcategory_weight: float = 0.6,
    interest_weight: float = 0.45,
    recency_weight: float = 0.35,
    popularity_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Combine behavioural signals into one candidate-level score.

    Baseline structure:

        interest_score =
            0.4 * category_preference
            + 0.6 * subcategory_preference

        recency_score =
            0.4 * category_recency
            + 0.6 * subcategory_recency

        behavioral_score =
            0.45 * interest_score
            + 0.35 * recency_score
            + 0.20 * smoothed_ctr

    The weights are baseline experimental choices and are
    not treated as optimized parameters.
    """

    print(
        f"\nBuilding behavioural score for "
        f"{split.upper()}..."
    )

    # --------------------------------------------------------
    # Validate weights
    # --------------------------------------------------------

    if category_weight < 0:
        raise ValueError(
            "category_weight must be non-negative."
        )

    if subcategory_weight < 0:
        raise ValueError(
            "subcategory_weight must be non-negative."
        )

    if abs(
        category_weight
        + subcategory_weight
        - 1.0
    ) > 1e-12:
        raise ValueError(
            "category_weight + subcategory_weight "
            "must equal 1."
        )

    if interest_weight < 0:
        raise ValueError(
            "interest_weight must be non-negative."
        )

    if recency_weight < 0:
        raise ValueError(
            "recency_weight must be non-negative."
        )

    if popularity_weight < 0:
        raise ValueError(
            "popularity_weight must be non-negative."
        )

    if abs(
        interest_weight
        + recency_weight
        + popularity_weight
        - 1.0
    ) > 1e-12:
        raise ValueError(
            "interest_weight + recency_weight + "
            "popularity_weight must equal 1."
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    category_path = (
        FEATURE_MIND
        / f"{split}_behavioral_category.parquet"
    )

    subcategory_path = (
        FEATURE_MIND
        / f"{split}_behavioral_subcategory.parquet"
    )

    recency_path = (
        FEATURE_MIND
        / f"{split}_behavioral_recency.parquet"
    )

    popularity_path = (
        FEATURE_MIND
        / f"{split}_behavioral_popularity.parquet"
    )

    required_paths = [
        category_path,
        subcategory_path,
        recency_path,
        popularity_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required behavioural feature file "
                f"not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load feature columns
    # --------------------------------------------------------

    category = pd.read_parquet(
        category_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
            "category_preference",
        ],
    )

    subcategory = pd.read_parquet(
        subcategory_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "subcategory_preference",
        ],
    )

    recency = pd.read_parquet(
        recency_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "category_recency",
            "subcategory_recency",
        ],
    )

    popularity = pd.read_parquet(
        popularity_path,
        columns=[
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "smoothed_ctr",
        ],
    )

    print(
        f"       Candidate rows: "
        f"{len(category):,}"
    )

    # --------------------------------------------------------
    # Verify candidate alignment
    # --------------------------------------------------------

    key_columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "article_id",
    ]

    base_keys = category[key_columns]

    for name, frame in [
        ("subcategory", subcategory),
        ("recency", recency),
        ("popularity", popularity),
    ]:
        if not base_keys.equals(
            frame[key_columns]
        ):
            raise ValueError(
                f"{name} feature rows are not aligned "
                "with category features."
            )

    # --------------------------------------------------------
    # Build score table
    # --------------------------------------------------------

    features = category[
        [
            "impression_id",
            "user_id",
            "timestamp",
            "article_id",
            "clicked",
            "category_preference",
        ]
    ].copy()

    features["subcategory_preference"] = (
        subcategory["subcategory_preference"]
        .to_numpy()
    )

    features["category_recency"] = (
        recency["category_recency"]
        .to_numpy()
    )

    features["subcategory_recency"] = (
        recency["subcategory_recency"]
        .to_numpy()
    )

    features["smoothed_ctr"] = (
        popularity["smoothed_ctr"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Interest score
    # --------------------------------------------------------

    features["interest_score"] = (
        category_weight
        * features["category_preference"]
        +
        subcategory_weight
        * features["subcategory_preference"]
    )

    # --------------------------------------------------------
    # Recency score
    # --------------------------------------------------------

    features["recency_score"] = (
        category_weight
        * features["category_recency"]
        +
        subcategory_weight
        * features["subcategory_recency"]
    )

    # --------------------------------------------------------
    # Final behavioural score
    # --------------------------------------------------------

    features["behavioral_score"] = (
        interest_weight
        * features["interest_score"]
        +
        recency_weight
        * features["recency_score"]
        +
        popularity_weight
        * features["smoothed_ctr"]
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    score_columns = [
        "category_preference",
        "subcategory_preference",
        "category_recency",
        "subcategory_recency",
        "smoothed_ctr",
        "interest_score",
        "recency_score",
        "behavioral_score",
    ]

    for column in score_columns:
        if features[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

    if (
        features["behavioral_score"] < 0
    ).any():
        raise ValueError(
            "behavioral_score contains negative values."
        )

    if (
        features["behavioral_score"] > 1
    ).any():
        raise ValueError(
            "behavioral_score contains values above 1."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        FEATURE_MIND
        / f"{split}_behavioral_score.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    print(
        f"       Output rows: "
        f"{len(features):,}"
    )

    print(
        f"       Behavioural score range: "
        f"{features['behavioral_score'].min():.6f}"
        f" to "
        f"{features['behavioral_score'].max():.6f}"
    )

    print(
        f"       Saved: {output}"
    )

    return features

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MIND BEHAVIOURAL FEATURES")
    print("=" * 70)

    # Existing descriptive feature.
    build_user_category_preferences()

    # Impression-time behavioural features.
    build_impression_category_preferences(
        "train"
    )

    build_impression_category_preferences(
        "validation"
    )

    build_impression_category_preferences(
        "test"
    )

    build_impression_subcategory_preferences(
    "train"
    )

    build_impression_subcategory_preferences(
    "validation"
    )

    build_impression_subcategory_preferences(
    "test"
    )

# --------------------------------------------------------
# Impression-time recency
# --------------------------------------------------------

    build_impression_recency_features(
        "train"
    )

    build_impression_recency_features(
        "validation"
    )

    build_impression_recency_features(
        "test"
    )



# --------------------------------------------------------
# Article popularity
# --------------------------------------------------------

    build_article_popularity_features(
        "train"
    )

    build_article_popularity_features(
        "validation"
    )

    build_article_popularity_features(
        "test"
    )

# --------------------------------------------------------
# Final behavioural score
# --------------------------------------------------------

    build_behavioral_score(
        "train"
    )

    build_behavioral_score(
        "validation"
    )

    build_behavioral_score(
        "test"
    )

    print("\n" + "=" * 70)
    print("BEHAVIOURAL FEATURE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()