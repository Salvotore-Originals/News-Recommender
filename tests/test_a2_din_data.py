"""
Phase 3.4 — Tests for the EB-NeRD → DIN data adapter.
"""

from pathlib import Path

import pandas as pd
import torch

from src.ranking.a2_din_data import (
    EBNeRDDINDataset,
    build_article_id_mapping,
    load_ebnerd_din_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ebnerd"
    / "small"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ebnerd"
    / "small"
)


def test_ebnerd_files_exist():
    """Required EB-NeRD Small artifacts must exist."""

    required_files = [
        FEATURE_DIR / "articles.parquet",
        FEATURE_DIR / "train_candidates.parquet",
        PROCESSED_DIR / "train_user_history.parquet",
    ]

    for path in required_files:
        assert path.exists(), f"Missing required file: {path}"


def test_article_id_mapping_is_valid():
    """Article IDs should map to positive embedding indices."""

    articles = pd.read_parquet(
        FEATURE_DIR / "articles.parquet"
    )

    mapping = build_article_id_mapping(articles)

    assert len(mapping) == len(
        articles["article_id"].dropna().unique()
    )

    assert len(mapping) > 0

    # 0 is reserved for padding/unknown.
    assert min(mapping.values()) >= 1

    # Mapping indices should be unique.
    assert len(mapping.values()) == len(
        set(mapping.values())
    )


def test_real_ebnerd_dataset_sample():
    """
    Construct the DIN dataset from real EB-NeRD data and inspect
    a small number of examples.
    """

    articles = pd.read_parquet(
        FEATURE_DIR / "articles.parquet"
    )

    candidates = pd.read_parquet(
        FEATURE_DIR / "train_candidates.parquet"
    )

    history = pd.read_parquet(
        PROCESSED_DIR / "train_user_history.parquet"
    )

    mapping = build_article_id_mapping(articles)

    # Only use a tiny candidate sample for this test.
    candidates_sample = candidates.head(100).copy()

    dataset = EBNeRDDINDataset(
        candidates=candidates_sample,
        user_history=history,
        article_id_to_index=mapping,
        max_history_length=20,
    )

    assert len(dataset) == 100

    example = dataset[0]

    assert "history_article_ids" in example
    assert "candidate_article_id" in example
    assert "clicked" in example
    assert "user_id" in example

    history_tensor = example["history_article_ids"]
    candidate_tensor = example["candidate_article_id"]
    clicked_tensor = example["clicked"]

    # History must have fixed length.
    assert history_tensor.shape == (20,)

    # Candidate is a single article index.
    assert candidate_tensor.shape == ()

    # Click label is scalar.
    assert clicked_tensor.shape == ()

    assert history_tensor.dtype == torch.long
    assert candidate_tensor.dtype == torch.long
    assert clicked_tensor.dtype == torch.float32


def test_history_padding_and_mask_are_consistent():
    """
    Valid history positions should be represented by non-zero article
    indices and padding positions by zero.
    """

    articles = pd.read_parquet(
        FEATURE_DIR / "articles.parquet"
    )

    candidates = pd.read_parquet(
        FEATURE_DIR / "train_candidates.parquet"
    )

    history = pd.read_parquet(
        PROCESSED_DIR / "train_user_history.parquet"
    )

    mapping = build_article_id_mapping(articles)

    dataset = EBNeRDDINDataset(
        candidates=candidates.head(100),
        user_history=history,
        article_id_to_index=mapping,
        max_history_length=20,
    )

    example = dataset[0]

    history_tensor = example["history_article_ids"]

    # Padding is represented by 0.
    padding_positions = history_tensor == 0

    valid_positions = history_tensor != 0

    assert torch.all(
        history_tensor[padding_positions] == 0
    )

    # Every valid article index must be positive.
    assert torch.all(
        history_tensor[valid_positions] > 0
    )


def test_dataset_can_feed_din_model():
    """
    Verify that real EB-NeRD tensors can directly enter DINModel.
    """

    from src.ranking.a2_din import DINModel

    articles = pd.read_parquet(
        FEATURE_DIR / "articles.parquet"
    )

    candidates = pd.read_parquet(
        FEATURE_DIR / "train_candidates.parquet"
    )

    history = pd.read_parquet(
        PROCESSED_DIR / "train_user_history.parquet"
    )

    mapping = build_article_id_mapping(articles)

    dataset = EBNeRDDINDataset(
        candidates=candidates.head(8),
        user_history=history,
        article_id_to_index=mapping,
        max_history_length=20,
    )

    batch = [dataset[i] for i in range(8)]

    histories = torch.stack(
        [
            item["history_article_ids"]
            for item in batch
        ]
    )

    candidates_tensor = torch.stack(
        [
            item["candidate_article_id"]
            for item in batch
        ]
    )

    # +1 because index 0 is reserved for padding.
    model = DINModel(
        num_articles=len(mapping) + 1,
        embedding_dim=16,
        hidden_dim=16,
        dropout=0.0,
        padding_idx=0,
    )

    logits = model(
        history_article_ids=histories,
        candidate_article_ids=candidates_tensor,
    )

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()

def test_history_events_are_strictly_before_impression():
    """
    Verify the DIN adapter's temporal-leakage invariant.

    Every history event used for an impression must satisfy:

        history_timestamp < impression_time
    """
    import pandas as pd

    project_root = Path(__file__).resolve().parents[1]

    dataset, _ = load_ebnerd_din_dataset(
        project_root=project_root,
        split="validation",
        max_history_length=20,
    )

    # Check a deterministic sample of candidate rows.
    sample = dataset.candidates.head(5000)

    violations = []

    for _, row in sample.iterrows():
        user_id = int(row["user_id"])
        impression_time = pd.Timestamp(row["impression_time"])

        events = dataset.history_by_user.get(user_id, [])

        # Reconstruct exactly the events that _get_history() is
        # allowed to consider before truncation.
        eligible_events = [
            (timestamp, article_id)
            for timestamp, article_id in events
            if timestamp < impression_time
        ]

        # The adapter's _get_history() must never return an event
        # outside this temporally valid set.
        returned_history = dataset._get_history(
            user_id=user_id,
            impression_time=impression_time,
        )

        # Verify that the source events selected by the adapter are
        # all temporally valid.
        #
        # We inspect the last max_history_length eligible events,
        # which are exactly the events used by _get_history().
        expected_events = eligible_events[-dataset.max_history_length:]

        expected_article_ids = [
            article_id
            for _, article_id in expected_events
        ]

        if returned_history != expected_article_ids:
            violations.append(
                {
                    "user_id": user_id,
                    "impression_time": impression_time,
                    "returned_history": returned_history,
                    "expected_history": expected_article_ids,
                }
            )

        # Explicitly verify every selected event.
        for timestamp, _ in expected_events:
            if not timestamp < impression_time:
                violations.append(
                    {
                        "user_id": user_id,
                        "impression_time": impression_time,
                        "history_timestamp": timestamp,
                    }
                )

    assert not violations, (
        "Temporal leakage detected. "
        f"Found {len(violations)} violations. "
        f"First violation: {violations[0]}"
    )