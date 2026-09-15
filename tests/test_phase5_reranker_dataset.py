from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.ranking.phase5_reranker_dataset import (
    EBNeRDPhase5Dataset,
    RETRIEVAL_FEATURE_COLUMNS,
    load_phase5_dataset,
)


def make_candidates():
    return pd.DataFrame(
        {
            "impression_id": [1, 1, 2],
            "user_id": [100, 100, 100],
            "article_id": [10, 20, 30],
            "impression_time": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "clicked": [0, 1, 0],
            "bm25_score": [50.0, 40.0, 30.0],
            "bm25_rank": [1, 2, 3],
            "semantic_score": [0.90, 0.80, 0.70],
            "semantic_rank": [1, 2, 3],
            "rrf_score": [0.0328, 0.0323, 0.0317],
        }
    )


def make_history():
    return pd.DataFrame(
        {
            "user_id": [100, 100, 100, 100, 100],
            "article_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 09:00:00",
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:30:00",
                    "2024-01-01 11:00:00",
                ]
            ),
        }
    )


@pytest.fixture
def article_mapping():
    return {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        10: 10,
        20: 20,
        30: 30,
    }


@pytest.fixture
def dataset(article_mapping):
    return EBNeRDPhase5Dataset(
        candidates=make_candidates(),
        history=make_history(),
        article_id_to_index=article_mapping,
        max_history_length=20,
    )


def test_dataset_length(dataset):
    assert len(dataset) == 3


def test_dataset_item_keys(dataset):
    example = dataset[0]

    assert set(example.keys()) == {
        "history_article_ids",
        "candidate_article_id",
        "retrieval_features",
        "clicked",
        "user_id",
    }


def test_history_shape_and_dtype(dataset):
    example = dataset[0]

    assert example["history_article_ids"].shape == (20,)
    assert example["history_article_ids"].dtype == torch.long


def test_candidate_shape_and_dtype(dataset):
    example = dataset[0]

    assert example["candidate_article_id"].shape == torch.Size([])
    assert example["candidate_article_id"].dtype == torch.long


def test_retrieval_features_shape_and_dtype(dataset):
    example = dataset[0]

    assert example["retrieval_features"].shape == (5,)
    assert example["retrieval_features"].dtype == torch.float32


def test_retrieval_feature_order(dataset):
    example = dataset[0]

    expected = torch.tensor(
        [
            50.0,
            1.0,
            0.90,
            1.0,
            0.0328,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        example["retrieval_features"],
        expected,
        atol=1e-6,
    )


def test_click_and_user_id(dataset):
    example = dataset[1]

    assert example["clicked"].dtype == torch.float32
    assert example["clicked"].item() == 1.0

    assert example["user_id"].dtype == torch.long
    assert example["user_id"].item() == 100


def test_strict_temporal_filtering(dataset):
    """
    Candidate impression is at 10:00.

    History:
        08:00 -> article 1   valid
        09:00 -> article 2   valid
        10:00 -> article 3   INVALID: equal timestamp
        10:30 -> article 4   INVALID: future
        11:00 -> article 5   INVALID: future

    Therefore only articles 1 and 2 may appear.
    """

    example = dataset[0]

    history = example["history_article_ids"].tolist()

    non_padding = [
        article_id
        for article_id in history
        if article_id != 0
    ]

    assert non_padding == [1, 2]


def test_future_history_is_excluded(dataset):
    """
    Candidate impression 2 occurs at 12:00.

    Articles 1-5 have all occurred before 12:00,
    so all five should be available.
    """

    example = dataset[2]

    history = example["history_article_ids"].tolist()

    non_padding = [
        article_id
        for article_id in history
        if article_id != 0
    ]

    assert non_padding == [1, 2, 3, 4, 5]


def test_history_is_right_aligned(dataset):
    example = dataset[0]

    history = example["history_article_ids"].tolist()

    assert history[:18] == [0] * 18
    assert history[-2:] == [1, 2]


def test_max_history_length_is_respected(
    article_mapping,
):
    candidates = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [100],
            "article_id": [10],
            "impression_time": pd.to_datetime(
                ["2024-01-01 12:00:00"]
            ),
            "clicked": [0],
            "bm25_score": [50.0],
            "bm25_rank": [1],
            "semantic_score": [0.9],
            "semantic_rank": [1],
            "rrf_score": [0.0328],
        }
    )

    history = pd.DataFrame(
        {
            "user_id": [100] * 5,
            "article_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 09:00:00",
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 11:30:00",
                ]
            ),
        }
    )

    dataset = EBNeRDPhase5Dataset(
        candidates=candidates,
        history=history,
        article_id_to_index=article_mapping,
        max_history_length=3,
    )

    example = dataset[0]

    assert example["history_article_ids"].tolist() == [
        3,
        4,
        5,
    ]


def test_no_history_is_zero_padded(
    article_mapping,
):
    candidates = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [999],
            "article_id": [10],
            "impression_time": pd.to_datetime(
                ["2024-01-01 12:00:00"]
            ),
            "clicked": [0],
            "bm25_score": [50.0],
            "bm25_rank": [1],
            "semantic_score": [0.9],
            "semantic_rank": [1],
            "rrf_score": [0.0328],
        }
    )

    history = make_history()

    dataset = EBNeRDPhase5Dataset(
        candidates=candidates,
        history=history,
        article_id_to_index=article_mapping,
        max_history_length=20,
    )

    example = dataset[0]

    assert torch.equal(
        example["history_article_ids"],
        torch.zeros(20, dtype=torch.long),
    )


def test_unknown_article_maps_to_zero():
    candidates = make_candidates()

    history = make_history()

    mapping = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    }

    dataset = EBNeRDPhase5Dataset(
        candidates=candidates,
        history=history,
        article_id_to_index=mapping,
        max_history_length=20,
    )

    example = dataset[0]

    assert example["candidate_article_id"].item() == 0


def test_retrieval_features_are_finite(dataset):
    values = dataset.retrieval_features

    assert np.isfinite(values).all()


def test_missing_candidate_column_is_rejected(
    article_mapping,
):
    candidates = make_candidates().drop(
        columns=["bm25_score"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        EBNeRDPhase5Dataset(
            candidates=candidates,
            history=make_history(),
            article_id_to_index=article_mapping,
        )


def test_missing_history_column_is_rejected(
    article_mapping,
):
    history = make_history().drop(
        columns=["timestamp"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        EBNeRDPhase5Dataset(
            candidates=make_candidates(),
            history=history,
            article_id_to_index=article_mapping,
        )


def test_invalid_history_length_is_rejected(
    article_mapping,
):
    with pytest.raises(
        ValueError,
        match="max_history_length",
    ):
        EBNeRDPhase5Dataset(
            candidates=make_candidates(),
            history=make_history(),
            article_id_to_index=article_mapping,
            max_history_length=0,
        )


def test_empty_article_mapping_is_rejected():
    with pytest.raises(
        ValueError,
        match="article_id_to_index",
    ):
        EBNeRDPhase5Dataset(
            candidates=make_candidates(),
            history=make_history(),
            article_id_to_index={},
        )


def test_repeated_dataset_access_is_deterministic(
    dataset,
):
    first = dataset[0]
    second = dataset[0]

    assert torch.equal(
        first["history_article_ids"],
        second["history_article_ids"],
    )

    assert torch.equal(
        first["candidate_article_id"],
        second["candidate_article_id"],
    )

    assert torch.equal(
        first["retrieval_features"],
        second["retrieval_features"],
    )

    assert torch.equal(
        first["clicked"],
        second["clicked"],
    )


def test_retrieval_feature_constant_is_correct():
    assert RETRIEVAL_FEATURE_COLUMNS == [
        "bm25_score",
        "bm25_rank",
        "semantic_score",
        "semantic_rank",
        "rrf_score",
    ]


def test_real_ebnerd_dataset_loading():
    """
    Integration test against the project's actual EB-NeRD files.

    This test is intentionally lightweight: it loads the dataset
    metadata and checks the first example rather than iterating
    through all 2.5M rows.
    """

    project_root = Path(__file__).resolve().parents[1]

    dataset, article_mapping = load_phase5_dataset(
        project_root=project_root,
        split="train",
        max_history_length=20,
    )

    assert len(dataset) == 2_585_747
    assert len(article_mapping) == 20_738

    example = dataset[0]

    assert example["history_article_ids"].shape == (20,)
    assert example["candidate_article_id"].shape == torch.Size([])
    assert example["retrieval_features"].shape == (5,)
    assert example["clicked"].item() in {0.0, 1.0}