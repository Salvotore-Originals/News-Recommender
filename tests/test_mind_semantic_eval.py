import numpy as np
import pandas as pd

from src.evaluation.mind_semantic import (
    build_history_lookup,
    evaluate_semantic_retrieval,
)


def test_build_history_lookup():

    history = pd.DataFrame(
        {
            "impression_id": [
                "1",
                "1",
                "2",
            ],
            "article_id": [
                "N1",
                "N2",
                "N3",
            ],
            "history_position": [
                0,
                1,
                0,
            ],
        }
    )

    lookup = build_history_lookup(
        history
    )

    assert lookup["1"] == [
        "N1",
        "N2",
    ]

    assert lookup["2"] == [
        "N3",
    ]


def test_semantic_candidate_evaluation():

    article_ids = np.array(
        [
            "N1",
            "N2",
            "N3",
        ]
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    history_lookup = {
        "I1": ["N1"],
    }

    candidates = pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I1",
                "I1",
            ],
            "user_id": [
                "U1",
                "U1",
                "U1",
            ],
            "timestamp": [
                "2019-11-12",
                "2019-11-12",
                "2019-11-12",
            ],
            "article_id": [
                "N2",
                "N3",
                "N1",
            ],
            "clicked": [
                1,
                0,
                0,
            ],
        }
    )

    metrics = evaluate_semantic_retrieval(
        candidates=candidates,
        history_lookup=history_lookup,
        article_embeddings=embeddings,
        article_ids=article_ids,
        max_history=10,
        ks=(1, 2),
    )

    assert metrics["impressions"] == 1

    assert metrics["Hit@1"] == 1.0

    assert metrics["Hit@2"] == 1.0

    assert metrics["MRR"] == 1.0