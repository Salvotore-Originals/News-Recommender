import numpy as np
import pandas as pd
import pytest

import src.evaluation.mind_inference as mind_inference


def test_predict_mind_candidates_is_label_free(
    monkeypatch,
):
    """
    The inference adapter must accept candidates without
    a clicked column and must remove clicked from output.
    """

    candidates = pd.DataFrame(
        {
            "impression_id": ["1", "1", "1"],
            "user_id": ["U1", "U1", "U1"],
            "timestamp": [
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
                "2019-11-12 10:00:00",
            ],
            "article_id": [
                "N1",
                "N2",
                "N3",
            ],
        }
    )

    history = pd.DataFrame(
        {
            "impression_id": ["1"],
            "article_id": ["N10"],
            "history_position": [0],
        }
    )

    articles = pd.DataFrame(
        {
            "article_id": [
                "N1",
                "N2",
                "N3",
                "N10",
            ],
            "title": [
                "Article one",
                "Article two",
                "Article three",
                "History article",
            ],
        }
    )

    behavioral_scores = pd.DataFrame(
        {
            "impression_id": ["1", "1", "1"],
            "article_id": [
                "N1",
                "N2",
                "N3",
            ],
            "behavioral_score": [
                0.3,
                0.2,
                0.1,
            ],
        }
    )

    def fake_candidate_scores(
        candidates,
        history,
        articles,
        bm25_retriever,
        article_embeddings,
        article_ids,
        behavioral_scores,
        max_history,
    ):
        # Verify that the adapter supplied the temporary label.
        assert "clicked" in candidates.columns
        assert (candidates["clicked"] == 0).all()

        return pd.DataFrame(
            {
                "impression_id": ["1", "1", "1"],
                "user_id": ["U1", "U1", "U1"],
                "timestamp": [
                    "2019-11-12 10:00:00",
                    "2019-11-12 10:00:00",
                    "2019-11-12 10:00:00",
                ],
                "article_id": [
                    "N1",
                    "N2",
                    "N3",
                ],
                "clicked": [0, 0, 0],
                "bm25_score": [
                    3.0,
                    2.0,
                    1.0,
                ],
                "semantic_score": [
                    0.9,
                    0.8,
                    0.7,
                ],
                "behavioral_score": [
                    0.3,
                    0.2,
                    0.1,
                ],
            }
        )

    monkeypatch.setattr(
        mind_inference,
        "build_hybrid_candidate_scores",
        fake_candidate_scores,
    )

    result = (
        mind_inference.predict_mind_candidates(
            candidates=candidates,
            history=history,
            articles=articles,
            bm25_retriever=object(),
            article_embeddings=np.zeros(
                (4, 384),
                dtype=np.float32,
            ),
            article_ids=np.array(
                ["N1", "N2", "N3", "N10"]
            ),
            behavioral_scores=behavioral_scores,
        )
    )

    assert "clicked" not in result.columns

    assert len(result) == 3

    assert result["rank"].tolist() == [
        1,
        2,
        3,
    ]

    assert result["article_id"].tolist() == [
        "N1",
        "N2",
        "N3",
    ]

    assert result["hybrid_score"].notna().all()


def test_predict_requires_candidate_identity_columns():

    candidates = pd.DataFrame(
        {
            "impression_id": ["1"],
            "article_id": ["N1"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Candidates are missing required columns",
    ):
        mind_inference.predict_mind_candidates(
            candidates=candidates,
            history=pd.DataFrame(),
            articles=pd.DataFrame(),
            bm25_retriever=object(),
            article_embeddings=np.zeros(
                (1, 384),
                dtype=np.float32,
            ),
            article_ids=np.array(["N1"]),
            behavioral_scores=pd.DataFrame(),
        )