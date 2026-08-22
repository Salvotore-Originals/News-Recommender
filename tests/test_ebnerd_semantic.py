import numpy as np
import pandas as pd

from src.retrieval.ebnerd_semantic import (
    load_articles,
    build_history_lookup,
    build_query_embedding,
    score_candidates,
)


def test_article_count():

    articles = load_articles()

    assert len(articles) == 20_738


def test_article_ids_are_unique():

    articles = load_articles()

    assert articles["article_id"].is_unique


def test_history_lookup_is_user_based():

    history = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "article_id": [10, 20, 30],
            "timestamp": [
                pd.Timestamp("2023-05-20"),
                pd.Timestamp("2023-05-21"),
                pd.Timestamp("2023-05-20"),
            ],
        }
    )

    lookup = build_history_lookup(
        history
    )

    assert set(lookup.keys()) == {
        1,
        2,
    }

    assert len(lookup[1]) == 2


def test_query_excludes_future_history():

    history = [
        (
            pd.Timestamp("2023-05-20"),
            1,
        ),
        (
            pd.Timestamp("2023-05-21"),
            2,
        ),
        (
            pd.Timestamp("2023-05-23"),
            3,
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_to_index = {
        1: 0,
        2: 1,
        3: 2,
    }

    query = build_query_embedding(
        history,
        pd.Timestamp("2023-05-22"),
        article_to_index,
        embeddings,
    )

    expected = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    expected /= np.linalg.norm(
        expected
    )

    np.testing.assert_allclose(
        query,
        expected,
        atol=1e-6,
    )


def test_query_returns_none_for_empty_history():

    query = build_query_embedding(
        [],
        pd.Timestamp("2023-05-22"),
        {},
        np.empty(
            (0, 2),
            dtype=np.float32,
        ),
    )

    assert query is None


def test_query_deduplicates_articles():

    history = [
        (
            pd.Timestamp("2023-05-20"),
            1,
        ),
        (
            pd.Timestamp("2023-05-21"),
            1,
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    query = build_query_embedding(
        history,
        pd.Timestamp("2023-05-22"),
        {1: 0},
        embeddings,
    )

    np.testing.assert_allclose(
        query,
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_candidate_scores_are_sorted():

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    embeddings = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    article_to_index = {
        10: 0,
        20: 1,
        30: 2,
    }

    results = score_candidates(
        query,
        [10, 20, 30],
        article_to_index,
        embeddings,
    )

    assert results[0][0] == 20
    assert results[1][0] == 30
    assert results[2][0] == 10


def test_candidate_set_is_preserved():

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_to_index = {
        10: 0,
        20: 1,
        30: 2,
    }

    results = score_candidates(
        query,
        [10, 30],
        article_to_index,
        embeddings,
    )

    result_ids = {
        article_id
        for article_id, score
        in results
    }

    assert result_ids == {
        10,
        30,
    }

def test_vectorized_scorer_matches_original():

    from src.retrieval.ebnerd_semantic import (
        score_candidates,
        score_candidates_vectorized,
    )

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.70710677, 0.70710677, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    article_to_index = {
        10: 0,
        20: 1,
        30: 2,
        40: 3,
    }

    candidates = [
        10,
        20,
        30,
        40,
    ]

    original = score_candidates(
        query,
        candidates,
        article_to_index,
        embeddings,
    )

    vectorized = score_candidates_vectorized(
        query,
        candidates,
        article_to_index,
        embeddings,
    )

    assert [
        article_id
        for article_id, score
        in original
    ] == [
        article_id
        for article_id, score
        in vectorized
    ]

    for (
        original_item,
        vectorized_item,
    ) in zip(
        original,
        vectorized,
    ):

        assert (
            original_item[0]
            == vectorized_item[0]
        )

        assert abs(
            original_item[1]
            - vectorized_item[1]
        ) < 1e-6

def test_vectorized_scorer_matches_original_on_real_candidates():

    from src.retrieval.ebnerd_semantic import (
        build_history_lookup,
        build_query_embedding,
        load_history,
        load_interactions,
        score_candidates,
        score_candidates_vectorized,
    )

    articles = load_articles()

    # Small deterministic synthetic embeddings are sufficient
    # to verify ranking behavior against real candidate sets.
    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(len(articles), 384)
    ).astype(
        np.float32
    )

    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    article_to_index = {
        article_id: index
        for index, article_id
        in enumerate(
            articles["article_id"]
        )
    }

    interactions = load_interactions(
        "train"
    )

    history = load_history(
        "train"
    )

    history_lookup = build_history_lookup(
        history
    )

    groups = interactions.groupby(
        "impression_id",
        sort=False,
    )

    checked = 0

    for impression_id, group in groups:

        if checked >= 100:
            break

        group = group.sort_values(
            "article_id"
        )

        user_id = group.iloc[0][
            "user_id"
        ]

        impression_time = group.iloc[0][
            "impression_time"
        ]

        candidates = (
            group["article_id"]
            .tolist()
        )

        user_history = history_lookup.get(
            user_id,
            [],
        )

        query = build_query_embedding(
            user_history,
            impression_time,
            article_to_index,
            embeddings,
        )

        original = score_candidates(
            query,
            candidates,
            article_to_index,
            embeddings,
        )

        vectorized = (
            score_candidates_vectorized(
                query,
                candidates,
                article_to_index,
                embeddings,
            )
        )

        assert [
            x[0]
            for x in original
        ] == [
            x[0]
            for x in vectorized
        ]

        for old, new in zip(
            original,
            vectorized,
        ):

            assert abs(
                old[1] - new[1]
            ) < 1e-6

        checked += 1

    assert checked == 100