import numpy as np
import pytest

from src.ranking.normalization import min_max_normalize


def test_min_max_normalization():
    scores = [2.0, 5.0, 8.0]

    result = min_max_normalize(scores)

    expected = np.array(
        [0.0, 0.5, 1.0]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_normalization_preserves_order():
    scores = [10.0, 2.0, 7.0]

    result = min_max_normalize(scores)

    assert result[1] < result[2]
    assert result[2] < result[0]


def test_constant_scores_return_zero():
    scores = [5.0, 5.0, 5.0]

    result = min_max_normalize(scores)

    expected = np.zeros(3)

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_negative_scores_are_supported():
    scores = [-5.0, 0.0, 5.0]

    result = min_max_normalize(scores)

    expected = np.array(
        [0.0, 0.5, 1.0]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_empty_scores_raise_error():
    with pytest.raises(ValueError):
        min_max_normalize([])


def test_multidimensional_scores_raise_error():
    scores = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]

    with pytest.raises(ValueError):
        min_max_normalize(scores)


def test_non_finite_scores_raise_error():
    scores = [
        1.0,
        np.nan,
        3.0,
    ]

    with pytest.raises(ValueError):
        min_max_normalize(scores)