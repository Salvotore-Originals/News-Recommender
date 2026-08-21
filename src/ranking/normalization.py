from __future__ import annotations

from typing import Sequence

import numpy as np


def min_max_normalize(
    scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Normalize scores to the [0, 1] range using Min-Max scaling.

    The minimum score becomes 0 and the maximum score becomes 1.

    Formula:

        x' = (x - min(x)) / (max(x) - min(x))

    If all scores are identical, the signal provides no
    ranking discrimination. In that case, all normalized
    scores are returned as 0.

    Parameters
    ----------
    scores:
        One-dimensional sequence of numeric scores.

    Returns
    -------
    np.ndarray
        Float64 array containing normalized scores.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, is empty,
        or contains non-finite values.
    """

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    # --------------------------------------------------
    # Validate dimensionality
    # --------------------------------------------------

    if values.ndim != 1:
        raise ValueError(
            "scores must be a one-dimensional sequence."
        )

    # --------------------------------------------------
    # Validate non-empty input
    # --------------------------------------------------

    if len(values) == 0:
        raise ValueError(
            "scores must not be empty."
        )

    # --------------------------------------------------
    # Validate finite values
    # --------------------------------------------------

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "scores must contain only finite values."
        )

    minimum = np.min(values)
    maximum = np.max(values)

    # --------------------------------------------------
    # Constant-score case
    # --------------------------------------------------

    if maximum == minimum:
        return np.zeros_like(
            values,
            dtype=np.float64,
        )

    # --------------------------------------------------
    # Min-Max normalization
    # --------------------------------------------------

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return normalized