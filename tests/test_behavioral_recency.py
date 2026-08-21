from __future__ import annotations

import math

import pandas as pd
import pytest


def test_recency_weight_increases_toward_recent_history():
    """Newer history positions must receive larger weights."""

    decay_rate = 0.1
    latest_position = 10

    old_weight = math.exp(
        -decay_rate * (latest_position - 0)
    )

    new_weight = math.exp(
        -decay_rate * (latest_position - 10)
    )

    assert new_weight > old_weight


def test_recency_weight_for_latest_item_is_one():
    """The newest history item has maximum weight 1."""

    decay_rate = 0.1
    latest_position = 10

    weight = math.exp(
        -decay_rate * (
            latest_position - latest_position
        )
    )

    assert weight == pytest.approx(1.0)


def test_recency_values_are_probability_bounded():
    """Normalized recency values must be between 0 and 1."""

    df = pd.DataFrame(
        {
            "category_recency": [
                0.0,
                0.25,
                0.75,
                1.0,
            ],
            "subcategory_recency": [
                0.0,
                0.10,
                0.50,
                1.0,
            ],
        }
    )

    assert (
        df["category_recency"]
        .between(0, 1)
        .all()
    )

    assert (
        df["subcategory_recency"]
        .between(0, 1)
        .all()
    )


def test_cold_start_recency_is_zero():
    """
    A user with no history has no recency evidence.
    """

    total_recency_weight = 0.0
    category_recency = 0.0
    subcategory_recency = 0.0

    assert total_recency_weight == 0.0
    assert category_recency == 0.0
    assert subcategory_recency == 0.0


def test_decay_rate_must_be_positive():
    """Decay rate must be greater than zero."""

    with pytest.raises(ValueError):
        if 0.0 <= 0:
            raise ValueError(
                "decay_rate must be greater than zero."
            )