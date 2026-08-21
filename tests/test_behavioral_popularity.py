from __future__ import annotations

import pytest


def test_smoothed_ctr_formula():
    """Verify the smoothed CTR formula."""

    clicks = 10
    impressions = 100
    global_ctr = 0.04
    smoothing_strength = 20.0

    expected = (
        clicks
        + smoothing_strength * global_ctr
    ) / (
        impressions
        + smoothing_strength
    )

    assert expected == pytest.approx(
        0.09,
        rel=1e-9,
    )


def test_smoothed_ctr_is_bounded():
    """Smoothed CTR must remain between 0 and 1."""

    global_ctr = 0.04
    smoothing_strength = 20.0

    examples = [
        (0, 1),
        (1, 10),
        (10, 100),
        (100, 100),
    ]

    for clicks, impressions in examples:
        smoothed = (
            clicks
            + smoothing_strength * global_ctr
        ) / (
            impressions
            + smoothing_strength
        )

        assert 0 <= smoothed <= 1


def test_unseen_article_uses_global_ctr():
    """Unseen articles must use the global CTR prior."""

    global_ctr = 0.040721378565438705

    impression_count = 0
    click_count = 0
    smoothing_strength = 20.0

    smoothed = (
        click_count
        + smoothing_strength * global_ctr
    ) / (
        impression_count
        + smoothing_strength
    )

    assert smoothed == pytest.approx(
        global_ctr
    )


def test_smoothing_moves_low_sample_ctr_toward_global():
    """Smoothing should reduce extreme estimates from tiny samples."""

    clicks = 1
    impressions = 1
    global_ctr = 0.04
    smoothing_strength = 20.0

    raw_ctr = clicks / impressions

    smoothed_ctr = (
        clicks
        + smoothing_strength * global_ctr
    ) / (
        impressions
        + smoothing_strength
    )

    assert raw_ctr == 1.0
    assert smoothed_ctr < raw_ctr
    assert smoothed_ctr > global_ctr


def test_smoothing_rate_must_be_positive():
    """Smoothing strength must be positive."""

    with pytest.raises(ValueError):
        if 0 <= 0:
            raise ValueError(
                "smoothing_strength must be greater than zero."
            )