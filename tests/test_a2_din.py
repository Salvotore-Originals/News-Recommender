"""
Tests for the Phase 3 local DIN-style baseline.

These tests verify the model architecture independently of the
real EB-NeRD dataset.
"""

import pytest
import torch

from src.ranking.a2_din import DINModel


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def model():
    """Create a small DIN model for testing."""
    return DINModel(
        num_articles=100,
        embedding_dim=16,
        hidden_dim=16,
        dropout=0.0,
        padding_idx=0,
    )


@pytest.fixture
def sample_inputs():
    """
    Create a small batch of synthetic user histories and candidates.

    Article ID 0 is reserved for padding.
    """
    history = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [13, 14, 15],
        dtype=torch.long,
    )

    return history, candidates


# ---------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------


def test_model_can_be_created(model):
    """The DIN model should initialize successfully."""
    assert isinstance(model, DINModel)


def test_model_contains_article_embedding(model):
    """The model should contain an article embedding layer."""
    assert isinstance(model.article_embedding, torch.nn.Embedding)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


def test_forward_output_shape(model, sample_inputs):
    """Forward pass should return one logit per candidate."""
    history, candidates = sample_inputs

    logits = model(
        history_article_ids=history,
        candidate_article_ids=candidates,
    )

    assert logits.shape == (3,)


def test_forward_output_is_finite(model, sample_inputs):
    """Model output should not contain NaN or infinite values."""
    history, candidates = sample_inputs

    logits = model(
        history_article_ids=history,
        candidate_article_ids=candidates,
    )

    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------
# Probability output
# ---------------------------------------------------------------------


def test_predict_proba_range(model, sample_inputs):
    """Predicted click probabilities must lie between 0 and 1."""
    history, candidates = sample_inputs

    probabilities = model.predict_proba(
        history_article_ids=history,
        candidate_article_ids=candidates,
    )

    assert probabilities.shape == (3,)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_predict_proba_is_finite(model, sample_inputs):
    """Predicted probabilities should be finite."""
    history, candidates = sample_inputs

    probabilities = model.predict_proba(
        history_article_ids=history,
        candidate_article_ids=candidates,
    )

    assert torch.isfinite(probabilities).all()


# ---------------------------------------------------------------------
# Padding / masking
# ---------------------------------------------------------------------


def test_padding_positions_are_supported(model):
    """
    Padded history positions should be accepted without producing
    invalid outputs.
    """
    history = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10, 11],
        dtype=torch.long,
    )

    logits = model(history, candidates)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_explicit_history_mask_is_supported(model):
    """The model should accept an explicit boolean history mask."""
    history = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10, 11],
        dtype=torch.long,
    )

    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ],
        dtype=torch.bool,
    )

    logits = model(
        history_article_ids=history,
        candidate_article_ids=candidates,
        history_mask=mask,
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_all_padding_history_is_supported(model):
    """
    A user with no usable history should still produce a valid
    prediction.
    """
    history = torch.zeros(
        (2, 5),
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10, 11],
        dtype=torch.long,
    )

    logits = model(history, candidates)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_invalid_history_dimensions_raise_error(model):
    """History input must be two-dimensional."""
    history = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(history, candidates)


def test_invalid_candidate_dimensions_raise_error(model):
    """Candidate input must be one-dimensional."""
    history = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [[10]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(history, candidates)


def test_mismatched_batch_sizes_raise_error(model):
    """History and candidate batch sizes must match."""
    history = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(history, candidates)


def test_invalid_history_mask_shape_raises_error(model):
    """History mask must match the history tensor shape."""
    history = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [10],
        dtype=torch.long,
    )

    invalid_mask = torch.tensor(
        [[True, True]],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError):
        model(
            history,
            candidates,
            history_mask=invalid_mask,
        )


# ---------------------------------------------------------------------
# Gradient / training behavior
# ---------------------------------------------------------------------


def test_gradients_flow_through_model(model, sample_inputs):
    """
    The model must support backpropagation because Phase 3 will train
    it using binary cross-entropy loss.
    """
    history, candidates = sample_inputs

    labels = torch.tensor(
        [1.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    logits = model(history, candidates)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
    )

    loss.backward()

    # At least one trainable parameter must receive a gradient.
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_optimizer_step_changes_parameters(model, sample_inputs):
    """
    A training step should be capable of changing model parameters.
    """
    history, candidates = sample_inputs

    labels = torch.tensor(
        [1.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    logits = model(history, candidates)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = False

    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter.detach()):
            changed = True
            break

    assert changed