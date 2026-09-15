import pytest
import torch

from src.ranking.phase5_reranker import Phase5Reranker


@pytest.fixture
def model():
    return Phase5Reranker(
        num_articles=100,
        embedding_dim=32,
        hidden_dim=32,
        retrieval_dim=5,
    )


@pytest.fixture
def sample_inputs():
    history = torch.tensor(
        [
            [0, 0, 3, 7, 12],
            [0, 0, 0, 0, 0],
            [2, 5, 8, 11, 14],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15, 20, 25],
        dtype=torch.long,
    )

    retrieval_features = torch.tensor(
        [
            [50.0, 1.0, 0.80, 2.0, 0.032],
            [40.0, 5.0, 0.70, 4.0, 0.029],
            [30.0, 10.0, 0.60, 8.0, 0.025],
        ],
        dtype=torch.float32,
    )

    return history, candidates, retrieval_features


def test_model_forward_shape(model, sample_inputs):
    history, candidates, retrieval_features = sample_inputs

    logits = model(
        history,
        candidates,
        retrieval_features,
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_predict_proba_shape_and_range(model, sample_inputs):
    history, candidates, retrieval_features = sample_inputs

    probabilities = model.predict_proba(
        history,
        candidates,
        retrieval_features,
    )

    assert probabilities.shape == (3,)
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_no_history_is_supported(model):
    history = torch.zeros(
        (3, 5),
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15, 20, 25],
        dtype=torch.long,
    )

    retrieval_features = torch.randn(
        3,
        5,
    )

    logits = model(
        history,
        candidates,
        retrieval_features,
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_single_example_batch(model):
    history = torch.tensor(
        [[0, 0, 3, 7, 12]],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15],
        dtype=torch.long,
    )

    retrieval_features = torch.tensor(
        [[50.0, 1.0, 0.8, 2.0, 0.032]],
        dtype=torch.float32,
    )

    logits = model(
        history,
        candidates,
        retrieval_features,
    )

    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_incorrect_retrieval_dimension_is_rejected(model):
    history = torch.tensor(
        [[0, 0, 3, 7, 12]],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15],
        dtype=torch.long,
    )

    # Model expects 5 retrieval features, but receives 4.
    retrieval_features = torch.randn(
        1,
        4,
    )

    with pytest.raises(ValueError, match="retrieval"):
        model(
            history,
            candidates,
            retrieval_features,
        )


def test_mismatched_batch_size_is_rejected(model):
    history = torch.tensor(
        [
            [0, 0, 3, 7, 12],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15, 20, 25],
        dtype=torch.long,
    )

    retrieval_features = torch.randn(
        3,
        5,
    )

    with pytest.raises(ValueError, match="batch"):
        model(
            history,
            candidates,
            retrieval_features,
        )


def test_history_mask_is_supported(model):
    history = torch.tensor(
        [
            [0, 0, 3, 7, 12],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    candidates = torch.tensor(
        [15, 20],
        dtype=torch.long,
    )

    retrieval_features = torch.randn(
        2,
        5,
    )

    history_mask = history != 0

    logits = model(
        history,
        candidates,
        retrieval_features,
        history_mask=history_mask,
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_gradients_flow(model, sample_inputs):
    history, candidates, retrieval_features = sample_inputs

    logits = model(
        history,
        candidates,
        retrieval_features,
    )

    loss = logits.mean()
    loss.backward()

    gradients_found = False

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            gradients_found = True

    assert gradients_found


def test_model_has_trainable_parameters(model):
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameters > 0


def test_model_is_deterministic_in_eval_mode(model, sample_inputs):
    history, candidates, retrieval_features = sample_inputs

    model.eval()

    with torch.no_grad():
        output_1 = model(
            history,
            candidates,
            retrieval_features,
        )

        output_2 = model(
            history,
            candidates,
            retrieval_features,
        )

    assert torch.allclose(output_1, output_2)


def test_different_retrieval_features_can_change_output(
    model,
    sample_inputs,
):
    history, candidates, retrieval_features = sample_inputs

    model.eval()

    retrieval_features_changed = retrieval_features.clone()
    retrieval_features_changed[:, 0] += 100.0

    with torch.no_grad():
        output_1 = model(
            history,
            candidates,
            retrieval_features,
        )

        output_2 = model(
            history,
            candidates,
            retrieval_features_changed,
        )

    assert not torch.allclose(output_1, output_2)