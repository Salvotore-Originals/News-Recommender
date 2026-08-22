from src.retrieval.ebnerd_weight_tuning import (
    generate_weight_grid,
)


def test_weight_grid_count():

    configs = generate_weight_grid(
        step=0.1
    )

    assert len(configs) == 66


def test_weights_sum_to_one():

    configs = generate_weight_grid(
        step=0.1
    )

    for config in configs:

        total = (
            config["lexical_weight"]
            + config["semantic_weight"]
            + config["behavioral_weight"]
        )

        assert abs(
            total - 1.0
        ) < 1e-9


def test_weights_non_negative():

    configs = generate_weight_grid(
        step=0.1
    )

    for config in configs:

        assert (
            config["lexical_weight"]
            >= 0
        )

        assert (
            config["semantic_weight"]
            >= 0
        )

        assert (
            config["behavioral_weight"]
            >= 0
        )


def test_pure_behavioral_exists():

    configs = generate_weight_grid(
        step=0.1
    )

    assert {
        "lexical_weight": 0.0,
        "semantic_weight": 0.0,
        "behavioral_weight": 1.0,
    } in configs


def test_equal_tenth_hybrid_exists():

    configs = generate_weight_grid(
        step=0.1
    )

    assert {
        "lexical_weight": 0.3,
        "semantic_weight": 0.3,
        "behavioral_weight": 0.4,
    } in configs