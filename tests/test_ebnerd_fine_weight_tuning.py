from src.retrieval.ebnerd_fine_weight_tuning import (
    generate_fine_weight_grid,
)


def test_fine_grid_count():

    configs = (
        generate_fine_weight_grid()
    )

    assert len(configs) == 17


def test_fine_weights_sum_to_one():

    configs = (
        generate_fine_weight_grid()
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


def test_fine_weights_non_negative():

    configs = (
        generate_fine_weight_grid()
    )

    for config in configs:

        assert config[
            "lexical_weight"
        ] >= 0

        assert config[
            "semantic_weight"
        ] >= 0

        assert config[
            "behavioral_weight"
        ] >= 0


def test_coarse_optimum_is_included():

    configs = (
        generate_fine_weight_grid()
    )

    assert {
        "lexical_weight": 0.0,
        "semantic_weight": 0.10,
        "behavioral_weight": 0.90,
    } in configs


def test_pure_behavioral_is_included():

    configs = (
        generate_fine_weight_grid()
    )

    assert {
        "lexical_weight": 0.0,
        "semantic_weight": 0.0,
        "behavioral_weight": 1.0,
    } in configs