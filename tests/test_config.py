from __future__ import annotations

from dataclasses import asdict

import pytest

from sglr.config import ExpertSpec, ModelConfig, experiment_from_dict
from sglr.presets.mnist import MNIST_PRESET_NAMES, get_mnist_preset, make_expert_pool


@pytest.mark.parametrize(
    "preset_name",
    MNIST_PRESET_NAMES,
)
def test_experiment_presets_are_valid(preset_name: str) -> None:
    experiment = get_mnist_preset(preset_name)

    assert experiment.schema_version == 3
    assert experiment.model.num_experts == 24
    assert experiment.model.hidden_size == 48


def test_full_preset_uses_all_training_images_and_seals_half_the_test_split() -> None:
    experiment = get_mnist_preset("full")

    assert experiment.training.batch_size == 256
    assert experiment.training.train_size == 60_000
    assert experiment.training.validation_size == 5_000
    assert experiment.training.test_size == 5_000
    assert experiment.training.validation_source == "official_test"


def test_primary_expert_order_is_stable() -> None:
    assert make_expert_pool() == (
        ExpertSpec("mlp_008", "mlp", hidden_size=8),
        ExpertSpec("mlp_012", "mlp", hidden_size=12),
        ExpertSpec("mlp_016", "mlp", hidden_size=16),
        ExpertSpec("mlp_024", "mlp", hidden_size=24),
        ExpertSpec("mlp_032", "mlp", hidden_size=32),
        ExpertSpec("mlp_048", "mlp", hidden_size=48),
        ExpertSpec("mlp_064", "mlp", hidden_size=64),
        ExpertSpec("mlp_096", "mlp", hidden_size=96),
        ExpertSpec("attention_008_h1", "attention", internal_size=8, num_heads=1),
        ExpertSpec("attention_012_h1", "attention", internal_size=12, num_heads=1),
        ExpertSpec("attention_016_h1", "attention", internal_size=16, num_heads=1),
        ExpertSpec("attention_016_h2", "attention", internal_size=16, num_heads=2),
        ExpertSpec("attention_024_h2", "attention", internal_size=24, num_heads=2),
        ExpertSpec("attention_024_h3", "attention", internal_size=24, num_heads=3),
        ExpertSpec("attention_032_h4", "attention", internal_size=32, num_heads=4),
        ExpertSpec("attention_048_h6", "attention", internal_size=48, num_heads=6),
        ExpertSpec("conv_token_3", "conv", channels=2, kernel_size=(3, 1)),
        ExpertSpec("conv_token_5", "conv", channels=2, kernel_size=(5, 1)),
        ExpertSpec("conv_feature_3", "conv", channels=2, kernel_size=(1, 3)),
        ExpertSpec("conv_feature_5", "conv", channels=2, kernel_size=(1, 5)),
        ExpertSpec("conv_joint_3x3", "conv", channels=2, kernel_size=(3, 3)),
        ExpertSpec("conv_joint_5x3", "conv", channels=3, kernel_size=(5, 3)),
        ExpertSpec(
            "conv_dilated_token",
            "conv",
            channels=2,
            kernel_size=(3, 1),
            dilation=(2, 1),
        ),
        ExpertSpec(
            "conv_dilated_joint",
            "conv",
            channels=3,
            kernel_size=(3, 3),
            dilation=(2, 1),
        ),
    )


def test_large_expert_pool_cycles_templates_with_unique_names() -> None:
    experts = make_expert_pool(mlp_experts=17, attention_experts=9, conv_experts=10)
    names = [expert.name for expert in experts]

    assert len(experts) == 36
    assert len(names) == len(set(names))
    assert names[8] == "mlp_008_r02"
    assert names[16] == "mlp_008_r03"


def test_expanded_preset_relaxes_canonical_parameter_guards() -> None:
    experiment = get_mnist_preset("pilot", experts_per_family=16)

    assert experiment.model.num_experts == 48
    assert experiment.experiment_name == "pilot_48_experts"
    assert experiment.model.parameter_budget is None
    assert not experiment.model.require_router_smaller_than_experts


def test_manifest_configuration_round_trip() -> None:
    experiment = get_mnist_preset("focused")

    assert experiment_from_dict(asdict(experiment)) == experiment


def test_unknown_preset_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="smoke, pilot, full, focused"):
        get_mnist_preset("unknown")


def test_duplicate_expert_names_are_rejected() -> None:
    config = ModelConfig(
        experts=(
            ExpertSpec("duplicate", "mlp", hidden_size=8),
            ExpertSpec("duplicate", "mlp", hidden_size=16),
        )
    )

    with pytest.raises(ValueError, match="unique"):
        config.validate()


def test_invalid_attention_shape_is_rejected() -> None:
    spec = ExpertSpec("bad_attention", "attention", internal_size=10, num_heads=3)

    with pytest.raises(ValueError, match="divisible"):
        spec.validate()


def test_zero_minimum_depth_is_rejected() -> None:
    config = ModelConfig(
        min_steps=0,
        experts=(ExpertSpec("mlp", "mlp", hidden_size=8),),
    )

    with pytest.raises(ValueError, match="between one"):
        config.validate()
