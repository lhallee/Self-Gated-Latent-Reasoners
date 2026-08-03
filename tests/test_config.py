from __future__ import annotations

import pytest

from sglr.config import ExpertSpec, ModelConfig, load_experiment_config


@pytest.mark.parametrize(
    "path",
    [
        "configs/mnist/smoke.toml",
        "configs/mnist/pilot.toml",
        "configs/mnist/full.toml",
        "configs/mnist/focused.toml",
    ],
)
def test_experiment_presets_are_valid(path: str) -> None:
    experiment = load_experiment_config(path)

    assert experiment.schema_version == 2
    assert experiment.model.num_experts == 24
    assert experiment.model.hidden_size == 48


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
