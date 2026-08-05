"""MNIST presets and scalable heterogeneous expert-pool builders."""

from __future__ import annotations

from dataclasses import replace

from sglr.config import ExperimentConfig, ExpertSpec, ModelConfig, SweepConfig, TrainingConfig


MLP_WIDTHS = (8, 12, 16, 24, 32, 48, 64, 96)
ATTENTION_SHAPES = ((8, 1), (12, 1), (16, 1), (16, 2), (24, 2), (24, 3), (32, 4), (48, 6))
CONV_TEMPLATES = (
    ExpertSpec("conv_token_3", "conv", channels=2, kernel_size=(3, 1)),
    ExpertSpec("conv_token_5", "conv", channels=2, kernel_size=(5, 1)),
    ExpertSpec("conv_feature_3", "conv", channels=2, kernel_size=(1, 3)),
    ExpertSpec("conv_feature_5", "conv", channels=2, kernel_size=(1, 5)),
    ExpertSpec("conv_joint_3x3", "conv", channels=2, kernel_size=(3, 3)),
    ExpertSpec("conv_joint_5x3", "conv", channels=3, kernel_size=(5, 3)),
    ExpertSpec("conv_dilated_token", "conv", channels=2, kernel_size=(3, 1), dilation=(2, 1)),
    ExpertSpec("conv_dilated_joint", "conv", channels=3, kernel_size=(3, 3), dilation=(2, 1)),
)


def make_expert_pool(
    mlp_experts: int = 8,
    attention_experts: int = 8,
    conv_experts: int = 8,
    dropout: float = 0.0,
) -> tuple[ExpertSpec, ...]:
    """Build any number of experts by cycling the canonical heterogeneous templates."""

    counts = (mlp_experts, attention_experts, conv_experts)
    if any(count < 0 for count in counts):
        raise ValueError("Expert-family counts must be non-negative")
    if sum(counts) == 0:
        raise ValueError("At least one expert is required")

    experts = (
        *_make_mlp_experts(mlp_experts, dropout),
        *_make_attention_experts(attention_experts, dropout),
        *_make_conv_experts(conv_experts, dropout),
    )
    return tuple(experts)


def _make_mlp_experts(count: int, dropout: float) -> list[ExpertSpec]:
    experts: list[ExpertSpec] = []
    for index in range(count):
        width = MLP_WIDTHS[index % len(MLP_WIDTHS)]
        base_name = f"mlp_{width:03d}"
        experts.append(
            ExpertSpec(
                name=_repeated_name(base_name, index, len(MLP_WIDTHS)),
                family="mlp",
                hidden_size=width,
                dropout=dropout,
            )
        )
    return experts


def _make_attention_experts(count: int, dropout: float) -> list[ExpertSpec]:
    experts: list[ExpertSpec] = []
    for index in range(count):
        internal_size, num_heads = ATTENTION_SHAPES[index % len(ATTENTION_SHAPES)]
        base_name = f"attention_{internal_size:03d}_h{num_heads}"
        experts.append(
            ExpertSpec(
                name=_repeated_name(base_name, index, len(ATTENTION_SHAPES)),
                family="attention",
                internal_size=internal_size,
                num_heads=num_heads,
                dropout=dropout,
            )
        )
    return experts


def _make_conv_experts(count: int, dropout: float) -> list[ExpertSpec]:
    experts: list[ExpertSpec] = []
    for index in range(count):
        template = CONV_TEMPLATES[index % len(CONV_TEMPLATES)]
        experts.append(
            replace(
                template,
                name=_repeated_name(template.name, index, len(CONV_TEMPLATES)),
                dropout=dropout,
            )
        )
    return experts


def _repeated_name(base_name: str, index: int, template_count: int) -> str:
    repetition = index // template_count
    return base_name if repetition == 0 else f"{base_name}_r{repetition + 1:02d}"


def smoke(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = pilot(experts_per_family=experts_per_family)
    return replace(
        experiment,
        experiment_name="smoke",
        training=replace(
            experiment.training,
            epochs=1,
            batch_size=256,
            patience=1,
            train_size=256,
            validation_size=128,
            test_size=128,
            log_interval=10,
        ),
    )


def pilot(*, experts_per_family: int = 8) -> ExperimentConfig:
    primary_pool = experts_per_family == 8
    experiment = ExperimentConfig(
        experiment_name="pilot",
        model=ModelConfig(
            experts=make_expert_pool(
                mlp_experts=experts_per_family,
                attention_experts=experts_per_family,
                conv_experts=experts_per_family,
            ),
            parameter_budget=200_000 if primary_pool else None,
            require_router_smaller_than_experts=primary_pool,
        ),
        training=TrainingConfig(
            epochs=5,
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=1e-4,
            warmup_fraction=0.05,
            grad_accum_steps=1,
            load_balance_coefficient=0.01,
            compute_penalty_coefficient=0.001,
            patience=3,
            train_size=12_000,
            validation_size=2_000,
            test_size=0,
            num_workers=0,
            seed=7,
            device="auto",
            log_interval=50,
        ),
    )
    experiment.validate()
    return experiment


def full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = pilot(experts_per_family=experts_per_family)
    return replace(
        experiment,
        experiment_name="full",
        training=replace(
            experiment.training,
            epochs=20,
            patience=5,
            train_size=60_000,
            validation_size=5_000,
            test_size=5_000,
            validation_source="official_test",
        ),
    )


def diverse_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="diverse_full",
        model=replace(experiment.model, max_steps=20),
        training=replace(
            experiment.training,
            epochs=15,
            patience=4,
            learning_rate=1e-3,
            load_balance_coefficient=0.03,
            within_family_balance_weight=1.0,
            route_mi_coefficient=0.1,
            compute_penalty_coefficient=0.025,
        ),
    )
    configured.validate()
    return configured


def fast_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_full",
        model=replace(
            experiment.model,
            max_steps=2,
            routing_mode="hard_argmax",
        ),
        training=replace(
            experiment.training,
            epochs=12,
            batch_size=2_048,
            learning_rate=2e-3,
            warmup_fraction=0.02,
            load_balance_coefficient=0.1,
            route_mi_coefficient=0.05,
            compute_penalty_coefficient=0.5,
            patience=3,
            log_interval=0,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_full",
        model=replace(
            experiment.model,
            encoder_width=32,
            parameter_budget=None,
            readout_hidden_size=128,
            require_router_smaller_than_experts=False,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_balanced_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_balanced_full",
        training=replace(
            experiment.training,
            load_balance_coefficient=0.2,
            within_family_balance_weight=2.0,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_balanced_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_full",
        model=replace(
            experiment.model,
            min_steps=5,
            max_steps=6,
        ),
        training=replace(
            experiment.training,
            epochs=8,
            batch_size=4_096,
            learning_rate=3e-3,
            patience=2,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_scaled_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_depth5_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_scaled_full",
        model=replace(
            experiment.model,
            expert_residual_scale=0.1,
        ),
        training=replace(
            experiment.training,
            load_balance_coefficient=0.5,
            compute_penalty_coefficient=3.0,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_curriculum_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_depth5_scaled_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_curriculum_full",
        training=replace(
            experiment.training,
            epochs=10,
            patience=3,
            routing_warmup_epochs=4,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_shared_router_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_depth5_curriculum_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_shared_router_full",
        model=replace(
            experiment.model,
            share_router_across_sources=True,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_sinkhorn_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_depth5_shared_router_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_sinkhorn_full",
        model=replace(
            experiment.model,
            sinkhorn_routing_iterations=100,
            sinkhorn_temperature=0.05,
        ),
    )
    configured.validate()
    return configured


def fast_cnn_depth5_capacity_full(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = fast_cnn_depth5_sinkhorn_full(experts_per_family=experts_per_family)
    configured = replace(
        experiment,
        experiment_name="fast_cnn_depth5_capacity_full",
        model=replace(
            experiment.model,
            capacity_balanced_evaluation=True,
        ),
    )
    configured.validate()
    return configured


def focused(*, experts_per_family: int = 8) -> ExperimentConfig:
    experiment = pilot(experts_per_family=experts_per_family)
    return replace(
        experiment,
        experiment_name="focused",
        sweep=SweepConfig(),
    )


MNIST_PRESETS = {
    "smoke": smoke,
    "pilot": pilot,
    "full": full,
    "diverse_full": diverse_full,
    "fast_full": fast_full,
    "fast_cnn_full": fast_cnn_full,
    "fast_cnn_balanced_full": fast_cnn_balanced_full,
    "fast_cnn_depth5_full": fast_cnn_depth5_full,
    "fast_cnn_depth5_scaled_full": fast_cnn_depth5_scaled_full,
    "fast_cnn_depth5_curriculum_full": fast_cnn_depth5_curriculum_full,
    "fast_cnn_depth5_shared_router_full": fast_cnn_depth5_shared_router_full,
    "fast_cnn_depth5_sinkhorn_full": fast_cnn_depth5_sinkhorn_full,
    "fast_cnn_depth5_capacity_full": fast_cnn_depth5_capacity_full,
    "focused": focused,
}
MNIST_PRESET_NAMES = tuple(MNIST_PRESETS)


def get_mnist_preset(name: str, *, experts_per_family: int | None = None) -> ExperimentConfig:
    try:
        factory = MNIST_PRESETS[name]
    except KeyError as error:
        choices = ", ".join(MNIST_PRESET_NAMES)
        raise ValueError(f"Unknown MNIST preset {name!r}; choose one of: {choices}") from error
    if experts_per_family is None:
        experiment = factory()
    else:
        experiment = factory(experts_per_family=experts_per_family)
        if experiment.model.num_experts != 24 and experiment.experiment_name == name:
            experiment = replace(
                experiment,
                experiment_name=f"{name}_{experiment.model.num_experts}_experts",
            )
    experiment.validate()
    return experiment
