"""Typed configuration for SGLR models and MNIST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping


SCHEMA_VERSION = 4
EXPERT_FAMILIES = {"mlp", "conv", "attention"}
ROUTING_MODES = {"straight_through", "hard_argmax", "frozen_random", "fixed_depth"}


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    """Configuration for one shape-preserving expert."""

    name: str
    family: Literal["mlp", "conv", "attention"]
    hidden_size: int = 0
    internal_size: int = 0
    num_heads: int = 0
    channels: int = 0
    kernel_size: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    dropout: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Expert names cannot be empty")
        if self.family not in EXPERT_FAMILIES:
            raise ValueError(f"Unsupported expert family: {self.family}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"Expert {self.name} dropout must be in [0, 1)")

        if self.family == "mlp" and self.hidden_size <= 0:
            raise ValueError(f"MLP expert {self.name} requires a positive hidden_size")
        if self.family == "attention":
            if self.internal_size <= 0 or self.num_heads <= 0:
                raise ValueError(f"Attention expert {self.name} requires positive internal_size and num_heads")
            if self.internal_size % self.num_heads != 0:
                raise ValueError(f"Attention expert {self.name} internal_size must be divisible by num_heads")
        if self.family == "conv":
            if self.channels <= 0:
                raise ValueError(f"Convolution expert {self.name} requires positive channels")
            if len(self.kernel_size) != 2 or len(self.dilation) != 2:
                raise ValueError(f"Convolution expert {self.name} expects two-dimensional kernel and dilation")
            if any(value <= 0 or value % 2 == 0 for value in self.kernel_size):
                raise ValueError(f"Convolution expert {self.name} kernel sizes must be positive and odd")
            if any(value <= 0 for value in self.dilation):
                raise ValueError(f"Convolution expert {self.name} dilation must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_size: int = 48
    image_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    max_steps: int = 5
    min_steps: int = 1
    router_hidden_size: int = 16
    router_dropout: float = 0.0
    routing_mode: str = "straight_through"
    experts: tuple[ExpertSpec, ...] = field(default_factory=tuple)
    parameter_budget: int | None = 200_000
    require_router_smaller_than_experts: bool = True

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def num_routes(self) -> int:
        return self.num_experts + 1

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 1 <= self.min_steps <= self.max_steps:
            raise ValueError("min_steps must be between one and max_steps")
        if self.router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive")
        if not 0.0 <= self.router_dropout < 1.0:
            raise ValueError("router_dropout must be in [0, 1)")
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
        if self.parameter_budget is not None and self.parameter_budget <= 0:
            raise ValueError("parameter_budget must be positive or None")
        if not self.experts:
            raise ValueError("At least one expert is required")

        expert_names = [expert.name for expert in self.experts]
        if len(expert_names) != len(set(expert_names)):
            raise ValueError("Expert names must be unique")
        for expert in self.experts:
            expert.validate()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    output_root: str = "runs"
    data_root: str = "data"
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    grad_accum_steps: int = 1
    load_balance_coefficient: float = 0.01
    within_family_balance_weight: float = 1.0
    route_mi_coefficient: float = 0.0
    compute_penalty_coefficient: float = 0.005
    patience: int = 3
    train_size: int = 12_000
    validation_size: int = 2_000
    test_size: int = 10_000
    validation_source: Literal["official_train", "official_test"] = "official_train"
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"
    log_interval: int = 50

    def validate(self) -> None:
        positive_values = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "grad_accum_steps": self.grad_accum_steps,
            "patience": self.patience,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.test_size < 0 or self.num_workers < 0:
            raise ValueError("test_size and num_workers must be non-negative")
        if self.validation_source not in {"official_train", "official_test"}:
            raise ValueError("validation_source must be 'official_train' or 'official_test'")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        auxiliary_coefficients = (
            self.load_balance_coefficient,
            self.within_family_balance_weight,
            self.route_mi_coefficient,
            self.compute_penalty_coefficient,
        )
        if any(coefficient < 0.0 for coefficient in auxiliary_coefficients):
            raise ValueError("Auxiliary loss coefficients must be non-negative")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    variants: tuple[str, ...] = (
        "straight_through",
        "hard_argmax",
        "frozen_random",
        "fixed_depth",
    )
    seeds: tuple[int, ...] = (7, 17, 27)

    def validate(self) -> None:
        if not self.variants or not self.seeds:
            raise ValueError("Sweep variants and seeds cannot be empty")
        unknown_variants = set(self.variants) - ROUTING_MODES
        if unknown_variants:
            raise ValueError(f"Unsupported sweep variants: {sorted(unknown_variants)}")
        if len(self.variants) != len(set(self.variants)):
            raise ValueError("Sweep variants must be unique")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Sweep seeds must be unique")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str = "sglr"
    sweep: SweepConfig | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version {SCHEMA_VERSION}, got {self.schema_version}")
        self.model.validate()
        self.training.validate()
        if not self.experiment_name or Path(self.experiment_name).name != self.experiment_name:
            raise ValueError("experiment_name must be a non-empty directory name")
        if self.sweep is not None:
            self.sweep.validate()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def experiment_from_dict(payload: object) -> ExperimentConfig:
    """Rehydrate a resolved manifest configuration without consulting a preset."""

    resolved_payload = _mapping(payload, "experiment")
    allowed_keys = {"schema_version", "experiment_name", "model", "training", "sweep"}
    unknown_keys = set(resolved_payload) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown experiment configuration keys: {sorted(unknown_keys)}")

    model_payload = dict(_mapping(resolved_payload.get("model"), "model"))
    raw_experts = model_payload.pop("experts", None)
    if not isinstance(raw_experts, (list, tuple)):
        raise ValueError("Resolved model configuration must contain an expert list")
    experts = tuple(_expert_from_dict(_mapping(value, "expert")) for value in raw_experts)
    model = ModelConfig(experts=experts, **model_payload)

    training = TrainingConfig(**dict(_mapping(resolved_payload.get("training"), "training")))
    raw_sweep = resolved_payload.get("sweep")
    sweep = None
    if raw_sweep is not None:
        sweep_payload = dict(_mapping(raw_sweep, "sweep"))
        sweep = SweepConfig(
            variants=tuple(str(value) for value in sweep_payload.get("variants", ())),
            seeds=tuple(int(value) for value in sweep_payload.get("seeds", ())),
        )

    experiment = ExperimentConfig(
        schema_version=int(resolved_payload.get("schema_version", 0)),
        experiment_name=str(resolved_payload.get("experiment_name", "")),
        model=model,
        training=training,
        sweep=sweep,
    )
    experiment.validate()
    return experiment


def _expert_from_dict(payload: Mapping[str, Any]) -> ExpertSpec:
    values = dict(payload)
    values["kernel_size"] = tuple(int(value) for value in values.get("kernel_size", (1, 1)))
    values["dilation"] = tuple(int(value) for value in values.get("dilation", (1, 1)))
    return ExpertSpec(**values)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Resolved {name} configuration must be a mapping")
    return value


def with_run_overrides(
    experiment: ExperimentConfig,
    routing_mode: str | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    overridden = replace(
        experiment,
        model=replace(
            experiment.model,
            routing_mode=experiment.model.routing_mode if routing_mode is None else routing_mode,
        ),
        training=replace(
            experiment.training,
            seed=experiment.training.seed if seed is None else seed,
        ),
    )
    overridden.validate()
    return overridden
