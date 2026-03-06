from __future__ import annotations

import argparse
from dataclasses import dataclass, replace


@dataclass(slots=True)
class ModelConfig:
    input_size: int = 784
    image_side: int = 28
    num_classes: int = 10
    num_mlp_experts: int = 4
    mlp_hidden_size: int = 2048
    mlp_dropout: float = 0.1
    num_conv_experts: int = 4
    conv_dropout: float = 0.1
    num_attention_experts: int = 4
    attention_hidden_size: int = 128
    attention_head_size: int = 64
    attention_dropout: float = 0.1
    router_hidden_size: int = 512
    router_dropout: float = 0.0
    max_repeats: int = 4
    min_steps_before_exit: int = 1

    @property
    def num_experts(self) -> int:
        return self.num_mlp_experts + self.num_conv_experts + self.num_attention_experts

    @property
    def num_routes(self) -> int:
        return self.num_experts + 1

    @property
    def max_steps(self) -> int:
        return self.num_experts * self.max_repeats

    def validate(self) -> None:
        assert self.input_size == self.image_side * self.image_side, "input_size must equal image_side squared"
        assert self.num_experts > 0, "At least one expert is required"
        assert self.max_repeats > 0, "max_repeats must be positive"
        assert self.min_steps_before_exit >= 0, "min_steps_before_exit must be non-negative"
        assert self.attention_hidden_size % self.attention_head_size == 0, "attention_hidden_size must divide by attention_head_size"
        assert self.attention_head_size % 2 == 0, "attention_head_size must be even for rotary embeddings"


@dataclass(slots=True)
class TrainingConfig:
    output_root: str = "runs"
    run_name: str = "sglr_mnist"
    data_root: str = "data"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_accum_steps: int = 1
    load_balancing_coef: float = 0.1
    patience: int = 5
    eval_batches: int = 0
    train_subset: int = 0
    test_subset: int = 0
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"
    plot_max_samples: int = 2000
    log_interval: int = 50
    smoke_test: bool = False


@dataclass(slots=True)
class ProbeTrainingConfig:
    base_run_dir: str = ""
    data_root: str = "data"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_accum_steps: int = 1
    classification_loss_coef: float = 1.0
    reconstruction_loss_coef: float = 1.0
    patience: int = 3
    train_subset: int = 0
    test_subset: int = 0
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"
    analysis_samples: int = 256
    log_interval: int = 50
    smoke_test: bool = False


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed_value


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-size", type=positive_int, default=784, help="Flattened latent size for each MNIST sample.")
    parser.add_argument("--image-side", type=positive_int, default=28, help="Image width and height used by conv and attention experts.")
    parser.add_argument("--num-classes", type=positive_int, default=10, help="Number of output classes.")
    parser.add_argument("--num-mlp-experts", type=positive_int, default=2, help="Number of MLP experts.")
    parser.add_argument("--mlp-hidden-size", type=positive_int, default=1024, help="Hidden width for each MLP expert.")
    parser.add_argument("--mlp-dropout", type=float, default=0.1, help="Dropout probability inside MLP experts.")
    parser.add_argument("--num-conv-experts", type=positive_int, default=2, help="Number of convolutional experts.")
    parser.add_argument("--conv-dropout", type=float, default=0.1, help="Dropout probability inside convolutional experts.")
    parser.add_argument("--num-attention-experts", type=positive_int, default=2, help="Number of attention experts.")
    parser.add_argument("--attention-hidden-size", type=positive_int, default=128, help="Hidden size for each attention expert.")
    parser.add_argument("--attention-head-size", type=positive_int, default=64, help="Per-head size for each attention expert.")
    parser.add_argument("--attention-dropout", type=float, default=0.1, help="Dropout probability inside attention experts.")
    parser.add_argument("--router-hidden-size", type=positive_int, default=512, help="Hidden size for the routing MLPs.")
    parser.add_argument("--router-dropout", type=float, default=0.0, help="Dropout probability inside the routing MLPs.")
    parser.add_argument("--max-repeats", type=positive_int, default=2, help="Maximum repeats per expert family, used to define the route horizon.")
    parser.add_argument("--min-steps-before-exit", type=int, default=1, help="Minimum number of routing steps before the exit route is allowed.")


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    model_config = ModelConfig(
        input_size=args.input_size,
        image_side=args.image_side,
        num_classes=args.num_classes,
        num_mlp_experts=args.num_mlp_experts,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_dropout=args.mlp_dropout,
        num_conv_experts=args.num_conv_experts,
        conv_dropout=args.conv_dropout,
        num_attention_experts=args.num_attention_experts,
        attention_hidden_size=args.attention_hidden_size,
        attention_head_size=args.attention_head_size,
        attention_dropout=args.attention_dropout,
        router_hidden_size=args.router_hidden_size,
        router_dropout=args.router_dropout,
        max_repeats=args.max_repeats,
        min_steps_before_exit=args.min_steps_before_exit,
    )
    model_config.validate()
    return model_config


def parse_train_configs(argv: list[str] | None = None) -> tuple[ModelConfig, TrainingConfig]:
    parser = argparse.ArgumentParser(
        description="Train the SGLR MNIST model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    path_group = parser.add_argument_group("paths")
    path_group.add_argument("--output-root", type=str, default="runs", help="Directory where training runs are written.")
    path_group.add_argument("--run-name", type=str, default="sglr_mnist", help="Human-readable name for this training run.")
    path_group.add_argument("--data-root", type=str, default="data", help="Directory where MNIST is downloaded or loaded from.")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--batch-size", type=positive_int, default=32, help="Microbatch size used for each forward pass.")
    data_group.add_argument("--train-subset", type=int, default=0, help="Optional prefix subset size for training data. Use 0 for the full dataset.")
    data_group.add_argument("--test-subset", type=int, default=0, help="Optional prefix subset size for evaluation data. Use 0 for the full dataset.")
    data_group.add_argument("--num-workers", type=int, default=0, help="Number of dataloader worker processes.")

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--epochs", type=positive_int, default=10, help="Number of training epochs.")
    optimization_group.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optimization_group.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    optimization_group.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps for the cosine scheduler.")
    optimization_group.add_argument("--grad-accum-steps", type=positive_int, default=1, help="Number of microbatches to merge into one optimizer step.")
    optimization_group.add_argument("--load-balancing-coef", type=float, default=0.1, help="Coefficient applied to the routing load-balancing loss.")
    optimization_group.add_argument("--patience", type=positive_int, default=5, help="Early stopping patience measured in epochs.")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--eval-batches", type=int, default=0, help="Optional cap on evaluation batches per epoch. Use 0 for all batches.")
    runtime_group.add_argument("--seed", type=int, default=7, help="Random seed for Python and Torch.")
    runtime_group.add_argument("--device", type=str, default="auto", help="Torch device string, or 'auto' to prefer CUDA.")
    runtime_group.add_argument("--log-interval", type=positive_int, default=50, help="Number of batches between progress updates.")
    runtime_group.add_argument("--smoke-test", action="store_true", help="Override several settings for a tiny verification run.")

    analysis_group = parser.add_argument_group("analysis")
    analysis_group.add_argument("--plot-max-samples", type=positive_int, default=2000, help="Maximum evaluation samples to use for route-pattern plots.")

    model_group = parser.add_argument_group("model")
    add_model_arguments(model_group)
    args = parser.parse_args(argv)

    model_config = build_model_config(args)
    training_config = TrainingConfig(
        output_root=args.output_root,
        run_name=args.run_name,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        load_balancing_coef=args.load_balancing_coef,
        patience=args.patience,
        eval_batches=args.eval_batches,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        plot_max_samples=args.plot_max_samples,
        log_interval=args.log_interval,
        smoke_test=args.smoke_test,
    )

    if training_config.smoke_test:
        training_config = replace(
            training_config,
            epochs=1,
            batch_size=min(training_config.batch_size, 8),
            eval_batches=4 if training_config.eval_batches == 0 else training_config.eval_batches,
            grad_accum_steps=min(training_config.grad_accum_steps, 4),
            patience=2,
            plot_max_samples=min(training_config.plot_max_samples, 128),
            train_subset=256 if training_config.train_subset == 0 else training_config.train_subset,
            test_subset=128 if training_config.test_subset == 0 else training_config.test_subset,
        )

    return model_config, training_config


def parse_probe_config(argv: list[str] | None = None) -> ProbeTrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train probes on a frozen SGLR MNIST model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    path_group = parser.add_argument_group("paths")
    path_group.add_argument("--base-run-dir", type=str, required=True, help="Run directory that already contains base model checkpoints.")
    path_group.add_argument("--data-root", type=str, default="data", help="Directory where MNIST is downloaded or loaded from.")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--batch-size", type=positive_int, default=32, help="Microbatch size used for each forward pass.")
    data_group.add_argument("--train-subset", type=int, default=0, help="Optional prefix subset size for training data. Use 0 for the full dataset.")
    data_group.add_argument("--test-subset", type=int, default=0, help="Optional prefix subset size for evaluation data. Use 0 for the full dataset.")
    data_group.add_argument("--num-workers", type=int, default=0, help="Number of dataloader worker processes.")

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--epochs", type=positive_int, default=5, help="Number of probe-training epochs.")
    optimization_group.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate for the probe heads.")
    optimization_group.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay for the probe heads.")
    optimization_group.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for the cosine scheduler.")
    optimization_group.add_argument("--grad-accum-steps", type=positive_int, default=1, help="Number of microbatches to merge into one optimizer step.")
    optimization_group.add_argument("--classification-loss-coef", type=float, default=1.0, help="Coefficient applied to the probe classification loss.")
    optimization_group.add_argument("--reconstruction-loss-coef", type=float, default=1.0, help="Coefficient applied to the probe reconstruction loss.")
    optimization_group.add_argument("--patience", type=positive_int, default=3, help="Early stopping patience measured in epochs.")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=7, help="Random seed for Python and Torch.")
    runtime_group.add_argument("--device", type=str, default="auto", help="Torch device string, or 'auto' to prefer CUDA.")
    runtime_group.add_argument("--analysis-samples", type=positive_int, default=256, help="Number of examples to reserve for optional downstream analysis.")
    runtime_group.add_argument("--log-interval", type=positive_int, default=50, help="Number of batches between progress updates.")
    runtime_group.add_argument("--smoke-test", action="store_true", help="Override several settings for a tiny verification run.")
    args = parser.parse_args(argv)

    probe_config = ProbeTrainingConfig(
        base_run_dir=args.base_run_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        classification_loss_coef=args.classification_loss_coef,
        reconstruction_loss_coef=args.reconstruction_loss_coef,
        patience=args.patience,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        analysis_samples=args.analysis_samples,
        log_interval=args.log_interval,
        smoke_test=args.smoke_test,
    )

    if probe_config.smoke_test:
        probe_config = replace(
            probe_config,
            epochs=1,
            batch_size=min(probe_config.batch_size, 8),
            grad_accum_steps=min(probe_config.grad_accum_steps, 4),
            patience=2,
            analysis_samples=min(probe_config.analysis_samples, 64),
            train_subset=256 if probe_config.train_subset == 0 else probe_config.train_subset,
            test_subset=128 if probe_config.test_subset == 0 else probe_config.test_subset,
        )

    return probe_config
