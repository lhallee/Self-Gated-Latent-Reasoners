"""Train or resume one SGLR MNIST experiment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from scripts.arguments import positive_int
from sglr.artifacts import run_directory, run_is_complete, validate_run_config
from sglr.config import ROUTING_MODES, with_run_overrides
from sglr.presets.mnist import MNIST_PRESET_NAMES, get_mnist_preset
from sglr.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one schema-versioned SGLR MNIST experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=MNIST_PRESET_NAMES, required=True)
    parser.add_argument(
        "--experts-per-family",
        type=positive_int,
        help="Override the selected preset's balanced expert-family count.",
    )
    parser.add_argument("--variant", choices=sorted(ROUTING_MODES), help="Override model.routing_mode.")
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download missing MNIST files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars for batch logs or CI.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate the validation split and leave the sealed test split untouched.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    experiment = with_run_overrides(
        get_mnist_preset(args.preset, experts_per_family=args.experts_per_family),
        routing_mode=args.variant,
        seed=args.seed,
    )
    if args.device is not None:
        experiment = replace(
            experiment,
            training=replace(experiment.training, device=args.device),
        )
    evaluation_split = "validation" if args.validation_only else "test"
    experiment_name = (
        f"{experiment.experiment_name}_validation"
        if args.validation_only
        else experiment.experiment_name
    )
    run_path = run_directory(
        experiment.training.output_root,
        experiment_name,
        experiment.model.routing_mode,
        experiment.training.seed,
    )
    if run_is_complete(run_path):
        validate_run_config(run_path, experiment)
        print(f"Completed run already exists: {run_path}")
        return

    command = [sys.executable, "-m", "scripts.train_mnist", *(argv if argv is not None else sys.argv[1:])]
    completed_path = run_experiment(
        experiment=experiment,
        experiment_name=experiment_name,
        download=args.download,
        command=command,
        show_progress=not args.no_progress,
        evaluation_split=evaluation_split,
    )
    print(f"Completed run: {completed_path}")


if __name__ == "__main__":
    main()
