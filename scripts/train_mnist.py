"""Train or resume one SGLR MNIST experiment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from sglr.artifacts import run_directory, run_is_complete, validate_run_config
from sglr.config import ROUTING_MODES, load_experiment_config, with_run_overrides
from sglr.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one schema-versioned SGLR MNIST experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML experiment configuration.")
    parser.add_argument("--variant", choices=sorted(ROUTING_MODES), help="Override model.routing_mode.")
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download missing MNIST files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    experiment = with_run_overrides(
        load_experiment_config(args.config),
        routing_mode=args.variant,
        seed=args.seed,
    )
    if args.device is not None:
        experiment = replace(
            experiment,
            training=replace(experiment.training, device=args.device),
        )
    experiment_name = experiment.experiment_name
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
    )
    print(f"Completed run: {completed_path}")


if __name__ == "__main__":
    main()
