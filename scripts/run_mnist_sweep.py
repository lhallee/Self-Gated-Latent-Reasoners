"""Run or resume every variant and seed in an SGLR sweep configuration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from sglr.artifacts import run_directory, run_is_complete, validate_run_config
from sglr.config import load_experiment_config, with_run_overrides
from sglr.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a schema-versioned SGLR MNIST sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML configuration containing [sweep].")
    parser.add_argument("--device", help="Override the configured Torch device for every run.")
    parser.add_argument("--download", action="store_true", help="Allow the first run to download missing MNIST files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base_experiment = load_experiment_config(args.config)
    if base_experiment.sweep is None:
        raise ValueError(f"Sweep configuration is missing [sweep]: {args.config}")
    experiment_name = base_experiment.experiment_name
    completed = 0
    skipped = 0
    allow_download = args.download

    for variant in base_experiment.sweep.variants:
        for seed in base_experiment.sweep.seeds:
            experiment = with_run_overrides(base_experiment, routing_mode=variant, seed=seed)
            if args.device is not None:
                experiment = replace(
                    experiment,
                    training=replace(experiment.training, device=args.device),
                )
            run_path = run_directory(
                experiment.training.output_root,
                experiment_name,
                variant,
                seed,
            )
            if run_is_complete(run_path):
                validate_run_config(run_path, experiment)
                print(f"Skipping completed run: {run_path}")
                skipped += 1
                continue

            command = [
                sys.executable,
                "-m",
                "scripts.run_mnist_sweep",
                *(argv if argv is not None else sys.argv[1:]),
            ]
            run_experiment(
                experiment=experiment,
                experiment_name=experiment_name,
                download=allow_download,
                command=command,
            )
            allow_download = False
            completed += 1

    print(f"Sweep complete: {completed} trained or resumed, {skipped} skipped")


if __name__ == "__main__":
    main()
