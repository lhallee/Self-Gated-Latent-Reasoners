"""Regenerate SGLR MNIST analyses from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

from sglr.artifacts import (
    load_checkpoint,
    load_json,
    run_is_complete,
    save_json,
)
from sglr.analysis import SweepRun, build_sweep_run, load_evaluation_records, load_json_object
from sglr.config import experiment_from_dict
from sglr.data import build_mnist_loaders
from sglr.evaluation import evaluate_model
from sglr.figures import generate_run_figures, generate_sweep_figure, load_image_archive
from sglr.model import build_mnist_model, count_parameters
from sglr.train import seed_everything, select_device


RECORD_FILENAMES = (
    "evaluation.jsonl",
    "evaluation_records.jsonl",
    "test_predictions.jsonl",
    "predictions.jsonl",
)
SUMMARY_FILENAMES = (
    "evaluation_summary.json",
    "test_summary.json",
    "run_summary.json",
    "training_summary.json",
)
MANIFEST_FILENAMES = ("run_manifest.json", "manifest.json")
IMAGE_FILENAMES = ("evaluation_images.npz", "test_images.npz", "images.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SGLR MNIST checkpoints or regenerate figures from frozen artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Analyze one completed evaluation run.")
    run_parser.add_argument("--run-dir", type=Path, required=True, help="Completed run directory.")
    run_parser.add_argument("--records", type=Path, help="Evaluation JSONL; discovered under run-dir when omitted.")
    run_parser.add_argument("--summary", type=Path, help="Run summary JSON; discovered under run-dir when omitted.")
    run_parser.add_argument("--manifest", type=Path, help="Run manifest JSON; discovered under run-dir when omitted.")
    run_parser.add_argument("--images", type=Path, help="Optional NPZ containing images and sample_indices arrays.")
    run_parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to RUN_DIR/analysis.")
    run_parser.add_argument(
        "--permutations", type=_positive_int, default=1000, help="Label permutations for the MI null."
    )
    run_parser.add_argument("--seed", type=int, default=7, help="Permutation RNG seed.")

    sweep_parser = subparsers.add_parser("sweep", help="Aggregate accuracy and compute across completed runs.")
    sweep_source = sweep_parser.add_mutually_exclusive_group(required=True)
    sweep_source.add_argument("--sweep-root", type=Path, help="Root searched recursively for run manifests.")
    sweep_source.add_argument(
        "--run-dir", type=Path, action="append", help="Completed run directory; repeat for each run."
    )
    sweep_parser.add_argument("--output-dir", type=Path, help="Output directory; defaults under the sweep root.")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint",
        help="Re-evaluate a frozen checkpoint, then regenerate its run figures.",
    )
    checkpoint_parser.add_argument("--run-dir", type=Path, required=True, help="Run containing best_model.pt.")
    checkpoint_parser.add_argument("--device", default="auto", help="Torch device used for evaluation.")
    checkpoint_parser.add_argument("--download", action="store_true", help="Allow missing MNIST files to download.")
    checkpoint_parser.add_argument("--permutations", type=_positive_int, default=1000, help="MI null permutations.")
    checkpoint_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the checkpoint evaluation progress bar.",
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def analyze_run(args: argparse.Namespace) -> list[Path]:
    run_directory = args.run_dir.resolve()
    records_path = args.records or _discover_required(run_directory, RECORD_FILENAMES, "evaluation JSONL")
    summary_path = args.summary or _discover_optional(run_directory, SUMMARY_FILENAMES)
    manifest_path = args.manifest or _discover_optional(run_directory, MANIFEST_FILENAMES)
    images_path = args.images or _discover_optional(run_directory, IMAGE_FILENAMES)
    output_directory = args.output_dir or run_directory / "analysis"

    records = load_evaluation_records(records_path)
    summary = load_json_object(summary_path) if summary_path else {}
    manifest = load_json_object(manifest_path) if manifest_path else {}
    images = load_image_archive(images_path) if images_path else None
    return generate_run_figures(
        records=records,
        output_directory=output_directory,
        summary=summary,
        manifest=manifest,
        images=images,
        permutations=args.permutations,
        seed=args.seed,
    )


def analyze_sweep(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        run_directories = [path.resolve() for path in args.run_dir]
        default_output = Path.cwd() / "sweep_analysis"
    else:
        sweep_root = args.sweep_root.resolve()
        run_directories = _discover_run_directories(sweep_root)
        default_output = sweep_root / "analysis"
    if not run_directories:
        raise ValueError("No completed run manifests were found")

    runs: list[SweepRun] = []
    manifests: list[dict[str, object]] = []
    failures: list[str] = []
    for run_directory in run_directories:
        try:
            if not run_is_complete(run_directory):
                raise ValueError("run does not satisfy the finalized completion contract")
            manifest_path = _discover_required(run_directory, MANIFEST_FILENAMES, "manifest JSON")
            summary_path = _discover_required(run_directory, SUMMARY_FILENAMES, "summary JSON")
            manifest = load_json_object(manifest_path)
            summary = load_json_object(summary_path)
            runs.append(build_sweep_run(run_directory, summary, manifest))
            manifests.append(manifest)
        except (FileNotFoundError, ValueError) as error:
            failures.append(f"{run_directory}: {error}")
    if failures:
        details = "\n".join(failures)
        raise ValueError(f"Sweep inputs are incomplete or invalid:\n{details}")

    output_directory = args.output_dir or default_output
    validation = _validate_sweep_inputs(runs, manifests)
    written = generate_sweep_figure(runs, output_directory)
    save_json(output_directory / "sweep_validation.json", validation)
    return written


def analyze_checkpoint(args: argparse.Namespace) -> list[Path]:
    """Rebuild frozen evaluation artifacts and figures without training."""

    run_directory = args.run_dir.resolve()
    manifest_path = run_directory / "manifest.json"
    manifest = load_json(manifest_path)
    manifest_variant = manifest.get("variant")
    manifest_seed = manifest.get("seed")
    if not isinstance(manifest_variant, str) or not isinstance(manifest_seed, int):
        raise ValueError("Run manifest must record a string variant and integer seed")
    experiment = experiment_from_dict(manifest.get("config"))
    if experiment.model.routing_mode != manifest_variant or experiment.training.seed != manifest_seed:
        raise ValueError("Run manifest variant or seed disagrees with its resolved configuration")
    seed_everything(experiment.training.seed)
    device = select_device(args.device)
    loaders = build_mnist_loaders(experiment.training, download=args.download, device=device)
    model = build_mnist_model(experiment.model).to(device)
    checkpoint = load_checkpoint(run_directory / "best_model.pt", device)
    model.load_state_dict(checkpoint["model_state"])
    summary_path = run_directory / "evaluation_summary.json"
    preserved_summary = load_json(summary_path) if summary_path.is_file() else {}
    (run_directory / "run_complete.json").unlink(missing_ok=True)
    reevaluated_summary = evaluate_model(
        model=model,
        data_loader=loaders.test,
        device=device,
        output_directory=run_directory,
        num_classes=experiment.model.num_classes,
        description="Checkpoint evaluation",
        show_progress=not args.no_progress,
    )
    summary = {**preserved_summary, **reevaluated_summary}
    summary.update(
        {
            "variant": experiment.model.routing_mode,
            "seed": experiment.training.seed,
            "total_parameters": count_parameters(model),
            "trainable_parameters": count_parameters(model, trainable_only=True),
        }
    )
    save_json(summary_path, summary)
    manifest["evaluation"] = summary
    save_json(manifest_path, manifest)
    records = load_evaluation_records(run_directory / "evaluation.jsonl")
    images = load_image_archive(run_directory / "evaluation_images.npz")
    written = generate_run_figures(
        records=records,
        output_directory=run_directory / "analysis",
        summary=summary,
        manifest=manifest,
        images=images,
        permutations=args.permutations,
        seed=experiment.training.seed,
    )
    save_json(
        run_directory / "run_complete.json",
        {
            "schema_version": experiment.schema_version,
            "variant": experiment.model.routing_mode,
            "seed": experiment.training.seed,
            "summary": "evaluation_summary.json",
        },
    )
    return written


def _validate_sweep_inputs(
    runs: Sequence[SweepRun],
    manifests: Sequence[dict[str, object]],
) -> dict[str, object]:
    pairs = [(run.variant, run.seed) for run in runs]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Sweep contains duplicate variant/seed runs")
    if len(runs) != len(manifests):
        raise ValueError("Every sweep run must have one manifest")

    signatures = {_config_signature(manifest) for manifest in manifests}
    if len(signatures) != 1:
        raise ValueError("Sweep run configurations differ beyond routing mode and seed")

    test_digests = {
        manifest.get("data_splits", {}).get("test_index_sha256")
        for manifest in manifests
        if isinstance(manifest.get("data_splits"), dict)
    }
    if len(test_digests) != 1 or None in test_digests:
        raise ValueError("Sweep runs must use the same recorded test split")

    expected_pairs: set[tuple[str, int]] | None = None
    first_config = manifests[0].get("config") if manifests else None
    if isinstance(first_config, dict) and isinstance(first_config.get("sweep"), dict):
        sweep = first_config["sweep"]
        variants = sweep.get("variants")
        seeds = sweep.get("seeds")
        if not isinstance(variants, list) or not isinstance(seeds, list):
            raise ValueError("Manifest sweep definition must contain variant and seed lists")
        expected_pairs = {(str(variant), int(seed)) for variant in variants for seed in seeds}
        if set(pairs) != expected_pairs:
            missing = sorted(expected_pairs - set(pairs))
            unexpected = sorted(set(pairs) - expected_pairs)
            raise ValueError(f"Sweep grid is incomplete: missing={missing}, unexpected={unexpected}")

    return {
        "valid": True,
        "runs": len(runs),
        "variant_seed_pairs": [list(pair) for pair in sorted(pairs)],
        "expected_grid_enforced": expected_pairs is not None,
        "test_index_sha256": next(iter(test_digests)),
    }


def _config_signature(manifest: dict[str, object]) -> str:
    config = copy.deepcopy(manifest.get("config"))
    if not isinstance(config, dict):
        raise ValueError("Sweep manifest is missing its resolved configuration")
    model = config.get("model")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("Sweep manifest has an invalid resolved configuration")
    model.pop("routing_mode", None)
    training.pop("seed", None)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _discover_required(directory: Path, filenames: Sequence[str], description: str) -> Path:
    path = _discover_optional(directory, filenames)
    if path is None:
        names = ", ".join(filenames)
        raise FileNotFoundError(f"Could not find {description} under {directory}; tried {names}")
    return path


def _discover_optional(directory: Path, filenames: Sequence[str]) -> Path | None:
    for filename in filenames:
        direct_path = directory / filename
        if direct_path.is_file():
            return direct_path
    for filename in filenames:
        matches = sorted(directory.rglob(filename))
        if matches:
            return matches[0]
    return None


def _discover_run_directories(sweep_root: Path) -> list[Path]:
    directories: set[Path] = set()
    for manifest_name in MANIFEST_FILENAMES:
        for manifest_path in sweep_root.rglob(manifest_name):
            directories.add(manifest_path.parent)
    return sorted(directories)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        written_paths = analyze_run(args)
    elif args.command == "sweep":
        written_paths = analyze_sweep(args)
    else:
        written_paths = analyze_checkpoint(args)

    print(f"Wrote analysis to {written_paths[0].parent}")
    for path in written_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
