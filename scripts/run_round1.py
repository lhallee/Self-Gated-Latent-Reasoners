"""Validation-only depth tuning followed by one sealed MNIST test evaluation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from sglr.analysis import load_evaluation_records
from sglr.artifacts import (
    build_run_manifest,
    ensure_directory,
    load_checkpoint,
    load_json,
    save_json,
)
from sglr.config import ExperimentConfig, load_experiment_config
from sglr.data import build_mnist_loaders
from sglr.evaluation import evaluate_model
from sglr.figures import generate_run_figures, load_image_archive
from sglr.model import build_mnist_model, count_parameters
from sglr.train import seed_everything, select_device, train_model


@dataclass(frozen=True, slots=True)
class RoundOneCandidate:
    name: str
    max_steps: int
    min_steps: int
    learning_rate: float
    load_balance_coefficient: float
    compute_penalty_coefficient: float


ROUND_ONE_CANDIDATES = (
    RoundOneCandidate("depth05_penalty1e3", 5, 1, 1e-3, 0.01, 1e-3),
    RoundOneCandidate("depth08_penalty1e3", 8, 1, 1e-3, 0.01, 1e-3),
    RoundOneCandidate("depth12_penalty1e3", 12, 1, 1e-3, 0.01, 1e-3),
    RoundOneCandidate("depth12_penalty3e4", 12, 1, 1e-3, 0.01, 3e-4),
    RoundOneCandidate("depth12_min02_penalty3e4", 12, 2, 1e-3, 0.01, 3e-4),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune recurrent depth on validation, then evaluate the winner on test once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mnist/pilot.toml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/round1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base_experiment = load_experiment_config(args.config)
    output_root = ensure_directory(args.output_root)
    device = select_device(args.device)
    candidate_results: list[dict[str, object]] = []

    for candidate in ROUND_ONE_CANDIDATES:
        experiment = _candidate_experiment(base_experiment, candidate, args.epochs, args.patience)
        candidate_path = ensure_directory(output_root / candidate.name / f"seed_{experiment.training.seed}")
        result_path = candidate_path / "validation_result.json"
        if result_path.is_file():
            print(f"Using completed validation candidate: {candidate.name}")
            candidate_results.append(load_json(result_path))
            continue

        print(f"Training validation candidate: {candidate.name}")
        seed_everything(experiment.training.seed)
        loaders = build_mnist_loaders(experiment.training, download=args.download)
        model = build_mnist_model(experiment.model).to(device)
        manifest = build_run_manifest(
            experiment=experiment,
            variant=experiment.model.routing_mode,
            seed=experiment.training.seed,
            device=device,
            run_path=candidate_path,
            command=[sys.executable, "-m", "scripts.run_round1", *sys.argv[1:]],
        )
        manifest["round_one_candidate"] = asdict(candidate)
        manifest["test_set_accessed"] = False
        manifest["total_parameters"] = count_parameters(model)
        save_json(candidate_path / "validation_manifest.json", manifest)

        training_result = train_model(model, experiment, loaders, device, candidate_path)
        validation_path = ensure_directory(candidate_path / "validation")
        validation_summary = evaluate_model(
            model=model,
            data_loader=loaders.validation,
            device=device,
            output_directory=validation_path,
            num_classes=experiment.model.num_classes,
        )
        result = {
            "candidate": asdict(candidate),
            "seed": experiment.training.seed,
            "best_epoch": training_result.best_epoch,
            "best_validation_accuracy": training_result.best_validation_accuracy,
            "validation_accuracy": validation_summary["accuracy"],
            "validation_nll": validation_summary["nll"],
            "mean_route_depth": validation_summary["mean_route_depth"],
            "forced_exit_rate": validation_summary["forced_exit_rate"],
            "route_entropy": validation_summary["route_entropy"],
            "training_elapsed_seconds": training_result.elapsed_seconds,
            "test_set_accessed": False,
            "path": str(candidate_path.resolve()),
        }
        save_json(result_path, result)
        candidate_results.append(result)

    selected_result = _select_candidate(candidate_results)
    selected_candidate = _candidate_from_result(selected_result)
    print(
        f"Selected {selected_candidate.name} at validation accuracy "
        f"{float(selected_result['validation_accuracy']):.4f}"
    )
    selection_path = output_root / "selection.json"
    selected_test_path = ensure_directory(output_root / "selected_test")
    if (selected_test_path / "run_complete.json").is_file():
        previous_selection = load_json(selection_path)
        if previous_selection.get("selected_candidate") != selected_candidate.name:
            raise RuntimeError("The validation winner changed after the sealed test was evaluated")
        print("Sealed test evaluation already exists; leaving it unchanged")
        return

    selected_experiment = _candidate_experiment(
        base_experiment,
        selected_candidate,
        args.epochs,
        args.patience,
    )
    seed_everything(selected_experiment.training.seed)
    loaders = build_mnist_loaders(selected_experiment.training, download=args.download)
    model = build_mnist_model(selected_experiment.model).to(device)
    checkpoint = load_checkpoint(Path(str(selected_result["path"])) / "best_model.pt", device)
    model.load_state_dict(checkpoint["model_state"])
    test_summary = evaluate_model(
        model=model,
        data_loader=loaders.test,
        device=device,
        output_directory=selected_test_path,
        num_classes=selected_experiment.model.num_classes,
    )
    test_summary.update(
        {
            "selected_candidate": selected_candidate.name,
            "selection_validation_accuracy": selected_result["validation_accuracy"],
            "seed": selected_experiment.training.seed,
            "total_parameters": count_parameters(model),
        }
    )
    save_json(selected_test_path / "evaluation_summary.json", test_summary)
    _write_selected_figures(selected_test_path, model.expert_names)
    selection = {
        "selection_rule": (
            "highest validation accuracy; within 0.10 percentage points prefer lower validation NLL, "
            "forced-exit rate, mean depth, then training time"
        ),
        "selected_candidate": selected_candidate.name,
        "selected_validation_accuracy": selected_result["validation_accuracy"],
        "sealed_test_accuracy": test_summary["accuracy"],
        "candidates": candidate_results,
    }
    save_json(selection_path, selection)
    save_json(
        selected_test_path / "run_complete.json",
        {
            "selected_candidate": selected_candidate.name,
            "seed": selected_experiment.training.seed,
            "summary": "evaluation_summary.json",
        },
    )
    print(f"Sealed test accuracy: {float(test_summary['accuracy']):.4f}")


def _candidate_experiment(
    base: ExperimentConfig,
    candidate: RoundOneCandidate,
    epochs: int,
    patience: int,
) -> ExperimentConfig:
    experiment = replace(
        base,
        experiment_name="round1",
        model=replace(
            base.model,
            max_steps=candidate.max_steps,
            min_steps=candidate.min_steps,
        ),
        training=replace(
            base.training,
            epochs=epochs,
            patience=patience,
            learning_rate=candidate.learning_rate,
            load_balance_coefficient=candidate.load_balance_coefficient,
            compute_penalty_coefficient=candidate.compute_penalty_coefficient,
        ),
        sweep=None,
    )
    experiment.validate()
    return experiment


def _select_candidate(results: list[dict[str, object]]) -> dict[str, object]:
    best_accuracy = max(float(result["validation_accuracy"]) for result in results)
    effectively_tied = [
        result for result in results if float(result["validation_accuracy"]) >= best_accuracy - 0.001
    ]
    return min(
        effectively_tied,
        key=lambda result: (
            float(result["validation_nll"]),
            float(result["forced_exit_rate"]),
            float(result["mean_route_depth"]),
            float(result["training_elapsed_seconds"]),
        ),
    )


def _candidate_from_result(result: dict[str, object]) -> RoundOneCandidate:
    candidate = result.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Validation result is missing its candidate configuration")
    return RoundOneCandidate(**candidate)


def _write_selected_figures(output_path: Path, expert_names: list[str]) -> None:
    records = load_evaluation_records(output_path / "evaluation.jsonl")
    images = load_image_archive(output_path / "evaluation_images.npz")
    generate_run_figures(
        records=records,
        output_directory=output_path / "analysis",
        summary=load_json(output_path / "evaluation_summary.json"),
        manifest={"expert_names": expert_names},
        images=images,
        permutations=1000,
        seed=7,
    )


if __name__ == "__main__":
    main()
