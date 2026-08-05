"""Verify the frozen MNIST SGLR acceptance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import torch

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from sglr.artifacts import load_checkpoint, load_json, save_json


MINIMUM_ACCURACY = 0.98
MAXIMUM_SECONDS = 180.0
MAXIMUM_FORCED_EXIT_RATE = 0.10
MINIMUM_MEAN_ROUTE_DEPTH = 5.0
MAXIMUM_MEAN_ROUTE_DEPTH = 10.0
MINIMUM_ROUTE_DIGIT_MI_NATS = 0.05
MAXIMUM_ROUTE_DIGIT_P_VALUE = 0.05
MINIMUM_NORMALIZED_UTILIZATION_ENTROPY = 0.95
MINIMUM_UTILIZATION = 0.5 / 24
MAXIMUM_UTILIZATION = 2.0 / 24


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a completed MNIST SGLR run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def verify_run(run_path: Path) -> dict[str, Any]:
    manifest = _mapping(load_json(run_path / "manifest.json"), "manifest")
    summary = _mapping(load_json(run_path / "evaluation_summary.json"), "summary")
    history = _mapping(load_json(run_path / "training_history.json"), "history")
    route_information = _mapping(
        load_json(run_path / "analysis" / "route_digit_mutual_information.json"),
        "route information",
    )
    checkpoint_path = run_path / "best_model.pt"
    checkpoint = _mapping(
        load_checkpoint(checkpoint_path, torch.device("cpu")),
        "checkpoint",
    )

    config = _mapping(manifest.get("config"), "config")
    model_config = _mapping(config.get("model"), "model config")
    experts = model_config.get("experts")
    if not isinstance(experts, list):
        raise ValueError("Model config must contain an expert list")
    expert_families = Counter(
        str(_mapping(expert, "expert").get("family")) for expert in experts
    )

    split_summary = _mapping(manifest.get("data_splits"), "data splits")
    epoch_records = history.get("epochs")
    if not isinstance(epoch_records, list) or not epoch_records:
        raise ValueError("Training history must contain at least one epoch")
    validation_accuracies = [
        float(_mapping(_mapping(epoch, "epoch").get("validation"), "validation")["accuracy"])
        for epoch in epoch_records
    ]
    best_history_index = max(
        range(len(validation_accuracies)),
        key=validation_accuracies.__getitem__,
    )
    best_history_epoch = int(_mapping(epoch_records[best_history_index], "epoch")["epoch"])

    utilization = summary.get("expert_utilization")
    if not isinstance(utilization, list):
        raise ValueError("Evaluation summary must contain expert utilization")
    utilization_values = [float(value) for value in utilization]
    utilization_entropy = float(summary["expert_utilization_entropy"])
    normalized_utilization_entropy = utilization_entropy / math.log(len(utilization_values))

    records = _load_records(run_path / "evaluation.jsonl")
    sample_indices = [int(record["sample_index"]) for record in records]
    route_records_are_valid = all(
        isinstance(record.get("route_ids"), list)
        and len(record["route_ids"]) == int(record["route_depth"])
        and int(record["route_depth"]) >= 1
        for record in records
    )

    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_accuracy = float(checkpoint["validation_accuracy"])
    summary_best_epoch = int(summary["best_epoch"])
    summary_best_accuracy = float(summary["best_validation_accuracy"])
    gates = {
        "data_protocol": (
            int(split_summary.get("train_size", 0)) == 60_000
            and int(split_summary.get("validation_size", 0)) == 5_000
            and int(split_summary.get("test_size", 0)) == 5_000
            and split_summary.get("train_source") == "official_train"
            and split_summary.get("validation_source") == "official_test"
            and split_summary.get("test_source") == "official_test"
        ),
        "validation_selected_checkpoint": (
            checkpoint_epoch == best_history_epoch == summary_best_epoch
            and checkpoint_accuracy == validation_accuracies[best_history_index]
            and checkpoint_accuracy == summary_best_accuracy
        ),
        "test_accuracy": (
            summary.get("evaluation_split") == "test"
            and int(summary.get("examples", 0)) == 5_000
            and float(summary.get("accuracy", 0.0)) > MINIMUM_ACCURACY
        ),
        "expert_composition": (
            len(experts) == 24
            and expert_families == Counter({"mlp": 8, "attention": 8, "conv": 8})
        ),
        "sequential_sparse_records": (
            len(records) == 5_000
            and len(set(sample_indices)) == 5_000
            and route_records_are_valid
        ),
        "runtime": float(summary.get("end_to_end_elapsed_seconds", math.inf)) <= MAXIMUM_SECONDS,
        "even_utilization": (
            len(utilization_values) == 24
            and all(value > 0.0 for value in utilization_values)
            and normalized_utilization_entropy >= MINIMUM_NORMALIZED_UTILIZATION_ENTROPY
            and min(utilization_values) >= MINIMUM_UTILIZATION
            and max(utilization_values) <= MAXIMUM_UTILIZATION
        ),
        "natural_termination": (
            float(summary.get("forced_exit_rate", 1.0)) < MAXIMUM_FORCED_EXIT_RATE
        ),
        "route_depth": (
            MINIMUM_MEAN_ROUTE_DEPTH
            <= float(summary.get("mean_route_depth", 0.0))
            <= MAXIMUM_MEAN_ROUTE_DEPTH
        ),
        "digit_differentiation": (
            float(route_information.get("observed_nats", 0.0)) >= MINIMUM_ROUTE_DIGIT_MI_NATS
            and float(route_information.get("permutation_p_value", 1.0))
            <= MAXIMUM_ROUTE_DIGIT_P_VALUE
            and int(route_information.get("permutations", 0)) >= 1_000
        ),
    }
    return {
        "all_passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "accuracy": float(summary["accuracy"]),
            "best_validation_accuracy": summary_best_accuracy,
            "best_epoch": summary_best_epoch,
            "end_to_end_elapsed_seconds": float(summary["end_to_end_elapsed_seconds"]),
            "forced_exit_rate": float(summary["forced_exit_rate"]),
            "mean_route_depth": float(summary["mean_route_depth"]),
            "normalized_utilization_entropy": normalized_utilization_entropy,
            "minimum_utilization": min(utilization_values),
            "maximum_utilization": max(utilization_values),
            "route_digit_mi_nats": float(route_information["observed_nats"]),
            "route_digit_permutation_p_value": float(
                route_information["permutation_p_value"]
            ),
        },
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Evaluation record {line_number} must be an object")
            records.append(payload)
    return records


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    report = verify_run(args.run_dir)
    save_json(args.run_dir / "acceptance.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
