"""Artifact-only routing analysis for MNIST experiments.

This module deliberately has no PyTorch dependency. Training writes stable JSONL
records, and analysis can therefore be repeated without a checkpoint or dataset.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DIGITS = tuple(range(10))


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One frozen evaluation result and its discrete expert route."""

    sample_index: int
    label: int
    prediction: int
    confidence: float
    correct: bool
    route_ids: tuple[int, ...]
    route_depth: int
    forced_exit: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], line_number: int) -> "EvaluationRecord":
        required = {
            "sample_index",
            "label",
            "prediction",
            "confidence",
            "correct",
            "route_ids",
            "route_depth",
            "forced_exit",
        }
        missing = required.difference(payload)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Evaluation JSONL line {line_number} is missing: {missing_text}")

        route_value = payload["route_ids"]
        if not isinstance(route_value, list) or not all(
            isinstance(route_id, int) and not isinstance(route_id, bool) for route_id in route_value
        ):
            raise ValueError(f"Evaluation JSONL line {line_number} has invalid route_ids")

        record = cls(
            sample_index=_strict_int(payload["sample_index"], "sample_index", line_number),
            label=_strict_int(payload["label"], "label", line_number),
            prediction=_strict_int(payload["prediction"], "prediction", line_number),
            confidence=_strict_float(payload["confidence"], "confidence", line_number),
            correct=_strict_bool(payload["correct"], "correct", line_number),
            route_ids=tuple(route_value),
            route_depth=_strict_int(payload["route_depth"], "route_depth", line_number),
            forced_exit=_strict_bool(payload["forced_exit"], "forced_exit", line_number),
        )
        record.validate(line_number)
        return record

    def validate(self, line_number: int = 0) -> None:
        location = f" on line {line_number}" if line_number else ""
        if self.sample_index < 0:
            raise ValueError(f"sample_index must be non-negative{location}")
        if self.label not in DIGITS or self.prediction not in DIGITS:
            raise ValueError(f"MNIST labels and predictions must be in [0, 9]{location}")
        if self.correct != (self.label == self.prediction):
            raise ValueError(f"correct must match the label/prediction comparison{location}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]{location}")
        if not self.route_ids:
            raise ValueError(f"route_ids must contain the required first expert step{location}")
        if self.route_depth != len(self.route_ids):
            raise ValueError(f"route_depth must equal len(route_ids){location}")
        if any(route_id < 0 for route_id in self.route_ids):
            raise ValueError(f"route_ids must be non-negative{location}")


@dataclass(frozen=True, slots=True)
class RouteStatistics:
    """Class-conditioned routing summaries.

    Matrix rows are MNIST digits. Expert matrices use expert IDs as columns;
    depth matrices use route depth as columns.
    """

    sample_counts: IntArray  # (10,)
    visitation: FloatArray  # (10, e), mean visits per sample
    visitation_lift: FloatArray  # (10, e), class rate / overall rate
    first_route: FloatArray  # (10, e), row-normalized probability
    exit_depth: FloatArray  # (10, max_depth + 1), row-normalized probability
    depth_counts: IntArray  # (10, max_depth + 1)
    forced_exit_rate: FloatArray  # (10,)
    transitions: FloatArray  # (e, e), row-normalized probability
    transition_counts: IntArray  # (e, e)


@dataclass(frozen=True, slots=True)
class MutualInformationResult:
    observed_nats: float
    shuffled_mean_nats: float
    shuffled_sd_nats: float
    permutation_p_value: float
    permutations: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": "mutual information between digit and first expert route",
            "observed_nats": self.observed_nats,
            "shuffled_mean_nats": self.shuffled_mean_nats,
            "shuffled_sd_nats": self.shuffled_sd_nats,
            "permutation_p_value": self.permutation_p_value,
            "permutations": self.permutations,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SweepRun:
    variant: str
    seed: int
    accuracy: float
    mean_expert_compute: float
    run_directory: str


@dataclass(frozen=True, slots=True)
class SweepAggregate:
    variant: str
    runs: int
    accuracy_mean: float
    accuracy_sd: float
    compute_mean: float
    compute_sd: float


def _strict_int(value: object, field: str, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Evaluation JSONL line {line_number} has non-integer {field}")
    return value


def _strict_float(value: object, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evaluation JSONL line {line_number} has non-numeric {field}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Evaluation JSONL line {line_number} has non-finite {field}")
    return parsed


def _strict_bool(value: object, field: str, line_number: int) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Evaluation JSONL line {line_number} has non-boolean {field}")
    return value


def load_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    """Load and validate frozen per-example evaluation records."""

    records_path = Path(path)
    records: list[EvaluationRecord] = []
    seen_indices: set[int] = set()

    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Evaluation JSONL line {line_number} must contain an object")

            record = EvaluationRecord.from_mapping(payload, line_number)
            if record.sample_index in seen_indices:
                raise ValueError(f"Duplicate sample_index {record.sample_index} on line {line_number}")
            seen_indices.add(record.sample_index)
            records.append(record)

    if not records:
        raise ValueError(f"No evaluation records found in {records_path}")
    return records


def load_json_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def infer_num_experts(records: Sequence[EvaluationRecord], expert_names: Sequence[str] = ()) -> int:
    routed_ids = [route_id for record in records for route_id in record.route_ids]
    inferred = max(routed_ids, default=-1) + 1
    num_experts = max(inferred, len(expert_names))
    if num_experts == 0:
        raise ValueError("Cannot infer expert count from records with no routes")
    return num_experts


def compute_route_statistics(records: Sequence[EvaluationRecord], num_experts: int) -> RouteStatistics:
    """Compute class-conditioned routing matrices from frozen routes."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    max_depth = max(record.route_depth for record in records)

    sample_counts = np.zeros(10, dtype=np.int64)  # (10,)
    visit_counts = np.zeros((10, num_experts), dtype=np.int64)  # (10, e)
    first_counts = np.zeros((10, num_experts), dtype=np.int64)  # (10, e)
    depth_counts = np.zeros((10, max_depth + 1), dtype=np.int64)  # (10, max_depth + 1)
    forced_counts = np.zeros(10, dtype=np.int64)  # (10,)
    transition_counts = np.zeros((num_experts, num_experts), dtype=np.int64)  # (e, e)

    for record in records:
        digit = record.label
        sample_counts[digit] += 1
        depth_counts[digit, record.route_depth] += 1
        forced_counts[digit] += int(record.forced_exit)

        for route_id in record.route_ids:
            if route_id >= num_experts:
                raise ValueError(f"Route ID {route_id} exceeds configured expert count {num_experts}")
            visit_counts[digit, route_id] += 1

        if record.route_ids:
            first_counts[digit, record.route_ids[0]] += 1
        for source, target in zip(record.route_ids, record.route_ids[1:], strict=False):
            transition_counts[source, target] += 1

    visitation = _divide_rows(visit_counts, sample_counts)  # (10, e)
    overall_visitation = visit_counts.sum(axis=0) / sample_counts.sum()  # (e,)
    visitation_lift = np.divide(  # (10, e)
        visitation,
        overall_visitation[None, :],
        out=np.zeros_like(visitation),
        where=overall_visitation[None, :] > 0,
    )
    first_route = _normalize_rows(first_counts)  # (10, e)
    exit_depth = _normalize_rows(depth_counts)  # (10, max_depth + 1)
    forced_exit_rate = np.divide(  # (10,)
        forced_counts,
        sample_counts,
        out=np.zeros(10, dtype=np.float64),
        where=sample_counts > 0,
    )
    transitions = _normalize_rows(transition_counts)  # (e, e)

    return RouteStatistics(
        sample_counts=sample_counts,
        visitation=visitation,
        visitation_lift=visitation_lift,
        first_route=first_route,
        exit_depth=exit_depth,
        depth_counts=depth_counts,
        forced_exit_rate=forced_exit_rate,
        transitions=transitions,
        transition_counts=transition_counts,
    )


def _divide_rows(counts: IntArray, denominators: IntArray) -> FloatArray:
    # counts: (r, c); denominators: (r,)
    return np.divide(
        counts,
        denominators[:, None],
        out=np.zeros(counts.shape, dtype=np.float64),
        where=denominators[:, None] > 0,
    )  # (r, c)


def _normalize_rows(counts: IntArray) -> FloatArray:
    # counts: (r, c)
    row_totals = counts.sum(axis=1)  # (r,)
    return _divide_rows(counts, row_totals)  # (r, c)


def discrete_mutual_information(left: IntArray, right: IntArray) -> float:
    """Return empirical discrete mutual information in natural-log units."""

    # left: (n,); right: (n,)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Mutual-information inputs must be same-length one-dimensional arrays")
    if left.size == 0:
        raise ValueError("Mutual information requires at least one observation")

    _, left_inverse = np.unique(left, return_inverse=True)  # (n,)
    _, right_inverse = np.unique(right, return_inverse=True)  # (n,)
    joint_counts = np.zeros(  # (n_left, n_right)
        (int(left_inverse.max()) + 1, int(right_inverse.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(joint_counts, (left_inverse, right_inverse), 1)
    joint = joint_counts / left.size  # (n_left, n_right)
    left_probability = joint.sum(axis=1, keepdims=True)  # (n_left, 1)
    right_probability = joint.sum(axis=0, keepdims=True)  # (1, n_right)
    independent = left_probability * right_probability  # (n_left, n_right)
    populated = joint > 0  # (n_left, n_right)
    terms = joint[populated] * np.log(joint[populated] / independent[populated])  # (n_populated,)
    return float(terms.sum())


def route_digit_mutual_information(
    records: Sequence[EvaluationRecord],
    permutations: int = 1000,
    seed: int = 7,
) -> MutualInformationResult:
    """Compare digit/first-route MI with a shuffled-label null distribution."""

    if permutations <= 0:
        raise ValueError("permutations must be positive")
    routed_records = [record for record in records if record.route_ids]
    if not routed_records:
        raise ValueError("Mutual information requires at least one non-empty route")

    labels = np.asarray([record.label for record in routed_records], dtype=np.int64)  # (n,)
    first_routes = np.asarray([record.route_ids[0] for record in routed_records], dtype=np.int64)  # (n,)
    observed = discrete_mutual_information(labels, first_routes)
    generator = np.random.default_rng(seed)
    shuffled = np.empty(permutations, dtype=np.float64)  # (p,)
    for permutation_index in range(permutations):
        shuffled_labels = generator.permutation(labels)  # (n,)
        shuffled[permutation_index] = discrete_mutual_information(shuffled_labels, first_routes)

    p_value = (1.0 + float(np.count_nonzero(shuffled >= observed))) / (permutations + 1.0)
    return MutualInformationResult(
        observed_nats=observed,
        shuffled_mean_nats=float(shuffled.mean()),
        shuffled_sd_nats=float(shuffled.std(ddof=1)) if permutations > 1 else 0.0,
        permutation_p_value=p_value,
        permutations=permutations,
        seed=seed,
    )


def resolve_expert_names(
    num_experts: int,
    manifest: Mapping[str, object] | None = None,
    summary: Mapping[str, object] | None = None,
) -> list[str]:
    """Read expert names from artifacts, falling back to stable numeric labels."""

    for payload in (manifest, summary):
        names = _find_value(payload, ("expert_names",)) if payload else None
        if isinstance(names, list) and len(names) == num_experts and all(isinstance(name, str) for name in names):
            return list(names)
        specs = _find_value(payload, ("experts",)) if payload else None
        if isinstance(specs, list) and len(specs) == num_experts:
            spec_names = [spec.get("name") for spec in specs if isinstance(spec, dict)]
            if len(spec_names) == num_experts and all(isinstance(name, str) for name in spec_names):
                return [name for name in spec_names if isinstance(name, str)]
    return [f"expert_{index}" for index in range(num_experts)]


def build_sweep_run(
    run_directory: str | Path,
    summary: Mapping[str, object],
    manifest: Mapping[str, object],
) -> SweepRun:
    """Extract one sweep row from nested run summary and manifest JSON."""

    accuracy = _required_number(
        (summary, manifest),
        ("test_accuracy", "accuracy", "eval_accuracy", "best_eval_accuracy"),
        "accuracy",
    )
    mean_compute = _required_number(
        (summary, manifest),
        ("mean_expert_compute", "expert_compute", "mean_compute", "analytical_expert_compute"),
        "mean expert compute",
    )
    variant_value = _first_value((manifest, summary), ("variant", "experiment_variant", "run_name"))
    seed_value = _first_value((manifest, summary), ("seed",))
    if not isinstance(variant_value, str):
        raise ValueError(f"Could not find a string variant in artifacts for {run_directory}")
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ValueError(f"Could not find an integer seed in artifacts for {run_directory}")
    return SweepRun(variant_value, seed_value, accuracy, mean_compute, str(run_directory))


def aggregate_sweep(runs: Sequence[SweepRun]) -> list[SweepAggregate]:
    grouped: dict[str, list[SweepRun]] = {}
    for run in runs:
        grouped.setdefault(run.variant, []).append(run)

    aggregates: list[SweepAggregate] = []
    for variant, variant_runs in sorted(grouped.items()):
        accuracies = np.asarray([run.accuracy for run in variant_runs], dtype=np.float64)  # (n_seeds,)
        compute = np.asarray([run.mean_expert_compute for run in variant_runs], dtype=np.float64)  # (n_seeds,)
        aggregates.append(
            SweepAggregate(
                variant=variant,
                runs=len(variant_runs),
                accuracy_mean=float(accuracies.mean()),
                accuracy_sd=float(accuracies.std(ddof=1)) if len(variant_runs) > 1 else 0.0,
                compute_mean=float(compute.mean()),
                compute_sd=float(compute.std(ddof=1)) if len(variant_runs) > 1 else 0.0,
            )
        )
    return aggregates


def _required_number(
    payloads: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    description: str,
) -> float:
    value = _first_value(payloads, keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Could not find numeric {description}; tried keys: {', '.join(keys)}")
    return float(value)


def _first_value(payloads: Sequence[Mapping[str, object]], keys: Sequence[str]) -> object | None:
    for payload in payloads:
        value = _find_value(payload, keys)
        if value is not None:
            return value
    return None


def _find_value(payload: Mapping[str, object] | None, keys: Sequence[str]) -> object | None:
    if payload is None:
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_value(value, keys)
            if found is not None:
                return found
    return None


def write_matrix_csv(
    path: str | Path,
    matrix: NDArray[np.generic],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    row_heading: str = "row",
) -> None:
    # matrix: (r, c)
    if matrix.shape != (len(row_labels), len(column_labels)):
        raise ValueError("CSV labels do not match matrix shape")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_heading, *column_labels])
        for row_label, row in zip(row_labels, matrix, strict=True):  # row: (c,)
            writer.writerow([row_label, *row.tolist()])


def write_rows_csv(path: str | Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
