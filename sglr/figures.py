"""Publication-ready figures from frozen SGLR evaluation artifacts."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from sglr.analysis import (
    DIGITS,
    EvaluationRecord,
    RouteStatistics,
    SweepAggregate,
    SweepRun,
    aggregate_sweep,
    compute_route_statistics,
    infer_num_experts,
    resolve_expert_names,
    route_digit_mutual_information,
    write_json,
    write_matrix_csv,
    write_rows_csv,
)


plt.switch_backend("Agg")


FloatArray = NDArray[np.float64]
ImageMap = Mapping[int, NDArray[np.generic]]
FIGURE_DPI = 300
DIGIT_LABELS = [str(digit) for digit in DIGITS]


def generate_run_figures(
    records: Sequence[EvaluationRecord],
    output_directory: str | Path,
    summary: Mapping[str, object] | None = None,
    manifest: Mapping[str, object] | None = None,
    images: ImageMap | None = None,
    permutations: int = 1000,
    seed: int = 7,
) -> list[Path]:
    """Write all route-analysis data and figures for one completed run."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    provisional_expert_names = _artifact_expert_names(manifest, summary)
    num_experts = infer_num_experts(records, provisional_expert_names)
    expert_names = resolve_expert_names(num_experts, manifest, summary)
    statistics = compute_route_statistics(records, num_experts)
    mutual_information = route_digit_mutual_information(records, permutations, seed)
    depth_labels = [str(depth) for depth in range(statistics.exit_depth.shape[1])]

    write_matrix_csv(
        output_path / "digit_expert_visitation.csv",
        statistics.visitation,
        DIGIT_LABELS,
        expert_names,
        "digit",
    )
    write_matrix_csv(
        output_path / "digit_expert_visitation_lift.csv",
        statistics.visitation_lift,
        DIGIT_LABELS,
        expert_names,
        "digit",
    )
    write_matrix_csv(
        output_path / "digit_first_route.csv",
        statistics.first_route,
        DIGIT_LABELS,
        expert_names,
        "digit",
    )
    write_matrix_csv(
        output_path / "digit_exit_depth.csv",
        statistics.exit_depth,
        DIGIT_LABELS,
        depth_labels,
        "digit",
    )
    write_matrix_csv(
        output_path / "expert_transitions.csv",
        statistics.transitions,
        expert_names,
        expert_names,
        "source_expert",
    )
    write_rows_csv(
        output_path / "digit_depth_forced_exit.csv",
        ("digit", "samples", "mean_depth", "forced_exit_rate"),
        _digit_depth_rows(statistics),
    )
    write_json(output_path / "route_digit_mutual_information.json", mutual_information.to_dict())
    write_json(
        output_path / "analysis_summary.json",
        {
            "samples": len(records),
            "num_experts": num_experts,
            "expert_names": expert_names,
            "accuracy": sum(record.correct for record in records) / len(records),
            "mean_route_depth": sum(record.route_depth for record in records) / len(records),
            "forced_exit_rate": sum(record.forced_exit for record in records) / len(records),
            "visitation_definition": "mean number of visits to each expert per sample of a digit",
            "lift_definition": "digit visitation divided by overall visitation",
            "mutual_information": mutual_information.to_dict(),
        },
    )

    figure_paths = [
        output_path / "digit_expert_visitation.png",
        output_path / "digit_first_route_exit_depth.png",
        output_path / "route_depth_forced_exit.png",
        output_path / "expert_transitions.png",
        output_path / "route_digit_mutual_information.png",
    ]
    plot_digit_expert_visitation(statistics, expert_names, figure_paths[0])
    plot_first_route_and_exit_depth(statistics, expert_names, figure_paths[1])
    plot_depth_and_forced_exit(statistics, figure_paths[2])
    plot_expert_transitions(statistics, expert_names, figure_paths[3])
    plot_mutual_information(mutual_information.to_dict(), figure_paths[4])

    if images:
        representative_path = output_path / "representative_routes.png"
        selected = _select_representative_records(records, images)
        if selected:
            write_rows_csv(
                output_path / "representative_routes.csv",
                (
                    "digit",
                    "sample_index",
                    "prediction",
                    "correct",
                    "confidence",
                    "route_ids",
                    "route_depth",
                    "forced_exit",
                ),
                (
                    (
                        digit,
                        record.sample_index,
                        record.prediction,
                        record.correct,
                        record.confidence,
                        "|".join(str(route_id) for route_id in record.route_ids),
                        record.route_depth,
                        record.forced_exit,
                    )
                    for digit, record in sorted(selected.items())
                ),
            )
        if plot_representative_routes(records, images, expert_names, representative_path):
            figure_paths.append(representative_path)

    return figure_paths


def _artifact_expert_names(
    manifest: Mapping[str, object] | None,
    summary: Mapping[str, object] | None,
) -> list[str]:
    for payload in (manifest, summary):
        names = _find_nested(payload, "expert_names") if payload else None
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return list(names)
        specs = _find_nested(payload, "experts") if payload else None
        if isinstance(specs, list):
            spec_names = [spec.get("name") for spec in specs if isinstance(spec, dict)]
            if len(spec_names) == len(specs) and all(isinstance(name, str) for name in spec_names):
                return [name for name in spec_names if isinstance(name, str)]
    return []


def _find_nested(payload: Mapping[str, object] | None, key: str) -> object | None:
    if payload is None:
        return None
    if key in payload:
        return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_nested(value, key)
            if found is not None:
                return found
    return None


def _digit_depth_rows(statistics: RouteStatistics) -> list[tuple[int, int, float, float]]:
    depth_values = np.arange(statistics.depth_counts.shape[1], dtype=np.float64)  # (max_depth + 1,)
    mean_depth = statistics.exit_depth @ depth_values  # (10,)
    return [
        (
            digit,
            int(statistics.sample_counts[digit]),
            float(mean_depth[digit]),
            float(statistics.forced_exit_rate[digit]),
        )
        for digit in DIGITS
    ]


def plot_digit_expert_visitation(
    statistics: RouteStatistics,
    expert_names: Sequence[str],
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(max(11.0, len(expert_names) * 0.45), 8.0))
    _heatmap(
        axes[0],
        statistics.visitation,  # (10, e)
        DIGIT_LABELS,
        expert_names,
        "Mean expert visits per sample",
        "viridis",
        "visits/sample",
    )
    lift_limit = max(2.0, float(np.nanpercentile(statistics.visitation_lift, 98)))
    _heatmap(
        axes[1],
        statistics.visitation_lift,  # (10, e)
        DIGIT_LABELS,
        expert_names,
        "Class-conditional visitation lift",
        "coolwarm",
        "lift",
        vmin=0.0,
        vmax=lift_limit,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_first_route_and_exit_depth(
    statistics: RouteStatistics,
    expert_names: Sequence[str],
    output_path: str | Path,
) -> None:
    width = max(12.0, len(expert_names) * 0.42 + statistics.exit_depth.shape[1] * 0.35)
    figure, axes = plt.subplots(1, 2, figsize=(width, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    _heatmap(
        axes[0],
        statistics.first_route,  # (10, e)
        DIGIT_LABELS,
        expert_names,
        "First expert by digit",
        "magma",
        "probability",
        vmin=0.0,
        vmax=1.0,
    )
    depth_labels = [str(depth) for depth in range(statistics.exit_depth.shape[1])]
    _heatmap(
        axes[1],
        statistics.exit_depth,  # (10, max_depth + 1)
        DIGIT_LABELS,
        depth_labels,
        "Exit depth by digit",
        "Blues",
        "probability",
        vmin=0.0,
        vmax=1.0,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_depth_and_forced_exit(statistics: RouteStatistics, output_path: str | Path) -> None:
    figure, (depth_axis, forced_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    depths = np.arange(statistics.exit_depth.shape[1], dtype=np.int64)  # (max_depth + 1,)
    bottom = np.zeros(10, dtype=np.float64)  # (10,)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(depths)))  # (max_depth + 1, 4)
    for depth, color in zip(depths, colors, strict=True):
        depth_probability = statistics.exit_depth[:, depth]  # (10,)
        depth_axis.bar(DIGITS, depth_probability, bottom=bottom, label=str(depth), color=color)
        bottom = bottom + depth_probability  # (10,)
    depth_axis.set(xlabel="Digit", ylabel="Fraction of samples", title="Route-depth distribution")
    depth_axis.set_xticks(DIGITS)
    depth_axis.set_ylim(0.0, 1.0)
    depth_axis.legend(title="Depth", ncols=min(3, len(depths)), fontsize=8)

    forced_axis.bar(DIGITS, statistics.forced_exit_rate, color="#b4474d")
    forced_axis.set(xlabel="Digit", ylabel="Forced-exit rate", title="Forced exits by digit")
    forced_axis.set_xticks(DIGITS)
    forced_axis.set_ylim(0.0, 1.0)
    _light_grid(depth_axis)
    _light_grid(forced_axis)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_expert_transitions(
    statistics: RouteStatistics,
    expert_names: Sequence[str],
    output_path: str | Path,
) -> None:
    size = max(7.0, len(expert_names) * 0.38)
    figure, axis = plt.subplots(figsize=(size, size))
    _heatmap(
        axis,
        statistics.transitions,  # (e, e)
        expert_names,
        expert_names,
        "Expert transition probabilities",
        "cividis",
        "probability",
        vmin=0.0,
        vmax=1.0,
        y_label="Source expert",
        x_label="Next expert",
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_mutual_information(metrics: Mapping[str, object], output_path: str | Path) -> None:
    observed = float(metrics["observed_nats"])
    shuffled_mean = float(metrics["shuffled_mean_nats"])
    shuffled_sd = float(metrics["shuffled_sd_nats"])
    p_value = float(metrics["permutation_p_value"])
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    bars = np.asarray([observed, shuffled_mean], dtype=np.float64)  # (2,)
    errors = np.asarray([0.0, shuffled_sd], dtype=np.float64)  # (2,)
    axis.bar(["Observed", "Shuffled labels"], bars, yerr=errors, capsize=5, color=["#386cb0", "#999999"])
    axis.set_ylabel("Mutual information (nats)")
    axis.set_title(f"Digit and first-route dependence\npermutation p = {p_value:.4g}")
    _light_grid(axis)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_representative_routes(
    records: Sequence[EvaluationRecord],
    images: ImageMap,
    expert_names: Sequence[str],
    output_path: str | Path,
) -> bool:
    """Plot a median-confidence example per digit, preferring correct predictions."""

    selected = _select_representative_records(records, images)
    if not selected:
        return False

    figure, axes = plt.subplots(2, 5, figsize=(13, 6.2))
    flat_axes = axes.ravel()  # (10,)
    for digit, axis in zip(DIGITS, flat_axes, strict=True):
        record = selected.get(digit)
        if record is None:
            axis.text(0.5, 0.5, "No image", ha="center", va="center")
            axis.set_title(f"Digit {digit}")
            axis.axis("off")
            continue
        image = np.asarray(images[record.sample_index])  # (h, w) or (1, h, w)
        image = np.squeeze(image)  # (h, w)
        if image.ndim != 2:
            raise ValueError(f"Image for sample {record.sample_index} is not two-dimensional after squeeze")
        route = " -> ".join(_short_expert_name(expert_names[route_id]) for route_id in record.route_ids)
        route_title = textwrap.fill(route or "no expert steps", width=28)
        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Digit {digit} -> {record.prediction}, p={record.confidence:.2f}\n{route_title}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Representative correct predictions and expert routes", fontsize=12)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return True


def _select_representative_records(
    records: Sequence[EvaluationRecord],
    images: ImageMap,
) -> dict[int, EvaluationRecord]:
    selected: dict[int, EvaluationRecord] = {}
    for digit in DIGITS:
        digit_records = [
            record for record in records if record.label == digit and record.sample_index in images
        ]
        correct_records = [record for record in digit_records if record.correct]
        candidates = sorted(
            correct_records if correct_records else digit_records,
            key=lambda record: (record.confidence, record.sample_index),
        )
        if candidates:
            selected[digit] = candidates[(len(candidates) - 1) // 2]
    return selected


def _short_expert_name(name: str) -> str:
    return name.replace("attention", "attn").replace("expert_", "e")


def generate_sweep_figure(
    runs: Sequence[SweepRun],
    output_directory: str | Path,
) -> list[Path]:
    if not runs:
        raise ValueError("At least one completed run is required for sweep analysis")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_sweep(runs)

    write_rows_csv(
        output_path / "sweep_runs.csv",
        ("variant", "seed", "accuracy", "mean_expert_compute", "run_directory"),
        (
            (run.variant, run.seed, run.accuracy, run.mean_expert_compute, run.run_directory)
            for run in sorted(runs, key=lambda item: (item.variant, item.seed))
        ),
    )
    write_rows_csv(
        output_path / "sweep_accuracy_vs_compute.csv",
        ("variant", "runs", "accuracy_mean", "accuracy_sd", "compute_mean", "compute_sd"),
        (
            (
                aggregate.variant,
                aggregate.runs,
                aggregate.accuracy_mean,
                aggregate.accuracy_sd,
                aggregate.compute_mean,
                aggregate.compute_sd,
            )
            for aggregate in aggregates
        ),
    )
    write_json(
        output_path / "sweep_accuracy_vs_compute.json",
        {"variants": [_aggregate_to_dict(aggregate) for aggregate in aggregates]},
    )

    figure_path = output_path / "sweep_accuracy_vs_compute.png"
    plot_accuracy_vs_compute(aggregates, figure_path)
    return [figure_path]


def _aggregate_to_dict(aggregate: SweepAggregate) -> dict[str, str | int | float]:
    return {
        "variant": aggregate.variant,
        "runs": aggregate.runs,
        "accuracy_mean": aggregate.accuracy_mean,
        "accuracy_sd": aggregate.accuracy_sd,
        "compute_mean": aggregate.compute_mean,
        "compute_sd": aggregate.compute_sd,
    }


def plot_accuracy_vs_compute(aggregates: Sequence[SweepAggregate], output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(aggregates)))  # (v, 4)
    for aggregate, color in zip(aggregates, colors, strict=True):
        axis.errorbar(
            aggregate.compute_mean,
            aggregate.accuracy_mean,
            xerr=aggregate.compute_sd,
            yerr=aggregate.accuracy_sd,
            fmt="o",
            capsize=4,
            markersize=7,
            color=color,
            label=f"{aggregate.variant} (n={aggregate.runs})",
        )
    axis.set_xlabel("Mean expert compute")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Accuracy versus routed expert compute")
    axis.legend(fontsize=8)
    _light_grid(axis)
    figure.tight_layout()
    _save_figure(figure, output_path)


def load_image_archive(path: str | Path) -> dict[int, NDArray[np.generic]]:
    """Load `images` and optional `sample_indices` arrays from an NPZ archive."""

    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        if "images" not in archive:
            raise ValueError(f"Image archive {archive_path} must contain an 'images' array")
        images = np.asarray(archive["images"])  # (n, h, w) or (n, 1, h, w)
        if images.ndim not in (3, 4):
            raise ValueError("Image archive images must have shape (n,h,w) or (n,1,h,w)")
        if "sample_indices" in archive:
            sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)  # (n,)
        else:
            sample_indices = np.arange(images.shape[0], dtype=np.int64)  # (n,)
        if sample_indices.ndim != 1 or sample_indices.shape[0] != images.shape[0]:
            raise ValueError("sample_indices must be one-dimensional and align with images")
        return {int(index): image for index, image in zip(sample_indices, images, strict=True)}


def _heatmap(
    axis: Axes,
    matrix: FloatArray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    color_map: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    y_label: str = "Digit",
    x_label: str = "Expert",
) -> None:
    # matrix: (r, c)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=color_map, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xticks(range(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)


def _light_grid(axis: Axes) -> None:
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: Figure, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
