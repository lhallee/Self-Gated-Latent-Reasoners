from __future__ import annotations

from pathlib import Path

import numpy as np

from sglr.analysis import EvaluationRecord, SweepRun, compute_route_statistics
from sglr.figures import generate_run_figures, generate_sweep_figure


def sample_records() -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            sample_index=digit * 2 + repeat,
            label=digit,
            prediction=digit,
            confidence=0.9,
            correct=True,
            route_ids=(digit % 3, (digit + repeat + 1) % 3),
            route_depth=2,
            forced_exit=bool(repeat),
        )
        for digit in range(10)
        for repeat in range(2)
    ]


def test_route_statistics_are_normalized_by_digit() -> None:
    statistics = compute_route_statistics(sample_records(), num_experts=3)

    assert statistics.visitation.shape == (10, 3)
    assert statistics.first_route.shape == (10, 3)
    assert statistics.exit_depth.shape == (10, 3)
    assert np.allclose(statistics.first_route.sum(axis=1), 1.0)
    assert np.allclose(statistics.exit_depth.sum(axis=1), 1.0)
    assert np.allclose(statistics.forced_exit_rate, 0.5)


def test_run_figures_render_with_machine_readable_sources(tmp_path: Path) -> None:
    written = generate_run_figures(
        records=sample_records(),
        output_directory=tmp_path,
        manifest={"expert_names": ["mlp", "conv", "attention"]},
        permutations=10,
        seed=7,
    )

    assert len(written) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    assert (tmp_path / "digit_expert_visitation.csv").is_file()
    assert (tmp_path / "route_digit_mutual_information.json").is_file()


def test_sweep_figure_aggregates_seed_variation(tmp_path: Path) -> None:
    runs = [
        SweepRun("straight_through", 7, 0.95, 3.0, "run_a"),
        SweepRun("straight_through", 17, 0.97, 2.8, "run_b"),
        SweepRun("hard_argmax", 7, 0.90, 3.5, "run_c"),
    ]

    written = generate_sweep_figure(runs, tmp_path)

    assert written == [tmp_path / "sweep_accuracy_vs_compute.png"]
    assert written[0].stat().st_size > 0
    assert (tmp_path / "sweep_accuracy_vs_compute.csv").is_file()
