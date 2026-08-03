from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_mnist import build_parser as build_analysis_parser
from scripts.run_mnist_sweep import build_parser as build_sweep_parser
from scripts.run_mnist_sweep import main as run_sweep
from scripts.run_round1 import _sealed_test_already_finished
from scripts.run_round1 import build_parser as build_round_one_parser
from scripts.train_mnist import build_parser as build_train_parser
from sglr.artifacts import save_json
from sglr.config import with_run_overrides
from sglr.presets.mnist import get_mnist_preset


def test_train_parser_accepts_python_preset_and_expert_override() -> None:
    args = build_train_parser().parse_args(
        [
            "--preset",
            "pilot",
            "--experts-per-family",
            "16",
            "--variant",
            "hard_argmax",
            "--seed",
            "17",
            "--device",
            "cpu",
        ]
    )
    experiment = with_run_overrides(
        get_mnist_preset(args.preset, experts_per_family=args.experts_per_family),
        routing_mode=args.variant,
        seed=args.seed,
    )

    assert experiment.experiment_name == "pilot_48_experts"
    assert experiment.model.routing_mode == "hard_argmax"
    assert experiment.training.seed == 17
    assert args.device == "cpu"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_expert_override_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit):
        build_train_parser().parse_args(["--preset", "pilot", "--experts-per-family", value])


def test_unknown_preset_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit):
        build_train_parser().parse_args(["--preset", "unknown"])


def test_non_sweep_preset_is_rejected_before_training() -> None:
    build_sweep_parser().parse_args(["--preset", "pilot"])
    with pytest.raises(ValueError, match="does not define a sweep"):
        run_sweep(["--preset", "pilot"])


def test_round_one_defaults_preserve_paused_pilot_paths() -> None:
    args = build_round_one_parser().parse_args([])

    assert args.preset == "pilot"
    assert args.experts_per_family is None
    assert args.output_root is None


def test_checkpoint_analysis_uses_only_the_run_manifest() -> None:
    args = build_analysis_parser().parse_args(["checkpoint", "--run-dir", "runs/example"])

    assert args.command == "checkpoint"
    assert args.run_dir == Path("runs/example")
    assert not hasattr(args, "config")


def test_started_sealed_test_cannot_be_evaluated_again(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    save_json(
        selection_path,
        {
            "selected_candidate": "depth12_penalty3e4",
            "status": "test_started",
            "test_set_accessed": True,
        },
    )

    with pytest.raises(RuntimeError, match="Refusing to evaluate"):
        _sealed_test_already_finished(
            selection_path,
            tmp_path / "run_complete.json",
            "depth12_penalty3e4",
        )
