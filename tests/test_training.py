from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sglr.analysis import load_evaluation_records
from sglr.artifacts import (
    require_manifest_for_resume,
    run_is_complete,
    save_checkpoint,
    save_json,
    save_jsonl,
)
from sglr.config import ExperimentConfig, ExpertSpec, ModelConfig, TrainingConfig
from sglr.data import MNISTDataLoaders
from sglr.evaluation import evaluate_model
from sglr.model import MNISTSGLR
from sglr.train import run_epoch, train_model


def training_model() -> MNISTSGLR:
    config = ModelConfig(
        hidden_size=16,
        max_steps=2,
        min_steps=1,
        router_hidden_size=8,
        experts=(
            ExpertSpec("mlp", "mlp", hidden_size=8),
            ExpertSpec("attention", "attention", internal_size=8, num_heads=1),
        ),
    )
    return MNISTSGLR(config)


def sample_loader() -> DataLoader:
    torch.manual_seed(5)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    indices = torch.arange(4)
    return DataLoader(TensorDataset(images, labels, indices), batch_size=4)


def test_single_batch_training_with_accumulation_one() -> None:
    model = training_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_epoch(
        model=model,
        data_loader=sample_loader(),
        device=torch.device("cpu"),
        variant="straight_through",
        expert_families=tuple(spec.family for spec in model.config.experts),
        load_balance_coefficient=0.01,
        within_family_balance_weight=1.0,
        route_mi_coefficient=0.01,
        compute_penalty_coefficient=0.001,
        optimizer=optimizer,
        grad_accum_steps=1,
    )

    assert metrics.loss > 0.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.mean_route_depth >= 1.0
    assert metrics.load_balance >= 0.0
    assert metrics.route_mutual_information >= 0.0


def test_training_progress_reports_live_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    model = training_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    run_epoch(
        model=model,
        data_loader=sample_loader(),
        device=torch.device("cpu"),
        variant="straight_through",
        expert_families=tuple(spec.family for spec in model.config.experts),
        load_balance_coefficient=0.01,
        within_family_balance_weight=1.0,
        route_mi_coefficient=0.01,
        compute_penalty_coefficient=0.001,
        optimizer=optimizer,
        description="Train 1/1",
        show_progress=True,
    )

    progress_output = capsys.readouterr().err
    assert "Train 1/1" in progress_output
    assert "loss=" in progress_output
    assert "acc=" in progress_output
    assert "depth=" in progress_output
    assert "balance=" in progress_output
    assert "route_mi=" in progress_output
    assert "lr=" in progress_output


def test_training_workflow_reports_epochs_and_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = training_model()
    loader = sample_loader()
    loaders = MNISTDataLoaders(
        train=loader,
        validation=loader,
        test=loader,
        train_indices=(0, 1, 2, 3),
        validation_indices=(0, 1, 2, 3),
        test_indices=(0, 1, 2, 3),
        train_generator=torch.Generator().manual_seed(7),
    )
    experiment = ExperimentConfig(
        model=model.config,
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            patience=1,
            train_size=4,
            validation_size=4,
            test_size=4,
            log_interval=1,
        ),
    )

    result = train_model(
        model=model,
        experiment=experiment,
        loaders=loaders,
        device=torch.device("cpu"),
        run_path=tmp_path,
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert result.best_epoch == 1
    assert "Training epochs" in captured.err
    assert "Train 1/1" in captured.err
    assert "Validate 1/1" in captured.err
    assert "Epoch 1/1" in captured.out
    assert "[new best]" in captured.out


def test_evaluation_artifacts_round_trip(tmp_path: Path) -> None:
    model = training_model()
    loader = sample_loader()
    images, _, _ = next(iter(loader))
    with torch.inference_mode():
        expected = model.eval()(images)

    summary = evaluate_model(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        output_directory=tmp_path,
    )
    records = load_evaluation_records(tmp_path / "evaluation.jsonl")

    assert summary["examples"] == 4
    assert len(records) == 4
    assert (tmp_path / "evaluation_summary.json").is_file()
    assert (tmp_path / "evaluation_images.npz").is_file()
    assert all(record.route_depth == len(record.route_ids) for record in records)
    expected_routes = []
    for batch_position in range(images.size(0)):
        active = expected.trace.active_mask[: expected.trace.executed_steps, batch_position]  # (s,)
        routes = expected.trace.route_ids[: expected.trace.executed_steps, batch_position][active]  # (s_active,)
        expected_routes.append(tuple(routes[routes.lt(expected.trace.num_experts)].tolist()))
    assert [record.route_ids for record in records] == expected_routes


def test_evaluation_progress_reports_live_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evaluate_model(
        model=training_model(),
        data_loader=sample_loader(),
        device=torch.device("cpu"),
        output_directory=tmp_path,
        description="Test evaluation",
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert "Test evaluation" in captured.err
    assert "acc=" in captured.err
    assert "nll=" in captured.err
    assert "depth=" in captured.err
    assert "ex/s=" in captured.err
    assert "Test evaluation complete" in captured.out


def test_same_seed_cpu_models_have_identical_routes() -> None:
    images = torch.randn(3, 1, 28, 28)
    torch.manual_seed(17)
    first = training_model().eval()
    torch.manual_seed(17)
    second = training_model().eval()

    with torch.inference_mode():
        first_output = first(images)
        second_output = second(images)

    assert torch.equal(first_output.trace.route_ids, second_output.trace.route_ids)
    assert torch.equal(first_output.trace.route_depth, second_output.trace.route_depth)
    assert torch.allclose(first_output.logits, second_output.logits)


def test_completion_marker_is_required(tmp_path: Path) -> None:
    save_json(tmp_path / "manifest.json", {"schema_version": 2})
    save_json(tmp_path / "evaluation_summary.json", {"accuracy": 0.5})
    save_jsonl(tmp_path / "evaluation.jsonl", [{"sample_index": 0}])
    (tmp_path / "evaluation_images.npz").write_bytes(b"npz")
    save_checkpoint(tmp_path / "best_model.pt", {"model_state": {}})
    save_checkpoint(tmp_path / "last_state.pt", {"model_state": {}})
    save_json(tmp_path / "training_history.json", {"epochs": []})

    assert not run_is_complete(tmp_path)
    save_json(
        tmp_path / "run_complete.json",
        {
            "schema_version": 2,
            "variant": None,
            "seed": None,
            "summary": "evaluation_summary.json",
        },
    )
    assert run_is_complete(tmp_path)


def test_resume_requires_manifest_in_nonempty_run_directory(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "last_state.pt", {"model_state": {}})

    with pytest.raises(FileNotFoundError, match="Refusing to resume"):
        require_manifest_for_resume(tmp_path)
