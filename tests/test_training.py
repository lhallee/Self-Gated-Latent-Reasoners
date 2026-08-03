from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from sglr.analysis import load_evaluation_records
from sglr.artifacts import run_is_complete, save_checkpoint, save_json, save_jsonl
from sglr.config import ExpertSpec, ModelConfig
from sglr.evaluation import evaluate_model
from sglr.model import MNISTSGLR
from sglr.train import run_epoch


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
        load_balance_coefficient=0.01,
        compute_penalty_coefficient=0.001,
        optimizer=optimizer,
        grad_accum_steps=1,
    )

    assert metrics.loss > 0.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.mean_route_depth >= 1.0


def test_evaluation_artifacts_round_trip(tmp_path: Path) -> None:
    model = training_model()

    summary = evaluate_model(
        model=model,
        data_loader=sample_loader(),
        device=torch.device("cpu"),
        output_directory=tmp_path,
    )
    records = load_evaluation_records(tmp_path / "evaluation.jsonl")

    assert summary["examples"] == 4
    assert len(records) == 4
    assert (tmp_path / "evaluation_summary.json").is_file()
    assert (tmp_path / "evaluation_images.npz").is_file()
    assert all(record.route_depth == len(record.route_ids) for record in records)


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
