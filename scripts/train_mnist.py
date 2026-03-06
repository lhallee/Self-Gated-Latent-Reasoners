from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from sglr.artifacts import create_run_directory, ensure_directory, save_json
from sglr.config import parse_train_configs
from sglr.model import SGLRModel
from sglr.train import build_mnist_dataloaders, count_parameter_count, save_route_artifacts, seed_everything, select_device, train_model


def main() -> None:
    model_config, training_config = parse_train_configs()
    seed_everything(training_config.seed)
    device = select_device(training_config.device)
    run_dir = create_run_directory(output_root=training_config.output_root, run_name=training_config.run_name)
    base_stage_dir = ensure_directory(run_dir / "base")

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(
        f"Microbatch size: {training_config.batch_size} | "
        f"Grad accum: {training_config.grad_accum_steps} | "
        f"Effective batch size: {training_config.batch_size * training_config.grad_accum_steps}"
    )

    train_loader, eval_loader = build_mnist_dataloaders(
        data_root=training_config.data_root,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        train_subset=training_config.train_subset,
        test_subset=training_config.test_subset,
    )
    model = SGLRModel(model_config).to(device)
    print(f"Model parameters: {count_parameter_count(model):,}")

    train_model(
        model=model,
        model_config=model_config,
        training_config=training_config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        stage_dir=base_stage_dir,
        device=device,
    )
    save_route_artifacts(
        model=model,
        data_loader=eval_loader,
        device=device,
        stage_dir=base_stage_dir,
        max_samples=training_config.plot_max_samples,
    )
    save_json(
        run_dir / "run_manifest.json",
        {
            "run_dir": str(run_dir),
            "base_stage_dir": str(base_stage_dir),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
        },
    )
    print(f"Finished base training. Artifacts are in {Path(run_dir)}")


if __name__ == "__main__":
    main()
