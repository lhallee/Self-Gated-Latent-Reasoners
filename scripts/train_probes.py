from __future__ import annotations

from pathlib import Path

from sglr.artifacts import ensure_directory, load_checkpoint, load_json, save_json
from sglr.config import ModelConfig, parse_probe_config
from sglr.model import SGLRModel
from sglr.probes import ExpertProbeSuite, summarize_probe_performance, train_probes
from sglr.train import build_mnist_dataloaders, seed_everything, select_device


def main() -> None:
    probe_config = parse_probe_config()
    seed_everything(probe_config.seed)
    device = select_device(probe_config.device)
    run_dir = Path(probe_config.base_run_dir)
    base_stage_dir = run_dir / "base"
    probes_stage_dir = ensure_directory(run_dir / "probes")
    model_config = ModelConfig(**load_json(base_stage_dir / "model_config.json"))

    print(f"Using device: {device}")
    print(f"Base run directory: {run_dir}")
    print(
        f"Probe microbatch size: {probe_config.batch_size} | "
        f"Grad accum: {probe_config.grad_accum_steps} | "
        f"Effective batch size: {probe_config.batch_size * probe_config.grad_accum_steps}"
    )

    backbone = SGLRModel(model_config).to(device)
    checkpoint = load_checkpoint(base_stage_dir / "best_model.pt", device=device)
    backbone.load_state_dict(checkpoint["model_state"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    probe_suite = ExpertProbeSuite(
        num_experts=model_config.num_experts,
        input_size=model_config.input_size,
        num_classes=model_config.num_classes,
    ).to(device)
    train_loader, eval_loader = build_mnist_dataloaders(
        data_root=probe_config.data_root,
        batch_size=probe_config.batch_size,
        num_workers=probe_config.num_workers,
        train_subset=probe_config.train_subset,
        test_subset=probe_config.test_subset,
    )

    train_probes(
        backbone=backbone,
        probe_suite=probe_suite,
        model_config=model_config,
        probe_config=probe_config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        stage_dir=probes_stage_dir,
        device=device,
    )

    probe_summary = summarize_probe_performance(
        backbone=backbone,
        probe_suite=probe_suite,
        data_loader=eval_loader,
        device=device,
        expert_names=backbone.expert_names,
    )
    save_json(probes_stage_dir / "per_expert_probe_summary.json", probe_summary)
    print(f"Finished probe training. Artifacts are in {probes_stage_dir}")


if __name__ == "__main__":
    main()
