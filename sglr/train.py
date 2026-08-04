"""Training orchestration for reproducible SGLR MNIST experiments."""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sglr.artifacts import (
    build_run_manifest,
    ensure_directory,
    load_checkpoint,
    require_manifest_for_resume,
    run_directory,
    save_checkpoint,
    save_json,
    validate_run_config,
)
from sglr.config import ExperimentConfig
from sglr.data import MNISTDataLoaders, build_mnist_loaders
from sglr.evaluation import evaluate_model
from sglr.model import MNISTOutput, build_mnist_model, count_parameters
from sglr.router import compute_penalty, load_balancing_loss


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    cross_entropy: float
    load_balance: float
    compute_penalty: float
    accuracy: float
    mean_route_depth: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    history: list[dict[str, object]]
    best_epoch: int
    best_validation_accuracy: float
    elapsed_seconds: float
    throughput_examples_per_second: float
    peak_cuda_memory_bytes: int


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_fraction)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)


def _auxiliary_losses(
    output: MNISTOutput,
    variant: str,
) -> tuple[Tensor, Tensor]:
    zero = output.logits.sum() * 0.0  # ()
    if variant in {"fixed_depth", "frozen_random"}:
        return zero, zero
    return load_balancing_loss(output.trace), compute_penalty(output.trace)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    variant: str,
    load_balance_coefficient: float,
    compute_penalty_coefficient: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_accum_steps: int = 1,
    log_interval: int = 0,
    description: str = "Epoch",
    show_progress: bool = False,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_examples = 0
    metric_totals = torch.zeros(6, device=device, dtype=torch.float32)  # (6,)
    total_batches = len(data_loader)

    batches = tqdm(
        data_loader,
        desc=description,
        total=total_batches,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch_index, (images, labels, _) in enumerate(batches):
        # images: (b, 1, 28, 28); labels: (b,)
        images = images.to(device, non_blocking=True)  # (b, 1, 28, 28)
        labels = labels.to(device, non_blocking=True)  # (b,)
        with torch.set_grad_enabled(training):
            output = model(images)
            if not isinstance(output, MNISTOutput):
                raise TypeError("MNIST training expects the model to return MNISTOutput")
            cross_entropy = F.cross_entropy(output.logits, labels)  # ()
            balance_loss, depth_loss = _auxiliary_losses(output, variant)
            loss = (
                cross_entropy
                + load_balance_coefficient * balance_loss
                + compute_penalty_coefficient * depth_loss
            )  # ()

            if training:
                window_start = (batch_index // grad_accum_steps) * grad_accum_steps
                window_size = min(grad_accum_steps, total_batches - window_start)
                (loss / window_size).backward()
                end_of_window = (batch_index + 1) % grad_accum_steps == 0
                final_batch = batch_index + 1 == total_batches
                if end_of_window or final_batch:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        b = labels.size(0)
        predictions = output.logits.argmax(dim=-1)  # (b,)
        total_examples += b
        batch_metric_totals = torch.stack(
            (
                loss.detach() * b,
                cross_entropy.detach() * b,
                balance_loss.detach() * b,
                depth_loss.detach() * b,
                predictions.eq(labels).sum(),
                output.trace.route_depth.sum(),
            )
        )  # (6,)
        metric_totals += batch_metric_totals  # (6,)

        should_report = (
            batch_index == 0
            or batch_index + 1 == total_batches
            or (log_interval > 0 and (batch_index + 1) % log_interval == 0)
        )
        if show_progress and should_report:
            current_metrics = _epoch_metrics(metric_totals, total_examples)
            postfix: dict[str, str] = {
                "loss": f"{current_metrics.loss:.4f}",
                "acc": f"{current_metrics.accuracy:.3f}",
                "depth": f"{current_metrics.mean_route_depth:.2f}",
            }
            if training:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            batches.set_postfix(postfix, refresh=True)
        elif log_interval and should_report:
            current_metrics = _epoch_metrics(metric_totals, total_examples)
            print(
                f"  batch {batch_index + 1}/{total_batches} "
                f"loss={current_metrics.loss:.4f} accuracy={current_metrics.accuracy:.3f}"
            )

    if total_examples == 0:
        raise ValueError("Data loader produced no examples")
    return _epoch_metrics(metric_totals, total_examples)


def _epoch_metrics(metric_totals: Tensor, total_examples: int) -> EpochMetrics:
    # metric_totals: (6,)
    loss, cross_entropy, load_balance, compute_penalty, correct, route_depth = metric_totals.tolist()
    return EpochMetrics(
        loss=loss / total_examples,
        cross_entropy=cross_entropy / total_examples,
        load_balance=load_balance / total_examples,
        compute_penalty=compute_penalty / total_examples,
        accuracy=correct / total_examples,
        mean_route_depth=route_depth / total_examples,
    )


def train_model(
    model: nn.Module,
    experiment: ExperimentConfig,
    loaders: MNISTDataLoaders,
    device: torch.device,
    run_path: Path,
    show_progress: bool = False,
) -> TrainingResult:
    config = experiment.training
    report = tqdm.write if show_progress else print
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(loaders.train) / config.grad_accum_steps)
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, config.epochs * optimizer_steps_per_epoch),
        warmup_fraction=config.warmup_fraction,
    )

    last_state_path = run_path / "last_state.pt"
    best_weights_path = run_path / "best_model.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation_accuracy = -1.0
    patience_counter = 0
    previous_elapsed_seconds = 0.0
    previous_peak_memory = 0
    if last_state_path.is_file():
        state = load_checkpoint(last_state_path, device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_validation_accuracy = float(state["best_validation_accuracy"])
        patience_counter = int(state["patience_counter"])
        previous_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        previous_peak_memory = int(state.get("training_peak_cuda_memory_bytes", 0))
        _restore_rng_state(state["rng_state"], loaders.train_generator)
        message = (
            f"Resuming at epoch {start_epoch}/{config.epochs}; "
            f"best validation accuracy={best_validation_accuracy:.3f} at epoch {best_epoch}"
        )
        report(message)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()
    epochs = tqdm(
        range(start_epoch, config.epochs + 1),
        desc="Training epochs",
        total=config.epochs,
        initial=start_epoch - 1,
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epochs:
        if patience_counter >= config.patience:
            message = (
                f"Early stopping before epoch {epoch}: no validation improvement for "
                f"{patience_counter} epoch(s)"
            )
            report(message)
            break
        train_metrics = run_epoch(
            model=model,
            data_loader=loaders.train,
            device=device,
            variant=experiment.model.routing_mode,
            load_balance_coefficient=config.load_balance_coefficient,
            compute_penalty_coefficient=config.compute_penalty_coefficient,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum_steps=config.grad_accum_steps,
            log_interval=config.log_interval,
            description=f"Train {epoch}/{config.epochs}",
            show_progress=show_progress,
        )
        validation_metrics = run_epoch(
            model=model,
            data_loader=loaders.validation,
            device=device,
            variant=experiment.model.routing_mode,
            load_balance_coefficient=config.load_balance_coefficient,
            compute_penalty_coefficient=config.compute_penalty_coefficient,
            description=f"Validate {epoch}/{config.epochs}",
            show_progress=show_progress,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        save_json(run_path / "training_history.json", {"epochs": history})
        improved = validation_metrics.accuracy > best_validation_accuracy
        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                best_weights_path,
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": best_validation_accuracy,
                },
            )
        else:
            patience_counter += 1

        epochs.set_postfix(
            {
                "train_acc": f"{train_metrics.accuracy:.3f}",
                "val_acc": f"{validation_metrics.accuracy:.3f}",
                "best": f"{best_validation_accuracy:.3f}",
                "patience": f"{patience_counter}/{config.patience}",
            },
            refresh=False,
        )
        epoch_message = (
            f"Epoch {epoch}/{config.epochs}: train loss={train_metrics.loss:.4f}, "
            f"train accuracy={train_metrics.accuracy:.3f}, "
            f"validation loss={validation_metrics.loss:.4f}, "
            f"validation accuracy={validation_metrics.accuracy:.3f}, "
            f"mean depth={validation_metrics.mean_route_depth:.2f}"
        )
        if improved:
            epoch_message += " [new best]"
        report(epoch_message)

        save_checkpoint(
            last_state_path,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_validation_accuracy,
                "patience_counter": patience_counter,
                "elapsed_seconds": previous_elapsed_seconds + time.perf_counter() - training_start,
                "rng_state": _capture_rng_state(loaders.train_generator),
                "training_peak_cuda_memory_bytes": max(
                    previous_peak_memory,
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                ),
            },
        )

    elapsed_seconds = previous_elapsed_seconds + time.perf_counter() - training_start
    if not best_weights_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    best_checkpoint = load_checkpoint(best_weights_path, device)
    model.load_state_dict(best_checkpoint["model_state"])
    examples_seen = len(loaders.train_indices) * len(history)
    throughput = examples_seen / max(elapsed_seconds, 1e-12)
    current_peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_memory = max(previous_peak_memory, current_peak_memory)
    return TrainingResult(
        history,
        best_epoch,
        best_validation_accuracy,
        elapsed_seconds,
        throughput,
        peak_memory,
    )


def run_experiment(
    experiment: ExperimentConfig,
    experiment_name: str,
    download: bool = False,
    command: list[str] | None = None,
    show_progress: bool = True,
) -> Path:
    experiment.validate()
    report = tqdm.write if show_progress else print
    seed_everything(experiment.training.seed)
    device = select_device(experiment.training.device)
    variant = experiment.model.routing_mode
    run_path = ensure_directory(
        run_directory(
            experiment.training.output_root,
            experiment_name,
            variant,
            experiment.training.seed,
        )
    )
    manifest_path = run_path / "manifest.json"
    require_manifest_for_resume(run_path)
    if manifest_path.is_file():
        validate_run_config(run_path, experiment)
    report(f"Preparing MNIST data for {experiment_name}...")
    loaders = build_mnist_loaders(experiment.training, download=download, device=device)
    report("Building model and checking parameter budgets...")
    model = build_mnist_model(experiment.model).to(device)

    total_parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)
    parameter_budget = experiment.model.parameter_budget
    if variant != "fixed_depth" and parameter_budget is not None and total_parameters >= parameter_budget:
        raise ValueError(
            f"SGLR model exceeds its {parameter_budget:,} parameter budget: {total_parameters:,}"
        )
    if variant != "fixed_depth" and experiment.model.require_router_smaller_than_experts:
        router_parameters = count_parameters(model.core.routers)
        expert_parameters = count_parameters(model.core.experts)
        if router_parameters > expert_parameters:
            raise ValueError(
                f"Router pool ({router_parameters:,}) outweighs expert pool ({expert_parameters:,})"
            )

    manifest = build_run_manifest(
        experiment=experiment,
        variant=variant,
        seed=experiment.training.seed,
        device=device,
        run_path=run_path,
        command=command,
    )
    manifest.update(
        {
            "expert_names": list(model.expert_names),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "data_splits": {
                "train_size": len(loaders.train_indices),
                "validation_size": len(loaders.validation_indices),
                "test_size": len(loaders.test_indices),
                "train_source": "official_train",
                "validation_source": experiment.training.validation_source,
                "test_source": "official_test",
                "train_index_sha256": _index_digest(loaders.train_indices),
                "validation_index_sha256": _index_digest(loaders.validation_indices),
                "test_index_sha256": _index_digest(loaders.test_indices),
            },
        }
    )
    save_json(manifest_path, manifest)

    run_summary = (
        f"Starting {variant} on {device}: {len(loaders.train_indices):,} train, "
        f"{len(loaders.validation_indices):,} validation, {len(loaders.test_indices):,} test; "
        f"{experiment.model.num_experts} experts, max depth {experiment.model.max_steps}; "
        f"{experiment.training.epochs} epochs, batch size {experiment.training.batch_size}, "
        f"learning rate {experiment.training.learning_rate:.1e}; "
        f"{total_parameters:,} parameters. Artifacts: {run_path}"
    )
    report(run_summary)
    training_result = train_model(
        model,
        experiment,
        loaders,
        device,
        run_path,
        show_progress=show_progress,
    )
    report(
        f"Training finished in {training_result.elapsed_seconds / 60:.1f} min; "
        f"best validation accuracy={training_result.best_validation_accuracy:.3f} "
        f"at epoch {training_result.best_epoch}. Evaluating the best checkpoint..."
    )
    evaluation_summary = evaluate_model(
        model=model,
        data_loader=loaders.test,
        device=device,
        output_directory=run_path,
        num_classes=experiment.model.num_classes,
        description="Test evaluation",
        show_progress=show_progress,
    )
    evaluation_summary.update(
        {
            "variant": variant,
            "seed": experiment.training.seed,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_epoch": training_result.best_epoch,
            "best_validation_accuracy": training_result.best_validation_accuracy,
            "training_elapsed_seconds": training_result.elapsed_seconds,
            "training_throughput_examples_per_second": training_result.throughput_examples_per_second,
            "training_peak_cuda_memory_bytes": training_result.peak_cuda_memory_bytes,
        }
    )
    save_json(run_path / "evaluation_summary.json", evaluation_summary)

    manifest.update(
        {
            "training_elapsed_seconds": training_result.elapsed_seconds,
            "training_throughput_examples_per_second": training_result.throughput_examples_per_second,
            "training_peak_cuda_memory_bytes": training_result.peak_cuda_memory_bytes,
            "evaluation": evaluation_summary,
            "best_epoch": training_result.best_epoch,
            "best_validation_accuracy": training_result.best_validation_accuracy,
            "checkpoints": {
                "best_model": str((run_path / "best_model.pt").resolve()),
                "last_state": str((run_path / "last_state.pt").resolve()),
            },
        }
    )
    save_json(manifest_path, manifest)
    save_json(
        run_path / "run_complete.json",
        {
            "schema_version": experiment.schema_version,
            "variant": variant,
            "seed": experiment.training.seed,
            "summary": "evaluation_summary.json",
        },
    )
    completion_message = (
        f"Run complete: test accuracy={float(evaluation_summary['accuracy']):.3f}, "
        f"NLL={float(evaluation_summary['nll']):.4f}, "
        f"mean depth={float(evaluation_summary['mean_route_depth']):.2f}, "
        f"training time={training_result.elapsed_seconds / 60:.1f} min"
    )
    report(completion_message)
    return run_path


def _index_digest(indices: tuple[int, ...]) -> str:
    encoded = ",".join(str(index) for index in sorted(indices)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _capture_rng_state(train_generator: torch.Generator) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "train_loader": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: object, train_generator: torch.Generator) -> None:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint RNG state must be a mapping")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_loader"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
