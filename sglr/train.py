from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from sglr.analysis import collect_digit_route_sequences, flatten_images, summarize_top_routes
from sglr.artifacts import plot_digit_route_patterns, plot_training_curves, save_checkpoint, save_json
from sglr.config import ModelConfig, TrainingConfig
from sglr.router import load_balancing_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_mnist_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def maybe_subset_dataset(dataset: Dataset, subset_size: int) -> Dataset:
    if subset_size <= 0:
        return dataset
    subset_size = min(subset_size, len(dataset))
    return Subset(dataset, range(subset_size))


def build_mnist_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    train_subset: int = 0,
    test_subset: int = 0,
) -> tuple[DataLoader, DataLoader]:
    transform = build_mnist_transforms()
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_dataset = maybe_subset_dataset(train_dataset, train_subset)
    test_dataset = maybe_subset_dataset(test_dataset, test_subset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def count_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def summarize_route_depth(trace) -> float:
    if trace.executed_steps == 0:
        return 0.0
    active_steps = trace.active_mask[: trace.executed_steps].sum(dim=0).float()
    exit_choices = trace.route_ids[: trace.executed_steps] == trace.exit_route_index
    expert_steps = active_steps - exit_choices.sum(dim=0).float()
    return float(expert_steps.mean().item())


def combine_outputs_for_loss(outputs: list, labels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert outputs, "At least one model output is required to build an accumulation window"
    assert labels, "At least one label tensor is required to build an accumulation window"
    combined_logits = torch.cat([output.logits for output in outputs], dim=0)
    combined_labels = torch.cat(labels, dim=0)
    combined_route_probs = torch.cat([output.trace.route_probs for output in outputs], dim=1)
    combined_active_mask = torch.cat([output.trace.active_mask for output in outputs], dim=1)
    return combined_logits, combined_labels, combined_route_probs, combined_active_mask


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
    load_balancing_coef: float,
    grad_accum_steps: int,
    log_interval: int,
    train_mode: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    if train_mode:
        model.train()
        assert optimizer is not None, "An optimizer is required for training"
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cross_entropy_loss = 0.0
    total_load_balancing_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_route_depth = 0.0
    mode_name = "Train" if train_mode else "Eval"
    progress_bar = tqdm(data_loader, desc=mode_name, leave=False)
    total_batches = len(data_loader) if max_batches == 0 else min(len(data_loader), max_batches)
    accumulation_outputs: list = []
    accumulation_labels: list[torch.Tensor] = []
    latest_loss_value = 0.0
    latest_accuracy_value = 0.0

    for batch_index, (images, labels) in enumerate(progress_bar, start=1):
        image_batch = flatten_images(images).to(device)
        label_batch = labels.to(device)

        with torch.set_grad_enabled(train_mode):
            output = model(image_batch)
            if train_mode:
                accumulation_outputs.append(output)
                accumulation_labels.append(label_batch)
                should_step = batch_index % grad_accum_steps == 0 or batch_index == total_batches
                if should_step:
                    combined_logits, combined_labels, combined_route_probs, combined_active_mask = combine_outputs_for_loss(
                        outputs=accumulation_outputs,
                        labels=accumulation_labels,
                    )
                    classification_loss = criterion(combined_logits, combined_labels)
                    balancing_loss = load_balancing_loss(combined_route_probs, combined_active_mask)
                    loss = classification_loss + load_balancing_coef * balancing_loss
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    accumulation_example_count = combined_labels.size(0)
                    total_loss += float(loss.item()) * accumulation_example_count
                    total_cross_entropy_loss += float(classification_loss.item()) * accumulation_example_count
                    total_load_balancing_loss += float(balancing_loss.item()) * accumulation_example_count
                    latest_loss_value = float(loss.item())
                    latest_accuracy_value = 100.0 * total_correct / total_examples
                    accumulation_outputs.clear()
                    accumulation_labels.clear()
            else:
                route_probs = output.trace.route_probs[: output.trace.executed_steps]
                active_mask = output.trace.active_mask[: output.trace.executed_steps]
                classification_loss = criterion(output.logits, label_batch)
                balancing_loss = load_balancing_loss(route_probs, active_mask)
                loss = classification_loss + load_balancing_coef * balancing_loss
                total_loss += float(loss.item()) * label_batch.size(0)
                total_cross_entropy_loss += float(classification_loss.item()) * label_batch.size(0)
                total_load_balancing_loss += float(balancing_loss.item()) * label_batch.size(0)
                latest_loss_value = float(loss.item())

        predictions = output.logits.argmax(dim=-1)
        batch_size = label_batch.size(0)
        total_examples += batch_size
        total_correct += int(predictions.eq(label_batch).sum().item())
        total_route_depth += summarize_route_depth(output.trace) * batch_size
        latest_accuracy_value = 100.0 * total_correct / total_examples

        if batch_index % log_interval == 0 or batch_index == total_batches:
            progress_bar.set_postfix(
                {
                    "loss": f"{latest_loss_value:.4f}",
                    "acc": f"{latest_accuracy_value:.2f}",
                }
            )

        if batch_index >= total_batches:
            break

    return {
        "loss": total_loss / total_examples,
        "cross_entropy_loss": total_cross_entropy_loss / total_examples,
        "load_balancing_loss": total_load_balancing_loss / total_examples,
        "accuracy": 100.0 * total_correct / total_examples,
        "route_depth": total_route_depth / total_examples,
    }


def train_model(
    model: nn.Module,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    stage_dir: str | Path,
    device: torch.device,
) -> dict[str, list[float]]:
    stage_path = Path(stage_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / training_config.grad_accum_steps))
    total_training_steps = max(1, training_config.epochs * optimizer_steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(training_config.warmup_steps, total_training_steps),
        num_training_steps=total_training_steps,
    )

    history = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_route_depth": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_route_depth": [],
    }
    best_eval_accuracy = -1.0
    best_state_dict = None
    patience_counter = 0

    save_json(stage_path / "model_config.json", asdict(model_config))
    save_json(stage_path / "training_config.json", asdict(training_config))

    for epoch_index in range(1, training_config.epochs + 1):
        print(f"Epoch {epoch_index}/{training_config.epochs}")
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            load_balancing_coef=training_config.load_balancing_coef,
            grad_accum_steps=training_config.grad_accum_steps,
            log_interval=training_config.log_interval,
            train_mode=True,
            max_batches=0,
        )
        eval_metrics = run_epoch(
            model=model,
            data_loader=eval_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            load_balancing_coef=training_config.load_balancing_coef,
            grad_accum_steps=1,
            log_interval=training_config.log_interval,
            train_mode=False,
            max_batches=training_config.eval_batches,
        )

        history["epochs"].append(epoch_index)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_route_depth"].append(train_metrics["route_depth"])
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_accuracy"].append(eval_metrics["accuracy"])
        history["eval_route_depth"].append(eval_metrics["route_depth"])

        print(
            "  "
            f"[Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}% | Depth: {train_metrics['route_depth']:.2f}"
        )
        print(
            "  "
            f"[Eval]  Loss: {eval_metrics['loss']:.4f} | Acc: {eval_metrics['accuracy']:.2f}% | Depth: {eval_metrics['route_depth']:.2f}"
        )

        checkpoint_payload = {
            "epoch": epoch_index,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "expert_names": list(model.expert_names),
        }
        save_checkpoint(stage_path / "last_model.pt", checkpoint_payload)
        save_json(stage_path / "training_history.json", history)

        if eval_metrics["accuracy"] > best_eval_accuracy:
            best_eval_accuracy = eval_metrics["accuracy"]
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_checkpoint(stage_path / "best_model.pt", checkpoint_payload)
        else:
            patience_counter += 1
            if patience_counter >= training_config.patience:
                print(f"Early stopping triggered after epoch {epoch_index}.")
                break

    assert best_state_dict is not None, "Training never produced a best checkpoint"
    model.load_state_dict(best_state_dict)
    plot_training_curves(history, stage_path / "training_curves.png")
    save_json(
        stage_path / "training_summary.json",
        {
            "best_eval_accuracy": best_eval_accuracy,
            "parameter_count": count_parameter_count(model),
            "max_steps": model_config.max_steps,
        },
    )
    return history


def save_route_artifacts(
    model,
    data_loader: DataLoader,
    device: torch.device,
    stage_dir: str | Path,
    max_samples: int,
) -> None:
    stage_path = Path(stage_dir)
    digit_to_sequences = collect_digit_route_sequences(model=model, data_loader=data_loader, device=device, max_samples=max_samples)
    route_summary = summarize_top_routes(digit_to_sequences, model.route_name)
    save_json(stage_path / "digit_route_summary.json", route_summary)
    plot_digit_route_patterns(
        digit_to_sequences=digit_to_sequences,
        output_path=stage_path / "digit_usage_patterns.png",
        route_name_fn=model.route_name,
    )


def save_reconstruction_preview(
    image_tensor: torch.Tensor,
    reconstruction_tensor: torch.Tensor,
    output_path: str | Path,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(image_tensor.view(28, 28).cpu(), cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(reconstruction_tensor.view(28, 28).cpu(), cmap="gray")
    axes[1].set_title("Probe")
    axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path_obj, dpi=300)
    plt.close(figure)
