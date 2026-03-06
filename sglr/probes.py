from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from sglr.analysis import flatten_images
from sglr.artifacts import save_checkpoint, save_json
from sglr.config import ModelConfig, ProbeTrainingConfig


@dataclass
class ProbeInferenceOutput:
    classifier_logits: torch.Tensor
    reconstructions: torch.Tensor
    visited_mask: torch.Tensor


class ExpertProbe(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_size, num_classes)
        self.reconstructor = nn.Linear(input_size, input_size)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(latents), self.reconstructor(latents)


class ExpertProbeSuite(nn.Module):
    def __init__(self, num_experts: int, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.probes = nn.ModuleList([ExpertProbe(input_size=input_size, num_classes=num_classes) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.input_size = input_size
        self.num_classes = num_classes

    def forward_expert(self, expert_index: int, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.probes[expert_index](latents)


def run_probes_on_trace(probe_suite: ExpertProbeSuite, trace) -> ProbeInferenceOutput:
    step_count = trace.executed_steps
    batch_size = trace.final_latent.size(0)
    latent_size = trace.final_latent.size(1)
    device = trace.final_latent.device
    classifier_logits = torch.zeros(step_count, batch_size, probe_suite.num_classes, device=device, dtype=trace.final_latent.dtype)
    reconstructions = torch.zeros(step_count, batch_size, latent_size, device=device, dtype=trace.final_latent.dtype)
    visited_mask = torch.zeros(step_count, batch_size, device=device, dtype=torch.bool)

    for expert_index in range(probe_suite.num_experts):
        expert_visits = trace.route_ids[:step_count] == expert_index
        if not expert_visits.any():
            continue
        step_indices, batch_indices = torch.where(expert_visits)
        visited_latents = trace.post_step_latents[step_indices, batch_indices]
        expert_logits, expert_reconstructions = probe_suite.forward_expert(expert_index, visited_latents)
        classifier_logits[step_indices, batch_indices] = expert_logits
        reconstructions[step_indices, batch_indices] = expert_reconstructions
        visited_mask[step_indices, batch_indices] = True

    return ProbeInferenceOutput(
        classifier_logits=classifier_logits,
        reconstructions=reconstructions,
        visited_mask=visited_mask,
    )


def compute_probe_losses(
    probe_output: ProbeInferenceOutput,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    assert probe_output.visited_mask.any(), "At least one probe position must be visited before computing probe losses"
    step_count = probe_output.visited_mask.size(0)
    expanded_images = images.unsqueeze(0).expand(step_count, -1, -1)
    expanded_labels = labels.unsqueeze(0).expand(step_count, -1)
    classifier_loss = nn.CrossEntropyLoss()(probe_output.classifier_logits[probe_output.visited_mask], expanded_labels[probe_output.visited_mask])
    reconstruction_loss = nn.MSELoss()(probe_output.reconstructions[probe_output.visited_mask], expanded_images[probe_output.visited_mask])
    predicted_classes = probe_output.classifier_logits[probe_output.visited_mask].argmax(dim=-1)
    label_targets = expanded_labels[probe_output.visited_mask]
    correct_predictions = int(predicted_classes.eq(label_targets).sum().item())
    total_predictions = int(label_targets.numel())
    return classifier_loss, reconstruction_loss, correct_predictions, total_predictions


def combine_probe_window(
    classifier_losses: list[torch.Tensor],
    reconstruction_losses: list[torch.Tensor],
    visited_positions: list[int],
    classification_loss_coef: float,
    reconstruction_loss_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    assert classifier_losses, "At least one probe loss is required to build an accumulation window"
    assert reconstruction_losses, "At least one probe reconstruction loss is required to build an accumulation window"
    assert visited_positions, "At least one visited-position count is required to build an accumulation window"
    total_positions = sum(visited_positions)
    classifier_loss = torch.stack(
        [
            loss * (position_count / total_positions)
            for loss, position_count in zip(classifier_losses, visited_positions)
        ]
    ).sum()
    reconstruction_loss = torch.stack(
        [
            loss * (position_count / total_positions)
            for loss, position_count in zip(reconstruction_losses, visited_positions)
        ]
    ).sum()
    total_loss = (
        classification_loss_coef * classifier_loss
        + reconstruction_loss_coef * reconstruction_loss
    )
    return total_loss, classifier_loss, reconstruction_loss, total_positions


def run_probe_epoch(
    backbone: nn.Module,
    probe_suite: ExpertProbeSuite,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
    probe_config: ProbeTrainingConfig,
    train_mode: bool,
) -> dict[str, float]:
    backbone.eval()
    if train_mode:
        probe_suite.train()
        assert optimizer is not None, "An optimizer is required when training probes"
        optimizer.zero_grad(set_to_none=True)
    else:
        probe_suite.eval()

    total_loss = 0.0
    total_classifier_loss = 0.0
    total_reconstruction_loss = 0.0
    total_correct = 0
    total_predictions = 0
    total_positions = 0
    mode_name = "ProbeTrain" if train_mode else "ProbeEval"
    progress_bar = tqdm(data_loader, desc=mode_name, leave=False)
    total_batches = len(data_loader)
    accumulation_classifier_losses: list[torch.Tensor] = []
    accumulation_reconstruction_losses: list[torch.Tensor] = []
    accumulation_visited_positions: list[int] = []
    latest_loss_value = 0.0
    latest_accuracy_value = 0.0

    for batch_index, (images, labels) in enumerate(progress_bar, start=1):
        image_batch = flatten_images(images).to(device)
        label_batch = labels.to(device)

        with torch.no_grad():
            backbone_output = backbone(image_batch)

        with torch.set_grad_enabled(train_mode):
            probe_output = run_probes_on_trace(probe_suite, backbone_output.trace)
            classifier_loss, reconstruction_loss, correct_predictions, total_predictions_in_batch = compute_probe_losses(
                probe_output=probe_output,
                images=image_batch,
                labels=label_batch,
            )
            loss = (
                probe_config.classification_loss_coef * classifier_loss
                + probe_config.reconstruction_loss_coef * reconstruction_loss
            )

            if train_mode:
                visited_positions = int(probe_output.visited_mask.sum().item())
                accumulation_classifier_losses.append(classifier_loss)
                accumulation_reconstruction_losses.append(reconstruction_loss)
                accumulation_visited_positions.append(visited_positions)
                should_step = batch_index % probe_config.grad_accum_steps == 0 or batch_index == total_batches
                if should_step:
                    total_loss_window, classifier_loss_window, reconstruction_loss_window, total_positions_window = combine_probe_window(
                        classifier_losses=accumulation_classifier_losses,
                        reconstruction_losses=accumulation_reconstruction_losses,
                        visited_positions=accumulation_visited_positions,
                        classification_loss_coef=probe_config.classification_loss_coef,
                        reconstruction_loss_coef=probe_config.reconstruction_loss_coef,
                    )
                    total_loss_window.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss += float(total_loss_window.item()) * total_positions_window
                    total_classifier_loss += float(classifier_loss_window.item()) * total_positions_window
                    total_reconstruction_loss += float(reconstruction_loss_window.item()) * total_positions_window
                    latest_loss_value = float(total_loss_window.item())
                    accumulation_classifier_losses.clear()
                    accumulation_reconstruction_losses.clear()
                    accumulation_visited_positions.clear()
            else:
                visited_positions = int(probe_output.visited_mask.sum().item())
                total_loss += float(loss.item()) * visited_positions
                total_classifier_loss += float(classifier_loss.item()) * visited_positions
                total_reconstruction_loss += float(reconstruction_loss.item()) * visited_positions
                latest_loss_value = float(loss.item())

        total_positions += visited_positions
        total_correct += correct_predictions
        total_predictions += total_predictions_in_batch
        latest_accuracy_value = 100.0 * total_correct / total_predictions

        if batch_index % probe_config.log_interval == 0 or batch_index == total_batches:
            progress_bar.set_postfix(
                {
                    "loss": f"{latest_loss_value:.4f}",
                    "acc": f"{latest_accuracy_value:.2f}",
                }
            )

    return {
        "loss": total_loss / total_positions,
        "classifier_loss": total_classifier_loss / total_positions,
        "reconstruction_loss": total_reconstruction_loss / total_positions,
        "classification_accuracy": 100.0 * total_correct / total_predictions,
        "visited_positions": float(total_positions),
    }


def train_probes(
    backbone: nn.Module,
    probe_suite: ExpertProbeSuite,
    model_config: ModelConfig,
    probe_config: ProbeTrainingConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    stage_dir: str | Path,
    device: torch.device,
) -> dict[str, list[float]]:
    stage_path = Path(stage_dir)
    optimizer = torch.optim.AdamW(probe_suite.parameters(), lr=probe_config.learning_rate, weight_decay=probe_config.weight_decay)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / probe_config.grad_accum_steps))
    total_training_steps = max(1, probe_config.epochs * optimizer_steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(probe_config.warmup_steps, total_training_steps),
        num_training_steps=total_training_steps,
    )

    history = {
        "epochs": [],
        "train_loss": [],
        "train_classifier_loss": [],
        "train_reconstruction_loss": [],
        "train_accuracy": [],
        "eval_loss": [],
        "eval_classifier_loss": [],
        "eval_reconstruction_loss": [],
        "eval_accuracy": [],
    }
    best_eval_accuracy = -1.0
    best_eval_reconstruction_loss = float("inf")
    best_probe_state = None
    patience_counter = 0

    save_json(stage_path / "probe_config.json", asdict(probe_config))
    save_json(stage_path / "probe_model_config.json", asdict(model_config))

    for epoch_index in range(1, probe_config.epochs + 1):
        print(f"Probe epoch {epoch_index}/{probe_config.epochs}")
        train_metrics = run_probe_epoch(
            backbone=backbone,
            probe_suite=probe_suite,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            probe_config=probe_config,
            train_mode=True,
        )
        eval_metrics = run_probe_epoch(
            backbone=backbone,
            probe_suite=probe_suite,
            data_loader=eval_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            probe_config=probe_config,
            train_mode=False,
        )

        history["epochs"].append(epoch_index)
        history["train_loss"].append(train_metrics["loss"])
        history["train_classifier_loss"].append(train_metrics["classifier_loss"])
        history["train_reconstruction_loss"].append(train_metrics["reconstruction_loss"])
        history["train_accuracy"].append(train_metrics["classification_accuracy"])
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_classifier_loss"].append(eval_metrics["classifier_loss"])
        history["eval_reconstruction_loss"].append(eval_metrics["reconstruction_loss"])
        history["eval_accuracy"].append(eval_metrics["classification_accuracy"])
        save_json(stage_path / "probe_history.json", history)

        print(
            "  "
            f"[Probe Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['classification_accuracy']:.2f}%"
        )
        print(
            "  "
            f"[Probe Eval]  Loss: {eval_metrics['loss']:.4f} | Acc: {eval_metrics['classification_accuracy']:.2f}%"
        )

        checkpoint_payload = {
            "epoch": epoch_index,
            "probe_state": probe_suite.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "probe_config": asdict(probe_config),
            "model_config": asdict(model_config),
        }
        save_checkpoint(stage_path / "last_probes.pt", checkpoint_payload)

        is_better_accuracy = eval_metrics["classification_accuracy"] > best_eval_accuracy
        is_better_reconstruction = (
            eval_metrics["classification_accuracy"] == best_eval_accuracy
            and eval_metrics["reconstruction_loss"] < best_eval_reconstruction_loss
        )
        if is_better_accuracy or is_better_reconstruction:
            best_eval_accuracy = eval_metrics["classification_accuracy"]
            best_eval_reconstruction_loss = eval_metrics["reconstruction_loss"]
            best_probe_state = copy.deepcopy(probe_suite.state_dict())
            patience_counter = 0
            save_checkpoint(stage_path / "best_probes.pt", checkpoint_payload)
        else:
            patience_counter += 1
            if patience_counter >= probe_config.patience:
                print(f"Probe early stopping triggered after epoch {epoch_index}.")
                break

    assert best_probe_state is not None, "Probe training never produced a best checkpoint"
    probe_suite.load_state_dict(best_probe_state)
    save_json(
        stage_path / "probe_summary.json",
        {
            "best_eval_accuracy": best_eval_accuracy,
            "best_eval_reconstruction_loss": best_eval_reconstruction_loss,
            "num_experts": model_config.num_experts,
        },
    )
    return history


def summarize_probe_performance(
    backbone: nn.Module,
    probe_suite: ExpertProbeSuite,
    data_loader: DataLoader,
    device: torch.device,
    expert_names: list[str],
) -> dict[str, list[dict[str, float | int | str]]]:
    backbone.eval()
    probe_suite.eval()
    per_expert_totals = [
        {"correct": 0, "count": 0, "reconstruction_loss": 0.0}
        for _ in range(len(expert_names))
    ]

    with torch.no_grad():
        for images, labels in data_loader:
            image_batch = flatten_images(images).to(device)
            label_batch = labels.to(device)
            trace = backbone(image_batch).trace
            probe_output = run_probes_on_trace(probe_suite, trace)
            expanded_images = image_batch.unsqueeze(0).expand(trace.executed_steps, -1, -1)
            expanded_labels = label_batch.unsqueeze(0).expand(trace.executed_steps, -1)

            for expert_index, expert_name in enumerate(expert_names):
                expert_visits = trace.route_ids[: trace.executed_steps] == expert_index
                if not expert_visits.any():
                    continue
                predicted_labels = probe_output.classifier_logits[expert_visits].argmax(dim=-1)
                target_labels = expanded_labels[expert_visits]
                reconstruction_mse = torch.mean(
                    (probe_output.reconstructions[expert_visits] - expanded_images[expert_visits]) ** 2,
                    dim=-1,
                )
                per_expert_totals[expert_index]["correct"] += int(predicted_labels.eq(target_labels).sum().item())
                per_expert_totals[expert_index]["count"] += int(target_labels.numel())
                per_expert_totals[expert_index]["reconstruction_loss"] += float(reconstruction_mse.sum().item())

    expert_rows: list[dict[str, float | int | str]] = []
    for expert_index, expert_name in enumerate(expert_names):
        count = per_expert_totals[expert_index]["count"]
        accuracy = 0.0 if count == 0 else 100.0 * per_expert_totals[expert_index]["correct"] / count
        reconstruction_loss = 0.0 if count == 0 else per_expert_totals[expert_index]["reconstruction_loss"] / count
        expert_rows.append(
            {
                "expert_name": expert_name,
                "visit_count": count,
                "classification_accuracy": accuracy,
                "reconstruction_loss": reconstruction_loss,
            }
        )
    return {"experts": expert_rows}
