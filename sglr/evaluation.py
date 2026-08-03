"""Frozen per-example evaluation records and aggregate MNIST metrics."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from sglr.artifacts import save_json, save_jsonl
from sglr.config import ExpertSpec
from sglr.model import MNISTOutput


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_directory: str | Path,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Evaluate once and write stable JSONL, summary, and representative-image inputs."""

    output_path = Path(output_directory)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    image_batches: list[np.ndarray] = []
    image_index_batches: list[np.ndarray] = []
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)  # (c, c)
    route_depths: list[int] = []
    forced_exits = 0
    total_nll = 0.0
    total_examples = 0
    expert_visits: Tensor | None = None
    probability_entropy_sum = 0.0
    probability_entropy_count = 0

    start_time = time.perf_counter()
    with torch.inference_mode():
        for images, labels, sample_indices in data_loader:
            # images: (b, 1, 28, 28); labels: (b,); sample_indices: (b,)
            images_device = images.to(device, non_blocking=True)  # (b, 1, 28, 28)
            labels_device = labels.to(device, non_blocking=True)  # (b,)
            output = model(images_device)
            if not isinstance(output, MNISTOutput):
                raise TypeError("MNIST evaluation expects the model to return MNISTOutput")

            logits = output.logits  # (b, c)
            probabilities = logits.softmax(dim=-1)  # (b, c)
            confidences, predictions = probabilities.max(dim=-1)  # (b,), (b,)
            total_nll += float(F.cross_entropy(logits, labels_device, reduction="sum").item())
            batch_confusion = torch.bincount(
                labels_device * num_classes + predictions,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)  # (c, c)
            confusion += batch_confusion.cpu()

            trace = output.trace
            if expert_visits is None:
                expert_visits = torch.zeros(trace.num_experts, dtype=torch.long)  # (e,)
            active_probs = trace.route_probs[: trace.executed_steps][
                trace.active_mask[: trace.executed_steps]
            ]  # (n_active, e + 1)
            if active_probs.numel():
                entropies = -(active_probs * active_probs.clamp_min(1e-12).log()).sum(dim=-1)  # (n_active,)
                probability_entropy_sum += float(entropies.sum().item())
                probability_entropy_count += entropies.numel()

            for batch_position in range(labels.size(0)):
                route_ids = _sample_expert_route(output, batch_position)
                for route_id in route_ids:
                    expert_visits[route_id] += 1
                route_depth = len(route_ids)
                forced_exit = bool(trace.forced_exit[batch_position].item())
                record = {
                    "sample_index": int(sample_indices[batch_position].item()),
                    "label": int(labels[batch_position].item()),
                    "prediction": int(predictions[batch_position].item()),
                    "confidence": float(confidences[batch_position].item()),
                    "correct": bool(predictions[batch_position].eq(labels_device[batch_position]).item()),
                    "route_ids": route_ids,
                    "route_depth": route_depth,
                    "forced_exit": forced_exit,
                }
                records.append(record)
                route_depths.append(route_depth)
                forced_exits += int(forced_exit)

            image_batches.append(images.cpu().numpy())  # (b, 1, 28, 28)
            image_index_batches.append(sample_indices.cpu().numpy())  # (b,)
            total_examples += labels.size(0)

    elapsed_seconds = time.perf_counter() - start_time
    if total_examples == 0 or expert_visits is None:
        raise ValueError("Evaluation loader produced no examples")

    expert_compute = _expert_mac_estimates(model)
    mean_expert_compute = sum(
        expert_compute[route_id] for record in records for route_id in record["route_ids"]
    ) / total_examples
    visit_total = int(expert_visits.sum().item())
    utilization = (
        expert_visits.to(torch.float64) / visit_total
        if visit_total
        else torch.zeros_like(expert_visits, dtype=torch.float64)
    )  # (e,)
    populated = utilization.gt(0)  # (e,)
    utilization_entropy = float(-(utilization[populated] * utilization[populated].log()).sum().item())
    mean_router_entropy = probability_entropy_sum / max(1, probability_entropy_count)
    sorted_depths = sorted(route_depths)
    p95_index = max(0, math.ceil(0.95 * len(sorted_depths)) - 1)
    summary: dict[str, Any] = {
        "accuracy": float(confusion.diag().sum().item() / total_examples),
        "test_accuracy": float(confusion.diag().sum().item() / total_examples),
        "nll": total_nll / total_examples,
        "confusion_matrix": confusion.tolist(),
        "mean_route_depth": sum(route_depths) / total_examples,
        "p95_route_depth": sorted_depths[p95_index],
        "forced_exit_rate": forced_exits / total_examples,
        "route_entropy": mean_router_entropy,
        "mean_router_probability_entropy": mean_router_entropy,
        "expert_utilization_entropy": utilization_entropy,
        "expert_utilization": utilization.tolist(),
        "expert_visit_counts": expert_visits.tolist(),
        "expert_mac_estimates": expert_compute,
        "mean_expert_compute": mean_expert_compute,
        "examples": total_examples,
        "elapsed_seconds": elapsed_seconds,
        "throughput_examples_per_second": total_examples / max(elapsed_seconds, 1e-12),
        "evaluation_peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    save_jsonl(output_path / "evaluation.jsonl", records)
    save_json(output_path / "evaluation_summary.json", summary)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "evaluation_images.npz",
        images=np.concatenate(image_batches, axis=0),  # (n, 1, 28, 28)
        sample_indices=np.concatenate(image_index_batches, axis=0),  # (n,)
    )
    return summary


def _sample_expert_route(output: MNISTOutput, batch_position: int) -> list[int]:
    trace = output.trace
    active = trace.active_mask[: trace.executed_steps, batch_position]  # (s,)
    choices = trace.route_ids[: trace.executed_steps, batch_position][active]  # (s_active,)
    expert_choices = choices[choices.lt(trace.num_experts)]  # (route_depth,)
    return [int(route_id) for route_id in expert_choices.tolist()]


def _expert_mac_estimates(model: nn.Module) -> list[int]:
    """Estimate multiply-accumulates for each expert delta at one recurrent step."""

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Cannot find model configuration for analytical compute estimates")
    specs: tuple[ExpertSpec, ...] = getattr(model, "specs", config.experts)
    length = config.num_patches
    hidden_size = config.hidden_size
    return [_expert_macs(spec, length, hidden_size) for spec in specs]


def _expert_macs(spec: ExpertSpec, length: int, hidden_size: int) -> int:
    if spec.family == "mlp":
        return 2 * length * hidden_size * spec.hidden_size
    if spec.family == "attention":
        projection_macs = 4 * length * hidden_size * spec.internal_size
        attention_macs = 2 * length * length * spec.internal_size
        return projection_macs + attention_macs
    if spec.family == "conv":
        kernel_area = spec.kernel_size[0] * spec.kernel_size[1]
        return 2 * length * hidden_size * spec.channels * kernel_area
    raise ValueError(f"Unsupported expert family: {spec.family}")
