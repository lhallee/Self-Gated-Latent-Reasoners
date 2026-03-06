from __future__ import annotations

import json
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def create_run_directory(output_root: str | Path, run_name: str) -> Path:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = run_name.replace(" ", "_")
    run_directory = output_root_path / f"{timestamp}_{safe_run_name}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(path: str | Path, payload: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    path_obj = Path(path)
    assert path_obj.exists(), f"Expected JSON file at {path_obj}"
    return json.loads(path_obj.read_text(encoding="utf-8"))


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict:
    path_obj = Path(path)
    assert path_obj.exists(), f"Expected checkpoint at {path_obj}"
    return torch.load(path_obj, map_location=device)


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epochs"]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))

    loss_axis.plot(epochs, history["train_loss"], label="Train Loss")
    loss_axis.plot(epochs, history["eval_loss"], label="Eval Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True)
    loss_axis.legend()

    accuracy_axis.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    accuracy_axis.plot(epochs, history["eval_accuracy"], label="Eval Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.grid(True)
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(output_path_obj, dpi=300)
    plt.close(figure)


def plot_digit_route_patterns(
    digit_to_sequences: dict[int, list[tuple[int, ...]]],
    output_path: str | Path,
    route_name_fn,
    top_k: int = 10,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 5, figsize=(18, 9))
    flattened_axes = axes.flatten()

    for digit in range(10):
        axis = flattened_axes[digit]
        route_counter = Counter(digit_to_sequences[digit])
        most_common_routes = route_counter.most_common(top_k)
        route_labels = [
            textwrap.fill(" -> ".join(route_name_fn(route_id) for route_id in sequence), width=20)
            for sequence, _ in most_common_routes
        ]
        route_counts = [count for _, count in most_common_routes]
        axis.bar(range(len(most_common_routes)), route_counts, color="skyblue")
        axis.set_xticks(range(len(most_common_routes)))
        axis.set_xticklabels(route_labels, rotation=40, ha="right")
        axis.set_title(f"Digit {digit}")
        axis.set_ylabel("Frequency")

    figure.subplots_adjust(bottom=0.2, hspace=0.8, wspace=0.3)
    figure.savefig(output_path_obj, dpi=300)
    plt.close(figure)
