from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from sglr.model import SGLRModel


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    assert images.ndim == 4, "Expected image tensors shaped as [batch, channel, height, width]"
    return images.view(images.size(0), -1)


def extract_route_sequence(trace, sample_index: int) -> tuple[int, ...]:
    route_sequence: list[int] = []
    for step_index in range(trace.executed_steps):
        if not trace.active_mask[step_index, sample_index]:
            break
        route_sequence.append(int(trace.route_ids[step_index, sample_index].item()))
    return tuple(route_sequence)


def collect_digit_route_sequences(
    model: "SGLRModel",
    data_loader: "DataLoader",
    device: torch.device,
    max_samples: int = 0,
) -> dict[int, list[tuple[int, ...]]]:
    digit_to_sequences = {digit: [] for digit in range(10)}
    processed_samples = 0
    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            image_batch = flatten_images(images).to(device)
            label_batch = labels.to(device)
            output = model(image_batch)

            for sample_index in range(image_batch.size(0)):
                digit = int(label_batch[sample_index].item())
                digit_to_sequences[digit].append(extract_route_sequence(output.trace, sample_index))
                processed_samples += 1
                if max_samples > 0 and processed_samples >= max_samples:
                    return digit_to_sequences

    return digit_to_sequences


def summarize_top_routes(
    digit_to_sequences: dict[int, list[tuple[int, ...]]],
    route_name_fn,
    top_k: int = 5,
) -> dict[str, list[dict[str, int | list[str]]]]:
    summary: dict[str, list[dict[str, int | list[str]]]] = {}
    for digit, sequences in digit_to_sequences.items():
        counter = Counter(sequences)
        digit_key = f"digit_{digit}"
        summary[digit_key] = []
        for sequence, count in counter.most_common(top_k):
            summary[digit_key].append(
                {
                    "count": count,
                    "route": [route_name_fn(route_id) for route_id in sequence],
                }
            )
    return summary
