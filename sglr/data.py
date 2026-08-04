"""Deterministic MNIST splits and indexed data loaders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sglr.config import TrainingConfig


MNIST_MEAN = (0.1307,)
MNIST_STANDARD_DEVIATION = (0.3081,)


@dataclass(frozen=True, slots=True)
class MNISTDataLoaders:
    """Loaders whose batches contain images, labels, and source-dataset indices."""

    train: DataLoader[tuple[Tensor, Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor, Tensor]]
    test: DataLoader[tuple[Tensor, Tensor, Tensor]]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_generator: torch.Generator


class TensorMNISTSubset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Materialize a normalized MNIST subset once instead of transforming every epoch."""

    def __init__(self, images: Tensor, labels: Tensor, indices: Sequence[int]) -> None:
        source_indices = torch.as_tensor(indices, dtype=torch.long)  # (n,)
        normalized_images = images.index_select(0, source_indices).unsqueeze(1).to(torch.float32)  # (n, 1, 28, 28)
        normalized_images.div_(255.0).sub_(MNIST_MEAN[0]).div_(MNIST_STANDARD_DEVIATION[0])
        self.images = normalized_images  # (n, 1, 28, 28)
        self.labels = labels.index_select(0, source_indices).to(torch.long)  # (n,)
        self.indices = source_indices  # (n,)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.images[position]  # (1, 28, 28)
        label = self.labels[position]  # ()
        source_index = self.indices[position]  # ()
        return image, label, source_index


def stratified_split_indices(
    labels: Sequence[int] | Tensor,
    first_size: int,
    second_size: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return disjoint, deterministic, approximately class-balanced index sets."""

    label_tensor = torch.as_tensor(labels, dtype=torch.long)  # (n,)
    if label_tensor.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if first_size < 0 or second_size < 0:
        raise ValueError("split sizes must be non-negative")
    if first_size + second_size > label_tensor.numel():
        raise ValueError("requested split sizes exceed the dataset size")

    generator = torch.Generator().manual_seed(seed)
    first_counts = _allocate_class_counts(label_tensor, first_size)  # (c,)
    remaining_counts = torch.bincount(label_tensor) - first_counts  # (c,)
    second_counts = _allocate_counts_from_capacity(remaining_counts, second_size)  # (c,)

    first_indices: list[int] = []
    second_indices: list[int] = []
    for class_index in range(first_counts.numel()):
        class_indices = torch.nonzero(label_tensor.eq(class_index), as_tuple=False).flatten()  # (n_class,)
        order = torch.randperm(class_indices.numel(), generator=generator)  # (n_class,)
        shuffled = class_indices.index_select(0, order)  # (n_class,)
        first_count = int(first_counts[class_index].item())
        second_count = int(second_counts[class_index].item())
        first_indices.extend(shuffled[:first_count].tolist())
        second_indices.extend(shuffled[first_count : first_count + second_count].tolist())

    first_order = torch.randperm(len(first_indices), generator=generator).tolist()  # (first_size,)
    second_order = torch.randperm(len(second_indices), generator=generator).tolist()  # (second_size,)
    first = tuple(first_indices[position] for position in first_order)
    second = tuple(second_indices[position] for position in second_order)
    return first, second


def _allocate_class_counts(labels: Tensor, requested_size: int) -> Tensor:
    # labels: (n,)
    capacities = torch.bincount(labels)  # (c,)
    return _allocate_counts_from_capacity(capacities, requested_size)  # (c,)


def _allocate_counts_from_capacity(capacities: Tensor, requested_size: int) -> Tensor:
    # capacities: (c,)
    if requested_size > int(capacities.sum().item()):
        raise ValueError("requested size exceeds available class capacity")
    if requested_size == 0:
        return torch.zeros_like(capacities)  # (c,)

    ideal = capacities.to(torch.float64) * requested_size / capacities.sum()  # (c,)
    allocated = torch.floor(ideal).to(torch.long)  # (c,)
    remainder = requested_size - int(allocated.sum().item())
    priority = ideal - allocated  # (c,)
    while remainder:
        eligible = allocated.lt(capacities)  # (c,)
        selected = torch.where(eligible, priority, torch.full_like(priority, -1.0)).argmax()
        allocated[selected] += 1
        priority[selected] = -1.0
        remainder -= 1
    return allocated  # (c,)


def mnist_split_indices(
    official_train_labels: Tensor,
    official_test_labels: Tensor,
    config: TrainingConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Resolve train, validation, and test indices without loading image tensors."""

    if config.validation_source == "official_train":
        train_indices, validation_indices = stratified_split_indices(
            official_train_labels,
            config.train_size,
            config.validation_size,
            config.seed,
        )
        requested_test_size = (
            official_test_labels.numel()
            if config.test_size == 0
            else min(config.test_size, official_test_labels.numel())
        )
        test_indices, _ = stratified_split_indices(
            official_test_labels,
            requested_test_size,
            0,
            config.seed,
        )
    else:
        train_indices, _ = stratified_split_indices(
            official_train_labels,
            config.train_size,
            0,
            config.seed,
        )
        validation_indices, test_indices = stratified_split_indices(
            official_test_labels,
            config.validation_size,
            config.test_size,
            config.seed,
        )

    return train_indices, validation_indices, test_indices


def build_mnist_loaders(
    config: TrainingConfig,
    download: bool = False,
    device: torch.device | str | None = None,
) -> MNISTDataLoaders:
    """Build deterministic, disjoint MNIST loaders from the configured sources."""

    from torchvision.datasets import MNIST
    data_root = Path(config.data_root)
    official_train = MNIST(data_root, train=True, download=download)
    official_test = MNIST(data_root, train=False, download=download)

    train_indices, validation_indices, test_indices = mnist_split_indices(
        official_train.targets,
        official_test.targets,
        config,
    )
    if config.validation_source == "official_train":
        validation_images = official_train.data
        validation_labels = official_train.targets
    else:
        validation_images = official_test.data
        validation_labels = official_test.targets

    train_dataset = TensorMNISTSubset(official_train.data, official_train.targets, train_indices)
    validation_dataset = TensorMNISTSubset(validation_images, validation_labels, validation_indices)
    test_dataset = TensorMNISTSubset(official_test.data, official_test.targets, test_indices)
    loader_generator = torch.Generator().manual_seed(config.seed)
    selected_device = torch.device(device) if device is not None else None
    pin_memory = selected_device.type == "cuda" if selected_device is not None else torch.cuda.is_available()
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        persistent_workers=config.num_workers > 0,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        persistent_workers=config.num_workers > 0,
        **loader_options,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return MNISTDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        train_generator=loader_generator,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
