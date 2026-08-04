from __future__ import annotations

import torch

from sglr.config import TrainingConfig
from sglr.data import (
    MNIST_MEAN,
    MNIST_STANDARD_DEVIATION,
    TensorMNISTSubset,
    mnist_split_indices,
    stratified_split_indices,
)


def test_stratified_split_is_deterministic_disjoint_and_balanced() -> None:
    labels = torch.arange(10).repeat_interleave(20)

    first, second = stratified_split_indices(labels, first_size=100, second_size=50, seed=7)
    repeated_first, repeated_second = stratified_split_indices(
        labels,
        first_size=100,
        second_size=50,
        seed=7,
    )

    assert first == repeated_first
    assert second == repeated_second
    assert len(first) == 100
    assert len(second) == 50
    assert set(first).isdisjoint(second)
    assert torch.bincount(labels[list(first)], minlength=10).tolist() == [10] * 10
    assert torch.bincount(labels[list(second)], minlength=10).tolist() == [5] * 10


def test_different_split_seed_changes_membership() -> None:
    labels = torch.arange(10).repeat_interleave(20)

    first_seed, _ = stratified_split_indices(labels, 100, 50, seed=7)
    second_seed, _ = stratified_split_indices(labels, 100, 50, seed=17)

    assert first_seed != second_seed


def test_tensor_subset_materializes_the_standard_mnist_normalization() -> None:
    images = torch.zeros(2, 28, 28, dtype=torch.uint8)
    images[1].fill_(255)
    labels = torch.tensor([3, 7])
    dataset = TensorMNISTSubset(images, labels, indices=(1, 0))

    bright_image, bright_label, bright_index = dataset[0]
    dark_image, dark_label, dark_index = dataset[1]
    expected_bright = (1.0 - MNIST_MEAN[0]) / MNIST_STANDARD_DEVIATION[0]
    expected_dark = -MNIST_MEAN[0] / MNIST_STANDARD_DEVIATION[0]

    assert bright_image.shape == (1, 28, 28)
    assert torch.allclose(bright_image, torch.full_like(bright_image, expected_bright))
    assert torch.allclose(dark_image, torch.full_like(dark_image, expected_dark))
    assert (int(bright_label), int(bright_index)) == (7, 1)
    assert (int(dark_label), int(dark_index)) == (3, 0)


def test_official_test_validation_protocol_is_disjoint_and_balanced() -> None:
    train_labels = torch.arange(10).repeat_interleave(12)
    test_labels = torch.arange(10).repeat_interleave(4)
    config = TrainingConfig(
        train_size=120,
        validation_size=20,
        test_size=20,
        validation_source="official_test",
    )

    train_indices, validation_indices, test_indices = mnist_split_indices(
        train_labels,
        test_labels,
        config,
    )

    assert len(train_indices) == 120
    assert len(validation_indices) == 20
    assert len(test_indices) == 20
    assert set(validation_indices).isdisjoint(test_indices)
    assert torch.bincount(test_labels[list(validation_indices)], minlength=10).tolist() == [2] * 10
    assert torch.bincount(test_labels[list(test_indices)], minlength=10).tolist() == [2] * 10
