from __future__ import annotations

import torch

from sglr.data import stratified_split_indices


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
