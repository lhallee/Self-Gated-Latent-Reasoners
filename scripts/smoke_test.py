"""Fast synthetic CPU smoke test for routing and optimization."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from sglr.config import ExpertSpec, ModelConfig
from sglr.model import MNISTSGLR, count_parameters
from sglr.router import compute_penalty, load_balancing_loss


def smoke_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=16,
        max_steps=2,
        min_steps=1,
        router_hidden_size=8,
        experts=(
            ExpertSpec("mlp", "mlp", hidden_size=8),
            ExpertSpec("conv", "conv", channels=2, kernel_size=(3, 3)),
            ExpertSpec("attention", "attention", internal_size=8, num_heads=1),
        ),
    )


def main() -> None:
    start = time.perf_counter()
    torch.manual_seed(7)
    model = MNISTSGLR(smoke_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    router_received_gradient = False

    for _ in range(2):
        images = torch.randn(8, 1, 28, 28)  # (b, 1, 28, 28)
        labels = torch.randint(0, 10, (8,))  # (b,)
        output = model(images)
        loss = (
            F.cross_entropy(output.logits, labels)
            + 0.01 * load_balancing_loss(output.trace)
            + 0.001 * compute_penalty(output.trace)
        )  # ()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        router_received_gradient |= any(
            parameter.grad is not None and bool(parameter.grad.abs().sum().item())
            for parameter in model.core.routers.parameters()
        )
        optimizer.step()

    if output.logits.shape != (8, 10):
        raise AssertionError("Smoke model returned an unexpected logit shape")
    if not router_received_gradient:
        raise AssertionError("Classification and auxiliary losses did not reach a router")
    if count_parameters(model) >= 20_000:
        raise AssertionError("Synthetic smoke model is unexpectedly large")
    elapsed = time.perf_counter() - start
    if elapsed >= 60.0:
        raise AssertionError(f"Synthetic CPU smoke exceeded 60 seconds: {elapsed:.1f}s")
    print(f"Synthetic smoke passed in {elapsed:.2f}s with {count_parameters(model):,} parameters")


if __name__ == "__main__":
    main()
