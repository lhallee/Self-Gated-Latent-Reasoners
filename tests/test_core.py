from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
import torch.nn as nn

from sglr.config import ExpertSpec, ModelConfig, load_experiment_config
from sglr.experts import ExpertDelta, build_expert
from sglr.model import MNISTSGLR, SGLRCore, build_mnist_model, count_parameters
from sglr.router import RouterHead, masked_mean


def small_config(routing_mode: str = "straight_through") -> ModelConfig:
    return ModelConfig(
        hidden_size=16,
        max_steps=3,
        min_steps=1,
        router_hidden_size=8,
        routing_mode=routing_mode,
        experts=(
            ExpertSpec("mlp", "mlp", hidden_size=8),
            ExpertSpec("conv", "conv", channels=2, kernel_size=(3, 3)),
            ExpertSpec("attention", "attention", internal_size=8, num_heads=1),
        ),
    )


@pytest.mark.parametrize(
    "spec",
    [
        ExpertSpec("mlp", "mlp", hidden_size=8),
        ExpertSpec("conv", "conv", channels=2, kernel_size=(3, 5)),
        ExpertSpec("attention", "attention", internal_size=8, num_heads=2),
    ],
)
def test_experts_preserve_shape_and_mask(spec: ExpertSpec) -> None:
    expert = build_expert(spec, hidden_size=16)
    hidden_states = torch.randn(3, 7, 16)
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )

    candidate = expert(hidden_states, attention_mask)

    assert candidate.shape == hidden_states.shape
    assert torch.count_nonzero(candidate[~attention_mask]) == 0


def test_zero_delta_expert_is_exact_identity() -> None:
    class ZeroDelta(nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.zeros_like(hidden_states)

    expert = ExpertDelta(hidden_size=8, expert=ZeroDelta())
    hidden_states = torch.randn(2, 5, 8)

    assert torch.equal(hidden_states + expert(hidden_states), hidden_states)


def test_convolution_mask_prevents_padded_values_from_leaking() -> None:
    torch.manual_seed(3)
    expert = build_expert(ExpertSpec("conv", "conv", channels=2, kernel_size=(3, 3)), 8)
    attention_mask = torch.tensor([[True, True, True, False, False]])
    first = torch.randn(1, 5, 8)
    second = first.clone()
    second[:, 3:] = 1000.0

    first_delta = expert(first, attention_mask)
    second_delta = expert(second, attention_mask)

    assert torch.allclose(first_delta[:, :3], second_delta[:, :3])


def test_router_and_pooling_ignore_masked_tokens() -> None:
    torch.manual_seed(4)
    router = RouterHead(hidden_size=8, router_hidden_size=4, num_routes=3, dropout=0.0)
    attention_mask = torch.tensor([[True, True, False, False]])
    first = torch.randn(1, 4, 8)
    second = first.clone()
    second[:, 2:] = -500.0

    assert torch.allclose(masked_mean(first, attention_mask), masked_mean(second, attention_mask))
    assert torch.allclose(router(first, attention_mask), router(second, attention_mask))


def test_straight_through_classification_loss_reaches_routers() -> None:
    torch.manual_seed(7)
    model = MNISTSGLR(small_config("straight_through"))
    model.train()
    output = model(torch.randn(4, 1, 28, 28))
    loss = nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 3]))

    loss.backward()

    router_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.core.routers.parameters()
        if parameter.grad is not None
    )
    assert router_gradient > 0.0
    downstream_gradient = sum(
        float(parameter.grad.abs().sum())
        for router in model.core.routers[:-1]
        for parameter in router.parameters()
        if parameter.grad is not None
    )
    assert downstream_gradient > 0.0


def test_hard_argmax_classification_loss_does_not_reach_routers() -> None:
    torch.manual_seed(7)
    model = MNISTSGLR(small_config("hard_argmax"))
    model.train()
    output = model(torch.randn(4, 1, 28, 28))
    loss = nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 3]))

    loss.backward()

    assert all(parameter.grad is None for parameter in model.core.routers.parameters())


def test_minimum_depth_then_natural_exit() -> None:
    config = ModelConfig(
        hidden_size=8,
        max_steps=3,
        min_steps=1,
        router_hidden_size=4,
        experts=(ExpertSpec("only", "mlp", hidden_size=4),),
    )
    core = SGLRCore(config)
    with torch.no_grad():
        for router in core.routers:
            router.input_projection.weight.zero_()
            router.input_projection.bias.zero_()
            router.output_projection.weight.zero_()
        core.routers[core.initial_router_index].output_projection.bias.copy_(torch.tensor([10.0, -10.0]))
        core.routers[0].output_projection.bias.copy_(torch.tensor([-10.0, 10.0]))

    output = core(torch.randn(2, 6, 8))

    assert output.trace.route_ids[:, 0].tolist() == [0, 1]
    assert output.trace.route_depth.tolist() == [1, 1]
    assert output.trace.forced_exit.tolist() == [False, False]


def test_self_routes_are_allowed_and_horizon_is_forced() -> None:
    config = ModelConfig(
        hidden_size=8,
        max_steps=3,
        min_steps=1,
        router_hidden_size=4,
        experts=(ExpertSpec("only", "mlp", hidden_size=4),),
    )
    core = SGLRCore(config)
    with torch.no_grad():
        for router in core.routers:
            router.input_projection.weight.zero_()
            router.input_projection.bias.zero_()
            router.output_projection.weight.zero_()
            router.output_projection.bias.copy_(torch.tensor([10.0, -10.0]))

    output = core(torch.randn(2, 6, 8))

    assert output.trace.route_ids[:, 0].tolist() == [0, 0, 0]
    assert output.trace.route_depth.tolist() == [3, 3]
    assert output.trace.forced_exit.tolist() == [True, True]


def test_straight_through_and_sparse_forward_states_match() -> None:
    torch.manual_seed(11)
    core = SGLRCore(small_config("straight_through"))
    hidden_states = torch.randn(3, 6, 16)
    attention_mask = torch.tensor(
        [[True] * 6, [True, True, True, True, False, False], [True] * 6]
    )

    core.train()
    dense_output = core(hidden_states, attention_mask)
    core.eval()
    sparse_output = core(hidden_states, attention_mask)

    assert torch.equal(dense_output.trace.route_ids, sparse_output.trace.route_ids)
    assert torch.allclose(
        dense_output.final_hidden_states,
        sparse_output.final_hidden_states,
        atol=1e-6,
        rtol=1e-6,
    )


def test_primary_and_fixed_depth_parameter_budgets() -> None:
    primary_config = load_experiment_config("configs/mnist/pilot.toml").model
    primary_model = MNISTSGLR(primary_config)
    primary_count = count_parameters(primary_model, trainable_only=True)
    router_count = count_parameters(primary_model.core.routers, trainable_only=True)
    expert_count = count_parameters(primary_model.core.experts, trainable_only=True)

    fixed_values = asdict(primary_config)
    fixed_values["routing_mode"] = "fixed_depth"
    fixed_model = build_mnist_model(ModelConfig.from_dict(fixed_values))
    fixed_count = count_parameters(fixed_model, trainable_only=True)

    assert primary_count < 200_000
    assert router_count < expert_count
    assert abs(fixed_count - primary_count) / primary_count <= 0.02
    assert fixed_model.specs[-1].hidden_size == 360
