from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from collections.abc import Callable
from dataclasses import replace

from sglr.config import ExpertSpec, ModelConfig
from sglr.experts import ExpertDelta, build_expert
from sglr.model import MNISTSGLR, SGLRCore, build_mnist_model, count_parameters
from sglr.presets.mnist import get_mnist_preset
from sglr.router import (
    RouterHead,
    RoutingTrace,
    capacity_balanced_routes,
    hierarchical_load_balancing_loss,
    masked_mean,
    routing_mutual_information,
    sinkhorn_balanced_probabilities,
)


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


def routing_trace(route_ids: torch.Tensor, route_probs: torch.Tensor) -> RoutingTrace:
    # route_ids: (s, b); route_probs: (s, b, e + 1)
    s, b = route_ids.shape
    num_experts = route_probs.size(-1) - 1
    return RoutingTrace(
        route_ids=route_ids,
        route_probs=route_probs,
        active_mask=route_ids.ge(0),
        exit_step=torch.full((b,), s, dtype=torch.long),
        route_depth=route_ids.lt(num_experts).sum(dim=0),
        forced_exit=torch.zeros(b, dtype=torch.bool),
        executed_steps=s,
        num_experts=num_experts,
        exit_route_index=num_experts,
    )


def test_hierarchical_switch_loss_uses_hard_assignments_and_excludes_exit() -> None:
    expert_families = ("mlp", "mlp", "conv", "conv", "attention", "attention")
    balanced_routes = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    balanced_probs = nn.functional.one_hot(balanced_routes, 7).to(torch.float32)
    balanced = routing_trace(balanced_routes, balanced_probs)

    collapsed_routes = torch.tensor([[0, 0, 0, 0, 0, 0]])
    collapsed_probs = nn.functional.one_hot(collapsed_routes, 7).to(torch.float32)
    collapsed = routing_trace(collapsed_routes, collapsed_probs)

    balanced_loss = hierarchical_load_balancing_loss(balanced, expert_families)
    collapsed_loss = hierarchical_load_balancing_loss(collapsed, expert_families)

    assert torch.allclose(balanced_loss, torch.tensor(1.0))
    assert collapsed_loss > balanced_loss


def test_hierarchical_switch_loss_reaches_soft_router_probabilities() -> None:
    route_ids = torch.tensor([[0, 0, 2, 2]])
    expert_probs = torch.tensor(
        [
            [
                [0.70, 0.10, 0.10, 0.10],
                [0.60, 0.10, 0.20, 0.10],
                [0.20, 0.10, 0.60, 0.10],
                [0.10, 0.10, 0.70, 0.10],
            ]
        ],
        requires_grad=True,
    )
    trace = routing_trace(route_ids, expert_probs)

    loss = hierarchical_load_balancing_loss(trace, ("mlp", "mlp", "conv"))
    loss.backward()

    assert expert_probs.grad is not None
    assert torch.count_nonzero(expert_probs.grad) > 0


def test_sinkhorn_probabilities_have_balanced_column_mass_and_gradients() -> None:
    torch.manual_seed(13)
    router_logits = torch.randn(96, 24, requires_grad=True)  # (b, e)

    route_probs = sinkhorn_balanced_probabilities(
        router_logits,
        iterations=100,
        temperature=0.05,
    )  # (b, e)
    column_mass = route_probs.sum(dim=0)  # (e,)
    route_probs.square().sum().backward()

    assert torch.allclose(route_probs.sum(dim=-1), torch.ones(96), atol=1e-6)
    assert torch.allclose(column_mass, torch.full((24,), 4.0), atol=0.1)
    assert router_logits.grad is not None
    assert torch.count_nonzero(router_logits.grad) > 0


def test_capacity_balanced_routes_use_equal_expert_quotas() -> None:
    torch.manual_seed(19)
    route_scores = torch.randn(101, 24)  # (b, e)

    route_ids = capacity_balanced_routes(route_scores)  # (b,)
    route_counts = torch.bincount(route_ids, minlength=24)  # (e,)

    assert route_ids.shape == (101,)
    assert int(route_counts.min()) == 4
    assert int(route_counts.max()) == 5


def test_route_mutual_information_rewards_different_orders_between_examples() -> None:
    diverse_routes = torch.tensor([[0, 1], [1, 0]])
    diverse_probs = nn.functional.one_hot(diverse_routes, 3).to(torch.float32)
    same_routes = torch.tensor([[0, 0], [1, 1]])
    same_probs = nn.functional.one_hot(same_routes, 3).to(torch.float32)

    diverse_information = routing_mutual_information(routing_trace(diverse_routes, diverse_probs))
    same_information = routing_mutual_information(routing_trace(same_routes, same_probs))

    assert diverse_information > 0.0
    assert torch.allclose(same_information, torch.tensor(0.0))


def test_route_mutual_information_reaches_soft_router_probabilities() -> None:
    route_ids = torch.tensor([[0, 1], [1, 0]])
    route_probs = torch.tensor(
        [
            [[0.80, 0.15, 0.05], [0.20, 0.75, 0.05]],
            [[0.25, 0.70, 0.05], [0.85, 0.10, 0.05]],
        ],
        requires_grad=True,
    )
    information = routing_mutual_information(routing_trace(route_ids, route_probs))

    information.backward()

    assert route_probs.grad is not None
    assert torch.count_nonzero(route_probs.grad) > 0


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


def test_sparse_expert_update_applies_configured_residual_scale() -> None:
    config = replace(
        small_config("hard_argmax"),
        max_steps=1,
        expert_residual_scale=0.25,
    )
    core = SGLRCore(config).eval()
    with torch.no_grad():
        initial_router = core.routers[core.initial_router_index]
        initial_router.input_projection.weight.zero_()
        initial_router.input_projection.bias.zero_()
        initial_router.output_projection.weight.zero_()
        initial_router.output_projection.bias.copy_(
            torch.tensor([10.0, -10.0, -10.0, -10.0])  # (e + 1,)
        )
        hidden_states = torch.randn(3, 6, config.hidden_size)  # (b, l, d)
        expert_delta = core.experts[0](hidden_states)  # (b, l, d)
        output = core(hidden_states)

    expected_states = hidden_states + 0.25 * expert_delta  # (b, l, d)
    assert torch.allclose(output.final_hidden_states, expected_states)


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


def test_shared_router_uses_one_head_for_all_recurrent_sources() -> None:
    config = replace(
        small_config("hard_argmax"),
        share_router_across_sources=True,
    )
    core = SGLRCore(config)
    hidden_states = torch.randn(4, 6, config.hidden_size)  # (b, l, d)

    output = core(hidden_states)

    assert len(core.routers) == 1
    assert core.initial_router_index == 0
    assert output.trace.num_experts == 3


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


def test_evaluation_dispatches_only_the_selected_expert() -> None:
    config = small_config("hard_argmax")
    core = SGLRCore(replace(config, max_steps=2)).eval()
    with torch.no_grad():
        for router in core.routers:
            router.input_projection.weight.zero_()
            router.input_projection.bias.zero_()
            router.output_projection.weight.zero_()
            router.output_projection.bias.zero_()
        core.routers[core.initial_router_index].output_projection.bias.copy_(
            torch.tensor([10.0, -10.0, -10.0, -10.0])  # (e + 1,)
        )
        core.routers[0].output_projection.bias.copy_(
            torch.tensor([-10.0, -10.0, -10.0, 10.0])  # (e + 1,)
        )

    call_sizes: list[list[int]] = [[] for _ in core.experts]

    def record_call(
        expert_index: int,
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            hidden_states = inputs[0]  # (n_selected, l, d)
            call_sizes[expert_index].append(hidden_states.size(0))

        return hook

    handles = [
        expert.register_forward_hook(record_call(expert_index))
        for expert_index, expert in enumerate(core.experts)
    ]
    hidden_states = torch.randn(5, 6, config.hidden_size)  # (b, l, d)
    try:
        output = core(hidden_states)
    finally:
        for handle in handles:
            handle.remove()

    assert call_sizes == [[5], [], []]
    assert output.trace.route_depth.tolist() == [1, 1, 1, 1, 1]
    assert output.trace.forced_exit.tolist() == [False, False, False, False, False]


def test_primary_and_fixed_depth_parameter_budgets() -> None:
    primary_config = get_mnist_preset("pilot").model
    primary_model = MNISTSGLR(primary_config)
    primary_count = count_parameters(primary_model, trainable_only=True)
    router_count = count_parameters(primary_model.core.routers, trainable_only=True)
    expert_count = count_parameters(primary_model.core.experts, trainable_only=True)

    fixed_model = build_mnist_model(replace(primary_config, routing_mode="fixed_depth"))
    fixed_count = count_parameters(fixed_model, trainable_only=True)

    assert primary_count < 200_000
    assert router_count < expert_count
    assert abs(fixed_count - primary_count) / primary_count <= 0.02
    assert fixed_model.specs[-1].hidden_size == 360


def test_fast_full_encoder_and_readout_preserve_output_contract() -> None:
    config = get_mnist_preset("fast_cnn_full").model
    model = MNISTSGLR(config).eval()
    images = torch.randn(3, 1, 28, 28)  # (b, 1, 28, 28)

    with torch.inference_mode():
        output = model(images)

    assert output.final_hidden_states.shape == (3, 49, config.hidden_size)
    assert output.logits.shape == (3, 10)
