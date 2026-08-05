"""Routing heads, traces, and auxiliary routing objectives."""

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor


@dataclass(slots=True)
class RoutingTrace:
    """Compact routing metadata without recurrent hidden-state snapshots."""

    route_ids: Tensor
    route_probs: Tensor
    active_mask: Tensor
    exit_step: Tensor
    route_depth: Tensor
    forced_exit: Tensor
    executed_steps: int
    num_experts: int
    exit_route_index: int


def masked_mean(hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
    # hidden_states: (b, l, d); attention_mask: (b, l)
    if attention_mask is None:
        return hidden_states.mean(dim=1)  # (b, d)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (b, l, 1)
    token_sum = (hidden_states * mask).sum(dim=1)  # (b, d)
    token_count = mask.sum(dim=1).clamp_min(1.0)  # (b, 1)
    return token_sum / token_count  # (b, d)


class RouterHead(nn.Module):
    """An independent, pooled router associated with one source module."""

    def __init__(
        self,
        hidden_size: int,
        router_hidden_size: int,
        num_routes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, router_hidden_size)
        self.output_projection = nn.Linear(router_hidden_size, num_routes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        normalized_states = self.normalization(hidden_states)  # (b, l, d)
        pooled_states = masked_mean(normalized_states, attention_mask)  # (b, d)
        router_states = torch.nn.functional.gelu(self.input_projection(pooled_states))  # (b, d_router)
        router_states = self.dropout(router_states)  # (b, d_router)
        return self.output_projection(router_states)  # (b, e + 1)


def straight_through_one_hot(route_probs: Tensor) -> tuple[Tensor, Tensor]:
    """Return hard forward gates with soft probability gradients."""
    # route_probs: (b, e + 1)
    route_ids = route_probs.argmax(dim=-1)  # (b,)
    hard_gates = torch.nn.functional.one_hot(route_ids, route_probs.size(-1)).to(route_probs.dtype)  # (b, e + 1)
    gates = hard_gates + route_probs - route_probs.detach()  # (b, e + 1)
    return gates, route_ids


def sinkhorn_balanced_probabilities(
    router_logits: Tensor,
    iterations: int,
    temperature: float,
) -> Tensor:
    """Balance expert probability mass across a batch with Sinkhorn updates."""
    # router_logits: (b, e)
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape (batch, experts)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    log_assignments = router_logits / temperature  # (b, e)
    for _ in range(iterations):
        log_assignments = log_assignments - torch.logsumexp(
            log_assignments,
            dim=-1,
            keepdim=True,
        )  # (b, e)
        log_assignments = log_assignments - torch.logsumexp(
            log_assignments,
            dim=0,
            keepdim=True,
        )  # (b, e)
    return log_assignments.softmax(dim=-1)  # (b, e)


def capacity_balanced_routes(route_scores: Tensor) -> Tensor:
    """Assign top-1 routes under equal per-expert batch capacities."""
    # route_scores: (b, e)
    if route_scores.ndim != 2:
        raise ValueError("route_scores must have shape (batch, experts)")
    b, e = route_scores.shape
    if b == 0 or e == 0:
        raise ValueError("route_scores must have non-empty batch and expert dimensions")

    base_capacity, extra_capacity = divmod(b, e)
    remaining_capacity = [
        base_capacity + int(expert_index < extra_capacity)
        for expert_index in range(e)
    ]
    flat_preference_order = (
        route_scores.detach().flatten().argsort(descending=True).cpu().tolist()
    )
    assignments = [-1] * b
    assigned_count = 0
    for flat_index in flat_preference_order:
        sample_index, expert_index = divmod(flat_index, e)
        if assignments[sample_index] >= 0 or remaining_capacity[expert_index] == 0:
            continue
        assignments[sample_index] = expert_index
        remaining_capacity[expert_index] -= 1
        assigned_count += 1
        if assigned_count == b:
            break

    if assigned_count != b:
        raise RuntimeError("Capacity routing did not assign every sample")
    return torch.tensor(assignments, device=route_scores.device, dtype=torch.long)  # (b,)


def hierarchical_load_balancing_loss(
    trace: RoutingTrace,
    expert_families: tuple[str, ...],
    within_family_weight: float = 1.0,
) -> Tensor:
    """Balance hard and soft utilization across families and experts."""

    if len(expert_families) != trace.num_experts:
        raise ValueError("expert_families must contain one entry per expert")
    if within_family_weight < 0.0:
        raise ValueError("within_family_weight must be non-negative")

    # trace.route_probs: (s, b, e + 1); trace.route_ids: (s, b)
    selected_expert = trace.active_mask & trace.route_ids.ge(0) & trace.route_ids.lt(trace.num_experts)  # (s, b)
    selected_routes = trace.route_ids[selected_expert]  # (n_selected,)
    selected_probs = trace.route_probs[..., : trace.num_experts][selected_expert]  # (n_selected, e)
    if selected_routes.numel() == 0:
        return trace.route_probs.sum() * 0.0  # ()

    conditional_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (n_selected, e)
    family_names = tuple(dict.fromkeys(expert_families))
    family_ids = torch.tensor(
        [family_names.index(family) for family in expert_families],
        device=trace.route_probs.device,
        dtype=torch.long,
    )  # (e,)
    selected_family_ids = family_ids.index_select(0, selected_routes)  # (n_selected,)
    family_probs = torch.stack(
        [conditional_probs[:, family_ids.eq(family_id)].sum(dim=-1) for family_id in range(len(family_names))],
        dim=-1,
    )  # (n_selected, f)
    family_loss = _switch_balance(family_probs, selected_family_ids)  # ()

    within_family_losses: list[Tensor] = []
    represented_families: list[Tensor] = []
    for family_id in range(len(family_names)):
        family_expert_indices = torch.nonzero(family_ids.eq(family_id), as_tuple=False).flatten()  # (e_family,)
        family_selected = selected_family_ids.eq(family_id)  # (n_selected,)
        family_count = family_selected.sum()  # ()
        local_probs = conditional_probs.index_select(1, family_expert_indices)  # (n_selected, e_family)
        local_probs = local_probs / local_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (n_selected, e_family)
        selected_local_probs = local_probs * family_selected.unsqueeze(-1)  # (n_selected, e_family)
        soft_fraction = selected_local_probs.sum(dim=0) / family_count.clamp_min(1)  # (e_family,)
        hard_counts = selected_routes.unsqueeze(-1).eq(family_expert_indices).sum(dim=0)  # (e_family,)
        hard_fraction = hard_counts.to(conditional_probs.dtype) / family_count.clamp_min(1)  # (e_family,)
        family_size = family_expert_indices.numel()
        within_family_losses.append(family_size * torch.sum(hard_fraction * soft_fraction))
        represented_families.append(family_count.gt(0).to(conditional_probs.dtype))

    if within_family_weight == 0.0:
        return family_loss  # ()
    family_presence = torch.stack(represented_families)  # (f,)
    mean_within_family_loss = (
        torch.stack(within_family_losses) * family_presence
    ).sum() / family_presence.sum().clamp_min(1.0)  # ()
    return (
        family_loss + within_family_weight * mean_within_family_loss
    ) / (1.0 + within_family_weight)  # ()


def _switch_balance(conditional_probs: Tensor, hard_routes: Tensor) -> Tensor:
    # conditional_probs: (n, r); hard_routes: (n,)
    num_routes = conditional_probs.size(-1)
    hard_assignments = torch.nn.functional.one_hot(hard_routes, num_routes).to(conditional_probs.dtype)  # (n, r)
    hard_fraction = hard_assignments.mean(dim=0)  # (r,)
    soft_fraction = conditional_probs.mean(dim=0)  # (r,)
    return num_routes * torch.sum(hard_fraction * soft_fraction)  # ()


def routing_mutual_information(trace: RoutingTrace) -> Tensor:
    """Measure confident, input-dependent expert choices at each recurrent step."""

    # trace.route_probs: (s, b, e + 1); trace.route_ids: (s, b)
    step_information: list[Tensor] = []
    selected_counts: list[int] = []
    for step_index in range(trace.executed_steps):
        selected_expert = (
            trace.active_mask[step_index]
            & trace.route_ids[step_index].ge(0)
            & trace.route_ids[step_index].lt(trace.num_experts)
        )  # (b,)
        step_probs = trace.route_probs[step_index, selected_expert, : trace.num_experts]  # (n_selected, e)
        if step_probs.size(0) < 2:
            continue

        conditional_probs = step_probs / step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (n_selected, e)
        per_example_entropy = _entropy(conditional_probs).mean()  # ()
        marginal_probs = conditional_probs.mean(dim=0)  # (e,)
        marginal_entropy = _entropy(marginal_probs)  # ()
        step_information.append(marginal_entropy - per_example_entropy)
        selected_counts.append(step_probs.size(0))

    if not step_information:
        return trace.route_probs.sum() * 0.0  # ()
    information_by_step = torch.stack(step_information)  # (n_steps,)
    step_weights = torch.tensor(
        selected_counts,
        device=trace.route_probs.device,
        dtype=trace.route_probs.dtype,
    )  # (n_steps,)
    return torch.sum(information_by_step * step_weights) / step_weights.sum()  # ()


def _entropy(probabilities: Tensor) -> Tensor:
    # probabilities: (..., e)
    safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)  # (..., e)
    return -(probabilities * safe_probabilities.log()).sum(dim=-1)  # (...,)


def compute_penalty(trace: RoutingTrace) -> Tensor:
    """Penalize differentiable probability mass assigned to continued computation."""
    # trace.route_probs: (s, b, e + 1); trace.active_mask: (s, b)
    continue_probs = 1.0 - trace.route_probs[..., trace.exit_route_index]  # (s, b)
    active_continue_probs = continue_probs[trace.active_mask]  # (n_active,)
    if active_continue_probs.numel() == 0:
        return trace.route_probs.sum() * 0.0  # ()
    return active_continue_probs.mean()  # ()
