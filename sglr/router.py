"""Routing heads, traces, and auxiliary routing objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
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


def load_balancing_loss(trace: RoutingTrace) -> Tensor:
    """Balance expert probabilities while leaving termination unconstrained."""
    # trace.route_probs: (s, b, e + 1); trace.active_mask: (s, b)
    expert_probs = trace.route_probs[..., : trace.num_experts]  # (s, b, e)
    active_expert_probs = expert_probs[trace.active_mask]  # (n_active, e)
    if active_expert_probs.numel() == 0:
        return trace.route_probs.sum() * 0.0  # ()

    expert_probability_mass = active_expert_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (n_active, 1)
    conditional_probs = active_expert_probs / expert_probability_mass  # (n_active, e)
    mean_expert_probs = conditional_probs.mean(dim=0)  # (e,)
    uniform_target = torch.full_like(mean_expert_probs, 1.0 / trace.num_experts)  # (e,)
    return torch.mean((mean_expert_probs - uniform_target) ** 2)  # ()


def compute_penalty(trace: RoutingTrace) -> Tensor:
    """Penalize differentiable probability mass assigned to continued computation."""
    # trace.route_probs: (s, b, e + 1); trace.active_mask: (s, b)
    continue_probs = 1.0 - trace.route_probs[..., trace.exit_route_index]  # (s, b)
    active_continue_probs = continue_probs[trace.active_mask]  # (n_active,)
    if active_continue_probs.numel() == 0:
        return trace.route_probs.sum() * 0.0  # ()
    return active_continue_probs.mean()  # ()
