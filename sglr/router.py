from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GroupedRoutingTrace:
    route_ids: torch.Tensor
    route_probs: torch.Tensor
    active_mask: torch.Tensor
    pre_step_latents: torch.Tensor
    post_step_latents: torch.Tensor
    exit_step: torch.Tensor
    final_latent: torch.Tensor
    executed_steps: int
    num_experts: int
    exit_route_index: int


class GroupedTop1Router(nn.Module):
    def __init__(self, num_experts: int, max_steps: int, min_steps_before_exit: int = 1) -> None:
        super().__init__()
        assert num_experts > 0, "GroupedTop1Router requires at least one expert"
        assert max_steps > 0, "max_steps must be positive"
        assert min_steps_before_exit >= 0, "min_steps_before_exit must be non-negative"
        self.num_experts = num_experts
        self.max_steps = max_steps
        self.min_steps_before_exit = min_steps_before_exit
        self.exit_route_index = num_experts

    def forward(
        self,
        latents: torch.Tensor,
        initial_logits: torch.Tensor,
        experts: nn.ModuleList,
        route_heads: nn.ModuleList,
    ) -> GroupedRoutingTrace:
        assert latents.ndim == 2, "Router expects latent vectors shaped as [batch, hidden]"
        assert len(experts) == self.num_experts, "Experts list must match router expert count"
        assert len(route_heads) == self.num_experts, "Route heads must match router expert count"
        assert initial_logits.ndim == 2 and initial_logits.size(0) == latents.size(0), "Initial router logits must align with the batch"
        assert initial_logits.size(1) == self.num_experts + 1, "Initial router must predict one logit per expert plus exit"

        batch_size, hidden_size = latents.shape
        device = latents.device
        route_probs = torch.zeros(self.max_steps, batch_size, self.num_experts + 1, device=device, dtype=latents.dtype)
        route_ids = torch.full((self.max_steps, batch_size), fill_value=-1, device=device, dtype=torch.long)
        active_mask = torch.zeros(self.max_steps, batch_size, device=device, dtype=torch.bool)
        pre_step_latents = torch.zeros(self.max_steps, batch_size, hidden_size, device=device, dtype=latents.dtype)
        post_step_latents = torch.zeros(self.max_steps, batch_size, hidden_size, device=device, dtype=latents.dtype)
        exit_step = torch.full((batch_size,), fill_value=-1, device=device, dtype=torch.long)

        current_latents = latents
        current_probs = initial_logits.softmax(dim=-1)
        sample_is_active = torch.ones(batch_size, device=device, dtype=torch.bool)
        executed_steps = 0

        for step_index in range(self.max_steps):
            active_indices = torch.nonzero(sample_is_active, as_tuple=False).flatten()
            if active_indices.numel() == 0:
                break

            pre_step_latents[step_index] = current_latents
            active_probabilities = current_probs.index_select(0, active_indices)
            selection_probabilities = active_probabilities.clone()
            if step_index < self.min_steps_before_exit:
                selection_probabilities[:, self.exit_route_index] = -1.0

            selected_routes = selection_probabilities.argmax(dim=-1)
            route_probs[step_index].index_copy_(0, active_indices, active_probabilities)
            route_ids[step_index].index_copy_(0, active_indices, selected_routes)
            active_mask[step_index].index_fill_(0, active_indices, True)

            next_latents = current_latents.clone()
            next_probs = current_probs.clone()

            exit_selection_mask = selected_routes == self.exit_route_index
            if exit_selection_mask.any():
                exit_indices = active_indices.index_select(0, torch.nonzero(exit_selection_mask, as_tuple=False).flatten())
                exit_step.index_fill_(0, exit_indices, step_index)
                sample_is_active.index_fill_(0, exit_indices, False)
                next_probs.index_fill_(0, exit_indices, 0.0)

            for expert_index in range(self.num_experts):
                expert_selection_mask = selected_routes == expert_index
                if not expert_selection_mask.any():
                    continue

                expert_batch_positions = torch.nonzero(expert_selection_mask, as_tuple=False).flatten()
                expert_sample_indices = active_indices.index_select(0, expert_batch_positions)
                expert_inputs = current_latents.index_select(0, expert_sample_indices)
                expert_outputs = experts[expert_index](expert_inputs)
                expert_route_logits = route_heads[expert_index](expert_outputs)
                expert_route_probs = expert_route_logits.softmax(dim=-1)
                next_latents.index_copy_(0, expert_sample_indices, expert_outputs)
                next_probs.index_copy_(0, expert_sample_indices, expert_route_probs)

            post_step_latents[step_index] = next_latents
            current_latents = next_latents
            current_probs = next_probs
            executed_steps = step_index + 1

        unfinished_mask = exit_step == -1
        if unfinished_mask.any():
            exit_step[unfinished_mask] = executed_steps

        return GroupedRoutingTrace(
            route_ids=route_ids,
            route_probs=route_probs,
            active_mask=active_mask,
            pre_step_latents=pre_step_latents,
            post_step_latents=post_step_latents,
            exit_step=exit_step,
            final_latent=current_latents,
            executed_steps=executed_steps,
            num_experts=self.num_experts,
            exit_route_index=self.exit_route_index,
        )


def load_balancing_loss(route_probs: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    assert route_probs.ndim == 3, "route_probs must have shape [steps, batch, routes]"
    assert active_mask.ndim == 2, "active_mask must have shape [steps, batch]"
    assert route_probs.size(0) == active_mask.size(0), "route_probs and active_mask must align on steps"
    assert route_probs.size(1) == active_mask.size(1), "route_probs and active_mask must align on batch size"

    per_step_losses: list[torch.Tensor] = []
    for step_index in range(route_probs.size(0)):
        current_active_mask = active_mask[step_index]
        if not current_active_mask.any():
            continue
        step_probs = route_probs[step_index][current_active_mask]
        mean_probs = step_probs.mean(dim=0)
        target_distribution = torch.full_like(mean_probs, fill_value=1.0 / mean_probs.numel())
        per_step_losses.append(F.mse_loss(mean_probs, target_distribution))

    if per_step_losses:
        return torch.stack(per_step_losses).mean()

    return route_probs.sum() * 0.0
