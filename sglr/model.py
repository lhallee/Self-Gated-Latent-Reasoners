"""SGLR recurrence core and MNIST classifier wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor

from sglr.config import ExpertSpec, ModelConfig
from sglr.experts import build_expert
from sglr.router import RouterHead, RoutingTrace, masked_mean, straight_through_one_hot


@dataclass(slots=True)
class SGLRCoreOutput:
    final_hidden_states: Tensor
    trace: RoutingTrace


@dataclass(slots=True)
class MNISTOutput:
    logits: Tensor
    final_hidden_states: Tensor
    trace: RoutingTrace


class SGLRCore(nn.Module):
    """Route Transformer hidden states through shape-preserving experts."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.routing_mode == "fixed_depth":
            raise ValueError("Use FixedDepthMNISTModel for the fixed_depth variant")

        self.config = config
        self.experts = nn.ModuleList(
            [build_expert(spec, config.hidden_size) for spec in config.experts]
        )
        self.expert_names = [spec.name for spec in config.experts]
        self.routers = nn.ModuleList(
            [
                RouterHead(
                    hidden_size=config.hidden_size,
                    router_hidden_size=config.router_hidden_size,
                    num_routes=config.num_routes,
                    dropout=config.router_dropout,
                )
                for _ in range(config.num_experts + 1)
            ]
        )
        self.initial_router_index = config.num_experts

        if config.routing_mode == "frozen_random":
            for parameter in self.routers.parameters():
                parameter.requires_grad = False

    @property
    def exit_route_index(self) -> int:
        return self.config.num_experts

    def route_name(self, route_id: int) -> str:
        if route_id == self.exit_route_index:
            return "exit"
        if not 0 <= route_id < self.config.num_experts:
            raise ValueError(f"Unknown route ID: {route_id}")
        return self.expert_names[route_id]

    def _source_router_logits(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        source_ids: Tensor,
        source_gates: Tensor | None = None,
    ) -> Tensor:
        # hidden_states: (b, l, d); source_ids: (b,); source_gates: (b, e + 1)
        logits_by_source = torch.stack(
            [router(hidden_states, attention_mask) for router in self.routers],
            dim=1,
        )  # (b, e + 1, e + 1)
        if source_gates is not None:
            return torch.einsum("bs,bsr->br", source_gates, logits_by_source)  # (b, e + 1)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)  # (b,)
        return logits_by_source[batch_indices, source_ids]  # (b, e + 1)

    def _dense_candidates(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        expert_candidates = [
            hidden_states + expert(hidden_states, attention_mask) for expert in self.experts
        ]  # e * (b, l, d)
        expert_candidates.append(hidden_states)  # exit candidate: (b, l, d)
        return torch.stack(expert_candidates, dim=1)  # (b, e + 1, l, d)

    def _sparse_update(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        selected_routes: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        # hidden_states: (b, l, d); selected_routes: (b,); active_mask: (b,)
        next_hidden_states = hidden_states  # (b, l, d)
        for expert_index, expert in enumerate(self.experts):
            selected_mask = active_mask & selected_routes.eq(expert_index)  # (b,)
            selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()  # (n_selected,)
            if selected_indices.numel() == 0:
                continue

            selected_states = hidden_states.index_select(0, selected_indices)  # (n_selected, l, d)
            selected_attention_mask = (
                None if attention_mask is None else attention_mask.index_select(0, selected_indices)
            )  # (n_selected, l)
            selected_candidates = selected_states + expert(
                selected_states,
                selected_attention_mask,
            )  # (n_selected, l, d)
            next_hidden_states = next_hidden_states.index_copy(
                0,
                selected_indices,
                selected_candidates,
            )  # (b, l, d)
        return next_hidden_states  # (b, l, d)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> SGLRCoreOutput:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        if hidden_states.ndim != 3:
            raise ValueError("SGLRCore expects hidden states shaped as (batch, length, hidden)")
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError("Hidden-state width does not match ModelConfig.hidden_size")
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape (batch, length)")
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)  # (b, l)

        b = hidden_states.size(0)
        device = hidden_states.device
        current_states = hidden_states  # (b, l, d)
        sample_is_active = torch.ones(b, device=device, dtype=torch.bool)  # (b,)
        source_ids = torch.full(
            (b,),
            self.initial_router_index,
            device=device,
            dtype=torch.long,
        )  # (b,)
        source_gates = torch.nn.functional.one_hot(
            source_ids,
            self.config.num_experts + 1,
        ).to(hidden_states.dtype)  # (b, e + 1)
        route_depth = torch.zeros(b, device=device, dtype=torch.long)  # (b,)
        exit_step = torch.full((b,), -1, device=device, dtype=torch.long)  # (b,)

        route_id_steps: list[Tensor] = []
        route_probability_steps: list[Tensor] = []
        active_mask_steps: list[Tensor] = []

        for step_index in range(self.config.max_steps):
            if not sample_is_active.any():
                break

            router_logits = self._source_router_logits(
                current_states,
                attention_mask,
                source_ids,
                source_gates if self.training and self.config.routing_mode == "straight_through" else None,
            )  # (b, e + 1)
            if step_index < self.config.min_steps:
                exit_mask = torch.zeros_like(router_logits, dtype=torch.bool)  # (b, e + 1)
                exit_mask[:, self.exit_route_index] = True
                router_logits = router_logits.masked_fill(exit_mask, torch.finfo(router_logits.dtype).min)  # (b, e + 1)

            route_probs = router_logits.softmax(dim=-1)  # (b, e + 1)
            hard_routes = route_probs.argmax(dim=-1)  # (b,)
            selected_routes = torch.where(
                sample_is_active,
                hard_routes,
                torch.full_like(hard_routes, self.exit_route_index),
            )  # (b,)
            recorded_routes = torch.where(
                sample_is_active,
                selected_routes,
                torch.full_like(selected_routes, -1),
            )  # (b,)
            recorded_probs = torch.where(
                sample_is_active.unsqueeze(-1),
                route_probs,
                torch.zeros_like(route_probs),
            )  # (b, e + 1)
            route_id_steps.append(recorded_routes)
            route_probability_steps.append(recorded_probs)
            active_mask_steps.append(sample_is_active)

            use_straight_through = self.training and self.config.routing_mode == "straight_through"
            if use_straight_through:
                gates, _ = straight_through_one_hot(route_probs)  # (b, e + 1)
                active_indices = torch.nonzero(sample_is_active, as_tuple=False).flatten()  # (n_active,)
                active_states = current_states.index_select(0, active_indices)  # (n_active, l, d)
                active_attention_mask = (
                    None if attention_mask is None else attention_mask.index_select(0, active_indices)
                )  # (n_active, l)
                candidates = self._dense_candidates(
                    active_states,
                    active_attention_mask,
                )  # (n_active, e + 1, l, d)
                active_gates = gates.index_select(0, active_indices)  # (n_active, e + 1)
                mixed_states = torch.einsum("br,brld->bld", active_gates, candidates)  # (n_active, l, d)
                next_states = current_states.index_copy(0, active_indices, mixed_states)  # (b, l, d)
            else:
                next_states = self._sparse_update(
                    current_states,
                    attention_mask,
                    selected_routes,
                    sample_is_active,
                )  # (b, l, d)

            selected_expert = sample_is_active & selected_routes.lt(self.config.num_experts)  # (b,)
            route_depth = route_depth + selected_expert.to(torch.long)  # (b,)
            selected_exit = sample_is_active & selected_routes.eq(self.exit_route_index)  # (b,)
            exit_step = torch.where(selected_exit & exit_step.lt(0), route_depth, exit_step)  # (b,)
            source_ids = torch.where(selected_expert, selected_routes, source_ids)  # (b,)
            if use_straight_through:
                expert_probs = route_probs[:, : self.config.num_experts]  # (b, e)
                conditional_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                hard_expert_gates = torch.nn.functional.one_hot(
                    selected_routes.clamp_max(self.config.num_experts - 1),
                    self.config.num_experts,
                ).to(route_probs.dtype)  # (b, e)
                expert_source_gates = hard_expert_gates + conditional_probs - conditional_probs.detach()
                source_gates = torch.cat(
                    [expert_source_gates, torch.zeros_like(expert_source_gates[:, :1])],
                    dim=-1,
                )  # (b, e + 1)
            sample_is_active = sample_is_active & ~selected_exit  # (b,)
            current_states = next_states  # (b, l, d)

        forced_exit = sample_is_active  # (b,)
        exit_step = torch.where(forced_exit, route_depth, exit_step)  # (b,)
        route_ids = torch.stack(route_id_steps, dim=0)  # (s, b)
        route_probs = torch.stack(route_probability_steps, dim=0)  # (s, b, e + 1)
        active_steps = torch.stack(active_mask_steps, dim=0)  # (s, b)
        trace = RoutingTrace(
            route_ids=route_ids,
            route_probs=route_probs,
            active_mask=active_steps,
            exit_step=exit_step,
            route_depth=route_depth,
            forced_exit=forced_exit,
            executed_steps=len(route_id_steps),
            num_experts=self.config.num_experts,
            exit_route_index=self.exit_route_index,
        )
        return SGLRCoreOutput(current_states, trace)  # (b, l, d)


class MNISTPatchEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.patch_projection = nn.Conv2d(
            1,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.empty(1, config.num_patches, config.hidden_size)
        )  # (1, 49, d)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        # images: (b, 1, 28, 28)
        if images.ndim != 4 or images.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("MNISTPatchEncoder expects images shaped as (batch, 1, image_size, image_size)")
        patch_grid = self.patch_projection(images)  # (b, d, 7, 7)
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # (b, 49, d)
        return patch_tokens + self.position_embeddings  # (b, 49, d)


class MNISTSGLR(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MNISTPatchEncoder(config)
        self.core = SGLRCore(config)
        self.decision_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    @property
    def expert_names(self) -> list[str]:
        return self.core.expert_names

    def route_name(self, route_id: int) -> str:
        return self.core.route_name(route_id)

    def forward(self, images: Tensor) -> MNISTOutput:
        # images: (b, 1, 28, 28)
        hidden_states = self.encoder(images)  # (b, 49, d)
        core_output = self.core(hidden_states)  # (b, 49, d)
        normalized_states = self.decision_norm(core_output.final_hidden_states)  # (b, 49, d)
        pooled_states = masked_mean(normalized_states, attention_mask=None)  # (b, d)
        logits = self.classifier(pooled_states)  # (b, 10)
        return MNISTOutput(logits, core_output.final_hidden_states, core_output.trace)


def _fixed_depth_specs(final_mlp_width: int) -> tuple[ExpertSpec, ...]:
    return (
        ExpertSpec(name="fixed_mlp_0", family="mlp", hidden_size=256),
        ExpertSpec(name="fixed_conv_0", family="conv", channels=16, kernel_size=(3, 3)),
        ExpertSpec(name="fixed_attention_0", family="attention", internal_size=192, num_heads=6),
        ExpertSpec(name="fixed_conv_1", family="conv", channels=16, kernel_size=(5, 3)),
        ExpertSpec(name="fixed_mlp_1", family="mlp", hidden_size=final_mlp_width),
    )


class FixedDepthMNISTModel(nn.Module):
    """Five-block fixed computation baseline matched to the routed model size."""

    def __init__(self, config: ModelConfig, target_parameter_count: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = MNISTPatchEncoder(config)
        self.blocks = nn.ModuleList()
        self.decision_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        width = self._matched_final_width(target_parameter_count)
        self.specs = _fixed_depth_specs(width)
        self.blocks.extend(build_expert(spec, config.hidden_size) for spec in self.specs)
        self.expert_names = [spec.name for spec in self.specs]

        actual_count = count_parameters(self)
        relative_difference = abs(actual_count - target_parameter_count) / target_parameter_count
        if relative_difference > 0.02:
            raise ValueError(
                f"Fixed-depth baseline has {actual_count:,} parameters; target is "
                f"{target_parameter_count:,} ({relative_difference:.2%} difference)"
            )

    def _matched_final_width(self, target_parameter_count: int) -> int:
        self.blocks.extend(
            build_expert(spec, self.config.hidden_size) for spec in _fixed_depth_specs(8)[:-1]
        )
        fixed_parameter_count = count_parameters(self)
        self.blocks = nn.ModuleList()

        d = self.config.hidden_size
        layer_norm_parameters = 2 * d
        output_bias_parameters = d
        variable_budget = max(
            1,
            target_parameter_count
            - fixed_parameter_count
            - layer_norm_parameters
            - output_bias_parameters,
        )
        approximate_width = variable_budget / (2 * d + 1)
        def projected_count(width: int) -> int:
            mlp_parameters = 2 * d * width + width + d
            return fixed_parameter_count + layer_norm_parameters + mlp_parameters

        maximum_width = max(8, int(approximate_width // 8 + 3) * 8)
        for width in range(8, maximum_width + 1, 8):
            relative_difference = abs(projected_count(width) - target_parameter_count) / target_parameter_count
            if relative_difference <= 0.02:
                return width
        raise ValueError("No multiple-of-eight MLP width can match the fixed-depth parameter target")

    def route_name(self, route_id: int) -> str:
        if not 0 <= route_id < len(self.expert_names):
            raise ValueError(f"Unknown fixed-depth route ID: {route_id}")
        return self.expert_names[route_id]

    def forward(self, images: Tensor) -> MNISTOutput:
        # images: (b, 1, 28, 28)
        hidden_states = self.encoder(images)  # (b, 49, d)
        route_ids: list[Tensor] = []
        for block_index, block in enumerate(self.blocks):
            hidden_states = hidden_states + block(hidden_states)  # (b, 49, d)
            route_ids.append(
                torch.full(
                    (images.size(0),),
                    block_index,
                    device=images.device,
                    dtype=torch.long,
                )
            )

        normalized_states = self.decision_norm(hidden_states)  # (b, 49, d)
        pooled_states = masked_mean(normalized_states, attention_mask=None)  # (b, d)
        logits = self.classifier(pooled_states)  # (b, 10)
        steps = len(self.blocks)
        b = images.size(0)
        route_id_tensor = torch.stack(route_ids, dim=0)  # (5, b)
        route_probs = torch.nn.functional.one_hot(route_id_tensor, steps + 1).to(logits.dtype)  # (5, b, 6)
        trace = RoutingTrace(
            route_ids=route_id_tensor,
            route_probs=route_probs,
            active_mask=torch.ones(steps, b, device=images.device, dtype=torch.bool),
            exit_step=torch.full((b,), steps, device=images.device, dtype=torch.long),
            route_depth=torch.full((b,), steps, device=images.device, dtype=torch.long),
            forced_exit=torch.zeros(b, device=images.device, dtype=torch.bool),
            executed_steps=steps,
            num_experts=steps,
            exit_route_index=steps,
        )
        return MNISTOutput(logits, hidden_states, trace)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def build_mnist_model(config: ModelConfig) -> nn.Module:
    if config.routing_mode != "fixed_depth":
        return MNISTSGLR(config)

    routed_values = asdict(config)
    routed_values["routing_mode"] = "straight_through"
    routed_values["experts"] = [
        spec if isinstance(spec, dict) else spec for spec in routed_values["experts"]
    ]
    routed_config = ModelConfig.from_dict(routed_values)
    target_parameter_count = count_parameters(MNISTSGLR(routed_config))
    return FixedDepthMNISTModel(config, target_parameter_count)
