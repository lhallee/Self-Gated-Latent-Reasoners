from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sglr.config import ModelConfig
from sglr.experts import AttentionExpert, ConvExpert, MLPExpert
from sglr.router import GroupedRoutingTrace, GroupedTop1Router


CONV_CHANNEL_CYCLE = (8, 16, 16, 32)
CONV_KERNEL_CYCLE = (3, 5, 7, 3)
CONV_DILATION_CYCLE = (1, 1, 2, 3)


@dataclass
class SGLROutput:
    logits: torch.Tensor
    final_latent: torch.Tensor
    trace: GroupedRoutingTrace


class RouterHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SGLRModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.experts = nn.ModuleList()
        self.expert_names: list[str] = []

        for expert_index in range(config.num_mlp_experts):
            self.experts.append(MLPExpert(input_size=config.input_size, hidden_size=config.mlp_hidden_size, dropout=config.mlp_dropout))
            self.expert_names.append(f"mlp_{expert_index}")

        for expert_index in range(config.num_conv_experts):
            cycle_index = expert_index % len(CONV_CHANNEL_CYCLE)
            self.experts.append(
                ConvExpert(
                    image_side=config.image_side,
                    hidden_channels=CONV_CHANNEL_CYCLE[cycle_index],
                    kernel_size=CONV_KERNEL_CYCLE[cycle_index],
                    dilation=CONV_DILATION_CYCLE[cycle_index],
                    dropout=config.conv_dropout,
                )
            )
            self.expert_names.append(f"conv_{expert_index}")

        for expert_index in range(config.num_attention_experts):
            self.experts.append(
                AttentionExpert(
                    image_side=config.image_side,
                    hidden_size=config.attention_hidden_size,
                    head_size=config.attention_head_size,
                    dropout=config.attention_dropout,
                )
            )
            self.expert_names.append(f"attention_{expert_index}")

        assert len(self.experts) == config.num_experts, "Expert count must match the configuration"
        self.initial_router = RouterHead(
            input_size=config.input_size,
            hidden_size=config.router_hidden_size,
            output_size=config.num_routes,
            dropout=config.router_dropout,
        )
        self.route_heads = nn.ModuleList(
            [
                RouterHead(
                    input_size=config.input_size,
                    hidden_size=config.router_hidden_size,
                    output_size=config.num_routes,
                    dropout=config.router_dropout,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.router = GroupedTop1Router(
            num_experts=config.num_experts,
            max_steps=config.max_steps,
            min_steps_before_exit=config.min_steps_before_exit,
        )
        self.final_classifier = nn.Linear(config.input_size, config.num_classes)

    @property
    def exit_route_index(self) -> int:
        return len(self.experts)

    def route_name(self, route_id: int) -> str:
        if route_id == self.exit_route_index:
            return "exit"
        assert 0 <= route_id < len(self.experts), "route_id must correspond to a known expert or exit"
        return self.expert_names[route_id]

    def forward(self, x: torch.Tensor) -> SGLROutput:
        assert x.ndim == 2, "SGLRModel expects flattened images shaped as [batch, input_size]"
        assert x.size(1) == self.config.input_size, "Input dimensionality must match the configured latent size"
        initial_logits = self.initial_router(x)
        trace = self.router(latents=x, initial_logits=initial_logits, experts=self.experts, route_heads=self.route_heads)
        logits = self.final_classifier(trace.final_latent)
        return SGLROutput(logits=logits, final_latent=trace.final_latent, trace=trace)
