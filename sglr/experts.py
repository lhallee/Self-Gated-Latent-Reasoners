"""Shape-preserving expert deltas for Transformer hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sglr.config import ExpertSpec


def apply_token_mask(hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Zero padded token positions without changing the hidden-state shape."""
    # hidden_states: (b, l, d); attention_mask: (b, l)
    if attention_mask is None:
        return hidden_states  # (b, l, d)
    return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (b, l, d)


class MLPExpert(nn.Module):
    """Mix the hidden-width axis independently at every token."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.output_projection = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        intermediate = self.input_projection(hidden_states)  # (b, l, d_mlp)
        intermediate = F.gelu(intermediate)  # (b, l, d_mlp)
        intermediate = self.dropout(intermediate)  # (b, l, d_mlp)
        delta = self.output_projection(intermediate)  # (b, l, d)
        return apply_token_mask(delta, attention_mask)  # (b, l, d)


class AttentionExpert(nn.Module):
    """Mix the sequence axis through a small projected attention space."""

    def __init__(
        self,
        hidden_size: int,
        internal_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if internal_size % num_heads != 0:
            raise ValueError("internal_size must be divisible by num_heads")

        self.internal_size = internal_size
        self.num_heads = num_heads
        self.head_size = internal_size // num_heads
        self.dropout = dropout
        self.query_projection = nn.Linear(hidden_size, internal_size)
        self.key_projection = nn.Linear(hidden_size, internal_size)
        self.value_projection = nn.Linear(hidden_size, internal_size)
        self.output_projection = nn.Linear(internal_size, hidden_size)

    def _split_heads(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (b, l, d_attention)
        b, l, _ = hidden_states.shape
        hidden_states = hidden_states.view(b, l, self.num_heads, self.head_size)  # (b, l, h, d_h)
        return hidden_states.transpose(1, 2)  # (b, h, l, d_h)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        b, l, _ = hidden_states.shape
        Q = self._split_heads(self.query_projection(hidden_states))  # (b, h, l, d_h)
        K = self._split_heads(self.key_projection(hidden_states))  # (b, h, l, d_h)
        V = self._split_heads(self.value_projection(hidden_states))  # (b, h, l, d_h)

        attention_bias: Tensor | None = None
        if attention_mask is not None:
            attention_bias = torch.zeros(
                b,
                1,
                1,
                l,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )  # (b, 1, 1, l)
            attention_bias = attention_bias.masked_fill(
                ~attention_mask[:, None, None, :],
                torch.finfo(hidden_states.dtype).min,
            )  # (b, 1, 1, l)

        attended = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (b, h, l, d_h)
        attended = attended.transpose(1, 2).contiguous()  # (b, l, h, d_h)
        attended = attended.view(b, l, self.internal_size)  # (b, l, d_attention)
        delta = self.output_projection(attended)  # (b, l, d)
        return apply_token_mask(delta, attention_mask)  # (b, l, d)


class ConvExpert(nn.Module):
    """Convolve jointly over token position and hidden-feature position."""

    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        padding = tuple(((kernel - 1) * rate) // 2 for kernel, rate in zip(kernel_size, dilation))
        self.input_convolution = nn.Conv2d(
            1,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.output_convolution = nn.Conv2d(
            channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        masked_states = apply_token_mask(hidden_states, attention_mask)  # (b, l, d)
        image_like_states = masked_states.unsqueeze(1)  # (b, 1, l, d)
        features = self.input_convolution(image_like_states)  # (b, c_conv, l, d)
        features = F.gelu(features)  # (b, c_conv, l, d)
        if attention_mask is not None:
            features = features * attention_mask[:, None, :, None].to(features.dtype)  # (b, c_conv, l, d)
        features = self.dropout(features)  # (b, c_conv, l, d)
        delta = self.output_convolution(features).squeeze(1)  # (b, l, d)
        return apply_token_mask(delta, attention_mask)  # (b, l, d)


class ExpertDelta(nn.Module):
    """Apply pre-normalization and return only the expert update."""

    def __init__(self, hidden_size: int, expert: nn.Module) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.expert = expert

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # hidden_states: (b, l, d); attention_mask: (b, l)
        normalized_states = self.normalization(hidden_states)  # (b, l, d)
        return self.expert(normalized_states, attention_mask)  # (b, l, d)


def build_expert(spec: ExpertSpec, hidden_size: int) -> ExpertDelta:
    if spec.family == "mlp":
        expert = MLPExpert(hidden_size, spec.hidden_size, spec.dropout)
    elif spec.family == "attention":
        expert = AttentionExpert(
            hidden_size=hidden_size,
            internal_size=spec.internal_size,
            num_heads=spec.num_heads,
            dropout=spec.dropout,
        )
    elif spec.family == "conv":
        expert = ConvExpert(
            channels=spec.channels,
            kernel_size=spec.kernel_size,
            dilation=spec.dilation,
            dropout=spec.dropout,
        )
    else:
        raise ValueError(f"Unsupported expert family: {spec.family}")

    return ExpertDelta(hidden_size, expert)
