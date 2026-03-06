from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPExpert(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
        )
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.net(self.norm1(x)) + x)


class ConvExpert(nn.Module):
    def __init__(
        self,
        image_side: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd so the convolution can stay centered"
        padding = ((kernel_size - 1) * dilation) // 2
        bottleneck_channels = hidden_channels * 2
        self.input_size = image_side * image_side
        self.image_side = image_side
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.down = nn.Sequential(
            nn.Conv2d(hidden_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.up = nn.ConvTranspose2d(bottleneck_channels, hidden_channels, kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, 1, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.size(1) == self.input_size, "ConvExpert expects flattened MNIST vectors"
        image = x.view(x.size(0), 1, self.image_side, self.image_side)
        skip = self.encoder(image)
        encoded = self.down(skip)
        bottleneck = self.bottleneck(encoded)
        upsampled = self.up(bottleneck)
        decoded = self.decoder(torch.cat((upsampled, skip), dim=1))
        return decoded.view(x.size(0), self.input_size)


class AttentionExpert(nn.Module):
    def __init__(
        self,
        image_side: int,
        hidden_size: int,
        head_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert hidden_size % head_size == 0, "hidden_size must be divisible by head_size"
        assert head_size % 2 == 0, "head_size must be even for rotary embeddings"
        self.image_side = image_side
        self.input_size = image_side * image_side
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.dropout = dropout
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(image_side, hidden_size)
        self.key_projection = nn.Linear(image_side, hidden_size)
        self.value_projection = nn.Linear(image_side, hidden_size)
        self.output_projection = nn.Linear(hidden_size, image_side)
        self.output_norm = nn.LayerNorm(image_side)

    def _apply_rotary_embedding(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(2)
        positions = torch.arange(sequence_length, device=x.device, dtype=x.dtype)
        inverse_frequencies = 1.0 / (10000 ** (torch.arange(0, self.head_size, 2, device=x.device, dtype=x.dtype) / self.head_size))
        angles = torch.outer(positions, inverse_frequencies)
        cosine = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sine = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cosine - x_odd * sine
        rotated_odd = x_even * sine + x_odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.size(1) == self.input_size, "AttentionExpert expects flattened MNIST vectors"
        batch_size = x.size(0)
        image = x.view(batch_size, self.image_side, self.image_side)
        residual = image

        query = self.query_norm(self.query_projection(image))
        key = self.key_norm(self.key_projection(image))
        value = self.value_projection(image)

        query = query.view(batch_size, self.image_side, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, self.image_side, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, self.image_side, self.num_heads, self.head_size).transpose(1, 2)
        query = self._apply_rotary_embedding(query)
        key = self._apply_rotary_embedding(key)

        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.image_side, self.hidden_size)
        projected = self.output_projection(attended)
        output = self.output_norm(projected + residual)
        return output.reshape(batch_size, self.input_size)
