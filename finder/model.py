from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    frequencies = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    frequencies = 1.0 / (10000**frequencies)
    values = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate([np.sin(values), np.cos(values)], axis=1)


def sincos_3d(embed_dim: int, grid_size: tuple[int, int, int], add_cls_token: bool = True) -> np.ndarray:
    if embed_dim % 16:
        raise ValueError("Prithvi embedding dimension must be divisible by 16")
    temporal, height, width = grid_size
    width_dim = embed_dim // 16 * 6
    height_dim = embed_dim // 16 * 6
    temporal_dim = embed_dim // 16 * 4
    width_embedding = np.tile(_sincos_1d(width_dim, np.arange(width)), (temporal * height, 1))
    height_embedding = np.tile(
        np.repeat(_sincos_1d(height_dim, np.arange(height)), width, axis=0),
        (temporal, 1),
    )
    temporal_embedding = np.repeat(
        _sincos_1d(temporal_dim, np.arange(temporal)), height * width, axis=0
    )
    result = np.concatenate([width_embedding, height_embedding, temporal_embedding], axis=1)
    if add_cls_token:
        result = np.concatenate([np.zeros((1, embed_dim)), result], axis=0)
    return result


class Mlp(nn.Module):
    """Parameter-compatible subset of timm's ViT MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(value)))))


class Attention(nn.Module):
    """Parameter-compatible subset of timm's multi-head self-attention."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        result = (attention @ val).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(result))


class ViTBlock(nn.Module):
    """A zero-drop-path ViT block whose state keys match the Prithvi checkpoint."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class CustomPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 6, embed_dim: int = 768) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_size = (46, 14, 14)
        self.patch_size = (16, 1, 1)
        self.grid_size = (3, 14, 14)
        self.num_patches = math.prod(self.grid_size)
        self.temporal_spatial_embedding_conv = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(16, 1, 1),
            stride=(16, 1, 1),
            padding=(1, 0, 0),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, *self.input_size)
        if tuple(value.shape[1:]) != expected:
            raise ValueError(f"Patch embedding received {tuple(value.shape[1:])}; expected {expected}")
        value = self.temporal_spatial_embedding_conv(value)
        return self.norm(value.flatten(2).transpose(1, 2))


class PrithviEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = CustomPatchEmbedding(6, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        position = sincos_3d(embed_dim, self.patch_embed.grid_size, add_cls_token=True)
        self.register_buffer("pos_embed", torch.from_numpy(position).float().unsqueeze(0))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _random_masking(sequence: torch.Tensor, mask_ratio: float):
        batch, length, channels = sequence.shape
        keep = int(length * (1 - mask_ratio))
        noise = torch.rand(batch, length, device=sequence.device)
        shuffle = torch.argsort(noise, dim=1)
        restore = torch.argsort(shuffle, dim=1)
        indexes = shuffle[:, :keep]
        unmasked = torch.gather(sequence, 1, indexes.unsqueeze(-1).repeat(1, 1, channels))
        mask = torch.ones((batch, length), device=sequence.device)
        mask[:, :keep] = 0
        return unmasked, torch.gather(mask, 1, restore), restore

    def forward(self, value: torch.Tensor, mask_ratio: float = 0.0):
        value = self.patch_embed(value)
        value = value + self.pos_embed[:, 1:, :]
        value, mask, restore = self._random_masking(value, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(value.shape[0], -1, -1)
        value = torch.cat([cls, value], dim=1)
        for block in self.blocks:
            value = block(value)
        return self.norm(value), mask, restore

    def load_pretrained(self, checkpoint_path: Union[str, Path]) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint = checkpoint.get("model", checkpoint)
        current = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for raw_key, value in checkpoint.items():
            key = raw_key
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("encoder."):
                key = key[len("encoder.") :]
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                compatible[key] = value
            else:
                skipped.append(raw_key)
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return {
            "loaded_tensor_count": len(compatible),
            "skipped_checkpoint_keys": skipped,
            "missing_model_keys": list(missing),
            "unexpected_model_keys": list(unexpected),
        }


class LegacyGatedMIL(nn.Module):
    """Gated attention MIL with normalization over the temporal axis."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(value)
        logits = self.attention_w(self.attention_V(features) * self.attention_U(features))
        logits = torch.transpose(logits, 3, 2)
        weights = F.softmax(logits, dim=-1)
        return torch.matmul(weights, features).squeeze(-2), weights


class NystromResidualLayer(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=10,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.attn(self.norm(value))


class AnnualAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mil_layer_time = LegacyGatedMIL(hidden_dim, 64)
        self.instan_layer1 = NystromResidualLayer(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.instan_layer2 = NystromResidualLayer(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cls_semantic = nn.Linear(hidden_dim, 2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value, weights = self.mil_layer_time(value)
        value = self.norm1(self.instan_layer1(value))
        value = self.norm2(self.instan_layer2(value))
        return value, weights


class LSTMTemporalBranch(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.semantic = nn.Linear(hidden_dim, 1)
        self.change_time = nn.Linear(hidden_dim, 1)
        self.aggregator = AnnualAggregator(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded, hidden = self.encoder(value)
        decoded, _ = self.decoder(encoded, hidden)
        features, _ = self.aggregator(decoded.reshape(-1, 10, 46, decoded.shape[-1]))
        return features

    def replaced_core_parameter_count(self) -> int:
        return sum(parameter.numel() for module in (self.encoder, self.decoder) for parameter in module.parameters())


class SpatialBranch(nn.Module):
    def __init__(self, embed_dim: int, depth: int, heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.prithvi_vit = PrithviEncoder(embed_dim, depth, heads)
        self.down = nn.Linear(embed_dim, hidden_dim)
        self.aggregator = AnnualAggregator(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, observations, bands, height, width = images.shape
        windows = images.reshape(batch, 10, 46, bands, height, width)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(batch * 10, bands, 46, height, width)
        encoded, _, restore = self.prithvi_vit(windows, mask_ratio=0.0)
        patches = encoded[:, 1:, :]
        patches = torch.gather(patches, dim=1, index=restore.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        patches = patches.reshape(batch, 10, 3, height * width, patches.shape[-1])
        patches = patches.permute(0, 3, 1, 2, 4).reshape(batch * height * width, 10, 3, patches.shape[-1])
        features, _ = self.aggregator(self.down(patches))
        return features


class FinderModel(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(model_config["temporal_hidden_dim"])
        self.spatial = SpatialBranch(
            int(model_config["prithvi_embed_dim"]),
            int(model_config["prithvi_depth"]),
            int(model_config["prithvi_heads"]),
            hidden,
        )
        self.temporal = LSTMTemporalBranch(int(model_config["bands"]), hidden)
        self.fusion_gate = nn.Parameter(torch.zeros(1))
        self.linear_semantic = nn.Linear(hidden, 2)

    def configure_prithvi(
        self,
        initialization: str,
        freeze_backbone: bool,
        pretrained_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"initialization": initialization}
        if initialization == "pretrained":
            if pretrained_path is None:
                raise ValueError("pretrained_path is required for pretrained initialization")
            report.update(self.spatial.prithvi_vit.load_pretrained(pretrained_path))
        elif initialization != "random":
            raise ValueError(f"Unsupported Prithvi initialization: {initialization}")

        if freeze_backbone:
            for parameter in self.spatial.prithvi_vit.parameters():
                parameter.requires_grad = False
            for parameter in self.spatial.prithvi_vit.patch_embed.parameters():
                parameter.requires_grad = True
        report["freeze_backbone"] = freeze_backbone
        return report

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, observations, bands, height, width = images.shape
        temporal_input = images.permute(0, 3, 4, 1, 2).reshape(batch * height * width, observations, bands)
        temporal_features = self.temporal(temporal_input)
        spatial_features = self.spatial(images)
        weight = torch.sigmoid(self.fusion_gate)
        return self.linear_semantic(weight * temporal_features + (1 - weight) * spatial_features)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "temporal_replaced_core": self.temporal.replaced_core_parameter_count(),
        }
