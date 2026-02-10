from typing import Any

import torch
from torch import nn
from torch.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
    RMSNorm,
)

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.initialization import TilableMixin, TileMode, tile_norm, tile_weight


class DynamicTanhNorm(TilableMixin, nn.Module):
    """Dynamic Tanh normalization.

    Attributes:
        alpha (nn.Parameter): learnable scalar input scale parameter.
        beta (nn.Parameter): learnable, per-channel shift parameter.
        gamma (nn.Parameter): learnable, per-channel scale parameter.

    Args:
        alpha (float): Initial alpha value.
        dim (int): Dimensionality of the input.

    References:
        - Transformers without Normalization (https://arxiv.org/pdf/2503.10622)

    """

    def __init__(self, alpha: float, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply dynamic tanh normalization.

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta

    def tile_from(
        self, pretrained: "DynamicTanhNorm", mode: str | TileMode = TileMode.tile_weights_from_middle
    ) -> None:
        """Tile weights from a smaller pretrained DynamicTanhNorm module.

        Tiles gamma (per-channel scale) and beta (per-channel shift) to the new dimension.
        The scalar alpha is copied directly as it has no spatial dimension to tile.

        Args:
            pretrained: Smaller pretrained DynamicTanhNorm module to tile from.
            mode: Tiling strategy to use.
        """
        with torch.no_grad():
            self.gamma.data = tile_weight(pretrained.gamma.data, self.gamma.data, mode=mode)
            self.beta.data = tile_weight(pretrained.beta.data, self.beta.data, mode=mode)
            self.alpha.data.copy_(pretrained.alpha.data)


class DeepNorm(TilableMixin, nn.Module):
    """DeepNorm normalization.

    References:
    - DeepNet: Scaling Transformers to 1,000 Layers (https://ieeexplore.ieee.org/document/10496231)

    """

    def __init__(
        self, alpha: float, normalized_shape: "int | list[int]", eps: float = 1e-5, **norm_kwargs: Any
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.layer_norm = LayerNorm(normalized_shape=normalized_shape, eps=eps, **norm_kwargs)

    def forward(self, x: "torch.Tensor", gx: "torch.Tensor") -> "torch.Tensor":
        """Apply DeepNorm.

        Args:
            x (torch.Tensor): Input tensor.
            gx (torch.Tensor): Gradient tensor to be scaled and added.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        return self.layer_norm(x + self.alpha * gx)

    def tile_from(self, pretrained: "DeepNorm", mode: str | TileMode = TileMode.tile_weights_from_middle) -> None:
        """Tile weights from a smaller pretrained DeepNorm module.

        Tiles the wrapped LayerNorm's weight and bias. The alpha scaling factor
        is a plain float attribute, not a parameter, and carries over unchanged.

        Args:
            pretrained: Smaller pretrained DeepNorm module to tile from.
            mode: Tiling strategy to use.
        """
        tile_norm(pretrained.layer_norm, self.layer_norm, mode=mode)


def get_norm(config: "BertBlocksConfig") -> "nn.Module":
    """Get the normalization layer specified in the configuration.

    This factory function returns the appropriate normalization layer based on
    the configuration. Supports different normalization techniques commonly used
    in transformer architectures.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.
        layer_id (int, optional): Layer ID to index into per-layer config definitions. Unused for scalar config values.

    Returns:
        A normalization module (nn.Module) that can normalize tensors.

    Raises:
        ValueError: If the specified normalization type is not supported.

    Supported normalization types:

        - `group`: Group normalization
        - `layer`: Layer normalization across the hidden dimension
        - `rms`: Root Mean Square layer normalization
        - `deep`: DeepNorm
        - `dynamictanh`: DynamicTanhNorm

    """
    match config.norm_fn:
        case "group":
            try:
                group_size = config.norm_params["group_size"]
                return GroupNorm(group_size, config.hidden_size, config.norm_eps)
            except KeyError:
                raise ValueError("When using GroupNorm, `group_size` must be specified in `config.norm_params`.")
        case "layer":
            return LayerNorm(config.hidden_size, config.norm_eps, bias=config.norm_bias)
        case "rms":
            return RMSNorm(config.hidden_size, config.norm_eps)
        case "deep":
            try:
                alpha = config.norm_params["alpha"]
                return DeepNorm(alpha, config.hidden_size, config.norm_eps)
            except KeyError:
                raise ValueError("When using DeepNorm, `alpha` must be specified in `config.norm_params`.")
        case "dynamictanh":
            try:
                alpha = config.norm_params["alpha"]
                return DynamicTanhNorm(alpha, config.hidden_size)
            except KeyError:
                raise ValueError("When using DynamicTanhNorm, `alpha` must be specified in `config.norm_params`.")
        case _:
            supported_norm_types = ["group", "layer", "rms", "deep", "dynamictanh"]
            raise ValueError(f"Unknown norm type {config.norm_fn}", f"Supported norm types: {supported_norm_types}")


__all__ = ["get_norm"]
