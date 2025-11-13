from typing import Any, Literal

import torch
from torch import nn
from torch.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
    RMSNorm,
)

from bertblocks.config import BertBlocksConfig


class DynamicTanhNorm(nn.Module):
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


class DeepNorm(nn.Module):
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


class LayerNormScaler(nn.Module):
    """Scales the output of the layer normalization inversely to the layer depth.

    Attributes:
        layer_id(int): zero-indexed layer id indicating index in the encoder stack.
        norm (nn.Module): Normalization layer or `nn.Identity` if not configured.
        scaling_factor(torch.Tensor): scaling factor.

    References:
    - The Curse of Depth in Large Language Models (https://arxiv.org/pdf/2502.05795)
    """
    def __init__(self, config: "BertBlocksConfig", layer_id: int):
        super().__init__()

        self.layer_id = layer_id
        self.norm = get_norm(config)
        self.scaling_factor = 1. / torch.sqrt(torch.tensor(self.layer_id + 1))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply layer norm scaling.

        Args:
            x (torch.Tensor): Input tensor to scale.

        Returns:
            torch.Tensor: Scaled tensor.

        """
        x = self.norm(x)
        self.scaling_factor = self.scaling_factor.to(x.device)
        return x * self.scaling_factor


def get_norm(config: "BertBlocksConfig") -> "nn.Module":
    """Get the normalization layer specified in the configuration.

    This factory function returns the appropriate normalization layer based on
    the configuration. Supports different normalization techniques commonly used
    in transformer architectures.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

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
                return GroupNorm(config.norm_params["group_size"], config.hidden_size, config.norm_eps)
            except KeyError:
                raise ValueError("When using GroupNorm, `group_size` must be specified in `config.norm_params`.")
        case "layer":
            return LayerNorm(config.hidden_size, config.norm_eps, bias=config.norm_bias)
        case "rms":
            return RMSNorm(config.hidden_size, config.norm_eps)
        case "deep":
            try:
                return DeepNorm(config.norm_params["alpha"], config.hidden_size, config.norm_eps)
            except KeyError:
                raise ValueError("When using DeepNorm, `alpha` must be specified in `config.norm_params`.")
        case "dynamictanh":
            try:
                return DynamicTanhNorm(config.norm_params["alpha"], config.hidden_size)
            except KeyError:
                raise ValueError("When using DynamicTanhNorm, `alpha` must be specified in `config.norm_params`.")
        case _:
            supported_norm_types = ["group", "layer", "rms", "deep", "dynamictanh"]
            raise ValueError(f"Unknown norm type {config.norm_fn}", f"Supported norm types: {supported_norm_types}")


def _get_norm_module(config: "BertBlocksConfig",
                     norm_kind: Literal["pre", "post"],
                     layer_id: int) -> nn.Module:
    """
    Get the appropriate normalization module for pre or post normalization based on the given config.

    If norm scaling is enabled, the normalization is wrapped in a LayerNormScaler.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.
        norm_kind: Position of normalization ("pre" or "post").
        layer_id: Zero-indexed layer id indicating index in the encoder stack.

    Returns:
        LayerNormScaler | nn.Module | nn.Identity: The normalization module, which can be
            a LayerNormScaler (if scaling is enabled), a standard normalization layer (if only normalization
            is enabled), or an Identity module (if no normalization is configured).
    """

    if config.norm_scaling in (norm_kind, "both"):
        return LayerNormScaler(config, layer_id)
    elif config.norm_kind in (norm_kind, "both"):
        return get_norm(config)

    return nn.Identity()


__all__ = ["get_norm"]
