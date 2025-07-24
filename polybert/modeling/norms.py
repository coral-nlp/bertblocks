from typing import Any

import torch
from torch import nn
from torch.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
    RMSNorm,
)

from polybert.modeling.config import PolyBertConfig


class DynamicTanhNorm(nn.Module):
    """Dynamic Tanh normalization.

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


def get_norm(config: "PolyBertConfig") -> "nn.Module":
    """Get the normalization layer specified in the configuration.

    This factory function returns the appropriate normalization layer based on
    the configuration. Supports different normalization techniques commonly used
    in transformer architectures.

    Args:
        config (PolyBertConfig): Configuration object determining model hyperparameters.

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
            return LayerNorm(config.hidden_size, config.norm_eps)
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
