from torch import nn
from torch.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
    RMSNorm,
)

from polybert.modeling.config import PolyBertConfig


def get_norm(config: PolyBertConfig) -> nn.Module:
    """Get the normalization layer specified in the configuration.

    This factory function returns the appropriate normalization layer based on
    the configuration. Supports different normalization techniques commonly used
    in transformer architectures.

    Args:
        config: PolyBert configuration object containing:
            - norm_type: Type of normalization ("group", "layer", "rms")
            - hidden_size: Size of the hidden dimension to normalize
            - norm_eps: Small constant for numerical stability

    Returns:
        A normalization module (nn.Module) that can normalize tensors.

    Raises:
        ValueError: If the specified normalization type is not supported.

    Supported normalization types:
        - "group": Group normalization with num_groups = hidden_size
        - "layer": Layer normalization across the hidden dimension
        - "rms": Root Mean Square layer normalization

    Note:
        For GroupNorm, the number of groups is set equal to hidden_size,
        effectively creating instance normalization.

    """
    match config.norm_type:
        case "group":
            return GroupNorm(config.hidden_size, config.hidden_size, config.norm_eps)
        case "layer":
            return LayerNorm(config.hidden_size, config.norm_eps)
        case "rms":
            return RMSNorm(config.hidden_size, config.norm_eps)
        case _:
            raise ValueError(f"Unknown norm type {config.norm_type}")
