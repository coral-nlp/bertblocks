from torch import nn
from torch.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
    RMSNorm,
)

from polybert.modeling.config import PolyBertConfig


def get_norm(config: PolyBertConfig) -> nn.Module:
    match config.norm_type:
        case "group":
            return GroupNorm(config.hidden_size, config.hidden_size, config.norm_eps)
        case "layer":
            return LayerNorm(config.hidden_size, config.norm_eps)
        case "rms":
            return RMSNorm(config.hidden_size, config.norm_eps)
        case _:
            raise ValueError(f"Unknown norm type {config.norm_type}")
