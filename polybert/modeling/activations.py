from torch import nn
from torch.nn.modules.activation import (
    GELU,
    SELU,
    LeakyReLU,
    LogSigmoid,
    PReLU,
    ReLU,
    Sigmoid,
    SiLU,
)

from polybert.modeling.config import PolyBertConfig


def get_actv_fn(config: PolyBertConfig) -> nn.Module:
    match config.actv_fn:
        case "relu":
            return ReLU()
        case "silu":
            return SiLU()
        case "gelu":
            return GELU()
        case "leakyrelu":
            return LeakyReLU()
        case "selu":
            return SELU()
        case "logsigmoid":
            return LogSigmoid()
        case "sigmoid":
            return Sigmoid()
        case "prelu":
            return PReLU()
        case _:
            raise ValueError(f"Unknown activation function {config.actv_fn}")
