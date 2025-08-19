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

from bertblocks.modeling.config import BertBlocksConfig


def get_actv_fn(config: "BertBlocksConfig") -> "nn.Module":
    """Get the activation function specified in the configuration.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

    Returns:
        nn.Module: An activation function module that can be called on tensors.

    Raises:
        ValueError: If the specified activation function is not supported.

    Supported activation functions:
        - `relu`: Rectified Linear Unit
        - `silu`: Sigmoid Linear Unit (Swish)
        - `gelu`: Gaussian Error Linear Unit
        - `leakyrelu`: Leaky Rectified Linear Unit
        - `selu`: Scaled Exponential Linear Unit
        - `logsigmoid`: Log-sigmoid activation
        - `sigmoid`: Standard sigmoid activation
        - `prelu`: Parametric Rectified Linear Unit

    """
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
