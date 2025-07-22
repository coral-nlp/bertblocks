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


def get_actv_fn(config: "PolyBertConfig") -> "nn.Module":
    """Get the activation function specified in the configuration.

    This factory function returns the appropriate activation function module
    based on the configuration. Supports various activation functions commonly
    used in transformer architectures.

    Args:
        config: PolyBert configuration object containing the activation function
                specification in model_config.actv_fn

    Returns:
        An activation function module (nn.Module) that can be called on tensors.

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
